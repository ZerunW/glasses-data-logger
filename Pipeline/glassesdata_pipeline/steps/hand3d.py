"""Step ``hand3d``: metric 3D hand joints in the camera frame and the depth map's scale.

Per frame and hand (slot 0 left, 1 right):

1. **Shape** (:func:`core.hand_pnp.hand_shape`): ``mp_world`` from ``hands2d`` rescaled to the
   user's bone lengths (``profile.json``; median ratio over the wrist->MCP bones and proximal
   phalanges), or the profile's flat rest pose when ``mp_world`` is missing (low confidence).
2. **Pose** (:func:`core.hand_pnp.solve_pose`): PnP on the undistorted keypoints weighted by their
   scores. Candidates: the shape's own camera-aligned orientation, the ray-aligned one, the
   previous frame's rotation, SQPnP, and the planar flip of each; all refined by Huber-robust
   Levenberg-Marquardt. Solutions behind the camera / outside ``depth_range_m`` are rejected; near
   ties (the flip ambiguity) are decided by the shape's orientation, the previous rotation and the
   depth map's relative depths at the palm. A hand is valid when its RMS reprojection error is at
   most ``max_reproj_frac`` of its 2D size.
3. **Depth scale** (:mod:`core.depth_scale`): at the ``depth_joints`` (default the four MCPs; the
   wrist's surface is ambiguous: forearm, sleeve, watch) the depth map sees the skin
   ``skin_offset_m`` (``skin_offset_wrist_m`` at the wrist) in front of the joint centre:
   ``ratio = (z_pnp - offset) / z_model``, median per frame over both hands ->
   ``depth_scale``; robust smoothing over time -> ``depth_scale_smooth`` (median filter +
   Huber-weighted Whittaker smoother, interpolated through frames without hands, held constant
   at the ends, global median for very few observations, ``default_scale`` with none).
4. **Fusion** (optional, ``fuse_depth``): each hand's PnP depth is combined with the scaled
   depth-map depth at its palm (inverse-variance weights ``pnp_depth_sigma`` /
   ``depthmap_depth_sigma``); the rigid hand moves along the ray of its palm centre.
5. **Depth smoothing** (optional, ``hand_depth_cutoff_s``): the palm-centre depth of each hand is
   smoothed over time (robust Whittaker) and the rigid hand moved along its ray accordingly.
6. **Ray snap** (optional, ``ray_snap``): each detected joint moves onto its observed ray at its
   depth (0 = off, 1 = fully): the 2D detector is usually more precise than the 3D shape.

Outputs (PIPELINE.md section 5) ``hand3d.npz``: ``joints_cam[T,2,21,3]`` (NaN where invalid),
``valid[T,2]``, ``reproj_px[T,2]`` (RMS reprojection error of the rigid fit, full-resolution
pixels), ``depth_scale[T]`` (raw metric/model ratio, NaN when unknown), ``depth_scale_smooth[T]``.
Extras: ``confidence[T,2]`` (0..1), ``shape_source[T,2]`` int8 (0 none, 1 mp_world, 2 rest pose),
``shape_scale[T,2]`` (factor applied to mp_world), ``depth_scale_hand[T,2]`` (per-hand ratio),
``flip_ambiguous[T,2]`` (a second pose fitted almost as well; decided by the priors),
``palm_obliquity_deg[T,2]`` (angle between palm normal and line of sight; hands beyond
``max_palm_obliquity_deg`` give no depth scale and get no fusion),
``depth_fused[T,2]`` (the fusion moved the hand).
"""

from __future__ import annotations

import logging
import math

import numpy as np

from .. import geometry as geo
from .. import skeleton as sk
from ..core import depth_scale as ds
from ..core import hand_pnp as hp
from ..episode import Episode
from ..errors import PipelineError

NAME = "hand3d"
VERSION = 1
REQUIRES = ["hands2d", "depth"]
INPUTS = [
    "derived/timeline.npz", "derived/hands2d.npz", "derived/depth/index.npz", "derived/depth/depth_f16.npy",
    "derived/camera.json", "derived/profile.json",
]
OUTPUTS = ["hand3d.npz"]
CONFIG_SECTIONS = ["hand3d"]

log = logging.getLogger(__name__)
SIDES = ("left", "right")


def depth_joints(c) -> tuple[list[int], np.ndarray]:
    """Joints where the depth map is sampled and their skin offsets (metres)."""
    joints = [int(j) for j in c.get("depth_joints", [5, 9, 13, 17])]
    off = np.array([float(c.get("skin_offset_wrist_m", 0.025)) if j == 0 else float(c.get("skin_offset_m", 0.014))
                    for j in joints])
    return joints, off


class DepthSource:
    """Depth maps of ``derived/depth`` by video frame (memory-mapped)."""

    def __init__(self, ep: Episode) -> None:
        self.maps = None
        self.rows: dict[int, int] = {}
        if ep.has("depth/index.npz") and ep.has("depth/depth_f16.npy"):
            index = ep.load_npz("depth/index")
            self.maps = np.load(ep.derived / "depth" / "depth_f16.npy", mmap_mode="r")
            self.rows = {int(f): r for r, f in enumerate(np.asarray(index["frames"]).reshape(-1))}

    @property
    def available(self) -> bool:
        return self.maps is not None and len(self.rows) > 0

    def get(self, frame: int) -> np.ndarray | None:
        row = self.rows.get(int(frame))
        return None if row is None or self.maps is None else np.asarray(self.maps[row], dtype=np.float32)


def run(ep: Episode, cfg) -> dict:
    c = cfg.get("hand3d", {}) or {}
    h2 = ep.load_npz("hands2d")
    kp = np.asarray(h2["kp"], dtype=np.float64)
    mp_world = np.asarray(h2.get("mp_world", np.full(kp.shape, np.nan)), dtype=np.float64)
    hand_score = np.asarray(h2.get("hand_score", np.full(kp.shape[:2], np.nan)), dtype=np.float64)
    T = kp.shape[0]
    camera = ep.camera()
    profile = ep.profile()
    bones = np.asarray(profile.get("bone_lengths_m") or sk.bone_lengths(profile), dtype=np.float64)
    depth = DepthSource(ep) if bool(c.get("use_depth", True)) else None
    if depth is not None and not depth.available:
        log.warning("no derived/depth maps: depth_scale is unknown and no depth fusion is done")
        depth = None
    t = np.asarray(ep.timeline()["t"], dtype=np.float64)
    fps = 1.0 / float(np.median(np.diff(t))) if len(t) > 1 else 24.0
    backend = (h2.get("meta") or {}).get("backend")
    result = reconstruct(kp, mp_world, hand_score, camera, profile, bones, depth, c, fps, backend)
    check_shapes(result, T)
    meta = ep.step_meta(backend=None, depth_scale_info=result.pop("info"))
    ep.save_npz("hand3d", meta, **result)
    valid = result["valid"]
    scale = result["depth_scale_smooth"]
    return {
        "frames": T,
        "valid_left": int(valid[:, 0].sum()),
        "valid_right": int(valid[:, 1].sum()),
        "reproj_px": round(float(np.nanmedian(result["reproj_px"][valid])), 2) if valid.any() else None,
        "scale_frames": int(np.isfinite(result["depth_scale"]).sum()),
        "scale_median": round(float(np.median(scale)), 4) if np.all(np.isfinite(scale)) else None,
    }


def reconstruct(kp, mp_world, hand_score, camera, profile, bones, depth, c, fps: float = 24.0,
                hands_backend: str | None = None) -> dict:
    """All of the step's arrays (no file I/O except reading depth maps through ``depth``)."""
    T = kp.shape[0]
    W, H = camera.width, camera.height
    f_px = 0.5 * (camera.fx + camera.fy)
    min_score = float(c.get("min_score", 0.3))
    min_joints = int(c.get("min_joints", 8))
    max_frac = float(c.get("max_reproj_frac", 0.1))
    zr = c.get("depth_range_m", [0.08, 1.5])
    depth_range = (float(zr[0]), float(zr[1]))
    shape_mode = str(c.get("shape_mode", "scale"))
    scale_bones = [int(b) for b in c.get("scale_bones", list(hp.SCALE_BONES))]
    max_gap = int(c.get("prev_max_gap", 12))
    snap = float(c.get("ray_snap", 0.0))
    radius = int(c.get("depth_sample_radius", 1))
    palm, offsets = depth_joints(c)
    solve_kwargs = dict(
        huber_px=float(c.get("huber_px", 3.0)) * W / 720.0,
        depth_range=depth_range,
        ambiguity_ratio=float(c.get("ambiguity_ratio", 1.5)),
        ambiguity_px=float(c.get("ambiguity_px", 1.0)) * W / 720.0,
        prior_sigma_rad=math.radians(float(c.get("prior_sigma_deg", 35.0))),
        prev_sigma_rad=math.radians(float(c.get("prev_sigma_deg", 15.0))),
        depth_sigma=float(c.get("depth_rel_sigma", 0.01)),
    )
    rot_sigma = math.radians(float(c.get("orientation_sigma_deg", 10.0)))
    anchor = float(c.get("kp_sigma_px", 3.0)) * W / 720.0 / rot_sigma if rot_sigma > 0 else 0.0
    shape_frame = str(c.get("shape_frame", "auto"))
    if shape_frame == "auto":
        shape_frame = "ray" if hands_backend in (None, "mediapipe") else "camera"
    use_depth_prior = bool(c.get("depth_flip_prior", True))

    joints = np.full((T, 2, 21, 3), np.nan)
    valid = np.zeros((T, 2), bool)
    reproj = np.full((T, 2), np.nan)
    confidence = np.zeros((T, 2))
    shape_source = np.zeros((T, 2), np.int8)
    shape_scale = np.full((T, 2), np.nan)
    ambiguous = np.zeros((T, 2), bool)
    ratio_hand = np.full((T, 2), np.nan)
    z_model = np.full((T, 2, len(palm)), np.nan)
    hand_off = np.full((T, 2, len(palm)), np.nan)  # skin offsets per hand (NaN: hand not used for depth)
    obliquity = np.full((T, 2), np.nan)
    max_obliquity = float(c.get("max_palm_obliquity_deg", 60.0))
    xy_all = camera.undistort_points(kp[..., :2])
    factors = shape_factors(mp_world, bones, scale_bones, float(c.get("shape_scale_window_s", 1.0)) * fps)
    last_R: list[np.ndarray | None] = [None, None]
    last_t = [-10**9, -10**9]
    grid_size = None

    for i in range(T):
        dmap = depth.get(i) if depth is not None else None
        if dmap is not None and grid_size is None:
            grid_size = (dmap.shape[1], dmap.shape[0])
        for h in range(2):
            uv = kp[i, h, :, :2]
            seen = np.all(np.isfinite(uv), axis=1)
            if seen.sum() < min_joints:
                continue
            w = np.where(seen, np.nan_to_num(kp[i, h, :, 2], nan=0.0), 0.0)
            w = np.where(w >= min_score, w, 0.0)
            if (w > 0).sum() < min_joints:
                continue
            X, source, factor = hp.hand_shape(mp_world[i, h], bones, profile, SIDES[h], shape_mode, scale_bones,
                                              factor=factors[i, h])
            if dmap is not None:
                z_model[i, h] = ds.sample_depth(dmap, ds.to_grid(uv[palm], (W, H), grid_size), radius)
            R_prior = None
            if source == 1:
                R_prior = np.eye(3)
                if shape_frame == "ray":
                    centre = np.nanmean(xy_all[i, h][w > 0], axis=0)
                    R_prior = geo.rot_from_two_vectors(np.array([0.0, 0.0, 1.0]), np.array([centre[0], centre[1], 1.0]))
            gap = i - last_t[h]
            R_prev = last_R[h] if gap <= max_gap else None
            res = hp.solve_pose(
                X, xy_all[i, h], w, f_px, R_prior=R_prior, R_prev=R_prev, prev_gap=gap,
                z_model_palm=z_model[i, h] if (use_depth_prior and dmap is not None) else None, depth_joints=palm,
                anchor_px_per_rad=anchor, **solve_kwargs,
            )
            if res is None:
                continue
            box = uv[seen]
            diag = float(np.linalg.norm(box.max(0) - box.min(0)))
            frac = res.rms_px / max(diag, 1.0)
            if frac > max_frac:
                continue
            joints[i, h] = res.joints
            valid[i, h] = True
            reproj[i, h] = res.rms_px
            shape_source[i, h] = source
            shape_scale[i, h] = factor
            ambiguous[i, h] = res.ambiguous
            confidence[i, h] = float(np.mean(w[w > 0])) * (1.0 if source == 1 else 0.3) * max(0.0, 1.0 - frac / max_frac)
            last_R[h], last_t[h] = res.R, i
            obliquity[i, h], hand_off[i, h] = palm_offsets(res.joints, palm, offsets, max_obliquity)
            if dmap is not None and np.all(np.isfinite(hand_off[i, h])):
                ratio_hand[i, h] = ds.frame_ratio(res.joints[palm, 2], z_model[i, h], hand_off[i, h],
                                                  min_samples=int(c.get("min_depth_samples", 3)))

    # ------------------------------------------------------------ depth scale
    raw = np.full(T, np.nan)
    weight = np.zeros(T)
    for i in range(T):
        samples = []
        for h in range(2):
            if valid[i, h] and np.isfinite(ratio_hand[i, h]):
                z = joints[i, h, palm, 2] - hand_off[i, h]
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = z / z_model[i, h]
                samples.append(r[np.isfinite(r)])
                weight[i] += confidence[i, h]
        if samples:
            raw[i] = float(np.median(np.concatenate(samples)))
    smooth, info = ds.smooth_scale(
        raw, weight,
        median_window=max(1, int(round(float(c.get("scale_median_window_s", 0.5)) * fps))),
        cutoff_frames=float(c.get("scale_cutoff_s", 2.0)) * fps,
        huber_k=float(c.get("scale_huber_k", 1.345)),
        min_observations=int(c.get("scale_min_observations", 5)),
        default=float(c.get("default_scale", 1.0)),
    )
    if info["method"] == "default" and depth is not None:
        log.warning("no frame gave a depth scale (no valid hand over the depth map); using %.3f", smooth[0])

    # ------------------------------------------------------------ fusion and ray snap
    fused = np.zeros((T, 2), bool)
    if depth is not None and bool(c.get("fuse_depth", True)):
        w_pnp = 1.0 / float(c.get("pnp_depth_sigma", 0.03)) ** 2
        w_dm = 1.0 / float(c.get("depthmap_depth_sigma", 0.03)) ** 2
        max_change = float(c.get("fuse_max_change", 0.25))
        for i, h in zip(*np.nonzero(valid)):
            zm = z_model[i, h]
            zj = joints[i, h, palm, 2]
            k = (smooth[i] * zm + hand_off[i, h]) / zj
            k = k[np.isfinite(k)]
            if k.size < int(c.get("min_depth_samples", 3)):
                continue
            k = float(np.median(k))
            if abs(k - 1.0) > max_change:
                continue
            f = (w_pnp + w_dm * k) / (w_pnp + w_dm)
            centre = joints[i, h, palm].mean(axis=0)
            joints[i, h] += (f - 1.0) * centre
            fused[i, h] = True
    depth_cutoff = float(c.get("hand_depth_cutoff_s", 0.0)) * fps
    if depth_cutoff >= 2:
        smooth_hand_depth(joints, valid, depth_cutoff, int(c.get("hand_depth_max_gap", 3)))
    if snap > 0:
        for i, h in zip(*np.nonzero(valid)):
            seen = np.all(np.isfinite(kp[i, h, :, :2]), axis=1) & (np.nan_to_num(kp[i, h, :, 2]) >= min_score)
            joints[i, h] = hp.snap_to_rays(joints[i, h], xy_all[i, h], snap, seen)

    info.update({"depth_joints": palm, "skin_offsets_m": offsets.tolist(), "shape_frame": shape_frame})
    return {
        "joints_cam": joints,
        "valid": valid,
        "reproj_px": reproj,
        "depth_scale": raw,
        "depth_scale_smooth": smooth,
        "confidence": confidence,
        "shape_source": shape_source,
        "shape_scale": shape_scale,
        "depth_scale_hand": ratio_hand,
        "flip_ambiguous": ambiguous,
        "palm_obliquity_deg": obliquity,
        "depth_fused": fused,
        "info": info,
    }


def palm_offsets(joints: np.ndarray, palm: list[int], offsets: np.ndarray, max_obliquity_deg: float,
                 max_stretch: float = 2.0) -> tuple[float, np.ndarray]:
    """Angle (degrees) between the palm normal and the line of sight, and the skin offsets of the
    depth joints for this view: ``offset / cos(angle)`` (a ray crossing a slab obliquely travels
    further through it; at most ``max_stretch`` times), all NaN beyond ``max_obliquity_deg`` (an
    edge-on palm: the far joints hide behind the hand and grazing depth is unreliable)."""
    J = np.asarray(joints, dtype=np.float64)
    n = np.cross(J[5] - J[0], J[17] - J[0])
    norm = np.linalg.norm(n)
    rays = J[palm] / np.linalg.norm(J[palm], axis=1, keepdims=True)
    if norm < 1e-9 or not np.all(np.isfinite(rays)):
        return math.nan, np.full(len(palm), np.nan)
    cos = np.abs(rays @ (n / norm))
    angle = float(np.degrees(np.arccos(np.clip(np.mean(cos), 0.0, 1.0))))
    if angle > max_obliquity_deg:
        return angle, np.full(len(palm), np.nan)
    return angle, offsets / np.maximum(cos, 1.0 / max_stretch) * rays[:, 2]


def smooth_hand_depth(joints: np.ndarray, valid: np.ndarray, cutoff_frames: float, max_gap: int) -> None:
    """In place: per hand and run of valid frames (gaps of at most ``max_gap`` frames bridged), the
    depth of the palm centre is smoothed over time (robust Whittaker smoother of log depth, cut-off
    period ``cutoff_frames``) and the rigid hand moves along the ray of its palm centre to it. The
    per-frame PnP depth is noisy along the line of sight; the hand moves smoothly at video rate."""
    palm = list(hp.PALM)
    for h in range(2):
        idx = np.flatnonzero(valid[:, h])
        if idx.size < 5:
            continue
        breaks = np.flatnonzero(np.diff(idx) > max_gap + 1) + 1
        for run in np.split(idx, breaks):
            if run.size < 5:
                continue
            lo, hi = int(run[0]), int(run[-1]) + 1
            z = np.full(hi - lo, np.nan)
            centres = joints[lo:hi, h][:, palm].mean(axis=1)
            ok = valid[lo:hi, h]
            z[ok] = centres[ok, 2]
            smooth, _ = ds.smooth_scale(z, median_window=3, cutoff_frames=cutoff_frames, min_sigma=0.002,
                                        min_observations=5, default=1.0)
            for k in np.flatnonzero(ok):
                f = smooth[k] / z[k]
                if np.isfinite(f) and 0.5 < f < 2.0:
                    joints[lo + k, h] += (f - 1.0) * centres[k]



def shape_factors(mp_world: np.ndarray, bones: np.ndarray, scale_bones, window_frames: float) -> np.ndarray:
    """``[T,2]`` scale factor of ``mp_world`` to the user's bones, per frame, then a running median
    over ``window_frames`` per hand (the hand's size does not change; the per-frame ratio of noisy
    bone lengths does). NaN without ``mp_world``."""
    T = mp_world.shape[0]
    out = np.full((T, 2), np.nan)
    for i in range(T):
        for h in range(2):
            if np.all(np.isfinite(mp_world[i, h])):
                out[i, h] = hp.scale_to_bones(mp_world[i, h], bones, scale_bones)
    if window_frames >= 2:
        for h in range(2):
            have = np.isfinite(out[:, h]) & (out[:, h] > 0)
            if have.any():
                med = np.exp(ds.running_median(np.where(have, np.log(np.where(have, out[:, h], 1.0)), np.nan),
                                               int(round(window_frames))))
                out[:, h] = np.where(have, med, np.nan)
    return out


def check_shapes(d: dict, T: int) -> None:
    for key, shape in (("joints_cam", (T, 2, 21, 3)), ("valid", (T, 2)), ("depth_scale_smooth", (T,))):
        if np.asarray(d[key]).shape != shape:
            raise PipelineError(f"hand3d.{key} has shape {np.asarray(d[key]).shape}, expected {shape}")
