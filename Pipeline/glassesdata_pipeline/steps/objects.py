"""Step ``objects``: unknown objects (no CAD, no category) from monocular video + metric depth +
hand tracking, and the supporting table, ready for MuJoCo (PIPELINE.md section 5).

1. **Table**: sequential RANSAC for horizontal planes over scene depth points (every
   ``table.frame_step``-th frame, box-filtered depth, hand pixels removed, metric via
   ``hand3d.depth_scale_smooth``, world via ``camera_traj``). The table is the largest horizontal
   plane just below the objects (the grasp points); its polygon is the 2-D convex hull of the
   inlier region (occupancy grid component) that contains the objects; the floor is the lowest
   plane at least ``floor_min_drop_m`` below it (``height_m`` = table above floor).
2. **Which objects**: grasp episodes from ``hands_world`` (:func:`core.grasp.detect_grasps`). Each
   episode gives a prompt: the grasp point at onset projected into the least occluded frame shortly
   before onset (snapped to the nearest depth surface), plus manual prompts (``objects.prompts``).
3. **Masks** per prompt from ``objects.mask_backend`` (``gt`` or ``sam2``) at depth resolution;
   prompts whose masks overlap (mean IoU >= ``merge_iou``) are one object.
4. **Geometry**: masked depth (eroded, hands removed) of the frames before the first grasp fused
   in a TSDF (``fusion.method = "points"``: a voxelised point cloud), outliers removed, completed by
   extrusion down to the table (optional mirror symmetry), meshed (``hull``: a convex hull that
   ignores the ``hull_trim_frac`` most extreme points per direction; ``open3d``; ``image_to_3d``),
   collision parts (``hull`` / ``coacd``), mass properties. Object frame: origin at the volume
   centroid, z = table normal, x/y from the footprint's minimum-area rectangle.
5. **Poses**: static before the first grasp (the fused pose). Inside a grasp the object follows
   the rigid motion of the whole hand skeleton (generalised Procrustes of all 21 joints of the
   grasping hand, :func:`core.grasp.hand_rigid_motion`), anchored at the pose before the grasp and
   Gaussian-smoothed (optionally one stacked-ICP correction of the hand-object offset,
   ``tracking.grasp_dof``); an object over the table polygon is lifted out of the table plane
   (``tracking.no_table_penetration``: the hand's depth error would otherwise push it into the
   table while it is picked up and put down). After a release: one robust point-to-plane ICP of
   the points of all frames of the static segment (translation only when the object is upright),
   then the object is put upright on the table. ``valid`` is False where the hand is missing (pose
   held) or the static ICP failed. ``grasped[T, 2]``: two fingertips within ``contact_m`` of the surface during
   a detected episode (or co-motion outside episodes). ``fitness`` is the per-frame fraction of
   observed points within ``max_correspondence_m`` of the posed model.

``poses.npz`` also has ``pose_source[T]`` int8 (0 measured, 1 assumed_static, 2 hand_motion,
3 interpolated; names in ``pose_source_names``; see :func:`track_object`), so poses assumed
before the grasp can be told from measured ones. The shape is fused only from frames before the
first grasp onset (:func:`fusion_frames`); an object with fewer than 3 such frames fails with
that reason.

Extra output (not in the section-5 table): ``objects/grasps.json`` (detected and rejected grasp
episodes, all prompts, failed objects with the reason).
"""

from __future__ import annotations

import logging
import math
import shutil
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import geometry as geo
from .. import skeleton as sk
from ..core import fusion as fu
from ..core import grasp as gr
from ..core import hull as hl
from ..core import plane as pl
from ..core.icp import average_poses, icp_point_to_plane
from ..core.knn import KNN
from ..episode import Episode
from ..errors import PipelineError
from ..groundtruth import unpack_mask

NAME = "objects"
VERSION = 3  # 2: near-surface test past the grasping hand, no table penetration while grasped; 3: pose_source
REQUIRES = ["hands2d", "depth", "hand3d", "world"]
OUTPUTS = ["objects/index.json", "objects/table.json"]
CONFIG_SECTIONS = ["objects", "synth"]

log = logging.getLogger(__name__)

_BASE_INPUTS = [
    "derived/timeline.npz", "derived/camera.json", "derived/depth/index.npz", "derived/depth/depth_f16.npy",
    "derived/hand3d.npz", "derived/camera_traj.npz", "derived/hands_world.npz", "derived/hands2d.npz",
]


def INPUTS(cfg: Any) -> list[str]:  # noqa: N802 - the runner's name for the declaration
    backend = str(cfg.get("objects.mask_backend", "sam2"))
    extra = ["gt/gt.npz", "gt/masks.npz"] if backend == "gt" else ["video.mp4"]
    return _BASE_INPUTS + extra


# =============================================================================================
# Inputs
# =============================================================================================


@dataclass
class Inputs:
    T: int
    fps: float
    t: np.ndarray
    camera: geo.Camera          # full-resolution video camera
    cam_d: geo.Camera           # depth-grid camera
    rays: np.ndarray            # (h, w, 3) rays of the depth grid
    depth: np.ndarray           # memmap (F, h, w) float16 (model units)
    row: dict[int, int]         # video frame -> depth row
    scale: np.ndarray           # (T,) metric = depth * scale
    T_wc: np.ndarray            # (T, 4, 4)
    cam_valid: np.ndarray       # (T,)
    joints: np.ndarray          # (T, 2, 21, 3) hands_world joints (NaN = missing)
    kp2d: np.ndarray | None     # (T, 2, 21, 3) hands2d keypoints (full-res pixels)
    hand_radius: float
    min_depth: float
    max_depth: float
    _hand_cache: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.depth.shape[1]), int(self.depth.shape[2])

    def first_frame_after(self, f: int, seconds: float) -> int:
        """The first frame at or after ``t[f] - seconds`` (real timestamps, not a frame count)."""
        return int(np.searchsorted(self.t, self.t[f] - seconds - 1e-6, side="left"))

    def has(self, f: int) -> bool:
        return f in self.row and bool(self.cam_valid[f]) and math.isfinite(float(self.scale[f]))

    def metric_depth(self, f: int) -> np.ndarray:
        d = np.asarray(self.depth[self.row[f]], dtype=np.float64) * float(self.scale[f])
        d[~np.isfinite(d) | (d < self.min_depth) | (d > self.max_depth)] = np.nan
        return d

    def to_depth_px(self, uv: np.ndarray) -> np.ndarray:
        """Full-resolution pixels -> depth-grid pixels (pixel-centre convention)."""
        s = np.array([self.cam_d.width / self.camera.width, self.cam_d.height / self.camera.height])
        return (np.asarray(uv, dtype=np.float64) + 0.5) * s - 0.5

    def to_full_px(self, uv_d: np.ndarray) -> np.ndarray:
        s = np.array([self.cam_d.width / self.camera.width, self.cam_d.height / self.camera.height])
        return (np.asarray(uv_d, dtype=np.float64) + 0.5) / s - 0.5

    def hand_mask(self, f: int) -> np.ndarray:
        """Depth-grid pixels covered by the hands in frame ``f`` (discs around the projected
        joints and bones of hands_world, plus hands2d keypoints)."""
        if f in self._hand_cache:
            return self._hand_cache[f]
        import cv2

        h, w = self.shape
        m = np.zeros((h, w), np.uint8)
        if self.cam_valid[f]:
            X = geo.transform(geo.inv_T(self.T_wc[f]), self.joints[f])  # (2,21,3)
            uv = self.cam_d.project(X)
            fx = self.cam_d.fx
            for hand in range(2):
                ok = np.all(np.isfinite(uv[hand]), axis=-1) & (X[hand, :, 2] > 0.05)
                r = np.where(ok, fx * self.hand_radius / np.where(ok, X[hand, :, 2], 1.0), 0.0)
                for j in range(21):
                    if not ok[j]:
                        continue
                    c = (int(round(uv[hand, j, 0])), int(round(uv[hand, j, 1])))
                    rr = max(1, int(math.ceil(r[j])))
                    cv2.circle(m, c, rr, 1, -1)
                    p = sk.PARENTS[j]
                    if p >= 0 and ok[p]:
                        c2 = (int(round(uv[hand, p, 0])), int(round(uv[hand, p, 1])))
                        cv2.line(m, c, c2, 1, max(1, int(round(r[j] + r[p]))))
                if self.kp2d is not None:
                    k = self.kp2d[f, hand, :, :2]
                    okk = np.all(np.isfinite(k), axis=-1)
                    kd = self.to_depth_px(k[okk])
                    # disc radius from the projected joints, else ~hand_radius at 0.5 m
                    rk = max(1, int(round(float(np.median(r[ok])) if np.any(ok) else fx * self.hand_radius / 0.5)))
                    for (u, v) in kd:
                        cv2.circle(m, (int(round(u)), int(round(v))), rk, 1, -1)
        out = m.astype(bool)
        if len(self._hand_cache) > 512:
            self._hand_cache.clear()
        self._hand_cache[f] = out
        return out

    def object_points(self, f: int, mask: np.ndarray, erode_px: int) -> np.ndarray:
        """World points of the (eroded) object mask in frame ``f`` without hand pixels."""
        if not self.has(f):
            return np.zeros((0, 3))
        m = fu.erode(mask, erode_px) & ~self.hand_mask(f)
        X, _ = fu.backproject(self.metric_depth(f), m, self.rays, self.min_depth, self.max_depth)
        return geo.transform(self.T_wc[f], X)


def _load_inputs(ep: Episode, cfg: Any) -> Inputs:
    ocfg = cfg.section("objects")
    tl = ep.timeline()
    T = int(len(tl["t"]))
    t = np.asarray(tl["t"], dtype=np.float64)
    fps = 1.0 / float(np.median(np.diff(t))) if T > 1 else 24.0
    camera = ep.camera()
    index = ep.load_npz("depth/index")
    depth = np.load(ep.derived / "depth" / "depth_f16.npy", mmap_mode="r")
    frames = np.asarray(index["frames"], dtype=np.int64)
    h, w = int(depth.shape[1]), int(depth.shape[2])
    cam_d = camera.resized(w, h)
    hand3d = ep.load_npz("hand3d")
    scale = np.asarray(hand3d.get("depth_scale_smooth", np.full(T, np.nan)), dtype=np.float64)[:T]
    ok = np.isfinite(scale) & (scale > 0)
    if not np.any(ok):
        log.warning("hand3d.depth_scale_smooth has no value: depth is used as metric")
        scale = np.ones(T)
    elif not np.all(ok):
        scale = np.interp(np.arange(T), np.flatnonzero(ok), scale[ok])
    traj = ep.load_npz("camera_traj")
    T_wc = np.asarray(traj["T_world_cam"], dtype=np.float64)
    cam_valid = np.asarray(traj.get("valid", np.ones(T, bool)), dtype=bool) & np.all(np.isfinite(T_wc), axis=(-2, -1))
    hw = ep.load_npz("hands_world")
    joints = np.asarray(hw["joints"], dtype=np.float64)
    kp2d = None
    if ep.has("hands2d.npz"):
        kp2d = np.asarray(ep.load_npz("hands2d")["kp"], dtype=np.float64)
    if joints.shape[0] != T or T_wc.shape[0] != T:
        raise PipelineError(f"objects: inputs disagree on the frame count (timeline {T}, camera_traj "
                            f"{T_wc.shape[0]}, hands_world {joints.shape[0]})")
    return Inputs(
        T=T, fps=fps, t=t, camera=camera, cam_d=cam_d, rays=np.asarray(cam_d.pixel_rays()), depth=depth,
        row={int(f): r for r, f in enumerate(frames)}, scale=scale, T_wc=T_wc, cam_valid=cam_valid,
        joints=joints, kp2d=kp2d, hand_radius=float(ocfg.get("hand_radius_m", 0.012)),
        min_depth=float(ocfg.get("min_depth_m", 0.1)), max_depth=float(ocfg.get("max_depth_m", 4.0)),
    )


# =============================================================================================
# Table
# =============================================================================================


def _box_blur_valid(d: np.ndarray, k: int) -> np.ndarray:
    import cv2

    if k <= 1:
        return d
    valid = np.isfinite(d).astype(np.float32)
    num = cv2.blur(np.where(valid > 0, d, 0.0).astype(np.float32), (k, k))
    den = cv2.blur(valid, (k, k))
    out = np.where(den > 0.5, num / np.maximum(den, 1e-6), np.nan)
    return np.where(valid > 0, out, np.nan)


def scene_points(inp: Inputs, tcfg: Any, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """World points (and their camera depth) of every ``frame_step``-th frame, hands removed."""
    step = max(1, int(tcfg.get("frame_step", 6)))
    blur = int(tcfg.get("blur_px", 5))
    pts, zs = [], []
    for f in range(0, inp.T, step):
        if not inp.has(f):
            continue
        d = _box_blur_valid(inp.metric_depth(f), blur)
        X, _ = fu.backproject(d, ~inp.hand_mask(f), inp.rays, inp.min_depth, inp.max_depth)
        pts.append(geo.transform(inp.T_wc[f], X))
        zs.append(X[:, 2])
    if not pts:
        return np.zeros((0, 3)), np.zeros(0)
    P, Z = np.concatenate(pts), np.concatenate(zs)
    n_max = int(tcfg.get("max_points", 80000))
    if len(P) > n_max:
        sel = np.random.default_rng(seed).choice(len(P), n_max, replace=False)
        P, Z = P[sel], Z[sel]
    return P, Z


def find_table(P: np.ndarray, Z: np.ndarray, anchors: np.ndarray, tcfg: Any) -> dict[str, Any]:
    """The support table and the floor from scene points ``P`` (world) with camera depths ``Z``."""
    out: dict[str, Any] = {"found": False, "planes": []}
    if len(P) < 100:
        return out
    thr = float(tcfg.get("threshold_m", 0.012)) + float(tcfg.get("threshold_rel", 0.012)) * Z
    planes = pl.horizontal_planes(
        P, thr, max_planes=int(tcfg.get("max_planes", 4)),
        min_inliers=max(50, int(float(tcfg.get("min_inlier_frac", 0.03)) * len(P))),
        max_tilt_deg=float(tcfg.get("max_tilt_deg", 12.0)), iterations=int(tcfg.get("iterations", 300)),
    )
    out["planes"] = [{"z": p["z"], "n_inliers": p["n_inliers"], "tilt_deg": pl.tilt_deg(p["normal"]),
                      "rms_m": p["rms"]} for p in planes]
    if not planes:
        return out
    anchors = np.asarray(anchors, dtype=np.float64).reshape(-1, 3)
    if len(anchors):
        a_low = float(anchors[:, 2].min())
        cands = [p for p in planes if p["z"] < a_low + 0.02 and p["z"] > a_low - 0.5]
        table = max(cands, key=lambda p: p["n_inliers"]) if cands else None
        if table is None:
            below = [p for p in planes if p["z"] < a_low + 0.02]
            table = max(below, key=lambda p: p["z"]) if below else max(planes, key=lambda p: p["n_inliers"])
    else:
        table = max(planes, key=lambda p: p["n_inliers"])
    idx = table["inliers"]
    xy = P[idx, :2]
    comp = pl.component_mask(xy, float(tcfg.get("cell_m", 0.03)), anchors[:, :2] if len(anchors) else None,
                             close_cells=1, min_count=2)
    sel = idx[comp] if comp.sum() >= 3 else idx
    normal, offset = pl.fit_plane(P[sel])
    poly = pl.convex_polygon(P[sel, :2])
    centre = poly.mean(axis=0) if len(poly) else P[sel, :2].mean(axis=0)
    top_z = float(pl.plane_height(normal, offset, centre))
    floor = [p for p in planes if p["z"] < top_z - float(tcfg.get("floor_min_drop_m", 0.3))]
    floor_z = None
    if floor:
        fl = min(floor, key=lambda p: p["z"])
        floor_z = float(pl.plane_height(fl["normal"], fl["offset"], centre))
    height = top_z - floor_z if floor_z is not None else float(tcfg.get("default_height_m", 0.75))
    resid = pl.plane_distance(P[sel], normal, offset)
    out.update({
        "found": True, "normal": normal, "offset": float(offset), "top_z": top_z, "floor_z": floor_z,
        "height_m": float(height), "height_source": "floor_plane" if floor_z is not None else "default",
        "polygon": poly, "n_inliers": int(len(sel)), "rms_m": float(np.sqrt(np.mean(resid**2))),
        "tilt_deg": pl.tilt_deg(normal),
    })
    return out


# =============================================================================================
# Prompts
# =============================================================================================


def _near_surface_fn(inp: Inputs, near_m: float, lookback_s: float = 0.6, min_free_frac: float = 0.25):
    """``near(hand, frame, point)`` for :func:`core.grasp.detect_grasps`: is the grasp point within
    ``near_m`` (camera depth) of an observed surface that is not the hand, in a +-2 cm window around
    its projection? The grasping hand usually hides that surface at the onset (a depth model sees
    the hand, and what shows around its silhouette is the background), so the decision uses the
    least occluded frame in ``[onset - lookback_s, onset]`` (the object is still static there): the
    one with the most non-hand depth pixels in the window. None (unknown, not a rejection) when even
    that frame has fewer than ``min_free_frac`` of the window free."""
    def near(hand: int, frame: int, point: np.ndarray) -> bool | None:
        best: tuple[int, int, bool] | None = None  # (free pixels, frame, decision); ties: nearest the onset
        start = min(inp.first_frame_after(frame, lookback_s), max(0, frame - 2)) if 0 <= frame < inp.T else frame
        for f in range(frame, start - 1, -1):
            if not (0 <= f < inp.T and inp.has(f)):
                continue
            X = geo.transform(geo.inv_T(inp.T_wc[f]), point)
            uv = inp.cam_d.project(X)
            if not np.all(np.isfinite(uv)) or X[2] <= 0:
                continue
            u, v = int(round(uv[0])), int(round(uv[1]))
            h, w = inp.shape
            if not (0 <= u < w and 0 <= v < h):
                continue
            r = max(1, int(round(inp.cam_d.fx * 0.02 / max(X[2], 0.1))))  # +-2 cm around the point
            win = (slice(max(0, v - r), v + r + 1), slice(max(0, u - r), u + r + 1))
            d = np.where(inp.hand_mask(f)[win], np.nan, inp.metric_depth(f)[win])  # the hand is no surface
            free = int(np.isfinite(d).sum())
            if free < max(1, min_free_frac * d.size) or (best is not None and free <= best[0]):
                continue
            best = (free, f, bool(np.nanmin(np.abs(d - X[2])) <= near_m))
        return None if best is None else best[2]

    return near


def _hand_samples(J: np.ndarray) -> np.ndarray:
    """Joints plus bone midpoints of one hand (21 + 20, 3)."""
    mids = 0.5 * (J[1:] + J[np.array(sk.PARENTS[1:])])
    return np.concatenate([J, mids])


def grasp_prompt(inp: Inputs, episode: dict[str, Any], gcfg: Any) -> dict[str, Any] | None:
    """The prompt of a grasp episode: frame in [onset - lookback, onset] where the projected grasp
    point is farthest from the projected hand, pixel snapped to the nearest depth surface."""
    point = np.asarray(episode["point"], dtype=np.float64)
    hand = int(episode["hand"])
    onset = int(episode["onset"])
    start = inp.first_frame_after(onset, float(gcfg.get("lookback_s", 0.6)))
    best = None
    for f in range(start, onset + 1):
        if not inp.has(f):
            continue
        T_cw = geo.inv_T(inp.T_wc[f])
        X = geo.transform(T_cw, point)
        uv = inp.camera.project(X)
        if not (np.all(np.isfinite(uv)) and inp.camera.in_image(uv, margin=2.0)):
            continue
        Hs = _hand_samples(inp.joints[f, hand])
        huv = inp.camera.project(geo.transform(T_cw, Hs))
        ok = np.all(np.isfinite(huv), axis=-1)
        clearance = float(np.min(np.linalg.norm(huv[ok] - uv, axis=-1))) if np.any(ok) else 1e9
        if best is None or clearance >= best[0] - 1e-9:
            best = (clearance, f, uv, X[2])
    if best is None:
        return None
    clearance, f, uv, z = best
    snapped = None
    if bool(gcfg.get("snap", True)):
        r_m = float(gcfg.get("snap_radius_m", 0.04))
        uvd = inp.to_depth_px(uv)
        r_px = max(1.0, inp.cam_d.fx * r_m / z)
        h, w = inp.shape
        vv, uu = np.mgrid[0:h, 0:w]
        disc = (uu - uvd[0]) ** 2 + (vv - uvd[1]) ** 2 <= r_px**2
        Xs, pix = fu.backproject(inp.metric_depth(f), disc & ~inp.hand_mask(f), inp.rays, inp.min_depth, inp.max_depth)
        if len(Xs):
            W = geo.transform(inp.T_wc[f], Xs)
            dist = np.linalg.norm(W - point, axis=1)
            k = int(np.argmin(dist))
            if dist[k] <= r_m:
                snapped = float(dist[k])
                uv = inp.to_full_px(pix[k, ::-1].astype(np.float64))
    half = float(inp.camera.fx * float(gcfg.get("prompt_box_m", 0.05)) / z)
    return {
        "source": "grasp", "frame": int(f), "x": float(uv[0]), "y": float(uv[1]),
        "box": [float(uv[0] - half), float(uv[1] - half), float(uv[0] + half), float(uv[1] + half)],
        "point_world": point.tolist(), "hand": "left" if hand == 0 else "right",
        "onset": onset, "release": int(episode["release"]), "hand_clearance_px": float(clearance),
        "snap_m": snapped,
    }


def manual_prompt(inp: Inputs, item: Any) -> dict[str, Any]:
    f, x, y = int(item["frame"]), float(item["x"]), float(item["y"])
    prompt: dict[str, Any] = {"source": "manual", "frame": f, "x": x, "y": y}
    if "box" in item:
        prompt["box"] = [float(v) for v in item["box"]]
    if inp.has(f):
        uvd = inp.to_depth_px(np.array([x, y]))
        u, v = int(round(uvd[0])), int(round(uvd[1]))
        h, w = inp.shape
        if 0 <= u < w and 0 <= v < h:
            d = inp.metric_depth(f)[max(0, v - 1):v + 2, max(0, u - 1):u + 2]
            if np.any(np.isfinite(d)):
                X = inp.rays[v, u] * float(np.nanmedian(d))
                prompt["point_world"] = geo.transform(inp.T_wc[f], X).tolist()
    return prompt


# =============================================================================================
# Masks
# =============================================================================================


def gt_prompt_masks(ep: Episode, cfg: Any, inp: Inputs, prompts: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    """``mask_backend = "gt"``: each prompt gets the ground-truth masks of the object under its
    pixel (or, when the pixel misses every object, of the nearest object inside the prompt box;
    ``hit`` is then False)."""
    from ..backends import gt as gtb

    gt = ep.gt
    if gt is None:
        raise PipelineError("objects.mask_backend = 'gt' needs a synthetic recording (gt/)")
    h, w = inp.shape
    H = gt.depth().shape[1]
    sub = cfg.merged({"depth": {"downsample": H / h}})
    per_object = gtb.masks(ep, sub)
    ids = gt.object_ids
    out: list[dict[str, Any] | None] = []
    for prompt in prompts:
        f = int(prompt["frame"])
        x, y = int(round(prompt["x"])), int(round(prompt["y"]))
        hit_k, hit = None, False
        best = np.inf
        for k in range(len(ids)):
            m = gt.mask(f, k)
            if 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and m[y, x]:
                hit_k, hit = k, True
                break
            box = prompt.get("box")
            if box is not None and np.any(m):
                ys, xs = np.nonzero(m)
                inside = (xs >= box[0]) & (xs <= box[2]) & (ys >= box[1]) & (ys <= box[3])
                if np.any(inside):
                    dmin = float(np.min(np.hypot(xs[inside] - x, ys[inside] - y)))
                    if dmin < best:
                        best, hit_k = dmin, k
        if hit_k is None:
            log.warning("gt masks: prompt at frame %d (%.0f, %.0f) hits no object", f, prompt["x"], prompt["y"])
            out.append(None)
            continue
        data = per_object[ids[hit_k]]
        out.append({"frames": data["frames"], "packed": data["packed"], "shape": (h, w), "hit": hit, "label": ids[hit_k]})
    return out


def prompt_masks(ep: Episode, cfg: Any, inp: Inputs, prompts: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    backend = str(cfg.get("objects.mask_backend", "sam2"))
    if not prompts:
        return []
    if backend == "gt":
        return gt_prompt_masks(ep, cfg, inp, prompts)
    if backend == "sam2":
        from ..backends import sam2 as sam2b

        return sam2b.video_masks(ep, cfg, prompts, inp.shape)
    raise PipelineError(f"unknown objects.mask_backend {backend!r} (gt or sam2)")


class MaskSet:
    """Packed masks of one object at depth resolution, frame -> bool mask."""

    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        self.packed: dict[int, np.ndarray] = {}

    def add(self, frames: np.ndarray, packed: np.ndarray) -> None:
        for f, p in zip(np.asarray(frames).tolist(), packed):
            f = int(f)
            if f in self.packed:
                self.packed[f] = np.bitwise_or(self.packed[f], p)
            else:
                self.packed[f] = np.asarray(p, dtype=np.uint8)

    def get(self, f: int) -> np.ndarray | None:
        p = self.packed.get(int(f))
        return None if p is None else unpack_mask(p, self.shape)

    @property
    def frames(self) -> list[int]:
        return sorted(self.packed)


def mask_iou(a: MaskSet, b: MaskSet, max_frames: int = 30) -> float:
    common = sorted(set(a.packed) & set(b.packed))
    if not common:
        return 0.0
    if len(common) > max_frames:
        common = [common[i] for i in np.linspace(0, len(common) - 1, max_frames).round().astype(int)]
    ious = []
    for f in common:
        ma, mb = a.get(f), b.get(f)
        union = np.logical_or(ma, mb).sum()
        ious.append(np.logical_and(ma, mb).sum() / union if union else 0.0)
    return float(np.mean(ious))


def merge_prompts(prompts: list[dict[str, Any]], masks: list[dict[str, Any] | None], merge_iou: float,
                  shape: tuple[int, int]) -> list[dict[str, Any]]:
    """Group prompts whose masks overlap; returns ``[{"prompts": [...], "masks": MaskSet, "hits": [...]}]``."""
    sets: list[tuple[int, MaskSet]] = []
    for i, m in enumerate(masks):
        if m is None or len(m["frames"]) == 0:
            continue
        ms = MaskSet(shape)
        ms.add(m["frames"], m["packed"])
        sets.append((i, ms))
    parent = list(range(len(sets)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            if find(a) != find(b) and mask_iou(sets[a][1], sets[b][1]) >= merge_iou:
                parent[find(b)] = find(a)
    groups: dict[int, list[int]] = {}
    for a in range(len(sets)):
        groups.setdefault(find(a), []).append(a)
    objects = []
    for members in groups.values():
        ms = MaskSet(shape)
        for a in members:
            i, s = sets[a]
            for f, p in s.packed.items():
                ms.add(np.array([f]), p[None])
        idx = [sets[a][0] for a in members]
        objects.append({"prompts": [prompts[i] for i in idx], "masks": ms,
                        "hits": [masks[i].get("hit") for i in idx], "labels": [masks[i].get("label") for i in idx]})
    objects.sort(key=lambda o: min(p["frame"] for p in o["prompts"]))
    return objects


# =============================================================================================
# Geometry
# =============================================================================================


def fusion_frames(inp: Inputs, masks: MaskSet, episodes: list[dict[str, Any]], fcfg: Any) -> list[int]:
    """Frames whose masked depth is fused: the object must be static there, so only frames before
    its first grasp onset (``margin_s`` before it when that leaves at least 3). Raises
    :class:`PipelineError` (the object is left out, with the reason in ``grasps.json``) when fewer
    than 3 masked frames precede the onset: frames after it are never fused, because the object
    moves with the hand there."""
    first_onset = min((int(e["onset"]) for e in episodes), default=inp.T)
    have = [f for f in masks.frames if inp.has(f)]
    if first_onset < inp.T:
        cut = inp.t[first_onset] - float(fcfg.get("margin_s", 0.25)) + 1e-6
        frames = [f for f in have if f < first_onset and inp.t[f] <= cut]
    else:
        frames = list(have)
    if len(frames) < 3:
        frames = [f for f in have if f < first_onset]
    if len(frames) < 3:
        raise PipelineError(f"only {len(frames)} frame(s) with a mask and depth before the first grasp (onset frame "
                            f"{first_onset}); the shape is fused only from frames where the object is static "
                            "(it is grasped too early in the clip)")
    n_max = int(fcfg.get("max_frames", 120))
    if len(frames) > n_max:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, n_max).round().astype(int)]
    return frames


def fuse_object(inp: Inputs, masks: MaskSet, frames: list[int], fcfg: Any, erode_px: int) -> dict[str, Any]:
    """Surface points + normals (world) of the object from ``frames``."""
    per_frame = {f: inp.object_points(f, masks.get(f), erode_px) for f in frames}
    chunks = [p for p in per_frame.values() if len(p)]
    allp = np.concatenate(chunks) if chunks else np.zeros((0, 3))
    if len(allp) < 10:
        raise PipelineError("objects: too few depth points on the object")
    voxel = float(fcfg.get("voxel_m", 0.004))
    method = str(fcfg.get("method", "tsdf"))
    info: dict[str, Any] = {"frames": len(frames), "raw_points": int(len(allp)), "method": method}
    P = N = None
    if method == "tsdf":
        margin = float(fcfg.get("bounds_margin_m", 0.04))
        lo = np.percentile(allp, 2, axis=0) - margin
        hi = np.percentile(allp, 98, axis=0) + margin
        size_cap = float(fcfg.get("max_size_m", 0.8))
        centre = 0.5 * (lo + hi)
        lo = np.maximum(lo, centre - size_cap / 2)
        hi = np.minimum(hi, centre + size_cap / 2)
        vol = fu.TSDFVolume(lo, hi, voxel, float(fcfg.get("trunc_m", 0.04)))
        carve = bool(fcfg.get("carve", True))
        for f in frames:
            m = masks.get(f)
            hand = inp.hand_mask(f)
            obj = fu.erode(m, erode_px) & ~hand
            free = (~fu.dilate(m, 2) & ~hand) if carve else None
            vol.integrate(inp.metric_depth(f), obj, inp.cam_d, inp.T_wc[f], free)
        P, N = vol.extract_surface(float(fcfg.get("min_weight", 4.0)))
        info.update(voxel_m=vol.voxel, grid=vol.dims.tolist(), tsdf_points=int(len(P)))
        if len(P) < 30:
            log.warning("objects: TSDF gave %d surface points; using the point cloud instead", len(P))
            P = N = None
            info["method"] = "points (tsdf fallback)"
    if P is None:
        P, _ = fu.voxel_downsample(allp, voxel)
        N = None
    keep = fu.statistical_outlier_mask(P, int(fcfg.get("outlier_k", 12)), float(fcfg.get("outlier_std", 2.0)))
    P = P[keep]
    N = N[keep] if N is not None else fu.estimate_normals(P, 16, toward=inp.T_wc[frames[len(frames) // 2], :3, 3])
    info["points"] = int(len(P))
    centre_cam = np.mean([inp.T_wc[f, :3, 3] for f in frames], axis=0)
    return {"points": P, "normals": N, "info": info, "view_centre": centre_cam, "raw": allp}


def complete_shape(P: np.ndarray, table: dict[str, Any], fcfg: Any, view_centre: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Points of the completed shape (seen surface + support extrusion + optional mirror)."""
    info: dict[str, Any] = {"extruded": False, "mirrored": False}
    pts = P.copy()
    if table.get("found"):
        n, off = np.asarray(table["normal"]), float(table["offset"])
        h = pl.plane_distance(pts, n, off)
        below = h < 0
        pts[below] -= h[below, None] * n  # nothing of a resting object is below its support
        h = pl.plane_distance(pts, n, off)
        gap = float(np.percentile(h, 2))
        info["support_gap_m"] = gap
        if bool(fcfg.get("extrude", True)) and gap <= float(fcfg.get("extrude_max_gap_m", 0.06)):
            pts = np.concatenate([pts, fu.extrude_to_plane(pts, n, off, int(fcfg.get("extrude_layers", 0)))])
            info["extruded"] = True
    if str(fcfg.get("symmetry", "none")) == "view":
        c = P.mean(axis=0)
        v = c - view_centre
        up = np.asarray(table.get("normal", [0, 0, 1.0]), dtype=np.float64)
        v = v - (v @ up) * up
        if np.linalg.norm(v) > 1e-6:
            pts = np.concatenate([pts, fu.mirror_points(pts, c, v)])
            info["mirrored"] = True
    return pts, info


def build_mesh(pts: np.ndarray, fused: dict[str, Any], ep: Episode, cfg: Any, inp: Inputs, masks: MaskSet,
               frames: list[int], table: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    """Visual mesh (world coordinates) from the configured backend; falls back to the hull when a
    backend fails or gives an open mesh."""
    backend = str(cfg.get("objects.mesh_backend", "hull"))
    if backend == "open3d":
        from ..backends import mesh_open3d

        try:
            V, F = mesh_open3d.reconstruct(pts, fu.estimate_normals(pts, 16, toward=fused["view_centre"]),
                                           cfg.section("objects.open3d"))
            if hl.is_closed_manifold(F):
                return V, F, "open3d"
            log.warning("objects: open3d mesh is not closed; using the convex hull")
        except Exception as error:  # noqa: BLE001 - optional backend
            log.warning("objects: open3d meshing failed (%s); using the convex hull", error)
    elif backend == "image_to_3d":
        from ..backends import image_to_3d

        try:
            f = frames[len(frames) // 2]
            V, F = image_to_3d.generate_aligned(ep, cfg, f, masks.get(f), inp, pts)
            if hl.is_closed_manifold(F):
                return V, F, "image_to_3d"
            log.warning("objects: generated mesh is not closed; using the convex hull")
        except Exception as error:  # noqa: BLE001
            log.warning("objects: image-to-3D failed (%s); using the convex hull", error)
    elif backend != "hull":
        raise PipelineError(f"unknown objects.mesh_backend {backend!r} (hull, open3d, image_to_3d)")
    return robust_hull(pts, cfg.section("objects.fusion"), table) + ("hull",)


def robust_hull(pts: np.ndarray, fcfg: Any, table: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Convex hull that ignores the noisiest extreme points (:func:`core.hull.support_polytope`,
    ``fusion.hull_trim_frac`` of the points per direction; 0 = the plain hull)."""
    frac = float(fcfg.get("hull_trim_frac", 0.008))
    if frac <= 0:
        return hl.hull_mesh(pts)
    up = np.asarray(table["normal"]) if table.get("found") else np.array([0.0, 0.0, 1.0])
    exact = -up[None] if table.get("found") else None
    # the footprint axes of the plain hull are among the directions, so the object-frame extents
    # come from trimmed supports exactly along them
    V0, _ = hl.hull_mesh(pts)
    R0 = fu.footprint_axes(V0, up, str(fcfg.get("frame_axes", "min_area_rect")))
    extra = np.concatenate([R0.T, -R0[:, :2].T, (up if exact is None else up)[None]])
    if exact is None:
        extra = np.concatenate([extra, -up[None]])
    V, F, _ = hl.support_polytope(pts, n_dirs=int(fcfg.get("hull_dirs", 256)), extra_dirs=extra,
                                  trim=max(3, int(round(frac * len(pts)))), exact_dirs=exact)
    return V, F


def collision_parts(V: np.ndarray, F: np.ndarray, cfg: Any) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    backend = str(cfg.get("objects.collision_backend", "hull"))
    if backend == "coacd":
        from ..backends import coacd as coacd_b

        try:
            return coacd_b.decompose(V, F, cfg.section("objects.coacd")), "coacd"
        except Exception as error:  # noqa: BLE001
            log.warning("objects: CoACD failed (%s); using the convex hull", error)
    elif backend != "hull":
        raise PipelineError(f"unknown objects.collision_backend {backend!r} (hull, coacd)")
    return [hl.hull_mesh(V)], "hull"


# =============================================================================================
# Tracking
# =============================================================================================


class Tracker:
    """ICP of an object's observed points (one frame or many stacked) to its model (object frame)."""

    def __init__(self, inp: Inputs, masks: MaskSet, model_pts: np.ndarray, model_nrm: np.ndarray, tcfg: Any,
                 erode_px: int) -> None:
        self.inp, self.masks, self.tcfg, self.erode_px = inp, masks, tcfg, erode_px
        self.model_pts, self.model_nrm = model_pts, model_nrm
        self.index = KNN(model_pts)
        self._obs: dict[int, np.ndarray] = {}

    def observe(self, f: int) -> np.ndarray:
        if f not in self._obs:
            m = self.masks.get(f)
            self._obs[f] = np.zeros((0, 3)) if m is None or not self.inp.has(f) else self.inp.object_points(f, m, self.erode_px)
        return self._obs[f]

    def _icp(self, src: np.ndarray, T_init_obj_src: np.ndarray, iterations: int | None, dof: str = "6") -> dict[str, Any]:
        c = self.tcfg
        return icp_point_to_plane(
            src, self.model_pts, self.model_nrm, T_init_obj_src, index=self.index,
            max_iterations=int(c.get("max_iterations", 25)) if iterations is None else iterations,
            max_correspondence=float(c.get("max_correspondence_m", 0.03)), robust=str(c.get("robust", "tukey")),
            min_points=int(c.get("min_points", 15)), dof=dof,
        )

    def fitness(self, f: int, T_world_obj: np.ndarray) -> float:
        """Fraction of the frame's points within ``max_correspondence_m`` of the posed model (NaN
        without points)."""
        src = self.observe(f)
        if len(src) == 0:
            return float("nan")
        return float(self._icp(src, geo.inv_T(T_world_obj), 0)["fitness"])

    def register_many(self, frames: list[int], poses: dict[int, np.ndarray], max_frames: int = 60,
                      dof: str = "6") -> dict[str, Any]:
        """One rigid correction for the stacked points of ``frames``, each brought into the object
        frame with its current pose ``poses[f]``. Returns ``{"C" (object-frame correction: new
        pose = pose @ inv(C)), "accepted", "fitness", "rmse", "n", "frames"}``."""
        c = self.tcfg
        use = [f for f in frames if len(self.observe(f))]
        if len(use) > max_frames:
            use = [use[i] for i in np.linspace(0, len(use) - 1, max_frames).round().astype(int)]
        if not use:
            return {"C": np.eye(4), "accepted": False, "fitness": 0.0, "rmse": float("nan"), "n": 0, "frames": 0}
        src = np.concatenate([geo.transform(geo.inv_T(poses[f]), self.observe(f)) for f in use])
        res = self._icp(src, np.eye(4), None, dof)
        C = res["T"]
        jump_t = float(np.linalg.norm(C[:3, 3]))
        jump_r = float(np.degrees(np.linalg.norm(geo.so3_log(C[:3, :3]))))
        accepted = (res["ok"] and res["fitness"] >= float(c.get("min_fitness", 0.5))
                    and jump_t <= float(c.get("max_jump_m", 0.05)) and jump_r <= float(c.get("max_jump_deg", 25.0)))
        return {"C": C, "accepted": bool(accepted), "fitness": res["fitness"], "rmse": res["rmse"], "n": len(src),
                "frames": len(use), "jump_m": jump_t, "jump_deg": jump_r}


def rest_on_table(T: np.ndarray, vertices: np.ndarray, table: dict[str, Any], max_tilt_deg: float) -> tuple[np.ndarray, bool]:
    """When the object's z axis is within ``max_tilt_deg`` of the table normal, make it exactly
    upright and put its lowest vertex on the table plane."""
    if not table.get("found"):
        return T, False
    n = np.asarray(table["normal"], dtype=np.float64)
    z = T[:3, 2]
    if np.degrees(np.arccos(np.clip(z @ n, -1.0, 1.0))) > max_tilt_deg:
        return T, False
    R = geo.rot_from_two_vectors(z, n) @ T[:3, :3]
    t = T[:3, 3].copy()
    low = float(np.min(pl.plane_distance(vertices @ R.T + t, n, float(table["offset"]))))
    return geo.make_T(R, t - low * n), True


def _inside_polygon(xy: np.ndarray, poly: np.ndarray) -> bool:
    """``xy`` inside the convex polygon ``poly`` (either orientation)."""
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(poly) < 3:
        return False
    edges = np.roll(poly, -1, axis=0) - poly
    rel = np.asarray(xy, dtype=np.float64)[None] - poly
    cross = edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0]
    return bool(np.all(cross >= 0) or np.all(cross <= 0))


def lift_above_table(T: np.ndarray, vertices: np.ndarray, table: dict[str, Any]) -> tuple[np.ndarray, float]:
    """An object whose centre is over the table polygon cannot be below the table plane: move it
    along the normal so that its lowest vertex is on the plane (no change when it is above, or not
    over the table). Returns the pose and the lift (m)."""
    if not table.get("found") or not _inside_polygon(T[:2, 3], table.get("polygon", [])):
        return T, 0.0
    n = np.asarray(table["normal"], dtype=np.float64)
    low = float(np.min(pl.plane_distance(vertices @ T[:3, :3].T + T[:3, 3], n, float(table["offset"]))))
    if low >= 0.0:
        return T, 0.0
    out = T.copy()
    out[:3, 3] -= low * n
    return out, -low


POSE_SOURCES = ("measured", "assumed_static", "hand_motion", "interpolated")
MEASURED, ASSUMED_STATIC, HAND_MOTION, INTERPOLATED = range(4)


def track_object(inp: Inputs, tracker: Tracker, T0: np.ndarray, episodes: list[dict[str, Any]], tcfg: Any,
                 table: dict[str, Any], vertices: np.ndarray, fused_frames: Any = ()) -> dict[str, Any]:
    """Poses ``T_world_obj[T]`` (see the module docstring), ``valid``, ``fitness``, ``pose_source``
    (:data:`POSE_SOURCES`: ``measured`` = the frame's own depth points defined the pose (a fused
    frame, or a static frame of an accepted ICP), ``assumed_static`` = a static pose held without
    the frame's own points, ``hand_motion`` = moved with the grasping hand, ``interpolated`` = inside
    a grasp where the hand is missing) and statistics."""
    T = inp.T
    poses = np.repeat(T0[None], T, axis=0)
    valid = np.zeros(T, dtype=bool)
    source = np.full(T, ASSUMED_STATIC, dtype=np.int8)
    fused = {int(f) for f in fused_frames}
    stats: dict[str, Any] = {"segments": []}
    segments: list[list[int]] = []
    for e in sorted(episodes, key=lambda e: int(e["onset"])):
        a, b, hand = int(e["onset"]), int(e["release"]), int(e["hand"])
        if segments and a <= segments[-1][1]:
            if hand == segments[-1][2]:
                segments[-1][1] = max(segments[-1][1], b)
                continue
            if b <= segments[-1][1]:
                continue  # the other hand helps inside the current grasp: keep following the first hand
            segments[-1][1] = a - 1  # hand-over: the new hand takes the object from its onset on
            if segments[-1][1] < segments[-1][0]:
                segments.pop()
        segments.append([a, b, hand])
    sigma = float(tcfg.get("smooth_frames", 2.0))
    snap_tilt = float(tcfg.get("rest_max_tilt_deg", 15.0))
    no_penetration = bool(tcfg.get("no_table_penetration", True))
    cur = T0.copy()

    def static_segment(a: int, b: int, first: bool) -> None:
        nonlocal cur
        if b < a:
            return
        if first:
            poses[a:b + 1] = T0
            valid[a:b + 1] = True
            for f in range(a, b + 1):
                source[f] = MEASURED if f in fused else ASSUMED_STATIC
            stats["segments"].append({"kind": "static", "frames": [a, b], "mode": "fused model"})
            return
        frames = list(range(a, b + 1))
        upright = table.get("found") and np.degrees(np.arccos(np.clip(cur[:3, 2] @ np.asarray(table["normal"]), -1, 1))) <= snap_tilt
        dof = str(tcfg.get("static_dof_upright", "4")) if upright else "6"
        res = tracker.register_many(frames, {f: cur for f in frames}, dof=dof)
        info = {"kind": "static", "frames": [a, b], "dof": dof, "n_frames": res["frames"],
                **{k: res[k] for k in ("accepted", "fitness", "n")}}
        if res["accepted"]:
            cur = cur @ geo.inv_T(res["C"])
            cur, snapped = rest_on_table(cur, vertices, table, snap_tilt) if snap_tilt > 0 else (cur, False)
            info["mode"] = "stacked icp" + (" + rest on table" if snapped else "")
            for f in frames:
                source[f] = MEASURED if len(tracker.observe(f)) else ASSUMED_STATIC
        else:
            info["mode"] = "hold"
        poses[a:b + 1] = cur
        valid[a:b + 1] = res["accepted"]
        stats["segments"].append(info)

    seg_start = 0
    for a, b, hand in segments:
        static_segment(seg_start, a - 1, seg_start == 0)
        pre = max(0, a - 2)
        frames = list(range(pre, b + 1))
        H = gr.hand_rigid_motion(inp.joints[:, hand], frames)
        okH = np.all(np.isfinite(H), axis=(-2, -1))
        info: dict[str, Any] = {"kind": "grasp", "frames": [a, b], "hand": "left" if hand == 0 else "right"}
        if okH.sum() < 2:
            poses[a:b + 1] = cur
            source[a:b + 1] = INTERPOLATED
            info["mode"] = "hold (no hand)"
            stats["segments"].append(info)
            seg_start = b + 1
            continue
        anchor = [i for i, f in enumerate(frames) if f <= a and okH[i]] or [int(np.flatnonzero(okH)[0])]
        X = average_poses(np.stack([geo.inv_T(H[i]) @ cur for i in anchor]))
        seg = {f: H[i] @ X for i, f in enumerate(frames) if f >= a}
        dof = str(tcfg.get("grasp_dof", "3"))
        res = tracker.register_many(list(range(a, b + 1)), seg, dof=dof) if dof != "0" else {"accepted": False}
        info.update({k: res.get(k) for k in ("accepted", "fitness", "n")}, dof=dof, n_frames=res.get("frames"))
        if res["accepted"]:
            X = X @ geo.inv_T(res["C"])
            info["mode"] = "hand motion + stacked icp"
        else:
            info["mode"] = "hand motion"
        raw = np.repeat(cur[None], len(frames), axis=0)
        last = cur
        for i, f in enumerate(frames):
            if f >= a and okH[i]:
                last = H[i] @ X
            raw[i] = last
        sm = gr.smooth_poses(raw, okH, sigma)
        lifts = []
        for i, f in enumerate(frames):
            if f >= a:
                poses[f] = sm[i]
                valid[f] = bool(okH[i])
                source[f] = HAND_MOTION if okH[i] else INTERPOLATED
                if no_penetration:
                    poses[f], lift = lift_above_table(poses[f], vertices, table)
                    lifts.append(lift)
        if lifts:
            info["lifted_frames"] = int(np.sum(np.asarray(lifts) > 0))
            info["max_lift_m"] = float(np.max(lifts))
        cur = poses[b].copy()
        stats["segments"].append(info)
        seg_start = b + 1
    static_segment(seg_start, T - 1, seg_start == 0)
    fitness = np.array([tracker.fitness(f, poses[f]) for f in range(T)])
    return {"T_world_obj": poses, "valid": valid, "fitness": fitness, "pose_source": source, "stats": stats}


# =============================================================================================
# Run
# =============================================================================================


def _py(x: Any) -> Any:
    """JSON-ready copy: numpy -> lists/floats, NaN -> None."""
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.ndarray):
        return _py(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _object(ep: Episode, cfg: Any, inp: Inputs, group: dict[str, Any], oid: str, k: int, table: dict[str, Any],
            meta: dict[str, Any], out_root: Any, shape: tuple[int, int], density: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Geometry, tracking and outputs of one object (see the module docstring)."""
    ocfg, fcfg, tcfg = cfg.section("objects"), cfg.section("objects.fusion"), cfg.section("objects.tracking")
    erode_px = int(ocfg.get("erode_px", 1))
    obj_eps = [p["_episode"] for p in group["prompts"] if "_episode" in p]
    masks: MaskSet = group["masks"]
    frames = fusion_frames(inp, masks, obj_eps, fcfg)
    fused = fuse_object(inp, masks, frames, fcfg, erode_px)
    pts, comp = complete_shape(fused["points"], table, fcfg, fused["view_centre"])
    V, F, mesh_backend = build_mesh(pts, fused, ep, cfg, inp, masks, frames, table)
    up = np.asarray(table["normal"]) if table.get("found") else np.array([0.0, 0.0, 1.0])
    R = fu.footprint_axes(V, up, str(fcfg.get("frame_axes", "min_area_rect")))
    _, centroid = hl.mesh_volume_centroid(V, F)
    T0 = geo.make_T(R, centroid)
    V_obj = (V - centroid) @ R
    parts, collision_backend = collision_parts(V_obj, F, cfg)
    mass, com, inertia = hl.mesh_inertia(V_obj, F, density)
    model_pts, model_nrm = hl.sample_surface(V_obj, F, int(fcfg.get("model_points", 3000)), seed=k)
    tracker = Tracker(inp, masks, model_pts, model_nrm, tcfg, erode_px)
    tr = track_object(inp, tracker, T0, obj_eps, tcfg, table, V_obj, fused_frames=frames)
    # grasped from contact
    tip = gr.fingertip_distances(inp.joints, tr["T_world_obj"], KNN(model_pts))
    obj_speed = gr.speed(tr["T_world_obj"][:, :3, 3], inp.fps, 3, inp.t)
    wrist = np.stack([gr.speed(gr.gaussian_smooth(inp.joints[:, h, 0], 0.06 * inp.fps), inp.fps, 3, inp.t)
                      for h in range(2)], axis=1)
    grasped = gr.contact_grasped(tip, obj_eps, obj_speed, wrist, float(tcfg.get("contact_m", 0.035)),
                                 float(tcfg.get("min_contact_frac", 0.3)), float(tcfg.get("co_motion_mps", 0.05)))
    # ---- write
    folder = out_root / oid
    if folder.exists():
        shutil.rmtree(folder)
    mframes = np.array(masks.frames, dtype=np.int64)
    packed = np.stack([masks.packed[f] for f in mframes]) if len(mframes) else np.zeros((0, (shape[0] * shape[1] + 7) // 8), np.uint8)
    ep.save_npz(f"objects/{oid}/masks", meta, frames=mframes, packed=packed, shape=np.array(shape))
    hl.write_obj(folder / "mesh.obj", V_obj, F, f"{oid}: visual mesh ({mesh_backend}) in the object frame, metres")
    for j, (pv, pf) in enumerate(parts):
        hl.write_obj(folder / "collision" / f"part_{j}.obj", pv, pf, f"{oid}: convex collision part {j} ({collision_backend})")
    ep.save_npz(f"objects/{oid}/poses", meta, T_world_obj=tr["T_world_obj"], valid=tr["valid"],
                fitness=tr["fitness"], grasped=grasped, pose_source=tr["pose_source"],
                pose_source_names=np.array(POSE_SOURCES))
    extent = hl.mesh_extents(V_obj)
    props = {
        "volume_m3": float(mass / density), "mass_kg": float(mass), "com": com.tolist(),
        "friction": [float(v) for v in ocfg.get("friction", [0.8, 0.02, 0.001])],
        "density_kg_m3": density, "inertia": inertia.tolist(), "extent_m": extent.tolist(),
        "mesh_backend": mesh_backend, "collision_backend": collision_backend, "collision_parts": len(parts),
        "fusion": _py({**fused["info"], **comp, "frames_used": [frames[0], frames[-1]]}),
    }
    ep.save_json(f"objects/{oid}/props.json", _py(props), meta)
    first_prompt = {k2: v for k2, v in group["prompts"][0].items() if not k2.startswith("_")}
    item = _py({
        "id": oid, "prompt": first_prompt,
        "prompts": [{k2: v for k2, v in p.items() if not k2.startswith("_")} for p in group["prompts"]],
        "first_frame": int(mframes[0]) if len(mframes) else 0,
        "last_frame": int(mframes[-1]) if len(mframes) else 0,
        "grasps": [{"hand": "left" if e["hand"] == 0 else "right", "onset": e["onset"], "release": e["release"]} for e in obj_eps],
        "mask_backend": str(ocfg.get("mask_backend", "sam2")), "mask_prompt_hits": group["hits"],
        "mesh_backend": mesh_backend, "collision_backend": collision_backend,
    })
    summary = {"id": oid, "extent_m": [round(float(v), 4) for v in extent],
               "valid": int(tr["valid"].sum()), "grasped": int(grasped.any(axis=1).sum()),
               "segments": [s_["mode"] for s_ in tr["stats"]["segments"]]}
    log.info("objects: %s extent %s m, %d/%d valid poses, segments %s", oid, np.round(extent, 3).tolist(),
             int(tr["valid"].sum()), inp.T, tr["stats"]["segments"])
    return item, summary


def run(ep: Episode, cfg: Any) -> dict[str, Any]:
    ocfg = cfg.section("objects")
    gcfg, tabcfg = cfg.section("objects.grasp"), cfg.section("objects.table")
    inp = _load_inputs(ep, cfg)
    meta = ep.step_meta(backend=str(ocfg.get("mask_backend", "sam2")))

    # ---- grasps and prompts
    episodes: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    if bool(ocfg.get("auto_prompts", True)):
        near = _near_surface_fn(inp, float(gcfg.get("near_surface_m", 0.08)), float(gcfg.get("lookback_s", 0.6)))
        episodes, rejected = gr.detect_grasps(inp.joints, inp.fps, gcfg, near, t=inp.t)
        for e in episodes:
            pr = grasp_prompt(inp, e, gcfg)
            if pr is None:
                log.warning("objects: grasp at frame %d has no usable prompt frame", e["onset"])
                continue
            pr["_episode"] = e
            prompts.append(pr)
    for item in ocfg.get("prompts", []) or []:
        prompts.append(manual_prompt(inp, item))
    log.info("objects: %d grasp episodes (%d rejected), %d prompts", len(episodes), len(rejected), len(prompts))

    # ---- table
    P, Z = scene_points(inp, tabcfg)
    anchors = np.array([p["point_world"] for p in prompts if "point_world" in p]).reshape(-1, 3)
    table = find_table(P, Z, anchors, tabcfg)
    if not table["found"]:
        log.warning("objects: no horizontal support plane found")

    # ---- masks and objects
    shape = inp.shape
    results = prompt_masks(ep, cfg, inp, prompts)
    groups = merge_prompts(prompts, results, float(ocfg.get("merge_iou", 0.5)), shape)
    out_root = ep.file("objects/index.json").parent
    index: list[dict[str, Any]] = []
    summary_objects = []
    density = float(ocfg.get("density_kg_m3", 500.0))
    if not density > 0:
        raise PipelineError(f"objects.density_kg_m3 must be positive, got {density}")
    failed: list[dict[str, Any]] = []
    k = -1
    for group in groups:
        k += 1
        oid = f"obj{k}"
        try:
            item, summary = _object(ep, cfg, inp, group, oid, k, table, meta, out_root, shape, density)
        except (PipelineError, ValueError, np.linalg.LinAlgError) as error:
            log.error("objects: %s (prompt frame %d) failed: %s", oid, group["prompts"][0]["frame"], error)
            failed.append({"prompt": {k2: v for k2, v in group["prompts"][0].items() if not k2.startswith("_")},
                           "error": f"{type(error).__name__}: {error}"})
            k -= 1
            continue
        index.append(item)
        summary_objects.append(summary)
    # stale object folders of an earlier run
    keep = {item["id"] for item in index}
    for child in out_root.iterdir() if out_root.exists() else []:
        if child.is_dir() and child.name.startswith("obj") and child.name not in keep:
            shutil.rmtree(child)
    table_json = {
        "normal": table.get("normal", np.array([0.0, 0.0, 1.0])), "offset": table.get("offset"),
        "height_m": table.get("height_m", float(tabcfg.get("default_height_m", 0.75))),
        "polygon": table.get("polygon", np.zeros((0, 2))), "top_z": table.get("top_z"), "floor_z": table.get("floor_z"),
        "height_source": table.get("height_source", "default"), "found": table["found"],
        "plane": "normal . p + offset = 0 (world); height_m = top above the floor", "n_inliers": table.get("n_inliers", 0),
        "rms_m": table.get("rms_m"), "tilt_deg": table.get("tilt_deg"), "planes": table["planes"],
    }
    ep.save_json("objects/table.json", _py(table_json), meta)
    # extra (not in the section-5 table): the grasp detection, for inspection and evaluation
    ep.save_json("objects/grasps.json", _py({"episodes": episodes, "rejected": rejected, "failed_objects": failed,
                                              "prompts": [{k2: v for k2, v in p.items() if not k2.startswith("_")} for p in prompts]}), meta)
    ep.save_json("objects/index.json", index)
    return {"objects": len(index), "prompts": len(prompts), "grasps": len(episodes),
            "table_z": round(float(table["top_z"]), 3) if table.get("found") else None, "items": summary_objects}
