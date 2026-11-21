"""Step ``slam``: metric camera trajectory (visual only) -> ``derived/slam.npz``.

Output (PIPELINE.md section 5): ``T_slam_cam[T,4,4]`` (camera-to-SLAM-frame, metres; the first
valid frame is the identity; NaN where invalid), ``valid[T]``, ``inliers[T]`` (PnP inliers),
``is_keyframe[T]``, ``source`` (``"builtin"`` or ``"import:<file>"``), plus ``kf_of[T]`` (the
keyframe a frame was tracked against, -1 none) and ``quality`` (JSON string).

Backend ``builtin`` (``[slam]`` in ``configs/slam.toml``)
---------------------------------------------------------
1. Frames are decoded grey, downsampled by ``slam.downsample`` (0 = auto: ``width //
   auto_width``, at least 1; 720 px -> 2, 360 px -> 1).
2. Hands are masked: the convex hull of each hand's 2D keypoints (``hands2d.npz``) extended along
   the forearm (``hand_forearm``) and dilated by ``hand_margin_px`` (processing-grid pixels), plus
   an image border of ``border_px``. Tracks that
   enter the mask are dropped.
3. Keyframes lift GFTT corners to 3D with the depth map (``depth/depth_f16.npy``, median filtered
   (``depth_median``), bilinear, rejecting samples next to depth edges: ``max/min > 1 +
   depth_edge_rel`` over a ``depth_edge_window`` neighbourhood of a median-``depth_edge_median``
   copy, so that per-pixel noise is not taken for an edge) times
   ``hand3d.depth_scale_smooth`` of that frame (the per-frame factor that makes the depth metric).
   Without it the factor is 1.0, with a warning and ``quality.depth_scale = "none"``.
4. Every frame: pyramidal KLT from the previous frame with a forward-backward check, then
   RANSAC PnP (EPnP, threshold ``pnp_thresh_px`` in the processing grid) against the current
   keyframe's landmarks on undistorted points (K = I) and LM refinement on the inliers. Outlier
   tracks (moving objects, bad depth) are dropped.
5. A new keyframe is made when the PnP inliers fall below ``kf_min_inliers`` or
   ``kf_track_ratio`` x the keyframe's landmarks, or the camera rotated more than
   ``kf_rotation_deg`` / moved more than ``kf_translation_rel`` x the median landmark depth since
   the keyframe (only on frames that have depth). It re-lifts the surviving tracks with its own
   depth and detects new corners in empty grid cells.
6. Pose graph over keyframes: the tracking edge to the previous keyframe plus ORB edges
   (mutual ratio-test matches, RANSAC PnP) to the ``edge_back`` previous keyframes in both
   directions (3D points of either keyframe, so two depth maps constrain every pair). Edges must
   agree with the odometry within ``edge_max_rot_deg`` / ``edge_max_trans_m``. Information:
   rotation ``sigma_rot / sqrt(n_inliers)``, translation ``sqrt(sigma_trans^2 / n + (scale_rel |t|)^2)``
   (the depth-scale error does not shrink with more points). Levenberg-Marquardt on SE(3)
   (:mod:`..core.pose_graph`) with a Huber kernel. Non-keyframes follow their keyframe:
   ``T_slam_cam = T_slam_kf (optimised) @ T_kf_cam (PnP)``.
7. Tracking loss: relocalisation by ORB matching against the last ``reloc_keyframes`` keyframes;
   when that fails for ``max_lost_frames`` frames, a new keyframe is started at the last valid
   pose (``quality.reinitialised`` counts these; the trajectory has a gap there).

Accuracy (synthetic 8 s clips, 360x640, 0.75 m head path, fabricated depth with a 10 % scale
error, +-5 % drift and 3 % pixel noise, corrected by ``depth_scale_smooth`` up to +-2 %): ATE
~0.7 mm (SE(3)) / ~0.6 mm (Sim(3), scale 0.995), RPE 0.3 mm / 0.01 deg per frame. The limit is
the metric scale: translation inherits the error of ``depth_scale_smooth`` (a 3 % bias gives
3 mm SE(3) vs 1.3 mm Sim(3) ATE on the same clip). Real video adds rolling shutter, motion
blur, calibration error, smooth-but-biased learned depth and longer paths without loop
closures (edges reach only ``edge_back`` keyframes back): expect centimetres there.

Backend ``import``: a trajectory from MegaSaM / DROID-SLAM / MASt3R-SLAM
------------------------------------------------------------------------
``slam.import_path`` (absolute or relative to the recording) is a TUM file, one pose per line,
camera-to-world::

    timestamp tx ty tz qx qy qz qw

``slam.import_time`` says what the timestamps are: ``"seconds"`` (episode seconds, i.e. video
frame ``i`` at ``timeline.t[i]``) or ``"frames"`` (video frame indices; what you get when the tool
ran on the frames extracted in order). Set it explicitly. ``"auto"`` decides only when that is
unambiguous: ``"frames"`` when every stamp is an integer and there is one stamp per video frame,
``"seconds"`` when some stamp is not an integer; whole-number stamps of another count (a
keyframe-only file) are an error asking for the unit. ``slam.import_time_offset`` (in the
stamps' unit: seconds or frames) is added to the stamps (the old name ``import_time_offset_s``
is still read, with a deprecation warning). Frames match the nearest stamp within
``import_max_dt_s`` (0 = half a frame); frames between two stamps less than ``import_max_gap_s``
apart are interpolated (slerp/lerp), which covers keyframe-only trajectories. The first valid pose
is moved to the identity. Scale: ``slam.import_scale`` is a number (multiplies translations) or
``"auto"``: the Sim(3) (Umeyama) scale to the builtin trajectory, computed in the same run.

Exporting (check against the version of the tool you run; all three give camera-to-world poses):

* Frames for the tools: ``python -c "from glassesdata_pipeline.episode import Episode; import cv2;
  [cv2.imwrite(f'frames/{i:06d}.png', im) for i, im in Episode('REC').frames()]"`` (or ``ffmpeg -i
  video.mp4 frames/%06d.png``, numbered from 1: set ``import_time_offset = -1``).
  Use the camera intrinsics in ``derived/camera.json``; undistort first if the tool assumes a
  pinhole without distortion.
* MASt3R-SLAM: its ``save_traj`` writes ``logs/<sequence>.txt`` in TUM format for the
  keyframes; the stamps are the dataset's timestamps (frame indices for an image folder):
  ``import_time = "frames"``; keyframe-only poses are interpolated (``import_max_gap_s``).
  Its scale is arbitrary: ``import_scale = "auto"``.
* DROID-SLAM: ``traj = droid.terminate(stream)`` returns ``(N, 7)`` = ``tx ty tz qx qy qz qw``
  camera-to-world for every frame of the stream; write ``np.savetxt('droid.txt',
  np.c_[np.arange(N), traj])`` (frame indices) -> ``import_time = "frames"``, scale ``"auto"``.
* MegaSaM: the final ``.npz`` holds camera-to-world matrices (``cam_c2w``, ``(N, 4, 4)``) for the
  input frames; convert with ``glassesdata_pipeline.core.align.write_tum('megasam.txt',
  np.arange(N), cam_c2w)`` -> ``import_time = "frames"``. MegaSaM is metric only up to its depth
  prior: keep ``import_scale = "auto"`` unless you trust it (then 1.0).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import geometry as geo
from ..core import pose_graph as pg
from ..core import vo
from ..episode import Episode
from ..errors import PipelineError

NAME = "slam"
VERSION = 1
REQUIRES = ["ingest", "hands2d", "depth", "hand3d"]
OUTPUTS = ["slam.npz"]
CONFIG_SECTIONS = ["slam"]

log = logging.getLogger(__name__)

_BASE_INPUTS = [
    "video.mp4", "derived/timeline.npz", "derived/camera.json", "derived/hands2d.npz",
    "derived/depth/index.npz", "derived/depth/depth_f16.npy", "derived/hand3d.npz",
]


def INPUTS(cfg) -> list[str]:  # noqa: N802 - the runner's contract name
    s = cfg.get("slam", {}) or {}
    files = list(_BASE_INPUTS)
    if s.get("backend", "builtin") == "import" and s.get("import_path"):
        files.append(str(s.get("import_path")))
    return files


DEFAULTS: dict[str, Any] = {
    "backend": "builtin",
    "downsample": 0,
    "auto_width": 400,
    "border_px": 8,
    "hand_margin_px": 10.0,
    "hand_forearm": 1.6,
    "max_features": 500,
    "min_distance_px": 7.0,
    "cell_px": 32,
    "per_cell": 5,
    "gftt_quality": 0.005,
    "klt_window": 21,
    "klt_levels": 3,
    "fb_thresh_px": 0.8,
    "depth_median": 3,
    "depth_edge_rel": 0.08,
    "depth_edge_median": 5,
    "depth_edge_window": 3,
    "min_depth_m": 0.15,
    "max_depth_m": 8.0,
    "pnp_thresh_px": 1.5,
    "pnp_iterations": 300,
    "min_inliers": 20,
    "max_step_m": 0.25,
    "kf_min_inliers": 100,
    "kf_track_ratio": 0.55,
    "kf_rotation_deg": 8.0,
    "kf_translation_rel": 0.06,
    "orb_features": 1000,
    "edge_back": 5,
    "edge_min_inliers": 30,
    "edge_min_ratio": 0.3,
    "edge_max_rot_deg": 5.0,
    "edge_max_trans_m": 0.10,
    "sigma_rot": 0.01,
    "sigma_trans": 0.02,
    "scale_rel": 0.02,
    "pg_iterations": 30,
    "pg_huber": 3.0,
    "reloc_keyframes": 5,
    "max_lost_frames": 12,
    "import_path": "",
    "import_time": "auto",
    "import_time_offset": 0.0,
    "import_max_dt_s": 0.0,
    "import_max_gap_s": 0.5,
    "import_scale": "auto",
}


def params(cfg) -> dict[str, Any]:
    """``[slam]`` merged over :data:`DEFAULTS` (the deprecated ``import_time_offset_s`` is mapped
    to ``import_time_offset``)."""
    section = dict((cfg.get("slam", {}) if hasattr(cfg, "get") else {}) or {})
    if "import_time_offset_s" in section:
        old = float(section.pop("import_time_offset_s"))
        new = float(section.get("import_time_offset", 0.0))
        log.warning("slam.import_time_offset_s is deprecated: use slam.import_time_offset (the same value, in the "
                    "stamps' unit)")
        if new != 0.0 and new != old:
            raise PipelineError(f"slam.import_time_offset ({new}) and the deprecated import_time_offset_s ({old}) differ")
        section["import_time_offset"] = old
    return {**DEFAULTS, **section}


# =============================================================================================
# Step
# =============================================================================================


def run(ep: Episode, cfg) -> dict:
    p = params(cfg)
    backend = str(p["backend"])
    if backend == "builtin":
        res = run_builtin(ep, p)
    elif backend == "import":
        res = run_import(ep, p)
    else:
        raise PipelineError(f"slam.backend must be 'builtin' or 'import', got {backend!r}")
    quality = res.pop("quality")
    ep.save_npz("slam", ep.step_meta(backend=backend), quality=json.dumps(quality), **res)
    valid = res["valid"]
    return {
        "backend": backend,
        "valid": f"{int(valid.sum())}/{len(valid)}",
        "keyframes": int(np.sum(res["is_keyframe"])),
        "path_m": round(_path_length(res["T_slam_cam"], valid), 3),
        **({"scale": quality["import_scale"]} if "import_scale" in quality else {}),
    }


def _path_length(T: np.ndarray, valid: np.ndarray) -> float:
    p = T[valid, :3, 3]
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 0.0


def anchor_first_valid(T: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``T`` re-expressed so that the first valid pose is the identity (invalid rows NaN)."""
    out = np.full_like(T, np.nan)
    if not np.any(valid):
        return out
    first = int(np.flatnonzero(valid)[0])
    inv0 = geo.inv_T(T[first])
    out[valid] = inv0 @ T[valid]
    out[first] = np.eye(4)
    return out


# =============================================================================================
# Inputs
# =============================================================================================


def _depth_scale(ep: Episode, T: int) -> tuple[np.ndarray, str]:
    """Per-frame factor that makes the depth metric (``hand3d.depth_scale_smooth``), NaNs filled
    by interpolation; ``(ones, "none")`` when unavailable."""
    if ep.has("hand3d.npz"):
        d = ep.load_npz("hand3d")
        s = np.asarray(d.get("depth_scale_smooth", np.full(T, np.nan)), dtype=np.float64).reshape(-1)
        if len(s) == T:
            ok = np.isfinite(s) & (s > 0)
            if ok.sum() >= 1:
                idx = np.arange(T)
                filled = np.interp(idx, idx[ok], s[ok])
                return filled, "hand3d" if ok.all() else "hand3d_interpolated"
    log.warning("slam: no usable hand3d.depth_scale_smooth; the depth is taken as metric (scale 1.0)")
    return np.ones(T), "none"


class _Depth:
    """Access to ``derived/depth`` by video frame index."""

    def __init__(self, ep: Episode) -> None:
        if not (ep.has("depth/index.npz") and ep.has("depth/depth_f16.npy")):
            raise PipelineError("slam needs derived/depth (run the depth step)")
        index = ep.load_npz("depth/index.npz")
        self.frames = np.asarray(index["frames"], dtype=np.int64)
        self.data = np.load(ep.derived / "depth" / "depth_f16.npy", mmap_mode="r")
        self.row = {int(f): r for r, f in enumerate(self.frames)}
        self.size = (int(self.data.shape[2]), int(self.data.shape[1]))  # (w, h)

    def has(self, frame: int) -> bool:
        return int(frame) in self.row

    def get(self, frame: int) -> np.ndarray:
        return np.asarray(self.data[self.row[int(frame)]], dtype=np.float32)


def _hand_kp(ep: Episode, T: int) -> np.ndarray | None:
    if not ep.has("hands2d.npz"):
        log.warning("slam: no hands2d.npz; hands are not masked")
        return None
    kp = np.asarray(ep.load_npz("hands2d")["kp"], dtype=np.float64)
    return kp if kp.shape[0] == T else None


def _choose_downsample(p: dict, width: int) -> int:
    ds = int(p["downsample"] or 0)
    if ds > 0:
        return ds
    return max(1, int(width // int(p["auto_width"])) if width > int(p["auto_width"]) else 1)


# =============================================================================================
# Builtin visual odometry
# =============================================================================================


@dataclass
class _Keyframe:
    frame: int
    T: np.ndarray  # T_slam_kf
    n_landmarks: int
    median_depth: float
    orb_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    orb_X: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    orb_des: np.ndarray | None = None


class _Builtin:
    def __init__(self, ep: Episode, p: dict) -> None:
        self.ep = ep
        self.p = p
        tl = ep.timeline()
        self.t = np.asarray(tl["t"], dtype=np.float64)
        self.T = len(self.t)
        self.camera = ep.camera()
        self.full_size = self.camera.size
        self.ds = _choose_downsample(p, self.camera.width)
        self.gcam = self.camera if self.ds == 1 else self.camera.downsampled(self.ds)
        self.gsize = self.gcam.size
        self.depth = _Depth(ep)
        self.scale, self.scale_source = _depth_scale(ep, self.T)
        self.kp = _hand_kp(ep, self.T)
        self.border = vo.border_mask(self.gsize, int(p["border_px"]))
        self.thr = float(p["pnp_thresh_px"]) / self.gcam.fx
        # outputs
        self.T_kf_cam = np.full((self.T, 4, 4), np.nan)
        self.kf_of = np.full(self.T, -1, dtype=np.int64)
        self.inliers = np.zeros(self.T, dtype=np.int64)
        self.is_kf = np.zeros(self.T, dtype=bool)
        self.keyframes: list[_Keyframe] = []
        self.edges: list[pg.Edge] = []
        self.counts = {"relocalised": 0, "reinitialised": 0, "lost_frames": 0, "edges_odometry": 0,
                       "edges_orb": 0, "edges_rejected": 0, "edges_weak": 0}

    # ---------------------------------------------------------------- helpers

    def usable(self, i: int) -> np.ndarray:
        hm = vo.hand_mask(None if self.kp is None else self.kp[i], self.full_size, self.gsize,
                          float(self.p["hand_margin_px"]), float(self.p["hand_forearm"]))
        return np.where(hm > 0, 0, self.border).astype(np.uint8)

    def metric_depth(self, i: int) -> vo.DepthMap:
        p = self.p
        d = vo.prepare_depth(self.depth.get(i), int(p["depth_median"]), float(p["depth_edge_rel"]),
                             int(p["depth_edge_median"]), int(p["depth_edge_window"]))
        return d.scaled(float(self.scale[i]))

    def lift(self, pts: np.ndarray, depth: vo.DepthMap) -> np.ndarray:
        """3D points (camera frame) of grid pixels ``pts``; NaN rows where depth is unusable."""
        if len(pts) == 0:
            return np.zeros((0, 3))
        h, w = depth.shape
        uv = vo.grid_to_grid(pts, self.gsize, (w, h))
        z = vo.sample_depth(depth, uv, float(self.p["min_depth_m"]), float(self.p["max_depth_m"]))
        xy = self.gcam.undistort_points(pts)
        return np.stack([xy[:, 0] * z, xy[:, 1] * z, z], axis=1)

    def make_keyframe(self, i: int, img: np.ndarray, T_slam: np.ndarray, pts: np.ndarray,
                      usable: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        depth = self.metric_depth(i)
        p = self.p
        new = vo.detect(img, usable, pts, int(p["max_features"]), float(p["gftt_quality"]),
                        float(p["min_distance_px"]), int(p["cell_px"]), int(p["per_cell"]))
        pts = np.concatenate([pts.reshape(-1, 2), new], axis=0)
        X = self.lift(pts, depth)
        keep = np.all(np.isfinite(X), axis=1)
        pts, X = pts[keep], X[keep]
        orb_pts, des = vo.orb_features(img, usable, int(p["orb_features"]))
        orb_X = self.lift(orb_pts, depth) if len(orb_pts) else np.zeros((0, 3))
        kf = _Keyframe(i, T_slam.copy(), int(len(X)), float(np.median(X[:, 2])) if len(X) else 1.0,
                       self.gcam.undistort_points(orb_pts) if len(orb_pts) else np.zeros((0, 2)), orb_X, des)
        self.keyframes.append(kf)
        self.is_kf[i] = True
        self.kf_of[i] = len(self.keyframes) - 1
        self.T_kf_cam[i] = np.eye(4)
        return pts, X

    def orb_relative(self, a: _Keyframe, b_xy: np.ndarray, b_des, guess: np.ndarray | None = None):
        """PnP of keyframe ``a``'s ORB 3D points in an image with ORB ``b_xy``/``b_des``:
        ``(ok, T_b_a, n_inliers, n_matches)``."""
        ia, ib = vo.match_descriptors(a.orb_des, b_des)
        if len(ia) == 0:
            return False, np.eye(4), 0, 0
        X = a.orb_X[ia]
        good = np.all(np.isfinite(X), axis=1)
        ia, ib = ia[good], ib[good]
        if len(ia) < int(self.p["edge_min_inliers"]):
            return False, np.eye(4), 0, len(ia)
        ok, T_b_a, inl = vo.pnp(a.orb_X[ia], b_xy[ib], self.thr * 1.5, int(self.p["pnp_iterations"]),
                                min_inliers=int(self.p["edge_min_inliers"]), guess=guess)
        n = int(inl.sum())
        ok = ok and n >= int(self.p["edge_min_inliers"]) and n >= float(self.p["edge_min_ratio"]) * len(ia)
        return ok, T_b_a, n, len(ia)

    def add_edge(self, i: int, j: int, Z: np.ndarray, n_inl: int, kind: str) -> None:
        p = self.p
        info = pg.information(n_inl, float(np.linalg.norm(Z[:3, 3])), float(p["sigma_rot"]),
                              float(p["sigma_trans"]), float(p["scale_rel"]))
        self.edges.append(pg.Edge(i, j, Z, info, kind))

    def extra_edges(self, k: int) -> None:
        """ORB edges between keyframe ``k`` and the ``edge_back`` previous keyframes."""
        p = self.p
        new = self.keyframes[k]
        for j in range(k - 1, max(-1, k - 1 - int(p["edge_back"])), -1):
            old = self.keyframes[j]
            predicted = geo.inv_T(old.T) @ new.T  # T_j_k from odometry
            candidates = []
            # reverse: 3D points of the new keyframe seen in the old image -> T_j_k directly
            ok, T_j_k, n, _ = self.orb_relative(new, old.orb_xy, old.orb_des, predicted)
            if ok:
                candidates.append((T_j_k, n, "orb_rev"))
            if j < k - 1:  # forward: the tracking edge already uses the previous keyframe's depth
                ok, T_k_j, n, _ = self.orb_relative(old, new.orb_xy, new.orb_des, geo.inv_T(predicted))
                if ok:
                    candidates.append((geo.inv_T(T_k_j), n, "orb_fwd"))
            for Z, n, kind in candidates:
                d_rot = math.degrees(float(geo.rotation_angle(Z[:3, :3], predicted[:3, :3])))
                d_t = float(np.linalg.norm(Z[:3, 3] - predicted[:3, 3]))
                if d_rot <= float(p["edge_max_rot_deg"]) and d_t <= float(p["edge_max_trans_m"]):
                    self.add_edge(j, k, Z, n, kind)
                    self.counts["edges_orb"] += 1
                else:
                    self.counts["edges_rejected"] += 1

    def relocalise(self, img: np.ndarray, usable: np.ndarray):
        """``(ok, T_slam_cam, kf_index, T_kf_cam, n)`` from ORB matching against recent keyframes."""
        pts, des = vo.orb_features(img, usable, int(self.p["orb_features"]))
        if des is None:
            return False, None, -1, None, 0
        xy = self.gcam.undistort_points(pts)
        best = None
        for j in range(len(self.keyframes) - 1, max(-1, len(self.keyframes) - 1 - int(self.p["reloc_keyframes"])), -1):
            ok, T_c_j, n, _ = self.orb_relative(self.keyframes[j], xy, des)
            if ok and (best is None or n > best[2]):
                best = (j, T_c_j, n)
        if best is None:
            return False, None, -1, None, 0
        j, T_c_j, n = best
        T_j_c = geo.inv_T(T_c_j)
        return True, self.keyframes[j].T @ T_j_c, j, T_j_c, n

    # ---------------------------------------------------------------- main loop

    def run(self) -> dict[str, Any]:
        p = self.p
        prev_img = None
        pts = np.zeros((0, 2))
        X = np.zeros((0, 3))
        kf: _Keyframe | None = None
        T_cur_kf_prev: np.ndarray | None = None
        last_valid_T: np.ndarray | None = None
        lost = 0
        for i, img in self.ep.frames(downsample=self.ds, gray=True):
            if i >= self.T:
                break
            usable = self.usable(i)
            if kf is None or lost > int(p["max_lost_frames"]):
                if self.depth.has(i):
                    T0 = np.eye(4) if last_valid_T is None else last_valid_T
                    if kf is not None:
                        self.counts["reinitialised"] += 1
                    pts, X = self.make_keyframe(i, img, T0, np.zeros((0, 2)), usable)
                    if kf is not None:  # weak link so the graph stays connected
                        prev_k = len(self.keyframes) - 2
                        Z = geo.inv_T(self.keyframes[prev_k].T) @ T0
                        self.edges.append(pg.Edge(prev_k, len(self.keyframes) - 1, Z, np.eye(6) * 1e-2, "weak"))
                        self.counts["edges_weak"] += 1
                    kf = self.keyframes[-1]
                    T_cur_kf_prev = np.eye(4)
                    last_valid_T = T0
                    self.inliers[i] = kf.n_landmarks
                    lost = 0
                prev_img = img
                continue

            # ---- track
            if len(pts):
                new_pts, ok = vo.track(prev_img, img, pts, int(p["klt_window"]), int(p["klt_levels"]),
                                       float(p["fb_thresh_px"]))
                ui = np.clip(np.round(new_pts[:, 0]).astype(int), 0, self.gsize[0] - 1)
                vi = np.clip(np.round(new_pts[:, 1]).astype(int), 0, self.gsize[1] - 1)
                ok &= usable[vi, ui] > 0
                pts, X = new_pts[ok], X[ok]
            success = False
            if len(pts) >= int(p["min_inliers"]):
                xy = self.gcam.undistort_points(pts)
                good = np.all(np.isfinite(xy), axis=1)
                okp, T_cur_kf, inl_g = vo.pnp(X[good], xy[good], self.thr, int(p["pnp_iterations"]),
                                              min_inliers=int(p["min_inliers"]), guess=T_cur_kf_prev)
                if okp:
                    T_kf_cur = geo.inv_T(T_cur_kf)
                    T_new = kf.T @ T_kf_cur
                    step = np.linalg.norm(T_new[:3, 3] - last_valid_T[:3, 3]) if last_valid_T is not None else 0.0
                    if step <= float(p["max_step_m"]) * max(1, lost + 1):
                        success = True
                        inl = np.zeros(len(pts), bool)
                        inl[np.flatnonzero(good)[inl_g]] = True
                        pts, X = pts[inl], X[inl]
                        n_inl = int(inl.sum())
            if success:
                lost = 0
                self.T_kf_cam[i] = T_kf_cur
                self.kf_of[i] = len(self.keyframes) - 1
                self.inliers[i] = n_inl
                T_cur_kf_prev = T_cur_kf
                last_valid_T = T_new
                angle = math.degrees(float(geo.rotation_angle(np.eye(3), T_kf_cur[:3, :3])))
                dist = float(np.linalg.norm(T_kf_cur[:3, 3]))
                need = (n_inl < int(p["kf_min_inliers"]) or n_inl < float(p["kf_track_ratio"]) * kf.n_landmarks
                        or angle > float(p["kf_rotation_deg"]) or dist > float(p["kf_translation_rel"]) * kf.median_depth)
                if need and self.depth.has(i):
                    k_prev = len(self.keyframes) - 1
                    pts, X = self.make_keyframe(i, img, T_new, pts, usable)
                    self.add_edge(k_prev, k_prev + 1, T_kf_cur, n_inl, "odometry")
                    self.counts["edges_odometry"] += 1
                    self.extra_edges(k_prev + 1)
                    kf = self.keyframes[-1]
                    T_cur_kf_prev = np.eye(4)
            else:
                self.counts["lost_frames"] += 1
                okr, T_r, j, T_j_c, n = self.relocalise(img, usable)
                if okr and self.depth.has(i):
                    self.counts["relocalised"] += 1
                    self.inliers[i] = n
                    k_prev = len(self.keyframes) - 1
                    pts, X = self.make_keyframe(i, img, T_r, np.zeros((0, 2)), usable)
                    self.add_edge(j, k_prev + 1, T_j_c, n, "reloc")
                    kf = self.keyframes[-1]
                    T_cur_kf_prev = np.eye(4)
                    last_valid_T = T_r
                    lost = 0
                else:
                    lost += 1
            prev_img = img
        return self.finish()

    def finish(self) -> dict[str, Any]:
        p = self.p
        K = len(self.keyframes)
        T_kf = np.stack([k.T for k in self.keyframes]) if K else np.zeros((0, 4, 4))
        report: dict[str, Any] = {}
        if K > 1 and self.edges:
            before = pg.edge_errors(T_kf, self.edges)
            T_opt, report = pg.optimize(T_kf, self.edges, fixed=(0,), iters=int(p["pg_iterations"]),
                                        huber=float(p["pg_huber"]))
            after = pg.edge_errors(T_opt, self.edges)
            report.update({
                "edge_rot_deg_before": float(np.median(before["rot_deg"])),
                "edge_rot_deg_after": float(np.median(after["rot_deg"])),
                "edge_trans_mm_before": float(np.median(before["trans_m"]) * 1000),
                "edge_trans_mm_after": float(np.median(after["trans_m"]) * 1000),
            })
            T_kf = T_opt
        valid = self.kf_of >= 0
        T = np.full((self.T, 4, 4), np.nan)
        for i in np.flatnonzero(valid):
            T[i] = T_kf[self.kf_of[i]] @ self.T_kf_cam[i]
        valid &= np.all(np.isfinite(T), axis=(1, 2))
        T = anchor_first_valid(T, valid)
        quality = {
            "backend": "builtin",
            "downsample": self.ds,
            "grid": list(self.gsize),
            "depth_scale": self.scale_source,
            "hands_masked": self.kp is not None,
            "keyframes": K,
            "edges": len(self.edges),
            **self.counts,
            "pose_graph": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in report.items()},
            "median_inliers": float(np.median(self.inliers[valid])) if np.any(valid) else 0.0,
        }
        if self.scale_source == "none":
            log.warning("slam: trajectory scale is that of the raw depth (no hand3d depth scale)")
        return {
            "T_slam_cam": T,
            "valid": valid,
            "inliers": self.inliers,
            "is_keyframe": self.is_kf & valid,
            "kf_of": self.kf_of,
            "source": "builtin",
            "quality": quality,
        }


def run_builtin(ep: Episode, p: dict) -> dict[str, Any]:
    """The builtin visual odometry (see the module docstring); returns the ``slam.npz`` arrays
    plus ``quality`` (dict)."""
    return _Builtin(ep, p).run()


# =============================================================================================
# Import backend
# =============================================================================================


def _interp_pose(Ta: np.ndarray, Tb: np.ndarray, alpha: float) -> np.ndarray:
    R = Ta[:3, :3] @ geo.so3_exp(alpha * geo.so3_log(Ta[:3, :3].T @ Tb[:3, :3]))
    return geo.make_T(R, (1 - alpha) * Ta[:3, 3] + alpha * Tb[:3, 3])


def associate_trajectory(frame_t: np.ndarray, stamps: np.ndarray, poses: np.ndarray, max_dt: float,
                         max_gap: float) -> tuple[np.ndarray, np.ndarray]:
    """Poses ``(T, 4, 4)`` at the frame times: nearest stamp within ``max_dt`` (one-to-one,
    :func:`core.align.associate`), else interpolated between the surrounding stamps when they
    are at most ``max_gap`` apart. Returns ``(T_out, valid)``."""
    from ..core.align import associate

    T = len(frame_t)
    out = np.full((T, 4, 4), np.nan)
    valid = np.zeros(T, bool)
    order = np.argsort(stamps)
    stamps, poses = stamps[order], poses[order]
    ia, ib = associate(frame_t, stamps, max_dt)
    out[ia] = poses[ib]
    valid[ia] = True
    if len(stamps) >= 2:
        for i in np.flatnonzero(~valid):
            k = int(np.searchsorted(stamps, frame_t[i]))
            if 0 < k < len(stamps) and stamps[k] - stamps[k - 1] <= max_gap:
                a = (frame_t[i] - stamps[k - 1]) / (stamps[k] - stamps[k - 1])
                out[i] = _interp_pose(poses[k - 1], poses[k], float(a))
                valid[i] = True
    return out, valid


def import_time_unit(mode: str, stamps: np.ndarray, n_frames: int, path: Any = "") -> str:
    """``"frames"`` or ``"seconds"`` for ``slam.import_time`` (see the module docstring: ``"auto"``
    never guesses from whole numbers alone)."""
    if mode in ("frames", "seconds"):
        return mode
    if mode != "auto":
        raise PipelineError(f"slam.import_time must be 'seconds', 'frames' or 'auto', got {mode!r}")
    whole = bool(np.all(stamps == np.round(stamps)))
    if not whole:
        return "seconds"
    if len(stamps) == n_frames:
        return "frames"
    raise PipelineError(f"{path}: the {len(stamps)} stamps are whole numbers but the video has {n_frames} frames, so "
                        "they could be frame indices (a keyframe-only file) or whole seconds; set slam.import_time "
                        "to 'frames' or 'seconds'")


def run_import(ep: Episode, p: dict) -> dict[str, Any]:
    from ..core.align import read_tum, umeyama

    raw = str(p.get("import_path") or "")
    if not raw:
        raise PipelineError("slam.backend = 'import' needs slam.import_path (a TUM trajectory)")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ep.path / path
    if not path.is_file():
        raise PipelineError(f"slam.import_path {path} does not exist")
    stamps, poses = read_tum(path)
    if len(stamps) == 0:
        raise PipelineError(f"{path} has no poses")
    tl = ep.timeline()
    t = np.asarray(tl["t"], dtype=np.float64)
    T = len(t)
    fps = 1.0 / float(np.median(np.diff(t))) if T > 1 else 24.0
    mode = import_time_unit(str(p["import_time"]), stamps, T, path)
    stamps = stamps + float(p["import_time_offset"])
    if mode == "frames":
        frame_t = np.arange(T, dtype=np.float64)
        max_dt = float(p["import_max_dt_s"]) * fps if float(p["import_max_dt_s"]) > 0 else 0.5
        max_gap = float(p["import_max_gap_s"]) * fps
    elif mode == "seconds":
        frame_t = t
        max_dt = float(p["import_max_dt_s"]) if float(p["import_max_dt_s"]) > 0 else 0.5 / fps
        max_gap = float(p["import_max_gap_s"])
    else:
        raise PipelineError(f"slam.import_time must be 'seconds', 'frames' or 'auto', got {mode!r}")
    T_raw, valid = associate_trajectory(frame_t, stamps, poses, max_dt, max_gap)
    if not np.any(valid):
        raise PipelineError(f"{path}: no pose matches a video frame (import_time = {mode!r})")
    T_anchor = anchor_first_valid(T_raw, valid)
    quality: dict[str, Any] = {"backend": "import", "file": str(path), "import_time": mode,
                               "matched": int(valid.sum()), "stamps": int(len(stamps))}
    scale_cfg = p["import_scale"]
    if isinstance(scale_cfg, str) and scale_cfg == "auto":
        builtin = run_builtin(ep, {**p, "backend": "builtin"})
        common = valid & builtin["valid"]
        if common.sum() < 3:
            raise PipelineError("slam.import_scale = 'auto' needs >= 3 frames valid in both trajectories")
        try:
            s, _, _ = umeyama(T_anchor[common, :3, 3], builtin["T_slam_cam"][common, :3, 3], with_scale=True)
        except ValueError as error:
            raise PipelineError(f"slam.import_scale = 'auto': {error}") from None
        quality["scale_reference"] = "builtin"
    else:
        s = float(scale_cfg)
    T_out = T_anchor.copy()
    T_out[valid, :3, 3] *= s
    quality["import_scale"] = float(s)
    inliers = np.zeros(T, np.int64)
    return {"T_slam_cam": T_out, "valid": valid, "inliers": inliers, "is_keyframe": np.zeros(T, bool),
            "kf_of": np.full(T, -1, np.int64), "source": f"import:{path.name}", "quality": quality}
