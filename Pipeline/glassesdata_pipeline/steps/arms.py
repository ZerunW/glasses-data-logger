"""Step ``arms``: shoulders, elbows and wrists in the world -> ``derived/arms.npz``.

Output (PIPELINE.md section 5): ``shoulder[T,2,3]``, ``elbow[T,2,3]``, ``wrist[T,2,3]``,
``elbow_conf[T,2]``, ``method[T,2]`` int8 (0 none, 1 forearm cue, 2 prior only); plus
``torso_yaw[T]`` (rad), ``swivel[T,2]`` (``solve_elbow`` convention), ``cue_plane[T,2,4]``,
``cue_conf[T,2]`` and ``quality`` (JSON string).

* Wrists: joint 0 of ``hands_world.joints`` (smoothed).
* Shoulders: :func:`skeleton.shoulders_from_head` with the camera centre of ``camera_traj`` and
  the torso yaw = the head yaw smoothed by a zero-phase first-order filter (forward-backward
  exponential, time constant ``arms.torso_yaw_tau_s``; the torso does not follow every head turn).
  ``torso_yaw_tau_s = 0`` uses the instantaneous head yaw, which is what the synthetic ground
  truth does.
* Elbows: :func:`core.elbow.solve_elbow` with the profile's upper-arm and forearm lengths, a
  prior direction "down, turned outward by ``prior_outward_deg``" in the torso frame, and the
  forearm cue chosen by ``arms.cue``:

  - ``"none"``: prior and temporal smoothness only;
  - ``"gt"`` (synthetic recordings): the ground-truth forearm as an image detector would see it:
    the wrist and the furthest in-image point of the true forearm projected with the true camera,
    with Gaussian pixel noise ``gt_cue_sigma_px``; at least ``gt_cue_min_len_px`` of forearm must be
    visible;
  - ``"skin"`` / ``"region"`` / ``"depth"``: image-based (``core.forearm_cue.forearm_axis``) on the
    video frame (downsampled by ``cue_downsample``) and the depth map when there is one: skin
    colour (bare forearms), colour-agnostic region growing + depth continuity (sleeves too), or
    depth continuity only. Cues below ``cue_min_conf`` are dropped.

  The 2D axis (wrist pixel, far pixel) becomes a plane through the *estimated* camera centre
  (``core.forearm_cue.cue_plane``) plus the side the forearm extends to.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import numpy as np

from .. import geometry as geo
from .. import skeleton as sk
from ..core import forearm_cue as fc
from ..core.elbow import solve_elbow
from ..episode import Episode
from ..errors import PipelineError

NAME = "arms"
VERSION = 1
REQUIRES = ["world"]
OUTPUTS = ["arms.npz"]
CONFIG_SECTIONS = ["arms"]

log = logging.getLogger(__name__)


def INPUTS(cfg) -> list[str]:  # noqa: N802
    a = cfg.get("arms", {}) or {}
    files = ["derived/timeline.npz", "derived/camera_traj.npz", "derived/hands_world.npz", "derived/profile.json"]
    cue = str(a.get("cue", DEFAULTS["cue"]))
    if cue == "gt":
        files.append("gt/gt.npz")
    elif cue in fc.MODES:
        files += ["video.mp4", "derived/hands2d.npz", "derived/depth/index.npz", "derived/depth/depth_f16.npy",
                  "derived/hand3d.npz", "derived/camera.json"]
    return files


DEFAULTS: dict[str, Any] = {
    "torso_yaw_tau_s": 1.0,
    "prior_swivel_deg": 20.0,
    "cue": "region",
    "cue_min_conf": 0.3,
    "cue_downsample": 0,
    "cue_auto_width": 360,
    "cue_use_depth": True,
    "w_cue": 1.0,
    "w_prior": 0.2,
    "w_smooth": 2.0,
    "n_samples": 360,
    "max_swivel_speed": 6.0,
    "gt_cue_sigma_px": 2.0,
    "gt_cue_min_len_px": 15.0,
    "seed": 0,
    "forearm": {},
}


def params(cfg) -> dict[str, Any]:
    section = cfg.get("arms", {}) if hasattr(cfg, "get") else {}
    return {**DEFAULTS, **dict(section or {})}


def run(ep: Episode, cfg) -> dict:
    p = params(cfg)
    res = solve(ep, p)
    quality = res.pop("quality")
    ep.save_npz("arms", ep.step_meta(backend=str(p["cue"])), quality=json.dumps(quality), **res)
    m = res["method"]
    return {"cue": p["cue"], "elbows": int(np.sum(m > 0)), "with_cue": int(np.sum(m == 1)),
            "prior_only": int(np.sum(m == 2))}


# =============================================================================================
# Shoulders
# =============================================================================================


def smooth_yaw(t: np.ndarray, yaw: np.ndarray, valid: np.ndarray, tau: float) -> np.ndarray:
    """Zero-phase (forward-backward) first-order low-pass of the unwrapped yaw; invalid frames
    are interpolated first. ``tau <= 0`` returns the (interpolated) yaw unchanged."""
    out = np.full(len(t), np.nan)
    if not np.any(valid):
        return out
    idx = np.flatnonzero(valid)
    y = np.unwrap(yaw[idx])
    full = np.interp(t, t[idx], y)
    if tau <= 0 or len(t) < 2:
        return full
    def one_pass(x, tt):
        o = np.empty_like(x)
        o[0] = x[0]
        for k in range(1, len(x)):
            a = 1.0 - math.exp(-abs(tt[k] - tt[k - 1]) / tau)
            o[k] = o[k - 1] + a * (x[k] - o[k - 1])
        return o
    fwd = one_pass(full, t)
    return one_pass(fwd[::-1], t[::-1])[::-1]


def torso_poses(T_world_cam: np.ndarray, valid: np.ndarray, t: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Head poses with the yaw replaced by the smoothed torso yaw (only the heading and the
    position matter to :func:`skeleton.shoulders_from_head`). Returns ``(T_torso, yaw)``."""
    yaw = np.full(len(t), np.nan)
    yaw[valid] = geo.yaw_of(T_world_cam[valid, :3, :3])
    ys = smooth_yaw(t, yaw, valid, tau)
    level = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # camera looking along +x
    T = np.full_like(T_world_cam, np.nan)
    T[valid] = geo.make_T(geo.rot_z(ys[valid]) @ level, T_world_cam[valid, :3, 3])
    return T, ys


def prior_directions(shoulder: np.ndarray, wrist: np.ndarray, upper: float, fore: float, swivel_deg: float,
                     side: str) -> np.ndarray:
    """``(T, 3)`` prior elbow direction (from the circle centre) at the swivel angle ``swivel_deg``
    of :func:`skeleton.elbow_from_swivel` (0 = lowest point, positive = outward). Defined in the
    circle plane, so the prior means "this swivel" however the arm axis is oriented (a fixed 3D
    direction projected onto the circle plane would not: with the forearm pointing down the
    projection of "down" vanishes and the sideways part takes over). NaN where undefined."""
    a = math.radians(swivel_deg)
    e0 = sk.elbow_from_swivel(shoulder, wrist, upper, fore, a, side)
    e1 = sk.elbow_from_swivel(shoulder, wrist, upper, fore, a + math.pi, side)
    return e0 - e1


# =============================================================================================
# Forearm cues
# =============================================================================================


def _gt_cues(ep: Episode, p: dict, T_world_cam: np.ndarray, cam_valid: np.ndarray, wrist: np.ndarray):
    gt = ep.gt
    if gt is None:
        raise PipelineError("arms.cue = 'gt' only works on synthetic recordings (no gt/)")
    camera = ep.camera()
    Tg = np.asarray(gt["T_world_cam"], dtype=np.float64)
    Wg = np.asarray(gt["joints_world"], dtype=np.float64)[:, :, 0]
    Eg = np.asarray(gt["elbow_world"], dtype=np.float64)
    T = Tg.shape[0]
    rng = np.random.default_rng([int(p["seed"]), 31])
    planes = np.full((T, 2, 4), np.nan)
    sides = np.full((T, 2, 3), np.nan)
    conf = np.zeros((T, 2))
    s = np.linspace(0.0, 1.0, 41)
    sigma = float(p["gt_cue_sigma_px"])
    for i in range(T):
        if not cam_valid[i]:
            continue
        Tcw = geo.inv_T(Tg[i])
        for h in range(2):
            pts = Wg[i, h][None] + s[:, None] * (Eg[i, h] - Wg[i, h])[None]
            uv = camera.project(geo.transform(Tcw, pts))
            inside = camera.in_image(uv) & np.all(np.isfinite(uv), axis=1)
            if not inside[0]:
                continue
            last = int(np.argmax(~inside)) - 1 if not inside.all() else len(s) - 1
            if last < 1 or np.linalg.norm(uv[last] - uv[0]) < float(p["gt_cue_min_len_px"]):
                continue
            noise = rng.normal(scale=sigma, size=(2, 2))
            plane, side = fc.cue_plane(camera, T_world_cam[i], uv[0] + noise[0], uv[last] + noise[1], wrist[i, h])
            if np.all(np.isfinite(plane)):
                planes[i, h], sides[i, h], conf[i, h] = plane, side, 1.0
    return planes, sides, conf


def _image_cues(ep: Episode, p: dict, T_world_cam: np.ndarray, cam_valid: np.ndarray, wrist: np.ndarray, mode: str):
    camera = ep.camera()
    T = len(cam_valid)
    planes = np.full((T, 2, 4), np.nan)
    sides = np.full((T, 2, 3), np.nan)
    conf = np.zeros((T, 2))
    if not ep.has("hands2d.npz"):
        log.warning("arms: no hands2d.npz; no image forearm cue")
        return planes, sides, conf, {}
    kp = np.asarray(ep.load_npz("hands2d")["kp"], dtype=np.float64)
    ds = int(p["cue_downsample"] or 0)
    if ds <= 0:
        ds = max(1, int(round(camera.width / float(p["cue_auto_width"]))))
    grid = geo.downsample_size(camera.width, camera.height, ds)
    depth = None
    if bool(p["cue_use_depth"]) and ep.has("depth/index.npz") and ep.has("depth/depth_f16.npy"):
        index = ep.load_npz("depth/index.npz")
        depth = (dict((int(f), r) for r, f in enumerate(index["frames"])),
                 np.load(ep.derived / "depth" / "depth_f16.npy", mmap_mode="r"))
    cp = fc.CueParams.from_mapping(p.get("forearm"))
    fore = float(ep.profile()["forearm_length_m"])
    from ..core.vo import grid_to_grid
    import cv2

    stats = {"tried": 0, "found": 0, "accepted": 0}
    frames = [i for i in range(T) if cam_valid[i] and np.any(np.isfinite(kp[i, :, 0, 0]))]
    for i, img in ep.frames(frames, downsample=ds):
        d_img = None
        if depth is not None and i in depth[0]:
            d = np.asarray(depth[1][depth[0][i]], dtype=np.float32)
            d_img = cv2.resize(d, grid, interpolation=cv2.INTER_LINEAR)
        for h in range(2):
            if not np.isfinite(kp[i, h, 0, 0]) or not np.all(np.isfinite(wrist[i, h])):
                continue
            stats["tried"] += 1
            k = grid_to_grid(kp[i, h, :, :2], camera.size, grid)
            z = float((geo.inv_T(T_world_cam[i]) @ np.append(wrist[i, h], 1.0))[2])
            forearm_px = camera.fx * (grid[0] / camera.width) * fore / z if z > 0.05 else None
            cue = fc.forearm_axis(img, k, mode, d_img, cp, forearm_px)
            if not cue.ok:
                continue
            stats["found"] += 1
            if cue.conf < float(p["cue_min_conf"]):
                continue
            p0 = grid_to_grid(cue.p0, grid, camera.size)
            p1 = grid_to_grid(cue.p1, grid, camera.size)
            plane, side = fc.cue_plane(camera, T_world_cam[i], p0, p1, wrist[i, h])
            if np.all(np.isfinite(plane)):
                planes[i, h], sides[i, h], conf[i, h] = plane, side, cue.conf
                stats["accepted"] += 1
    return planes, sides, conf, stats


# =============================================================================================
# Solve
# =============================================================================================


def solve(ep: Episode, p: dict[str, Any]) -> dict[str, Any]:
    if not (ep.has("camera_traj.npz") and ep.has("hands_world.npz")):
        raise PipelineError("arms needs camera_traj.npz and hands_world.npz (run the world step)")
    cam = ep.load_npz("camera_traj")
    T_wc = np.asarray(cam["T_world_cam"], dtype=np.float64)
    cam_valid = np.asarray(cam["valid"], dtype=bool) & np.all(np.isfinite(T_wc), axis=(1, 2))
    hw = ep.load_npz("hands_world")
    joints = np.asarray(hw["joints"], dtype=np.float64)
    T = joints.shape[0]
    t = np.asarray(ep.timeline()["t"], dtype=np.float64)[:T]
    profile = ep.profile()
    quality: dict[str, Any] = {"cue": str(p["cue"]), "torso_yaw_tau_s": float(p["torso_yaw_tau_s"])}

    T_torso, yaw = torso_poses(T_wc, cam_valid, t, float(p["torso_yaw_tau_s"]))
    shoulder = np.full((T, 2, 3), np.nan)
    if cam_valid.any():
        shoulder[cam_valid] = sk.shoulders_from_head(T_torso[cam_valid], profile)
    wrist = joints[:, :, 0].copy()

    cue = str(p["cue"])
    planes = np.full((T, 2, 4), np.nan)
    sides = np.full((T, 2, 3), np.nan)
    cue_conf = np.zeros((T, 2))
    if cue == "gt":
        planes, sides, cue_conf = _gt_cues(ep, p, T_wc, cam_valid, wrist)
    elif cue in fc.MODES:
        planes, sides, cue_conf, stats = _image_cues(ep, p, T_wc, cam_valid, wrist, cue)
        quality["cue_stats"] = stats
    elif cue != "none":
        raise PipelineError(f"arms.cue must be none, gt, skin, region or depth, got {cue!r}")

    elbow = np.full((T, 2, 3), np.nan)
    swivel = np.full((T, 2), np.nan)
    conf = np.zeros((T, 2))
    method = np.zeros((T, 2), np.int8)
    dt = float(np.median(np.diff(t))) if T > 1 else 1.0 / 24.0
    upper, fore = float(profile["upper_arm_length_m"]), float(profile["forearm_length_m"])
    for h in range(2):
        S = shoulder[:, h]
        W = wrist[:, h]
        with np.errstate(invalid="ignore", divide="ignore"):
            prior = prior_directions(S, W, upper, fore, float(p["prior_swivel_deg"]), ("left", "right")[h])
        bad = ~np.all(np.isfinite(prior), axis=1) | (np.linalg.norm(np.nan_to_num(prior), axis=1) < 1e-6)
        prior[bad] = (0.0, 0.0, -1.0)
        e, phi, c, m = solve_elbow(S, W, upper, fore, prior_dir=prior, cue_plane=planes[:, h],
                                   cue_side=sides[:, h], w_cue=float(p["w_cue"]), w_prior=float(p["w_prior"]),
                                   w_smooth=float(p["w_smooth"]), n_samples=int(p["n_samples"]), dt=dt,
                                   max_swivel_speed=float(p["max_swivel_speed"]))
        elbow[:, h], swivel[:, h], conf[:, h], method[:, h] = e, phi, c, m
        quality[["left", "right"][h]] = {"elbows": int(np.sum(m > 0)), "cue_frames": int(np.sum(m == 1))}
    return {
        "shoulder": shoulder, "elbow": elbow, "wrist": wrist, "elbow_conf": conf, "method": method,
        "torso_yaw": yaw, "swivel": swivel, "cue_plane": planes, "cue_conf": cue_conf, "quality": quality,
    }
