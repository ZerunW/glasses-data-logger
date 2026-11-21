"""Step ``world``: gravity-aligned camera trajectory and smoothed metric hands in the world.

Outputs (PIPELINE.md section 5):

* ``camera_traj.npz``: ``T_world_cam[T,4,4]`` (NaN where invalid), ``valid[T]``.
  ``T_world_cam = T_world_slam @ T_slam_cam`` with ``T_world_slam = [R_world_slam | 0]`` from
  ``imu.npz``, then re-anchored to the section-1 convention: origin at the camera centre of the
  first valid frame, that camera's optical axis (projected on the horizontal plane) along +x.
  Without a gravity estimate (``imu.quality.gravity = "none"`` or no ``imu.npz``) the world is the
  SLAM frame (first camera = identity, *not* gravity aligned) with a warning and
  ``gravity_aligned = False``.
* ``hands_world.npz``: ``raw[T,2,21,3]`` = ``hand3d.joints_cam`` mapped by ``T_world_cam``
  (NaN where the hand or the camera is invalid); ``joints`` = ``raw`` smoothed per hand with the
  RTS smoother (``core.smoothing.rts_smooth``: constant-acceleration model per coordinate,
  bidirectional chi2 gating of outliers, gaps up to ``max_gap_s`` filled) and then refined to the
  profile's bone lengths (``core.smoothing.refine_bone_lengths``); ``observed[T,2]`` = raw hand
  present and accepted by the gate (at least half its joints); ``filled[T,2]`` = smoothed hand
  present where it was not observed; ``bone_lengths[2,20]`` = target bone lengths used.

Tuning (``configs/world.toml``; measured on synthetic recordings with ~1 cm joint noise):
``q_accel`` is the white-jerk spectral density of the motion model (m^2/s^5): larger follows fast
hand motion, smaller smooths more; ``r_meas`` the joint measurement variance (m^2, about the
squared per-coordinate error of ``hand3d``); ``max_gap_s`` the longest gap that is bridged
(longer gaps stay NaN). When the gate rejects more than ``max_reject_fraction`` of a hand's
samples, the motion is faster than ``q_accel`` allows (the filter diverges); the hand is smoothed
again with ``q_accel`` x ``q_ladder_factor`` (at most ``q_ladder_steps`` times). ``gate_chi2`` rejects samples further than ``sqrt(gate_chi2)`` predicted
standard deviations.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from .. import geometry as geo
from .. import skeleton as sk
from ..core.smoothing import refine_bone_lengths, rts_smooth
from ..episode import Episode
from ..errors import PipelineError

NAME = "world"
VERSION = 2  # 2: a hand seen in a single frame is kept (not smoothed)
REQUIRES = ["ingest", "hand3d", "slam", "imu"]
INPUTS = ["derived/timeline.npz", "derived/slam.npz", "derived/imu.npz", "derived/hand3d.npz", "derived/profile.json"]
OUTPUTS = ["camera_traj.npz", "hands_world.npz"]
CONFIG_SECTIONS = ["world"]

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "q_accel": 20.0,
    "r_meas": 5e-5,
    "gate_chi2": 16.0,
    "max_gap_s": 0.5,
    "init_var": [1e-2, 1.0, 10.0],
    "bone_lengths": "profile",
    "refine_bones": True,
    "w_data": 1.0,
    "w_bone": 100.0,
    "w_smooth": 0.1,
    "bone_iters": 10,
    "min_observed_joints": 0.5,
    "max_reject_fraction": 0.2,
    "q_ladder_factor": 10.0,
    "q_ladder_steps": 2,
}


def params(cfg) -> dict[str, Any]:
    section = cfg.get("world", {}) if hasattr(cfg, "get") else {}
    return {**DEFAULTS, **dict(section or {})}


def run(ep: Episode, cfg) -> dict:
    p = params(cfg)
    cam = camera_trajectory(ep)
    meta = ep.step_meta(backend="gravity" if cam["gravity_aligned"] else "slam_frame")
    ep.save_npz("camera_traj", {**meta, "gravity_aligned": cam["gravity_aligned"]},
                T_world_cam=cam["T_world_cam"], valid=cam["valid"])
    summary: dict[str, Any] = {"camera_valid": f"{int(cam['valid'].sum())}/{len(cam['valid'])}",
                               "gravity_aligned": cam["gravity_aligned"]}
    if ep.has("hand3d.npz"):
        hands = hands_world(ep, cam["T_world_cam"], cam["valid"], p)
        quality = hands.pop("quality")
        ep.save_npz("hands_world", meta, quality=json.dumps(quality), **hands)
        summary.update({"observed": hands["observed"].sum(axis=0).tolist(), "filled": hands["filled"].sum(axis=0).tolist()})
    else:
        log.warning("world: no hand3d.npz; hands_world.npz is written empty")
        T = len(cam["valid"])
        ep.save_npz("hands_world", meta, joints=np.full((T, 2, 21, 3), np.nan), raw=np.full((T, 2, 21, 3), np.nan),
                    observed=np.zeros((T, 2), bool), filled=np.zeros((T, 2), bool),
                    bone_lengths=np.tile(sk.bone_lengths(ep.profile()), (2, 1)), quality=json.dumps({"hands": "none"}))
    return summary


# =============================================================================================


def camera_trajectory(ep: Episode) -> dict[str, Any]:
    """``{"T_world_cam", "valid", "gravity_aligned"}`` (see the module docstring)."""
    if not ep.has("slam.npz"):
        raise PipelineError("world needs derived/slam.npz (run the slam step)")
    slam = ep.load_npz("slam")
    T_sc = np.asarray(slam["T_slam_cam"], dtype=np.float64)
    valid = np.asarray(slam["valid"], dtype=bool) & np.all(np.isfinite(T_sc), axis=(1, 2))
    R_ws = np.eye(3)
    aligned = False
    if ep.has("imu.npz"):
        imu = ep.load_npz("imu")
        quality = json.loads(imu.get("quality", "{}") or "{}")
        if quality.get("gravity", "none") not in ("none", "imu_no_slam", "phone_no_slam"):
            R_ws = np.asarray(imu["R_world_slam"], dtype=np.float64)
            aligned = True
    if not aligned:
        log.warning("world: no gravity estimate (imu.npz missing or gravity = none); the world frame is the "
                    "SLAM frame, NOT gravity aligned")
    T = np.full_like(T_sc, np.nan)
    if np.any(valid):
        T_w = geo.make_T(R_ws) @ T_sc[valid]
        first = T_w[0]
        # Section-1 convention: origin at the first valid camera, its heading along +x (only a
        # rotation about z, so gravity stays -z).
        R_fix = geo.gravity_align(first[:3, :3]) if aligned else np.eye(3)
        T_fix = geo.make_T(R_fix, -R_fix @ first[:3, 3])
        T[valid] = T_fix @ T_w
    return {"T_world_cam": T, "valid": valid, "gravity_aligned": aligned}


def _bone_targets(ep: Episode, raw: np.ndarray, p: dict) -> np.ndarray:
    profile_lengths = np.asarray(ep.profile().get("bone_lengths_m") or sk.bone_lengths(ep.profile()), dtype=np.float64)
    out = np.tile(profile_lengths, (2, 1))
    if str(p["bone_lengths"]) == "data":
        for h in range(2):
            J = raw[:, h]
            lengths = np.stack([np.linalg.norm(J[:, i] - J[:, sk.PARENTS[i]], axis=-1) for i in range(1, 21)], axis=-1)
            med = np.nanmedian(np.where(np.isfinite(lengths), lengths, np.nan), axis=0) if np.isfinite(lengths).any() else None
            if med is not None:
                out[h] = np.where(np.isfinite(med), med, profile_lengths)
    return out


def hands_world(ep: Episode, T_world_cam: np.ndarray, cam_valid: np.ndarray, p: dict) -> dict[str, Any]:
    """The ``hands_world.npz`` arrays plus ``quality`` (dict)."""
    h3 = ep.load_npz("hand3d")
    J = np.asarray(h3["joints_cam"], dtype=np.float64)
    T = J.shape[0]
    hv = np.asarray(h3.get("valid", np.ones((T, 2), bool)), dtype=bool)
    t = np.asarray(ep.timeline()["t"], dtype=np.float64)[:T]
    ok = hv & cam_valid[:, None]
    raw = np.full_like(J, np.nan)
    for h in range(2):
        rows = np.flatnonzero(ok[:, h])
        if rows.size:
            raw[rows, h] = geo.transform(T_world_cam[rows][:, None], J[rows, h])
    targets = _bone_targets(ep, raw, p)
    joints = np.full_like(raw, np.nan)
    observed = np.zeros((T, 2), bool)
    quality: dict[str, Any] = {}
    for h in range(2):
        z = raw[:, h].reshape(T, -1)
        present = np.any(np.isfinite(z), axis=1)
        if present.sum() == 1:  # nothing to smooth with: keep the one measurement as it is
            f = int(np.flatnonzero(present)[0])
            keep = bool(np.all(np.isfinite(raw[f, h])))
            if keep:
                joints[f, h] = raw[f, h]
                observed[f, h] = True
            quality[["left", "right"][h]] = {"present": 1, "rejected": int(not keep), "smoothed": 0,
                                              "single_frame": True}
            continue
        if present.sum() == 0:
            continue
        # Adaptive process noise: when the gate rejects too many present hands, the motion is
        # faster than q_accel allows (the filter diverges and rejects everything after); retry
        # with q_accel x q_ladder_factor.
        q = float(p["q_accel"])
        for attempt in range(int(p["q_ladder_steps"]) + 1):
            x, _, inlier = rts_smooth(z, t, q_accel=q, r_meas=float(p["r_meas"]),
                                      gate_chi2=float(p["gate_chi2"]), max_gap_s=float(p["max_gap_s"]),
                                      init_var=tuple(float(v) for v in p["init_var"]))
            inl_joint = inlier.reshape(T, 21, 3).all(axis=-1)
            frac = inl_joint.sum(axis=1) / 21.0
            obs = present & (frac >= float(p["min_observed_joints"]))
            reject_frac = 1.0 - obs.sum() / max(int(present.sum()), 1)
            if reject_frac <= float(p["max_reject_fraction"]):
                break
            if attempt < int(p["q_ladder_steps"]):
                q *= float(p["q_ladder_factor"])
        xs = x.reshape(T, 21, 3)
        observed[:, h] = obs
        # A frame is kept only when every joint is defined (rts leaves NaN outside the bridged span).
        complete = np.all(np.isfinite(xs), axis=(1, 2))
        xs[~complete] = np.nan
        if bool(p["refine_bones"]) and complete.any():
            refined = refine_bone_lengths(xs, np.asarray(sk.PARENTS), targets[h], observed=np.isfinite(xs[..., 0]),
                                          w_data=float(p["w_data"]), w_bone=float(p["w_bone"]),
                                          w_smooth=float(p["w_smooth"]), iters=int(p["bone_iters"]))
            refined[~complete] = np.nan
            xs = refined
        joints[:, h] = xs
        rejected = present & ~observed[:, h]
        quality[["left", "right"][h]] = {"present": int(present.sum()), "rejected": int(rejected.sum()),
                                          "smoothed": int(complete.sum()), "q_accel": q}
        if q != float(p["q_accel"]):
            log.info("world: %s hand smoothed with q_accel = %g (gate rejected too many samples at %g)",
                     ["left", "right"][h], q, float(p["q_accel"]))
    smoothed_present = np.all(np.isfinite(joints), axis=(2, 3))
    filled = smoothed_present & ~observed
    return {"joints": joints, "raw": raw, "observed": observed, "filled": filled, "bone_lengths": targets,
            "quality": quality}
