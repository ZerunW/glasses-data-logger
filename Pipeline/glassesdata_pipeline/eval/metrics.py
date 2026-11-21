"""Metric functions on arrays (no file I/O). PIPELINE.md section 9.

Trajectory and joint metrics use :mod:`glassesdata_pipeline.core.align` (imported lazily); when it
is missing they return ``{"status": "pending", ...}`` instead of failing.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from .. import geometry as geo

PENDING = {"status": "pending", "reason": "glassesdata_pipeline.core.align is not available yet"}


def align_module() -> Any | None:
    """``glassesdata_pipeline.core.align`` or None."""
    try:
        return importlib.import_module("glassesdata_pipeline.core.align")
    except ImportError:
        return None


def _stats(err: np.ndarray, prefix: str = "") -> dict[str, Any]:
    err = np.asarray(err, dtype=np.float64)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return {f"{prefix}n": 0}
    return {
        f"{prefix}mean": float(err.mean()),
        f"{prefix}median": float(np.median(err)),
        f"{prefix}rmse": float(np.sqrt(np.mean(err**2))),
        f"{prefix}max": float(err.max()),
        f"{prefix}n": int(err.size),
    }


# ---------------------------------------------------------------- depth


def depth_errors(pred: np.ndarray, gt: np.ndarray, min_depth: float = 0.05, max_depth: float = 10.0) -> dict[str, Any]:
    """AbsRel, RMSE (m), delta < 1.25 over pixels with finite positive prediction and ground truth
    in ``[min_depth, max_depth]``."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    m = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt >= min_depth) & (gt <= max_depth)
    if not np.any(m):
        return {"n": 0}
    p, g = pred[m], gt[m]
    ratio = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "n": int(m.sum()),
    }


# ---------------------------------------------------------------- 2D keypoints


def keypoint2d_errors(kp: np.ndarray, gt_uv: np.ndarray, gt_visible: np.ndarray) -> dict[str, Any]:
    """Pixel error of detected joints the ground truth marks visible, and the detection rate
    (hands with >= 6 visible ground-truth joints that have a finite hand detection)."""
    kp = np.asarray(kp, dtype=np.float64)
    vis = np.asarray(gt_visible, dtype=bool)
    detected = np.all(np.isfinite(kp[..., :2]), axis=-1)
    m = vis & detected
    err = np.linalg.norm(kp[..., :2] - gt_uv, axis=-1)[m]
    present = vis.sum(axis=-1) >= 6
    found = np.any(detected, axis=-1)
    out = _stats(err, "px_")
    out["detection_rate"] = float(np.mean(found[present])) if np.any(present) else float("nan")
    out["false_detections"] = int(np.sum(found & ~np.any(vis, axis=-1)))
    return out


# ---------------------------------------------------------------- trajectories


def trajectory_errors(est: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None,
                      rpe_deltas: Sequence[int] = (1, 24), aligns: Sequence[str] = ("none", "se3", "sim3")) -> dict[str, Any]:
    """ATE (per alignment) and RPE of camera poses ``(T, 4, 4)``."""
    align = align_module()
    if align is None:
        return dict(PENDING)
    est = np.asarray(est, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    out: dict[str, Any] = {"status": "ok"}
    for mode in aligns:
        try:
            r = align.ate(est, gt, align=mode, valid=valid)
            out[f"ate_{mode}"] = {k: r[k] for k in ("rmse", "mean", "median", "max", "scale", "n")}
        except ValueError as error:
            out[f"ate_{mode}"] = {"status": "failed", "reason": str(error)}
    for delta in rpe_deltas:
        if est.shape[0] > delta:
            r = align.rpe(est, gt, delta=int(delta), valid=valid)
            out[f"rpe_{delta}"] = {k: r[k] for k in ("trans_rmse", "trans_mean", "rot_rmse_deg", "rot_mean_deg", "n", "scale")}
    return out


def per_frame_position_error(est: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None,
                             align_mode: str = "sim3") -> np.ndarray:
    """``(T,)`` camera position error after alignment (NaN where invalid)."""
    align = align_module()
    est = np.asarray(est, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    aligned = est
    if align is not None:
        try:
            aligned = align.ate(est, gt, align=align_mode, valid=valid)["aligned"]
        except ValueError:
            aligned = est
    err = np.linalg.norm(aligned[:, :3, 3] - gt[:, :3, 3], axis=-1)
    if valid is not None:
        err = np.where(np.asarray(valid, dtype=bool), err, np.nan)
    return err


# ---------------------------------------------------------------- joints


def joint_errors(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None, procrustes: bool = True) -> dict[str, Any]:
    """MPJPE (and PA-MPJPE) of ``(T, 2, 21, 3)`` joints; ``valid`` is ``(T, 2)`` or ``(T, 2, 21)``."""
    align = align_module()
    if align is None:
        return dict(PENDING)
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    m = np.ones(pred.shape[:-1], dtype=bool)
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        m &= v[..., None] if v.ndim == pred.ndim - 2 else v
    out: dict[str, Any] = {"status": "ok", "mpjpe": align.mpjpe(pred, gt, m)}
    for slot, side in enumerate(("left", "right")):
        out[f"mpjpe_{side}"] = align.mpjpe(pred[:, slot], gt[:, slot], m[:, slot])
        out[f"n_{side}"] = int(np.sum(np.any(m[:, slot] & np.all(np.isfinite(pred[:, slot]), axis=-1), axis=-1)))
    if procrustes:
        hands = np.any(m & np.all(np.isfinite(pred), axis=-1), axis=-1)
        p = np.where(m[..., None], pred, np.nan)[hands]
        g = gt[hands]
        out["pa_mpjpe"] = align.pa_mpjpe(p, g) if len(p) else float("nan")
    return out


def per_frame_joint_error(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """``(T, 2)`` mean joint error per frame and hand (NaN where invalid)."""
    err = np.linalg.norm(np.asarray(pred, dtype=np.float64) - np.asarray(gt, dtype=np.float64), axis=-1)
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        err = np.where(v[..., None] if v.ndim == err.ndim - 1 else v, err, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
        return np.nanmean(err, axis=-1)


def point_errors(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None) -> dict[str, Any]:
    """Distances of ``(T, 2, 3)`` points (shoulders, elbows) per side."""
    err = np.linalg.norm(np.asarray(pred, dtype=np.float64) - np.asarray(gt, dtype=np.float64), axis=-1)
    if valid is not None:
        err = np.where(np.asarray(valid, dtype=bool), err, np.nan)
    out: dict[str, Any] = {}
    for slot, side in enumerate(("left", "right")):
        out[side] = _stats(err[:, slot])
    out["all"] = _stats(err.reshape(-1))
    return out


# ---------------------------------------------------------------- objects


def object_pose_errors(est: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None) -> dict[str, Any]:
    """Translation error of the object origin (centroid) and the error of the rotation *change*
    since the first valid frame, expressed in the world (``R_t R_0^T``). The estimated object
    frame's axes are arbitrary, so absolute rotations are not comparable, but world-frame rotation
    changes are invariant to that choice."""
    est = np.asarray(est, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    m = np.all(np.isfinite(est), axis=(-2, -1))
    if valid is not None:
        m &= np.asarray(valid, dtype=bool)
    if not np.any(m):
        return {"n": 0}
    trans = np.linalg.norm(est[m, :3, 3] - gt[m, :3, 3], axis=-1)
    first = int(np.flatnonzero(m)[0])
    d_est = est[m, :3, :3] @ np.swapaxes(est[first, :3, :3], -1, -2)
    d_gt = gt[m, :3, :3] @ np.swapaxes(gt[first, :3, :3], -1, -2)
    rot = np.degrees(geo.rotation_angle(d_est, d_gt))
    return {"translation_m": _stats(trans), "rotation_change_deg": _stats(rot), "n": int(m.sum())}


# ---------------------------------------------------------------- IMU


def imu_errors(est: dict[str, Any], gt: Any, T_world_cam_gt: np.ndarray) -> dict[str, Any]:
    """Errors of the ``imu`` step's calibration against the synthetic truth."""
    out: dict[str, Any] = {}
    if "time_offset_s" in est:
        out["time_offset_error_s"] = float(est["time_offset_s"]) - float(gt["imu_time_offset_s"])
        out["time_offset_s"] = float(est["time_offset_s"])
        out["time_offset_true_s"] = float(gt["imu_time_offset_s"])
    if "R_cam_imu" in est:
        out["R_cam_imu_error_deg"] = float(np.degrees(geo.rotation_angle(np.asarray(est["R_cam_imu"]), np.asarray(gt["R_cam_imu"]))))
    if "gyro_bias" in est:
        out["gyro_bias_error_rad_s"] = float(np.linalg.norm(np.asarray(est["gyro_bias"]) - np.asarray(gt["gyro_bias"])))
    if "accel_scale" in est:
        out["accel_scale"] = float(est["accel_scale"])
        out["accel_scale_error"] = float(est["accel_scale"]) - 9.80665
    if "gravity_cam" in est:
        g_est = np.asarray(est["gravity_cam"], dtype=np.float64)
        g_true = np.einsum("tji,j->ti", np.asarray(T_world_cam_gt)[:, :3, :3], np.array([0.0, 0.0, -1.0]))
        n = np.linalg.norm(g_est, axis=-1)
        cos = np.sum(g_est * g_true, axis=-1) / np.where(n > 0, n, np.nan)
        out["gravity_error_deg"] = _stats(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return out
