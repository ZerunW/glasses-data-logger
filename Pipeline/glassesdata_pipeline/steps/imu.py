"""Step ``imu``: camera-IMU calibration and gravity -> ``derived/imu.npz``.

Output (PIPELINE.md section 5): ``R_cam_imu[3,3]`` (``v_cam = R_cam_imu @ v_imu``),
``time_offset_s`` (IMU time = video time + offset), ``gyro_bias[3]`` (IMU frame, rad/s, after
unit conversion), ``accel_scale`` (raw accelerometer units -> m/s^2), ``gravity_cam[T,3]`` (unit,
pointing down, camera frame), ``R_world_slam[3,3]``, ``quality`` (JSON string), plus
``accel_bias[3]`` (IMU frame, m/s^2) and ``gyro_scale`` (raw gyro units -> rad/s).

Method (parameters in ``configs/imu.toml``):

1. IMU samples: the glasses rows of ``timeline.npz``. Without them the phone's IMU is used only
   when ``imu.use_phone_imu`` (the phone is not rigidly attached to the head: its calibration is
   meaningless for the head); otherwise the outputs are identity / NaN with
   ``quality.gravity = "none"`` and a warning (gravity from the dominant horizontal plane is not
   implemented).
2. Units: accelerometer from the median norm (``core.imu_filter.detect_accel_scale``) unless
   ``imu.accel_units`` is ``"g"`` or ``"m/s2"``; gyroscope from ``imu.gyro_units`` (``"rad/s"``,
   ``"deg/s"`` or ``"auto"``: the median ratio of gyro to visual rotation rates,
   ``gyro_scale_check``, ~1 -> rad/s, ~57 -> deg/s).
3. Visual angular velocity: ``log(R_i^T R_{i+1}) / dt`` between consecutive valid ``slam`` frames,
   compared with the exact mean of the (interpolated) gyro over the same interval shifted by the
   offset (no aliasing between 24 fps and the IMU rate).
4. Time offset: cross-correlation of the rate norms (``estimate_time_offset``, rotation
   independent), then refined to sub-millisecond by minimising the vector residual over
   ``+- refine_window_s`` once the rotation is known.
5. Rotation ``R_cam_imu``: Kabsch with Huber IRLS (``estimate_rotation_extrinsic``) alternated
   with the gyro bias, or ``imu.extrinsic = "fixed"`` with ``imu.R_cam_imu``.
6. Gyro bias: still periods (``estimate_gyro_bias``) when they cover at least
   ``min_still_fraction`` of the clip, else the mean residual of the extrinsic fit.
7. Gravity in the SLAM frame and accelerometer bias by "preintegration": with a metric
   trajectory, position double differences over frame triplets (``preint_window_s``) eliminate
   the velocities and leave a linear system in ``g_slam`` and ``b_acc``
   (``core.imu_calib.gravity_preintegration``, with ``|g| = 9.81`` imposed and a weak prior
   ``preint_bias_sigma`` on the bias, which is only observable through head rotation: short
   clips give ``quality.preint_tilt_std_deg`` > 1 and a warning); the free ``|g|`` must come out
   near 9.81 m/s^2
   (``quality.preint_g_norm``; a large deviation means a wrong scale or offset, and the Mahony
   average is used instead). ``quality.slam_scale_from_imu`` is the trajectory scale the
   accelerometer suggests (diagnostic only).
8. Gravity per frame (``gravity_cam``): Mahony filter (``core.imu_filter.mahony``) on the
   accelerometer minus its bias and, with ``compensate_linear_acc``, minus the SLAM linear
   acceleration rotated into the IMU frame (local quadratic fit of positions over
   ``+-linear_acc_half_window`` frames); run forward and backward and averaged
   (``mahony_bidirectional``); sampled at frame time + offset, rotated by ``R_cam_imu``.
   ``imu.gravity_cam = "slam"`` instead uses ``R_slam_cam^T g_slam`` on valid frames
   (consistent with the world trajectory by construction), Mahony elsewhere.
9. ``R_world_slam``: rotates ``g_slam`` (``imu.gravity_world = "preintegration"`` or
   ``"mahony"`` = the mean of ``R_slam_cam gravity_cam`` over valid frames) to ``-z``, then about
   z so that the first valid camera looks along +x (PIPELINE.md section 1).
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import numpy as np

from .. import geometry as geo
from ..core import imu_calib as ic
from ..core import imu_filter as imf
from ..episode import Episode
from ..errors import PipelineError

NAME = "imu"
VERSION = 1
REQUIRES = ["ingest", "slam"]
INPUTS = ["derived/timeline.npz", "derived/slam.npz"]
OUTPUTS = ["imu.npz"]
CONFIG_SECTIONS = ["imu"]

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "use_phone_imu": False,
    "accel_units": "auto",
    "gyro_units": "auto",
    "max_offset_s": 0.2,
    "offset_rate_hz": 200.0,
    "refine_window_s": 0.02,
    "refine_step_s": 0.0005,
    "extrinsic": "estimate",
    "R_cam_imu": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "fixed_time_offset_s": 0.0,
    "huber_rad_s": 0.1,
    "min_rate_pairs": 20,
    "bias_window_s": 0.5,
    "bias_gyro_thresh": 0.05,
    "bias_acc_thresh": 0.3,
    "min_still_fraction": 0.2,
    "accel_bias": "estimate",
    "preint_window_s": 0.5,
    "preint_substeps": 8,
    "preint_max_g_error": 0.5,
    "preint_fix_g_norm": True,
    "preint_bias_sigma": 0.5,
    "preint_eq_sigma": 0.1,
    "gravity_world": "preintegration",
    "gravity_cam": "mahony",
    "compensate_linear_acc": True,
    "linear_acc_half_window": 3,
    "mahony_kp": 1.0,
    "mahony_ki": 0.0,
    "mahony_bidirectional": True,
}


def params(cfg) -> dict[str, Any]:
    section = cfg.get("imu", {}) if hasattr(cfg, "get") else {}
    return {**DEFAULTS, **dict(section or {})}


def run(ep: Episode, cfg) -> dict:
    p = params(cfg)
    res = calibrate(ep, p)
    quality = res.pop("quality")
    ep.save_npz("imu", ep.step_meta(backend=quality.get("gravity")), quality=json.dumps(quality), **res)
    summary = {"gravity": quality.get("gravity"), "source": quality.get("source")}
    if quality.get("gravity") != "none":
        summary.update({
            "offset_ms": round(1000 * float(res["time_offset_s"]), 2),
            "extrinsic_rms": round(float(quality.get("extrinsic_rms", float("nan"))), 4),
            "g_norm": round(float(quality.get("preint_g_norm", float("nan"))), 3),
        })
    return summary


# =============================================================================================


def _empty(T: int, quality: dict[str, Any]) -> dict[str, Any]:
    quality = {"gravity": "none", **quality}
    return {
        "R_cam_imu": np.eye(3), "time_offset_s": 0.0, "gyro_bias": np.zeros(3), "accel_scale": float("nan"),
        "gravity_cam": np.full((T, 3), np.nan), "R_world_slam": np.eye(3), "accel_bias": np.zeros(3),
        "gyro_scale": 1.0, "quality": quality,
    }


def _select_samples(tl: dict[str, Any], p: dict[str, Any]) -> tuple[dict[str, np.ndarray] | None, str]:
    t = np.asarray(tl.get("imu_t", np.zeros(0)), dtype=np.float64)
    if t.size == 0:
        return None, "none"
    glasses = np.asarray(tl.get("imu_is_glasses", np.zeros(t.size, bool)), dtype=bool)
    source = "glasses"
    sel = glasses
    if not np.any(glasses):
        if bool(p["use_phone_imu"]):
            log.warning("imu: no glasses IMU; using the PHONE IMU (not rigidly attached to the head)")
            sel = ~glasses
            source = "phone"
        else:
            return None, "none"
    acc = np.asarray(tl["acc"], dtype=np.float64)[sel]
    gyro = np.asarray(tl["gyro"], dtype=np.float64)[sel]
    tt = t[sel]
    ok = np.isfinite(tt) & np.all(np.isfinite(acc), axis=1) & np.all(np.isfinite(gyro), axis=1)
    tt, acc, gyro = tt[ok], acc[ok], gyro[ok]
    order = np.argsort(tt, kind="stable")
    tt, acc, gyro = tt[order], acc[order], gyro[order]
    keep = np.concatenate([[True], np.diff(tt) > 0])
    if tt.size < 10:
        return None, "none"
    return {"t": tt[keep], "acc": acc[keep], "gyro": gyro[keep]}, source


def _accel_scale(acc: np.ndarray, units: str) -> float:
    if units == "g":
        return imf.G0
    if units in ("m/s2", "m/s^2"):
        return 1.0
    return float(imf.detect_accel_scale(acc))


def _gyro_scale(units: str, w_cam: np.ndarray, w_imu: np.ndarray, quality: dict) -> float:
    if units == "rad/s":
        return 1.0
    if units == "deg/s":
        return math.pi / 180.0
    ratio = imf.gyro_scale_check(np.linalg.norm(w_cam, axis=1), np.linalg.norm(w_imu, axis=1))
    quality["gyro_ratio"] = float(ratio)
    if not np.isfinite(ratio):
        log.warning("imu: gyro units undetermined (no usable visual rates); assuming rad/s")
        return 1.0
    if abs(math.log(ratio)) > abs(math.log(ratio / (180.0 / math.pi))):
        if not 30 < ratio < 110:
            log.warning("imu: gyro/visual rate ratio %.3g is far from 57.3; assuming deg/s", ratio)
        return math.pi / 180.0
    if not 0.5 < ratio < 2.0:
        log.warning("imu: gyro/visual rate ratio %.3g is far from 1; assuming rad/s", ratio)
    return 1.0


def _mahony_gravity(t: np.ndarray, acc: np.ndarray, gyro: np.ndarray, p: dict) -> np.ndarray:
    """Unit gravity (down) in the IMU frame at the IMU samples."""
    kp, ki = float(p["mahony_kp"]), float(p["mahony_ki"])
    g_f = imf.gravity_in_sensor(imf.mahony(t, acc, gyro, kp, ki))
    if not bool(p["mahony_bidirectional"]):
        return g_f
    t_rev = t[-1] - t[::-1]
    g_b = imf.gravity_in_sensor(imf.mahony(t_rev, acc[::-1], -gyro[::-1], kp, ki))[::-1]
    g = g_f + g_b
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def calibrate(ep: Episode, p: dict[str, Any]) -> dict[str, Any]:
    """All ``imu.npz`` arrays plus ``quality`` (dict)."""
    tl = ep.timeline()
    t = np.asarray(tl["t"], dtype=np.float64)
    T = len(t)
    samples, source = _select_samples(tl, p)
    quality: dict[str, Any] = {"source": source}
    if samples is None:
        log.warning("imu: no glasses IMU samples (imu.use_phone_imu = false): gravity unknown, "
                    "R_world_slam = identity; the world frame will not be gravity aligned")
        return _empty(T, quality)
    t_imu, acc_raw, gyro_raw = samples["t"], samples["acc"], samples["gyro"]
    quality["samples"] = int(t_imu.size)
    quality["rate_hz"] = float(1.0 / np.median(np.diff(t_imu)))
    try:
        accel_scale = _accel_scale(acc_raw, str(p["accel_units"]))
    except ValueError as error:
        raise PipelineError(f"imu: {error} (set imu.accel_units)") from None
    acc = acc_raw * accel_scale
    quality["accel_units"] = "g" if accel_scale != 1.0 else "m/s2"

    slam = ep.load_npz("slam") if ep.has("slam.npz") else None
    if slam is not None:
        T_sc = np.asarray(slam["T_slam_cam"], dtype=np.float64)
        valid = np.asarray(slam["valid"], dtype=bool) & np.all(np.isfinite(T_sc), axis=(1, 2))
    else:
        T_sc = np.full((T, 4, 4), np.nan)
        valid = np.zeros(T, bool)
    t0, t1, w_cam, _ = ic.visual_rates(t, T_sc[:, :3, :3], valid)
    quality["rate_pairs"] = int(len(t0))
    have_visual = len(t0) >= int(p["min_rate_pairs"])
    mid = 0.5 * (t0 + t1)

    # ---- gyro units
    units = str(p["gyro_units"])
    if units == "auto" and not have_visual:
        log.warning("imu: gyro_units = auto needs the slam trajectory; assuming rad/s")
        units = "rad/s"
    w_imu0 = ic.interval_mean(t_imu, gyro_raw, t0, t1) if have_visual else np.zeros((0, 3))
    gyro_scale = _gyro_scale(units, w_cam, w_imu0, quality)
    gyro = gyro_raw * gyro_scale
    quality["gyro_units"] = "rad/s" if gyro_scale == 1.0 else "deg/s"

    # ---- still-period bias
    bias_still, still_fraction = imf.estimate_gyro_bias(t_imu, gyro, acc, float(p["bias_window_s"]),
                                                        float(p["bias_gyro_thresh"]), float(p["bias_acc_thresh"]))
    quality["still_fraction"] = float(still_fraction)
    use_still = still_fraction >= float(p["min_still_fraction"])

    huber = float(p["huber_rad_s"])
    fixed = str(p["extrinsic"]) == "fixed"
    if fixed:
        R_cam_imu = geo.project_to_so3(np.asarray(p["R_cam_imu"], dtype=np.float64))
    if not have_visual:
        log.warning("imu: fewer than %d visual rate pairs; time offset = imu.fixed_time_offset_s and "
                    "R_cam_imu = %s", int(p["min_rate_pairs"]), "imu.R_cam_imu" if fixed else "identity")
        offset = float(p["fixed_time_offset_s"])
        R_cam_imu = R_cam_imu if fixed else np.eye(3)
        bias = bias_still if use_still else np.zeros(3)
        quality.update({"offset_method": "fixed", "extrinsic_method": "fixed" if fixed else "identity",
                        "bias_method": "still" if use_still else "none"})
        rms = float("nan")
    else:
        # ---- coarse offset (rate norms)
        d, corr = imf.estimate_time_offset(mid, w_cam, t_imu, gyro, float(p["max_offset_s"]), float(p["offset_rate_hz"]))
        offset = -d if np.isfinite(d) else float(p["fixed_time_offset_s"])
        quality.update({"offset_coarse_s": float(offset), "offset_corr": float(corr)})
        # ---- extrinsic + bias, offset refinement, again
        bias = bias_still.copy() if use_still else np.zeros(3)
        for _ in range(2):
            w_imu = ic.interval_mean(t_imu, gyro, t0 + offset, t1 + offset)
            if fixed:
                if not use_still:
                    r = w_imu - w_cam @ R_cam_imu
                    ok = np.all(np.isfinite(r), axis=1)
                    bias = np.median(r[ok], axis=0) if ok.any() else np.zeros(3)
            else:
                R_cam_imu, b_fit, rms = ic.fit_extrinsic_bias(w_cam, w_imu, bias, not use_still, huber)
                if not use_still:
                    bias = b_fit
            offset, grid, cost = ic.refine_offset(t0, t1, w_cam, t_imu, gyro, R_cam_imu, bias, offset,
                                                  float(p["refine_window_s"]), float(p["refine_step_s"]), huber)
        w_imu = ic.interval_mean(t_imu, gyro, t0 + offset, t1 + offset)
        if not fixed:
            R_cam_imu, b_fit, rms = ic.fit_extrinsic_bias(w_cam, w_imu, bias, not use_still, huber)
            if not use_still:
                bias = b_fit
        res = np.linalg.norm(w_cam - (w_imu - bias) @ R_cam_imu.T, axis=1)
        rms = float(np.sqrt(np.nanmean(res**2)))
        quality.update({"offset_method": "xcorr+refine", "extrinsic_method": "fixed" if fixed else "kabsch",
                        "bias_method": "still" if use_still else "extrinsic_residual",
                        "extrinsic_rms": rms, "offset_curve": {"grid_s": grid.tolist(), "cost": [float(c) if np.isfinite(c) else None for c in cost]}})

    gyro_c = gyro - bias

    # ---- gravity in the SLAM frame + accelerometer bias (preintegration)
    acc_bias = np.zeros(3)
    g_slam_pre = None
    if have_visual and str(p["accel_bias"]) in ("estimate", "none"):
        common = dict(window_s=float(p["preint_window_s"]), substeps=int(p["preint_substeps"]),
                      estimate_bias=str(p["accel_bias"]) == "estimate", bias_sigma=float(p["preint_bias_sigma"]),
                      eq_sigma=float(p["preint_eq_sigma"]))
        args = (t, T_sc, valid, t_imu, acc, R_cam_imu, offset)
        free = ic.gravity_preintegration(*args, fix_g_norm=False, **common)
        if free.get("n", 0) > 0:
            pre = ic.gravity_preintegration(*args, fix_g_norm=bool(p["preint_fix_g_norm"]), **common)
            scale_fit = ic.gravity_preintegration(*args, fix_g_norm=False, estimate_scale=True, **common)
            g_dir = pre["g"] / np.linalg.norm(pre["g"])
            b_std = np.asarray(pre["bias_std"], dtype=np.float64)
            # tilt uncertainty from the bias components perpendicular to gravity (IMU frame ~ any frame here)
            tilt_std = float(np.degrees(np.sqrt(np.nansum(b_std**2) * 2.0 / 3.0) / imf.G0)) if np.all(np.isfinite(b_std)) else None
            quality.update({"preint_g_norm": free["g_norm"], "preint_rms": pre["rms"], "preint_triplets": pre["n"],
                            "accel_bias_std_m_s2": b_std.tolist(), "preint_tilt_std_deg": tilt_std,
                            "slam_scale_from_imu": float(1.0 / scale_fit["scale"]) if scale_fit.get("n", 0) and scale_fit["scale"] else None})
            if tilt_std is not None and tilt_std > 1.0:
                log.warning("imu: the accelerometer bias is poorly observable (gravity tilt uncertainty ~%.1f deg); "
                            "a longer clip with more head rotation helps", tilt_std)
            if abs(free["g_norm"] - imf.G0) <= float(p["preint_max_g_error"]) * imf.G0:
                g_slam_pre = g_dir
                acc_bias = np.asarray(pre["bias"], dtype=np.float64)
            else:
                log.warning("imu: preintegration gave |g| = %.2f m/s^2 (expected 9.81): slam scale or time offset "
                            "is off; using the Mahony gravity", free["g_norm"])
    quality["accel_bias_m_s2"] = acc_bias.tolist()

    # ---- Mahony gravity per frame
    acc_c = acc - acc_bias
    if bool(p["compensate_linear_acc"]) and have_visual:
        a_slam = ic.savgol_second_derivative(t, T_sc[:, :3, 3], valid, int(p["linear_acc_half_window"]))
        R_cam = ic.slerp_rotations(t, T_sc[:, :3, :3], valid, t_imu - offset)
        a_at = np.stack([np.interp(t_imu - offset, t, a_slam[:, k], left=np.nan, right=np.nan) for k in range(3)], axis=1)
        R_slam_imu = R_cam @ R_cam_imu
        lin = np.einsum("nji,nj->ni", R_slam_imu, a_at)  # R_imu_slam @ a
        lin_ok = np.all(np.isfinite(lin), axis=1)
        acc_c = acc_c - np.where(lin_ok[:, None], lin, 0.0)
        quality["linear_acc_compensated"] = float(lin_ok.mean())
    g_imu = _mahony_gravity(t_imu, acc_c, gyro_c, p)
    g_frames = imf.resample(t_imu, g_imu, t + offset)
    g_cam_mahony = g_frames @ R_cam_imu.T
    n = np.linalg.norm(g_cam_mahony, axis=1, keepdims=True)
    g_cam_mahony = g_cam_mahony / np.where(n > 0, n, np.nan)

    # ---- gravity direction in the SLAM frame, world alignment
    g_slam_mahony = None
    vm = valid & np.all(np.isfinite(g_cam_mahony), axis=1)
    if vm.any():
        gs = np.einsum("tij,tj->ti", T_sc[vm, :3, :3], g_cam_mahony[vm])
        g_slam_mahony = gs.mean(axis=0)
        g_slam_mahony /= np.linalg.norm(g_slam_mahony)
        quality["mahony_spread_deg"] = float(np.median(ic.angle_deg(gs, g_slam_mahony)))
    if g_slam_pre is not None and g_slam_mahony is not None:
        quality["preint_vs_mahony_deg"] = float(ic.angle_deg(g_slam_pre, g_slam_mahony))
    method = str(p["gravity_world"])
    g_slam = g_slam_pre if (method == "preintegration" and g_slam_pre is not None) else g_slam_mahony
    quality["gravity_world"] = "preintegration" if (g_slam is g_slam_pre and g_slam is not None) else "mahony"
    if g_slam is not None and valid.any():
        first = int(np.flatnonzero(valid)[0])
        R_world_slam = ic.gravity_alignment(g_slam, T_sc[first, :3, :3])
        quality["gravity"] = "imu" if source == "glasses" else "phone"
    else:
        R_world_slam = np.eye(3)
        quality["gravity"] = "imu_no_slam" if source == "glasses" else "phone_no_slam"
        log.warning("imu: no valid slam frames; R_world_slam = identity")

    if str(p["gravity_cam"]) == "slam" and g_slam is not None:
        g_cam = g_cam_mahony.copy()
        g_cam[valid] = np.einsum("tji,j->ti", T_sc[valid, :3, :3], g_slam)
        quality["gravity_cam"] = "slam"
    else:
        g_cam = g_cam_mahony
        quality["gravity_cam"] = "mahony"
    if g_slam is not None and valid.any():
        g_slam_cam = np.einsum("tji,j->ti", T_sc[valid, :3, :3], g_slam)
        quality["gravity_cam_vs_slam_deg"] = float(np.nanmedian(ic.angle_deg(g_cam_mahony[valid], g_slam_cam)))
    quality["time_offset_s"] = float(offset)
    return {
        "R_cam_imu": R_cam_imu, "time_offset_s": float(offset), "gyro_bias": bias, "accel_scale": float(accel_scale),
        "gravity_cam": g_cam, "R_world_slam": R_world_slam, "accel_bias": acc_bias, "gyro_scale": float(gyro_scale),
        "quality": quality,
    }
