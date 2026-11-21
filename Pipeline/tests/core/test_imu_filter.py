"""Deterministic NumPy-only tests for glassesdata_pipeline.core.imu_filter."""

from __future__ import annotations

import numpy as np
import pytest

from glassesdata_pipeline.core.imu_filter import (
    G0,
    detect_accel_scale,
    estimate_gyro_bias,
    estimate_rotation_extrinsic,
    estimate_time_offset,
    gravity_in_sensor,
    gyro_scale_check,
    mahony,
    mat_to_quat,
    quat_mul,
    quat_to_mat,
    resample,
    so3_exp,
    so3_log,
)


def _angle_deg(ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
    c = (np.trace(np.swapaxes(ra, -1, -2) @ rb, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _vec_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _rot_xyz(roll, pitch, yaw):
    """R = Rz(yaw) Ry(pitch) Rx(roll), vectorised over the angle arrays."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    r = np.empty(np.shape(roll) + (3, 3))
    r[..., 0, 0] = cy * cp
    r[..., 0, 1] = cy * sp * sr - sy * cr
    r[..., 0, 2] = cy * sp * cr + sy * sr
    r[..., 1, 0] = sy * cp
    r[..., 1, 1] = sy * sp * sr + cy * cr
    r[..., 1, 2] = sy * sp * cr - cy * sr
    r[..., 2, 0] = -sp
    r[..., 2, 1] = cp * sr
    r[..., 2, 2] = cp * cr
    return r


# ---------------------------------------------------------------- 1. SO(3) / quaternions


def test_so3_exp_log_round_trip_including_near_zero_and_near_pi() -> None:
    rng = np.random.default_rng(1)
    axes = rng.normal(size=(12, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = np.array([0.0, 1e-12, 1e-7, 1e-4, 0.3, 1.0, 2.0, 3.0,
                       np.pi - 1e-4, np.pi - 1e-7, np.pi - 1e-10, np.pi])
    w = axes * angles[:, None]
    r = so3_exp(w)
    # orthonormal, det +1
    np.testing.assert_allclose(r @ np.swapaxes(r, -1, -2), np.broadcast_to(np.eye(3), r.shape), atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-14)
    w_back = so3_log(r)
    np.testing.assert_allclose(so3_exp(w_back), r, atol=1e-13)
    # away from exactly pi the vector itself is recovered (at pi, w and -w are the same rotation)
    np.testing.assert_allclose(w_back[:-2], w[:-2], atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(w_back, axis=1), angles, atol=1e-9)
    # first-order behaviour at tiny angles
    np.testing.assert_allclose(so3_log(so3_exp(np.array([1e-9, -2e-9, 3e-9]))),
                               [1e-9, -2e-9, 3e-9], rtol=1e-6, atol=0)


def test_so3_log_exactly_pi_and_quaternion_180() -> None:
    r = np.diag([1.0, -1.0, -1.0])           # 180 deg about x
    w = so3_log(r)
    np.testing.assert_allclose(np.abs(w), [np.pi, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(so3_exp(w), r, atol=1e-12)
    q = mat_to_quat(r)
    assert q[0] >= 0.0
    np.testing.assert_allclose(np.abs(q), [0.0, 1.0, 0.0, 0.0], atol=1e-12)
    # 180 deg about a skew axis
    axis = np.array([1.0, 2.0, -2.0]) / 3.0
    r2 = so3_exp(np.pi * axis)
    np.testing.assert_allclose(np.abs(so3_log(r2)), np.pi * np.abs(axis), atol=1e-9)
    np.testing.assert_allclose(quat_to_mat(mat_to_quat(r2)), r2, atol=1e-12)


def test_quaternion_matrix_round_trip_and_product() -> None:
    rng = np.random.default_rng(2)
    q = rng.normal(size=(200, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q_can = np.where(q[:, :1] < 0, -q, q)
    r = quat_to_mat(q)
    np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-12)
    q_back = mat_to_quat(r)
    assert np.all(q_back[:, 0] >= 0)
    np.testing.assert_allclose(q_back, q_can, atol=1e-12)
    # Hamilton product composes like matrix product
    a, b = q[:100], q[100:]
    np.testing.assert_allclose(quat_to_mat(quat_mul(a, b)), quat_to_mat(a) @ quat_to_mat(b), atol=1e-12)
    # [w,x,y,z]: 90 deg about z maps x -> y
    qz = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    np.testing.assert_allclose(quat_to_mat(qz) @ [1.0, 0, 0], [0, 1.0, 0], atol=1e-15)
    # so3_exp agrees with the quaternion of the same rotation
    w = rng.normal(size=(50, 3))
    th = np.linalg.norm(w, axis=1, keepdims=True)
    qw = np.hstack([np.cos(th / 2), np.sin(th / 2) * w / th])
    np.testing.assert_allclose(so3_exp(w), quat_to_mat(qw), atol=1e-12)


# ---------------------------------------------------------------- 2. Mahony on a synthetic trajectory


def _synthetic_imu(seed: int = 3, rate: float = 200.0, duration: float = 20.0,
                   bias=(0.01, -0.02, 0.005)):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, 1.0 / rate)

    def ramp(tt):
        x = np.clip((tt - 2.0) / 1.0, 0.0, 1.0)  # still for 2 s, then 1 s ease-in
        return x * x * (3 - 2 * x)

    def angles(tt):
        s = ramp(tt)
        roll = 0.10 + s * (0.35 * np.sin(2 * np.pi * 0.30 * tt) + 0.10 * np.sin(2 * np.pi * 1.1 * tt + 0.5))
        pitch = -0.05 + s * (0.25 * np.sin(2 * np.pi * 0.20 * tt + 1.0) + 0.10 * np.sin(2 * np.pi * 0.9 * tt))
        yaw = 0.30 + s * (0.80 * np.sin(2 * np.pi * 0.10 * tt) + 0.20 * np.sin(2 * np.pi * 0.7 * tt + 2.0))
        return roll, pitch, yaw

    r = _rot_xyz(*angles(t))
    h = 1e-5
    dr = (_rot_xyz(*angles(t + h)) - _rot_xyz(*angles(t - h))) / (2 * h)
    k = np.swapaxes(r, -1, -2) @ dr            # R^T dR/dt = [w_body]x
    w_body = np.stack([k[:, 2, 1], k[:, 0, 2], k[:, 1, 0]], axis=1)
    s = ramp(t)[:, None]
    a_lin = 0.2 * s * np.stack([np.sin(2 * np.pi * 0.5 * t), np.cos(2 * np.pi * 0.4 * t),
                                0.5 * np.sin(2 * np.pi * 0.7 * t)], axis=1)
    f_world = a_lin + np.array([0.0, 0.0, 9.81])
    acc = np.einsum("nji,nj->ni", r, f_world) + rng.normal(0.0, 0.03, (t.size, 3))
    gyro = w_body + np.asarray(bias) + rng.normal(0.0, 0.003, (t.size, 3))
    return t, r, w_body, acc, gyro


def test_mahony_gravity_error_with_estimated_bias() -> None:
    bias_true = np.array([0.01, -0.02, 0.005])
    t, r_true, _, acc, gyro = _synthetic_imu(bias=bias_true)
    bias, still = estimate_gyro_bias(t, gyro, acc)
    np.testing.assert_allclose(bias, bias_true, atol=1e-3)
    assert 0.08 <= still <= 0.2                      # ~2 s of 20 s

    q = mahony(t, acc, gyro, bias=bias)
    assert q.shape == (t.size, 4)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-12)
    g_est = gravity_in_sensor(q)
    g_true = -r_true[:, 2, :]
    err = _vec_angle_deg(g_est, g_true)[t >= 1.0]
    assert np.median(err) < 0.5          # spec asks < 2 deg; we are well inside
    assert np.max(err) < 2.0
    # yaw is unobservable but must not drift much once bias is removed:
    # R_est R_true^T is (nearly) a constant rotation about world z
    rel = quat_to_mat(q) @ np.swapaxes(r_true, -1, -2)
    drift = _angle_deg(rel[200], rel[-1])
    assert drift < 1.0


def test_mahony_without_bias_is_worse_and_static_tilt_converges() -> None:
    t, r_true, _, acc, gyro = _synthetic_imu(bias=(0.03, -0.03, 0.0))
    g_true = -r_true[:, 2, :]
    e_raw = np.median(_vec_angle_deg(gravity_in_sensor(mahony(t, acc, gyro)), g_true)[t >= 1.0])
    b, _ = estimate_gyro_bias(t, gyro, acc)
    e_fix = np.median(_vec_angle_deg(gravity_in_sensor(mahony(t, acc, gyro, bias=b)), g_true)[t >= 1.0])
    assert e_fix < 0.5 * e_raw

    # static, tilted sensor, wrong initial guess: converges to the accelerometer
    n = 2000
    ts = np.arange(n) / 200.0
    up = np.array([0.3, -0.4, 0.866])
    up /= np.linalg.norm(up)
    acc_s = np.tile(9.81 * up, (n, 1))
    acc_s[100:110] = np.nan                              # dropped samples are skipped
    q = mahony(ts, acc_s, np.zeros((n, 3)), kp=2.0, q0=[1.0, 0.0, 0.0, 0.0])
    assert np.all(np.isfinite(q))
    assert _vec_angle_deg(gravity_in_sensor(q[-1]), -up) < 0.05
    # default initialisation from the first accelerometer sample is exact
    q2 = mahony(ts[:5], acc_s[:5], np.zeros((5, 3)))
    assert _vec_angle_deg(gravity_in_sensor(q2[0]), -up) < 1e-6


def test_mahony_downweights_large_linear_acceleration() -> None:
    n = 1000
    t = np.arange(n) / 200.0
    acc = np.tile([0.0, 0.0, 9.81], (n, 1))
    acc[400:600] += [2.5, 0.0, 6.0]       # 1 s bump: |acc| - g ~ 6.2 m/s^2, 8.5 deg off vertical
    q = mahony(t, acc, np.zeros((n, 3)))
    tilt = _vec_angle_deg(gravity_in_sensor(q), [0.0, 0.0, -1.0])
    # fully trusted (kp = 1 for 1 s) the estimate would tilt ~5 deg towards the bump
    assert tilt.max() < 0.5
    # a deviation inside the 2 m/s^2 band is trusted as usual
    acc2 = np.tile([0.0, 0.0, 9.81], (n, 1))
    acc2[400:600] += [1.0, 0.0, 0.5]
    tilt2 = _vec_angle_deg(gravity_in_sensor(mahony(t, acc2, np.zeros((n, 3)))), [0.0, 0.0, -1.0])
    assert tilt2.max() > 2.0


# ---------------------------------------------------------------- 3. accelerometer units


def test_detect_accel_scale_g_and_ms2() -> None:
    rng = np.random.default_rng(4)
    up = rng.normal(size=(500, 3))
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    acc_g = up * (1.0 + 0.05 * rng.normal(size=(500, 1)))
    assert detect_accel_scale(acc_g) == pytest.approx(9.80665)
    assert detect_accel_scale(acc_g * 9.80665) == 1.0
    with pytest.raises(ValueError):
        detect_accel_scale(acc_g * 4.0)        # neither g nor m/s^2
    with pytest.raises(ValueError):
        detect_accel_scale(np.full((5, 3), np.nan))


def test_gyro_bias_no_still_window_and_resample() -> None:
    t = np.arange(400) / 200.0
    gyro = np.tile([0.3, 0.0, 0.0], (400, 1))     # always rotating
    acc = np.tile([0.0, 0.0, G0], (400, 1))
    bias, frac = estimate_gyro_bias(t, gyro, acc)
    np.testing.assert_array_equal(bias, np.zeros(3))
    assert frac == 0.0

    ts = np.array([0.0, 1.0, 2.0])
    xs = np.array([[0.0, 10.0], [1.0, 20.0], [4.0, 40.0]])
    out = resample(ts, xs, np.array([-0.5, 0.0, 0.5, 1.5, 2.0, 2.5]))
    np.testing.assert_allclose(out[1:5], [[0, 10], [0.5, 15], [2.5, 30], [4, 40]])
    assert np.isnan(out[0]).all() and np.isnan(out[5]).all()
    assert resample(ts, xs[:, 0], np.array([0.25])).shape == (1,)


# ---------------------------------------------------------------- 4. time offset


def _omega(t: np.ndarray) -> np.ndarray:
    return np.stack([
        1.2 * np.sin(2 * np.pi * 0.31 * t) + 0.5 * np.sin(2 * np.pi * 1.3 * t + 1.0),
        0.9 * np.sin(2 * np.pi * 0.53 * t + 2.0) + 0.4 * np.sin(2 * np.pi * 1.7 * t),
        0.7 * np.sin(2 * np.pi * 0.23 * t + 0.5) + 0.3 * np.sin(2 * np.pi * 2.1 * t + 3.0),
    ], axis=1)


@pytest.mark.parametrize("offset", [0.035, -0.035, 0.0, 0.11])
def test_estimate_time_offset_60hz_vs_24hz(offset: float) -> None:
    rng = np.random.default_rng(5)
    t_a = np.arange(0.0, 40.0, 1.0 / 60.0)
    t_b = np.arange(0.5, 39.5, 1.0 / 24.0)
    w_a = _omega(t_a) + rng.normal(0.0, 0.02, (t_a.size, 3))
    rot = so3_exp(np.array([0.4, -1.1, 0.7]))              # unknown extrinsic
    w_b = _omega(t_b + offset) @ rot.T + rng.normal(0.0, 0.02, (t_b.size, 3))

    d, peak = estimate_time_offset(t_a, w_a, t_b, w_b)

    assert abs(d - offset) < 0.003
    assert 0.95 < peak <= 1.0


# ---------------------------------------------------------------- 5. rotation extrinsic


@pytest.mark.parametrize("outlier_frac", [0.0, 0.1])
def test_estimate_rotation_extrinsic(outlier_frac: float) -> None:
    rng = np.random.default_rng(6)
    r_true = so3_exp(rng.normal(size=3) * 1.2)
    w_imu = rng.normal(size=(500, 3))
    w_cam = w_imu @ r_true.T + rng.normal(0.0, 0.01, (500, 3))
    bad = rng.choice(500, int(outlier_frac * 500), replace=False)
    w_cam[bad] = rng.normal(0.0, 2.0, (bad.size, 3))
    w_cam[7] = np.nan                                       # NaN row ignored

    r, rms = estimate_rotation_extrinsic(w_cam, w_imu)

    assert _angle_deg(r, r_true) < 0.5
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)
    if outlier_frac == 0.0:
        assert rms < 0.03                                   # ~ sqrt(3) * 0.01


def test_rotation_extrinsic_never_returns_reflection() -> None:
    rng = np.random.default_rng(7)
    w_imu = rng.normal(size=(100, 3))
    w_cam = w_imu * np.array([1.0, 1.0, -1.0])             # data generated by a reflection
    r, _ = estimate_rotation_extrinsic(w_cam, w_imu)
    assert np.linalg.det(r) == pytest.approx(1.0)


# ---------------------------------------------------------------- 6. gyro units


def test_gyro_scale_check_deg_per_s() -> None:
    rng = np.random.default_rng(8)
    w = _omega(np.arange(0.0, 30.0, 1.0 / 30.0))
    cam = np.linalg.norm(w, axis=1) + rng.normal(0.0, 0.01, w.shape[0])
    imu = np.degrees(np.linalg.norm(w, axis=1))
    assert gyro_scale_check(cam, imu) == pytest.approx(57.2958, rel=0.01)
    assert gyro_scale_check(cam, np.linalg.norm(w, axis=1)) == pytest.approx(1.0, rel=0.01)
