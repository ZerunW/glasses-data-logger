"""Deterministic NumPy-only tests for glassesdata_pipeline.core.align."""

from __future__ import annotations

import numpy as np
import pytest

from glassesdata_pipeline.core.align import (
    apply_sim3,
    associate,
    ate,
    mpjpe,
    pa_mpjpe,
    read_tum,
    rotation_angle_deg,
    rpe,
    umeyama,
    write_tum,
)


def _rot(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _random_rot(rng):
    return _rot(rng.normal(size=3), rng.uniform(0.3, 3.0))


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    out = np.zeros(np.shape(a) + (3, 3))
    out[..., 0, 0], out[..., 0, 1], out[..., 1, 0], out[..., 1, 1], out[..., 2, 2] = c, -s, s, c, 1
    return out


def _trajectory(n=200, seed=10):
    """Smooth, non-planar camera trajectory with varying orientation."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n)
    pos = np.stack([2 * np.cos(t), 1.5 * np.sin(t), 0.3 * np.sin(3 * t)], axis=1)
    poses = np.zeros((n, 4, 4))
    poses[:, 3, 3] = 1
    for k in range(n):
        poses[k, :3, :3] = _rot([0.2, 0.1, 1.0], t[k]) @ _rot([1, 0, 0], 0.2 * np.sin(2 * t[k]))
    poses[:, :3, 3] = pos
    return poses, rng


# ---------------------------------------------------------------- 1. umeyama


def test_umeyama_exact_noisy_noscale_and_reflection() -> None:
    rng = np.random.default_rng(1)
    src = rng.normal(size=(50, 3))
    s0, r0, t0 = rng.uniform(0.5, 2.0), _random_rot(rng), rng.normal(size=3)
    dst = s0 * src @ r0.T + t0

    s, r, t = umeyama(src, dst)
    assert abs(s - s0) < 1e-9
    np.testing.assert_allclose(r, r0, atol=1e-9)
    np.testing.assert_allclose(t, t0, atol=1e-9)

    s_n, r_n, t_n = umeyama(src, dst + rng.normal(0, 0.01, dst.shape))
    assert abs(s_n - s0) < 0.01 and rotation_angle_deg(r_n, r0) < 0.5
    np.testing.assert_allclose(t_n, t0, atol=0.02)

    s1, r1, _ = umeyama(src, dst, with_scale=False)
    assert s1 == 1.0
    assert rotation_angle_deg(r1, r0) < 1e-6          # rotation still exact for a pure similarity

    # mirrored data: best orthogonal map is a reflection, Umeyama must return a rotation
    mirrored = 1.5 * src * np.array([1.0, 1.0, -1.0])
    sm, rm, tm = umeyama(src, mirrored)
    assert np.linalg.det(rm) == pytest.approx(1.0)
    np.testing.assert_allclose(rm @ rm.T, np.eye(3), atol=1e-12)
    # (s, t) must still be least-squares optimal for the returned rotation
    xs, xd = src - src.mean(0), mirrored - mirrored.mean(0)
    s_opt = np.sum(xd * (xs @ rm.T)) / np.sum(xs * xs)
    assert sm == pytest.approx(s_opt, rel=1e-9)
    np.testing.assert_allclose(tm, mirrored.mean(0) - sm * rm @ src.mean(0), atol=1e-12)


def test_umeyama_nan_rows_weights_and_degenerate_inputs() -> None:
    rng = np.random.default_rng(2)
    src = rng.normal(size=(30, 3))
    r0 = _random_rot(rng)
    dst = 1.3 * src @ r0.T + [1.0, 2.0, 3.0]
    dst[3] = np.nan
    src[5, 1] = np.nan
    dst[10] += 5.0                                   # corrupted row ...
    w = np.ones(30)
    w[10] = 0.0                                      # ... removed by weight 0
    s, r, t = umeyama(src, dst, weights=w)
    assert abs(s - 1.3) < 1e-9 and rotation_angle_deg(r, r0) < 1e-6
    with pytest.raises(ValueError):
        umeyama(src[:2], dst[:2])
    line = np.outer(np.arange(5.0), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        umeyama(line, line + 1.0)
    with pytest.raises(ValueError):
        umeyama(np.full((5, 3), np.nan), np.zeros((5, 3)))


def test_apply_sim3_on_poses() -> None:
    poses, rng = _trajectory(10)
    s, r, t = 2.0, _random_rot(rng), np.array([1.0, -2.0, 0.5])
    out = apply_sim3(s, r, t, poses)
    np.testing.assert_allclose(out[:, :3, 3], s * poses[:, :3, 3] @ r.T + t, atol=1e-12)
    np.testing.assert_allclose(out[:, :3, :3], r @ poses[:, :3, :3], atol=1e-12)
    np.testing.assert_array_equal(out[:, 3], np.tile([0, 0, 0, 1.0], (10, 1)))


# ---------------------------------------------------------------- 2. ATE


def test_ate_recovers_sim3_and_noise_level() -> None:
    gt, rng = _trajectory()
    s0, r0, t0 = 0.4, _random_rot(rng), np.array([3.0, -1.0, 2.0])
    est = apply_sim3(s0, r0, t0, gt)
    sigma = 0.01
    est[:, :3, 3] += rng.normal(0, sigma * s0, (gt.shape[0], 3))   # noise of sigma metres in gt units
    est[17] = np.nan                                               # lost pose
    valid = np.ones(gt.shape[0], bool)
    valid[50] = False

    res = ate(est, gt, align="sim3", valid=valid)
    assert res["n"] == gt.shape[0] - 2
    assert res["scale"] == pytest.approx(1 / s0, rel=0.005)
    expected = sigma * np.sqrt(3)
    assert 0.8 * expected < res["rmse"] < 1.1 * expected
    assert res["median"] <= res["rmse"] <= res["max"]
    assert res["aligned"].shape == gt.shape
    assert np.isnan(res["aligned"][17]).all()
    np.testing.assert_allclose(res["aligned"][50, :3, 3], gt[50, :3, 3], atol=0.06)  # aligned though invalid

    assert ate(est, gt, align="none")["rmse"] > 1.0
    se3 = ate(est, gt, align="se3")
    assert se3["scale"] == 1.0 and se3["rmse"] > 10 * res["rmse"]  # wrong scale cannot be absorbed
    with pytest.raises(ValueError):
        ate(est, gt, align="bogus")


# ---------------------------------------------------------------- 3. RPE


def test_rpe_identical_and_constant_drift() -> None:
    gt, _ = _trajectory()
    zero = rpe(gt, gt)
    assert zero["n"] == gt.shape[0] - 1
    assert zero["trans_rmse"] < 1e-12 and zero["rot_rmse_deg"] < 1e-6

    # constant translational drift v per frame (in world): every relative
    # translation is off by exactly |v| * delta, rotations are untouched
    v = np.array([0.01, -0.02, 0.005])
    est = gt.copy()
    est[:, :3, 3] += np.arange(gt.shape[0])[:, None] * v
    for delta in (1, 5):
        res = rpe(est, gt, delta=delta, align_scale=False)
        assert res["n"] == gt.shape[0] - delta
        assert res["trans_mean"] == pytest.approx(np.linalg.norm(v) * delta, rel=1e-9)
        assert res["trans_rmse"] == pytest.approx(np.linalg.norm(v) * delta, rel=1e-9)
        assert res["rot_mean_deg"] < 1e-6

    # constant yaw drift eps per frame on a yaw-only trajectory: rot error = eps * delta
    yaw_gt = np.zeros((50, 4, 4))
    yaw_gt[:, 3, 3] = 1
    yaw_gt[:, :3, :3] = _rz(np.linspace(0, 2, 50))
    yaw_gt[:, :3, 3] = np.stack([np.cos(np.arange(50) / 7), np.sin(np.arange(50) / 5), np.arange(50) / 30], 1)
    eps = np.radians(0.3)
    yaw_est = yaw_gt.copy()
    yaw_est[:, :3, :3] = _rz(np.arange(50) * eps) @ yaw_gt[:, :3, :3]
    res = rpe(yaw_est, yaw_gt, delta=2, align_scale=False)
    assert res["rot_mean_deg"] == pytest.approx(0.6, rel=1e-6)
    assert res["rot_rmse_deg"] == pytest.approx(0.6, rel=1e-6)


def test_rpe_scale_alignment_and_invalid_pairs() -> None:
    gt, _ = _trajectory(60)
    est = gt.copy()
    est[:, :3, 3] *= 3.0                                     # monocular scale ambiguity
    assert rpe(est, gt, align_scale=True)["trans_rmse"] < 1e-9
    assert rpe(est, gt, align_scale=False)["trans_rmse"] > 0.05
    valid = np.ones(60, bool)
    valid[[10, 30]] = False
    est[45] = np.nan
    res = rpe(est, gt, delta=1, valid=valid)
    assert res["n"] == 59 - 6                                # pairs touching 10, 30, 45 dropped
    # straight-line trajectory: sim3 rotation undefined, scale still recovered
    line = np.tile(np.eye(4), (20, 1, 1))
    line[:, 0, 3] = np.arange(20) * 0.1
    est_line = line.copy()
    est_line[:, 0, 3] *= 0.5
    res = rpe(est_line, line)
    assert res["scale"] == pytest.approx(2.0) and res["trans_rmse"] < 1e-12


# ---------------------------------------------------------------- 4. rotation angle


def test_rotation_angle_deg() -> None:
    assert rotation_angle_deg(np.eye(3), _rz(np.pi / 2)) == pytest.approx(90.0, abs=1e-12)
    assert rotation_angle_deg(_rz(0.3), _rz(0.3 + 1e-8)) == pytest.approx(np.degrees(1e-8), rel=1e-6)
    assert rotation_angle_deg(np.eye(3), np.diag([1.0, -1.0, -1.0])) == pytest.approx(180.0)
    batch = rotation_angle_deg(np.tile(np.eye(3), (4, 1, 1)), _rz(np.radians([0, 10, 45, 170])))
    np.testing.assert_allclose(batch, [0, 10, 45, 170], atol=1e-9)
    # slightly non-orthonormal input must not produce NaN
    assert np.isfinite(rotation_angle_deg(np.eye(3) * (1 + 1e-9), np.eye(3)))


# ---------------------------------------------------------------- 5. MPJPE


def test_mpjpe_and_pa_mpjpe() -> None:
    rng = np.random.default_rng(5)
    gt = rng.normal(0, 0.1, (20, 21, 3))
    pred = gt + np.array([0.03, 0.0, -0.04])                # constant 5 cm offset
    assert mpjpe(pred, gt) == pytest.approx(0.05)
    pred_nan = pred.copy()
    pred_nan[0, 3] = np.nan
    pred_nan[4, 7] += 10.0
    valid = np.ones((20, 21), bool)
    valid[4, 7] = False
    assert mpjpe(pred_nan, gt, valid=valid) == pytest.approx(0.05)

    # per-sample random similarity: PA-MPJPE ~ 0, MPJPE large
    sims = np.empty_like(gt)
    for k in range(20):
        sims[k] = rng.uniform(0.5, 2) * gt[k] @ _random_rot(rng).T + rng.normal(size=3)
    assert pa_mpjpe(sims, gt) < 1e-9
    assert mpjpe(sims, gt) > 0.1
    noisy = sims + rng.normal(0, 0.005, sims.shape)
    assert 0.003 < pa_mpjpe(noisy, gt) < 0.02

    # NaN joints are ignored, samples with < 3 valid joints are skipped
    sims[2, :5] = np.nan
    sims[3, 2:] = np.nan                                    # only 2 valid joints
    sims[3, :2] += 100.0
    assert pa_mpjpe(sims, gt) < 1e-9
    noisy[2, :5] = np.nan
    noisy[3, 2:] = np.nan                                   # 2 joints fit exactly: must not count
    assert pa_mpjpe(noisy, gt) == pytest.approx(pa_mpjpe(np.delete(noisy, 3, 0), np.delete(gt, 3, 0)),
                                                rel=1e-12)
    assert np.isnan(pa_mpjpe(np.full((2, 5, 3), np.nan), np.zeros((2, 5, 3))))


# ---------------------------------------------------------------- 6. TUM I/O and association


def test_tum_round_trip_and_column_order(tmp_path) -> None:
    poses, _ = _trajectory(30)
    ts = 1.7e9 + np.arange(30) / 30.0
    path = tmp_path / "traj.txt"
    write_tum(path, ts, poses)
    ts2, poses2 = read_tum(path)
    np.testing.assert_array_equal(ts2, ts)
    np.testing.assert_allclose(poses2, poses, atol=1e-12)

    # on disk: tx ty tz qx qy qz qw (scalar last); 90 deg about z -> qz = qw = sqrt(1/2)
    p = np.eye(4)[None].copy()
    p[0, :3, :3] = _rz(np.pi / 2)
    p[0, :3, 3] = [1.0, 2.0, 3.0]
    write_tum(path, [0.5], p)
    line = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")][0]
    vals = np.array(line.split(), float)
    np.testing.assert_allclose(vals, [0.5, 1, 2, 3, 0, 0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-12)

    # comments, blank lines, unnormalised quaternion
    path.write_text("# header\n\n1.0 0 0 0 0 0 0 2\n2.0 1 1 1 0 0 1 1\n")
    t3, p3 = read_tum(path)
    np.testing.assert_array_equal(t3, [1.0, 2.0])
    np.testing.assert_allclose(p3[0], np.eye(4))
    assert rotation_angle_deg(np.eye(3), p3[1, :3, :3]) == pytest.approx(90.0)


def test_associate_unequal_rates_one_to_one() -> None:
    t_a = np.arange(0.0, 2.0, 1 / 30)                       # 30 Hz
    t_b = np.arange(0.004, 2.0, 1 / 24)                     # 24 Hz, offset start
    ia, ib = associate(t_a, t_b, max_dt=0.02)
    assert len(ia) == len(ib) == len(np.unique(ia)) == len(np.unique(ib))
    assert np.all(np.diff(t_a[ia]) > 0)
    assert np.all(np.abs(t_a[ia] - t_b[ib]) <= 0.02)
    # every b has a 30 Hz neighbour within 1/60 s, so every b is matched
    assert len(ib) == len(t_b)
    # each match is the nearest a for that b
    nearest = np.abs(t_b[ib][:, None] - t_a[None, :]).argmin(axis=1)
    np.testing.assert_array_equal(ia, nearest)
    # greedy conflict: one a near two b's -> only the closer b gets it
    ia2, ib2 = associate([1.0], [0.99, 1.005], max_dt=0.02)
    assert ia2.tolist() == [0] and ib2.tolist() == [1]
    # unsorted input, nothing within max_dt
    ia3, ib3 = associate([3.0, 1.0, 2.0], [2.001, 0.999], max_dt=0.01)
    assert sorted(zip(ia3.tolist(), ib3.tolist())) == [(1, 1), (2, 0)]
    assert ia3.tolist() == [1, 2]
    assert associate([0.0], [1.0])[0].size == 0
