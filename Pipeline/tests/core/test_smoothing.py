"""Deterministic NumPy-only tests for the offline hand-trajectory smoothers."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from glassesdata_pipeline.core.smoothing import refine_bone_lengths, rts_smooth


def _rmse(x: np.ndarray, truth: np.ndarray) -> float:
    """Root mean square coordinate error, ignoring deliberately masked gaps."""
    return float(np.sqrt(np.nanmean((x - truth) ** 2)))


def _smooth_trajectory(t: np.ndarray) -> np.ndarray:
    """A smooth 3D trajectory of realistic, centimetre-scale hand motion."""
    return np.column_stack((
        0.09 * np.sin(2.0 * np.pi * 0.40 * t),
        0.11 * np.cos(2.0 * np.pi * 0.30 * t + 0.4),
        0.07 * np.sin(2.0 * np.pi * 0.25 * t + 0.7),
    ))


def _chain_truth(n: int = 240) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create 21 connected 3D joints with exactly specified, moving bones."""
    parents = np.arange(21, dtype=np.int64) - 1
    lengths = np.linspace(0.025, 0.040, 20)
    t = np.arange(n, dtype=np.float64) / 24.0
    truth = np.empty((n, 21, 3), dtype=np.float64)
    truth[:, 0] = np.column_stack((
        0.035 * np.sin(0.35 * t),
        0.030 * np.cos(0.40 * t),
        0.25 + 0.020 * np.sin(0.3 * t),
    ))
    for j in range(1, 21):
        direction = np.column_stack((
            np.cos(0.45 * j + 0.22 * t),
            np.sin(0.45 * j + 0.22 * t),
            np.full(n, 0.40),
        ))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        truth[:, j] = truth[:, parents[j]] + lengths[j - 1] * direction
    return truth, parents, lengths


def _bone_mae(x: np.ndarray, parents: np.ndarray, lengths: np.ndarray) -> float:
    """Mean absolute violation of the target bone lengths."""
    measured = np.linalg.norm(x[:, 1:] - x[:, parents[1:]], axis=-1)
    return float(np.nanmean(np.abs(measured - lengths[None, :])))


def test_sine_gaussian_noise_reduces_rmse_by_at_least_half() -> None:
    rng = np.random.default_rng(101)
    t = np.arange(240) / 24.0
    truth = _smooth_trajectory(t)
    z = truth + rng.normal(0.0, 0.01, size=truth.shape)

    x, variance, inlier = rts_smooth(z, t)

    assert _rmse(x, truth) < 0.5 * _rmse(z, truth)
    assert np.all(inlier)
    assert np.isfinite(x).all()
    assert np.isfinite(variance).all()
    assert np.all(variance >= 0.0)


def test_outlier_spikes_are_rejected_and_rmse_below_one_cm() -> None:
    rng = np.random.default_rng(202)
    t = np.arange(240) / 24.0
    truth = _smooth_trajectory(t)
    z = truth + rng.normal(0.0, 0.01, size=truth.shape)
    spikes = np.zeros_like(z, dtype=np.bool_)
    # Protect the first sample: before initialization there is no innovation
    # against which a measurement could be classified as an outlier.
    eligible = np.arange(z.shape[1], z.size)
    pick = rng.choice(eligible, size=round(0.05 * z.size), replace=False)
    spikes.flat[pick] = True
    z[spikes] += 0.5

    x, _, inlier = rts_smooth(z, t)

    assert not np.any(inlier[spikes])
    assert _rmse(x, truth) < 0.01
    assert np.all(inlier[0])
    # Gating must not throw away good data wholesale either.
    assert np.mean(inlier[~spikes]) > 0.97


def test_short_gap_filled_but_long_gap_remains_nan() -> None:
    rng = np.random.default_rng(303)
    t = np.arange(360) / 24.0
    truth = _smooth_trajectory(t)
    z = truth + rng.normal(0.0, 0.004, size=truth.shape)
    short = (t >= 4.0) & (t < 4.2)
    long = (t >= 8.0) & (t < 9.0)
    z[short] = np.nan
    z[long] = np.nan

    x, var, inlier = rts_smooth(z, t, max_gap_s=0.5)

    assert np.all(np.isfinite(x[short]))
    assert np.max(np.abs(x[short] - truth[short])) < 0.02
    assert np.isnan(x[long]).all()
    assert np.isnan(var[long]).all()
    assert not np.any(inlier[short | long])
    assert np.all(np.isfinite(x[~(short | long)]))


def test_jittered_timestamps_and_per_sample_variances() -> None:
    rng = np.random.default_rng(404)
    steps = (1.0 / 24.0) * rng.uniform(0.75, 1.25, size=239)
    t = np.concatenate(([0.0], np.cumsum(steps)))
    truth = _smooth_trajectory(t)
    z = truth + rng.normal(0.0, 0.01, size=truth.shape)
    r_time = np.full(t.shape, 1e-4)
    r_full = np.broadcast_to(r_time[:, None], z.shape).copy()

    a, avar, amask = rts_smooth(z, t, r_meas=r_time)
    b, bvar, bmask = rts_smooth(z, t, r_meas=r_full)

    assert _rmse(a, truth) < 0.5 * _rmse(z, truth)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(avar, bvar, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(amask, bmask)


def test_all_nan_column_and_boundary_gaps_do_not_warn() -> None:
    rng = np.random.default_rng(505)
    t = np.arange(120) / 24.0
    z = _smooth_trajectory(t) + rng.normal(0.0, 0.006, size=(120, 3))
    z[:, 2] = np.nan
    z[:5, 1] = np.nan
    z[-7:, 1] = np.nan

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", RuntimeWarning)
        with np.errstate(all="raise"):
            x, var, inlier = rts_smooth(z, t)

    assert not any(issubclass(w.category, RuntimeWarning) for w in seen)
    assert np.isnan(x[:, 2]).all()
    assert np.isnan(var[:, 2]).all()
    assert not inlier[:, 2].any()
    assert np.isnan(x[:5, 1]).all()
    assert np.isnan(x[-7:, 1]).all()
    assert np.isfinite(x[5:-7, 1]).all()
    assert np.isfinite(x[:, 0]).all()


def test_refine_chain_bones_fivefold_and_positions_within_one_cm() -> None:
    rng = np.random.default_rng(606)
    truth, parents, lengths = _chain_truth(240)
    noisy = truth + rng.normal(0.0, 0.003, size=truth.shape)

    refined = refine_bone_lengths(noisy, parents, lengths, iters=10)

    assert _bone_mae(refined, parents, lengths) < _bone_mae(noisy, parents, lengths) / 5.0
    assert np.max(np.linalg.norm(refined - truth, axis=2)) < 0.01
    assert refined.shape == truth.shape
    assert np.isfinite(refined).all()


def test_refinement_missing_joint_and_whole_nan_frame() -> None:
    rng = np.random.default_rng(707)
    truth, parents, lengths = _chain_truth(96)
    noisy = truth + rng.normal(0.0, 0.002, size=truth.shape)
    noisy[35:37] = np.nan
    noisy[48:52, 10] = np.nan
    observed = np.isfinite(noisy[..., 0])

    refined = refine_bone_lengths(noisy, parents, lengths, observed=observed)

    assert np.isnan(refined[35:37]).all()
    assert np.isfinite(refined[48:52, 10]).all()
    # The unobserved joint is placed by its bones and temporal smoothness.
    assert np.max(np.linalg.norm(refined[48:52, 10] - truth[48:52, 10], axis=-1)) < 0.01
    assert _bone_mae(refined, parents, lengths) < _bone_mae(noisy, parents, lengths) / 5.0


def test_shapes_dtypes_and_empty_inputs() -> None:
    t = np.array([0.0, 0.05, 0.13, 0.19], dtype=np.float32)
    z = np.array([[0.0, 0.1], [0.1, np.nan], [0.2, 0.3], [0.3, 0.4]],
                 dtype=np.float32)
    x, var, inlier = rts_smooth(z, t, r_meas=np.full(z.shape, 1e-4))
    assert x.shape == var.shape == inlier.shape == z.shape
    assert x.dtype == var.dtype == np.float64
    assert inlier.dtype == np.bool_

    empty, empty_var, empty_mask = rts_smooth(np.empty((0, 2)), np.empty(0))
    assert empty.shape == empty_var.shape == empty_mask.shape == (0, 2)

    tree = np.array([-1, 0, 0], dtype=np.int64)
    points = np.array([[[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, 0.04, 0.0]]],
                      dtype=np.float32)
    refined = refine_bone_lengths(points, tree, np.array([0.03, 0.04]))
    assert refined.shape == (1, 3, 3)
    assert refined.dtype == np.float64
    assert np.isfinite(refined).all()


@pytest.mark.parametrize("bad_times", [
    [0.0, 0.1, 0.1],
    [0.0, 0.2, 0.1],
    [0.0, np.nan, 0.2],
])
def test_rts_rejects_nonmonotone_timestamps(bad_times: list[float]) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        rts_smooth(np.zeros((3, 2)), np.asarray(bad_times))


# --- Added in review: checks the reply's tests did not cover. -------------


def _dense_ca_posterior(z, t, q, r, init_var):
    """Independent reference: solve the full 3T x 3T information system of the
    1-D white-jerk CA model (prior at the first sample, which is not reused as
    a measurement) and return posterior mean and marginal variance of p."""
    n = len(z)
    a = np.zeros((3 * n, 3 * n))
    b = np.zeros(3 * n)
    p0_inv = np.diag(1.0 / np.asarray(init_var))
    a[:3, :3] += p0_inv
    b[:3] += p0_inv @ np.array([z[0], 0.0, 0.0])
    for k in range(n - 1):
        dt = t[k + 1] - t[k]
        phi = np.array([[1.0, dt, dt * dt / 2.0], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        qm = q * np.array([[dt**5 / 20, dt**4 / 8, dt**3 / 6],
                           [dt**4 / 8, dt**3 / 3, dt**2 / 2],
                           [dt**3 / 6, dt**2 / 2, dt]])
        m = np.hstack([-phi, np.eye(3)])
        a[3 * k:3 * k + 6, 3 * k:3 * k + 6] += m.T @ np.linalg.inv(qm) @ m
    for k in range(1, n):
        if np.isfinite(z[k]):
            a[3 * k, 3 * k] += 1.0 / r[k]
            b[3 * k] += z[k] / r[k]
    return np.linalg.solve(a, b)[0::3], np.diag(np.linalg.inv(a))[0::3]


def test_rts_equals_dense_batch_posterior_exactly() -> None:
    """Exact Phi/Q discretisation and a correct RTS pass give the batch MAP
    solution (and marginal variances) to round-off, for non-uniform t,
    per-sample r and missing samples. Approximate Q or a wrong smoother gain
    would still pass the RMSE tests above but not this one."""
    rng = np.random.default_rng(808)
    t = np.cumsum(rng.uniform(0.03, 0.06, 80))
    z = np.sin(3.0 * t) + rng.normal(0.0, 0.01, 80)
    z[20:24] = np.nan
    r = rng.uniform(5e-5, 2e-4, 80)
    ref_x, ref_var = _dense_ca_posterior(z, t, 50.0, r, (1e-2, 1.0, 10.0))

    x, var, inlier = rts_smooth(z[:, None], t, r_meas=r, gate_chi2=1e12, max_gap_s=10.0)

    assert inlier[np.isfinite(z), 0].all()
    np.testing.assert_allclose(x[:, 0], ref_x, rtol=0, atol=1e-9)
    np.testing.assert_allclose(var[:, 0], ref_var, rtol=1e-7)


def test_spike_on_first_sample_and_right_after_gap_is_rejected() -> None:
    """A forward-only gate initialises on a first-sample spike and lets a
    spike right after an occlusion gap through (the prediction variance has
    grown). Both must be rejected and must not distort the output."""
    rng = np.random.default_rng(909)
    t = np.arange(240) / 24.0
    truth = _smooth_trajectory(t)
    z = truth + rng.normal(0.0, 0.01, size=truth.shape)
    z[0, 0] += 0.5                 # spike on the very first sample
    z[100:110] = np.nan            # 0.42 s occlusion ...
    z[110, 1] += 0.5               # ... and a spike on re-acquisition
    z[239, 2] -= 0.5               # spike on the last sample

    x, _, inlier = rts_smooth(z, t, max_gap_s=1.0)

    assert not inlier[0, 0] and not inlier[110, 1] and not inlier[239, 2]
    assert np.nanmax(np.abs(x - truth)) < 0.03
    assert _rmse(x, truth) < 0.01


def test_refine_runtime_t2000_j21_under_5s() -> None:
    import time

    rng = np.random.default_rng(1001)
    truth, parents, lengths = _chain_truth(2000)
    noisy = truth + rng.normal(0.0, 0.003, size=truth.shape)
    start = time.perf_counter()
    refined = refine_bone_lengths(noisy, parents, lengths)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0
    assert _bone_mae(refined, parents, lengths) < _bone_mae(noisy, parents, lengths) / 5.0
