"""Deterministic NumPy-only tests for glassesdata_pipeline.core.elbow."""

from __future__ import annotations

import time

import numpy as np
import pytest

from glassesdata_pipeline.core.elbow import (
    circle_point,
    elbow_circle,
    plane_from_image_line,
    solve_elbow,
)

L_UP, L_FORE = 0.30, 0.27
DOWN = np.array([0.0, 0.0, -1.0])


def _arm(T: int = 240, swivel=None, seed: int = 0):
    """Synthetic right arm seen from a head camera (x forward, y left, z up).

    Returns camera centres, shoulder, wrist and the true elbow, whose swivel
    angle (circle_point convention, reference = straight down) is ``swivel(t)``.
    """
    t = np.arange(T) / 24.0
    cam = np.stack([0.05 * np.sin(0.5 * t), 0.02 * np.sin(0.3 * t), np.full(T, 1.65)], axis=1)
    shoulder = cam + np.array([-0.05, -0.18, -0.25])
    d = np.stack([0.8 + 0.1 * np.sin(0.7 * t), 0.1 + 0.2 * np.sin(0.5 * t), -0.4 + 0.2 * np.sin(0.9 * t)], axis=1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    reach = 0.42 + 0.06 * np.sin(1.1 * t)
    wrist = shoulder + reach[:, None] * d
    phi_true = swivel(t) if swivel is not None else 0.1 + 0.35 * np.sin(2 * np.pi * t / 6.0)
    c, n, r, st = elbow_circle(shoulder, wrist, L_UP, L_FORE)
    assert np.all(st == 0)
    elbow = circle_point(c, n, r, DOWN, phi_true)
    return t, cam, shoulder, wrist, elbow, phi_true


def _cue_planes(cam, wrist, elbow, noise_deg: float, rng) -> np.ndarray:
    n, d = plane_from_image_line(cam, wrist - cam, elbow - cam)
    if noise_deg > 0:
        n = n + rng.normal(0.0, np.radians(noise_deg), n.shape)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        d = -np.sum(n * wrist, axis=1)          # plane still passes through the wrist ray
    return np.column_stack([n, d])


def _err(a, b):
    return np.linalg.norm(a - b, axis=-1)


# ---------------------------------------------------------------- 1. elbow_circle


def test_elbow_circle_known_configuration_and_degenerate_cases() -> None:
    c, n, r, st = elbow_circle([0.0, 0, 0], [0.4, 0, 0], 0.3, 0.3)
    np.testing.assert_allclose(c, [0.2, 0, 0], atol=1e-15)
    np.testing.assert_allclose(n, [1.0, 0, 0], atol=1e-15)
    assert r == pytest.approx(np.sqrt(0.3**2 - 0.2**2))
    assert st == 0

    c, n, r, st = elbow_circle([0.0, 0, 0], [0.0, 0.7, 0], 0.3, 0.3)       # too far
    assert st == 1 and r == 0.0
    np.testing.assert_allclose(c, [0, 0.3, 0], atol=1e-15)

    c, n, r, st = elbow_circle([0.0, 0, 0], [0.0, 0, 0.05], 0.3, 0.2)      # too close
    assert st == 2 and r == 0.0 and np.all(np.isfinite(c)) and np.all(np.isfinite(n))
    np.testing.assert_allclose(c, [0, 0, 0.3], atol=1e-15)

    c, n, r, st = elbow_circle([1.0, 1, 1], [1.0, 1, 1], 0.3, 0.3)         # S == W
    assert st == 2 and np.all(np.isfinite(c)) and np.linalg.norm(n) == pytest.approx(1.0)
    assert np.linalg.norm(c - [1, 1, 1]) == pytest.approx(0.3)

    c, n, r, st = elbow_circle([[0.0, 0, 0], [np.nan, 0, 0]], [[0.4, 0, 0], [0.4, 0, 0]], 0.3, 0.3)
    assert st.tolist() == [0, 3] and np.isnan(c[1]).all() and np.isnan(r[1])
    assert st.dtype == np.int8


# ---------------------------------------------------------------- 2. circle points satisfy both lengths


def test_circle_points_satisfy_segment_lengths_and_convention() -> None:
    rng = np.random.default_rng(1)
    S = rng.normal(size=(200, 3))
    W = S + rng.normal(size=(200, 3)) * 0.1 + np.array([0.3, 0.0, 0.0])
    lu = rng.uniform(0.25, 0.35, 200)
    lf = rng.uniform(0.22, 0.30, 200)
    c, n, r, st = elbow_circle(S, W, lu, lf)
    ok = st == 0
    assert ok.sum() > 150
    phi = rng.uniform(-np.pi, np.pi, 200)
    ref = rng.normal(size=(200, 3))
    E = circle_point(c, n, r, ref, phi)
    np.testing.assert_allclose(_err(E[ok], S[ok]), lu[ok], atol=1e-9)
    np.testing.assert_allclose(_err(E[ok], W[ok]), lf[ok], atol=1e-9)

    # phi = 0 is the circle point closest to ref_dir; +pi/2 is n x u (right-handed)
    c1, n1, r1, _ = elbow_circle([0.0, 0, 0], [0.4, 0, 0], 0.3, 0.3)
    e0 = circle_point(c1, n1, r1, [0.2, 0.0, -1.0], 0.0)
    np.testing.assert_allclose(e0, [0.2, 0.0, -r1], atol=1e-12)
    e90 = circle_point(c1, n1, r1, [0.0, 0.0, -1.0], np.pi / 2)
    np.testing.assert_allclose(e90, c1 + r1 * np.cross([1.0, 0, 0], [0, 0, -1.0]), atol=1e-12)
    # ref_dir parallel to the normal: deterministic, finite, on the circle
    ep = circle_point(c1, n1, r1, [1.0, 0, 0], 0.3)
    assert np.all(np.isfinite(ep)) and _err(ep, c1) == pytest.approx(r1)
    np.testing.assert_array_equal(ep, circle_point(c1, n1, r1, [5.0, 0, 0], 0.3))


def test_plane_from_image_line_contains_rays_and_orientation() -> None:
    rng = np.random.default_rng(2)
    cam = rng.normal(size=(50, 3))
    a = rng.normal(size=(50, 3))
    b = rng.normal(size=(50, 3))
    n, d = plane_from_image_line(cam, a, b)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-12)
    for pt in (cam, cam + 2.0 * a, cam - 0.5 * b):
        np.testing.assert_allclose(np.sum(n * pt, axis=1) + d, 0.0, atol=1e-12)
    assert np.all(np.sum(n * np.cross(a, b - a), axis=1) > 0)
    n_par, d_par = plane_from_image_line(cam[0], a[0], 3.0 * a[0])
    assert np.isnan(n_par).all() and np.isnan(d_par)


# ---------------------------------------------------------------- 3. cues


def test_solve_with_cue_planes_median_error_below_1cm() -> None:
    rng = np.random.default_rng(3)
    t, cam, S, W, E_true, phi_true = _arm()
    planes = _cue_planes(cam, W, E_true, noise_deg=0.3, rng=rng)
    planes[::7] = np.nan                                          # some frames without a cue
    has = np.isfinite(planes[:, 0])

    E, phi, conf, method = solve_elbow(S, W, L_UP, L_FORE, cue_plane=planes)

    err = _err(E, E_true)
    assert np.median(err) < 0.01
    assert np.max(err) < 0.04
    assert np.all(method[has] == 1) and np.all(method[~has] == 2)
    assert np.all((conf >= 0) & (conf <= 1))
    assert np.median(conf[has]) > 2 * np.median(conf[~has])
    # returned phi reproduces the elbow through circle_point(prior_dir)
    c, n, r, _ = elbow_circle(S, W, L_UP, L_FORE)
    np.testing.assert_allclose(circle_point(c, n, r, DOWN, phi), E, atol=1e-12)
    np.testing.assert_allclose(_err(E, S), L_UP, atol=1e-9)
    np.testing.assert_allclose(_err(E, W), L_FORE, atol=1e-9)


def test_cue_side_overrides_a_wrong_prior() -> None:
    """The cue plane cuts the circle twice; with the prior pointing up the
    prior favours the wrong intersection, cue_side must pick the right one."""
    rng = np.random.default_rng(4)
    t, cam, S, W, E_true, _ = _arm(swivel=lambda tt: 0.2 + 0.25 * np.sin(tt))
    planes = _cue_planes(cam, W, E_true, noise_deg=0.2, rng=rng)
    # what the image shows: the forearm's direction from the wrist pixel, i.e.
    # the in-plane component perpendicular to the wrist's viewing ray
    ray = (W - cam) / np.linalg.norm(W - cam, axis=1, keepdims=True)
    fore = E_true - W
    side = fore - np.sum(fore * ray, axis=1, keepdims=True) * ray
    side -= np.sum(side * planes[:, :3], axis=1, keepdims=True) * planes[:, :3]

    E_wrong, *_ = solve_elbow(S, W, L_UP, L_FORE, prior_dir=(0, 0, 1.0), cue_plane=planes)
    E, _, conf, _ = solve_elbow(S, W, L_UP, L_FORE, prior_dir=(0, 0, 1.0), cue_plane=planes, cue_side=side)

    assert np.median(_err(E_wrong, E_true)) > 0.05                # test is meaningful
    assert np.median(_err(E, E_true)) < 0.01


# ---------------------------------------------------------------- 4. prior only


def test_prior_only_hanging_elbow_error_below_3cm() -> None:
    t, cam, S, W, E_true, _ = _arm(swivel=lambda tt: np.zeros_like(tt))
    E, phi, conf, method = solve_elbow(S, W, L_UP, L_FORE)
    assert np.max(_err(E, E_true)) < 0.03
    assert np.all(method == 2)
    assert np.all(conf <= 0.25)
    assert np.max(np.abs(phi)) < 0.02


# ---------------------------------------------------------------- 5. cues in the first half only


def test_cues_first_half_then_prior_is_smooth() -> None:
    rng = np.random.default_rng(5)
    t, cam, S, W, E_true, _ = _arm(swivel=lambda tt: 0.6 + 0.2 * np.sin(tt))
    planes = _cue_planes(cam, W, E_true, noise_deg=0.3, rng=rng)
    planes[120:] = np.nan
    E, phi, conf, method = solve_elbow(S, W, L_UP, L_FORE, cue_plane=planes)
    jumps = _err(E[1:], E[:-1])
    assert np.max(jumps) < 0.05
    assert np.median(_err(E[:110], E_true[:110])) < 0.01          # cues followed
    assert np.median(np.abs(phi[200:])) < 0.2                     # relaxes towards the prior
    assert np.all(method[:120] == 1) and np.all(method[120:] == 2)
    # the true elbow itself moves less than 5 cm per frame, the estimate must too
    assert np.max(_err(E_true[1:], E_true[:-1])) < 0.05


# ---------------------------------------------------------------- 6. NaN frames


def test_nan_frames() -> None:
    rng = np.random.default_rng(6)
    t, cam, S, W, E_true, _ = _arm()
    planes = _cue_planes(cam, W, E_true, noise_deg=0.3, rng=rng)
    S = S.copy()
    W = W.copy()
    S[50:55] = np.nan
    W[100, 1] = np.nan
    E, phi, conf, method = solve_elbow(S, W, L_UP, L_FORE, cue_plane=planes)
    bad = np.zeros(240, bool)
    bad[50:55] = True
    bad[100] = True
    assert np.isnan(E[bad]).all() and np.isnan(phi[bad]).all()
    assert np.all(method[bad] == 0) and np.all(conf[bad] == 0)
    assert np.all(np.isfinite(E[~bad]))
    assert np.median(_err(E[~bad], E_true[~bad])) < 0.01
    # all-NaN input
    E2, phi2, conf2, m2 = solve_elbow(np.full((5, 3), np.nan), np.full((5, 3), np.nan), L_UP, L_FORE)
    assert np.isnan(E2).all() and np.all(m2 == 0) and np.all(conf2 == 0)


# ---------------------------------------------------------------- extra edge cases


def test_arm_axis_parallel_to_prior_stays_finite_and_smooth() -> None:
    """Upper arm and forearm folded with the wrist straight below the shoulder:
    the prior's projection on the circle plane vanishes. No cue: the result
    must be finite, on the circle, and continuous (no frame-to-frame flips)."""
    T = 120
    t = np.arange(T) / 24.0
    S = np.tile([0.0, 0.0, 1.4], (T, 1))
    W = S + np.stack([0.02 * np.sin(t), 0.02 * np.cos(1.3 * t), np.full(T, -0.40)], axis=1)
    E, phi, conf, method = solve_elbow(S, W, L_UP, L_FORE)
    assert np.all(np.isfinite(E)) and np.all(method == 2)
    np.testing.assert_allclose(_err(E, S), L_UP, atol=1e-9)
    assert np.max(_err(E[1:], E[:-1])) < 0.02
    assert np.all(conf < 0.1)


def test_degenerate_straight_arm_frames() -> None:
    T = 30
    S = np.zeros((T, 3))
    W = np.tile([0.0, 0.6, 0.0], (T, 1))                      # beyond reach -> straight arm
    W[10:20] = [0.0, 0.45, 0.0]                                # reachable in the middle
    E, phi, conf, method = solve_elbow(S, W, L_UP, L_FORE)
    np.testing.assert_allclose(E[:10], np.tile([0, L_UP, 0], (10, 1)), atol=1e-12)
    np.testing.assert_allclose(conf[:10], 0.5)
    assert np.all(np.isfinite(E)) and np.all(method == 2)


def test_runtime_t3000_n360_under_5s() -> None:
    rng = np.random.default_rng(7)
    t, cam, S, W, E_true, _ = _arm(T=3000)
    planes = _cue_planes(cam, W, E_true, noise_deg=0.3, rng=rng)
    planes[1000:1500] = np.nan
    start = time.perf_counter()
    E, *_ = solve_elbow(S, W, L_UP, L_FORE, cue_plane=planes, n_samples=360)
    assert time.perf_counter() - start < 5.0
    assert np.median(_err(E, E_true)) < 0.01


def test_parabolic_refinement_beats_the_grid() -> None:
    """With a coarse 10 deg grid (about 3 cm between samples on the circle) the
    sub-sample refinement must still land within a few millimetres."""
    rng = np.random.default_rng(8)
    t, cam, S, W, E_true, _ = _arm()
    planes = _cue_planes(cam, W, E_true, noise_deg=0.3, rng=rng)
    E, *_ = solve_elbow(S, W, L_UP, L_FORE, cue_plane=planes, n_samples=36)
    assert np.median(_err(E, E_true)) < 0.004
