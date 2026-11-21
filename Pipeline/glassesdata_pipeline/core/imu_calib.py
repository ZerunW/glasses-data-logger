"""Camera-IMU calibration helpers on top of :mod:`.imu_filter` (NumPy only, pure functions).

* :func:`visual_rates` - mean body angular velocity of the camera over each pair of consecutive
  valid frames, from a trajectory's rotations.
* :func:`interval_mean` - exact mean of a linearly interpolated signal over time intervals (the
  IMU counterpart of a visual rate: no aliasing between 24 fps and the IMU rate).
* :func:`fit_extrinsic_bias` - alternate ``R_cam_imu`` (Kabsch/Huber) and the gyro bias.
* :func:`refine_offset` - time offset that minimises the vector rate residual (sub-sample).
* :func:`gravity_preintegration` - gravity in the trajectory frame and accelerometer bias from a
  metric trajectory and the accelerometer (linear least squares on frame triplets, velocities
  eliminated), optionally with a scale factor.
* :func:`savgol_second_derivative` - local quadratic fit acceleration of frame positions.

Conventions: ``v_cam = R_cam_imu @ v_imu``; an IMU sample stamped ``t`` measures the motion at
video time ``t - offset`` (PIPELINE.md: IMU time = video time + offset); the gyroscope reads
``R_cam_imu^T omega_cam + bias``; the accelerometer reads ``R_imu_world (a - g) + b_acc`` (m/s^2,
``g = (0, 0, -9.81)`` in a z-up world).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import geometry as geo
from .imu_filter import G0, estimate_rotation_extrinsic


def visual_rates(t: ArrayLike, R: ArrayLike, valid: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Mean camera angular velocity (camera frame, rad/s) between consecutive valid frames.

    Returns ``(t0 (M,), t1 (M,), omega (M, 3), index (M,))`` for frame pairs ``(i, i+1)`` both
    valid; ``omega = log(R_i^T R_{i+1}) / (t_{i+1} - t_i)``.
    """
    t = np.asarray(t, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    v = np.asarray(valid, dtype=bool) & np.all(np.isfinite(R), axis=(1, 2))
    i = np.flatnonzero(v[:-1] & v[1:])
    if i.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros((0, 3)), i
    dR = np.swapaxes(R[i], -1, -2) @ R[i + 1]
    dt = t[i + 1] - t[i]
    return t[i], t[i + 1], geo.so3_log(dR) / dt[:, None], i


def _cumulative(t: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cumulative trapezoid integral of ``x`` (N, D) at the sample times."""
    c = np.zeros_like(x)
    c[1:] = np.cumsum(0.5 * (x[1:] + x[:-1]) * np.diff(t)[:, None], axis=0)
    return c


def _integral_at(t: NDArray[np.float64], x: NDArray[np.float64], c: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Exact integral from ``t[0]`` to ``q`` of the piecewise-linear interpolant of ``x``."""
    k = np.clip(np.searchsorted(t, q, side="right") - 1, 0, len(t) - 2)
    h = (q - t[k])[:, None]
    dt = (t[k + 1] - t[k])[:, None]
    slope = (x[k + 1] - x[k]) / dt
    return c[k] + x[k] * h + 0.5 * slope * h * h


def interval_mean(t: ArrayLike, x: ArrayLike, a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Mean of the linear interpolant of ``x`` (N, D) over ``[a_k, b_k]`` (M,) each.

    NaN where an interval leaves ``[t[0], t[-1]]`` or touches a NaN sample."""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    x2 = x.reshape(len(t), -1)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.full((len(a), x2.shape[1]), np.nan)
    if len(t) < 2 or len(a) == 0:
        return out.reshape((len(a),) + x.shape[1:])
    bad = ~np.all(np.isfinite(x2), axis=1)
    xz = np.where(bad[:, None], 0.0, x2)
    c = _cumulative(t, xz)
    nb = np.concatenate([[0], np.cumsum(bad)])  # bad samples before index k: nb[k]
    inside = (a >= t[0]) & (b <= t[-1]) & (b > a)
    if np.any(inside):
        ia = np.clip(np.searchsorted(t, a[inside], side="right") - 1, 0, len(t) - 1)
        ib = np.clip(np.searchsorted(t, b[inside], side="left"), 0, len(t) - 1)
        touched = (nb[ib + 1] - nb[ia]) > 0
        val = (_integral_at(t, xz, c, b[inside]) - _integral_at(t, xz, c, a[inside])) / (b[inside] - a[inside])[:, None]
        val[touched] = np.nan
        out[inside] = val
    return out.reshape((len(a),) + x.shape[1:])


def fit_extrinsic_bias(
    w_cam: ArrayLike,
    w_imu: ArrayLike,
    bias: ArrayLike | None = None,
    estimate_bias: bool = True,
    huber: float = 0.1,
    iters: int = 8,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Alternate ``R = argmin sum |w_cam - R (w_imu - b)|`` (Huber IRLS) and
    ``b = mean_huber(w_imu - R^T w_cam)``. Returns ``(R_cam_imu, bias, rms)``."""
    wc = np.asarray(w_cam, dtype=np.float64).reshape(-1, 3)
    wi = np.asarray(w_imu, dtype=np.float64).reshape(-1, 3)
    b = np.zeros(3) if bias is None else np.asarray(bias, dtype=np.float64).reshape(3)
    ok = np.all(np.isfinite(wc), axis=1) & np.all(np.isfinite(wi), axis=1)
    wc, wi = wc[ok], wi[ok]
    R, rms = estimate_rotation_extrinsic(wc, wi - b, huber=huber)
    if estimate_bias:
        for _ in range(int(iters)):
            res = wi - wc @ R  # = w_imu - R^T w_cam, rows
            r = res - b
            n = np.linalg.norm(r, axis=1)
            w = np.where(n > huber, huber / np.maximum(n, 1e-300), 1.0)
            b = np.sum(res * w[:, None], axis=0) / np.sum(w)
            R, rms = estimate_rotation_extrinsic(wc, wi - b, huber=huber)
    return R, b, rms


def refine_offset(
    t0: ArrayLike, t1: ArrayLike, w_cam: ArrayLike, t_imu: ArrayLike, gyro: ArrayLike,
    R_cam_imu: ArrayLike, bias: ArrayLike, center: float, half_window: float = 0.02, step: float = 0.0005,
    huber: float = 0.1,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Offset (s, IMU time = video time + offset) minimising the Huber cost of
    ``w_cam - R (mean_gyro(t0 + o, t1 + o) - b)`` over a grid ``center +- half_window``, refined
    by a parabola through the best grid point. Returns ``(offset, grid, cost)``."""
    R = np.asarray(R_cam_imu, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    wc = np.asarray(w_cam, dtype=np.float64)
    grid = center + np.arange(-half_window, half_window + 0.5 * step, step)
    cost = np.full(len(grid), np.inf)
    t0 = np.asarray(t0, dtype=np.float64)
    t1 = np.asarray(t1, dtype=np.float64)
    for k, o in enumerate(grid):
        wi = interval_mean(t_imu, gyro, t0 + o, t1 + o)
        r = np.linalg.norm(wc - (wi - b) @ R.T, axis=1)
        r = r[np.isfinite(r)]
        if r.size < 10:
            continue
        cost[k] = float(np.mean(np.where(r <= huber, r * r, 2 * huber * r - huber * huber)))
    k = int(np.argmin(cost))
    best = float(grid[k])
    if 0 < k < len(grid) - 1 and np.all(np.isfinite(cost[k - 1:k + 2])):
        den = cost[k - 1] - 2 * cost[k] + cost[k + 1]
        if den > 0:
            best += float(np.clip(0.5 * (cost[k - 1] - cost[k + 1]) / den, -0.5, 0.5)) * step
    return best, grid, cost


def slerp_rotations(t: ArrayLike, R: ArrayLike, valid: ArrayLike, q: ArrayLike) -> NDArray[np.float64]:
    """Rotations at times ``q`` by geodesic interpolation between consecutive valid frames
    (NaN outside or across an invalid frame)."""
    t = np.asarray(t, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    v = np.asarray(valid, dtype=bool)
    q = np.asarray(q, dtype=np.float64)
    out = np.full((len(q), 3, 3), np.nan)
    k = np.clip(np.searchsorted(t, q, side="right") - 1, 0, len(t) - 2)
    ok = (q >= t[0]) & (q <= t[-1]) & v[k] & v[k + 1]
    if np.any(ok):
        kk = k[ok]
        a = (q[ok] - t[kk]) / (t[kk + 1] - t[kk])
        d = geo.so3_log(np.swapaxes(R[kk], -1, -2) @ R[kk + 1])
        out[ok] = R[kk] @ geo.so3_exp(d * a[:, None])
    return out


def gravity_preintegration(
    t: ArrayLike,
    T_slam_cam: ArrayLike,
    valid: ArrayLike,
    t_imu: ArrayLike,
    acc: ArrayLike,
    R_cam_imu: ArrayLike,
    offset: float,
    window_s: float = 0.5,
    substeps: int = 8,
    estimate_bias: bool = True,
    estimate_scale: bool = False,
    huber: float = 0.3,
    iters: int = 5,
    fix_g_norm: bool = True,
    bias_sigma: float | None = 0.5,
    eq_sigma: float = 0.1,
) -> dict[str, Any]:
    """Gravity vector in the trajectory frame (and accelerometer bias, IMU frame) from a metric
    trajectory and the accelerometer, velocities eliminated.

    For frames ``i < j < k`` (``j - i = k - j = m`` frames, ``m ~ window_s / 2`` seconds, all frames
    in between valid), with ``P_ab = int_{t_a}^{t_b} (t_b - s) R(s) f(s) ds`` and
    ``Q_ab = int (t_b - s) R(s) ds`` (``R = R_slam_imu``, slerped between frames on a grid of
    ``substeps`` per frame; ``f`` the accelerometer in m/s^2 at IMU time ``s + offset``)::

        dp_ij dt_ik - dp_ik dt_ij = 0.5 g (dt_ij^2 dt_ik - dt_ik^2 dt_ij)
                                    + (P_ij dt_ik - P_ik dt_ij) - (Q_ij dt_ik - Q_ik dt_ij) b

    is linear in ``g`` (3) and ``b`` (3) (and in a position scale ``s`` multiplying ``dp`` when
    ``estimate_scale``). Solved by least squares with Huber reweighting (``huber`` m/s^2 of the
    equivalent acceleration residual).

    The bias is only observable through rotation of the head (a bias fixed in the IMU frame
    turns with it, gravity does not); on short or rotation-poor clips it trades off against the
    gravity direction. Two regularisers keep it in check: ``fix_g_norm`` constrains ``|g| = 9.81``
    (linearised penalty rows, iterated), and ``bias_sigma`` (m/s^2) is a zero-mean Gaussian prior
    on each bias component relative to an equation noise of ``eq_sigma`` (m/s^2). ``bias_std``
    in the result is the posterior standard deviation of the bias (observability check).

    Returns ``{"g" (3,), "bias" (3,), "scale", "rms", "n", "g_norm"}`` or ``{"n": 0}`` when there
    are too few triplets.
    """
    t = np.asarray(t, dtype=np.float64)
    Tc = np.asarray(T_slam_cam, dtype=np.float64)
    v = np.asarray(valid, dtype=bool) & np.all(np.isfinite(Tc), axis=(1, 2))
    Rci = np.asarray(R_cam_imu, dtype=np.float64)
    t_imu = np.asarray(t_imu, dtype=np.float64)
    acc = np.asarray(acc, dtype=np.float64)
    n = len(t)
    fps = 1.0 / float(np.median(np.diff(t))) if n > 1 else 24.0
    m = max(1, int(round(0.5 * window_s * fps)))
    S = int(substeps)
    # Fine grid aligned to the frames: index i * S is frame i.
    alpha = np.arange(S) / S
    grid = (t[:-1, None] + (t[1:] - t[:-1])[:, None] * alpha[None, :]).reshape(-1)
    grid = np.concatenate([grid, t[-1:]])
    Rcam = slerp_rotations(t, Tc[:, :3, :3], v, grid)
    R = Rcam @ Rci
    f = np.stack([np.interp(grid + offset, t_imu, acc[:, d], left=np.nan, right=np.nan) for d in range(3)], axis=1)
    good = np.all(np.isfinite(R), axis=(1, 2)) & np.all(np.isfinite(f), axis=1)
    X = np.einsum("nij,nj->ni", np.where(good[:, None, None], R, 0.0), np.where(good[:, None], f, 0.0))
    Y = np.where(good[:, None, None], R, 0.0).reshape(-1, 9)
    C0x, C1x = _cumulative(grid, X), _cumulative(grid, X * grid[:, None])
    C0y, C1y = _cumulative(grid, Y), _cumulative(grid, Y * grid[:, None])
    badc = np.concatenate([[0], np.cumsum(~good)])

    def integral(a_idx, b_idx, C0, C1):
        tb = grid[b_idx][:, None]
        return tb * (C0[b_idx] - C0[a_idx]) - (C1[b_idx] - C1[a_idx])

    i = np.arange(0, n - 2 * m)
    j, k = i + m, i + 2 * m
    gi, gj, gk = i * S, j * S, k * S
    seg_ok = (badc[gk + 1] - badc[gi]) == 0
    seg_ok &= np.array([v[a:c + 1].all() for a, c in zip(i, k)], dtype=bool) if len(i) else np.zeros(0, bool)
    i, j, k, gi, gj, gk = (x[seg_ok] for x in (i, j, k, gi, gj, gk))
    if len(i) < 4:
        return {"n": 0}
    p = Tc[:, :3, 3]
    dt_ij = (t[j] - t[i])[:, None]
    dt_ik = (t[k] - t[i])[:, None]
    P_ij, P_ik = integral(gi, gj, C0x, C1x), integral(gi, gk, C0x, C1x)
    Q_ij = integral(gi, gj, C0y, C1y).reshape(-1, 3, 3)
    Q_ik = integral(gi, gk, C0y, C1y).reshape(-1, 3, 3)
    dp = (p[j] - p[i]) * dt_ik - (p[k] - p[i]) * dt_ij               # (E, 3)
    cg = 0.5 * (dt_ij**2 * dt_ik - dt_ik**2 * dt_ij)                 # (E, 1)
    Pt = P_ij * dt_ik - P_ik * dt_ij                                  # (E, 3)
    Qt = Q_ij * dt_ik[:, :, None] - Q_ik * dt_ij[:, :, None]          # (E, 3, 3)
    E = len(i)
    cols = 3 + (3 if estimate_bias else 0) + (1 if estimate_scale else 0)
    A = np.zeros((E, 3, cols))
    A[:, :, :3] = cg[:, :, None] * np.eye(3)[None]
    c = 3
    if estimate_bias:
        A[:, :, c:c + 3] = -Qt
        c += 3
    rhs = dp - Pt
    if estimate_scale:
        A[:, :, c] = -dp
        rhs = -Pt
    # Normalise each equation to acceleration units (divide by |cg|).
    norm = np.abs(cg[:, 0])
    A /= norm[:, None, None]
    rhs = rhs / norm[:, None]
    w = np.ones(E)
    x = np.zeros(cols)
    g_dir = None
    prior_rows = np.zeros((0, cols))
    if estimate_bias and bias_sigma is not None and bias_sigma > 0:
        prior_rows = np.zeros((3, cols))
        prior_rows[:, 3:6] = np.eye(3) * (eq_sigma / bias_sigma)
    for _ in range(max(1, int(iters)) + (3 if fix_g_norm else 0)):
        sw = np.sqrt(w)[:, None, None]
        rows = [(A * sw).reshape(-1, cols), prior_rows]
        rhs_rows = [(rhs * sw[:, :, 0]).reshape(-1), np.zeros(len(prior_rows))]
        if fix_g_norm and g_dir is not None:
            c_row = np.zeros((1, cols))
            c_row[0, :3] = g_dir * 100.0
            rows.append(c_row)
            rhs_rows.append(np.array([100.0 * G0]))
        AA, bb = np.concatenate(rows), np.concatenate(rhs_rows)
        x, *_ = np.linalg.lstsq(AA, bb, rcond=None)
        g_dir = x[:3] / max(np.linalg.norm(x[:3]), 1e-9)
        r = np.linalg.norm(np.einsum("eij,j->ei", A, x) - rhs, axis=1)
        w = np.where(r <= huber, 1.0, huber / np.maximum(r, 1e-300))
    r = np.linalg.norm(np.einsum("eij,j->ei", A, x) - rhs, axis=1)
    bias_std = np.full(3, np.nan)
    if estimate_bias:
        sw = np.sqrt(w)[:, None, None]
        H = (A * sw).reshape(-1, cols)
        H = np.concatenate([H / eq_sigma, prior_rows / eq_sigma])
        try:
            cov = np.linalg.inv(H.T @ H)
            bias_std = np.sqrt(np.clip(np.diag(cov)[3:6], 0, None))
        except np.linalg.LinAlgError:
            pass
    out: dict[str, Any] = {
        "g": x[:3], "g_norm": float(np.linalg.norm(x[:3])), "rms": float(np.sqrt(np.mean(r * r))), "n": int(E),
        "bias": x[3:6] if estimate_bias else np.zeros(3), "scale": 1.0, "window_frames": int(2 * m),
        "bias_std": bias_std,
    }
    if estimate_scale:
        out["scale"] = float(x[-1])
    return out


def savgol_second_derivative(t: ArrayLike, p: ArrayLike, valid: ArrayLike, half_window: int) -> NDArray[np.float64]:
    """Acceleration ``(T, D)`` of positions ``p`` by a local quadratic least-squares fit over
    ``+- half_window`` frames (uniform frame spacing assumed; NaN where the window has an invalid
    frame or leaves the clip)."""
    t = np.asarray(t, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    v = np.asarray(valid, dtype=bool) & np.all(np.isfinite(p), axis=1)
    T = len(t)
    out = np.full(p.shape, np.nan)
    k = int(half_window)
    if T < 2 * k + 1 or k < 1:
        return out
    dt = float(np.median(np.diff(t)))
    x = np.arange(-k, k + 1) * dt
    V = np.stack([np.ones_like(x), x, x * x], axis=1)
    coef = np.linalg.pinv(V)[2] * 2.0  # second derivative = 2 c2
    pz = np.where(v[:, None], p, 0.0)
    bad = (~v).astype(np.int64)
    nb = np.concatenate([[0], np.cumsum(bad)])
    for c_idx in range(k, T - k):
        if nb[c_idx + k + 1] - nb[c_idx - k] == 0:
            out[c_idx] = coef @ pz[c_idx - k:c_idx + k + 1]
    return out


def gravity_alignment(g_slam: ArrayLike, R_slam_cam_first: ArrayLike) -> NDArray[np.float64]:
    """``R_world_slam``: rotates the gravity direction ``g_slam`` to ``-z`` and then about z so
    that the first camera's optical axis (projected on the horizontal plane) points along +x."""
    g = np.asarray(g_slam, dtype=np.float64)
    R_tilt = geo.rot_from_two_vectors(g / np.linalg.norm(g), np.array([0.0, 0.0, -1.0]))
    R_world_cam0 = R_tilt @ np.asarray(R_slam_cam_first, dtype=np.float64)
    return geo.gravity_align(R_world_cam0) @ R_tilt


def angle_deg(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Angle (degrees) between vectors ``(..., 3)`` (NaN when either is zero/NaN)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.sum(a * b, axis=-1) / (na * nb)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


__all__ = [
    "visual_rates", "interval_mean", "fit_extrinsic_bias", "refine_offset", "slerp_rotations",
    "gravity_preintegration", "savgol_second_derivative", "gravity_alignment", "angle_deg", "G0",
]
