"""IMU helpers for camera glasses: rotations, gyro bias, Mahony attitude filter,
camera-IMU time offset and rotation extrinsic.

Conventions
-----------
* Quaternions are ``[w, x, y, z]``, Hamilton product, unit norm.
* Rotation matrices act on column vectors: ``v_world = R_world_sensor @ v_sensor``.
* The accelerometer measures specific force: at rest it reads ``+g`` along the
  sensor's "up" direction, i.e. ``acc = R_world_sensor.T @ (0, 0, g)``.
* The world frame is gravity aligned with z up; yaw is arbitrary.

Only NumPy is used. All functions are pure (no I/O, no global state).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

G0 = 9.80665  # standard gravity, m/s^2

_SMALL_ANGLE = 1e-4  # below this, Taylor series are used in exp/log


# ----------------------------------------------------------------------------
# Rotation helpers
# ----------------------------------------------------------------------------


def _hat(w: NDArray[np.float64]) -> NDArray[np.float64]:
    """Skew-symmetric matrix of (..., 3) vectors -> (..., 3, 3)."""
    k = np.zeros(w.shape[:-1] + (3, 3), dtype=np.float64)
    k[..., 0, 1] = -w[..., 2]
    k[..., 0, 2] = w[..., 1]
    k[..., 1, 0] = w[..., 2]
    k[..., 1, 2] = -w[..., 0]
    k[..., 2, 0] = -w[..., 1]
    k[..., 2, 1] = w[..., 0]
    return k


def quat_to_mat(q: ArrayLike) -> NDArray[np.float64]:
    """Unit quaternion(s) ``[w, x, y, z]`` (..., 4) -> rotation matrices (..., 3, 3).

    The input is normalised first, so slightly non-unit quaternions are fine.
    """
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.where(norm > 0, norm, 1.0)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    r[..., 0, 0] = 1 - 2 * (y * y + z * z)
    r[..., 0, 1] = 2 * (x * y - w * z)
    r[..., 0, 2] = 2 * (x * z + w * y)
    r[..., 1, 0] = 2 * (x * y + w * z)
    r[..., 1, 1] = 1 - 2 * (x * x + z * z)
    r[..., 1, 2] = 2 * (y * z - w * x)
    r[..., 2, 0] = 2 * (x * z - w * y)
    r[..., 2, 1] = 2 * (y * z + w * x)
    r[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def mat_to_quat(R: ArrayLike) -> NDArray[np.float64]:
    """Rotation matrices (..., 3, 3) -> unit quaternions (..., 4) with ``w >= 0``.

    Shepperd's method: for every matrix the largest of ``w, x, y, z`` is
    computed from the diagonal (well conditioned) and the others from the
    off-diagonal terms, so it is accurate for all angles including 180 deg.
    """
    m = np.asarray(R, dtype=np.float64)
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    tr = m00 + m11 + m22
    d21 = m[..., 2, 1] - m[..., 1, 2]
    d02 = m[..., 0, 2] - m[..., 2, 0]
    d10 = m[..., 1, 0] - m[..., 0, 1]
    s01 = m[..., 0, 1] + m[..., 1, 0]
    s02 = m[..., 0, 2] + m[..., 2, 0]
    s12 = m[..., 1, 2] + m[..., 2, 1]
    # 4 * (component)^2 for each candidate pivot, clipped to stay >= 0.
    big = np.stack([1 + tr, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22,
                    1 - m00 - m11 + m22], axis=-1)
    pivot = np.argmax(big, axis=-1)
    root = np.sqrt(np.maximum(np.take_along_axis(big, pivot[..., None], -1)[..., 0], 1e-300))
    # rows: candidate quaternions times 2*root (i.e. 4*pivot component * value)
    cand = np.stack([
        np.stack([root * root, d21, d02, d10], axis=-1),
        np.stack([d21, root * root, s01, s02], axis=-1),
        np.stack([d02, s01, root * root, s12], axis=-1),
        np.stack([d10, s02, s12, root * root], axis=-1),
    ], axis=-2)
    q = np.take_along_axis(cand, pivot[..., None, None], axis=-2)[..., 0, :]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., :1] < 0, -q, q)


def quat_mul(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Hamilton product ``a * b`` of (..., 4) quaternions (broadcasting).

    With ``q = a * b``, ``quat_to_mat(q) == quat_to_mat(a) @ quat_to_mat(b)``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def so3_exp(w: ArrayLike) -> NDArray[np.float64]:
    """Rotation vector(s) (..., 3) -> rotation matrices (..., 3, 3) (Rodrigues).

    Uses Taylor expansions of ``sin(t)/t`` and ``(1-cos t)/t^2`` for tiny angles.
    """
    w = np.asarray(w, dtype=np.float64)
    th2 = np.sum(w * w, axis=-1)
    th = np.sqrt(th2)
    small = th < _SMALL_ANGLE
    ths = np.where(small, 1.0, th)
    a = np.where(small, 1.0 - th2 / 6.0 + th2 * th2 / 120.0, np.sin(ths) / ths)
    b = np.where(small, 0.5 - th2 / 24.0 + th2 * th2 / 720.0, (1.0 - np.cos(ths)) / (ths * ths))
    k = _hat(w)
    eye = np.broadcast_to(np.eye(3), k.shape)
    return eye + a[..., None, None] * k + b[..., None, None] * (k @ k)


def so3_log(R: ArrayLike) -> NDArray[np.float64]:
    """Rotation matrices (..., 3, 3) -> rotation vectors (..., 3), angle in [0, pi].

    The angle is ``atan2(|v|, (tr-1)/2)`` with ``v = vee(R - R^T)/2 = sin(t) a``,
    which is well conditioned at every angle. Below 90 deg the axis comes from
    ``v``; above it from the symmetric part ``(R + R^T)/2 - cos(t) I =
    (1 - cos t) a a^T`` (largest column), with its sign taken from ``v``. That
    keeps full accuracy near pi, where ``v`` vanishes.
    """
    m = np.asarray(R, dtype=np.float64)
    v = 0.5 * np.stack([m[..., 2, 1] - m[..., 1, 2],
                        m[..., 0, 2] - m[..., 2, 0],
                        m[..., 1, 0] - m[..., 0, 1]], axis=-1)
    s = np.linalg.norm(v, axis=-1)
    c = 0.5 * (m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2] - 1.0)
    th = np.arctan2(s, c)
    # Branch 1 (angle < 90 deg): w = th / sin(th) * v.
    small = th < _SMALL_ANGLE
    s_safe = np.where(s > 0, s, 1.0)
    scale = np.where(small, 1.0 + th * th / 6.0 + 7.0 * th**4 / 360.0, th / s_safe)
    w_small = scale[..., None] * v
    # Branch 2 (angle >= 90 deg): axis from the symmetric part.
    sym = 0.5 * (m + np.swapaxes(m, -1, -2))
    one_m_c = np.where(c < 0, 1.0 - c, 1.0)
    aat = (sym - c[..., None, None] * np.eye(3)) / one_m_c[..., None, None]
    diag = np.stack([aat[..., 0, 0], aat[..., 1, 1], aat[..., 2, 2]], axis=-1)
    col = np.argmax(diag, axis=-1)
    axis = np.take_along_axis(aat, col[..., None, None], axis=-1)[..., 0]
    piv = np.sqrt(np.maximum(np.take_along_axis(diag, col[..., None], -1)[..., 0], 1e-300))
    axis = axis / piv[..., None]
    nrm = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = axis / np.where(nrm > 0, nrm, 1.0)  # zero only off-branch (c >= 0)
    flip = np.sum(axis * v, axis=-1) < 0
    axis = np.where(flip[..., None], -axis, axis)
    w_big = th[..., None] * axis
    return np.where((c < 0)[..., None], w_big, w_small)


# ----------------------------------------------------------------------------
# Accelerometer / gyroscope helpers
# ----------------------------------------------------------------------------


def detect_accel_scale(acc: ArrayLike) -> float:
    """Return the factor that converts ``acc`` to m/s^2.

    The median of the per-sample norm is compared with gravity: about 1 means
    the data is in units of g (returns 9.80665), about 9.8 means it is already
    in m/s^2 (returns 1.0). Accepted bands are [0.5, 2] and [5, 20]; anything
    else (or no finite sample) raises ``ValueError``.
    """
    a = np.asarray(acc, dtype=np.float64).reshape(-1, 3)
    n = np.linalg.norm(a, axis=1)
    n = n[np.isfinite(n)]
    if n.size == 0:
        raise ValueError("accelerometer data has no finite sample")
    med = float(np.median(n))
    if 0.5 <= med <= 2.0:
        return G0
    if 5.0 <= med <= 20.0:
        return 1.0
    raise ValueError(f"cannot infer accelerometer units: median norm {med:.3g}")


def estimate_gyro_bias(
    t: ArrayLike,
    gyro: ArrayLike,
    acc: ArrayLike,
    window_s: float = 0.5,
    gyro_thresh: float = 0.05,
    acc_thresh: float = 0.3,
) -> tuple[NDArray[np.float64], float]:
    """Gyro bias from still periods.

    The time axis is cut into consecutive windows of ``window_s``. A sample is
    quiet when ``|gyro| < gyro_thresh`` (rad/s) and ``| |acc| - g | <
    acc_thresh`` (acc in m/s^2). A window is still when at least 90% of its
    samples (and at least 3) are quiet; the 10% allowance absorbs isolated
    noise spikes. The bias is the mean gyro over the quiet samples of all
    still windows.

    Returns ``(bias (3,), still_fraction)`` where ``still_fraction`` is the
    fraction of samples that lie in still windows. If there is no still
    window the bias is zero and the fraction 0.
    """
    tt = np.asarray(t, dtype=np.float64).ravel()
    g = np.asarray(gyro, dtype=np.float64).reshape(-1, 3)
    a = np.asarray(acc, dtype=np.float64).reshape(-1, 3)
    if not (tt.size == g.shape[0] == a.shape[0]):
        raise ValueError("t, gyro and acc must have the same length")
    if tt.size == 0 or window_s <= 0:
        return np.zeros(3), 0.0
    finite = np.isfinite(tt) & np.all(np.isfinite(g), axis=1) & np.all(np.isfinite(a), axis=1)
    gn = np.linalg.norm(np.where(finite[:, None], g, np.inf), axis=1)
    an = np.linalg.norm(np.where(finite[:, None], a, np.inf), axis=1)
    quiet = finite & (gn < gyro_thresh) & (np.abs(an - G0) < acc_thresh)
    t0 = np.min(tt[np.isfinite(tt)]) if np.any(np.isfinite(tt)) else 0.0
    win = np.where(np.isfinite(tt), np.floor((np.where(np.isfinite(tt), tt, t0) - t0) / window_s), -1)
    win = win.astype(np.int64)
    ok = win >= 0
    nwin = int(win[ok].max()) + 1 if np.any(ok) else 0
    if nwin == 0:
        return np.zeros(3), 0.0
    total = np.bincount(win[ok], minlength=nwin)
    nquiet = np.bincount(win[ok], weights=quiet[ok].astype(np.float64), minlength=nwin)
    still_win = (nquiet >= 3) & (nquiet >= 0.9 * total)
    in_still = ok & still_win[np.where(ok, win, 0)]
    use = in_still & quiet
    if not np.any(use):
        return np.zeros(3), 0.0
    bias = g[use].mean(axis=0)
    return bias, float(np.count_nonzero(in_still) / tt.size)


def _accel_tilt_quat(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Quaternion R_world_sensor with zero yaw whose up axis matches ``a``.

    ``R = Rz(0) Ry(pitch) Rx(roll)`` with ``roll = atan2(ay, az)`` and
    ``pitch = atan2(-ax, hypot(ay, az))``, so that ``R.T @ e_z = a / |a|``.
    """
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.hypot(ay, az))
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    # q = q_y(pitch) * q_x(roll)
    return np.array([cp * cr, cp * sr, sp * cr, -sp * sr])


def mahony(
    t: ArrayLike,
    acc: ArrayLike,
    gyro: ArrayLike,
    kp: float = 1.0,
    ki: float = 0.0,
    bias: ArrayLike | None = None,
    q0: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Mahony complementary filter.

    Parameters
    ----------
    t : (N,) seconds, increasing. acc : (N, 3) m/s^2 (specific force).
    gyro : (N, 3) rad/s. kp, ki : proportional / integral gains (1/s, 1/s^2).
    bias : optional (3,) gyro bias subtracted from every sample.
    q0 : optional initial quaternion; otherwise the orientation is initialised
        from the first finite accelerometer sample with yaw = 0.

    Returns
    -------
    q : (N, 4) quaternions of R_world_sensor (sensor -> gravity-aligned world,
        z up, yaw continuous but arbitrary), ``w >= 0`` is *not* enforced so
        that the sequence stays continuous.

    Notes
    -----
    Between samples k-1 and k the body rate is the trapezoidal mean of the two
    gyro samples (minus bias) plus the correction ``kp * w_acc * e + ki *
    integral(e)``, where ``e = a_hat x (R^T e_z)`` rotates the predicted up
    direction towards the measured one. ``w_acc`` is 1 while
    ``dev = | |acc| - g | <= 2 m/s^2`` and ``exp(-((dev - 2) / 0.75)^2)``
    beyond (0.28 at 2.85, 0.02 at 3.5, ~0 past 4 m/s^2), so linear
    accelerations that change the norm barely tilt the estimate. (Sideways
    accelerations change the norm only to second order and cannot be detected
    this way.) Non-finite samples skip the
    correction (acc) or the propagation (gyro); non-increasing time steps are
    skipped. The loop runs on Python floats (about 5 us per sample).
    """
    tt = np.asarray(t, dtype=np.float64).ravel()
    a = np.asarray(acc, dtype=np.float64).reshape(-1, 3)
    g = np.asarray(gyro, dtype=np.float64).reshape(-1, 3)
    n = tt.size
    if not (a.shape[0] == n and g.shape[0] == n):
        raise ValueError("t, acc and gyro must have the same length")
    out = np.empty((n, 4), dtype=np.float64)
    if n == 0:
        return out
    b = np.zeros(3) if bias is None else np.asarray(bias, dtype=np.float64).reshape(3)
    gc = g - b

    if q0 is not None:
        q = np.asarray(q0, dtype=np.float64).reshape(4)
        q = q / np.linalg.norm(q)
    else:
        good = np.flatnonzero(np.all(np.isfinite(a), axis=1) & (np.linalg.norm(
            np.nan_to_num(a), axis=1) > 0))
        q = _accel_tilt_quat(a[good[0]]) if good.size else np.array([1.0, 0.0, 0.0, 0.0])
    qw, qx, qy, qz = (float(v) for v in q)
    out[0] = (qw, qx, qy, qz)

    al = a.tolist()
    gl = gc.tolist()
    tl = tt.tolist()
    ix = iy = iz = 0.0  # integral of the error
    kp = float(kp)
    ki = float(ki)
    isfinite = math.isfinite
    for k in range(1, n):
        dt = tl[k] - tl[k - 1]
        g0, g1 = gl[k - 1], gl[k]
        if not (dt > 0.0 and isfinite(dt)):
            out[k] = (qw, qx, qy, qz)
            continue
        wx = 0.5 * (g0[0] + g1[0])
        wy = 0.5 * (g0[1] + g1[1])
        wz = 0.5 * (g0[2] + g1[2])
        if not (isfinite(wx) and isfinite(wy) and isfinite(wz)):
            # fall back to whichever endpoint is finite, else hold
            if all(isfinite(v) for v in g0):
                wx, wy, wz = g0
            elif all(isfinite(v) for v in g1):
                wx, wy, wz = g1
            else:
                out[k] = (qw, qx, qy, qz)
                continue
        ax_, ay_, az_ = al[k - 1]
        an = math.sqrt(ax_ * ax_ + ay_ * ay_ + az_ * az_) if (
            isfinite(ax_) and isfinite(ay_) and isfinite(az_)) else 0.0
        if an > 0.0:
            ax_ /= an
            ay_ /= an
            az_ /= an
            # predicted up direction in the sensor frame: third row of R
            vx = 2.0 * (qx * qz - qw * qy)
            vy = 2.0 * (qy * qz + qw * qx)
            vz = 1.0 - 2.0 * (qx * qx + qy * qy)
            ex = ay_ * vz - az_ * vy
            ey = az_ * vx - ax_ * vz
            ez = ax_ * vy - ay_ * vx
            dev = abs(an - G0)
            wacc = 1.0 if dev <= 2.0 else math.exp(-((dev - 2.0) / 0.75) ** 2)
            if ki > 0.0:
                ix += ki * wacc * ex * dt
                iy += ki * wacc * ey * dt
                iz += ki * wacc * ez * dt
            wx += kp * wacc * ex + ix
            wy += kp * wacc * ey + iy
            wz += kp * wacc * ez + iz
        else:
            wx += ix
            wy += iy
            wz += iz
        # q <- q * exp(w dt / 2)
        hx, hy, hz = 0.5 * wx * dt, 0.5 * wy * dt, 0.5 * wz * dt
        h = math.sqrt(hx * hx + hy * hy + hz * hz)
        if h < 1e-8:
            dw, sc = 1.0 - 0.5 * h * h, 1.0 - h * h / 6.0
        else:
            dw, sc = math.cos(h), math.sin(h) / h
        dx, dy, dz = sc * hx, sc * hy, sc * hz
        nw = qw * dw - qx * dx - qy * dy - qz * dz
        nx = qw * dx + qx * dw + qy * dz - qz * dy
        ny = qw * dy - qx * dz + qy * dw + qz * dx
        nz = qw * dz + qx * dy - qy * dx + qz * dw
        inv = 1.0 / math.sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
        qw, qx, qy, qz = nw * inv, nx * inv, ny * inv, nz * inv
        out[k] = (qw, qx, qy, qz)
    return out


def gravity_in_sensor(q: ArrayLike) -> NDArray[np.float64]:
    """Unit gravity direction (pointing down) in the sensor frame: ``-R^T e_z``.

    ``q`` is (..., 4) R_world_sensor; returns (..., 3).
    """
    r = quat_to_mat(q)
    return -r[..., 2, :]


def resample(t_src: ArrayLike, x_src: ArrayLike, t_dst: ArrayLike) -> NDArray[np.float64]:
    """Linear interpolation of ``x_src`` (N,) or (N, D) at ``t_dst`` (M,).

    Samples are sorted by time first; non-finite source times are dropped.
    Destinations outside ``[t_src.min(), t_src.max()]`` are NaN. NaN source
    values propagate to the destinations that interpolate from them.
    """
    ts = np.asarray(t_src, dtype=np.float64).ravel()
    xs = np.asarray(x_src, dtype=np.float64)
    td = np.asarray(t_dst, dtype=np.float64).ravel()
    one_d = xs.ndim == 1
    xs2 = xs.reshape(xs.shape[0], -1)
    if xs2.shape[0] != ts.size:
        raise ValueError("t_src and x_src lengths differ")
    keep = np.isfinite(ts)
    ts, xs2 = ts[keep], xs2[keep]
    out = np.full((td.size, xs2.shape[1]), np.nan)
    if ts.size == 0:
        return out[:, 0] if one_d else out
    order = np.argsort(ts, kind="stable")
    ts, xs2 = ts[order], xs2[order]
    inside = np.isfinite(td) & (td >= ts[0]) & (td <= ts[-1])
    if np.any(inside):
        for j in range(xs2.shape[1]):
            out[inside, j] = np.interp(td[inside], ts, xs2[:, j])
    return out[:, 0] if one_d else out


def _rate_norm(w: ArrayLike) -> NDArray[np.float64]:
    w = np.asarray(w, dtype=np.float64)
    return np.linalg.norm(w, axis=-1) if w.ndim == 2 else w.ravel()


def estimate_time_offset(
    t_a: ArrayLike,
    w_a: ArrayLike,
    t_b: ArrayLike,
    w_b: ArrayLike,
    max_offset_s: float = 0.2,
    rate_hz: float = 200.0,
) -> tuple[float, float]:
    """Offset ``d`` such that signal b(t) ~ signal a(t + d).

    ``w_a``/``w_b`` are angular velocities (N, 3) (or their norms (N,)). Only
    the norm is used, which does not depend on the unknown camera-IMU
    rotation. Both norms are resampled to a common ``rate_hz`` grid over the
    overlap of the two time ranges. For every integer lag L in
    ``[-max_offset_s, max_offset_s] * rate_hz`` the Pearson correlation of
    ``B[i]`` and ``A[i + L]`` is computed over the pairs where both exist
    (each segment has its own mean removed). The best lag is refined by a
    parabola through its two neighbours.

    Returns ``(d_seconds, peak_correlation)``; ``(nan, nan)`` if there are
    too few overlapping samples.
    """
    ta = np.asarray(t_a, dtype=np.float64).ravel()
    tb = np.asarray(t_b, dtype=np.float64).ravel()
    na = _rate_norm(w_a)
    nb = _rate_norm(w_b)
    lo = max(np.nanmin(ta), np.nanmin(tb))
    hi = min(np.nanmax(ta), np.nanmax(tb))
    if not hi > lo:
        return float("nan"), float("nan")
    grid = np.arange(lo, hi, 1.0 / rate_hz)
    a = resample(ta, na, grid)
    b = resample(tb, nb, grid)
    n = grid.size
    max_lag = int(round(max_offset_s * rate_hz))
    max_lag = min(max_lag, n - 3)
    if max_lag < 1:
        return float("nan"), float("nan")
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.full(lags.size, -np.inf)
    min_pairs = max(10, n // 4)
    for i, lag in enumerate(lags):
        if lag >= 0:
            sa, sb = a[lag:], b[:n - lag]
        else:
            sa, sb = a[:n + lag], b[-lag:]
        ok = np.isfinite(sa) & np.isfinite(sb)
        if np.count_nonzero(ok) < min_pairs:
            continue
        xa = sa[ok] - sa[ok].mean()
        xb = sb[ok] - sb[ok].mean()
        den = math.sqrt(float(xa @ xa) * float(xb @ xb))
        if den > 0:
            corr[i] = float(xa @ xb) / den
    k = int(np.argmax(corr))
    if not np.isfinite(corr[k]):
        return float("nan"), float("nan")
    peak = corr[k]
    delta = 0.0
    if 0 < k < lags.size - 1 and np.isfinite(corr[k - 1]) and np.isfinite(corr[k + 1]):
        cm, cp = corr[k - 1], corr[k + 1]
        den = cm - 2.0 * peak + cp
        if den < 0:
            delta = float(np.clip(0.5 * (cm - cp) / den, -0.5, 0.5))
            peak = peak - 0.25 * (cm - cp) * delta
    return float((lags[k] + delta) / rate_hz), float(min(peak, 1.0))


def _kabsch(x: NDArray[np.float64], y: NDArray[np.float64], w: NDArray[np.float64]) -> NDArray[np.float64]:
    """R minimising sum w_i |y_i - R x_i|^2 (proper rotation)."""
    h = (x * w[:, None]).T @ y
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    if d == 0:
        d = 1.0
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def estimate_rotation_extrinsic(
    w_cam: ArrayLike,
    w_imu: ArrayLike,
    weights: ArrayLike | None = None,
    iters: int = 3,
    huber: float = 0.1,
) -> tuple[NDArray[np.float64], float]:
    """Rotation ``R_cam_imu`` minimising ``sum ||w_cam - R w_imu||^2``.

    Kabsch/SVD with determinant correction, followed by ``iters`` rounds of
    iteratively reweighted least squares with Huber weights
    ``min(1, huber / |residual|)`` (``huber`` in rad/s), multiplied by the
    optional per-sample ``weights``. Inputs are time-aligned (M, 3) arrays;
    rows with any NaN (or a NaN/non-positive weight) are ignored.

    Returns ``(R (3, 3), rms_residual)``; the RMS is the plain (unweighted)
    root-mean-square residual over all valid rows, so outliers do raise it.
    Raises ``ValueError`` with fewer than 2 valid rows.
    """
    wc = np.asarray(w_cam, dtype=np.float64).reshape(-1, 3)
    wi = np.asarray(w_imu, dtype=np.float64).reshape(-1, 3)
    if wc.shape != wi.shape:
        raise ValueError("w_cam and w_imu must have the same shape")
    base = np.ones(wc.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    ok = np.all(np.isfinite(wc), axis=1) & np.all(np.isfinite(wi), axis=1) & np.isfinite(base) & (base > 0)
    if np.count_nonzero(ok) < 2:
        raise ValueError("need at least 2 valid samples")
    wc, wi, base = wc[ok], wi[ok], base[ok]
    r = _kabsch(wi, wc, base)
    for _ in range(max(int(iters), 0)):
        res = np.linalg.norm(wc - wi @ r.T, axis=1)
        hub = np.where(res > huber, huber / np.maximum(res, 1e-300), 1.0)
        r = _kabsch(wi, wc, base * hub)
    res = np.linalg.norm(wc - wi @ r.T, axis=1)
    return r, float(np.sqrt(np.mean(res * res)))


def gyro_scale_check(w_cam_norm: ArrayLike, w_imu_norm: ArrayLike) -> float:
    """Ratio ``median(|w_imu|) / median(|w_cam|)`` over moving samples.

    Inputs are time-aligned norms (N,) (or (N, 3) vectors, whose norms are
    taken); the camera rate is assumed to be in rad/s. Moving samples are
    those with a finite pair and camera rate above 0.1 rad/s (all finite
    pairs if fewer than 10 qualify). About 1 means both are rad/s; about
    57.3 suggests the IMU reports deg/s. Returns NaN if nothing is usable.
    """
    c = _rate_norm(w_cam_norm)
    i = _rate_norm(w_imu_norm)
    if c.shape != i.shape:
        raise ValueError("inputs must have the same length")
    ok = np.isfinite(c) & np.isfinite(i)
    moving = ok & (c > 0.1)
    if np.count_nonzero(moving) < 10:
        moving = ok
    if not np.any(moving):
        return float("nan")
    mc = float(np.median(c[moving]))
    if mc <= 0:
        return float("nan")
    return float(np.median(i[moving]) / mc)
