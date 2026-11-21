"""Offline smoothing and geometric refinement of hand-joint trajectories.

Only NumPy is required.  The temporal smoother works independently on every
coordinate; bone refinement couples joints within frames and regularizes time.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _ca_transition(dt: float, q_accel: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Exact discrete transition and process covariance for integrated white jerk."""
    d2 = dt * dt
    d3 = d2 * dt
    d4 = d3 * dt
    d5 = d4 * dt
    phi = np.array([[1.0, dt, d2 / 2.0],
                    [0.0, 1.0, dt],
                    [0.0, 0.0, 1.0]], dtype=np.float64)
    q = q_accel * np.array([[d5 / 20.0, d4 / 8.0, d3 / 6.0],
                            [d4 / 8.0, d3 / 3.0, d2 / 2.0],
                            [d3 / 6.0, d2 / 2.0, dt]], dtype=np.float64)
    return phi, q


def rts_smooth(
    z: ArrayLike,
    t: ArrayLike,
    q_accel: float = 50.0,
    r_meas: ArrayLike = 1e-4,
    gate_chi2: float = 16.0,
    max_gap_s: float = 0.5,
    init_var: tuple[float, float, float] = (1e-2, 1.0, 10.0),
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Kalman filter and Rauch--Tung--Striebel smoother for independent CA states.

    Parameters
    ----------
    z : array_like, shape (T, D)
        Positions, with NaNs designating missing measurements.
    t : array_like, shape (T,)
        Strictly increasing time stamps in seconds; intervals need not be equal.
    q_accel : float
        Nonnegative continuous-time white-jerk spectral density, units^2/s^5.
        The state is ``[position, velocity, acceleration]``.
    r_meas : float or array_like, shape (T,) or (T, D)
        Positive measurement variance(s); a nonfinite or nonpositive entry
        makes that measurement unavailable.
    gate_chi2 : float
        Maximum normalized squared innovation ``nu**2 / S``; larger values
        are rejected as outliers. Must be nonnegative. The gate is applied by
        a forward and a time-reversed filter and a sample is an inlier only
        if both accept it (see Notes).
    max_gap_s : float
        Maximum time *between surrounding accepted measurements* over which
        rejected/missing positions are interpolated. Longer interiors are NaN.
    init_var : tuple[float, float, float]
        Strictly positive initial position, velocity, and acceleration variances.
        Each dimension initializes at its first accepted observation
        (p = z, v = a = 0, covariance diag(init_var)); that observation is
        not used again as a measurement update.

    Returns
    -------
    x, var, inlier : ndarray, ndarray, ndarray
        Smoothed positions and their marginal variances, both shape (T, D),
        and a boolean acceptance mask. Smoothed positions/variances are NaN
        outside the first/last inliers and in intervals longer than
        ``max_gap_s``. Observed inliers remain defined.

    Notes
    -----
    The process noise is integrated continuous-time white jerk. Its exact
    covariance has entries ``q * [[dt^5/20, dt^4/8, dt^3/6],
    [dt^4/8, dt^3/3, dt^2/2], [dt^3/6, dt^2/2, dt]]``. Filtering and RTS
    updates are batched over D; only time is traversed in Python.

    Outlier gating runs the gated filter twice, forward and on the
    time-reversed sequence, and keeps a measurement only if both directions
    accept it. A forward-only gate keeps a spike that is the first sample of a
    track (it initialises on it and rejects the next true sample instead) and
    a spike right after a gap (where the predicted variance is large). A
    final ungated filter + RTS pass over the accepted samples gives x and var.
    Cost is two or three filter passes plus one RTS pass.
    """
    obs = np.asarray(z, dtype=np.float64)
    times = np.asarray(t, dtype=np.float64)
    if obs.ndim != 2:
        raise ValueError("z must have shape (T, D)")
    n, dims = obs.shape
    if times.shape != (n,) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("t must be finite, shape (T,), and strictly increasing")
    if not np.isfinite(q_accel) or q_accel < 0:
        raise ValueError("q_accel must be nonnegative and finite")
    if not np.isfinite(gate_chi2) or gate_chi2 < 0:
        raise ValueError("gate_chi2 must be nonnegative and finite")
    if not np.isfinite(max_gap_s) or max_gap_s < 0:
        raise ValueError("max_gap_s must be nonnegative and finite")
    iv = np.asarray(init_var, dtype=np.float64)
    if iv.shape != (3,) or not np.all(np.isfinite(iv)) or np.any(iv <= 0):
        raise ValueError("init_var must contain three strictly positive, finite entries")
    noise = np.asarray(r_meas, dtype=np.float64)
    if noise.ndim == 0:
        noise = np.broadcast_to(noise, (n, dims))
    elif noise.shape == (n,):
        noise = np.broadcast_to(noise[:, None], (n, dims))
    elif noise.shape != (n, dims):
        raise ValueError("r_meas must be scalar, (T,), or (T, D)")

    xout = np.full((n, dims), np.nan, dtype=np.float64)
    vout = np.full((n, dims), np.nan, dtype=np.float64)
    if n == 0 or dims == 0:
        return xout, vout, np.zeros((n, dims), dtype=np.bool_)

    # Bidirectional gating. A forward-only gate cannot judge the sample it
    # initialises on (a spike there is kept and the next true sample is
    # rejected instead), and after a gap the predicted variance is so large
    # that a spike right after the gap passes. Running the same gated filter
    # on the time-reversed sequence sees those samples from the other side;
    # a measurement is an inlier only if both directions accept it. The final
    # filter/RTS pass then uses exactly those inliers, ungated.
    fwd = _ca_forward(obs, noise, times, q_accel, gate_chi2, iv)
    rev_times = times[-1] - times[::-1]
    bwd = _ca_forward(obs[::-1], noise[::-1], rev_times, q_accel, gate_chi2, iv)
    inlier = fwd[-1] & bwd[-1][::-1]
    if np.array_equal(inlier, fwd[-1]):
        filtered, f_cov, predicted, p_cov, active, _ = fwd
    else:
        masked = np.where(inlier, obs, np.nan)
        filtered, f_cov, predicted, p_cov, active, _ = _ca_forward(
            masked, noise, times, q_accel, np.inf, iv)

    smooth = filtered.copy()
    smooth_cov = f_cov.copy()
    identity = np.eye(3, dtype=np.float64)
    for k in range(n - 2, -1, -1):
        phi, _ = _ca_transition(float(times[k + 1] - times[k]), q_accel)
        cross = f_cov[k] @ phi.T
        valid = active[k]
        # An uninitialized channel has singular (zero) predicted covariance.
        # Give it an identity system and zero gain; it remains masked below.
        lhs = np.where(valid[:, None, None], p_cov[k + 1], identity)
        rhs = np.where(valid[:, None, None], np.swapaxes(cross, -1, -2), 0.0)
        gain = np.swapaxes(np.linalg.solve(lhs, rhs), -1, -2)
        ds = smooth[k + 1] - predicted[k + 1]
        smooth[k] += (gain @ ds[..., None])[..., 0]
        d_cov = smooth_cov[k + 1] - p_cov[k + 1]
        smooth_cov[k] += gain @ d_cov @ np.swapaxes(gain, -1, -2)
        smooth_cov[k] = (smooth_cov[k] + np.swapaxes(smooth_cov[k], -1, -2)) * 0.5

    xout[:] = smooth[..., 0]
    vout[:] = np.maximum(smooth_cov[..., 0, 0], 0.0)
    # Every coordinate uses its own nearest accepted observations. The scan is
    # vectorized across both time and D, with no per-column Python loop.
    sample = np.arange(n, dtype=np.int64)[:, None]
    left = np.maximum.accumulate(np.where(inlier, sample, -1), axis=0)
    right = np.minimum.accumulate(
        np.where(inlier, sample, n)[::-1], axis=0
    )[::-1]
    left_safe = np.clip(left, 0, n - 1)
    right_safe = np.clip(right, 0, n - 1)
    keep = ((left >= 0) & (right < n)
            & ((times[right_safe] - times[left_safe]) <= max_gap_s))
    xout[~keep] = np.nan
    vout[~keep] = np.nan
    return xout, vout, inlier


def _ca_forward(
    obs: NDArray[np.float64],
    noise: NDArray[np.float64],
    times: NDArray[np.float64],
    q_accel: float,
    gate_chi2: float,
    iv: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
           NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Gated forward Kalman filter for D independent CA channels.

    Returns (filtered (T,D,3), filtered cov (T,D,3,3), predicted (T,D,3),
    predicted cov (T,D,3,3), active (T,D) bool = channel initialised,
    inlier (T,D) bool). Each channel starts at its first usable sample
    (p = z, v = a = 0, cov = diag(iv)), which is accepted without gating.
    ``gate_chi2 = inf`` disables gating.
    """
    n, dims = obs.shape
    inlier = np.zeros((n, dims), dtype=np.bool_)
    filtered = np.zeros((n, dims, 3), dtype=np.float64)
    f_cov = np.zeros((n, dims, 3, 3), dtype=np.float64)
    predicted = np.zeros_like(filtered)
    p_cov = np.zeros_like(f_cov)
    active = np.zeros((n, dims), dtype=np.bool_)
    state = np.zeros((dims, 3), dtype=np.float64)
    cov = np.zeros((dims, 3, 3), dtype=np.float64)
    initial_cov = np.diag(iv)
    identity = np.eye(3, dtype=np.float64)
    initialized = np.zeros(dims, dtype=np.bool_)

    for k in range(n):
        if k:
            phi, q = _ca_transition(float(times[k] - times[k - 1]), q_accel)
            state_prior = state @ phi.T
            cov_prior = phi @ cov @ phi.T + q[None, :, :] * initialized[:, None, None]
        else:
            state_prior = state.copy()
            cov_prior = cov.copy()

        predicted[k] = state_prior
        p_cov[k] = cov_prior
        usable = np.isfinite(obs[k]) & np.isfinite(noise[k]) & (noise[k] > 0)
        first = usable & ~initialized
        if np.any(first):
            state_prior[first] = 0.0
            state_prior[first, 0] = obs[k, first]
            cov_prior[first] = initial_cov
            inlier[k, first] = True

        candidates = usable & initialized
        safe_noise = np.where(np.isfinite(noise[k]) & (noise[k] > 0), noise[k], 1.0)
        innovation = np.where(candidates, obs[k] - state_prior[:, 0], 0.0)
        s = cov_prior[:, 0, 0] + safe_noise
        accepted = candidates & (innovation * innovation <= gate_chi2 * s)
        inlier[k] |= accepted
        gain = cov_prior[:, :, 0] / s[:, None]
        state = state_prior + gain * np.where(accepted, innovation, 0.0)[:, None]
        # A Joseph-form covariance update avoids loss of positive definiteness.
        kh = np.zeros((dims, 3, 3), dtype=np.float64)
        kh[:, :, 0] = gain * accepted[:, None]
        m = identity - kh
        cov = m @ cov_prior @ np.swapaxes(m, -1, -2)
        cov += ((safe_noise * accepted)[:, None, None]
                * gain[:, :, None] * gain[:, None, :])
        cov = (cov + np.swapaxes(cov, -1, -2)) * 0.5
        initialized |= first
        filtered[k] = state
        f_cov[k] = cov
        active[k] = initialized

    return filtered, f_cov, predicted, p_cov, active, inlier


def _fill_missing_joints(
    joints: NDArray[np.float64], parents: NDArray[np.int64],
    lengths: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate joint coordinates in time before geometric optimization."""
    n, nj, _ = joints.shape
    result = np.empty_like(joints)
    time = np.arange(n)
    for j in range(nj):
        valid = np.flatnonzero(np.all(np.isfinite(joints[:, j]), axis=-1))
        if valid.size:
            hi_index = np.searchsorted(valid, time)
            lo = valid[np.clip(hi_index - 1, 0, valid.size - 1)]
            hi = valid[np.clip(hi_index, 0, valid.size - 1)]
            alpha = (time - lo) / np.maximum(hi - lo, 1)
            result[:, j] = ((1.0 - alpha[:, None]) * joints[lo, j]
                            + alpha[:, None] * joints[hi, j])
        elif j == 0:
            result[:, j] = 0.0
        else:
            result[:, j] = result[:, parents[j]]
            result[:, j, 0] += lengths[j - 1]
    return result


def _pentadiagonal_cholesky(n: int, rho: float, w_smooth: float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Factor ``rho * I + 2*w_smooth * D2.T @ D2`` into two-band L."""
    main = np.full(n, rho, dtype=np.float64)
    low1 = np.zeros(n, dtype=np.float64)
    low2 = np.zeros(n, dtype=np.float64)
    if n >= 3 and w_smooth:
        # Accumulate the three overlapping coefficients of each [1,-2,1]
        # second-difference row.  This also handles n == 3 and 4 exactly.
        weights = 2.0 * w_smooth
        main[:-2] += weights
        main[1:-1] += 4.0 * weights
        main[2:] += weights
        low1[1:] = -2.0 * weights
        if n > 3:
            low1[2:-1] = -4.0 * weights
        low2[2:] = weights
    diagonal = np.empty(n, dtype=np.float64)
    for k in range(n):
        if k >= 2:
            low2[k] /= diagonal[k - 2]
        if k >= 1:
            low1[k] = (low1[k] - (low2[k] * low1[k - 1] if k >= 2 else 0.0)) / diagonal[k - 1]
        diagonal[k] = np.sqrt(main[k] - low1[k] ** 2 - low2[k] ** 2)
    return diagonal, low1, low2


def _pentadiagonal_solve(
    rhs: NDArray[np.float64],
    chol: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Solve a fixed SPD pentadiagonal system for all trailing RHS at once."""
    d, l1, l2 = chol
    n = rhs.shape[0]
    y = np.empty_like(rhs)
    for k in range(n):
        value = rhs[k].copy()
        if k >= 1:
            value -= l1[k] * y[k - 1]
        if k >= 2:
            value -= l2[k] * y[k - 2]
        y[k] = value / d[k]
    x = np.empty_like(rhs)
    for k in range(n - 1, -1, -1):
        value = y[k].copy()
        if k + 1 < n:
            value -= l1[k + 1] * x[k + 1]
        if k + 2 < n:
            value -= l2[k + 2] * x[k + 2]
        x[k] = value / d[k]
    return x


def _bone_gn_step(
    positions: NDArray[np.float64],
    data: NDArray[np.float64],
    observed: NDArray[np.bool_],
    parents: NDArray[np.int64],
    lengths: NDArray[np.float64],
    w_data: float,
    w_bone: float,
    prox_target: NDArray[np.float64],
    prox_weight: float,
) -> NDArray[np.float64]:
    """One batched per-frame Gauss--Newton step, using tree block elimination.

    Every Hessian block is 3x3. Eliminating leaves to root is O(T*J) rather
    than the O(T*J**3) work of factoring 63x63 dense matrices per frame.
    """
    n, nj, _ = positions.shape
    eye = np.eye(3, dtype=np.float64)
    diag = np.zeros((n, nj, 3, 3), dtype=np.float64)
    diag[:] = (w_data * observed + prox_weight + 1e-9)[..., None, None] * eye
    gradient = w_data * observed[..., None] * (positions - data)
    gradient += prox_weight * (positions - prox_target)
    offdiag = np.zeros((n, nj, 3, 3), dtype=np.float64)

    for j in range(1, nj):
        p = parents[j]
        direction = positions[:, j] - positions[:, p]
        distance = np.linalg.norm(direction, axis=-1)
        unit = direction / np.maximum(distance[:, None], 1e-12)
        zeros = distance < 1e-12
        if np.any(zeros):
            unit[zeros] = (1.0, 0.0, 0.0)
        residual = distance - lengths[j - 1]
        force = (w_bone * residual)[:, None] * unit
        gradient[:, j] += force
        gradient[:, p] -= force
        hess = w_bone * unit[:, :, None] * unit[:, None, :]
        diag[:, j] += hess
        diag[:, p] += hess
        offdiag[:, j] = -hess

    rhs = -gradient
    offsets = np.zeros((n, nj, 3), dtype=np.float64)
    transfers = np.zeros((n, nj, 3, 3), dtype=np.float64)
    for j in range(nj - 1, 0, -1):
        p = parents[j]
        coupled = np.concatenate((offdiag[:, j], rhs[:, j, :, None]), axis=2)
        solved = np.linalg.solve(diag[:, j], coupled)
        transfer = solved[:, :, :3]
        offset = solved[:, :, 3]
        transfers[:, j] = transfer
        offsets[:, j] = offset
        diag[:, p] -= np.swapaxes(offdiag[:, j], 1, 2) @ transfer
        rhs[:, p] -= (np.swapaxes(offdiag[:, j], 1, 2) @ offset[..., None])[..., 0]

    delta = np.empty_like(positions)
    delta[:, 0] = np.linalg.solve(diag[:, 0], rhs[:, 0, :, None])[..., 0]
    for j in range(1, nj):
        delta[:, j] = offsets[:, j] - (
            transfers[:, j] @ delta[:, parents[j], :, None]
        )[..., 0]
    return positions + delta


def refine_bone_lengths(
    joints: ArrayLike,
    parents: ArrayLike,
    target_len: ArrayLike,
    observed: ArrayLike | None = None,
    w_data: float = 1.0,
    w_bone: float = 100.0,
    w_smooth: float = 0.1,
    iters: int = 10,
) -> NDArray[np.float64]:
    """Refine joint coordinates using bone lengths and temporal regularity.

    Minimizes approximately

    ``w_data * sum_observed ||X - joints||^2``
    ``+ w_bone * sum_bones (||X_child-X_parent|| - L)^2``
    ``+ w_smooth * sum_t ||X[t+1]-2*X[t]+X[t-1]||^2``.

    ``joints`` is (T, J, 3), ``parents`` is (J,) with ``parents[0]=-1``
    and ``0 <= parents[j] < j``, and ``target_len`` is (J-1,).
    ``observed`` is a (T, J) boolean data-term mask, defaulting to finite
    first coordinates; incomplete 3D vectors also have no data term.
    Entirely NaN input frames remain entirely NaN in the returned array.

    Algorithm
    ---------
    This is scaled ADMM with a Gauss--Newton X-substep and an exact temporal
    Z-substep. Each X-substep performs a *batched* tree-structured 3x3-block
    elimination, O(T*J); it does not form or invert a dense (3J)x(3J) Hessian.
    The Z-substep solves ``(rho I + 2*w_smooth*D2.T@D2)Z = rho*(X+U)``
    with a custom pentadiagonal Cholesky factorization and time-axis forward/
    backward substitutions, simultaneously for all 3J coordinates. This is
    O(T*J) work and O(T*J) storage, and supports T=2000 without dense solves.
    The returned X is the final bone-refined (primal) trajectory.
    """
    data = np.asarray(joints, dtype=np.float64)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError("joints must have shape (T, J, 3)")
    n, nj, _ = data.shape
    if nj == 0:
        raise ValueError("joints must contain at least the root joint")
    parent = np.asarray(parents)
    if parent.shape != (nj,) or not np.issubdtype(parent.dtype, np.integer):
        raise ValueError("parents must be an integer array of shape (J,)")
    parent = parent.astype(np.int64, copy=False)
    if parent[0] != -1 or np.any(parent[1:] < 0) or np.any(parent[1:] >= np.arange(1, nj)):
        raise ValueError("parents must define a tree in topological order")
    lengths = np.asarray(target_len, dtype=np.float64)
    if lengths.shape != (nj - 1,) or np.any(lengths < 0) or not np.all(np.isfinite(lengths)):
        raise ValueError("target_len must contain J-1 finite nonnegative lengths")
    if observed is None:
        mask = ~np.isnan(data[..., 0])
    else:
        mask = np.asarray(observed, dtype=np.bool_)
        if mask.shape != (n, nj):
            raise ValueError("observed must have shape (T, J)")
    mask = mask & np.all(np.isfinite(data), axis=-1)
    for name, weight in (("w_data", w_data), ("w_bone", w_bone), ("w_smooth", w_smooth)):
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not isinstance(iters, (int, np.integer)) or iters < 0:
        raise ValueError("iters must be a nonnegative integer")
    if n == 0 or iters == 0:
        return data.copy()
    all_missing = np.all(np.isnan(data), axis=(1, 2))
    if np.all(all_missing):
        return data.copy()

    # The proximal coefficient rho/2 stabilizes otherwise unanchored frames.
    rho = 2.0 * max(float(w_data), 1.0)
    filled = _fill_missing_joints(data, parent, lengths)
    target = np.where(mask[..., None], data, 0.0)
    x = filled.copy()
    z = filled.copy()
    dual = np.zeros_like(filled)
    chol = _pentadiagonal_cholesky(n, rho, float(w_smooth))
    for _ in range(iters):
        x = _bone_gn_step(x, target, mask, parent, lengths,
                          float(w_data), float(w_bone), z - dual, rho / 2.0)
        if w_smooth:
            z = _pentadiagonal_solve((rho * (x + dual)).reshape(n, -1), chol).reshape(n, nj, 3)
        else:
            z = x + dual
        dual += x - z

    x[all_missing] = np.nan
    return x
