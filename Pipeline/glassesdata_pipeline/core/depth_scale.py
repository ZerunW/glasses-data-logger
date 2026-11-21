"""Scale of a learned depth map from metric hands, and its robust smoothing over time (pure numpy).

Used by step ``hand3d``. A depth model's "metric" output is off by a slowly varying factor;
``depth_scale = metric / model``. Per frame it is measured at the palm joints of the metric hands:
the model sees the skin, ``skin_offset`` in front of the joint centre, so
``ratio_j = (z_joint - offset_j) / d_model(u_j, v_j)``.

:func:`smooth_scale` turns the per-frame ratios (NaN where unknown) into a smooth curve:

1. work in ``log`` (the error is multiplicative);
2. a running median over ``median_window`` frames gives a robust first estimate;
3. a Whittaker smoother (penalised second differences, cut-off period ``cutoff_frames``) with
   Huber weights (IRLS; residual scale from the MAD) fits the observations; it interpolates
   frames without observations;
4. before the first and after the last observation the curve is held constant;
5. with fewer than ``min_observations`` observations the curve is the global median; with none it
   is ``default`` (the model is then taken as metric).
"""

from __future__ import annotations

import math

import numpy as np


# =============================================================================================
# Sampling
# =============================================================================================


def to_grid(uv: np.ndarray, full_size: tuple[int, int], grid_size: tuple[int, int]) -> np.ndarray:
    """Full-resolution pixels -> pixels of a resized grid (pixel-centre convention)."""
    uv = np.asarray(uv, dtype=np.float64)
    sx = grid_size[0] / full_size[0]
    sy = grid_size[1] / full_size[1]
    return np.stack([(uv[..., 0] + 0.5) * sx - 0.5, (uv[..., 1] + 0.5) * sy - 0.5], axis=-1)


def sample_depth(depth: np.ndarray, uv_grid: np.ndarray, radius: int = 1) -> np.ndarray:
    """Median of the finite positive depth values in a ``(2 radius + 1)^2`` window around each
    rounded pixel ``(..., 2)``; NaN outside the image or without valid values."""
    d = np.asarray(depth, dtype=np.float32)
    h, w = d.shape
    uv = np.asarray(uv_grid, dtype=np.float64)
    flat = uv.reshape(-1, 2)
    out = np.full(len(flat), np.nan)
    for k, (u, v) in enumerate(flat):
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        iu, iv = int(round(u)), int(round(v))
        if not (0 <= iu < w and 0 <= iv < h):
            continue
        patch = d[max(iv - radius, 0): iv + radius + 1, max(iu - radius, 0): iu + radius + 1]
        vals = patch[np.isfinite(patch) & (patch > 0)]
        if vals.size:
            out[k] = float(np.median(vals))
    return out.reshape(uv.shape[:-1])


def frame_ratio(z_joints: np.ndarray, z_model: np.ndarray, offsets: np.ndarray,
                bounds: tuple[float, float] = (0.2, 5.0), min_samples: int = 3) -> float:
    """Robust (median) ``(z_joint - offset) / z_model`` over joints with data; NaN when fewer than
    ``min_samples`` ratios inside ``bounds``."""
    z = np.asarray(z_joints, dtype=np.float64) - np.asarray(offsets, dtype=np.float64)
    m = np.asarray(z_model, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = z / m
    r = r[np.isfinite(r) & (r >= bounds[0]) & (r <= bounds[1])]
    if r.size < min_samples:
        return math.nan
    return float(np.median(r))


# =============================================================================================
# Smoothing
# =============================================================================================


def _second_difference_bands(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagonals (main, first, second) of ``D^T D`` for the (n-2) x n second-difference matrix."""
    d0 = np.zeros(n)
    d1 = np.zeros(max(n - 1, 0))
    d2 = np.zeros(max(n - 2, 0))
    if n < 3:
        return d0, d1, d2
    stencil = np.array([1.0, -2.0, 1.0])
    for a in range(3):
        d0[a: n - 2 + a] += stencil[a] ** 2
    for a in range(2):
        d1[a: n - 2 + a] += stencil[a] * stencil[a + 1]
    d2[: n - 2] += stencil[0] * stencil[2]
    return d0, d1, d2


def solve_pentadiagonal(d0: np.ndarray, d1: np.ndarray, d2: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``A x = b`` for a symmetric positive definite pentadiagonal ``A`` given by its main,
    first and second diagonals (banded Cholesky, O(n))."""
    n = len(d0)
    D = np.zeros(n)
    E = np.zeros(n)  # L[i, i-1]
    F = np.zeros(n)  # L[i, i-2]
    for i in range(n):
        if i >= 2:
            F[i] = d2[i - 2] / D[i - 2]
        if i >= 1:
            E[i] = (d1[i - 1] - F[i] * E[i - 1]) / D[i - 1]
        D[i] = math.sqrt(max(d0[i] - E[i] ** 2 - F[i] ** 2, 1e-300))
    y = np.zeros(n)
    for i in range(n):
        acc = b[i]
        if i >= 1:
            acc -= E[i] * y[i - 1]
        if i >= 2:
            acc -= F[i] * y[i - 2]
        y[i] = acc / D[i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        acc = y[i]
        if i + 1 < n:
            acc -= E[i + 1] * x[i + 1]
        if i + 2 < n:
            acc -= F[i + 2] * x[i + 2]
        x[i] = acc / D[i]
    return x


def whittaker(y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """``argmin_x sum w (y - x)^2 + lam sum (second difference of x)^2``; ``w = 0`` where ``y`` is
    missing (those entries of ``y`` are ignored)."""
    y = np.where(np.asarray(w) > 0, np.asarray(y, dtype=np.float64), 0.0)
    w = np.asarray(w, dtype=np.float64)
    n = len(y)
    d0, d1, d2 = _second_difference_bands(n)
    ridge = 1e-10 * max(lam, 1.0)
    return solve_pentadiagonal(w + lam * d0 + ridge, lam * d1, lam * d2, w * y)


def running_median(y: np.ndarray, window: int) -> np.ndarray:
    """Median of the finite values of ``y`` within ``+-window // 2`` (NaN where none)."""
    y = np.asarray(y, dtype=np.float64)
    half = max(int(window) // 2, 0)
    out = np.full(len(y), np.nan)
    finite = np.isfinite(y)
    idx = np.flatnonzero(finite)
    if idx.size == 0:
        return out
    vals = y[idx]
    lo = np.searchsorted(idx, np.arange(len(y)) - half, side="left")
    hi = np.searchsorted(idx, np.arange(len(y)) + half, side="right")
    for i in range(len(y)):
        if hi[i] > lo[i]:
            out[i] = float(np.median(vals[lo[i]:hi[i]]))
    return out


def lambda_for_cutoff(cutoff_frames: float) -> float:
    """Whittaker ``lam`` whose response falls to one half at a period of ``cutoff_frames``
    (second-difference penalty: gain ``1 / (1 + lam (2 sin(pi f))^4)``)."""
    f = 1.0 / max(cutoff_frames, 2.0)
    return 1.0 / (2.0 * math.sin(math.pi * f)) ** 4


def smooth_scale(
    ratio: np.ndarray,
    weights: np.ndarray | None = None,
    median_window: int = 13,
    cutoff_frames: float = 48.0,
    huber_k: float = 1.345,
    min_sigma: float = 0.003,
    iterations: int = 10,
    min_observations: int = 5,
    default: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Smooth per-frame scale ratios (NaN = unknown). Returns ``(smooth[T], info)``."""
    r = np.asarray(ratio, dtype=np.float64)
    T = len(r)
    obs = np.isfinite(r) & (r > 0)
    base = np.ones(T) if weights is None else np.where(np.isfinite(weights), np.asarray(weights, dtype=np.float64), 0.0)
    base = np.where(obs, np.clip(base, 0.0, None), 0.0)
    obs &= base > 0
    n_obs = int(obs.sum())
    info: dict = {"observations": n_obs}
    if n_obs == 0:
        info["method"] = "default"
        return np.full(T, float(default)), info
    y = np.full(T, np.nan)
    y[obs] = np.log(r[obs])
    if n_obs < min_observations:
        info["method"] = "global_median"
        return np.full(T, float(np.exp(np.median(y[obs])))), info
    first, last = int(np.flatnonzero(obs)[0]), int(np.flatnonzero(obs)[-1])
    seg = slice(first, last + 1)
    ys, ws, os_ = y[seg], base[seg], obs[seg]
    lam = lambda_for_cutoff(cutoff_frames)
    med = running_median(ys, median_window)
    # initial curve: the running median, interpolated over gaps
    have = np.isfinite(med)
    xi = np.arange(len(ys))
    x = np.interp(xi, xi[have], med[have])
    sigma = math.nan
    for _ in range(max(1, iterations)):
        res = np.where(os_, ys - x, 0.0)
        mad = float(np.median(np.abs(res[os_] - np.median(res[os_]))))
        sigma = max(1.4826 * mad, min_sigma)
        u = np.abs(res) / (huber_k * sigma)
        hub = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
        x_new = whittaker(ys, ws * hub * os_, lam)
        if np.max(np.abs(x_new - x)) < 1e-7:
            x = x_new
            break
        x = x_new
    out = np.empty(T)
    out[seg] = x
    out[:first] = x[0]
    out[last + 1:] = x[-1]
    info.update({"method": "whittaker_huber", "lambda": lam, "sigma_log": sigma,
                 "first": first, "last": last})
    return np.exp(out), info
