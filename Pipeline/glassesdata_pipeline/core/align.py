"""Trajectory alignment and evaluation metrics.

Umeyama similarity alignment, absolute trajectory error (ATE), relative pose
error (RPE), (Procrustes-aligned) mean per-joint position error, and TUM
trajectory file I/O.

Poses are 4x4 arrays ``T_world_cam`` (camera-to-world); positions are the
translation column. Only NumPy is used; functions are pure except the two
TUM file helpers.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

_COLLINEAR_TOL = 1e-10  # relative second singular value below which points are collinear


# ----------------------------------------------------------------------------
# small SO(3) / quaternion helpers (private; quaternions [w, x, y, z])
# ----------------------------------------------------------------------------


def _quat_to_mat(q: NDArray[np.float64]) -> NDArray[np.float64]:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.empty(q.shape[:-1] + (3, 3))
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


def _mat_to_quat(m: NDArray[np.float64]) -> NDArray[np.float64]:
    """Shepperd's method, (..., 3, 3) -> (..., 4) with w >= 0."""
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    big = np.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                    1 - m00 + m11 - m22, 1 - m00 - m11 + m22], axis=-1)
    d21 = m[..., 2, 1] - m[..., 1, 2]
    d02 = m[..., 0, 2] - m[..., 2, 0]
    d10 = m[..., 1, 0] - m[..., 0, 1]
    s01 = m[..., 0, 1] + m[..., 1, 0]
    s02 = m[..., 0, 2] + m[..., 2, 0]
    s12 = m[..., 1, 2] + m[..., 2, 1]
    cand = np.stack([
        np.stack([big[..., 0], d21, d02, d10], axis=-1),
        np.stack([d21, big[..., 1], s01, s02], axis=-1),
        np.stack([d02, s01, big[..., 2], s12], axis=-1),
        np.stack([d10, s02, s12, big[..., 3]], axis=-1),
    ], axis=-2)
    piv = np.argmax(big, axis=-1)
    q = np.take_along_axis(cand, piv[..., None, None], axis=-2)[..., 0, :]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., :1] < 0, -q, q)


def _invert_rigid(poses: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of rigid 4x4 transforms (..., 4, 4) in closed form."""
    r = poses[..., :3, :3]
    t = poses[..., :3, 3]
    out = np.zeros_like(poses)
    rt = np.swapaxes(r, -1, -2)
    out[..., :3, :3] = rt
    out[..., :3, 3] = -(rt @ t[..., None])[..., 0]
    out[..., 3, 3] = 1.0
    return out


def rotation_angle_deg(Ra: ArrayLike, Rb: ArrayLike) -> NDArray[np.float64] | float:
    """Geodesic angle in degrees between rotations ``Ra`` and ``Rb`` (..., 3, 3).

    Computed as ``atan2(|vee(M - M^T)| / 2, (tr M - 1) / 2)`` with
    ``M = Ra^T Rb`` (both arguments clipped to their valid ranges), which is
    accurate for tiny angles as well as near 180 deg, unlike a bare arccos.
    Returns a float for single matrices, an array otherwise.
    """
    a = np.asarray(Ra, dtype=np.float64)
    b = np.asarray(Rb, dtype=np.float64)
    m = np.swapaxes(a, -1, -2) @ b
    v = np.stack([m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0],
                  m[..., 1, 0] - m[..., 0, 1]], axis=-1)
    s = np.clip(0.5 * np.linalg.norm(v, axis=-1), 0.0, 1.0)
    c = np.clip(0.5 * (np.trace(m, axis1=-2, axis2=-1) - 1.0), -1.0, 1.0)
    ang = np.degrees(np.arctan2(s, c))
    return float(ang) if np.ndim(ang) == 0 else ang


# ----------------------------------------------------------------------------
# Umeyama
# ----------------------------------------------------------------------------


def _umeyama_batch(
    src: NDArray[np.float64],
    dst: NDArray[np.float64],
    w: NDArray[np.float64],
    with_scale: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Batched weighted Umeyama on (B, N, 3) with weights (B, N) >= 0.

    Rows with zero weight do not contribute (their coordinates may be
    anything finite). Returns (s (B,), R (B,3,3), t (B,3), ok (B,)) where
    ``ok`` is False for batches with < 3 positively weighted points or
    collinear (or coincident) source/destination points.
    """
    wsum = w.sum(axis=1)
    count = np.count_nonzero(w > 0, axis=1)
    wn = w / np.where(wsum > 0, wsum, 1.0)[:, None]
    mu_s = np.einsum("bn,bnk->bk", wn, src)
    mu_d = np.einsum("bn,bnk->bk", wn, dst)
    xs = src - mu_s[:, None, :]
    xd = dst - mu_d[:, None, :]
    sq = np.sqrt(wn)[..., None]
    sv_s = np.linalg.svd(sq * xs, compute_uv=False)
    sv_d = np.linalg.svd(sq * xd, compute_uv=False)
    ok = ((count >= 3)
          & (sv_s[:, 1] > _COLLINEAR_TOL * np.maximum(sv_s[:, 0], 1e-300))
          & (sv_d[:, 1] > _COLLINEAR_TOL * np.maximum(sv_d[:, 0], 1e-300)))
    var_s = np.einsum("bn,bnk,bnk->b", wn, xs, xs)
    sigma = np.einsum("bn,bni,bnj->bij", wn, xd, xs)       # dst-src cross-covariance
    u, d, vt = np.linalg.svd(sigma)
    sign = np.sign(np.linalg.det(u) * np.linalg.det(vt))
    sign = np.where(sign == 0, 1.0, sign)
    corr = np.ones((sigma.shape[0], 3))
    corr[:, 2] = sign
    r = u @ (corr[:, :, None] * vt)
    if with_scale:
        s = np.sum(d * corr, axis=1) / np.where(var_s > 0, var_s, 1.0)
    else:
        s = np.ones(sigma.shape[0])
    t = mu_d - s[:, None] * np.einsum("bij,bj->bi", r, mu_s)
    return s, r, t, ok


def umeyama(
    src: ArrayLike,
    dst: ArrayLike,
    with_scale: bool = True,
    weights: ArrayLike | None = None,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Least-squares similarity ``dst ~ s * R @ src + t`` (Umeyama 1991).

    ``src``, ``dst``: (N, 3). Rows with any NaN in either array (or a
    non-finite / non-positive weight) are ignored. ``weights`` (N,) weight
    the squared residuals. The reflection correction guarantees
    ``det(R) = +1``. With ``with_scale=False``, ``s`` is exactly 1.

    Returns ``(s, R (3, 3), t (3,))``. Raises ``ValueError`` if fewer than 3
    valid points remain or the valid source or destination points are
    collinear (the rotation about that line is then undetermined).
    """
    a = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if a.shape != b.shape:
        raise ValueError("src and dst must have the same shape")
    w = np.ones(a.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if w.shape[0] != a.shape[0]:
        raise ValueError("weights must have one entry per point")
    valid = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1) & np.isfinite(w) & (w > 0)
    if np.count_nonzero(valid) < 3:
        raise ValueError("umeyama needs at least 3 valid points")
    s, r, t, ok = _umeyama_batch(a[valid][None], b[valid][None], w[valid][None], with_scale)
    if not ok[0]:
        raise ValueError("umeyama points are collinear or coincident")
    return float(s[0]), r[0], t[0]


def apply_sim3(s: float, R: ArrayLike, t: ArrayLike, poses: ArrayLike) -> NDArray[np.float64]:
    """Apply a similarity to camera-to-world poses (T, 4, 4) (or a single 4x4).

    Positions become ``s R p + t`` and rotations ``R Rc``; the result keeps a
    proper rigid 4x4 form (bottom row ``[0, 0, 0, 1]``). NaN poses stay NaN.
    """
    p = np.asarray(poses, dtype=np.float64)
    r = np.asarray(R, dtype=np.float64)
    tt = np.asarray(t, dtype=np.float64).reshape(3)
    out = np.zeros_like(p)
    out[..., :3, :3] = r @ p[..., :3, :3]
    out[..., :3, 3] = float(s) * (p[..., :3, 3] @ r.T) + tt
    out[..., 3, :3] = 0.0
    out[..., 3, 3] = 1.0
    bad = ~np.all(np.isfinite(p), axis=(-2, -1))
    if np.any(bad):
        out[bad] = np.nan
    return out


def _valid_poses(est: NDArray[np.float64], gt: NDArray[np.float64],
                 valid: ArrayLike | None) -> NDArray[np.bool_]:
    m = np.all(np.isfinite(est), axis=(1, 2)) & np.all(np.isfinite(gt), axis=(1, 2))
    if valid is not None:
        v = np.asarray(valid, dtype=bool).ravel()
        if v.shape[0] != m.shape[0]:
            raise ValueError("valid must have one entry per pose")
        m &= v
    return m


def ate(est: ArrayLike, gt: ArrayLike, align: str = "sim3", valid: ArrayLike | None = None) -> dict[str, Any]:
    """Absolute trajectory error of camera positions.

    ``est``, ``gt``: (T, 4, 4) corresponding poses. ``align``: ``'sim3'``
    (Umeyama with scale), ``'se3'`` (rigid) or ``'none'``; the alignment is
    estimated from the camera positions of the valid poses (``valid`` mask,
    and both poses finite).

    Returns a dict with ``rmse``, ``mean``, ``median``, ``max`` (metres, over
    valid poses), ``scale`` (1.0 unless sim3), ``n`` (number of valid poses),
    ``R``, ``t`` (the alignment) and ``aligned`` (T, 4, 4): every estimated
    pose mapped by the alignment. Metrics are NaN when ``n == 0`` with
    ``align='none'``; sim3/se3 raise ``ValueError`` (via ``umeyama``) with
    fewer than 3 valid, non-collinear positions.
    """
    e = np.asarray(est, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if e.shape != g.shape or e.ndim != 3 or e.shape[1:] != (4, 4):
        raise ValueError("est and gt must both be (T, 4, 4)")
    m = _valid_poses(e, g, valid)
    pe, pg = e[m, :3, 3], g[m, :3, 3]
    if align == "sim3":
        s, r, t = umeyama(pe, pg, with_scale=True)
    elif align == "se3":
        s, r, t = umeyama(pe, pg, with_scale=False)
    elif align == "none":
        s, r, t = 1.0, np.eye(3), np.zeros(3)
    else:
        raise ValueError("align must be 'sim3', 'se3' or 'none'")
    aligned = apply_sim3(s, r, t, e)
    err = np.linalg.norm(aligned[m, :3, 3] - pg, axis=1)
    if err.size:
        stats = dict(rmse=float(np.sqrt(np.mean(err * err))), mean=float(err.mean()),
                     median=float(np.median(err)), max=float(err.max()))
    else:
        stats = dict(rmse=float("nan"), mean=float("nan"), median=float("nan"), max=float("nan"))
    return dict(stats, scale=float(s), n=int(m.sum()), R=r, t=t, aligned=aligned)


def _scale_estimate(pe: NDArray[np.float64], pg: NDArray[np.float64]) -> float:
    """Sim3 scale gt/est; falls back to the ratio of position spreads when the
    positions are collinear (rotation undetermined, scale still defined)."""
    try:
        return umeyama(pe, pg, with_scale=True)[0]
    except ValueError:
        if pe.shape[0] < 2:
            return 1.0
        vs = np.sum((pe - pe.mean(0)) ** 2)
        vg = np.sum((pg - pg.mean(0)) ** 2)
        return float(np.sqrt(vg / vs)) if vs > 0 else 1.0


def rpe(est: ArrayLike, gt: ArrayLike, delta: int = 1, valid: ArrayLike | None = None,
        align_scale: bool = True) -> dict[str, Any]:
    """Relative pose error over index pairs ``(i, i + delta)`` that are both valid.

    For each pair, ``Q = inv(gt_i) gt_j`` and ``P = inv(est_i) est_j``. The
    translation error is ``|s * t(P) - t(Q)|`` where ``s`` is the sim3 scale
    from aligning est to gt positions (as in ``ate``) when ``align_scale`` is
    True, else 1; the rotation error is the angle between ``R(P)`` and
    ``R(Q)`` in degrees. (Rotations of the poses are assumed orthonormal.)

    Returns a dict ``trans_rmse``, ``trans_mean`` (metres), ``rot_rmse_deg``,
    ``rot_mean_deg``, ``n`` (number of pairs) and ``scale``. Metrics are NaN
    when there is no pair.
    """
    e = np.asarray(est, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if e.shape != g.shape or e.ndim != 3 or e.shape[1:] != (4, 4):
        raise ValueError("est and gt must both be (T, 4, 4)")
    if int(delta) < 1:
        raise ValueError("delta must be >= 1")
    delta = int(delta)
    m = _valid_poses(e, g, valid)
    s = _scale_estimate(e[m, :3, 3], g[m, :3, 3]) if align_scale else 1.0
    i = np.flatnonzero(m[:-delta] & m[delta:]) if e.shape[0] > delta else np.zeros(0, int)
    j = i + delta
    nan = float("nan")
    if i.size == 0:
        return dict(trans_rmse=nan, trans_mean=nan, rot_rmse_deg=nan, rot_mean_deg=nan, n=0, scale=s)
    q = _invert_rigid(g[i]) @ g[j]
    p = _invert_rigid(e[i]) @ e[j]
    terr = np.linalg.norm(s * p[:, :3, 3] - q[:, :3, 3], axis=1)
    rerr = np.atleast_1d(rotation_angle_deg(q[:, :3, :3], p[:, :3, :3]))
    return dict(trans_rmse=float(np.sqrt(np.mean(terr ** 2))), trans_mean=float(terr.mean()),
                rot_rmse_deg=float(np.sqrt(np.mean(rerr ** 2))), rot_mean_deg=float(rerr.mean()),
                n=int(i.size), scale=float(s))


# ----------------------------------------------------------------------------
# joint errors
# ----------------------------------------------------------------------------


def mpjpe(pred: ArrayLike, gt: ArrayLike, valid: ArrayLike | None = None) -> float:
    """Mean per-joint position error over (..., J, 3) arrays, in the input units.

    Joints with any NaN in ``pred`` or ``gt`` (or ``valid`` False, shape
    (..., J) or broadcastable) are excluded. Returns NaN if none is valid.
    """
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if p.shape != g.shape or p.shape[-1] != 3:
        raise ValueError("pred and gt must have the same (..., J, 3) shape")
    m = np.all(np.isfinite(p), axis=-1) & np.all(np.isfinite(g), axis=-1)
    if valid is not None:
        m &= np.broadcast_to(np.asarray(valid, dtype=bool), m.shape)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.linalg.norm(p[m] - g[m], axis=-1)))


def pa_mpjpe(pred: ArrayLike, gt: ArrayLike) -> float:
    """Procrustes-aligned MPJPE for (N, J, 3) arrays.

    Each sample's prediction is aligned to its ground truth with a similarity
    (Umeyama with scale, reflection-corrected) over the joints valid in both;
    samples with fewer than 3 valid joints (or collinear ones) are skipped.
    The result is the mean error over all valid joints of the kept samples
    (equal to the mean of per-sample errors when no joint is missing).
    Batched over samples (no Python loop). Returns NaN if nothing is kept.
    """
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if p.ndim == 2:
        p, g = p[None], g[None]
    if p.shape != g.shape or p.ndim != 3 or p.shape[-1] != 3:
        raise ValueError("pred and gt must both be (N, J, 3)")
    m = np.all(np.isfinite(p), axis=-1) & np.all(np.isfinite(g), axis=-1)
    if p.shape[0] == 0:
        return float("nan")
    w = m.astype(np.float64)
    p0 = np.where(m[..., None], p, 0.0)
    g0 = np.where(m[..., None], g, 0.0)
    s, r, t, ok = _umeyama_batch(p0, g0, w, with_scale=True)
    aligned = s[:, None, None] * np.einsum("bij,bnj->bni", r, p0) + t[:, None, :]
    keep = m & ok[:, None]
    if not np.any(keep):
        return float("nan")
    return float(np.mean(np.linalg.norm(aligned[keep] - g0[keep], axis=-1)))


# ----------------------------------------------------------------------------
# TUM files and timestamp association
# ----------------------------------------------------------------------------


def read_tum(path: str | os.PathLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read a TUM trajectory: ``timestamp tx ty tz qx qy qz qw`` per line.

    Blank lines and lines starting with ``#`` are skipped; commas are accepted
    as separators. Quaternions are normalised. Returns ``(timestamps (N,),
    poses (N, 4, 4))`` camera-to-world. Raises ``ValueError`` naming the line
    for a line that does not have 8 numbers, a non-finite value, or a
    quaternion of (near) zero length.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) != 8:
                raise ValueError(f"{path}:{lineno}: expected 8 values (timestamp tx ty tz qx qy qz qw), got {len(parts)}")
            try:
                row = [float(x) for x in parts]
            except ValueError:
                raise ValueError(f"{path}:{lineno}: not a number in {s!r}") from None
            if not all(math.isfinite(x) for x in row):
                raise ValueError(f"{path}:{lineno}: non-finite value in {s!r}")
            if math.sqrt(sum(x * x for x in row[4:8])) < 1e-9:
                raise ValueError(f"{path}:{lineno}: zero-length quaternion in {s!r}")
            rows.append(row)
    if not rows:
        return np.zeros(0), np.zeros((0, 4, 4))
    a = np.asarray(rows, dtype=np.float64)
    poses = np.zeros((a.shape[0], 4, 4))
    poses[:, :3, :3] = _quat_to_mat(a[:, [7, 4, 5, 6]])     # file order qx qy qz qw -> [w,x,y,z]
    poses[:, :3, 3] = a[:, 1:4]
    poses[:, 3, 3] = 1.0
    return a[:, 0], poses


def write_tum(path: str | os.PathLike, timestamps: ArrayLike, poses: ArrayLike) -> None:
    """Write a TUM trajectory (``timestamp tx ty tz qx qy qz qw``).

    Numbers are written with ``repr`` precision, so a write/read round trip is
    exact up to quaternion normalisation. Rows whose timestamp or pose is not
    finite are skipped (TUM has no missing-value marker).
    """
    ts = np.asarray(timestamps, dtype=np.float64).ravel()
    p = np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)
    if ts.shape[0] != p.shape[0]:
        raise ValueError("timestamps and poses lengths differ")
    ok = np.isfinite(ts) & np.all(np.isfinite(p), axis=(1, 2))
    q = np.zeros((p.shape[0], 4))
    if np.any(ok):
        q[ok] = _mat_to_quat(p[ok, :3, :3])
    with open(path, "w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for k in np.flatnonzero(ok):
            vals = (ts[k], p[k, 0, 3], p[k, 1, 3], p[k, 2, 3], q[k, 1], q[k, 2], q[k, 3], q[k, 0])
            f.write(" ".join(repr(float(v)) for v in vals) + "\n")


def associate(t_a: ArrayLike, t_b: ArrayLike, max_dt: float = 0.02) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """One-to-one nearest-timestamp matching.

    All pairs with ``|t_a[i] - t_b[j]| <= max_dt`` are candidates; they are
    accepted greedily in order of increasing time difference, each index used
    at most once (the scheme of TUM's ``associate.py``). Inputs need not be
    sorted. Returns ``(ia, ib)`` int arrays sorted by ``t_a``.
    """
    a = np.asarray(t_a, dtype=np.float64).ravel()
    b = np.asarray(t_b, dtype=np.float64).ravel()
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64))
    if a.size == 0 or b.size == 0:
        return empty
    ia_all = np.flatnonzero(np.isfinite(a))
    ib_all = np.flatnonzero(np.isfinite(b))
    order_b = ib_all[np.argsort(b[ib_all], kind="stable")]
    bs = b[order_b]
    lo = np.searchsorted(bs, a[ia_all] - max_dt, side="left")
    hi = np.searchsorted(bs, a[ia_all] + max_dt, side="right")
    cnt = hi - lo
    if cnt.sum() == 0:
        return empty
    ci = np.repeat(ia_all, cnt)
    starts = np.repeat(lo - np.cumsum(np.r_[0, cnt[:-1]]), cnt) + np.arange(cnt.sum())
    cj = order_b[starts]
    diff = np.abs(a[ci] - b[cj])
    keep = diff <= max_dt
    ci, cj, diff = ci[keep], cj[keep], diff[keep]
    order = np.lexsort((cj, ci, diff))
    used_a = np.zeros(a.size, bool)
    used_b = np.zeros(b.size, bool)
    out_a, out_b = [], []
    for k in order:
        i, j = ci[k], cj[k]
        if not used_a[i] and not used_b[j]:
            used_a[i] = used_b[j] = True
            out_a.append(i)
            out_b.append(j)
    ia = np.asarray(out_a, dtype=np.int64)
    ib = np.asarray(out_b, dtype=np.int64)
    srt = np.lexsort((ib, a[ia]))
    return ia[srt], ib[srt]
