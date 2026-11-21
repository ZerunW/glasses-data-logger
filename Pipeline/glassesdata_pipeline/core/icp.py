"""Rigid and similarity registration of point clouds (numpy only).

* :func:`icp_point_to_plane` - robust (Tukey or Huber) point-to-plane ICP of a source cloud to a
  model with normals, Gauss-Newton on se(3) with a small point-to-point term so that a single
  plane does not leave the problem singular. Returns the pose, Open3D-style ``fitness`` (fraction
  of source points with a model point within ``max_correspondence``) and the inlier RMSE.
* :func:`align_similarity` - scale + rotation + translation of a shape to target points
  (point-to-point ICP with Umeyama steps from several initial rotations about ``up``), used to put
  an image-to-3D mesh, which has an arbitrary frame and scale, onto the fused depth points.
* :func:`se3_exp`, :func:`average_poses` - small SE(3) helpers.

Poses map source coordinates to model coordinates: ``x_model = T @ x_source``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import geometry as geo
from .knn import KNN


def se3_exp(xi: Any) -> np.ndarray:
    """``[omega (3), v (3)]`` -> 4x4 (rotation ``so3_exp(omega)``, translation with the SE(3) V matrix)."""
    xi = np.asarray(xi, dtype=np.float64)
    w, v = xi[:3], xi[3:]
    R = geo.so3_exp(w)
    th = float(np.linalg.norm(w))
    K = geo.skew(w)
    if th < 1e-8:
        V = np.eye(3) + 0.5 * K
    else:
        V = np.eye(3) + (1 - np.cos(th)) / th**2 * K + (th - np.sin(th)) / th**3 * (K @ K)
    return geo.make_T(R, V @ v)


def _robust_weights(r: np.ndarray, kind: str, scale: float) -> np.ndarray:
    a = np.abs(r)
    if kind == "tukey":
        c = 4.685 * scale
        w = (1.0 - (a / c) ** 2) ** 2
        w[a >= c] = 0.0
        return w
    if kind == "huber":
        c = 1.345 * scale
        return np.where(a <= c, 1.0, c / np.maximum(a, 1e-12))
    return np.ones_like(a)


def icp_point_to_plane(
    source: Any,
    model_points: Any,
    model_normals: Any,
    T_init: Any = None,
    *,
    index: KNN | None = None,
    max_iterations: int = 30,
    max_correspondence: float = 0.03,
    robust: str = "tukey",
    min_scale: float = 0.002,
    point_weight: float = 0.05,
    min_points: int = 10,
    tolerance: float = 1e-6,
    dof: str | np.ndarray = "6",
) -> dict[str, Any]:
    """Register ``source`` ``(N, 3)`` to a model (points and unit normals ``(M, 3)``).

    ``dof`` restricts the update (expressed in the model frame): ``"6"`` (full rigid), ``"4"``
    (translation + rotation about the model's z axis, e.g. an upright object on a table),
    ``"3"`` (translation only), or a ``(6, k)`` basis of allowed ``[omega, v]`` directions.

    Returns ``{"T" (4x4 source -> model), "fitness", "rmse", "n_inliers", "iterations",
    "converged", "ok"}``; ``ok`` is False when fewer than ``min_points`` correspondences remain (``T``
    is then the last estimate). ``max_correspondence`` (metres) bounds the matching distance;
    residual weights use ``robust`` ("tukey", "huber" or "none") with a scale of 1.4826 x MAD of the
    point-to-plane residuals (at least ``min_scale``). ``point_weight`` adds that fraction of the
    point-to-point error, which keeps the problem regular when the visible part is one plane.
    """
    src = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    src = src[np.all(np.isfinite(src), axis=1)]
    mp = np.asarray(model_points, dtype=np.float64)
    mn = np.asarray(model_normals, dtype=np.float64)
    index = index if index is not None else KNN(mp)
    T = np.eye(4) if T_init is None else np.asarray(T_init, dtype=np.float64).copy()
    B = dof_basis(dof)
    out = {"T": T, "fitness": 0.0, "rmse": float("nan"), "n_inliers": 0, "iterations": 0, "converged": False,
           "ok": False}
    if len(src) < min_points:
        return out
    converged = False
    it = 0
    for it in range(1, max_iterations + 1):
        x = src @ T[:3, :3].T + T[:3, 3]
        dist, idx = index.nearest(x)
        m = dist <= max_correspondence
        if m.sum() < min_points:
            out.update(T=T, iterations=it)
            return out
        xm, q, n = x[m], mp[idx[m]], mn[idx[m]]
        r = np.einsum("ij,ij->i", n, xm - q)
        scale = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), min_scale)
        w = _robust_weights(r, robust, scale)
        if np.count_nonzero(w) < min_points:
            out.update(T=T, iterations=it)
            return out
        # point-to-plane rows: r + (x cross n) . omega + n . v
        J = np.concatenate([np.cross(xm, n), n], axis=1)
        H = (J * w[:, None]).T @ J
        g = (J * w[:, None]).T @ r
        if point_weight > 0:
            # point-to-point rows: (x - q) + omega cross x + v = (x - q) - [x]_x omega + v
            e = xm - q
            wp = point_weight * w
            Sx = geo.skew(xm)  # (k,3,3): Sx @ omega = x cross omega
            Jp = np.concatenate([-Sx, np.broadcast_to(np.eye(3), Sx.shape)], axis=2)  # (k,3,6)
            H += np.einsum("k,kai,kaj->ij", wp, Jp, Jp)
            g += np.einsum("k,kai,ka->i", wp, Jp, e)
        Hr = B.T @ H @ B
        gr = B.T @ g
        Hr += 1e-9 * max(np.trace(Hr), 1e-12) / len(Hr) * np.eye(len(Hr))
        try:
            delta = -B @ np.linalg.solve(Hr, gr)
        except np.linalg.LinAlgError:
            break
        T = se3_exp(delta) @ T
        if np.linalg.norm(delta[:3]) < tolerance and np.linalg.norm(delta[3:]) < tolerance:
            converged = True
            break
    x = src @ T[:3, :3].T + T[:3, 3]
    dist, idx = index.nearest(x)
    m = dist <= max_correspondence
    r = np.einsum("ij,ij->i", mn[idx[m]], x[m] - mp[idx[m]]) if np.any(m) else np.zeros(0)
    out.update(
        T=T, fitness=float(m.mean()), rmse=float(np.sqrt(np.mean(r**2))) if len(r) else float("nan"),
        n_inliers=int(m.sum()), iterations=it, converged=converged, ok=bool(m.sum() >= min_points),
    )
    return out


def dof_basis(dof: str | np.ndarray) -> np.ndarray:
    """``(6, k)`` basis of the allowed ``[omega, v]`` update directions."""
    if isinstance(dof, np.ndarray):
        return np.asarray(dof, dtype=np.float64).reshape(6, -1)
    I = np.eye(6)
    if str(dof) == "6":
        return I
    if str(dof) == "4":
        return I[:, [2, 3, 4, 5]]
    if str(dof) == "3":
        return I[:, 3:]
    raise ValueError(f"unknown ICP dof {dof!r} (6, 4, 3 or a basis)")


def average_poses(Ts: Any, weights: Any = None) -> np.ndarray:
    """Weighted mean of rigid transforms ``(N, 4, 4)``: chordal L2 mean of the rotations (SVD
    projection of the weighted sum) and weighted median of the translations (per axis)."""
    Ts = np.asarray(Ts, dtype=np.float64)
    w = np.ones(len(Ts)) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(Ts) == 0 or w.sum() <= 0:
        raise ValueError("no poses to average")
    R = geo.project_to_so3(np.einsum("n,nij->ij", w, Ts[:, :3, :3]))
    t = np.array([_weighted_median(Ts[:, a, 3], w) for a in range(3)])
    return geo.make_T(R, t)


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cw, 0.5 * cw[-1])])


def align_similarity(
    shape_points: Any,
    target_points: Any,
    *,
    up: Any = (0.0, 0.0, 1.0),
    n_yaw: int = 8,
    iterations: int = 40,
    trim: float = 0.8,
    with_scale: bool = True,
) -> dict[str, Any]:
    """Similarity ``target ~ s R shape + t`` from point-to-point ICP with Umeyama updates.

    Initialisation: centroids matched, scale from the RMS radii, rotations ``n_yaw`` turns about
    ``up`` (both clouds are assumed to be roughly upright - image-to-3D models output upright
    objects and the target is in a gravity-aligned frame). Each iteration matches every target
    point to its nearest transformed shape point (the target is partial, the shape complete) and
    keeps the ``trim`` fraction with the smallest distances. Returns ``{"s", "R", "t", "rmse",
    "T" (4x4 with the scale folded into the rotation block)}`` of the best start.
    """
    from .align import umeyama

    S = np.asarray(shape_points, dtype=np.float64)
    P = np.asarray(target_points, dtype=np.float64)
    cs, cp = S.mean(axis=0), P.mean(axis=0)
    s0 = float(np.sqrt(np.mean(np.sum((P - cp) ** 2, axis=1))) / max(np.sqrt(np.mean(np.sum((S - cs) ** 2, axis=1))), 1e-12))
    up_v = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    best: dict[str, Any] | None = None
    for k in range(max(n_yaw, 1)):
        R = geo.so3_exp(up_v * (2 * np.pi * k / max(n_yaw, 1)))
        s = s0 if with_scale else 1.0
        t = cp - s * R @ cs
        rmse = np.inf
        for _ in range(iterations):
            X = s * S @ R.T + t
            d, i = KNN(X).nearest(P)
            keep = d <= np.quantile(d, trim)
            try:
                s_new, R_new, t_new = umeyama(S[i[keep]], P[keep], with_scale=with_scale)
            except ValueError:
                break
            s, R, t = s_new, R_new, t_new
            new_rmse = float(np.sqrt(np.mean(d[keep] ** 2)))
            if abs(rmse - new_rmse) < 1e-7:
                rmse = new_rmse
                break
            rmse = new_rmse
        if best is None or rmse < best["rmse"]:
            best = {"s": float(s), "R": R, "t": t, "rmse": float(rmse)}
    assert best is not None
    T = np.eye(4)
    T[:3, :3] = best["s"] * best["R"]
    T[:3, 3] = best["t"]
    best["T"] = T
    return best
