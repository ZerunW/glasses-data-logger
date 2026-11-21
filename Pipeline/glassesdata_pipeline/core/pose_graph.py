"""Pose-graph optimisation on SE(3) with Levenberg-Marquardt (NumPy only).

Nodes are rigid poses ``T_world_i`` (4x4, camera-to-world). An edge ``(i, j, Z_ij, Omega_ij)``
says that the relative pose ``T_i^-1 T_j`` was measured as ``Z_ij`` with 6x6 information
``Omega_ij`` (order: rotation, translation). The residual of an edge is

    D = Z_ij^-1 T_i^-1 T_j,     r = [log_SO3(R_D), t_D]      (6,)

(rotation in radians, translation in the frame of ``j`` as predicted by ``Z_ij``), and the cost
``sum rho(r^T Omega r)`` with an optional Huber kernel on the Mahalanobis norm.

Poses are updated on the right, decoupled: ``R <- R exp(dr)``, ``t <- t + R dt``. Jacobians are
central finite differences, vectorised over edges (12 perturbations per iteration), and the
normal equations are solved densely: fine for a few hundred nodes (6 N unknowns).
The gauge is fixed by holding the ``fixed`` nodes (default: node 0) constant.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import geometry as geo


@dataclass
class Edge:
    """A relative-pose measurement ``T_i^-1 T_j ~ Z`` with information ``info`` (6x6)."""

    i: int
    j: int
    Z: NDArray[np.float64]
    info: NDArray[np.float64]
    kind: str = "odometry"


def _residuals(Ti: NDArray[np.float64], Tj: NDArray[np.float64], Zinv: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(E, 6)`` residuals for batched poses ``Ti``, ``Tj`` and inverse measurements ``Zinv``."""
    D = Zinv @ geo.inv_T(Ti) @ Tj
    return np.concatenate([geo.so3_log(D[:, :3, :3]), D[:, :3, 3]], axis=-1)


def _perturb(T: NDArray[np.float64], delta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Right, decoupled update of ``(..., 4, 4)`` poses by ``(..., 6)`` increments."""
    out = T.copy()
    R = T[..., :3, :3]
    out[..., :3, :3] = R @ geo.so3_exp(delta[..., :3])
    out[..., :3, 3] = T[..., :3, 3] + np.einsum("...ij,...j->...i", R, delta[..., 3:])
    return out


def _jacobians(Ti, Tj, Zinv, eps: float = 1e-6):
    """Residuals ``(E, 6)`` and Jacobians w.r.t. the increments of ``i`` and ``j`` ``(E, 6, 6)``."""
    r0 = _residuals(Ti, Tj, Zinv)
    E = r0.shape[0]
    Ji = np.zeros((E, 6, 6))
    Jj = np.zeros((E, 6, 6))
    for k in range(6):
        d = np.zeros((E, 6))
        d[:, k] = eps
        Ji[:, :, k] = (_residuals(_perturb(Ti, d), Tj, Zinv) - _residuals(_perturb(Ti, -d), Tj, Zinv)) / (2 * eps)
        Jj[:, :, k] = (_residuals(Ti, _perturb(Tj, d), Zinv) - _residuals(Ti, _perturb(Tj, -d), Zinv)) / (2 * eps)
    return r0, Ji, Jj


def _robust_weights(chi2: NDArray[np.float64], huber: float | None) -> NDArray[np.float64]:
    if huber is None or not np.isfinite(huber) or huber <= 0:
        return np.ones_like(chi2)
    e = np.sqrt(np.maximum(chi2, 0.0))
    return np.where(e <= huber, 1.0, huber / np.maximum(e, 1e-300))


def _cost(chi2: NDArray[np.float64], huber: float | None) -> float:
    if huber is None or not np.isfinite(huber) or huber <= 0:
        return float(np.sum(chi2))
    e = np.sqrt(np.maximum(chi2, 0.0))
    return float(np.sum(np.where(e <= huber, chi2, 2.0 * huber * e - huber * huber)))


def edge_errors(T: ArrayLike, edges: Sequence[Edge]) -> dict[str, NDArray[np.float64]]:
    """Per-edge rotation error (deg), translation error (m) and chi2 for poses ``T`` (N, 4, 4)."""
    T = np.asarray(T, dtype=np.float64)
    if not edges:
        return {"rot_deg": np.zeros(0), "trans_m": np.zeros(0), "chi2": np.zeros(0)}
    ii = np.array([e.i for e in edges])
    jj = np.array([e.j for e in edges])
    Zinv = geo.inv_T(np.stack([e.Z for e in edges]))
    info = np.stack([e.info for e in edges])
    r = _residuals(T[ii], T[jj], Zinv)
    return {
        "rot_deg": np.degrees(np.linalg.norm(r[:, :3], axis=1)),
        "trans_m": np.linalg.norm(r[:, 3:], axis=1),
        "chi2": np.einsum("ei,eij,ej->e", r, info, r),
    }


def optimize(
    T_init: ArrayLike,
    edges: Sequence[Edge],
    fixed: Sequence[int] = (0,),
    iters: int = 30,
    huber: float | None = None,
    lambda0: float = 1e-4,
    tol: float = 1e-10,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Levenberg-Marquardt over all poses.

    Args:
        T_init: ``(N, 4, 4)`` initial poses.
        edges: relative-pose measurements (node indices into ``T_init``).
        fixed: nodes held constant (gauge); at least one is needed for a unique solution.
        iters: maximum LM iterations.
        huber: Huber threshold on ``sqrt(r^T Omega r)`` (None = plain least squares).
        lambda0: initial damping (multiplies the Hessian diagonal).
        tol: stop when the relative cost decrease falls below this.

    Returns:
        ``(T_opt (N, 4, 4), info)`` with ``info = {"cost0", "cost", "iterations", "accepted"}``.
    """
    T = np.array(T_init, dtype=np.float64, copy=True)
    N = T.shape[0]
    report: dict[str, Any] = {"cost0": 0.0, "cost": 0.0, "iterations": 0, "accepted": 0}
    if N == 0 or not edges:
        return T, report
    ii = np.array([e.i for e in edges], dtype=np.int64)
    jj = np.array([e.j for e in edges], dtype=np.int64)
    if ii.min() < 0 or jj.min() < 0 or max(ii.max(), jj.max()) >= N:
        raise ValueError("edge node index out of range")
    Zinv = geo.inv_T(np.stack([np.asarray(e.Z, dtype=np.float64) for e in edges]))
    info = np.stack([np.asarray(e.info, dtype=np.float64) for e in edges])
    free = np.ones(N, dtype=bool)
    free[list(fixed)] = False
    if not np.any(free):
        return T, report
    col = -np.ones(N, dtype=np.int64)
    col[free] = np.arange(int(free.sum()))
    n_var = 6 * int(free.sum())

    def evaluate(Tc):
        r = _residuals(Tc[ii], Tc[jj], Zinv)
        chi2 = np.einsum("ei,eij,ej->e", r, info, r)
        return r, chi2

    r, chi2 = evaluate(T)
    cost = _cost(chi2, huber)
    report["cost0"] = cost
    lam = float(lambda0)
    for it in range(int(iters)):
        report["iterations"] = it + 1
        r, Ji, Jj = _jacobians(T[ii], T[jj], Zinv)
        chi2 = np.einsum("ei,eij,ej->e", r, info, r)
        w = _robust_weights(chi2, huber)
        W = info * w[:, None, None]
        H = np.zeros((n_var, n_var))
        g = np.zeros(n_var)
        blocks = (("i", ii, Ji), ("j", jj, Jj))
        for _, na, Ja in blocks:
            JtW = np.einsum("eki,ekl->eil", Ja, W)  # (E, 6, 6) = Ja^T W
            ga = np.einsum("eil,el->ei", JtW, r)
            for e in np.flatnonzero(free[na]):
                c = 6 * col[na[e]]
                g[c:c + 6] += ga[e]
            for _, nb, Jb in blocks:
                Hab = np.einsum("eil,elj->eij", JtW, Jb)
                for e in np.flatnonzero(free[na] & free[nb]):
                    ca, cb = 6 * col[na[e]], 6 * col[nb[e]]
                    H[ca:ca + 6, cb:cb + 6] += Hab[e]
        diag = np.diag(H).copy()
        diag[diag <= 0] = 1e-12
        improved = False
        for _ in range(10):
            A = H + lam * np.diag(diag)
            try:
                step = np.linalg.solve(A, -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            delta = np.zeros((N, 6))
            delta[free] = step.reshape(-1, 6)
            T_new = _perturb(T, delta)
            _, chi2_new = evaluate(T_new)
            cost_new = _cost(chi2_new, huber)
            if cost_new < cost:
                rel = (cost - cost_new) / max(cost, 1e-300)
                T, cost = T_new, cost_new
                lam = max(lam / 10.0, 1e-12)
                improved = True
                report["accepted"] += 1
                break
            lam *= 10.0
        if not improved or rel < tol:
            break
    # Re-orthonormalise the rotations (finite-precision drift of repeated updates).
    T[:, :3, :3] = geo.project_to_so3(T[:, :3, :3])
    report["cost"] = cost
    return T, report


def information(n_inliers: float, translation_m: float, sigma_rot: float = 0.01, sigma_trans: float = 0.02,
                scale_rel: float = 0.02, floor_trans: float = 1e-3) -> NDArray[np.float64]:
    """Diagonal information of a PnP relative pose from its inlier count.

    Rotation sigma ``sigma_rot / sqrt(n)`` (rad); translation sigma
    ``sqrt(sigma_trans^2 / n + (scale_rel * |t|)^2 + floor^2)`` (m): the second term models the
    depth-scale error of the keyframe whose depth gave the 3D points (proportional to the
    distance travelled), which more inliers do not reduce.
    """
    n = max(float(n_inliers), 1.0)
    s_r = sigma_rot / np.sqrt(n)
    s_t = np.sqrt(sigma_trans**2 / n + (scale_rel * float(translation_m)) ** 2 + floor_trans**2)
    return np.diag([1 / s_r**2] * 3 + [1 / s_t**2] * 3)


__all__ = ["Edge", "optimize", "edge_errors", "information"]
