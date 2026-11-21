"""Metric hand pose from 2D keypoints and a known 3D hand shape (pure numpy/OpenCV, no file I/O).

Used by step ``hand3d``. One hand in one frame:

1. :func:`hand_shape`: a metric 21x3 shape. MediaPipe's world landmarks (``mp_world``: metres,
   origin near the hand centre, x/y along the image axes, z away from the camera) are rescaled so
   that the robust median ratio of the user's bone lengths to their bone lengths over reliable
   bones is 1 (``mode="scale"``), or every bone is set to the user's length along MediaPipe's
   direction (``mode="retarget"``). Without ``mp_world`` the profile's flat rest pose is used.
2. :func:`solve_pose`: rigid pose (R, t) with ``X_cam = R @ X + t`` from undistorted keypoints,
   weighted by keypoint score: candidates from the shape's own (camera-aligned) orientation, the
   ray-aligned orientation, OpenCV's SQPnP and the planar "flip" of each; every candidate is refined
   by a Huber-weighted Levenberg-Marquardt on the reprojection error. Candidates behind the camera
   or outside the depth range are rejected. Among candidates whose reprojection error is close to
   the best (the planar flip ambiguity), the one most consistent with the priors wins: the
   shape's own orientation, the previous frame's rotation and the depth map's relative depths.
3. :func:`snap_to_rays`: optionally moves each joint (part of the way) onto its observed ray,
   keeping its depth: the 2D detector is usually more precise than the 3D shape.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import geometry as geo
from .. import skeleton as sk

PARENTS = np.array(sk.PARENTS)
PALM = (0, 5, 9, 13, 17)
#: Bones (by child joint index) whose MediaPipe length is used to fix the shape's scale: the
#: wrist->MCP bones and the proximal phalanges.
SCALE_BONES = (5, 9, 13, 17, 6, 10, 14, 18)


# =============================================================================================
# Shape
# =============================================================================================


def bone_lengths(joints: np.ndarray) -> np.ndarray:
    """``(..., 21, 3)`` -> ``(..., 20)`` lengths of bones 1..20."""
    J = np.asarray(joints, dtype=np.float64)
    return np.linalg.norm(J[..., 1:, :] - J[..., PARENTS[1:], :], axis=-1)


def scale_to_bones(shape: np.ndarray, target: np.ndarray, bones: Sequence[int] = SCALE_BONES) -> float:
    """Median of ``target / length`` over ``bones`` (child joint indices) with valid lengths; NaN
    when none is usable."""
    lengths = bone_lengths(shape)
    idx = np.asarray(bones, dtype=int) - 1
    ratio = np.asarray(target, dtype=np.float64)[idx] / lengths[idx]
    ratio = ratio[np.isfinite(ratio) & (lengths[idx] > 1e-4)]
    return float(np.median(ratio)) if ratio.size else math.nan


def retarget_bones(shape: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Same bone directions, the ``target`` lengths, wrist kept fixed (kinematic-tree order)."""
    J = np.asarray(shape, dtype=np.float64)
    out = np.empty_like(J)
    out[0] = J[0]
    for i in range(1, 21):
        d = J[i] - J[PARENTS[i]]
        n = np.linalg.norm(d)
        out[i] = out[PARENTS[i]] + (d / n * target[i - 1] if n > 1e-9 else d)
    return out


def hand_shape(mp_world: np.ndarray | None, bones_m: np.ndarray, profile: Any, side: str,
               mode: str = "scale", scale_bones: Sequence[int] = SCALE_BONES,
               max_scale_change: float = 0.5, factor: float | None = None) -> tuple[np.ndarray, int, float]:
    """``(shape (21, 3), source, factor)``: source 1 = ``mp_world`` (factor = applied scale; pass
    ``factor`` to use a precomputed, e.g. temporally smoothed, one), 2 = rest pose of the profile
    (no orientation information; factor NaN)."""
    if mp_world is not None and np.all(np.isfinite(mp_world)):
        if factor is None or not np.isfinite(factor):
            factor = scale_to_bones(mp_world, bones_m, scale_bones)
        if np.isfinite(factor) and abs(math.log(factor)) < math.log1p(max_scale_change) + 1e-12:
            shape = mp_world * factor
            if mode == "retarget":
                shape = retarget_bones(shape, bones_m)
            return shape, 1, factor
    return sk.rest_pose(profile, side), 2, math.nan


# =============================================================================================
# Pose
# =============================================================================================


@dataclass
class PoseResult:
    R: np.ndarray
    t: np.ndarray
    joints: np.ndarray  # (21, 3) camera frame
    rms_px: float  # weighted RMS reprojection error of the used joints, pixels
    n_used: int
    ambiguous: bool  # another candidate had a similar reprojection error
    choice: str  # which prior decided ("cost", "prior", ...)
    candidates: list[dict[str, Any]] = field(default_factory=list)


def _project(P: np.ndarray) -> np.ndarray:
    return P[:, :2] / P[:, 2:3]


def translation_for_rotation(X: np.ndarray, xy: np.ndarray, w: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Least-squares translation for a fixed rotation: ``x_j (RX_j + t)_z = (RX_j + t)_x`` (and y),
    weighted by ``w``."""
    P = X @ R.T
    n = len(X)
    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)
    A[0::2, 0] = 1.0
    A[0::2, 2] = -xy[:, 0]
    b[0::2] = xy[:, 0] * P[:, 2] - P[:, 0]
    A[1::2, 1] = 1.0
    A[1::2, 2] = -xy[:, 1]
    b[1::2] = xy[:, 1] * P[:, 2] - P[:, 1]
    sw = np.sqrt(np.repeat(w, 2))
    t, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
    return t


def refine_pose(X: np.ndarray, xy: np.ndarray, w: np.ndarray, R: np.ndarray, t: np.ndarray, f_px: float,
                huber_px: float = 3.0, iterations: int = 20, R_anchor: np.ndarray | None = None,
                anchor_px_per_rad: float = 0.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Levenberg-Marquardt on the weighted, Huber-robust reprojection error.

    Parametrisation about the shape's centroid (well conditioned at hand distances):
    ``X_cam = exp(w) R (X - Xbar) + c``. With ``R_anchor`` an orientation prior is added: the
    residual ``anchor_px_per_rad * log(R R_anchor^T)`` (pixel-equivalent units, i.e.
    ``sigma_px / sigma_rot``); it fixes the weakly observed out-of-plane rotation of a nearly planar
    hand. Returns ``(R, t, cost)``: the robust weighted sum of squared pixel residuals plus the
    prior term."""
    Xbar = np.average(X, axis=0, weights=w)
    Y = X - Xbar
    c = R @ Xbar + t
    lam = 1e-3
    k = float(anchor_px_per_rad) if R_anchor is not None else 0.0
    Ra = np.asarray(R_anchor, dtype=np.float64) if R_anchor is not None else np.eye(3)

    def residuals(R_: np.ndarray, c_: np.ndarray):
        P = Y @ R_.T + c_
        if np.any(P[:, 2] <= 1e-6):
            return None
        r = (_project(P) - xy) * f_px  # pixels
        norm = np.linalg.norm(r, axis=1)
        hw = np.where(norm <= huber_px, 1.0, huber_px / np.maximum(norm, 1e-12))
        cost = float(np.sum(w * np.where(norm <= huber_px, norm**2, 2 * huber_px * norm - huber_px**2)))
        rp = k * geo.so3_log(R_ @ Ra.T) if k > 0 else np.zeros(3)
        return P, r, hw, cost + float(rp @ rp), rp

    state = residuals(R, c)
    if state is None:
        return R, t, math.inf
    P, r, hw, cost, rp = state
    done = False
    for _ in range(iterations):
        # Jacobian of the projection (pixels) w.r.t. [omega (3), c (3)]
        z = P[:, 2]
        J_proj = np.zeros((len(P), 2, 3))
        J_proj[:, 0, 0] = 1.0 / z
        J_proj[:, 0, 2] = -P[:, 0] / z**2
        J_proj[:, 1, 1] = 1.0 / z
        J_proj[:, 1, 2] = -P[:, 1] / z**2
        J_proj *= f_px
        RY = Y @ R.T
        J_rot = -geo.skew(RY)  # d(exp(w) RY)/dw at 0 = -[RY]x
        J = np.concatenate([J_proj @ J_rot, J_proj], axis=2).reshape(-1, 6)
        ww = np.repeat(w * hw, 2)
        rr = r.reshape(-1)
        H = (J * ww[:, None]).T @ J
        g = (J * ww[:, None]).T @ rr
        if k > 0:  # d log(exp(w) R Ra^T) / dw ~ identity near the anchor
            H[:3, :3] += k * k * np.eye(3)
            g[:3] += k * rp
        improved = False
        for _ in range(8):
            try:
                step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            R_new = geo.so3_exp(step[:3]) @ R
            c_new = c + step[3:]
            new = residuals(R_new, c_new)
            if new is not None and new[3] < cost:
                R, c = R_new, c_new
                P, r, hw, cost_new, rp = new
                done = cost - cost_new < 1e-10 * max(cost, 1e-12)
                cost = cost_new
                lam = max(lam / 3, 1e-9)
                improved = True
                break
            lam *= 10
        if not improved or done:
            break
    return R, c - R @ Xbar, cost


def flip_candidate(X: np.ndarray, w: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The planar "Necker" flip of a pose: the shape mirrored through the plane perpendicular to the
    line of sight at its centroid, composed with the mirror through the shape's own best-fit plane
    (a rotation again). For a planar shape both poses project almost identically."""
    Xbar = np.average(X, axis=0, weights=w)
    Y = X - Xbar
    _, _, Vt = np.linalg.svd(Y * np.sqrt(w)[:, None], full_matrices=False)
    n_obj = Vt[-1]
    c = R @ Xbar + t
    v = c / np.linalg.norm(c)
    H = np.eye(3) - 2.0 * np.outer(v, v)
    M = np.eye(3) - 2.0 * np.outer(n_obj, n_obj)
    R_flip = geo.project_to_so3(H @ R @ M)
    return R_flip, c - R_flip @ Xbar


def _sqpnp(X: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    import cv2

    if len(X) < 4:
        return None
    try:
        ok, rvec, tvec = cv2.solvePnP(X.astype(np.float64), xy.astype(np.float64), np.eye(3), None,
                                      flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return None
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def depth_consistency(z_joints: np.ndarray, z_model: np.ndarray) -> float:
    """Mean absolute difference of relative depths (``z / median(z) - 1``) between candidate joints
    and the depth map samples at the same joints; NaN with fewer than 3 samples."""
    m = np.isfinite(z_joints) & np.isfinite(z_model) & (z_model > 0)
    if m.sum() < 3:
        return math.nan
    a = z_joints[m] / np.median(z_joints[m]) - 1.0
    b = z_model[m] / np.median(z_model[m]) - 1.0
    return float(np.mean(np.abs(a - b)))


def solve_pose(
    X: np.ndarray,
    xy: np.ndarray,
    weights: np.ndarray,
    f_px: float,
    R_prior: np.ndarray | None = None,
    R_prev: np.ndarray | None = None,
    prev_gap: int = 1,
    z_model_palm: np.ndarray | None = None,
    huber_px: float = 3.0,
    min_z: float = 0.03,
    depth_range: tuple[float, float] = (0.05, 3.0),
    ambiguity_ratio: float = 1.5,
    ambiguity_px: float = 1.0,
    prior_sigma_rad: float = 0.6,
    prev_sigma_rad: float = 0.25,
    depth_sigma: float = 0.01,
    anchor_px_per_rad: float = 0.0,
    depth_joints: Sequence[int] = PALM,
) -> PoseResult | None:
    """Rigid pose of shape ``X`` (21, 3) from normalized undistorted points ``xy`` (21, 2).

    Args:
        weights: (21,) keypoint weights; joints with weight <= 0 or NaN data are ignored.
        f_px: focal length in pixels (to express residuals in pixels).
        R_prior: the rotation the shape's own orientation suggests (identity for camera-aligned
            MediaPipe world landmarks); None when unknown (rest-pose fallback).
        R_prev: the hand's rotation in a previous frame ``prev_gap`` frames ago, or None.
        z_model_palm: depth-map values (any scale) at ``depth_joints`` or None.
        ambiguity_ratio, ambiguity_px: candidates with ``cost_rms <= best * ratio + px`` compete on
            the priors.
        prior_sigma_rad, prev_sigma_rad, depth_sigma: scales of the prior terms (the previous-frame
            term grows with ``sqrt(prev_gap)``).
        anchor_px_per_rad: weight of the orientation prior inside the refinement (``sigma_px /
            sigma_rot``, see :func:`refine_pose`); 0 = none. Uses ``R_prior``.
    """
    X = np.asarray(X, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    use = np.isfinite(w) & (w > 0) & np.all(np.isfinite(xy), axis=1) & np.all(np.isfinite(X), axis=1)
    n_used = int(use.sum())
    if n_used < 4:
        return None
    Xu, xyu, wu = X[use], xy[use], w[use] / np.max(w[use])
    inits: list[tuple[str, np.ndarray]] = []
    if R_prior is not None:
        inits.append(("prior", np.asarray(R_prior, dtype=np.float64)))
        centre = np.average(xyu, axis=0, weights=wu)
        ray = np.array([centre[0], centre[1], 1.0])
        inits.append(("ray", geo.rot_from_two_vectors(np.array([0.0, 0.0, 1.0]), ray) @ np.asarray(R_prior)))
    if R_prev is not None:
        inits.append(("prev", np.asarray(R_prev, dtype=np.float64)))
    sq = _sqpnp(Xu, xyu)
    seeds: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, R0 in inits:
        seeds.append((name, R0, translation_for_rotation(Xu, xyu, wu, R0)))
    if sq is not None:
        seeds.append(("sqpnp", sq[0], sq[1]))
    results: list[dict[str, Any]] = []
    for name, R0, t0 in seeds:
        for flipped in (False, True):
            if flipped:
                if not np.all(np.isfinite(t0)) or np.linalg.norm(R0 @ np.average(Xu, 0, weights=wu) + t0) < 1e-6:
                    continue
                R0f, t0f = flip_candidate(Xu, wu, R0, t0)
            else:
                R0f, t0f = R0, t0
            if not np.all(np.isfinite(t0f)):
                continue
            R, t, cost = refine_pose(Xu, xyu, wu, R0f, t0f, f_px, huber_px, R_anchor=R_prior,
                                     anchor_px_per_rad=anchor_px_per_rad)
            if not math.isfinite(cost):
                continue
            P = X @ R.T + t
            zc = float(np.average(P[use, 2], weights=wu))
            if np.any(P[use, 2] < min_z) or not depth_range[0] <= zc <= depth_range[1]:
                continue
            err = np.linalg.norm((_project(P[use]) - xyu) * f_px, axis=1)
            rms = float(np.sqrt(np.sum(wu * err**2) / np.sum(wu)))
            if any(geo.rotation_angle(R, r["R"]) < math.radians(2.0) and np.linalg.norm(t - r["t"]) < 0.01 for r in results):
                continue
            results.append({"name": name + ("+flip" if flipped else ""), "R": R, "t": t, "rms": rms, "cost": cost, "P": P})
    if not results:
        return None
    best = min(r["rms"] for r in results)
    close = [r for r in results if r["rms"] <= best * ambiguity_ratio + ambiguity_px]
    choice = "cost"
    if len(close) > 1:
        for r in close:
            score = (r["rms"] - best) ** 2 / max(best, 1.0) ** 2
            terms = []
            if R_prior is not None:
                score += (geo.rotation_angle(r["R"], R_prior) / prior_sigma_rad) ** 2
                terms.append("prior")
            if R_prev is not None:
                score += (geo.rotation_angle(r["R"], R_prev) / (prev_sigma_rad * math.sqrt(max(prev_gap, 1)))) ** 2
                terms.append("prev")
            if z_model_palm is not None:
                dc = depth_consistency(r["P"][list(depth_joints), 2], np.asarray(z_model_palm, dtype=np.float64))
                if math.isfinite(dc):
                    score += (dc / depth_sigma) ** 2
                    terms.append("depth")
            r["score"] = float(score)
        chosen = min(close, key=lambda r: r["score"])
        choice = "+".join(terms) if terms else "cost"
    else:
        chosen = close[0]
    return PoseResult(
        R=chosen["R"], t=chosen["t"], joints=chosen["P"], rms_px=chosen["rms"], n_used=n_used,
        ambiguous=len(close) > 1, choice=choice,
        candidates=[{k: v for k, v in r.items() if k in ("name", "rms", "score")} for r in results],
    )


def snap_to_rays(joints: np.ndarray, xy: np.ndarray, amount: float = 1.0, mask: np.ndarray | None = None) -> np.ndarray:
    """Move joints (21, 3) towards the closest point on their observed rays ``(x, y, 1)``
    (``amount`` 0 = unchanged, 1 = on the ray); joints without an observation stay."""
    J = np.array(joints, dtype=np.float64, copy=True)
    if amount <= 0:
        return J
    rays = np.concatenate([np.asarray(xy, dtype=np.float64), np.ones((len(J), 1))], axis=1)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    ok = np.all(np.isfinite(rays), axis=1) & np.all(np.isfinite(J), axis=1)
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    on_ray = np.sum(J * rays, axis=1, keepdims=True) * rays
    J[ok] += amount * (on_ray[ok] - J[ok])
    return J
