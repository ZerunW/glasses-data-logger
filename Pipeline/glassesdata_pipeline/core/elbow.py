"""Elbow position from shoulder, wrist, segment lengths and optional image cues.

Given the shoulder S and wrist W (metres, world frame, z up) and the upper-arm
and forearm lengths, the elbow lies on the circle where the sphere (S, l_upper)
meets the sphere (W, l_fore). This module parameterises that circle by a swivel
angle and picks the angle per frame from an optional image cue (a plane
through the camera centre that contains the visible forearm), a direction
prior (elbows usually hang down) and temporal smoothness, solved jointly over
the trajectory with a banded Viterbi.

Only NumPy is used. All functions are pure.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

CUE_SIGMA = 0.05       # m; plane distance that costs w_cue in the cue term
SIDE_WEIGHT = 4.0      # hinge weight of the cue_side term, relative to w_cue
_FALLBACK_NORMAL = np.array([0.0, 0.0, 1.0])

STATUS_OK, STATUS_TOO_FAR, STATUS_TOO_CLOSE, STATUS_INVALID = 0, 1, 2, 3
METHOD_INVALID, METHOD_CUE, METHOD_PRIOR = 0, 1, 2


def _norm(v: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sqrt(np.sum(v * v, axis=-1))


def _fallback_axis(n: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit vector perpendicular to unit ``n`` (...,3): the coordinate axis
    least aligned with ``n``, made orthogonal. Deterministic."""
    eye = np.eye(3)
    pick = np.argmin(np.abs(np.nan_to_num(n)), axis=-1)
    e = eye[pick]
    u = e - np.sum(e * n, axis=-1, keepdims=True) * n
    nu = _norm(u)[..., None]
    return u / np.where(nu > 0, nu, 1.0)


def _circle_basis(normal: NDArray[np.float64], ref_dir: NDArray[np.float64]
                  ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Orthonormal in-plane basis (u, v = n x u) with u the projection of
    ``ref_dir`` onto the plane normal to ``normal``; if that projection is
    (nearly) zero or not finite, ``_fallback_axis`` is used."""
    n, ref = np.broadcast_arrays(normal, ref_dir)
    ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
    u = ref - np.sum(ref * n, axis=-1, keepdims=True) * n
    nu = _norm(u)
    nref = _norm(ref)
    degenerate = ~(nu > 1e-9 * np.maximum(nref, 1e-300)) | ~(nref > 0)
    u = np.where(degenerate[..., None], _fallback_axis(n), u / np.where(nu > 0, nu, 1.0)[..., None])
    return u, np.cross(n, u)


def elbow_circle(shoulder: ArrayLike, wrist: ArrayLike, l_upper: ArrayLike, l_fore: ArrayLike
                 ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int8]]:
    """Circle of feasible elbow positions.

    Parameters
    ----------
    shoulder, wrist : (..., 3). l_upper, l_fore : scalars or (...) lengths.

    Returns
    -------
    center (..., 3), normal (..., 3) unit vector S -> W, radius (...),
    status (...) int8:

    * 0 ok;
    * 1 too far (``|SW| >= l_upper + l_fore``): radius 0, center on the
      segment at ``l_upper`` from S (the straight-arm elbow);
    * 2 too close (``|SW| <= |l_upper - l_fore|``): radius 0, best-effort
      center at ``l_upper`` from S towards W if ``l_upper >= l_fore``, away
      from W otherwise (the point of the shoulder sphere closest to the wrist
      sphere). If S == W the normal falls back to +z;
    * 3 invalid input (non-finite values or non-positive lengths): all
      outputs NaN. (Extension of the 0/1/2 codes.)
    """
    s = np.asarray(shoulder, dtype=np.float64)
    w = np.asarray(wrist, dtype=np.float64)
    s, w = np.broadcast_arrays(s, w)
    shape = s.shape[:-1]
    lu = np.broadcast_to(np.asarray(l_upper, dtype=np.float64), shape)
    lf = np.broadcast_to(np.asarray(l_fore, dtype=np.float64), shape)
    valid = (np.all(np.isfinite(s), axis=-1) & np.all(np.isfinite(w), axis=-1)
             & np.isfinite(lu) & np.isfinite(lf) & (lu > 0) & (lf > 0))
    s0 = np.where(valid[..., None], s, 0.0)
    w0 = np.where(valid[..., None], w, 0.0)
    lu0 = np.where(valid, lu, 1.0)
    lf0 = np.where(valid, lf, 1.0)
    d = w0 - s0
    dist = _norm(d)
    pos = dist > 0
    dist_safe = np.where(pos, dist, 1.0)
    n = np.where(pos[..., None], d / dist_safe[..., None], _FALLBACK_NORMAL)
    too_far = dist >= lu0 + lf0
    too_close = (dist <= np.abs(lu0 - lf0)) & ~too_far
    ok = ~too_far & ~too_close
    a = (dist * dist + lu0 * lu0 - lf0 * lf0) / (2.0 * dist_safe)
    r = np.sqrt(np.maximum(lu0 * lu0 - a * a, 0.0))
    along = np.where(ok, a, np.where(too_far, lu0, np.where(lu0 >= lf0, lu0, -lu0)))
    center = s0 + along[..., None] * n
    radius = np.where(ok, r, 0.0)
    status = np.where(ok, STATUS_OK, np.where(too_far, STATUS_TOO_FAR, STATUS_TOO_CLOSE)).astype(np.int8)
    status = np.where(valid, status, STATUS_INVALID).astype(np.int8)
    center = np.where(valid[..., None], center, np.nan)
    n = np.where(valid[..., None], n, np.nan)
    radius = np.where(valid, radius, np.nan)
    return center, n, radius, status


def circle_point(center: ArrayLike, normal: ArrayLike, radius: ArrayLike, ref_dir: ArrayLike,
                 phi: ArrayLike) -> NDArray[np.float64]:
    """Point on the circle at swivel angle ``phi`` (radians).

    ``phi = 0`` is the point closest to ``ref_dir`` (the direction of its
    projection onto the circle plane, seen from the center); positive ``phi``
    rotates right-handed about ``normal``, i.e. towards ``normal x u``. If
    ``ref_dir`` is parallel to ``normal`` (or zero), the coordinate axis
    least aligned with ``normal`` is projected instead. Broadcasts over
    leading dimensions: center/normal/ref_dir (..., 3), radius/phi (...).
    """
    c = np.asarray(center, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    ref = np.asarray(ref_dir, dtype=np.float64)
    r = np.asarray(radius, dtype=np.float64)
    ph = np.asarray(phi, dtype=np.float64)
    u, v = _circle_basis(n, ref)
    return c + r[..., None] * (np.cos(ph)[..., None] * u + np.sin(ph)[..., None] * v)


def plane_from_image_line(cam_center: ArrayLike, ray_a: ArrayLike, ray_b: ArrayLike
                          ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Plane through the camera centre containing two viewing rays.

    ``ray_a``, ``ray_b`` are world-frame directions (e.g. through the wrist
    pixel and a pixel further along the visible forearm); ``cam_center`` is
    (..., 3). Returns ``(n, d)`` with unit ``n`` (..., 3) and ``n . x + d = 0``
    for points x of the plane.

    Orientation: ``n`` is parallel to ``ray_a x ray_b``, so rotating ``ray_a``
    towards ``ray_b`` is right-handed about ``n``; equivalently
    ``n . (ray_a x (ray_b - ray_a)) > 0``. Parallel or zero rays give NaN.
    """
    c = np.asarray(cam_center, dtype=np.float64)
    a = np.asarray(ray_a, dtype=np.float64)
    b = np.asarray(ray_b, dtype=np.float64)
    cr = np.cross(a, b)
    ncr = _norm(cr)
    scale = _norm(a) * _norm(b)
    good = ncr > 1e-12 * scale
    n = np.where(good[..., None], cr / np.where(good, ncr, 1.0)[..., None], np.nan)
    d = -np.sum(n * c, axis=-1)
    return n, d


def _viterbi_banded(unary: NDArray[np.float64], gaps: NDArray[np.int64], k: int,
                    w_smooth: float, step: float) -> NDArray[np.int64]:
    """Min-cost path over a ring of N states.

    ``unary`` (M, N); ``gaps`` (M,) frame distance to the previous row (gaps[0]
    unused). Between rows only shifts ``|s| <= min(k * gap, N // 2)`` are
    allowed, at cost ``w_smooth * (s * step)^2 / gap``. O(M * N * k).
    """
    m, n = unary.shape
    cols = np.arange(n)
    cache: dict[int, tuple[NDArray[np.int64], NDArray[np.int64]]] = {}
    back = np.empty((m, n), dtype=np.int32)
    cost = unary[0].copy()
    for i in range(1, m):
        g = int(gaps[i])
        kk = min(k * g, n // 2)
        if kk not in cache:
            shifts = np.arange(-kk, kk + 1)
            cache[kk] = (shifts, (cols[None, :] - shifts[:, None]) % n)
        shifts, idx = cache[kk]
        tc = (w_smooth / g) * (shifts * step) ** 2
        cand = cost[idx] + tc[:, None]
        best = np.argmin(cand, axis=0)
        back[i] = idx[best, cols]
        cost = cand[best, cols] + unary[i]
        cost -= cost.min()
    path = np.empty(m, dtype=np.int64)
    path[-1] = int(np.argmin(cost))
    for i in range(m - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def solve_elbow(
    shoulder: ArrayLike,
    wrist: ArrayLike,
    l_upper: ArrayLike,
    l_fore: ArrayLike,
    prior_dir: ArrayLike = (0.0, 0.0, -1.0),
    cue_plane: ArrayLike | None = None,
    cue_side: ArrayLike | None = None,
    w_cue: float = 1.0,
    w_prior: float = 0.2,
    w_smooth: float = 2.0,
    n_samples: int = 360,
    dt: float = 1.0 / 24.0,
    max_swivel_speed: float = 6.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int8]]:
    """Elbow trajectory from shoulder/wrist trajectories and optional cues.

    Parameters
    ----------
    shoulder, wrist : (T, 3), NaN rows allowed.
    l_upper, l_fore : scalars or (T,).
    prior_dir : (3,) or (T, 3) world direction the elbow tends to point to
        from the circle center (default straight down).
    cue_plane : optional (T, 4) ``[nx, ny, nz, d]`` (n need not be unit), NaN
        where there is no forearm cue. Typically from ``plane_from_image_line``.
    cue_side : optional (T, 3) direction in which the forearm extends from the
        wrist as seen in the image; circle points with ``(E - W) . cue_side < 0``
        are penalised. NaN rows = no side information.
    w_cue, w_prior, w_smooth : weights of the three cost terms.
    n_samples : swivel angles per frame (grid step ``2 pi / n_samples``).
    dt, max_swivel_speed : frame period (s) and the largest swivel speed
        (rad/s) the transition band allows (see Notes).

    Cost per frame at swivel angle psi, elbow ``E(psi)``:

    * ``w_cue * dist(E, cue_plane)^2 / 0.05^2`` when a cue plane exists;
    * ``w_cue * 4 * max(0, -cos(E - W, cue_side))`` when cue_side exists;
    * ``w_prior * (1 - u(psi) . p)`` with ``u`` the unit vector from the center
      to E and ``p`` the unit prior. This equals the spec's ``w_prior * (1 -
      cos(angle to the projected prior))`` scaled by ``|p_perp|`` (the length
      of the prior's projection onto the circle plane) plus a constant, so the
      prior fades out smoothly when ``prior_dir`` is (nearly) parallel to the
      arm axis, where its projection, and so the projected-angle cost, is undefined.

    and consecutive valid frames add ``w_smooth * wrapped(dpsi)^2`` (divided by
    the frame gap when NaN frames lie between them).

    Returns
    -------
    elbow : (T, 3), NaN for invalid frames.
    phi : (T,) swivel angle in the convention of ``circle_point(center,
        normal, radius, prior_dir, phi)``, which reproduces ``elbow``.
    conf : (T,) in [0, 1]; see Notes.
    method : (T,) int8, 0 invalid, 1 cue plane present, 2 prior only.

    Notes
    -----
    *Solver.* Viterbi over ``n_samples`` angles with a banded transition: only
    shifts of at most ``k = ceil(max_swivel_speed * dt / step)`` grid steps
    per frame (15 steps = 0.26 rad at the defaults), times the gap across NaN
    frames, capped at the full circle. Cost O(T * n_samples * k): about 0.1 s
    for T = 3000, n = 360. The chosen angle is then refined by a parabola
    through the conditional cost of its two neighbours (which is minimal at
    the chosen sample, so the shift stays within half a step).

    *Reference.* Swivel angles are measured from a reference direction that is
    parallel-transported from frame to frame (projected onto the new circle
    plane), so the smoothness term measures real swivel change even when the
    prior's projection flips (arm axis passing through the prior direction).

    *Confidence.* For cue frames ``conf = sharp * unambiguous`` where
    ``sharp = 1 / (1 + (sigma_pos / 0.1 m)^2)``, ``sigma_pos = radius *
    sqrt(2 / curvature)`` of the frame cost at its minimum, and ``unambiguous
    = 1 - exp(-margin / 0.3)`` with ``margin`` the cost gap to the best angle
    more than 30 deg away (the other plane intersection). Prior-only frames get
    ``0.25 * |p_perp|``; degenerate circles (status 1/2, radius 0) get 0.5;
    invalid frames 0.
    """
    S = np.asarray(shoulder, dtype=np.float64).reshape(-1, 3)
    W = np.asarray(wrist, dtype=np.float64).reshape(-1, 3)
    if S.shape != W.shape:
        raise ValueError("shoulder and wrist must both be (T, 3)")
    T = S.shape[0]
    N = int(n_samples)
    if N < 8:
        raise ValueError("n_samples must be >= 8")
    elbow = np.full((T, 3), np.nan)
    phi_out = np.full(T, np.nan)
    conf = np.zeros(T)
    method = np.zeros(T, dtype=np.int8)
    if T == 0:
        return elbow, phi_out, conf, method

    lu = np.broadcast_to(np.asarray(l_upper, dtype=np.float64), (T,))
    lf = np.broadcast_to(np.asarray(l_fore, dtype=np.float64), (T,))
    center, normal, radius, status = elbow_circle(S, W, lu, lf)
    p = np.broadcast_to(np.asarray(prior_dir, dtype=np.float64), (T, 3)).copy()
    pn = _norm(p)
    p_ok = np.isfinite(pn) & (pn > 0)
    p = np.where(p_ok[:, None], p / np.where(p_ok, pn, 1.0)[:, None], 0.0)

    if cue_plane is not None:
        cp = np.asarray(cue_plane, dtype=np.float64).reshape(T, 4)
        cn = _norm(cp[:, :3])
        has_cue = np.all(np.isfinite(cp), axis=1) & (cn > 0)
        cp = np.where(has_cue[:, None], cp / np.where(has_cue, cn, 1.0)[:, None], 0.0)
    else:
        cp, has_cue = np.zeros((T, 4)), np.zeros(T, bool)
    if cue_side is not None:
        cs = np.asarray(cue_side, dtype=np.float64).reshape(T, 3)
        sn = _norm(cs)
        has_side = np.all(np.isfinite(cs), axis=1) & (sn > 0)
        cs = np.where(has_side[:, None], cs / np.where(has_side, sn, 1.0)[:, None], 0.0)
    else:
        cs, has_side = np.zeros((T, 3)), np.zeros(T, bool)

    valid = status != STATUS_INVALID
    vi = np.flatnonzero(valid)
    if vi.size == 0:
        return elbow, phi_out, conf, method
    M = vi.size
    C, n, r = center[vi], normal[vi], radius[vi]
    pv = p[vi]

    # Parallel-transported reference direction per valid frame.
    u_prior, _ = _circle_basis(n, pv)
    ref = np.empty((M, 3))
    ref[0] = u_prior[0]
    for i in range(1, M):
        w_ = ref[i - 1] - np.dot(ref[i - 1], n[i]) * n[i]
        nw = math.sqrt(float(np.dot(w_, w_)))
        ref[i] = w_ / nw if nw > 0.1 else u_prior[i]
    ref_x = np.cross(n, ref)

    step = 2.0 * math.pi / N
    psi = np.arange(N) * step
    cpsi, spsi = np.cos(psi), np.sin(psi)
    dirs = cpsi[None, :, None] * ref[:, None, :] + spsi[None, :, None] * ref_x[:, None, :]   # (M, N, 3)
    E = C[:, None, :] + r[:, None, None] * dirs

    unary = w_prior * (1.0 - np.einsum("mnk,mk->mn", dirs, pv))
    cue_v = has_cue[vi]
    cue_cost = np.zeros((M, N))
    if np.any(cue_v):
        cpv = cp[vi]
        dist = np.einsum("mnk,mk->mn", E, cpv[:, :3]) + cpv[:, 3:4]
        cue_cost = np.where(cue_v[:, None], w_cue * (dist / CUE_SIGMA) ** 2, 0.0)
    side_v = has_side[vi]
    if np.any(side_v):
        fore = E - W[vi][:, None, :]
        fn = _norm(fore)
        cosang = np.einsum("mnk,mk->mn", fore, cs[vi]) / np.where(fn > 0, fn, 1.0)
        cue_cost = cue_cost + np.where(side_v[:, None], w_cue * SIDE_WEIGHT * np.maximum(0.0, -cosang), 0.0)
    unary = unary + cue_cost

    gaps = np.empty(M, dtype=np.int64)
    gaps[0] = 1
    gaps[1:] = np.diff(vi)
    k = max(1, int(math.ceil(max_swivel_speed * dt / step))) if dt > 0 and max_swivel_speed > 0 else N // 2
    path = _viterbi_banded(unary, gaps, k, float(w_smooth), step)

    # Parabolic refinement on the conditional cost (neighbours fixed).
    rows = np.arange(M)
    offs = np.array([-1, 0, 1])
    js = (path[:, None] + offs[None, :]) % N                      # (M, 3)
    f = unary[rows[:, None], js].copy()

    def _wrap(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return (x + math.pi) % (2.0 * math.pi) - math.pi

    if M > 1:
        dprev = _wrap((js[1:] - path[:-1, None]) * step)
        f[1:] += (w_smooth / gaps[1:, None]) * dprev ** 2
        dnext = _wrap((path[1:, None] - js[:-1]) * step)
        f[:-1] += (w_smooth / gaps[1:, None]) * dnext ** 2
    den = f[:, 0] - 2.0 * f[:, 1] + f[:, 2]
    delta = np.where(den > 0, 0.5 * (f[:, 0] - f[:, 2]) / np.where(den > 0, den, 1.0), 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    ang = (path + delta) * step
    d_sel = np.cos(ang)[:, None] * ref + np.sin(ang)[:, None] * ref_x
    E_sel = C + r[:, None] * d_sel

    # phi in the circle_point(prior_dir) convention.
    up, vp = _circle_basis(n, pv)
    phi_v = np.arctan2(np.sum(d_sel * vp, axis=1), np.sum(d_sel * up, axis=1))

    # Confidence.
    p_perp = _norm(pv - np.sum(pv * n, axis=1, keepdims=True) * n)
    conf_v = 0.25 * np.clip(p_perp, 0.0, 1.0)
    cue_rows = np.flatnonzero(cue_v & (status[vi] == STATUS_OK))
    if cue_rows.size:
        fc = unary[cue_rows]
        jc = path[cue_rows]
        fm = fc[np.arange(cue_rows.size), jc]
        curv = (fc[np.arange(cue_rows.size), (jc - 1) % N] - 2.0 * fm
                + fc[np.arange(cue_rows.size), (jc + 1) % N]) / step ** 2
        sig_pos = r[cue_rows] * np.sqrt(2.0 / np.maximum(curv, 1e-12))
        sharp = 1.0 / (1.0 + (sig_pos / 0.1) ** 2)
        far = np.abs(_wrap((np.arange(N)[None, :] - jc[:, None]) * step)) > math.radians(30.0)
        margin = np.min(np.where(far, fc, np.inf), axis=1) - fm
        unamb = 1.0 - np.exp(-np.maximum(margin, 0.0) / 0.3)
        conf_v[cue_rows] = sharp * unamb
    degenerate = status[vi] != STATUS_OK
    conf_v[degenerate] = 0.5

    elbow[vi] = E_sel
    phi_out[vi] = phi_v
    conf[vi] = np.clip(conf_v, 0.0, 1.0)
    method[vi] = np.where(cue_v, METHOD_CUE, METHOD_PRIOR)
    return elbow, phi_out, conf, method
