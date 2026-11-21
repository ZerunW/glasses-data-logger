"""Array-module helpers shared by the RL math (``xp`` is ``numpy`` or ``torch``).

Every function here takes the array module as its first argument so the same code runs on numpy
arrays (unit tests on the Mac) and on torch tensors (training on the GPU). Only operations whose
spelling is identical in numpy and torch are used directly (``x.sum(-1)``, ``xp.exp``,
``xp.where``, indexing); the rest goes through the small wrappers below.

Quaternions are ``[w, x, y, z]`` (PIPELINE.md section 1, MuJoCo, MuJoCo Warp).
"""

from __future__ import annotations

from typing import Any

import numpy as np

Array = Any


# =============================================================================================
# array-module wrappers
# =============================================================================================


def is_torch(xp: Any) -> bool:
    return getattr(xp, "__name__", "") == "torch"


def cat(xp: Any, arrays: list[Array], axis: int = -1) -> Array:
    return xp.cat(list(arrays), dim=axis) if is_torch(xp) else xp.concatenate(list(arrays), axis=axis)


def stack(xp: Any, arrays: list[Array], axis: int = -1) -> Array:
    return xp.stack(list(arrays), dim=axis) if is_torch(xp) else xp.stack(list(arrays), axis=axis)


def atan2(xp: Any, y: Array, x: Array) -> Array:
    return xp.atan2(y, x) if is_torch(xp) else xp.arctan2(y, x)


def clip(xp: Any, x: Array, lo: Any, hi: Any) -> Array:
    """``lo``/``hi``: both Python numbers or both arrays broadcastable to ``x``."""
    return xp.clamp(x, lo, hi) if is_torch(xp) else xp.clip(x, lo, hi)


def norm(xp: Any, x: Array, axis: int = -1) -> Array:
    return xp.sqrt((x * x).sum(axis))


def amax(xp: Any, x: Array, axis: int = -1) -> Array:
    return xp.amax(x, axis)


def amin(xp: Any, x: Array, axis: int = -1) -> Array:
    return xp.amin(x, axis)


def to_int(xp: Any, x: Array) -> Array:
    return x.long() if is_torch(xp) else x.astype(np.int64)


def to_float(xp: Any, x: Array, like: Array) -> Array:
    return x.to(like.dtype) if is_torch(xp) else x.astype(like.dtype)


def asarray(xp: Any, data: Any, like: Array | None = None, dtype: Any = None, device: Any = None) -> Array:
    """``data`` as an array of ``like``'s dtype/device (or the given ``dtype``/``device``)."""
    if is_torch(xp):
        if like is not None:
            dtype = like.dtype if dtype is None else dtype
            device = like.device if device is None else device
        return xp.as_tensor(np.asarray(data) if not isinstance(data, xp.Tensor) else data, dtype=dtype, device=device)
    if like is not None and dtype is None:
        dtype = like.dtype
    return np.asarray(data, dtype=dtype)


def index_array(xp: Any, data: Any, device: Any = None) -> Array:
    """An int64 index array (``torch.long`` on ``device``)."""
    if is_torch(xp):
        return xp.as_tensor(np.asarray(data, dtype=np.int64), dtype=xp.long, device=device)
    return np.asarray(data, dtype=np.int64)


def nonzero(xp: Any, mask: Array) -> Array:
    """Indices of the true entries of a 1-D boolean array."""
    return xp.nonzero(mask).reshape(-1) if is_torch(xp) else np.nonzero(mask)[0]


def zeros(xp: Any, shape: tuple[int, ...], like: Array) -> Array:
    if is_torch(xp):
        return xp.zeros(shape, dtype=like.dtype, device=like.device)
    return np.zeros(shape, dtype=like.dtype)


def isfinite_rows(xp: Any, x: Array) -> Array:
    """``(N, ...)`` -> ``(N,)`` true where every entry of the row is finite."""
    return xp.isfinite(x).reshape(x.shape[0], -1).all(-1)


def nan_to_zero(xp: Any, x: Array) -> Array:
    """NaN and +-inf -> 0 (``numpy.nan_to_num`` / ``torch.nan_to_num``)."""
    return xp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class Rng:
    """Seeded uniform / normal samples for numpy or torch (``device`` for torch)."""

    def __init__(self, xp: Any, seed: int, device: Any = None, dtype: Any = None) -> None:
        self.xp = xp
        self.device = device
        if is_torch(xp):
            self.dtype = dtype or xp.float32
            self.device = device if device is not None else "cpu"
            self.gen = xp.Generator(device=self.device)
            self.gen.manual_seed(int(seed))
        else:
            self.dtype = dtype or np.float64
            self.gen = np.random.default_rng(int(seed))

    def uniform(self, shape: tuple[int, ...], lo: float = 0.0, hi: float = 1.0) -> Array:
        if is_torch(self.xp):
            u = self.xp.rand(shape, generator=self.gen, device=self.device, dtype=self.dtype)
        else:
            u = self.gen.random(shape).astype(self.dtype)
        return lo + (hi - lo) * u

    def normal(self, shape: tuple[int, ...]) -> Array:
        if is_torch(self.xp):
            return self.xp.randn(shape, generator=self.gen, device=self.device, dtype=self.dtype)
        return self.gen.standard_normal(shape).astype(self.dtype)


# =============================================================================================
# vectors and quaternions ([w, x, y, z])
# =============================================================================================


def cross(xp: Any, a: Array, b: Array) -> Array:
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return stack(xp, [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], -1)


def quat_normalize(xp: Any, q: Array, eps: float = 1e-9) -> Array:
    n = norm(xp, q)[..., None]
    return q / xp.where(n > eps, n, n * 0 + 1.0)


def quat_conj(xp: Any, q: Array) -> Array:
    return cat(xp, [q[..., :1], -q[..., 1:]], -1)


def quat_mul(xp: Any, a: Array, b: Array) -> Array:
    """Hamilton product; ``R(a * b) = R(a) R(b)``."""
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return stack(xp, [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], -1)


def quat_rotate(xp: Any, q: Array, v: Array) -> Array:
    """Rotate vectors ``v`` (..., 3) by unit quaternions ``q`` (..., 4)."""
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * cross(xp, u, v)
    return v + w * t + cross(xp, u, t)


def quat_rotate_inv(xp: Any, q: Array, v: Array) -> Array:
    return quat_rotate(xp, quat_conj(xp, q), v)


def quat_from_rotvec(xp: Any, r: Array) -> Array:
    """Rotation vector (axis * angle) -> unit quaternion (exact; small-angle safe)."""
    theta = norm(xp, r)[..., None]
    half = 0.5 * theta
    small = theta < 1e-6
    safe = xp.where(small, theta * 0 + 1.0, theta)
    k = xp.where(small, 0.5 - theta * theta / 48.0, xp.sin(half) / safe)
    return cat(xp, [xp.cos(half), r * k], -1)


def quat_to_rotvec(xp: Any, q: Array) -> Array:
    """Unit quaternion -> rotation vector with angle in [0, pi] (q and -q give the same)."""
    w = q[..., :1]
    sign = xp.where(w < 0, w * 0 - 1.0, w * 0 + 1.0)
    w = w * sign
    v = q[..., 1:] * sign
    s = norm(xp, v)[..., None]
    angle = 2.0 * atan2(xp, s, w)
    small = s < 1e-9
    k = xp.where(small, 2.0 / xp.where(w > 1e-12, w, w * 0 + 1.0), angle / xp.where(small, s * 0 + 1.0, s))
    return v * k


def quat_angle(xp: Any, a: Array, b: Array) -> Array:
    """Geodesic angle (radians, [0, pi]) between unit quaternions ``a`` and ``b``."""
    d = quat_mul(xp, quat_conj(xp, a), b)
    return 2.0 * atan2(xp, norm(xp, d[..., 1:]), xp.abs(d[..., 0]))


def quat_to_mat(xp: Any, q: Array) -> Array:
    q = quat_normalize(xp, q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rows = [
        stack(xp, [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        stack(xp, [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        stack(xp, [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ]
    return stack(xp, rows, -2)


def quat_to_rot6d(xp: Any, q: Array) -> Array:
    """The 6D rotation representation (Zhou et al. 2019): the first two columns of R, i.e.
    ``[R00, R10, R20, R01, R11, R21]``."""
    R = quat_to_mat(xp, q)
    return cat(xp, [R[..., :, 0], R[..., :, 1]], -1)


def rot6d_to_mat(xp: Any, r: Array) -> Array:
    """Inverse of :func:`quat_to_rot6d` (Gram-Schmidt), for checks and decoding."""
    a1, a2 = r[..., 0:3], r[..., 3:6]
    b1 = a1 / norm(xp, a1)[..., None]
    a2 = a2 - (b1 * a2).sum(-1)[..., None] * b1
    b2 = a2 / norm(xp, a2)[..., None]
    b3 = cross(xp, b1, b2)
    return stack(xp, [b1, b2, b3], -1)


def exp_kernel(xp: Any, err: Array, sigma: float) -> Array:
    """``exp(-(err / sigma)^2)``: 1 at zero error, 0.37 at ``err = sigma``."""
    return xp.exp(-((err / sigma) ** 2))
