"""Camera model and SE(3)/SO(3) helpers (numpy only).

Conventions (PIPELINE.md section 1):

* Camera frame: OpenCV, x right, y down, z forward. Pixel (u, v) has its centre at integer
  coordinates, origin at the top-left pixel; the image covers [-0.5, W-0.5] x [-0.5, H-0.5].
* ``T_a_b`` is a 4x4 matrix mapping frame-b coordinates to frame a: ``p_a = T_a_b @ p_b``.
* Quaternions are ``[w, x, y, z]``.
* "Normalized" image coordinates are undistorted ``(x, y) = (X/Z, Y/Z)``; a "ray" is ``(x, y, 1)``
  so that ``X_cam = ray * depth`` with depth = z (not the Euclidean range).

:class:`Camera` implements OpenCV's ``pinhole`` model (``dist`` with 4, 5, 8, 12 or 14 values:
k1 k2 p1 p2 [k3 [k4 k5 k6 [s1 s2 s3 s4 [tauX tauY]]]]) and ``fisheye`` model (``cv2.fisheye``,
``dist = [k1, k2, k3, k4]``) in numpy. ``project`` matches ``cv2.projectPoints`` /
``cv2.fisheye.projectPoints``; ``undistort_points`` inverts it with Newton iterations (round trip
error < 1e-6 px inside the image). Points behind the camera (z <= 0) or beyond the radius where the
distortion polynomial stops being monotonic (it would fold far-away points back into the image)
project to NaN.

All functions accept batches: leading dimensions broadcast.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .errors import PipelineError

ArrayLike = Any

# =============================================================================================
# Image sizes
# =============================================================================================


def downsample_size(width: int, height: int, downsample: float) -> tuple[int, int]:
    """Size ``(w, h)`` of an image downsampled by ``downsample`` (rounded, at least 1).

    Every downsampled grid in the pipeline (``Episode.frames``, depth maps, masks,
    ``Camera.pixel_rays``) uses this so that their pixels line up.
    """
    if downsample <= 0:
        raise ValueError("downsample must be positive")
    return max(1, int(round(width / downsample))), max(1, int(round(height / downsample)))


# =============================================================================================
# Camera
# =============================================================================================

_PINHOLE_SIZES = (4, 5, 8, 12, 14)


class Camera:
    """A calibrated camera for one image size.

    Build it with :meth:`from_dict` (a ``calibration.json`` / ``camera.json`` dict),
    :meth:`fov_default` or the constructor.

    Attributes: ``K`` (3x3), ``dist`` (1-D array, possibly empty), ``width``, ``height``,
    ``model`` (``"pinhole"`` or ``"fisheye"``), ``source`` (``"charuco-web"``, ``"synthetic"``,
    ``"default"``, ...).
    """

    def __init__(
        self,
        K: ArrayLike,
        dist: ArrayLike | None,
        width: int,
        height: int,
        model: str = "pinhole",
        source: str = "default",
    ) -> None:
        self.K = np.array(K, dtype=np.float64).reshape(3, 3)
        self.dist = np.zeros(0) if dist is None else np.array(dist, dtype=np.float64).reshape(-1)
        self.width = int(width)
        self.height = int(height)
        self.model = str(model)
        self.source = str(source)
        if self.model == "pinhole":
            if self.dist.size not in (0,) + _PINHOLE_SIZES:
                raise PipelineError(f"pinhole dist must have 4, 5, 8, 12 or 14 values, got {self.dist.size}")
        elif self.model == "fisheye":
            if self.dist.size not in (0, 4):
                raise PipelineError(f"fisheye dist must have 4 values, got {self.dist.size}")
        else:
            raise PipelineError(f"unknown camera model {self.model!r} (pinhole or fisheye)")
        self._Kinv = np.linalg.inv(self.K)
        self._coeffs = self._padded_coefficients()
        self._tilt, self._tilt_inv = self._tilt_matrices()
        self._max_radius = _max_radius_for(self.model, tuple(self._coeffs.tolist()))
        self._rays: dict[float, np.ndarray] = {}

    # ---------------------------------------------------------------- construction / I/O

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], width: int | None = None, height: int | None = None) -> "Camera":
        """From a calibration dict (PIPELINE.md section 3). Given ``width``/``height`` (the video's
        size), K is rescaled when they differ from the calibration's; a different aspect ratio
        raises :class:`PipelineError`."""
        try:
            camera = cls(
                data["K"], data.get("dist"), int(data["width"]), int(data["height"]),
                data.get("model", "pinhole"), data.get("source", "default"),
            )
        except KeyError as error:
            raise PipelineError(f"calibration lacks {error}") from None
        if width is not None and height is not None and (int(width), int(height)) != (camera.width, camera.height):
            if abs(width / height - camera.width / camera.height) > 0.01 * (width / height):
                raise PipelineError(
                    f"calibration is {camera.width}x{camera.height} but the video is {width}x{height}: "
                    "different aspect ratio"
                )
            camera = camera.resized(int(width), int(height))
        return camera

    def to_dict(self) -> dict[str, Any]:
        """The calibration dict (PIPELINE.md section 3) without ``rms_px``/``created``/``device``."""
        return {
            "version": 1,
            "model": self.model,
            "width": self.width,
            "height": self.height,
            "K": self.K.tolist(),
            "dist": self.dist.tolist(),
            "source": self.source,
        }

    @classmethod
    def fov_default(cls, width: int, height: int, hfov_deg: float = 65.0, source: str = "default") -> "Camera":
        """A distortion-free pinhole camera with the given horizontal field of view, square
        pixels and the principal point at the image centre ``((W-1)/2, (H-1)/2)``."""
        fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        K = [[fx, 0.0, (width - 1) / 2.0], [0.0, fx, (height - 1) / 2.0], [0.0, 0.0, 1.0]]
        return cls(K, [], width, height, "pinhole", source)

    def resized(self, width: int, height: int) -> "Camera":
        """The same camera for the image resized (``cv2.resize``) to ``width`` x ``height``.

        Uses the pixel-centre convention: ``u' = (u + 0.5) * sx - 0.5``, so ``cx' = (cx + 0.5) * sx - 0.5``.
        Distortion coefficients (in normalized coordinates) are unchanged.
        """
        sx, sy = width / self.width, height / self.height
        K = self.K.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        K[0, 2] = (self.K[0, 2] + 0.5) * sx - 0.5
        K[1, 2] = (self.K[1, 2] + 0.5) * sy - 0.5
        return Camera(K, self.dist, width, height, self.model, self.source)

    def scaled(self, factor: float) -> "Camera":
        """``scaled(0.25)`` is the camera of the image downsampled by 4 (see :func:`downsample_size`)."""
        w, h = downsample_size(self.width, self.height, 1.0 / factor)
        return self.resized(w, h)

    def downsampled(self, downsample: float) -> "Camera":
        """The camera of the grid used for ``downsample`` (same as ``scaled(1/downsample)``)."""
        w, h = downsample_size(self.width, self.height, downsample)
        return self.resized(w, h)

    # ---------------------------------------------------------------- properties

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)``."""
        return self.width, self.height

    @property
    def max_radius(self) -> float:
        """Largest normalized radius sqrt(x^2+y^2) (tan of the angle off-axis) that projects."""
        return self._max_radius

    def __repr__(self) -> str:
        return (
            f"Camera({self.model}, {self.width}x{self.height}, fx={self.fx:.2f}, fy={self.fy:.2f}, "
            f"cx={self.cx:.2f}, cy={self.cy:.2f}, dist={np.round(self.dist, 5).tolist()}, source={self.source!r})"
        )

    # ---------------------------------------------------------------- projection

    def project(self, X_cam: ArrayLike) -> np.ndarray:
        """Camera-frame points ``(..., 3)`` -> pixels ``(..., 2)``; NaN when not projectable."""
        X = np.asarray(X_cam, dtype=np.float64)
        z = X[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = X[..., 0] / z
            y = X[..., 1] / z
        bad = ~(z > 1e-9)
        xy = self.distort(np.stack([x, y], axis=-1))
        xy[bad] = np.nan
        return self._to_pixels(xy)

    def distort(self, xy: ArrayLike) -> np.ndarray:
        """Normalized undistorted ``(..., 2)`` -> normalized distorted ``(..., 2)`` (before K).
        NaN beyond :attr:`max_radius`."""
        xy = np.asarray(xy, dtype=np.float64)
        x, y = xy[..., 0], xy[..., 1]
        r = np.hypot(x, y)
        if self.model == "fisheye":
            xd, yd = self._fisheye_forward(x, y, r)
        else:
            xd, yd = self._pinhole_forward(x, y)[:2]
            if self._tilt is not None:
                xd, yd = _apply_homography(self._tilt, xd, yd)
        out = np.stack([xd, yd], axis=-1)
        out[~(r < self._max_radius)] = np.nan
        return out

    def undistort_points(self, uv: ArrayLike, iterations: int = 20) -> np.ndarray:
        """Pixels ``(..., 2)`` -> normalized undistorted ``(x, y)`` ``(..., 2)``; NaN where the
        inverse does not exist (outside the model's valid region)."""
        uv = np.asarray(uv, dtype=np.float64)
        xd, yd = self._from_pixels(uv)
        if self.model == "fisheye":
            return self._fisheye_inverse(xd, yd, iterations)
        if self._tilt_inv is not None:
            xd, yd = _apply_homography(self._tilt_inv, xd, yd)
        return self._pinhole_inverse(xd, yd, iterations)

    def unproject(self, uv: ArrayLike, depth: ArrayLike) -> np.ndarray:
        """Pixels ``(..., 2)`` and z-depth ``(...)`` -> camera-frame points ``(..., 3)``."""
        xy = self.undistort_points(uv)
        d = np.asarray(depth, dtype=np.float64)
        return np.stack([xy[..., 0] * d, xy[..., 1] * d, np.broadcast_to(d, xy.shape[:-1])], axis=-1)

    def pixel_rays(self, downsample: float = 1) -> np.ndarray:
        """Rays ``(h, w, 3)`` = ``(x, y, 1)`` for every pixel centre of the image downsampled by
        ``downsample`` (grid of :func:`downsample_size`). Cached; the array is read-only."""
        key = float(downsample)
        if key not in self._rays:
            camera = self if key == 1.0 else self.downsampled(key)
            u, v = np.meshgrid(np.arange(camera.width, dtype=np.float64), np.arange(camera.height, dtype=np.float64))
            xy = camera.undistort_points(np.stack([u, v], axis=-1))
            rays = np.concatenate([xy, np.ones(xy.shape[:-1] + (1,))], axis=-1)
            rays.setflags(write=False)
            self._rays[key] = rays
        return self._rays[key]

    def in_image(self, uv: ArrayLike, margin: float = 0.0) -> np.ndarray:
        """True where ``uv`` lies inside the image (pixel extents ``[-0.5, W-0.5]``) by ``margin`` px."""
        uv = np.asarray(uv, dtype=np.float64)
        u, v = uv[..., 0], uv[..., 1]
        with np.errstate(invalid="ignore"):
            return (
                (u >= -0.5 + margin) & (u <= self.width - 0.5 - margin)
                & (v >= -0.5 + margin) & (v <= self.height - 0.5 - margin)
            )

    # ---------------------------------------------------------------- internals

    def _to_pixels(self, xy: np.ndarray) -> np.ndarray:
        K = self.K
        u = K[0, 0] * xy[..., 0] + K[0, 1] * xy[..., 1] + K[0, 2]
        v = K[1, 1] * xy[..., 1] + K[1, 2]
        return np.stack([u, v], axis=-1)

    def _from_pixels(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Ki = self._Kinv
        u, v = uv[..., 0], uv[..., 1]
        y = Ki[1, 1] * v + Ki[1, 2]
        x = Ki[0, 0] * u + Ki[0, 1] * v + Ki[0, 2]
        return x, y

    def _padded_coefficients(self) -> np.ndarray:
        if self.model == "fisheye":
            c = np.zeros(4)
            c[: self.dist.size] = self.dist
            return c
        c = np.zeros(14)
        c[: self.dist.size] = self.dist
        return c

    def _tilt_matrices(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.model != "pinhole" or self.dist.size < 14 or (self.dist[12] == 0 and self.dist[13] == 0):
            return None, None
        tau_x, tau_y = self.dist[12], self.dist[13]
        cx_, sx_ = math.cos(tau_x), math.sin(tau_x)
        cy_, sy_ = math.cos(tau_y), math.sin(tau_y)
        rot_x = np.array([[1, 0, 0], [0, cx_, sx_], [0, -sx_, cx_]])
        rot_y = np.array([[cy_, 0, -sy_], [0, 1, 0], [sy_, 0, cy_]])
        rot_xy = rot_y @ rot_x
        proj_z = np.array([[rot_xy[2, 2], 0, -rot_xy[0, 2]], [0, rot_xy[2, 2], -rot_xy[1, 2]], [0, 0, 1]])
        tilt = proj_z @ rot_xy
        return tilt, np.linalg.inv(tilt)

    def _pinhole_forward(self, x: np.ndarray, y: np.ndarray, jacobian: bool = False):
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = self._coeffs[:12]
        r2 = x * x + y * y
        a = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        b = 1 + r2 * (k4 + r2 * (k5 + r2 * k6))
        radial = a / b
        xy2 = 2 * x * y
        xd = x * radial + p1 * xy2 + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r2 * r2
        yd = y * radial + p1 * (r2 + 2 * y * y) + p2 * xy2 + s3 * r2 + s4 * r2 * r2
        if not jacobian:
            return xd, yd, None
        da = k1 + r2 * (2 * k2 + 3 * k3 * r2)
        db = k4 + r2 * (2 * k5 + 3 * k6 * r2)
        dR = (da * b - a * db) / (b * b)  # d radial / d r2
        j11 = radial + 2 * x * x * dR + 2 * p1 * y + 6 * p2 * x + 2 * s1 * x + 4 * s2 * r2 * x
        j12 = 2 * x * y * dR + 2 * p1 * x + 2 * p2 * y + 2 * s1 * y + 4 * s2 * r2 * y
        j21 = 2 * x * y * dR + 2 * p1 * x + 2 * p2 * y + 2 * s3 * x + 4 * s4 * r2 * x
        j22 = radial + 2 * y * y * dR + 6 * p1 * y + 2 * p2 * x + 2 * s3 * y + 4 * s4 * r2 * y
        return xd, yd, (j11, j12, j21, j22)

    def _pinhole_inverse(self, xd: np.ndarray, yd: np.ndarray, iterations: int) -> np.ndarray:
        x = np.array(xd, dtype=np.float64, copy=True)
        y = np.array(yd, dtype=np.float64, copy=True)
        if not np.any(self._coeffs[:12]):
            return np.stack([x, y], axis=-1)
        with np.errstate(all="ignore"):
            for _ in range(iterations):
                fx_, fy_, (j11, j12, j21, j22) = self._pinhole_forward(x, y, jacobian=True)
                ex, ey = fx_ - xd, fy_ - yd
                det = j11 * j22 - j12 * j21
                dx = (j22 * ex - j12 * ey) / det
                dy = (-j21 * ex + j11 * ey) / det
                x = x - dx
                y = y - dy
                if np.nanmax(np.abs(dx) + np.abs(dy), initial=0.0) < 1e-15:
                    break
            fx_, fy_, _ = self._pinhole_forward(x, y)
            residual = np.hypot(fx_ - xd, fy_ - yd)
        bad = ~(residual < 1e-9) | ~(np.hypot(x, y) < self._max_radius)
        out = np.stack([x, y], axis=-1)
        out[bad] = np.nan
        return out

    def _fisheye_theta_d(self, theta: np.ndarray) -> np.ndarray:
        k1, k2, k3, k4 = self._coeffs
        t2 = theta * theta
        return theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    def _fisheye_forward(self, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.arctan(r)
        theta_d = self._fisheye_theta_d(theta)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(r > 1e-12, theta_d / np.where(r > 1e-12, r, 1.0), 1.0)
        return x * scale, y * scale

    def _fisheye_inverse(self, xd: np.ndarray, yd: np.ndarray, iterations: int) -> np.ndarray:
        k1, k2, k3, k4 = self._coeffs
        theta_d = np.hypot(xd, yd)
        theta = theta_d.copy()
        with np.errstate(all="ignore"):
            for _ in range(iterations):
                t2 = theta * theta
                f = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d
                df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + 9 * k4 * t2)))
                step = f / df
                theta = theta - step
                if np.nanmax(np.abs(step), initial=0.0) < 1e-15:
                    break
            residual = np.abs(self._fisheye_theta_d(theta) - theta_d)
            r = np.tan(theta)
            scale = np.where(theta_d > 1e-12, r / np.where(theta_d > 1e-12, theta_d, 1.0), 1.0)
        out = np.stack([xd * scale, yd * scale], axis=-1)
        bad = ~(residual < 1e-9) | ~(theta >= 0) | ~(theta < math.atan(self._max_radius))
        out[bad] = np.nan
        return out


@functools.lru_cache(maxsize=64)
def _max_radius_for(model: str, coeffs: tuple[float, ...]) -> float:
    """Normalized radius where the radial distortion map stops increasing (cached per model)."""
    probe = Camera.__new__(Camera)
    probe.model = model
    probe._coeffs = np.array(coeffs)
    if model == "fisheye":
        theta = np.linspace(0.0, math.pi / 2, 20001)[1:]
        d = np.diff(probe._fisheye_theta_d(np.concatenate([[0.0], theta])))
        stop = np.nonzero(d <= 0)[0]
        if stop.size == 0:
            return 1e6
        return float(math.tan(theta[max(stop[0] - 1, 0)]))
    if not any(coeffs[:12]):
        return math.inf
    r = np.concatenate([np.linspace(0.0, 5.0, 50001)[1:], np.linspace(5.0, 1000.0, 20000)[1:]])
    rd = []
    for angle in np.linspace(0.0, 2 * math.pi, 16, endpoint=False):
        x, y = r * math.cos(angle), r * math.sin(angle)
        xd, yd, _ = probe._pinhole_forward(x, y)
        rd.append(np.hypot(xd, yd))
    rd = np.min(np.stack(rd), axis=0)
    with np.errstate(invalid="ignore"):
        stop = np.nonzero(~(np.diff(rd) > 0))[0]
    if stop.size == 0:
        return float(r[-1])
    return float(r[stop[0]])


def _apply_homography(H: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    w = np.where(w != 0, w, 1.0)
    return (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w, (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w


# =============================================================================================
# SO(3) / SE(3)
# =============================================================================================


def make_T(R: ArrayLike | None = None, t: ArrayLike | None = None) -> np.ndarray:
    """4x4 transform(s) from rotation ``(..., 3, 3)`` and translation ``(..., 3)`` (either may be None)."""
    R_ = np.eye(3) if R is None else np.asarray(R, dtype=np.float64)
    t_ = np.zeros(3) if t is None else np.asarray(t, dtype=np.float64)
    batch = np.broadcast_shapes(R_.shape[:-2], t_.shape[:-1])
    T = np.zeros(batch + (4, 4))
    T[..., :3, :3] = R_
    T[..., :3, 3] = t_
    T[..., 3, 3] = 1.0
    return T


def inv_T(T: ArrayLike) -> np.ndarray:
    """Inverse of rigid transform(s) ``(..., 4, 4)``."""
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, T[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def transform(T: ArrayLike, X: ArrayLike) -> np.ndarray:
    """Apply ``T`` ``(..., 4, 4)`` to points ``(..., 3)`` (leading dimensions broadcast).

    For one transform and many points use ``transform(T, X)`` with ``T`` (4, 4) and ``X`` (N, 3);
    for per-frame transforms ``T`` (F, 4, 4) and points (F, J, 3) use ``transform(T[:, None], X)``.
    """
    T = np.asarray(T, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    return np.einsum("...ij,...j->...i", T[..., :3, :3], X) + T[..., :3, 3]


def rotate(R: ArrayLike, v: ArrayLike) -> np.ndarray:
    """Rotate vectors ``(..., 3)`` by ``R`` ``(..., 3, 3)`` (broadcasting)."""
    return np.einsum("...ij,...j->...i", np.asarray(R, dtype=np.float64), np.asarray(v, dtype=np.float64))


def skew(w: ArrayLike) -> np.ndarray:
    """``(..., 3)`` -> skew-symmetric ``(..., 3, 3)`` with ``skew(a) @ b == cross(a, b)``."""
    w = np.asarray(w, dtype=np.float64)
    S = np.zeros(w.shape[:-1] + (3, 3))
    S[..., 0, 1], S[..., 0, 2] = -w[..., 2], w[..., 1]
    S[..., 1, 0], S[..., 1, 2] = w[..., 2], -w[..., 0]
    S[..., 2, 0], S[..., 2, 1] = -w[..., 1], w[..., 0]
    return S


def so3_exp(w: ArrayLike) -> np.ndarray:
    """Rotation vector (axis * angle, radians) ``(..., 3)`` -> rotation matrix ``(..., 3, 3)``."""
    w = np.asarray(w, dtype=np.float64)
    theta = np.linalg.norm(w, axis=-1)[..., None, None]
    K = skew(w)
    small = theta < 1e-8
    safe = np.where(small, 1.0, theta)
    a = np.where(small, 1.0 - theta**2 / 6.0, np.sin(safe) / safe)
    b = np.where(small, 0.5 - theta**2 / 24.0, (1.0 - np.cos(safe)) / safe**2)
    return np.eye(3) + a * K + b * (K @ K)


def so3_log(R: ArrayLike) -> np.ndarray:
    """Rotation matrix ``(..., 3, 3)`` -> rotation vector ``(..., 3)`` (angle in [0, pi])."""
    q = mat_to_quat(R)
    w = np.clip(q[..., 0], -1.0, 1.0)
    v = q[..., 1:]
    n = np.linalg.norm(v, axis=-1)
    angle = 2.0 * np.arctan2(n, w)
    small = n < 1e-12
    scale = np.where(small, 2.0 / np.where(np.abs(w) > 0, w, 1.0), angle / np.where(small, 1.0, n))
    return v * scale[..., None]


def rot_x(angle: ArrayLike) -> np.ndarray:
    a = np.asarray(angle, dtype=np.float64)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    R[..., 0, 0] = 1
    R[..., 1, 1], R[..., 1, 2], R[..., 2, 1], R[..., 2, 2] = c, -s, s, c
    return R


def rot_y(angle: ArrayLike) -> np.ndarray:
    a = np.asarray(angle, dtype=np.float64)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    R[..., 1, 1] = 1
    R[..., 0, 0], R[..., 0, 2], R[..., 2, 0], R[..., 2, 2] = c, s, -s, c
    return R


def rot_z(angle: ArrayLike) -> np.ndarray:
    a = np.asarray(angle, dtype=np.float64)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    R[..., 2, 2] = 1
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1] = c, -s, s, c
    return R


def quat_to_mat(q: ArrayLike) -> np.ndarray:
    """Quaternion(s) ``[w, x, y, z]`` ``(..., 4)`` (normalized here) -> ``(..., 3, 3)``."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def mat_to_quat(R: ArrayLike) -> np.ndarray:
    """Rotation matrix ``(..., 3, 3)`` -> unit quaternion ``[w, x, y, z]`` with ``w >= 0``."""
    R = np.asarray(R, dtype=np.float64)
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    trace = m00 + m11 + m22
    cand = np.stack(
        [
            np.stack([1 + trace, R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1),
            np.stack([R[..., 2, 1] - R[..., 1, 2], 1 + m00 - m11 - m22, R[..., 0, 1] + R[..., 1, 0], R[..., 0, 2] + R[..., 2, 0]], -1),
            np.stack([R[..., 0, 2] - R[..., 2, 0], R[..., 0, 1] + R[..., 1, 0], 1 - m00 + m11 - m22, R[..., 1, 2] + R[..., 2, 1]], -1),
            np.stack([R[..., 1, 0] - R[..., 0, 1], R[..., 0, 2] + R[..., 2, 0], R[..., 1, 2] + R[..., 2, 1], 1 - m00 - m11 + m22], -1),
        ],
        axis=-2,
    )  # (..., 4, 4): row i is 4 * q * q[i]
    pick = np.argmax(np.stack([trace, m00, m11, m22], -1), axis=-1)
    q = np.take_along_axis(cand, pick[..., None, None], axis=-2)[..., 0, :]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., :1] < 0, -q, q)


def rot_from_two_vectors(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """The smallest rotation taking direction ``a`` to direction ``b`` (``(..., 3)`` each)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    a, b = np.broadcast_arrays(a, b)
    v = np.cross(a, b)
    c = np.sum(a * b, axis=-1)
    s = np.linalg.norm(v, axis=-1)
    angle = np.arctan2(s, c)
    # Axis: v/|v|; for (anti)parallel vectors any axis perpendicular to a.
    helper = np.where(np.abs(a[..., :1]) < 0.9, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    perp = np.cross(a, helper)
    perp = perp / np.linalg.norm(perp, axis=-1, keepdims=True)
    axis = np.where((s > 1e-12)[..., None], v / np.where(s > 1e-12, s, 1.0)[..., None], perp)
    return so3_exp(axis * angle[..., None])


def look_at(eye: ArrayLike, target: ArrayLike, up: ArrayLike = (0.0, 0.0, 1.0)) -> np.ndarray:
    """``T_world_cam`` of an OpenCV camera at ``eye`` looking at ``target``; image "up" (-y) is
    as close to ``up`` as possible. Raises when looking along ``up``."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    z = target - eye
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    x = np.cross(z, up)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(n < 1e-9):
        raise ValueError("look_at: viewing direction is parallel to up")
    x = x / n
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=-1)
    return make_T(R, eye)


def rotation_angle(Ra: ArrayLike, Rb: ArrayLike) -> np.ndarray:
    """Geodesic angle (radians) between rotations ``Ra`` and ``Rb`` ``(..., 3, 3)``."""
    Ra = np.asarray(Ra, dtype=np.float64)
    Rb = np.asarray(Rb, dtype=np.float64)
    R = np.swapaxes(Ra, -1, -2) @ Rb
    return np.linalg.norm(so3_log(R), axis=-1)


def project_to_so3(M: ArrayLike) -> np.ndarray:
    """Nearest rotation matrix (SVD) to ``M`` ``(..., 3, 3)``."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=np.float64))
    D = np.ones(U.shape[:-1])
    D[..., -1] = np.sign(np.linalg.det(U @ Vt))
    return (U * D[..., None, :]) @ Vt


def yaw_of(R_world_cam: ArrayLike) -> np.ndarray:
    """Heading (radians, about world +z, 0 = +x) of a camera's optical axis projected onto the
    horizontal plane. Falls back to the camera's up vector (-y) when looking straight down/up."""
    R = np.asarray(R_world_cam, dtype=np.float64)
    f = R[..., :, 2]
    up = -R[..., :, 1]
    use = np.where((np.hypot(f[..., 0], f[..., 1]) > 1e-3)[..., None], f, np.where((f[..., 2:3] < 0), up, -up))
    return np.arctan2(use[..., 1], use[..., 0])


def gravity_align(R_world_cam: ArrayLike) -> np.ndarray:
    """Rotation about world z that brings the first camera's heading to +x (see section 1)."""
    return rot_z(-yaw_of(R_world_cam))


def as_sequence(x: ArrayLike) -> Sequence[float]:
    return [float(v) for v in np.asarray(x, dtype=np.float64).reshape(-1)]
