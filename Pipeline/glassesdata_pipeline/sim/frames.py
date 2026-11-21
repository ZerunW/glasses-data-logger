"""Frame and camera conventions between the pipeline (PIPELINE.md section 1) and MuJoCo (numpy only).

Pipeline: world z up; camera frame OpenCV (x right, y down, z forward); ``T_a_b`` maps frame b to
frame a; quaternions ``[w, x, y, z]``.

MuJoCo: quaternions are also ``[w, x, y, z]``; body ``pos``/``quat`` give the body frame in its
parent (``T_parent_body``); a ``<camera>`` looks along its **-z** axis with **+x right and +y up**
(MJCF reference, ``body/camera``), the principal point is the image centre and ``fovy`` is the
vertical field of view in degrees.

So a MuJoCo camera frame is the OpenCV camera frame rotated by pi about x:
``p_cv = CV_FROM_MJ @ p_mj`` with ``CV_FROM_MJ = diag(1, -1, -1)``. The scene puts the head mocap
body at ``T_world_cam`` (OpenCV) and its camera at ``quat = QUAT_CV_MJ = [0, 1, 0, 0]`` inside it,
so the MuJoCo camera's world pose is ``T_world_cam @ diag(1, -1, -1)`` (:func:`mujoco_camera_pose`).

:func:`project_mujoco` is a numpy model of MuJoCo's perspective camera in the pipeline's pixel
convention (pixel centres at integer coordinates): ``u = (W - 1)/2 + f x / (-z)``,
``v = (H - 1)/2 - f y / (-z)`` with ``f = (H / 2) / tan(fovy / 2)``. It is what MuJoCo renders,
and the unit tests check it against :meth:`geometry.Camera.project`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .. import geometry as geo

CV_FROM_MJ = np.diag([1.0, -1.0, -1.0])
"""``p_cv = CV_FROM_MJ @ p_mj`` for camera-frame points (its own inverse)."""
QUAT_CV_MJ = np.array([0.0, 1.0, 0.0, 0.0])
"""Quaternion of ``CV_FROM_MJ`` (pi about x): the MuJoCo camera inside an OpenCV-camera body."""


# =============================================================================================
# Poses and quaternions
# =============================================================================================


def pos_quat(T: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(..., 4, 4)`` -> position ``(..., 3)`` and quaternion ``[w, x, y, z]`` ``(..., 4)``."""
    T = np.asarray(T, dtype=np.float64)
    return T[..., :3, 3].copy(), geo.mat_to_quat(T[..., :3, :3])


def pose_from_pos_quat(pos: Any, quat: Any) -> np.ndarray:
    """Inverse of :func:`pos_quat`."""
    return geo.make_T(geo.quat_to_mat(quat), pos)


def quat_mul(a: Any, b: Any) -> np.ndarray:
    """Hamilton product of ``[w, x, y, z]`` quaternions (broadcasting); ``R(a*b) = R(a) R(b)``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def slerp(q0: Any, q1: Any, s: Any) -> np.ndarray:
    """Spherical linear interpolation between unit quaternions (shortest arc), ``s`` in [0, 1]."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    small = theta < 1e-6
    sin_t = np.where(small, 1.0, np.sin(theta))
    w0 = np.where(small, 1.0 - s, np.sin((1.0 - s) * theta) / sin_t)
    w1 = np.where(small, s, np.sin(s * theta) / sin_t)
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def align_hemisphere(q: Any) -> np.ndarray:
    """``(T, 4)`` quaternions with signs flipped so consecutive ones have a positive dot product
    (NaN rows are skipped and kept)."""
    q = np.array(q, dtype=np.float64, copy=True)
    prev = None
    for i in range(len(q)):
        if not np.all(np.isfinite(q[i])):
            continue
        if prev is not None and float(q[i] @ prev) < 0:
            q[i] = -q[i]
        prev = q[i]
    return q


def smooth_series(x: Any, sigma: float, valid: Any = None) -> np.ndarray:
    """Zero-phase Gaussian smoothing along axis 0 (``sigma`` in samples) by normalized convolution:
    NaN or invalid samples get no weight; a sample with no valid neighbour within 3 sigma is NaN.
    ``sigma <= 0`` returns a copy with invalid samples set to NaN."""
    x = np.asarray(x, dtype=np.float64)
    shape = x.shape
    flat = x.reshape(len(x), -1)
    ok = np.all(np.isfinite(flat), axis=1)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    if sigma <= 0 or len(x) == 0:
        out = flat.copy()
        out[~ok] = np.nan
        return out.reshape(shape)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    w = ok.astype(np.float64)
    data = np.where(ok[:, None], flat, 0.0)
    den = np.convolve(w, k, mode="same")
    out = np.empty_like(flat)
    for d in range(flat.shape[1]):
        out[:, d] = np.convolve(data[:, d], k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = out / den[:, None]
    out[den < 1e-9] = np.nan
    return out.reshape(shape)


def smooth_quaternions(q: Any, sigma: float, valid: Any = None) -> np.ndarray:
    """Gaussian smoothing of a ``(T, 4)`` quaternion sequence (hemisphere-aligned average, then
    normalized); fine for the small per-frame rotations of a tracked hand or head."""
    q = align_hemisphere(q)
    out = smooth_series(q, sigma, valid)
    n = np.linalg.norm(out, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return out / n


# =============================================================================================
# Cameras
# =============================================================================================


def mujoco_camera_pose(T_world_cam: Any) -> tuple[np.ndarray, np.ndarray]:
    """World position and ``[w, x, y, z]`` orientation of the MuJoCo camera that sees what the
    OpenCV camera ``T_world_cam`` (``(..., 4, 4)``) sees."""
    T = np.asarray(T_world_cam, dtype=np.float64)
    R = T[..., :3, :3] @ CV_FROM_MJ
    return T[..., :3, 3].copy(), geo.mat_to_quat(R)


def fovy_deg(camera: geo.Camera) -> float:
    """Vertical field of view (degrees) of the pinhole part of ``camera``: ``2 atan((H/2) / fy)``
    (the image spans H pixels of size 1 around the principal point)."""
    return math.degrees(2.0 * math.atan2(camera.height / 2.0, camera.fy))


def mujoco_K(fovy: float, width: int, height: int) -> np.ndarray:
    """Intrinsics (pipeline pixel convention) of a MuJoCo camera rendered at ``width x height``:
    square pixels, principal point at the image centre ``((W-1)/2, (H-1)/2)``."""
    f = (height / 2.0) / math.tan(math.radians(fovy) / 2.0)
    return np.array([[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


def project_mujoco(points_world: Any, cam_pos: Any, cam_quat: Any, fovy: float, width: int, height: int) -> np.ndarray:
    """Pixels ``(..., 2)`` of world points in a MuJoCo camera (pose ``cam_pos``/``cam_quat`` in
    the world, looking along its -z axis, +y up). NaN behind the camera."""
    P = np.asarray(points_world, dtype=np.float64)
    R = geo.quat_to_mat(cam_quat)
    local = (P - np.asarray(cam_pos, dtype=np.float64)) @ R  # R^T (p - c), row vectors
    depth = -local[..., 2]
    f = (height / 2.0) / math.tan(math.radians(fovy) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (width - 1) / 2.0 + f * local[..., 0] / depth
        v = (height - 1) / 2.0 - f * local[..., 1] / depth
    uv = np.stack([u, v], axis=-1)
    uv[~(depth > 1e-9)] = np.nan
    return uv


def mujoco_camera_matching(camera: geo.Camera) -> geo.Camera:
    """The distortion-free :class:`geometry.Camera` that a MuJoCo camera with
    ``fovy = fovy_deg(camera)`` rendered at the same size represents (used to undistort source
    frames onto the sim image grid)."""
    return geo.Camera(mujoco_K(fovy_deg(camera), camera.width, camera.height), [], camera.width, camera.height,
                      "pinhole", camera.source)


def look_at_mujoco(eye: Any, target: Any, up: Any = (0.0, 0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """Position and quaternion of a MuJoCo camera at ``eye`` looking at ``target`` (image up as
    close to ``up`` as possible)."""
    return mujoco_camera_pose(geo.look_at(eye, target, up))


# =============================================================================================
# MJCF number formatting
# =============================================================================================


def fmt(values: Any, digits: int = 6) -> str:
    """Space-separated numbers for MJCF attributes (``%.{digits}g``, ``-0`` printed as ``0``)."""
    arr = np.atleast_1d(np.asarray(values, dtype=np.float64)).reshape(-1)
    out = []
    for v in arr:
        if not math.isfinite(v):
            raise ValueError(f"non-finite value in MJCF attribute: {values!r}")
        text = f"{v:.{digits}g}"
        out.append("0" if text in ("-0", "0", "-0.0") else text)
    return " ".join(out)
