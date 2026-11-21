"""IMU synthesis from a smooth camera trajectory.

Model (documented in ``gt/meta.json``):

* The IMU is rigidly attached at the camera centre (no lever arm); ``R_cam_imu`` maps IMU-frame
  vectors to the camera frame: ``v_cam = R_cam_imu @ v_imu``.
* Time: a sample stamped ``t_imu`` (episode seconds, from ``phone_time_ns``) measures the motion
  at video time ``t_imu - imu_time_offset_s`` (PIPELINE.md: IMU time = video time + offset).
* Gyroscope (rad/s, IMU frame): ``R_cam_imu.T @ omega_cam + gyro_bias + noise``, with ``omega_cam``
  the body angular velocity of the camera.
* Accelerometer (g, IMU frame): ``R_cam_imu.T @ R_world_cam.T @ (a_world - g_world) / G + accel_bias
  + noise`` with ``g_world = (0, 0, -G)``, ``G = 9.80665``: at rest it reads +1 g "up".
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from .. import geometry as geo
from .motion import finite_difference_rates

G = 9.80665


def r_cam_imu(permutation, misalignment_deg) -> np.ndarray:
    """``permutation @ exp(misalignment)``; ``permutation`` must be a proper rotation."""
    P = np.asarray(permutation, dtype=np.float64)
    if not np.allclose(P @ P.T, np.eye(3)) or np.linalg.det(P) < 0:
        raise ValueError("synth.imu.permutation must be a rotation matrix (det +1)")
    return P @ geo.so3_exp(np.radians(np.asarray(misalignment_deg, dtype=np.float64)))


def synthesize(
    pose_fn: Callable[[np.ndarray], np.ndarray],
    t_imu: np.ndarray,
    R_cam_imu: np.ndarray,
    time_offset_s: float,
    gyro_bias: np.ndarray,
    accel_bias: np.ndarray,
    gyro_noise: float,
    accel_noise: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Samples at ``t_imu``: ``acc`` (g), ``gyro`` (rad/s) in the IMU frame, plus the noise-free
    ``omega_cam`` (rad/s) and ``specific_force_cam`` (m/s^2) at the physical times."""
    t_phys = np.asarray(t_imu, dtype=np.float64) - float(time_offset_s)
    omega_cam, acc_world = finite_difference_rates(pose_fn, t_phys)
    R_world_cam = pose_fn(t_phys)[:, :3, :3]
    f_world = acc_world - np.array([0.0, 0.0, -G])
    f_cam = np.einsum("nji,nj->ni", R_world_cam, f_world)
    Rt = np.asarray(R_cam_imu).T
    gyro = omega_cam @ Rt.T + np.asarray(gyro_bias) + rng.normal(scale=gyro_noise, size=omega_cam.shape)
    acc = (f_cam @ Rt.T) / G + np.asarray(accel_bias) + rng.normal(scale=accel_noise, size=f_cam.shape)
    return {"acc": acc, "gyro": gyro, "omega_cam": omega_cam, "specific_force_cam": f_cam}


def integrate_gyro(t: np.ndarray, omega: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Rotation ``R(t0)^T R(t1)`` from body rates ``omega`` sampled at ``t`` (trapezoid on a fine
    grid with linear interpolation). Used by tests and for sanity checks."""
    grid = np.linspace(t0, t1, max(2, int(math.ceil((t1 - t0) * 2000)) + 1))
    w = np.stack([np.interp(grid, t, omega[:, k]) for k in range(3)], axis=-1)
    R = np.eye(3)
    for k in range(len(grid) - 1):
        R = R @ geo.so3_exp(0.5 * (w[k] + w[k + 1]) * (grid[k + 1] - grid[k]))
    return R
