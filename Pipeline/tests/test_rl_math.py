"""xp-generic quaternion / array helpers (rl/xmath.py) checked with numpy against geometry.py."""

from __future__ import annotations

import numpy as np
import pytest

from glassesdata_pipeline import geometry as geo
from glassesdata_pipeline.rl import xmath as xm


def random_quats(n: int, seed: int = 0) -> np.ndarray:
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def test_quat_to_mat_matches_geometry():
    q = random_quats(20)
    np.testing.assert_allclose(xm.quat_to_mat(np, q), geo.quat_to_mat(q), atol=1e-12)


def test_quat_mul_composes_rotations():
    a, b = random_quats(10, 1), random_quats(10, 2)
    np.testing.assert_allclose(xm.quat_to_mat(np, xm.quat_mul(np, a, b)),
                               geo.quat_to_mat(a) @ geo.quat_to_mat(b), atol=1e-12)


def test_rotate_and_inverse():
    q = random_quats(10, 3)
    v = np.random.default_rng(4).normal(size=(10, 3))
    np.testing.assert_allclose(xm.quat_rotate(np, q, v), np.einsum("nij,nj->ni", geo.quat_to_mat(q), v), atol=1e-12)
    np.testing.assert_allclose(xm.quat_rotate_inv(np, q, xm.quat_rotate(np, q, v)), v, atol=1e-12)


def test_rotvec_roundtrip_and_small_angles():
    r = np.random.default_rng(5).normal(size=(20, 3))
    r = r / np.linalg.norm(r, axis=-1, keepdims=True) * np.linspace(0.0, 3.0, 20)[:, None]
    q = xm.quat_from_rotvec(np, r)
    np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(xm.quat_to_rotvec(np, q), r, atol=1e-9)
    np.testing.assert_allclose(xm.quat_to_mat(np, q), geo.so3_exp(r), atol=1e-12)
    tiny = np.array([[1e-9, 0.0, 0.0]])
    np.testing.assert_allclose(xm.quat_to_rotvec(np, xm.quat_from_rotvec(np, tiny)), tiny, atol=1e-15)
    # q and -q are the same rotation
    np.testing.assert_allclose(xm.quat_to_rotvec(np, -q), r, atol=1e-9)


def test_quat_angle_matches_geometry():
    a, b = random_quats(15, 6), random_quats(15, 7)
    np.testing.assert_allclose(xm.quat_angle(np, a, b), geo.rotation_angle(geo.quat_to_mat(a), geo.quat_to_mat(b)), atol=1e-9)
    np.testing.assert_allclose(xm.quat_angle(np, a, -a), 0.0, atol=1e-7)


def test_rot6d_roundtrip():
    q = random_quats(10, 8)
    r6 = xm.quat_to_rot6d(np, q)
    assert r6.shape == (10, 6)
    R = geo.quat_to_mat(q)
    np.testing.assert_allclose(r6[:, :3], R[:, :, 0], atol=1e-12)
    np.testing.assert_allclose(r6[:, 3:], R[:, :, 1], atol=1e-12)
    np.testing.assert_allclose(xm.rot6d_to_mat(np, r6), R, atol=1e-12)
    # continuity: q and -q give the same 6D vector
    np.testing.assert_allclose(xm.quat_to_rot6d(np, -q), r6, atol=1e-12)


def test_array_wrappers_numpy():
    a = np.arange(6.0).reshape(2, 3)
    assert xm.cat(np, [a, a], -1).shape == (2, 6)
    assert xm.stack(np, [a, a], 0).shape == (2, 2, 3)
    np.testing.assert_allclose(xm.clip(np, a, 1.0, 4.0), np.clip(a, 1, 4))
    np.testing.assert_allclose(xm.norm(np, a), np.linalg.norm(a, axis=-1))
    np.testing.assert_array_equal(xm.nonzero(np, np.array([False, True, True])), [1, 2])
    bad = np.array([[1.0, np.nan], [np.inf, 2.0], [1.0, 2.0]])
    np.testing.assert_array_equal(xm.isfinite_rows(np, bad), [False, False, True])
    np.testing.assert_array_equal(xm.nan_to_zero(np, bad), [[1.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
    assert xm.exp_kernel(np, np.array(0.05), 0.05) == pytest.approx(np.exp(-1.0))


def test_rng_reproducible():
    a = xm.Rng(np, 3).uniform((5,), 2.0, 4.0)
    b = xm.Rng(np, 3).uniform((5,), 2.0, 4.0)
    np.testing.assert_array_equal(a, b)
    assert ((a >= 2.0) & (a < 4.0)).all()
    assert xm.Rng(np, 3).normal((4, 2)).shape == (4, 2)


def test_torch_wrappers_when_available():
    torch = pytest.importorskip("torch")
    q = torch.as_tensor(random_quats(8, 9))
    v = torch.as_tensor(np.random.default_rng(1).normal(size=(8, 3)))
    np.testing.assert_allclose(xm.quat_rotate(torch, q, v).numpy(), xm.quat_rotate(np, q.numpy(), v.numpy()), atol=1e-12)
    np.testing.assert_allclose(xm.quat_to_rot6d(torch, q).numpy(), xm.quat_to_rot6d(np, q.numpy()), atol=1e-12)
    r = torch.as_tensor(np.random.default_rng(2).normal(size=(8, 3)))
    np.testing.assert_allclose(xm.quat_to_rotvec(torch, xm.quat_from_rotvec(torch, r)).numpy(), r.numpy(), atol=1e-9)
