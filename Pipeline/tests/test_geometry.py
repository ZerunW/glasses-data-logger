import math

import cv2
import numpy as np
import pytest

from glassesdata_pipeline import geometry as geo
from glassesdata_pipeline.errors import PipelineError

K = np.array([[512.0, 0.0, 359.3], [0.0, 509.0, 641.8], [0.0, 0.0, 1.0]])
PINHOLE_DISTS = [
    [-0.05, 0.01, 0.001, -0.002],
    [-0.05, 0.01, 0.001, -0.002, 0.003],
    [0.1, -0.05, 0.001, 0.002, 0.01, 0.05, -0.01, 0.002],
    [0.1, -0.05, 0.001, 0.002, 0.01, 0.05, -0.01, 0.002, 0.001, -0.002, 0.0005, 0.001],
    [0.1, -0.05, 0.001, 0.002, 0.01, 0.05, -0.01, 0.002, 0.001, -0.002, 0.0005, 0.001, 0.02, -0.01],
]


def _points(rng, n=400):
    xy = np.c_[rng.uniform(-0.6, 0.6, n), rng.uniform(-1.1, 1.1, n)]
    z = rng.uniform(0.3, 4.0, n)
    return np.c_[xy * z[:, None], z]


@pytest.mark.parametrize("dist", PINHOLE_DISTS, ids=lambda d: f"{len(d)}coef")
def test_pinhole_matches_opencv_and_round_trips(dist):
    rng = np.random.default_rng(len(dist))
    cam = geo.Camera(K, dist, 720, 1280)
    X = _points(rng)
    uv = cam.project(X)
    ref, _ = cv2.projectPoints(X.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, np.array(dist))
    assert np.max(np.abs(uv - ref.reshape(-1, 2))) < 1e-6
    back = cam.unproject(uv, X[:, 2])
    assert np.max(np.abs(back - X)) < 1e-6
    assert np.max(np.abs(cam.project(back) - uv)) < 1e-6


def test_fisheye_matches_opencv_and_round_trips():
    rng = np.random.default_rng(3)
    D = np.array([0.05, -0.01, 0.002, -0.0005])
    Kf = np.array([[400.0, 0.0, 359.5], [0.0, 400.0, 639.5], [0.0, 0.0, 1.0]])
    cam = geo.Camera(Kf, D, 720, 1280, "fisheye")
    X = np.c_[rng.uniform(-1.5, 1.5, 300), rng.uniform(-2, 2, 300), np.ones(300)] * rng.uniform(0.5, 3, 300)[:, None]
    uv = cam.project(X)
    ref, _ = cv2.fisheye.projectPoints(X.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), Kf, D)
    assert np.max(np.abs(uv - ref.reshape(-1, 2))) < 1e-6
    assert np.max(np.abs(cam.unproject(uv, X[:, 2]) - X)) < 1e-6


def test_undistort_round_trip_over_the_whole_image():
    cam = geo.Camera(K, [-0.05, 0.01, 0.0005, -0.0003, 0.002], 720, 1280)
    u, v = np.meshgrid(np.linspace(0, 719, 37), np.linspace(0, 1279, 65))
    uv = np.stack([u, v], -1)
    xy = cam.undistort_points(uv)
    rays = np.concatenate([xy, np.ones(xy.shape[:-1] + (1,))], -1)
    assert np.max(np.abs(cam.project(rays) - uv)) < 1e-6


def test_behind_camera_and_folded_points_are_nan():
    cam = geo.Camera(K, [-0.3, 0.0, 0.0, 0.0, 0.0], 720, 1280)  # r(1 - 0.3 r^2) folds at r = 1.054
    assert np.isclose(cam.max_radius, 1 / math.sqrt(0.9), rtol=1e-3)
    uv = cam.project(np.array([[0.1, 0.1, -1.0], [1.5, 0.0, 1.0], [0.2, 0.1, 1.0]]))
    assert np.all(np.isnan(uv[:2])) and np.all(np.isfinite(uv[2]))


def test_pixel_rays_hit_their_pixels():
    cam = geo.Camera(K, [-0.05, 0.01, 0.0, 0.0, 0.0], 720, 1280)
    for ds in (1, 4):
        rays = cam.pixel_rays(ds)
        w, h = geo.downsample_size(720, 1280, ds)
        assert rays.shape == (h, w, 3) and np.all(rays[..., 2] == 1)
        assert not rays.flags.writeable
        small = cam.downsampled(ds)
        uv = small.project(rays)
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        assert np.max(np.abs(uv - np.stack([u, v], -1))) < 1e-6
    assert cam.pixel_rays(4) is cam.pixel_rays(4)  # cached


def test_resized_camera_maps_pixel_centres():
    cam = geo.Camera(K, [-0.05, 0.01, 0, 0, 0], 720, 1280)
    small = cam.scaled(0.25)
    assert (small.width, small.height) == (180, 320)
    X = np.array([[0.1, -0.2, 1.5], [0.3, 0.4, 2.0]])
    uv, uv_small = cam.project(X), small.project(X)
    assert np.allclose(uv_small, (uv + 0.5) * 0.25 - 0.5)  # cv2.resize pixel-centre convention


def test_from_dict_scales_and_checks_aspect():
    data = {"version": 1, "model": "pinhole", "width": 720, "height": 1280, "K": K.tolist(),
            "dist": [-0.05, 0.01, 0, 0, 0], "source": "charuco-web"}
    cam = geo.Camera.from_dict(data, 360, 640)
    assert (cam.width, cam.height, cam.source) == (360, 640, "charuco-web")
    assert np.isclose(cam.fx, K[0, 0] / 2)
    assert geo.Camera.from_dict(cam.to_dict()).to_dict() == cam.to_dict()
    with pytest.raises(PipelineError, match="aspect"):
        geo.Camera.from_dict(data, 720, 720)
    with pytest.raises(PipelineError):
        geo.Camera(K, [0.1, 0.2, 0.3], 720, 1280)


def test_fov_default():
    cam = geo.Camera.fov_default(720, 1280, 65.0)
    assert cam.source == "default" and cam.dist.size == 0
    edge = cam.project(np.array([[math.tan(math.radians(32.5)), 0.0, 1.0]]))[0]
    assert np.isclose(edge[0], 719.5)  # right edge of the last pixel
    assert np.isclose(cam.cx, 359.5) and np.isclose(cam.cy, 639.5)


def test_se3_helpers():
    rng = np.random.default_rng(0)
    R = geo.so3_exp(rng.normal(size=(10, 3)))
    t = rng.normal(size=(10, 3))
    T = geo.make_T(R, t)
    assert np.allclose(geo.inv_T(T) @ T, np.eye(4))
    X = rng.normal(size=(10, 5, 3))
    Y = geo.transform(T[:, None], X)
    assert np.allclose(Y, np.einsum("nij,nkj->nki", R, X) + t[:, None])
    assert np.allclose(geo.transform(geo.inv_T(T)[:, None], Y), X)
    q = geo.mat_to_quat(R)
    assert np.all(q[:, 0] >= 0) and np.allclose(np.linalg.norm(q, axis=1), 1)
    assert np.allclose(geo.quat_to_mat(q), R)
    assert np.allclose(geo.quat_to_mat([1, 0, 0, 0]), np.eye(3))
    assert np.allclose(geo.quat_to_mat([0, 0, 0, 1]), geo.rot_z(np.pi))  # [w, x, y, z]
    assert np.allclose(geo.so3_exp(geo.so3_log(R)), R)
    assert np.allclose(geo.so3_exp(geo.so3_log(geo.rot_x(np.pi))), geo.rot_x(np.pi))
    assert np.allclose(geo.rotation_angle(np.eye(3), geo.rot_y(0.3)), 0.3)


def test_rot_from_two_vectors():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(20, 3)), rng.normal(size=(20, 3))
    R = geo.rot_from_two_vectors(a, b)
    assert np.allclose(np.einsum("nij,nj->ni", R, a / np.linalg.norm(a, axis=1, keepdims=True)),
                       b / np.linalg.norm(b, axis=1, keepdims=True))
    for v in ([1, 0, 0], [0, 0, 1]):
        Rv = geo.rot_from_two_vectors(v, np.negative(v))  # antiparallel
        assert np.allclose(Rv @ v, np.negative(v)) and np.isclose(np.linalg.det(Rv), 1)
    assert np.allclose(geo.rot_from_two_vectors([0, 1, 0], [0, 1, 0]), np.eye(3))


def test_look_at_is_opencv():
    T = geo.look_at([0, 0, 1.5], [2, 0, 1.5])
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    assert np.allclose(R[:, 2], [1, 0, 0])  # z forward
    assert np.allclose(R[:, 1], [0, 0, -1])  # y down
    assert np.allclose(R[:, 0], [0, -1, 0])  # x right
    T2 = geo.look_at([0, 0, 1.5], [1, 1, 0.5])
    p = geo.transform(geo.inv_T(T2), np.array([1, 1, 0.5]))
    assert np.allclose(p[:2], 0) and p[2] > 0
    assert np.isclose(geo.yaw_of(T2[:3, :3]), np.pi / 4)
