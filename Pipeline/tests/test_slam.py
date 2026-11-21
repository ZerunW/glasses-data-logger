"""Step ``slam`` (builtin visual odometry + pose graph, import backend) and core.vo / core.pose_graph."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/fabricate.py (pytest runs in importlib mode)
from fabricate import fabricate_upstream  # noqa: E402
from glassesdata_pipeline import geometry as geo
from glassesdata_pipeline.core import align, pose_graph as pg, vo
from glassesdata_pipeline.episode import Episode
from glassesdata_pipeline.runner import Runner
from glassesdata_pipeline.steps import slam


@pytest.fixture(scope="module")
def slam_rec(synth_recording, tmp_path_factory):
    dst = tmp_path_factory.mktemp("slam") / "rec"
    shutil.copytree(synth_recording, dst, ignore=shutil.ignore_patterns("derived"))
    fabricate_upstream(dst)
    report = Runner(Episode(dst)).run(steps=["slam"], deps=False)
    assert report.ok, report.table()
    return dst


# ---------------------------------------------------------------- core.pose_graph


def _circle_poses(n=24, radius=1.0):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return geo.make_T(geo.rot_z(ang), np.stack([radius * np.cos(ang), radius * np.sin(ang), 0.1 * np.sin(3 * ang)], -1))


def test_pose_graph_consistent_edges_are_exact():
    T = _circle_poses()
    edges = [pg.Edge(i, i + 1, geo.inv_T(T[i]) @ T[i + 1], np.eye(6)) for i in range(len(T) - 1)]
    init = T.copy()
    init[1:, :3, 3] += 0.05  # perturbed start
    init[1:, :3, :3] = init[1:, :3, :3] @ geo.so3_exp(np.array([0.02, -0.01, 0.03]))
    out, rep = pg.optimize(init, edges, fixed=(0,))
    assert np.allclose(out, T, atol=1e-6)
    assert rep["cost"] < 1e-10 < rep["cost0"]


def test_pose_graph_loop_edges_reduce_drift():
    rng = np.random.default_rng(0)
    T = _circle_poses(30)
    edges = []
    odo = [T[0]]
    for i in range(len(T) - 1):
        Z = geo.inv_T(T[i]) @ T[i + 1]
        Z = Z @ geo.make_T(geo.so3_exp(rng.normal(scale=0.01, size=3)), rng.normal(scale=0.01, size=3))
        edges.append(pg.Edge(i, i + 1, Z, pg.information(100, np.linalg.norm(Z[:3, 3]))))
        odo.append(odo[-1] @ Z)
    odo = np.stack(odo)
    for i in range(0, len(T) - 5, 3):  # accurate edges to keyframes 5 back
        edges.append(pg.Edge(i, i + 5, geo.inv_T(T[i]) @ T[i + 5], np.eye(6) * 1e6))
    edges.append(pg.Edge(len(T) - 1, 0, geo.inv_T(T[-1]) @ T[0], np.eye(6) * 1e6))  # loop closure
    out, _ = pg.optimize(odo, edges, fixed=(0,))
    err_odo = np.linalg.norm(odo[:, :3, 3] - T[:, :3, 3], axis=1).max()
    err_opt = np.linalg.norm(out[:, :3, 3] - T[:, :3, 3], axis=1).max()
    assert err_opt < 0.25 * err_odo
    assert err_opt < 0.05  # the odometry edges carry 1 cm / 0.6 deg noise each
    e = pg.edge_errors(out, edges)
    assert e["rot_deg"].shape == (len(edges),)


# ---------------------------------------------------------------- core.vo


def test_pnp_recovers_pose_with_outliers():
    rng = np.random.default_rng(1)
    X = rng.uniform([-1, -1, 2], [1, 1, 4], size=(200, 3))
    T_true = geo.make_T(geo.so3_exp([0.05, -0.1, 0.02]), [0.1, -0.05, 0.2])
    Xc = geo.transform(T_true, X)
    xy = Xc[:, :2] / Xc[:, 2:] + rng.normal(scale=0.5 / 300, size=(200, 2))
    xy[:60] += rng.uniform(-0.2, 0.2, size=(60, 2))  # 30 % outliers
    ok, T, inl = vo.pnp(X, xy, thresh=2.0 / 300)
    assert ok
    assert inl[60:].mean() > 0.95 and inl[:60].mean() < 0.1
    assert np.degrees(geo.rotation_angle(T[:3, :3], T_true[:3, :3])) < 0.2
    assert np.linalg.norm(T[:3, 3] - T_true[:3, 3]) < 0.01


def test_sample_depth_bilinear_and_edges():
    d = np.full((40, 60), 2.0, np.float32)
    d[:, 30:] = 1.0  # depth step at u = 29.5
    d[5, 5] = np.nan
    dm = vo.prepare_depth(d, median=0, edge_rel=0.08)
    z = vo.sample_depth(dm, np.array([[10.5, 20.2], [29.6, 20.0], [50.0, 10.0], [5.2, 5.1], [-1.0, 3.0]]))
    assert z[0] == pytest.approx(2.0) and z[2] == pytest.approx(1.0)
    assert np.isnan(z[1]) and np.isnan(z[3]) and np.isnan(z[4])
    noisy = (2.0 * (1 + np.random.default_rng(2).normal(scale=0.03, size=d.shape))).astype(np.float32)
    dmn = vo.prepare_depth(noisy, median=3, edge_rel=0.08)
    assert dmn.ok[5:-5, 5:-5].mean() > 0.9  # pixel noise is not taken for an edge


def test_hand_mask_and_grid_mapping():
    kp = np.full((2, 21, 3), np.nan)
    rng = np.random.default_rng(3)
    kp[1, :, :2] = rng.uniform([300, 500], [400, 600], size=(21, 2))
    kp[1, 0, :2] = (350, 640)      # wrist
    kp[1, 9, :2] = (350, 520)      # middle knuckle: the forearm extends downwards
    kp[1, :, 2] = 0.9
    m = vo.hand_mask(kp, (720, 1280), (360, 640), margin_px=4, forearm=1.5)
    uv = vo.grid_to_grid(kp[1, :, :2], (720, 1280), (360, 640))
    assert np.all(m[np.round(uv[:, 1]).astype(int), np.round(uv[:, 0]).astype(int)] == 255)
    assert m[int(round(vo.grid_to_grid(np.array([350.0, 760.0]), (720, 1280), (360, 640))[1])), 175] == 255  # forearm
    assert m[10, 10] == 0
    assert np.allclose(vo.grid_to_grid(np.array([[0.0, 0.0]]), (720, 1280), (360, 640)), [[-0.25, -0.25]])


def test_klt_track_subpixel_shift():
    rng = np.random.default_rng(4)
    img = cv2.GaussianBlur((rng.random((120, 160)) * 255).astype(np.uint8), (5, 5), 1.5)
    M = np.float32([[1, 0, 2.3], [0, 1, -1.1]])
    moved = cv2.warpAffine(img, M, (160, 120), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    pts = vo.detect(img, vo.border_mask((160, 120), 20), max_corners=50, min_distance=8, cell_px=40, per_cell=10)
    assert len(pts) >= 10
    new, ok = vo.track(img, moved, pts, win=15, levels=2, fb_thresh=0.5)
    assert ok.mean() > 0.8
    assert np.allclose(np.median(new[ok] - pts[ok], axis=0), [2.3, -1.1], atol=0.1)


# ---------------------------------------------------------------- step


def test_slam_step_on_synthetic(slam_rec):
    ep = Episode(slam_rec)
    d = ep.load_npz("slam")
    T = d["T_slam_cam"]
    valid = d["valid"]
    n = ep.gt.n_frames
    assert T.shape == (n, 4, 4) and valid.shape == (n,) and d["inliers"].shape == (n,)
    assert d["source"] == "builtin" and d["is_keyframe"].sum() >= 2
    assert valid.mean() > 0.9
    first = int(np.flatnonzero(valid)[0])
    assert np.allclose(T[first], np.eye(4))
    assert np.all(np.isnan(T[~valid]))
    gt = ep.gt["T_world_cam"]
    se3 = align.ate(T, gt, align="se3", valid=valid)
    sim3 = align.ate(T, gt, align="sim3", valid=valid)
    assert se3["rmse"] < 0.02            # 2 s, 180x320, fast motion: ~1-2 mm in practice
    assert abs(sim3["scale"] - 1) < 0.05  # metric (depth x hand3d.depth_scale_smooth)
    rpe = align.rpe(T, gt, delta=1, valid=valid, align_scale=False)
    assert rpe["rot_rmse_deg"] < 0.5
    q = json.loads(d["quality"])
    assert q["depth_scale"].startswith("hand3d") and q["keyframes"] >= 2 and q["edges"] >= 1


def test_depth_scale_fallback(synth_recording, tmp_path, caplog):
    dst = tmp_path / "rec"
    shutil.copytree(synth_recording, dst, ignore=shutil.ignore_patterns("derived"))
    fabricate_upstream(dst, write=("hands2d", "depth"))
    ep = Episode(dst)
    with caplog.at_level(logging.WARNING):
        s, src = slam._depth_scale(ep, ep.gt.n_frames)
    assert src == "none" and np.all(s == 1.0)
    assert any("depth_scale" in r.message for r in caplog.records)


def test_import_backend_frames_auto_scale(slam_rec, tmp_path):
    ep = Episode(slam_rec)
    gt = ep.gt["T_world_cam"]
    n = len(gt)
    # A foreign world frame and scale, keyframes only (every 3rd frame), stamps = frame indices.
    G = geo.make_T(geo.so3_exp([0.3, -1.2, 0.5]), [1.0, 2.0, -0.5])
    poses = G @ gt
    poses[:, :3, 3] *= 2.7
    keep = np.arange(0, n, 3)
    path = tmp_path / "traj.txt"
    align.write_tum(path, keep.astype(float), poses[keep])
    p = slam.params(ep.config)
    p.update(backend="import", import_path=str(path), import_time="auto", import_scale="auto")
    res = slam.run_import(ep, p)
    v = res["valid"]
    assert v[: keep[-1] + 1].all()                           # interpolated between keyframes
    assert np.allclose(res["T_slam_cam"][0], np.eye(4), atol=1e-9)
    assert abs(res["quality"]["import_scale"] - 1 / 2.7) < 0.02
    assert align.ate(res["T_slam_cam"], gt, align="se3", valid=v)["rmse"] < 0.01
    # seconds stamps and a numeric scale
    t = ep.timeline()["t"]
    align.write_tum(path, t, poses)
    p.update(import_time="seconds", import_scale=1 / 2.7)
    res = slam.run_import(ep, p)
    assert res["valid"].all()
    rel = geo.inv_T(gt[0]) @ gt
    assert np.allclose(res["T_slam_cam"][:, :3, 3], rel[:, :3, 3], atol=1e-6)
