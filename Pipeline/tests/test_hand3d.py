"""Step hand3d: PnP core, depth-scale smoothing, the step on a synthetic recording with rendered hands,
and the hand3d metrics of eval."""

from __future__ import annotations

import math

import numpy as np
import pytest

from glassesdata_pipeline import geometry as geo
from glassesdata_pipeline import skeleton as sk
from glassesdata_pipeline.backends import gt as gtb
from glassesdata_pipeline.core import depth_scale as ds
from glassesdata_pipeline.core import hand_pnp as hp
from glassesdata_pipeline.episode import Episode
from glassesdata_pipeline.runner import Runner

PROFILE = sk.complete_profile({"hand_length_m": 0.19})
BONES = sk.bone_lengths(PROFILE)


def _posed_hand(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flex = np.clip(rng.normal(0.4, 0.3, (5, 3)), 0, 1.2)
    return sk.pose_hand(PROFILE, "right", np.eye(4), flex, rng.normal(0, 0.1, 5))


# ------------------------------------------------------------------ shape


def test_hand_shape_scale_retarget_and_fallback():
    J = _posed_hand()
    shape, source, factor = hp.hand_shape(J * 1.15, BONES, PROFILE, "right")
    assert source == 1 and np.isclose(factor, 1 / 1.15) and np.allclose(shape, J)
    noisy = J + np.random.default_rng(1).normal(0, 0.003, J.shape)
    shape, _, _ = hp.hand_shape(noisy, BONES, PROFILE, "right", mode="retarget")
    assert np.allclose(hp.bone_lengths(shape), BONES)
    shape, source, factor = hp.hand_shape(np.full((21, 3), np.nan), BONES, PROFILE, "left")
    assert source == 2 and math.isnan(factor) and np.allclose(shape, sk.rest_pose(PROFILE, "left"))


# ------------------------------------------------------------------ pose


def _observe(J, R, t, sigma=0.0, seed=0, f=600.0):
    P = J @ R.T + t
    xy = P[:, :2] / P[:, 2:]
    return xy + np.random.default_rng(seed).normal(0, sigma / f, xy.shape), P


def test_solve_pose_recovers_pose():
    J = _posed_hand(2)
    for seed in range(4):
        R = geo.so3_exp(np.random.default_rng(seed).normal(0, 0.5, 3))
        t = np.array([0.05, 0.1, 0.45])
        xy, P = _observe(J, R, t, sigma=1.0, seed=seed)
        res = hp.solve_pose(J, xy, np.ones(21), 600.0)
        assert res is not None and res.rms_px < 2.0
        assert np.mean(np.linalg.norm(res.joints - P, axis=1)) < 0.006, seed


def test_flip_ambiguity_resolved_by_prior():
    flat = sk.rest_pose(PROFILE, "right")
    flat[:, 2] = 0.0  # exactly planar: the flipped pose explains the image almost as well
    R = geo.so3_exp([0.7, 0.0, 0.2])
    t = np.array([0.0, 0.05, 0.9])
    xy0, P = _observe(flat, R, t)
    Rf, tf = hp.flip_candidate(flat, np.ones(21), R, t)
    Rf, tf, _ = hp.refine_pose(flat, xy0, np.ones(21), Rf, tf, 600.0)
    assert np.degrees(geo.rotation_angle(R, Rf)) > 30
    Pf = flat @ Rf.T + tf
    assert np.sqrt(np.mean(np.sum((Pf[:, :2] / Pf[:, 2:] - xy0) ** 2, axis=1))) * 600 < 3.0
    xy, _ = _observe(flat, R, t, sigma=2.0, seed=3)  # with keypoint noise both poses are plausible
    kw = dict(anchor_px_per_rad=0.0, ambiguity_px=4.0)
    for prior in (R, Rf):
        res = hp.solve_pose(flat, xy, np.ones(21), 600.0, R_prior=prior, **kw)
        assert res.ambiguous and np.degrees(geo.rotation_angle(res.R, prior)) < 10
    # the depth map's relative palm depths pick the right one without any rotation prior
    z_model = P[[5, 9, 13, 17], 2] * 1.3
    res = hp.solve_pose(flat, xy, np.ones(21), 600.0, z_model_palm=z_model, depth_joints=[5, 9, 13, 17], **kw)
    assert np.degrees(geo.rotation_angle(res.R, R)) < 10 and "depth" in res.choice


def test_orientation_anchor_pulls_out_of_plane_rotation():
    J = _posed_hand(3)
    R = geo.so3_exp([0.3, -0.2, 0.1])
    xy, _ = _observe(J, R, np.array([0.0, 0.0, 0.5]), sigma=3.0, seed=5)
    free = hp.solve_pose(J, xy, np.ones(21), 600.0, R_prior=R)
    anchored = hp.solve_pose(J, xy, np.ones(21), 600.0, R_prior=R, anchor_px_per_rad=50.0)
    assert geo.rotation_angle(anchored.R, R) <= geo.rotation_angle(free.R, R) + 1e-9


def test_snap_to_rays():
    J = np.array([[0.01, 0.02, 0.5], [0.0, 0.0, 0.4]])
    xy = np.array([[0.03, 0.04], [np.nan, np.nan]])
    out = hp.snap_to_rays(J, xy, 1.0)
    assert np.allclose(out[0, :2] / out[0, 2], xy[0], atol=1e-12) and np.allclose(out[1], J[1])
    assert np.allclose(hp.snap_to_rays(J, xy, 0.0), J)


# ------------------------------------------------------------------ depth scale


def test_pentadiagonal_solver_matches_dense():
    rng = np.random.default_rng(0)
    n = 30
    w = rng.random(n)
    d0, d1, d2 = ds._second_difference_bands(n)
    A = np.diag(w + 50 * d0) + np.diag(50 * d1, 1) + np.diag(50 * d1, -1) + np.diag(50 * d2, 2) + np.diag(50 * d2, -2)
    b = rng.normal(size=n)
    assert np.allclose(ds.solve_pentadiagonal(w + 50 * d0, 50 * d1, 50 * d2, b), np.linalg.solve(A, b))


def test_smooth_scale_robust_interpolating_and_constant_ends():
    T = 240
    t = np.arange(T) / 24
    truth = 0.9 * (1 + 0.05 * np.sin(2 * np.pi * t / 6))
    rng = np.random.default_rng(0)
    raw = truth * (1 + rng.normal(0, 0.02, T))
    raw[rng.random(T) < 0.1] *= 1.4  # outliers
    raw[:30] = np.nan  # no hands at the start
    raw[100:140] = np.nan  # a gap
    raw[200:] = np.nan  # none at the end
    smooth, info = ds.smooth_scale(raw, cutoff_frames=48)
    assert info["method"] == "whittaker_huber" and info["first"] == 30 and info["last"] == 199
    assert np.max(np.abs(smooth[30:200] / truth[30:200] - 1)) < 0.025
    assert np.all(smooth[:30] == smooth[30]) and np.all(smooth[200:] == smooth[199])
    none, info = ds.smooth_scale(np.full(10, np.nan), default=1.0)
    assert info["method"] == "default" and np.all(none == 1.0)
    few, info = ds.smooth_scale(np.array([np.nan, 0.8, np.nan, 0.9, 1.0]), min_observations=5)
    assert info["method"] == "global_median" and np.allclose(few, 0.9)


def test_sampling_helpers():
    d = np.arange(20, dtype=np.float32).reshape(4, 5) + 1
    d[1, 1] = np.nan
    assert np.isclose(ds.sample_depth(d, np.array([[1.0, 1.0]]), 1)[0], np.median([1, 2, 3, 6, 8, 11, 12, 13]))
    assert np.isnan(ds.sample_depth(d, np.array([[9.0, 1.0]]), 1)[0])
    assert np.allclose(ds.to_grid(np.array([[-0.5, -0.5], [719.5, 1279.5]]), (720, 1280), (180, 320)),
                       [[-0.5, -0.5], [179.5, 319.5]])
    assert np.isclose(ds.frame_ratio(np.array([0.5, 0.52, 0.51]), np.array([0.4, 0.41, 0.4]), np.full(3, 0.014)),
                      np.median((np.array([0.5, 0.52, 0.51]) - 0.014) / np.array([0.4, 0.41, 0.4])))
    assert np.isnan(ds.frame_ratio(np.array([0.5, 0.5]), np.array([0.4, 0.4]), np.zeros(2)))


# ------------------------------------------------------------------ the step on rendered hands


@pytest.fixture(scope="module")
def hands_recording(tmp_path_factory):
    """3 s, 270x480, hands rendered (the gt depth then contains them); hands2d and depth written
    from the ground truth with the gt backends (the files of the contract), ingest run."""
    from glassesdata_pipeline.synth import generate

    rec = tmp_path_factory.mktemp("hand3d") / "rec"
    generate(rec, seconds=3.0, seed=1, size=(270, 480), draw_hands=True)
    ep = Episode(rec)
    assert Runner(ep).run(steps=["ingest"]).ok
    h = gtb.hands2d(ep, ep.config)
    ep.save_npz("hands2d", {"step": "hands2d", "backend": "gt"}, **h)
    d = gtb.depth(ep, ep.config)
    mm = np.lib.format.open_memmap(ep.file("depth/depth_f16.npy"), mode="w+", dtype=np.float16, shape=d["depth"].shape)
    mm[:] = d["depth"]
    mm.flush()
    del mm
    ep.save_npz("depth/index", {"step": "depth", "backend": "gt", "params": {"synth": dict(ep.config["synth"])}},
                frames=d["frames"], scale=d["scale"])
    return rec


def _run_hand3d(rec, overrides=None):
    ep = Episode(rec, overrides)
    report = Runner(ep).run(steps=["hand3d"], deps=False, force=True)
    assert report.ok, report.table()
    return ep, ep.load_npz("hand3d")


def test_hand3d_metric_joints_and_depth_scale(hands_recording):
    ep, d = _run_hand3d(hands_recording)
    gt = ep.gt
    T = gt.n_frames
    assert d["joints_cam"].shape == (T, 2, 21, 3) and d["valid"].shape == (T, 2)
    assert d["depth_scale"].shape == (T,) and np.all(np.isfinite(d["depth_scale_smooth"]))
    V = d["valid"]
    assert V.sum() > 0.8 * np.isfinite(ep.load_npz("hands2d")["hand_score"]).sum()
    assert np.all(np.isnan(d["joints_cam"][~V]))
    err = np.linalg.norm(d["joints_cam"] - gt["joints_cam"], axis=-1)[V]
    assert np.mean(err) < 0.02, np.mean(err)
    dz = (d["joints_cam"][..., 0, 2] - gt["joints_cam"][..., 0, 2])[V]
    assert abs(np.mean(dz)) < 0.006 and np.mean(np.abs(dz)) < 0.015
    truth = 1 / gtb.depth_scale_truth(ep, ep.config)
    rel = d["depth_scale_smooth"] / truth - 1
    assert np.mean(np.abs(rel)) < 0.025, np.mean(np.abs(rel))
    assert d["meta"]["depth_scale_info"]["method"] == "whittaker_huber"
    assert np.all(d["shape_source"][V] == 1) and np.all(d["confidence"][V] > 0)


def test_hand3d_rest_pose_fallback_and_no_depth(hands_recording, tmp_path):
    import shutil

    rec = tmp_path / "rec"
    shutil.copytree(hands_recording, rec)
    ep = Episode(rec)
    h = ep.load_npz("hands2d")
    ep.save_npz("hands2d", {"step": "hands2d", "backend": "gt"}, kp=h["kp"], hand_score=h["hand_score"],
                mp_world=np.full(h["mp_world"].shape, np.nan))
    (ep.derived / "depth" / "depth_f16.npy").unlink()
    (ep.derived / "depth" / "index.npz").unlink()
    from glassesdata_pipeline.steps import hand3d

    hand3d.run(ep, ep.config)  # the runner would (rightly) block without depth outputs
    d = ep.load_npz("hand3d")
    V = d["valid"]
    assert V.any() and np.all(d["shape_source"][V] == 2) and np.all(d["confidence"][V] <= 0.3 + 1e-9)
    assert np.all(np.isnan(d["depth_scale"])) and np.all(d["depth_scale_smooth"] == 1.0)
    err = np.linalg.norm(d["joints_cam"] - ep.gt["joints_cam"], axis=-1)[V]
    assert np.median(err) < 0.1  # a flat template for flexed, grasping hands: coarse (about 7 cm) but metric


def test_eval_reports_hand3d_and_scale_metrics(hands_recording, tmp_path):
    import shutil

    from glassesdata_pipeline.eval.report import evaluate

    rec = tmp_path / "rec"
    shutil.copytree(hands_recording, rec)
    _run_hand3d(rec)
    m = evaluate(Episode(rec))["hand3d"]
    assert m["wrist_depth_abs_error_m"]["mean"] < 0.015 and abs(m["wrist_depth_bias_m"]) < 0.006
    assert m["joint_error_m"]["median"] < 0.02 and m["root_relative_error_m"]["n"] > 0
    assert m["depth_scale"]["smooth_rel_abs"]["mean"] < 0.025 and m["depth_scale"]["raw_frames"] > 0
