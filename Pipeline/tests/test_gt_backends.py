import math

import cv2
import numpy as np
import pytest

from glassesdata_pipeline.backends import gt as gtb
from glassesdata_pipeline.episode import Episode
from glassesdata_pipeline.errors import PipelineError
from glassesdata_pipeline.groundtruth import unpack_mask


@pytest.fixture(scope="module")
def ep(synth_recording):
    return Episode(synth_recording)


def test_hands2d_shapes_noise_and_visibility(ep):
    out = gtb.hands2d(ep, ep.config)
    T = ep.gt.n_frames
    assert out["kp"].shape == (T, 2, 21, 3) and out["hand_score"].shape == (T, 2) and out["mp_world"].shape == (T, 2, 21, 3)
    uv = ep.camera().project(ep.gt["joints_cam"])
    vis = ep.gt["visible"]
    detected = np.isfinite(out["kp"][..., 0])
    assert not np.any(detected & ~vis)  # invisible joints are NaN
    res = (out["kp"][..., :2] - uv)[detected]
    assert 1.6 < res.std() < 2.4 and abs(res.mean()) < 0.3  # sigma 2 px
    assert np.all(out["kp"][..., 2][detected] == 0.9)
    has = np.isfinite(out["hand_score"])
    assert np.all(np.any(detected, axis=-1) == has)
    eligible = vis.sum(-1) >= 6
    assert 0.8 < has[eligible].mean() < 1.0  # 5 % random dropouts
    centre = ep.gt["joints_cam"][..., [0, 5, 9, 13, 17], :].mean(axis=-2, keepdims=True)
    err = (out["mp_world"] - (ep.gt["joints_cam"] - centre))[has]
    assert np.nanstd(err) < 0.005
    again = gtb.hands2d(ep, ep.config)
    assert np.array_equal(out["kp"], again["kp"], equal_nan=True)  # reproducible


def test_hands2d_config_changes_noise(ep):
    cfg = ep.config.merged({"synth": {"gt_noise": {"hands2d_sigma_px": 0.0, "hands2d_dropout": 0.0, "mp_world_sigma_m": -1}}})
    out = gtb.hands2d(ep, cfg)
    uv = ep.camera().project(ep.gt["joints_cam"])
    ok = np.isfinite(out["kp"][..., 0])
    assert np.allclose(out["kp"][..., :2][ok], uv[ok])
    assert np.all(np.isnan(out["mp_world"]))
    eligible = ep.gt["visible"].sum(-1) >= 6
    assert np.all(np.isfinite(out["hand_score"])[eligible])


def test_depth_scale_drift_and_noise(ep):
    cfg = ep.config.merged({"depth": {"downsample": 4}})
    out = gtb.depth(ep, cfg, frames=range(0, 48, 6))
    assert out["depth"].shape == (8, 80, 45) and out["depth"].dtype == np.float16
    assert list(out["frames"]) == list(range(0, 48, 6)) and math.isclose(out["scale"], 0.25)
    truth = gtb.depth_scale_truth(ep, cfg)
    full = ep.gt.depth()
    for row, f in enumerate(out["frames"]):
        g = cv2.resize(np.asarray(full[f], np.float32), (45, 80), interpolation=cv2.INTER_AREA)
        ratio = out["depth"][row].astype(np.float64) / g
        assert abs(np.median(ratio) - truth[f]) < 0.01
        assert 0.025 < np.std(ratio / truth[f]) < 0.035
    assert np.isclose(truth.mean(), 1.10, atol=0.06) and truth.max() - truth.min() > 0.01
    one = dict(gtb.depth_frames(ep, cfg, frames=[12]))[12]
    assert np.array_equal(one, out["depth"][2])  # noise depends on the frame only
    clean = gtb.depth(ep, ep.config.merged({"depth": {"downsample": 1}, "synth": {"gt_noise": {
        "depth_scale": 1.0, "depth_drift": 0.0, "depth_noise": 0.0}}}), frames=[5])
    assert np.allclose(clean["depth"][0].astype(np.float64), full[5].astype(np.float64), rtol=1e-3)


def test_masks_at_depth_resolution(ep):
    cfg = ep.config.merged({"depth": {"downsample": 4}})
    out = gtb.masks(ep, cfg)
    assert set(out) == {"box_a", "box_b"}
    gt_masks = ep.gt.masks()
    for k, obj in enumerate(("box_a", "box_b")):
        item = out[obj]
        assert item["shape"] == (80, 45) and item["packed"].shape[1] == (80 * 45 + 7) // 8
        assert len(item["frames"]) > 40
        f = int(item["frames"][len(item["frames"]) // 2])
        row = list(item["frames"]).index(f)
        small = unpack_mask(item["packed"][row], item["shape"])
        full = unpack_mask(gt_masks["packed"][f, k], gt_masks["shape"])
        assert abs(small.sum() * 16 - full.sum()) < 0.35 * full.sum() + 40
        x0, y0, x1, y1 = item["prompt"]["bbox"]
        first = unpack_mask(item["packed"][0], item["shape"])
        ys, xs = np.nonzero(first)
        assert (xs.min(), ys.min(), xs.max(), ys.max()) == (x0, y0, x1, y1)


def test_gt_backends_need_ground_truth(tmp_path):
    (tmp_path / "rec").mkdir()
    with pytest.raises(PipelineError, match="synthetic"):
        gtb.hands2d(Episode(tmp_path / "rec"))
