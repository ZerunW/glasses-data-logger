"""Step ``world``: gravity-aligned camera trajectory and smoothed hands in the world."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/fabricate.py (importlib mode)
from fabricate import fabricate_upstream  # noqa: E402
from glassesdata_pipeline import geometry as geo  # noqa: E402
from glassesdata_pipeline import skeleton as sk  # noqa: E402
from glassesdata_pipeline.core import align  # noqa: E402
from glassesdata_pipeline.episode import Episode  # noqa: E402
from glassesdata_pipeline.runner import Runner  # noqa: E402
from glassesdata_pipeline.steps import world  # noqa: E402


@pytest.fixture(scope="module")
def world_rec(synth_recording, tmp_path_factory):
    dst = tmp_path_factory.mktemp("world") / "rec"
    shutil.copytree(synth_recording, dst, ignore=shutil.ignore_patterns("derived"))
    fabricate_upstream(dst)
    report = Runner(Episode(dst)).run(steps=["slam", "imu", "world"], deps=False)
    assert report.ok, report.table()
    return dst


def test_camera_traj_convention_and_accuracy(world_rec):
    ep = Episode(world_rec)
    d = ep.load_npz("camera_traj")
    T, valid = d["T_world_cam"], d["valid"]
    assert d["meta"]["gravity_aligned"] is True
    first = int(np.flatnonzero(valid)[0])
    assert np.allclose(T[first, :3, 3], 0.0, atol=1e-9)
    f = T[first, :3, 2]
    assert abs(np.arctan2(f[1], f[0])) < 1e-9          # first optical axis along +x
    gt = ep.gt["T_world_cam"]
    rot_err = np.degrees(geo.rotation_angle(T[valid, :3, :3], gt[valid, :3, :3]))
    assert np.median(rot_err) < 2.0                     # gravity + yaw alignment (2 s clip)
    assert align.ate(T, gt, align="none", valid=valid)["rmse"] < 0.03


def test_hands_world_smoothing_and_bones(world_rec):
    ep = Episode(world_rec)
    d = ep.load_npz("hands_world")
    gt = ep.gt
    T = gt.n_frames
    for key, shape in (("joints", (T, 2, 21, 3)), ("raw", (T, 2, 21, 3)), ("observed", (T, 2)), ("filled", (T, 2)),
                       ("bone_lengths", (2, 20))):
        assert d[key].shape == shape, key
    raw, J = d["raw"], d["joints"]
    # raw = hand3d joints mapped by the camera trajectory
    h3 = ep.load_npz("hand3d")
    cam = ep.load_npz("camera_traj")
    i = int(np.flatnonzero(h3["valid"][:, 0] & cam["valid"])[0])
    assert np.allclose(raw[i, 0], geo.transform(cam["T_world_cam"][i], h3["joints_cam"][i, 0]))
    observed, filled = d["observed"], d["filled"]
    assert not np.any(observed & filled)
    assert np.all(np.isfinite(J[filled]))
    assert np.all(np.isfinite(raw[observed][..., 0]))
    # bone lengths follow the profile
    target = np.asarray(ep.profile()["bone_lengths_m"])
    assert np.allclose(d["bone_lengths"], target[None])
    lengths = np.stack([np.linalg.norm(J[:, :, k] - J[:, :, sk.PARENTS[k]], axis=-1) for k in range(1, 21)], -1)
    err = np.abs(lengths - target)
    # Soft constraint (w_bone vs the temporal term): exact to ~0.1 mm at realistic speeds (8 s
    # clip: max 0.1 mm); the 2 s clip's 4x-speed right hand bends a few bones by up to ~1.5 cm.
    assert np.nanmedian(err) < 0.001 and np.nanpercentile(err, 95) < 0.003
    # The slow left hand gets better; the 2 s clip's right hand moves unrealistically fast.
    Jg = gt["joints_world"]
    left_raw = align.mpjpe(raw[:, 0], Jg[:, 0])
    left_smooth = align.mpjpe(J[:, 0], Jg[:, 0], observed[:, 0, None])
    assert left_smooth < left_raw
    assert align.mpjpe(J, Jg) < 0.05
    q = json.loads(d["quality"])
    assert set(q) <= {"left", "right"}


def test_world_without_imu_is_not_gravity_aligned(world_rec, tmp_path, caplog):
    dst = tmp_path / "rec"
    shutil.copytree(world_rec, dst)
    (dst / "derived" / "imu.npz").unlink()
    ep = Episode(dst)
    cam = world.camera_trajectory(ep)
    assert cam["gravity_aligned"] is False
    first = int(np.flatnonzero(cam["valid"])[0])
    assert np.allclose(cam["T_world_cam"][first], np.eye(4))
    assert any("NOT gravity aligned" in r.message for r in caplog.records)


def test_hand_gaps_and_outliers(world_rec):
    """A long gap stays NaN, a short one is filled, a gross outlier is rejected."""
    ep = Episode(world_rec)
    t = ep.timeline()["t"]
    T = len(t)
    Jg = ep.gt["joints_world"]
    cam = world.camera_trajectory(ep)
    # Feed ground truth (camera frame) with small noise through the real code path.
    h3 = ep.load_npz("hand3d")
    J = np.asarray(ep.gt["joints_cam"], dtype=np.float64).copy()
    J += np.random.default_rng(5).normal(scale=0.003, size=J.shape)
    valid = np.ones((T, 2), bool)
    valid[10:14, 0] = False     # 4 frames = 0.17 s: bridged
    valid[24:40, 0] = False     # 16 frames = 0.67 s > max_gap_s: not bridged
    J[6, 0] += 0.2              # gross outlier
    meta = h3.pop("meta")
    h3.update(joints_cam=np.where(valid[..., None, None], J, np.nan), valid=valid)
    ep.save_npz("hand3d", meta, **h3)
    p = world.params(ep.config)
    out = world.hands_world(ep, cam["T_world_cam"], cam["valid"], p)
    left = out["joints"][:, 0]
    assert np.all(np.isfinite(left[10:14])) and out["filled"][10:14, 0].all()
    assert np.all(np.isnan(left[26:38]))
    assert not out["observed"][6, 0] and out["filled"][6, 0]
    assert np.linalg.norm(left[6] - Jg[6, 0], axis=-1).mean() < 0.03


def test_hand_seen_in_a_single_frame_is_kept(world_rec, tmp_path):
    dst = tmp_path / "rec"
    shutil.copytree(world_rec, dst)
    ep = Episode(dst)
    h3 = ep.load_npz("hand3d")
    meta = h3.pop("meta")
    valid = np.asarray(h3["valid"]).copy()
    f = int(np.flatnonzero(valid[:, 0])[len(np.flatnonzero(valid[:, 0])) // 2])
    valid[:, 0] = False
    valid[f, 0] = True  # the left hand is seen once
    joints = np.asarray(h3["joints_cam"]).copy()
    joints[~valid] = np.nan
    ep.save_npz("hand3d", meta, **{**h3, "valid": valid, "joints_cam": joints})
    cam = world.camera_trajectory(ep)
    out = world.hands_world(ep, cam["T_world_cam"], cam["valid"], world.params(ep.config))
    assert np.allclose(out["joints"][f, 0], out["raw"][f, 0]) and np.isfinite(out["joints"][f, 0]).all()
    assert out["observed"][f, 0] and out["observed"][:, 0].sum() == 1 and not out["filled"][:, 0].any()
    assert out["quality"]["left"]["single_frame"] is True
    assert np.isfinite(out["joints"][:, 1]).any()  # the right hand is smoothed as before
