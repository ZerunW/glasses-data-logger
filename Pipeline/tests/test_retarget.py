import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/fabricate_sim.py (importlib mode)
from fabricate_sim import fabricate_sim  # noqa: E402

from glassesdata_pipeline import geometry as geo  # noqa: E402
from glassesdata_pipeline import skeleton as sk  # noqa: E402
from glassesdata_pipeline.sim import robots  # noqa: E402
from glassesdata_pipeline.sim.retarget_mink import (  # noqa: E402
    Joint, choose_hands, hold_invalid, keypoint_error, mean_over_hands, smooth_qpos,
)

ROBOT = robots.SHADOW


def _bones(kp):
    return np.array([np.linalg.norm(kp[i] - kp[sk.PARENTS[i]]) for i in range(1, 21)])


def _random_T(rng):
    return geo.make_T(geo.so3_exp(rng.normal(size=3)), rng.normal(size=3))


@pytest.mark.parametrize("side", ["right", "left"])
def test_robot_shaped_hand_maps_to_the_robot_rest_pose(side):
    rng = np.random.default_rng(1)
    rest = ROBOT.rest_keypoints(side)
    mapping = robots.hand_mapping(rest, side, _bones(rest), "wrist")
    assert np.allclose(mapping.chain_scale, 1.0) and np.allclose(mapping.finger_scale, 1.0)
    for _ in range(3):
        T = _random_T(rng)  # the robot palm placed anywhere
        human = geo.transform(T, rest)
        T_palm, targets = robots.keypoint_targets(human, mapping)
        assert np.allclose(T_palm, T, atol=1e-9)
        assert np.allclose(targets, human, atol=1e-9)


def test_palm_frame_axes_match_the_menagerie_description():
    # canonical hand frame (x radial for the right hand, y distal, z palmar) inside the palm frame
    for side in ("right", "left"):
        R = robots.canonical_in_palm(ROBOT.rest_keypoints(side), side)[:3, :3]
        assert R[:, 1] @ [0, 0, 1] > 0.99  # distal = palm +z
        assert R[:, 2] @ [0, -1, 0] > 0.99  # palmar = palm -y
        assert R[:, 0] @ [1, 0, 0] > 0.99  # x_c = x_p (towards the thumb for the right hand, mirrored left)
    right, left = ROBOT.rest_keypoints("right"), ROBOT.rest_keypoints("left")
    assert right[1, 0] > 0 and left[1, 0] < 0  # thumb base on +x (right) / -x (left)


def test_scaling_makes_hand_size_irrelevant_in_wrist_mode():
    rng = np.random.default_rng(2)
    rest = ROBOT.rest_keypoints("right")
    profile = {"hand_length_m": 0.19}
    flexion = rng.uniform(0, 0.8, size=(5, 3))
    T = _random_T(rng)
    small = sk.pose_hand(profile, "right", T, flexion)
    big = geo.transform(T, 1.3 * geo.transform(geo.inv_T(T), small))  # same pose, 30 % larger hand
    bones = sk.bone_lengths(profile)
    t_small = robots.keypoint_targets(small, robots.hand_mapping(rest, "right", bones, "wrist"))
    t_big = robots.keypoint_targets(big, robots.hand_mapping(rest, "right", 1.3 * bones, "wrist"))
    assert np.allclose(t_small[0], t_big[0], atol=1e-9) and np.allclose(t_small[1], t_big[1], atol=1e-9)
    # a flat human hand reaches about as far as the open robot hand (fingers; the thumb differs)
    flat = sk.pose_hand(profile, "right", np.eye(4))
    _, targets = robots.keypoint_targets(flat, robots.hand_mapping(rest, "right", bones, "wrist"))
    for name in ("index", "middle", "ring", "pinky"):
        tip = sk.FINGERS[name][-1]
        assert np.linalg.norm(targets[tip] - targets[0]) == pytest.approx(np.linalg.norm(rest[tip]), rel=0.05)


def test_base_mode_anchors_fingers_at_the_robot_knuckles():
    rest = ROBOT.rest_keypoints("right")
    profile = {"hand_length_m": 0.18}
    human = sk.pose_hand(profile, "right", np.eye(4), np.full((5, 3), 0.3))
    mapping = robots.hand_mapping(rest, "right", sk.bone_lengths(profile), "base")
    T_palm, targets = robots.keypoint_targets(human, mapping)
    local = geo.transform(geo.inv_T(T_palm), targets)  # targets in the palm frame
    for name in sk.FINGER_NAMES:
        base = sk.FINGERS[name][0]
        assert np.allclose(local[base], rest[base], atol=1e-9)
    assert np.allclose(local[0], 0, atol=1e-12)
    none = robots.hand_mapping(rest, "right", sk.bone_lengths(profile), "none")
    assert np.allclose(robots.keypoint_targets(human, none)[1], human)
    with pytest.raises(Exception):
        robots.hand_mapping(rest, "right", sk.bone_lengths(profile), "stretch")


def test_keypoint_table_and_weights():
    frames = ROBOT.keypoint_frames("left")
    assert [f[0] for f in frames] == list(range(21))
    assert frames[0][2] == "lh_palm" and frames[8] == (8, "site", "lh_fftip", "tip") and frames[17][2] == "lh_lfknuckle"
    w = robots.keypoint_weights(ROBOT, {"tip": 1.0, "middle": 0.25, "base": 0.1, "wrist": 0.0})
    assert w[[4, 8, 12, 16, 20]].tolist() == [1.0] * 5 and w[0] == 0 and w[5] == 0.1 and w[6] == 0.25
    sites = {ROBOT.resolve(s.name, "right") for s in ROBOT.added_sites}
    assert {f[2] for f in ROBOT.keypoint_frames("right") if f[1] == "site"} == sites


def test_numpy_helpers():
    valid = np.zeros((10, 2), bool)
    valid[:, 1] = True
    valid[:1, 0] = True
    assert choose_hands(valid, "auto", 0.2) == ["right"]
    assert choose_hands(valid, "auto", 0.05) == ["right", "left"]
    assert choose_hands(valid, ["left"], 0.5) == ["left"] and choose_hands(valid, "right", 0.5) == ["right"]
    with pytest.raises(Exception):
        choose_hands(valid, ["middle"], 0.5)
    q = np.arange(30, dtype=float).reshape(10, 3)
    ok = np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 0], bool)
    held = hold_invalid(q.copy(), ok, [0, 2])
    assert held[:, 0].tolist() == [6, 6, 6, 9, 9, 9, 18, 18, 18, 18] and np.array_equal(held[:, 1], q[:, 1])
    names = robots.qpos_column_names([("right/base_free", "free", 0), ("right/rh_FFJ4", "hinge", 7)], 8)
    assert names[0] == "right/base_free:px" and names[6] == "right/base_free:qz" and names[7] == "right/rh_FFJ4"
    assert np.allclose(mean_over_hands(np.array([[1.0, np.nan], [np.nan, np.nan], [1.0, 3.0]]))[[0, 2]], [1.0, 2.0])
    assert np.isnan(mean_over_hands(np.array([[np.nan, np.nan]]))[0])
    assert keypoint_error(np.zeros((21, 3)), np.full((21, 3), [0.03, 0.04, 0.0])) == pytest.approx(0.05)


def test_smooth_qpos_handles_free_joints_and_limits():
    rng = np.random.default_rng(3)
    T = 40
    q = np.zeros((T, 8))
    q[:, :3] = np.linspace(0, 1, T)[:, None] + rng.normal(scale=0.01, size=(T, 3))
    quats = geo.mat_to_quat(geo.rot_z(np.linspace(0, 1, T)))
    quats[::2] *= -1  # sign flips must not matter
    q[:, 3:7] = quats
    q[:, 7] = 0.99 + rng.normal(scale=0.05, size=T)
    joints = [Joint("right/base_free", "free", 0, 0, None), Joint("right/j", "hinge", 7, 6, (0.0, 1.0))]
    out = smooth_qpos(q, joints, 2.0, {"right/": np.ones(T, bool)})
    assert np.all(np.abs(np.linalg.norm(out[:, 3:7], axis=1) - 1) < 1e-9)
    angles = geo.rotation_angle(geo.quat_to_mat(out[:, 3:7]), geo.rot_z(np.linspace(0, 1, T)))
    assert angles[3:-3].max() < 0.01
    assert out[:, 7].max() <= 1.0 and np.std(np.diff(out[:, 0])) < np.std(np.diff(q[:, 0]))
    assert np.array_equal(smooth_qpos(q, joints, 0.0), q)


@pytest.mark.skipif(not os.environ.get("MUJOCO_MENAGERIE"), reason="set MUJOCO_MENAGERIE to a Menagerie clone")
def test_retarget_step_with_mink(recording_copy):
    pytest.importorskip("mujoco")
    pytest.importorskip("mink")
    from glassesdata_pipeline.episode import Episode
    from glassesdata_pipeline.runner import Runner

    fabricate_sim(recording_copy)
    ep = Episode(recording_copy, {"retarget": {"menagerie_path": os.environ["MUJOCO_MENAGERIE"]}})
    report = Runner(ep).run(steps=["retarget"], deps=False)
    assert report.ok, report.table()
    out = ep.load_npz("retarget_shadow")
    assert out["qpos"].shape == (ep.gt.n_frames, len(out["qpos_names"]))
    err = out["err_m"][np.isfinite(out["err_m"])]
    assert np.median(err) < 0.02
