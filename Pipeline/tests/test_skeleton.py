import numpy as np
import pytest

from glassesdata_pipeline import geometry as geo
from glassesdata_pipeline import skeleton as sk


def test_topology():
    assert len(sk.PARENTS) == len(sk.JOINT_NAMES) == 21
    assert sk.PARENTS == [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    assert sorted(j for f in sk.FINGERS.values() for j in f) == list(range(1, 21))
    assert sk.JOINT_NAMES[8] == "index_tip" and sk.JOINT_NAMES[17] == "pinky_mcp"


def test_complete_profile_records_defaults():
    p = sk.complete_profile({"hand_length_m": 0.2, "forearm_length_m": "bad", "measured": {"hand_length_m": True}})
    assert p["hand_length_m"] == 0.2
    assert p["forearm_length_m"] == sk.DEFAULT_PROFILE["forearm_length_m"]
    assert "forearm_length_m" in p["defaulted"] and "hand_length_m" not in p["defaulted"]
    assert set(p["defaulted"]) == set(sk.PROFILE_KEYS) - {"hand_length_m"}
    assert p["measured"] == {"hand_length_m": True}
    assert sk.complete_profile(None)["defaulted"] == list(sk.PROFILE_KEYS)


@pytest.mark.parametrize("L", [0.17, 0.187, 0.21])
def test_bone_lengths_follow_buchholz(L):
    profile = {"hand_length_m": L}
    bones = sk.bone_lengths(profile)
    assert bones.shape == (20,)
    for name, joints in sk.FINGERS.items():
        ratios = sk.BUCHHOLZ[name]
        seg = [bones[j - 1] for j in joints[1:]]  # the three moving segments of the finger
        expected = ratios[:3] if name == "thumb" else ratios[1:]
        assert np.allclose(seg, np.array(expected) * L)
        if name != "thumb":  # wrist -> MCP = metacarpal + carpal offset
            assert np.isclose(bones[joints[0] - 1], (ratios[0] + sk.CARPAL_FRACTION) * L)
    rest = sk.rest_pose(profile)
    assert np.isclose(np.linalg.norm(rest[12]), L)  # straight middle finger spans the hand length


def test_palm_width_scales_mcp_spacing():
    narrow = sk.rest_pose({"palm_width_m": 0.07})
    wide = sk.rest_pose({"palm_width_m": 0.098})
    assert np.isclose((wide[5, 0] - wide[17, 0]) / (narrow[5, 0] - narrow[17, 0]), 0.098 / 0.07)
    assert np.allclose(np.linalg.norm(wide[[5, 9, 13, 17]], axis=1), np.linalg.norm(narrow[[5, 9, 13, 17]], axis=1))


def test_canonical_frame_and_mirror():
    right = sk.rest_pose(None, "right")
    left = sk.rest_pose(None, "left")
    assert np.allclose(right[0], 0)
    assert right[4, 0] > 0.03 and left[4, 0] < -0.03  # thumb radial: +x right, -x left
    assert np.allclose(right[9, [0, 2]], 0) and right[9, 1] > 0  # +y through the middle MCP
    assert np.allclose(left * [-1, 1, 1], right)


def test_pose_hand_keeps_bones_and_flexes_to_the_palm():
    rng = np.random.default_rng(0)
    T = geo.make_T(geo.so3_exp(rng.normal(size=3)), rng.normal(size=3))
    flex = np.tile([0.6, 0.9, 0.5], (5, 1))
    abd = np.full(5, 0.1)
    for side in ("left", "right"):
        J = sk.pose_hand(None, side, T, flex, abd)
        bones = np.array([np.linalg.norm(J[i] - J[sk.PARENTS[i]]) for i in range(1, 21)])
        assert np.allclose(bones, sk.bone_lengths(None))
        assert np.allclose(sk.wrist_frame(J, side), T, atol=1e-9)
        local = geo.transform(geo.inv_T(T), J)
        assert local[8, 2] > 0.02  # flexed index tip moves to the palmar side (+z)
        flat = geo.transform(geo.inv_T(T), sk.pose_hand(None, side, T))
        assert np.allclose(flat, sk.rest_pose(None, side))


def test_abduction_spreads_fingers():
    spread = sk.pose_hand(None, "right", np.eye(4), None, np.full(5, 0.2))
    rest = sk.rest_pose(None, "right")
    assert spread[8, 0] > rest[8, 0] and spread[20, 0] < rest[20, 0]


def test_arm_model():
    profile = sk.complete_profile({"upper_arm_length_m": 0.31, "forearm_length_m": 0.27, "shoulder_width_m": 0.4})
    T_cam = geo.look_at([0.1, 0.0, 1.4], [0.6, 0.0, 0.8])
    S = sk.shoulders_from_head(T_cam, profile)
    assert np.allclose(S[0], [0.1 - 0.07, 0.2, 1.2]) and np.allclose(S[1], [0.1 - 0.07, -0.2, 1.2])
    wrist = np.array([0.4, -0.15, 0.85])
    for phi in (-0.4, 0.0, 0.3, 0.8):
        E = sk.elbow_from_swivel(S[1], wrist, 0.31, 0.27, phi, "right")
        assert np.isclose(np.linalg.norm(E - S[1]), 0.31) and np.isclose(np.linalg.norm(E - wrist), 0.27)
        assert np.isclose(sk.swivel_of_elbow(S[1], wrist, E, "right"), phi)
    low = sk.elbow_from_swivel(S[1], wrist, 0.31, 0.27, 0.0, "right")
    out = sk.elbow_from_swivel(S[1], wrist, 0.31, 0.27, 0.5, "right")
    assert out[1] < low[1]  # positive swivel moves the right elbow to the person's right (-y)
    left_wrist = wrist * [1, -1, 1]
    left_low = sk.elbow_from_swivel(S[0], left_wrist, 0.31, 0.27, 0.0, "left")
    left_out = sk.elbow_from_swivel(S[0], left_wrist, 0.31, 0.27, 0.5, "left")
    assert left_out[1] > left_low[1]  # ... and the left elbow to the person's left (+y)
    assert sk.reach_ratio(S[1], wrist, 0.31, 0.27) < 1
