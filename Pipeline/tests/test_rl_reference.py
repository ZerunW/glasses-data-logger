"""Reference trajectory handling (rl/reference.py): gaps, resampling, velocities, column mapping,
object choice, and load_demo on a small fabricated recording."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from glassesdata_pipeline import geometry as geo
from glassesdata_pipeline.episode import Episode
from glassesdata_pipeline.errors import PipelineError
from glassesdata_pipeline.rl import reference as R

sys.path.insert(0, str(Path(__file__).resolve().parent))  # importlib mode: shared RL fakes
from test_rl_fixtures import task_model  # noqa: E402


def yaw_quat(yaw: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], -1)


def test_fill_gaps_linear_and_hold():
    t = np.arange(6.0)
    x = np.array([np.nan, 1.0, np.nan, np.nan, 4.0, np.nan])
    valid = np.isfinite(x)
    np.testing.assert_allclose(R.fill_gaps(t, x, valid), [1, 1, 2, 3, 4, 4])
    with pytest.raises(PipelineError):
        R.fill_gaps(t, x, np.zeros(6, bool))


def test_quat_gap_fill_and_resample_are_slerp():
    t = np.linspace(0, 1, 11)
    q = yaw_quat(1.2 * t)
    valid = np.ones(11, bool)
    valid[3:7] = False
    filled = R.fill_quat_gaps(t, np.where(valid[:, None], q, np.nan), valid)
    np.testing.assert_allclose(filled, q, atol=1e-12)
    t2 = np.linspace(0, 1, 37)
    np.testing.assert_allclose(R.resample_quat(t, q, t2), yaw_quat(1.2 * t2), atol=1e-12)
    # sign flips in the input do not matter
    flipped = q.copy()
    flipped[::2] *= -1
    out = R.resample_quat(t, flipped, t2)
    np.testing.assert_allclose(np.abs(np.sum(out * yaw_quat(1.2 * t2), -1)), 1.0, atol=1e-12)


def test_resample_linear_and_control_times():
    tc = R.control_times(0.1, 0.5, 0.1)
    np.testing.assert_allclose(tc, [0.1, 0.2, 0.3, 0.4, 0.5])
    x = np.stack([np.arange(5.0), 2 * np.arange(5.0)], -1)
    out = R.resample_linear(np.arange(5.0), x, np.array([0.5, 3.25, 10.0]))
    np.testing.assert_allclose(out, [[0.5, 1.0], [3.25, 6.5], [4.0, 8.0]])


def test_velocities():
    dt = 0.02
    t = dt * np.arange(50)
    np.testing.assert_allclose(R.finite_difference(np.stack([3 * t, -t], -1), dt), np.tile([3.0, -1.0], (50, 1)), atol=1e-9)
    w = R.angular_velocity_world(yaw_quat(0.7 * t), dt)
    np.testing.assert_allclose(w, np.tile([0, 0, 0.7], (50, 1)), atol=1e-9)
    # world vs body frame for a free joint: body angular velocity = R^T w
    tm = task_model()
    q = np.zeros((50, 9))
    q[:, 0] = 0.1 * t
    q[:, 3:7] = R.quat_mul(yaw_quat(0.4 * t), R.quat_mul(np.array([np.cos(0.3), np.sin(0.3), 0, 0]), np.array([1.0, 0, 0, 0])))
    q[:, 7] = np.sin(t)
    v = R.hand_qvel_from_qpos(q, dt, tm.hand_joints, 8)
    np.testing.assert_allclose(v[:, 0], 0.1, atol=1e-9)
    w_world = np.tile([0, 0, 0.4], (50, 1))
    w_body = np.einsum("nji,nj->ni", geo.quat_to_mat(q[:, 3:7]), w_world)
    np.testing.assert_allclose(v[:, 3:6], w_body, atol=1e-6)
    np.testing.assert_allclose(v[2:-2, 6], np.cos(t[2:-2]), atol=1e-3)


def test_smoothing_keeps_constants_and_unit_quats():
    x = np.ones((30, 2))
    np.testing.assert_allclose(R.gaussian_smooth(x, 2.0), x)
    for T, sigma in ((5, 1.0), (20, 5.0), (3, 10.0)):  # kernel longer than the series
        short = np.arange(2.0 * T).reshape(T, 2)
        out = R.gaussian_smooth(short, sigma)
        assert out.shape == short.shape and np.isfinite(out).all()
    q = R.smooth_quats(yaw_quat(np.linspace(0, 1, 30)), 2.0)
    np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1.0)


def test_columns_and_mapping():
    tm = task_model()
    cols = R.column_names(tm.hand_joints)
    assert cols[:7] == [f"right/base_free:{s}" for s in R.FREE_SUFFIXES] and cols[7:] == ["right/j1", "right/j2"]
    widths = {"right/base_free": 7, "right/j1": 1, "right/j2": 1}
    expanded = R.expand_joint_columns(["left/x", "right/base_free", "right/j1", "right/j2"], 11, widths)
    assert expanded[0] == "left/x:0" and expanded[2:9] == cols[:7]
    assert R.expand_joint_columns(cols, 9, {}) == cols
    with pytest.raises(PipelineError):
        R.expand_joint_columns(["a", "b"], 11, widths)
    idx = R.map_columns(["right/j2", "zzz"] + cols[:8], cols)
    np.testing.assert_array_equal(idx, list(range(2, 10)) + [0])
    with pytest.raises(PipelineError, match="not in the retarget"):
        R.map_columns(cols[:8], cols)


def test_choose_object_and_robot_names():
    g = np.zeros((10, 2), bool)
    objs = {"a": {"grasped": g, "valid": np.ones(10, bool)}, "b": {"grasped": g.copy(), "valid": np.ones(10, bool)}}
    objs["b"]["grasped"][2:6, 1] = True
    assert R.choose_object(objs, 1) == "b"
    assert R.choose_object(objs, 0) == "b"  # ties on the hand -> any hand
    assert R.choose_object(objs, 1, "a") == "a"
    with pytest.raises(PipelineError):
        R.choose_object(objs, 1, "c")
    assert R.parse_robot("shadow_right") == ("shadow", "right")
    assert R.parse_robot("shadow_left") == ("shadow", "left")
    assert R.parse_robot("allegro") == ("allegro", "right")
    assert R.hand_slot("right") == 1 and R.hand_slot("right", "left") == 0 and R.hand_slot("left", "auto") == 0
    with pytest.raises(PipelineError, match="train.hand"):
        R.hand_slot("right", "middle")


# =============================================================================================
# load_demo on a fabricated recording
# =============================================================================================


def fabricate(tmp_path: Path, step_format: bool = False) -> Episode:
    rec = tmp_path / "rec"
    rec.mkdir()
    ep = Episode(rec)
    T = 40
    rng = np.random.default_rng(0)
    t = np.arange(T) / 30.0 + rng.uniform(-0.002, 0.002, T)
    t[0] = 0.0
    meta = {"step": "test"}
    ep.save_npz("timeline", meta, t=t, frame_index=np.arange(T))
    tm = task_model()
    cols = R.column_names(tm.hand_joints)
    names = ["left/j1"] + cols[::-1]  # extra column, reversed order
    qpos = np.zeros((T, 10))
    hand = np.zeros((T, 9))
    hand[:, 0:3] = np.stack([0.2 + 0.1 * t, 0 * t, 0.15 + 0 * t], -1)
    hand[:, 3:7] = yaw_quat(0.3 * t)
    hand[:, 7] = 0.5 * t
    hand[:, 8] = 0.25
    qpos[:, 1:] = hand[:, ::-1]
    qpos[:, 0] = 9.0
    qpos[10:13, 1:] = np.nan
    if step_format:  # what steps/retarget.py writes: per-joint names, per-column qpos_names, per-hand validity
        qpos[10:13, 1:] = 0.0  # the solver's value on invalid frames is not NaN; valid[] marks them
        valid_h = np.ones((T, 2), bool)
        valid_h[10:13, 1] = False
        ep.save_npz("retarget_shadow", meta, qpos=qpos, joint_names=np.array(["left/j1", "right/j2", "right/j1", "right/base_free"]),
                    qpos_names=np.array(names), valid=valid_h, hands=np.array(["left", "right"]),
                    err_m=np.full(T, 0.004), err_m_hand=np.full((T, 2), 0.003), robot="shadow", base_body="palm")
    else:
        ep.save_npz("retarget_shadow", meta, qpos=qpos, joint_names=np.array(names), err_m=np.full(T, 0.004), robot="shadow")
    ep.save_json("objects/index.json", [{"id": "obj0"}, {"id": "obj1"}])
    T_wo = np.tile(np.eye(4), (T, 1, 1))
    T_wo[:, :3, 3] = np.stack([0.3 + 0.2 * t, 0 * t, 0.05 + 0.1 * t], -1)
    T_wo[:, :3, :3] = geo.quat_to_mat(yaw_quat(0.5 * t))
    valid = np.ones(T, bool)
    valid[:3] = False
    T_wo[:3] = np.nan
    grasped = np.zeros((T, 2), bool)
    grasped[15:31, 1] = True
    ep.save_npz("objects/obj0/poses", meta, T_world_obj=T_wo, valid=valid, fitness=np.ones(T), grasped=grasped)
    ep.save_npz("objects/obj1/poses", meta, T_world_obj=np.tile(np.eye(4), (T, 1, 1)), valid=np.ones(T, bool),
                fitness=np.ones(T), grasped=np.zeros((T, 2), bool))
    return ep


@pytest.mark.parametrize("step_format", [False, True])
def test_load_demo(tmp_path, step_format):
    ep = fabricate(tmp_path, step_format)
    tm = task_model()
    train = {"control_dt": 0.02, "smooth_sigma_s": 0.0, "object": "auto", "hand": "auto"}
    ref = R.load_demo(ep, "shadow", "right", tm.hand_joints, 8, train)
    t_frames = ep.timeline()["t"]
    assert ref.object_id == "obj0"
    assert ref.t[0] == pytest.approx(t_frames[3])
    assert ref.t[-1] <= t_frames[-1] + 1e-9 and ref.t[-1] > t_frames[-1] - 0.02
    np.testing.assert_allclose(np.diff(ref.t), 0.02)
    tc = ref.t
    np.testing.assert_allclose(ref.obj_pos, np.stack([0.3 + 0.2 * tc, 0 * tc, 0.05 + 0.1 * tc], -1), atol=1e-9)
    np.testing.assert_allclose(np.abs(np.sum(ref.obj_quat * yaw_quat(0.5 * tc), -1)), 1.0, atol=1e-9)
    np.testing.assert_allclose(ref.obj_linvel, np.tile([0.2, 0, 0.1], (len(tc), 1)), atol=1e-6)
    np.testing.assert_allclose(ref.obj_angvel, np.tile([0, 0, 0.5], (len(tc), 1)), atol=1e-6)
    # hand: columns mapped by name, the gap in frames 10-12 filled linearly
    np.testing.assert_allclose(ref.hand_qpos[:, 7], 0.5 * tc, atol=1e-9)
    np.testing.assert_allclose(ref.hand_qpos[:, 8], 0.25)
    np.testing.assert_allclose(ref.hand_qpos[:, 0], 0.2 + 0.1 * tc, atol=1e-9)
    np.testing.assert_allclose(ref.hand_qvel[:, 6], 0.5, atol=1e-6)
    # grasp flag of the right hand, resampled
    inside = (tc > t_frames[15] + 0.02) & (tc < t_frames[30] - 0.02)
    assert np.all(ref.grasp[inside] == 1.0) and ref.grasp[0] == 0.0
    assert not ref.observed.all() and ref.observed[0]
    assert ref.info["frames"] == [3, 39]
    assert ref.info["retarget_err_m"] == pytest.approx(0.003 if step_format else 0.004)
    with pytest.raises(PipelineError, match="missing"):
        ref.check_complete()


def test_load_demo_segment_and_errors(tmp_path):
    ep = fabricate(tmp_path)
    tm = task_model()
    ref = R.load_demo(ep, "shadow", "right", tm.hand_joints, 8,
                      {"control_dt": 0.05, "segment_s": [0.5, 1.0], "object": "obj0"})
    assert ref.t[0] >= 0.5 - 1e-9 and ref.t[-1] <= 1.0 + 1e-9
    with pytest.raises(PipelineError, match="retarget"):
        R.load_demo(ep, "allegro", "right", tm.hand_joints, 8, {})


def test_load_demo_wrong_side(tmp_path):
    ep = fabricate(tmp_path, step_format=True)
    tm = task_model()
    ep.save_npz("retarget_shadow", {"step": "test"}, **{k: v for k, v in ep.load_npz("retarget_shadow").items()
                                                       if k not in ("meta", "hands")}, hands=np.array(["left"]))
    with pytest.raises(PipelineError, match="hands"):
        R.load_demo(ep, "shadow", "right", tm.hand_joints, 8, {})


def test_gap_masks():
    t = np.arange(60) / 24.0
    valid = np.ones(60, bool)
    valid[10:13] = False  # 3 frames: 0.125 s
    valid[30:50] = False  # 20 frames: 0.83 s > 0.5 s
    tc = np.arange(0.0, t[-1] + 1e-9, 0.02)
    observed, long_gap = R.gap_masks(t, valid, tc, 0.5)
    nearest = np.abs(tc[:, None] - t[None]).argmin(axis=1)
    np.testing.assert_array_equal(observed, valid[nearest])
    np.testing.assert_array_equal(long_gap, (nearest >= 30) & (nearest < 50))
    assert not (observed & long_gap).any()
    _, none = R.gap_masks(t, valid, tc, 1.0)
    assert not none.any()
