"""Task logic (rl/task.py): config parsing, phase / start sampling, initial-state noise, action
scaling, termination, and the TrackingTask core driven by a fake numpy backend."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from glassesdata_pipeline.config import load_config
from glassesdata_pipeline.errors import ConfigError
from glassesdata_pipeline.rl import obs as obs_mod
from glassesdata_pipeline.rl import task as T
from glassesdata_pipeline.rl import xmath as xm

sys.path.insert(0, str(Path(__file__).resolve().parent))  # importlib mode: shared RL fakes
from test_rl_fixtures import FakeSim, make_reference, task_model  # noqa: E402

# =============================================================================================
# configuration
# =============================================================================================


def test_shipped_train_toml_parses():
    cfg = load_config()
    train = cfg.section("train").to_dict()
    tc = T.parse_task_config(train)
    assert tc.control_dt == pytest.approx(0.02)
    assert tc.reward.weights["obj_pos"] == pytest.approx(5.0)
    assert tc.reward.sigmas["obj_pos"] == pytest.approx(0.05)
    assert tc.domain.mass_scale == (0.7, 1.3)
    assert tc.termination.obj_pos_m == pytest.approx(0.10)
    assert train["backend"] in ("warp", "mujoco") and train["num_envs"] > 0


def test_config_defaults_and_overrides():
    tc = T.parse_task_config({})
    assert tc.lookahead_steps == 5 and tc.noise.finger_qpos == pytest.approx(0.05)
    tc = T.parse_task_config({"reward": {"obj_pos": 2, "sigma": {"tips": 0.1}}, "noise": {"obj_rot": 0},
                              "domain": {"mass_scale": [1, 1]}, "lookahead_steps": 3})
    assert tc.reward.weights["obj_pos"] == 2.0 and tc.reward.sigmas["tips"] == 0.1
    assert tc.noise.obj_rot == 0.0 and tc.domain.mass_scale == (1.0, 1.0) and tc.lookahead_steps == 3


@pytest.mark.parametrize("bad", [
    {"nope": 1},
    {"reward": {"obj_pos_typo": 1.0}},
    {"reward": {"sigma": {"obj_pos": 0.0}}},
    {"reward": {"obj_pos": -1.0}},
    {"noise": {"finger_qpos": "a"}},
    {"noise": {"bogus": 1.0}},
    {"domain": {"mass_scale": [1.3, 0.7]}},
    {"domain": {"resample_on_reset": 1}},
    {"termination": {"grace_steps": 1.5}},
    {"control_dt": 0},
    {"lookahead_steps": 0},
    {"p_start_at_zero": 2.0},
    {"action": {"clip": 0.0}},
    {"hand": "middle"},
    {"backend": "gpu"},
])
def test_config_errors(bad):
    with pytest.raises(ConfigError):
        T.parse_task_config(bad)


# =============================================================================================
# pure functions
# =============================================================================================


def test_start_sampling_range_and_zero_probability():
    u = np.linspace(0, 0.999999, 1000)
    idx = T.sample_start_steps(np, u, np.ones(1000), num_steps=50, min_episode_steps=10, p_start_at_zero=0.1)
    assert idx.min() == 0 and idx.max() == 39 and idx.dtype == np.int64
    assert len(np.unique(idx)) == 40
    idx = T.sample_start_steps(np, u, np.full(1000, 0.05), 50, 10, 0.1)
    assert (idx == 0).all()
    idx = T.sample_start_steps(np, u, np.ones(1000), 5, 10, 0.0)  # demo shorter than min_episode_steps
    assert (idx == 0).all()


def test_offsets_and_phase():
    idx = np.array([0, 5, 9])
    np.testing.assert_array_equal(T.offset_indices(np, idx, 3, 10), [3, 8, 9])
    np.testing.assert_allclose(T.phase(np, idx, 10, np.zeros(3)), [0, 5 / 9, 1])


def test_initial_state_noise():
    tm = task_model()
    q = np.tile(np.r_[0, 0, 0.2, 1, 0, 0, 0, 1.4, 0.1], (500, 1))
    normals = np.random.default_rng(0).normal(size=q.shape)
    std = tm.qpos_noise_std(T.NoiseConfig(finger_qpos=0.2, base_pos=0.01, base_rot=0.1))
    out = T.noisy_hand_qpos(np, q, normals, std, tm.quat_blocks, tm.hand_q_lo, tm.hand_q_hi)
    np.testing.assert_allclose(np.linalg.norm(out[:, 3:7], axis=-1), 1.0)
    assert out[:, 7].max() <= 1.5 and out[:, 8].min() >= 0.0  # clipped to the joint limits
    assert 0.005 < np.std(out[:, 0]) < 0.015
    noise = T.NoiseConfig(obj_pos_xy=0.01, obj_pos_z=0.002, obj_rot=0.05)
    pos = np.tile([0.3, 0.0, 0.05], (500, 1))
    quat = np.tile([1.0, 0, 0, 0], (500, 1))
    p, qq = T.noisy_object_pose(np, pos, quat, normals[:, :3], normals[:, 3:6], noise)
    assert (p[:, 2] >= 0.05).all()  # never pushed into the table
    assert 0.007 < np.std(p[:, 0]) < 0.013
    ang = xm.quat_angle(np, quat, qq)
    assert 0.03 < np.median(ang) < 0.12


def test_action_scaling_and_mocap():
    a = np.array([[2.0, -0.5, np.nan]])
    np.testing.assert_allclose(T.clip_actions(np, a, 1.0), [[1.0, -0.5, 0.0]])
    ctrl = T.ctrl_targets(np, np.array([[1.0, -1.0]]), np.array([[1.4, 0.1]]), np.array([0.3, 0.6]),
                          np.array([-0.5, 0.0]), np.array([1.5, 3.1]))
    np.testing.assert_allclose(ctrl, [[1.5, 0.0]])
    ref_pos, ref_quat = np.array([[0.1, 0.2, 0.3]]), np.array([[1.0, 0, 0, 0]])
    pos, quat = T.base_targets(np, np.array([[1.0, 0, -1, 0, 0, 1.0]]), ref_pos, ref_quat, 0.02, 0.1)
    np.testing.assert_allclose(pos, [[0.12, 0.2, 0.28]])
    assert xm.quat_angle(np, ref_quat, quat)[0] == pytest.approx(0.1)
    # mocap offset: the root sits at T_mocap_root inside the mocap frame
    off_p, off_q = np.array([0.0, 0.0, 0.05]), xm.quat_from_rotvec(np, np.array([0.0, 0.0, 0.3]))
    mp, mq = T.mocap_from_root(np, pos, quat, off_p, off_q)
    root_q = xm.quat_mul(np, mq, off_q)
    root_p = mp + xm.quat_rotate(np, mq, off_p)
    np.testing.assert_allclose(root_p, pos, atol=1e-12)
    assert xm.quat_angle(np, root_q, quat)[0] == pytest.approx(0.0, abs=1e-7)


def test_termination_and_timeouts():
    cfg = T.TerminationConfig(obj_pos_m=0.1, obj_rot_rad=1.0, fall_below_table_m=0.05, base_pos_m=0.2, grace_steps=2)
    err = np.array([0.05, 0.2, 0.05, 0.05, 0.05, 0.2])
    rot = np.array([0.1, 0.1, 1.5, 0.1, 0.1, 0.1])
    z = np.array([0.8, 0.8, 0.8, 0.7, 0.8, 0.8])
    base = np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.0])
    finite = np.array([True, True, True, True, True, True])
    steps = np.array([5, 5, 5, 5, 5, 1])
    done, reasons = T.termination(np, err, rot, z, 0.8, base, finite, steps, cfg)
    np.testing.assert_array_equal(done, [False, True, True, True, True, False])  # last: grace period
    assert reasons["fell"][3] and reasons["obj_rot"][2] and reasons["base_pos"][4]
    finite[0] = False
    done, reasons = T.termination(np, err, rot, z, float("nan"), base, finite, steps, T.TerminationConfig(obj_rot_rad=0.0))
    assert done[0] and reasons["nonfinite"][0] and not reasons["fell"].any() and not reasons["obj_rot"].any()
    np.testing.assert_array_equal(T.time_outs(np, np.array([8, 9, 3]), np.array([1, 1, 7]), 10, 0), [False, True, False])
    np.testing.assert_array_equal(T.time_outs(np, np.array([8, 9, 3]), np.array([1, 1, 7]), 10, 7), [False, True, True])


# =============================================================================================
# TrackingTask with a fake backend
# =============================================================================================


def make_task(n=8, follow=True, lift=0.1, **cfg_over):
    tm = task_model()
    ref = make_reference(tm, lift=lift)
    sim = FakeSim(tm, n, ref, follow=follow)
    cfg = T.parse_task_config(cfg_over)
    task = T.TrackingTask(np, sim, tm, ref, cfg, seed=0)
    sim.task = task
    return task, sim, tm, ref


def test_task_shapes_and_reset():
    task, sim, tm, ref = make_task()
    assert task.num_actions == 6 + 2
    policy, priv = task.observations()
    layout = obs_mod.policy_layout(2, 2, task.num_actions)
    assert policy.shape == (8, sum(n for _, n in layout)) and task.num_policy_obs == policy.shape[1]
    assert priv.shape == (8, 1 + 1 + 1 + 6 + 1 + 1)
    assert np.isfinite(policy).all() and np.isfinite(priv).all()
    assert len(sim.resets) == 1 and (task.idx <= ref.num_steps - 1 - task.cfg.min_episode_steps).all()
    assert task.max_episode_length == ref.num_steps - 1
    lo, hi = task.cfg.domain.mass_scale
    assert ((sim.domain[0] >= lo) & (sim.domain[0] <= hi)).all()


def test_perfect_tracking_runs_to_the_demo_end():
    task, sim, tm, ref = make_task(n=4, p_start_at_zero=1.0, noise={k: 0.0 for k in
                                   ("finger_qpos", "base_pos", "base_rot", "obj_pos_xy", "obj_pos_z", "obj_rot")})
    assert (task.idx == 0).all()
    zero = np.zeros((4, task.num_actions))
    rewards, outs = [], []
    for k in range(ref.num_steps - 1):
        _, _, r, term, trunc, info = task.step(zero)
        assert not term.any(), {key: v for key, v in info["reasons"].items() if v.any()}
        rewards.append(r)
        outs.append(trunc)
        if k < ref.num_steps - 2:
            assert not trunc.any()
            np.testing.assert_allclose(info["errors"]["obj_pos"], 0.0, atol=1e-12)
            np.testing.assert_allclose(info["errors"]["hand_qpos"], 0.0, atol=1e-9)
            np.testing.assert_allclose(info["errors"]["base_pos"], 0.0, atol=1e-12)
    assert outs[-1].all()  # time-out at the demo end, then reset
    assert len(sim.resets) == 2
    w = task.cfg.reward.weights
    best = w["obj_pos"] + w["obj_rot"] + w["hand_qpos"] + w["base_pos"] + w["base_rot"]
    assert rewards[3].min() > best - 1e-6
    # the zero residual sends exactly the reference targets
    mp, mq = sim.last_mocap
    assert mp.shape == (4, 3) and mq.shape == (4, 4)


def test_static_object_terminates_when_the_reference_moves():
    task, sim, tm, ref = make_task(n=4, follow=False, lift=0.15, p_start_at_zero=1.0, noise={"obj_pos_xy": 0.0, "obj_pos_z": 0.0})
    zero = np.zeros((4, task.num_actions))
    first = None
    for k in range(ref.num_steps - 1):
        _, _, r, term, trunc, info = task.step(zero)
        if term.any():
            first = k
            assert info["reasons"]["obj_pos"].all()
            assert (r < 0).all()  # termination penalty dominates
            break
    # the reference rises 0.15 m between steps 15 and 35: the error passes 0.1 m at demo step 29
    assert first == 28
    assert (task.idx[term] <= ref.num_steps - 1 - task.cfg.min_episode_steps).all()
    assert (task.ep_steps[term] == 0).all() and (task.prev_action[term] == 0).all()


def test_nonfinite_state_resets_only_that_world():
    task, sim, tm, ref = make_task(n=4)
    sim.inject_nan = 2
    _, _, r, term, trunc, info = task.step(np.zeros((4, task.num_actions)))
    sim.inject_nan = None
    assert term[2] and info["reasons"]["nonfinite"][2] and info["reasons"]["nonfinite"].sum() == 1
    assert np.isfinite(r).all()
    np.testing.assert_array_equal(sim.resets[-1], [2])


def test_actions_reach_the_backend_scaled_and_clipped():
    task, sim, tm, ref = make_task(n=2)
    a = np.zeros((2, task.num_actions))
    a[:, 6] = 0.5
    a[:, 7] = 5.0  # clipped to 1
    a[0, 0] = 1.0  # base x residual
    j = np.minimum(task.idx + 1, task.T - 1)
    expected_ctrl = np.minimum(np.maximum(ref.ctrl[j] + np.array([0.5 * 0.3, 1.0 * 0.6]), tm.ctrl_lo), tm.ctrl_hi)
    base_ref = ref.base_pos[j].copy()
    task.step(a)
    np.testing.assert_allclose(sim.ctrl[:, tm.act_ids], expected_ctrl)
    mp, _ = sim.last_mocap
    np.testing.assert_allclose(mp[0], base_ref[0] + [0.02, 0, 0])
    np.testing.assert_allclose(mp[1], base_ref[1])
    np.testing.assert_allclose(task.prev_action[:, 7], 1.0)


def test_play_mode_keeps_final_state():
    task, sim, tm, ref = make_task(n=1, p_start_at_zero=1.0)
    task.auto_reset = False
    for _ in range(ref.num_steps - 1):
        _, _, _, term, trunc, _ = task.step(np.zeros((1, task.num_actions)))
        if term[0] or trunc[0]:
            break
    assert trunc[0] and task.idx[0] == ref.num_steps - 1 and len(sim.resets) == 1
