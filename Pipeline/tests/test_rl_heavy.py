"""Heavy RL tests (Linux machine): the task core with torch, the real MuJoCo model path
(build_task_model, kinematics, the CPU backend), one rsl_rl PPO iteration, and MuJoCo Warp vs CPU.
Each test skips itself when its packages are missing (all of them skip on the development Mac)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from glassesdata_pipeline.rl.reference import DemoReference, angular_velocity_world, finite_difference, hand_qvel_from_qpos
from glassesdata_pipeline.rl.task import TrackingTask, parse_task_config

sys.path.insert(0, str(Path(__file__).resolve().parent))  # importlib mode: shared RL fakes
from test_rl_fixtures import FakeSim, make_reference, task_model  # noqa: E402

MJCF = """
<mujoco model="rl_test">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic"/>
  <worldbody>
    <geom name="table" type="plane" size="1 1 0.01"/>
    <body name="right/palm" pos="0 0 0.2">
      <freejoint name="right/base_free"/>
      <geom name="palm" type="box" size="0.04 0.04 0.01" mass="0.3"/>
      <body name="right/f1" pos="0.03 0 -0.01">
        <joint name="right/j1" type="hinge" axis="0 1 0" range="-0.5 1.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.04" size="0.008" mass="0.02"/>
        <body name="right/f2" pos="0 0 -0.04">
          <joint name="right/j2" type="hinge" axis="0 1 0" range="0 1.6"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.03" size="0.007" mass="0.01"/>
          <site name="right/tip1" pos="0 0 -0.03"/>
        </body>
      </body>
      <body name="right/f3" pos="-0.03 0 -0.01">
        <joint name="right/j3" type="hinge" axis="0 1 0" range="-1.5 0.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.05" size="0.008" mass="0.02"/>
        <site name="right/tip2" pos="0 0 -0.05"/>
      </body>
    </body>
    <body name="right/base_target" mocap="true" pos="0 0 0.2"/>
    <body name="obj_box" pos="0 0 0.025">
      <freejoint name="obj_box_free"/>
      <geom name="box" type="box" size="0.025 0.025 0.025" mass="0.1"/>
    </body>
  </worldbody>
  <equality>
    <weld name="right/base_weld" body1="right/base_target" body2="right/palm" relpose="0 0 0 1 0 0 0" solref="0.02 1"/>
  </equality>
  <tendon>
    <fixed name="right/T12"><joint joint="right/j1" coef="1"/><joint joint="right/j2" coef="1"/></fixed>
  </tendon>
  <actuator>
    <position name="right/A12" tendon="right/T12" kp="5" ctrlrange="0 3"/>
    <position name="right/A3" joint="right/j3" kp="5" ctrlrange="-1.5 0.5"/>
  </actuator>
</mujoco>
"""


def real_model(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    from glassesdata_pipeline.rl import scene_adapter as sa

    xml = tmp_path / "rl_test.xml"
    xml.write_text(MJCF)
    m = mujoco.MjModel.from_xml_path(str(xml))
    sa.check_enums(mujoco)
    names = sa.model_names(mujoco, m)
    scene = sa.SceneNames(xml=xml, robot="test", side="right", root_body="right/palm", mocap_body="right/base_target",
                          tip_sites=["right/tip1", "right/tip2"], object_body="obj_box", table_z=0.0)
    offset = sa.mocap_offset(mujoco, m, sa.resolve_name(names["body"], "right/palm", "body"),
                             sa.resolve_name(names["body"], "right/base_target", "body"))
    tm = sa.build_task_model(m, names, scene, 0.02, offset)
    return mujoco, m, tm


def still_reference(tm, T: int = 40, dt: float = 0.02) -> DemoReference:
    t = dt * np.arange(T)
    hand = np.zeros((T, tm.num_hand_q))
    hand[:, 0:3] = [0.0, 0.0, 0.2]
    hand[:, 3] = 1.0
    hand[:, 7] = 0.3 * np.sin(t)
    obj_pos = np.tile([0.0, 0.0, 0.025], (T, 1))
    obj_quat = np.tile([1.0, 0.0, 0.0, 0.0], (T, 1))
    return DemoReference(dt=dt, t=t, hand_qpos=hand, hand_qvel=hand_qvel_from_qpos(hand, dt, tm.hand_joints, len(tm.hand_vadr)),
                         obj_pos=obj_pos, obj_quat=obj_quat, obj_linvel=finite_difference(obj_pos, dt),
                         obj_angvel=angular_velocity_world(obj_quat, dt), grasp=np.zeros(T), observed=np.ones(T, bool))


def quiet_cfg(**over):
    cfg = {"p_start_at_zero": 1.0, "domain": {"mass_scale": [1.0, 1.0], "friction_scale": [1.0, 1.0]},
           "noise": {k: 0.0 for k in ("finger_qpos", "base_pos", "base_rot", "obj_pos_xy", "obj_pos_z", "obj_rot")}}
    cfg.update(over)
    return parse_task_config(cfg)


# =============================================================================================
# torch
# =============================================================================================


class TorchFake:
    """FakeSim behind torch tensors (exercises the torch code paths of task/obs/rewards)."""

    def __init__(self, torch, fake: FakeSim) -> None:
        self.torch, self.fake, self.num_envs = torch, fake, fake.num_envs

    def _t(self, d):
        return {k: self.torch.as_tensor(v, dtype=self.torch.float32) for k, v in d.items()}

    def read(self):
        return self._t(self.fake.read())

    def apply(self, ctrl, mp, mq):
        self.fake.apply(ctrl.double().numpy(), mp.double().numpy(), mq.double().numpy())

    def reset(self, ids, qpos, qvel, ctrl, mp, mq):
        self.fake.reset(ids.numpy(), qpos.double().numpy(), qvel.double().numpy(), ctrl.double().numpy(), None, None)

    def set_domain(self, ids, ms, fs):
        self.fake.set_domain(ids.numpy(), ms.double().numpy(), fs.double().numpy())


def test_task_with_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    tm = task_model()
    ref = make_reference(tm)
    cfg = quiet_cfg()
    fake_np, fake_t = FakeSim(tm, 4, ref, follow=True), FakeSim(tm, 4, ref, follow=True)
    task_np = TrackingTask(np, fake_np, tm, ref, cfg, seed=0)
    task_t = TrackingTask(torch, TorchFake(torch, fake_t), tm, ref, cfg, seed=0, device="cpu")
    fake_np.task, fake_t.task = task_np, task_t
    a = np.random.default_rng(0).uniform(-1, 1, (4, task_np.num_actions))
    for _ in range(5):
        p_np, _, r_np, *_ = task_np.step(a)
        p_t, _, r_t, *_ = task_t.step(torch.as_tensor(a, dtype=torch.float32))
    np.testing.assert_allclose(p_t.numpy(), p_np, atol=2e-4)
    np.testing.assert_allclose(r_t.numpy(), r_np, rtol=1e-3, atol=1e-3)


# =============================================================================================
# MuJoCo (CPU)
# =============================================================================================


def test_task_model_from_real_mujoco(tmp_path):
    mujoco, m, tm = real_model(tmp_path)
    assert tm.base_mode == "mocap" and tm.decimation == 10
    assert [j.name for j in tm.hand_joints] == ["right/base_free", "right/j1", "right/j2", "right/j3"]
    np.testing.assert_allclose(tm.mocap_offset_pos, 0.0, atol=1e-9)
    np.testing.assert_allclose(np.abs(tm.mocap_offset_quat), [1, 0, 0, 0], atol=1e-9)
    row = tm.names["actuators"].index("right/A12")
    np.testing.assert_allclose(tm.ctrl_map[row, 7:10], [1.0, 1.0, 0.0])
    from glassesdata_pipeline.rl import scene_adapter as sa

    ref = sa.attach_kinematics(mujoco, m, tm, still_reference(tm))
    np.testing.assert_allclose(ref.base_pos, np.tile([0, 0, 0.2], (ref.num_steps, 1)), atol=1e-9)
    assert ref.tips_pos.shape == (ref.num_steps, 2, 3)
    ref.check_complete()
    assert sa.warp_compat(mujoco, m) == []


def test_cpu_backend_task_steps(tmp_path):
    torch = pytest.importorskip("torch")
    mujoco, m, tm = real_model(tmp_path)
    from glassesdata_pipeline.rl import scene_adapter as sa
    from glassesdata_pipeline.rl.env_mujoco import MujocoBackend

    ref = sa.attach_kinematics(mujoco, m, tm, still_reference(tm))
    sim = MujocoBackend(m, tm, 4, xp=torch, device="cpu")
    task = TrackingTask(torch, sim, tm, ref, quiet_cfg(domain={"mass_scale": [0.5, 2.0]}), seed=0, device="cpu")
    np.testing.assert_allclose(task.state["base_pos"].numpy(), np.tile([0, 0, 0.2], (4, 1)), atol=1e-5)
    masses = [float(mm.body_mass[tm.object_body_id]) for mm in sim.models]
    assert len(set(np.round(masses, 6))) > 1  # per-world randomisation
    zero = torch.zeros(4, task.num_actions)
    for _ in range(20):
        policy, priv, reward, term, trunc, info = task.step(zero)
        assert torch.isfinite(policy).all() and torch.isfinite(reward).all()
    # the welded hand holds its reference height and the object rests on the table
    assert (info["errors"]["base_pos"] < 0.02).all(), info["errors"]["base_pos"]
    assert (info["errors"]["obj_pos"] < 0.01).all(), info["errors"]["obj_pos"]
    assert not term.any()


# =============================================================================================
# rsl_rl
# =============================================================================================


def test_one_ppo_iteration_and_reload(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    pytest.importorskip("tensordict")
    from glassesdata_pipeline.rl import train as Tr
    from glassesdata_pipeline.rl.vecenv import wrap

    tm = task_model()
    ref = make_reference(tm)
    fake = FakeSim(tm, 8, ref, follow=True)
    task = TrackingTask(torch, TorchFake(torch, fake), tm, ref, parse_task_config({}), seed=0, device="cpu")
    fake.task = task
    env = wrap(task, {})
    obs = env.get_observations()
    assert obs["policy"].shape == (8, task.num_policy_obs) and obs["privileged"].shape == (8, task.num_privileged_obs)
    cfg = Tr.runner_config({"num_steps_per_env": 4, "save_interval": 1, "actor_hidden_dims": [32],
                            "critic_hidden_dims": [32], "num_mini_batches": 2})
    OnPolicyRunner = Tr._import_runner()
    runner = OnPolicyRunner(env, cfg, log_dir=str(tmp_path), device="cpu")
    runner.learn(num_learning_iterations=2)
    cps = Tr.checkpoints(tmp_path)
    assert cps, list(tmp_path.iterdir())
    runner2 = OnPolicyRunner(env, cfg, log_dir=None, device="cpu")
    runner2.load(str(cps[-1][1]), map_location="cpu")
    policy = runner2.get_inference_policy(device="cpu")
    with torch.inference_mode():
        actions = policy(env.get_observations())
        assert actions.shape == (8, task.num_actions)
        _, rew, dones, extras = env.step(actions)
    assert rew.shape == (8,) and dones.dtype == torch.long and extras["time_outs"].dtype == torch.bool
    assert all("/" in k for k in extras["log"])


# =============================================================================================
# MuJoCo Warp (GPU)
# =============================================================================================


def test_warp_backend_matches_cpu(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    pytest.importorskip("mujoco_warp")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    mujoco, m, tm = real_model(tmp_path)
    from glassesdata_pipeline.rl import scene_adapter as sa
    from glassesdata_pipeline.rl.env_mujoco import MujocoBackend
    from glassesdata_pipeline.rl.env_warp import WarpBackend

    ref = sa.attach_kinematics(mujoco, m, tm, still_reference(tm))
    cfg = quiet_cfg()
    cpu = TrackingTask(torch, MujocoBackend(m, tm, 2, xp=torch, device="cpu"), tm, ref, cfg, seed=0, device="cpu")
    gpu_sim = WarpBackend(m, tm, 64, device="cuda:0", nconmax=32, njmax=128)
    gpu = TrackingTask(torch, gpu_sim, tm, ref, cfg, seed=0, device=torch.device("cuda:0"))
    for _ in range(10):
        _, _, _, _, _, info_c = cpu.step(torch.zeros(2, cpu.num_actions))
        _, _, _, term, _, info_g = gpu.step(torch.zeros(64, gpu.num_actions, device="cuda:0"))
    assert not term.any()
    for key in ("obj_pos", "base_pos", "hand_qpos"):
        assert abs(float(info_g["errors"][key].mean()) - float(info_c["errors"][key].mean())) < 5e-3, key
    assert int((gpu_sim.overflow != 0).sum()) == 0
