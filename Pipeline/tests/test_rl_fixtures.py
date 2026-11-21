"""Fakes for the RL tests (numpy only): a tiny mjModel-like object, a kinematic fake physics
backend and a synthetic reference. Imported by the other ``test_rl_*.py`` files; the tests here
check :func:`scene_adapter.build_task_model` on the fake model."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from glassesdata_pipeline.errors import PipelineError
from glassesdata_pipeline.rl.reference import DemoReference, reference_ctrl
from glassesdata_pipeline.rl.scene_adapter import SceneNames, build_task_model

# Bodies: 0 world, 1 hand root (free joint), 2 finger link (hinge j1), 3 finger tip link (hinge j2),
# 4 mocap target, 5 object (free joint). Actuator 0: joint j1; actuator 1: fixed tendon j1 + j2.
NAMES = {
    "body": ["world", "right/palm", "right/f1", "right/f2", "right/base_target", "obj_box"],
    "joint": ["right/base_free", "right/j1", "right/j2", "obj_box_free"],
    "site": ["right/tip1", "right/tip2"],
    "actuator": ["right/A_j1", "right/A_j0"],
    "geom": ["floor", "f1", "f2", "box"],
}
TIP_OFFSETS = np.array([[0.05, 0.0, 0.0], [0.0, 0.05, 0.0]])


def fake_model(**over) -> SimpleNamespace:
    qpos0 = np.zeros(16)
    qpos0[3] = 1.0
    qpos0[12] = 1.0
    biasprm = np.zeros((2, 10))
    biasprm[:, 1] = -10.0
    gear = np.zeros((2, 6))
    gear[:, 0] = 1.0
    m = SimpleNamespace(
        nq=16, nv=14, nu=2, nbody=6, njnt=4, ngeom=4, nsite=2, qpos0=qpos0,
        body_parentid=np.array([0, 0, 1, 2, 0, 0]), body_mocapid=np.array([-1, -1, -1, -1, 0, -1]),
        jnt_type=np.array([0, 3, 3, 0]), jnt_bodyid=np.array([1, 2, 3, 5]), jnt_qposadr=np.array([0, 7, 8, 9]),
        jnt_dofadr=np.array([0, 6, 7, 8]), jnt_limited=np.array([0, 1, 1, 0]),
        jnt_range=np.array([[0.0, 0.0], [-0.5, 1.5], [0.0, 1.6], [0.0, 0.0]]),
        actuator_trntype=np.array([0, 3]), actuator_trnid=np.array([[1, -1], [0, -1]]), actuator_gear=gear,
        actuator_gaintype=np.array([0, 0]), actuator_biastype=np.array([1, 1]), actuator_biasprm=biasprm,
        actuator_ctrllimited=np.array([1, 1]), actuator_ctrlrange=np.array([[-0.5, 1.5], [0.0, 3.1]]),
        tendon_adr=np.array([0]), tendon_num=np.array([2]), wrap_type=np.array([1, 1]),
        wrap_objid=np.array([1, 2]), wrap_prm=np.array([1.0, 1.0]),
        geom_bodyid=np.array([0, 2, 3, 5]), geom_contype=np.ones(4, dtype=int), geom_conaffinity=np.ones(4, dtype=int),
        opt=SimpleNamespace(timestep=0.002),
    )
    for k, v in over.items():
        setattr(m, k, v)
    return m


def scene_names(**over) -> SceneNames:
    kw = dict(xml="x.xml", robot="shadow", side="right", root_body="right/palm", mocap_body="right/base_target",
              tip_sites=["tip1", "tip2"], object_body="obj_box", table_z=0.0)
    kw.update(over)
    return SceneNames(**kw)


def task_model():
    return build_task_model(fake_model(), NAMES, scene_names(), control_dt=0.02)


def make_reference(tm, T: int = 60, dt: float = 0.02, lift: float = 0.1) -> DemoReference:
    """Object at (0.3, 0, 0.05) on the table, lifted by ``lift`` between steps 15 and 35 and turned
    about z; the hand base follows 0.1 m above it; fingers close while grasping."""
    t = dt * np.arange(T)
    s = np.clip((np.arange(T) - 15) / 20.0, 0.0, 1.0)
    obj_pos = np.stack([0.3 + 0 * s, 0 * s, 0.05 + lift * s], -1)
    yaw = 0.5 * s
    obj_quat = np.stack([np.cos(yaw / 2), 0 * s, 0 * s, np.sin(yaw / 2)], -1)
    hand = np.zeros((T, tm.num_hand_q))
    hand[:, 0:3] = obj_pos + np.array([0.0, 0.0, 0.1])
    hand[:, 3] = 1.0
    grasp = ((np.arange(T) >= 12) & (np.arange(T) <= 40)).astype(float)
    hand[:, 7] = 0.8 * np.clip((np.arange(T) - 5) / 7.0, 0.0, 1.0)
    hand[:, 8] = 0.5 * np.clip((np.arange(T) - 5) / 7.0, 0.0, 1.0)
    from glassesdata_pipeline.rl.reference import angular_velocity_world, finite_difference, hand_qvel_from_qpos

    ref = DemoReference(dt=dt, t=t, hand_qpos=hand, hand_qvel=hand_qvel_from_qpos(hand, dt, tm.hand_joints, len(tm.hand_vadr)),
                        obj_pos=obj_pos, obj_quat=obj_quat, obj_linvel=finite_difference(obj_pos, dt),
                        obj_angvel=angular_velocity_world(obj_quat, dt), grasp=grasp, observed=np.ones(T, bool),
                        object_id="box", side="right")
    ref.base_pos = hand[:, 0:3].copy()
    ref.base_quat = hand[:, 3:7].copy()
    ref.tips_pos = ref.base_pos[:, None, :] + TIP_OFFSETS[None] + np.array([0, 0, -0.08])
    ref.ctrl = reference_ctrl(hand, tm.ctrl_map)
    return ref


class FakeSim:
    """Kinematic stand-in for a physics backend (numpy): fingers reach their position targets in
    one control step (j2 = tendon target - j1), the hand root jumps to the mocap target, and the
    object either stays put or (``follow``) is placed at the reference of the task's next step."""

    def __init__(self, tm, num_envs: int, ref: DemoReference | None = None, follow: bool = False) -> None:
        self.tm = tm
        self.num_envs = num_envs
        self.qpos = np.tile(tm.qpos0, (num_envs, 1))
        self.qvel = np.zeros((num_envs, tm.nv))
        self.ctrl = np.zeros((num_envs, tm.nu))
        self.ref = ref
        self.follow = follow
        self.task = None
        self.apply_calls = 0
        self.resets: list[np.ndarray] = []
        self.domain = (np.ones(num_envs), np.ones(num_envs))
        self.last_mocap = None
        self.inject_nan: int | None = None

    def read(self):
        root = self.qpos[:, 0:3]
        obj = self.qpos[:, 9:12]
        tips = root[:, None, :] + TIP_OFFSETS[None] + np.array([0, 0, -0.08])
        tips = tips + 0.01 * self.qpos[:, None, 7:9].sum(-1, keepdims=True)
        return {"qpos": self.qpos.copy(), "qvel": self.qvel.copy(), "qfrc_actuator": np.zeros_like(self.qvel),
                "body_pos": np.stack([root, obj], 1), "body_quat": np.stack([self.qpos[:, 3:7], self.qpos[:, 12:16]], 1),
                "site_pos": tips}

    def apply(self, ctrl, mocap_pos, mocap_quat):
        self.apply_calls += 1
        self.ctrl[:] = ctrl
        self.qpos[:, 7] = ctrl[:, 0]
        self.qpos[:, 8] = ctrl[:, 1] - ctrl[:, 0]
        self.last_mocap = (mocap_pos.copy(), mocap_quat.copy())
        self.qpos[:, 0:3] = mocap_pos
        self.qpos[:, 3:7] = mocap_quat
        if self.follow and self.task is not None:
            j = np.minimum(self.task.idx + 1, self.task.T - 1)
            self.qpos[:, 9:12] = self.ref.obj_pos[j]
            self.qpos[:, 12:16] = self.ref.obj_quat[j]
        if self.inject_nan is not None:
            self.qpos[self.inject_nan, 9] = np.nan

    def reset(self, env_ids, qpos, qvel, ctrl, mocap_pos, mocap_quat):
        self.resets.append(np.array(env_ids))
        self.qpos[env_ids] = qpos
        self.qvel[env_ids] = qvel
        self.ctrl[env_ids] = ctrl

    def set_domain(self, env_ids, mass_scale, friction_scale):
        self.domain[0][env_ids] = mass_scale
        self.domain[1][env_ids] = friction_scale


# =============================================================================================
# build_task_model on the fake model
# =============================================================================================


def test_task_model_layout():
    tm = task_model()
    assert tm.base_mode == "mocap" and tm.mocap_id == 0
    assert [j.name for j in tm.hand_joints] == ["right/base_free", "right/j1", "right/j2"]
    np.testing.assert_array_equal(tm.hand_qadr, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(tm.hand_vadr, [0, 1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(tm.finger_q_cols, [7, 8])
    np.testing.assert_array_equal(tm.finger_v_cols, [6, 7])
    np.testing.assert_array_equal(tm.base_v_cols, [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(tm.obj_qadr, np.arange(9, 16))
    np.testing.assert_array_equal(tm.obj_vadr, np.arange(8, 14))
    assert tm.hand_q_kind == ["base_pos"] * 3 + ["base_quat"] * 4 + ["finger", "finger"]
    assert tm.quat_blocks == [3]
    np.testing.assert_array_equal(tm.hand_q_lo[7:], [-0.5, 0.0])
    assert np.isinf(tm.hand_q_lo[:7]).all()
    np.testing.assert_array_equal(tm.tip_site_ids, [0, 1])
    np.testing.assert_array_equal(tm.hand_geom_ids, [1, 2])
    np.testing.assert_array_equal(tm.object_geom_ids, [3])
    assert tm.decimation == 10


def test_ctrl_map_joint_and_fixed_tendon():
    tm = task_model()
    expected = np.zeros((2, 9))
    expected[0, 7] = 1.0
    expected[1, 7] = expected[1, 8] = 1.0
    np.testing.assert_allclose(tm.ctrl_map, expected)
    assert tm.act_kind == ["finger", "finger"]
    from glassesdata_pipeline.rl.task import ActionConfig, NoiseConfig

    np.testing.assert_allclose(tm.ctrl_scale(ActionConfig(finger=0.3)), [0.3, 0.6])
    std = tm.qpos_noise_std(NoiseConfig(finger_qpos=0.05, base_pos=0.01, base_rot=0.02))
    np.testing.assert_allclose(std, [0.01] * 3 + [0.01] * 4 + [0.05, 0.05])


def test_task_model_rejects_unsupported():
    with pytest.raises(PipelineError, match="mocap"):
        build_task_model(fake_model(), NAMES, scene_names(mocap_body=None), 0.02)
    with pytest.raises(PipelineError, match="spatial"):
        build_task_model(fake_model(wrap_type=np.array([1, 3])), NAMES, scene_names(), 0.02)
    with pytest.raises(PipelineError, match="position servo"):
        build_task_model(fake_model(actuator_biastype=np.array([0, 1])), NAMES, scene_names(), 0.02)
    with pytest.raises(PipelineError, match="no body"):
        build_task_model(fake_model(), NAMES, scene_names(object_body="nope"), 0.02)


def test_name_resolution_suffix():
    from glassesdata_pipeline.rl.scene_adapter import resolve_name

    assert resolve_name(["a", "right/rh_fftip"], "rh_fftip", "site") == 1
    with pytest.raises(PipelineError, match="ambiguous"):
        resolve_name(["left/tip", "right/tip"], "tip", "site")
