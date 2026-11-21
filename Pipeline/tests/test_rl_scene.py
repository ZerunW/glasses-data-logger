"""Scene adapter glue that needs no MuJoCo: base body choice, table height, explicit-XML scenes,
reference clipping and the ctrl map of a joints-mode (6 base joints) model."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from glassesdata_pipeline.episode import Episode
from glassesdata_pipeline.errors import PipelineError
from glassesdata_pipeline.rl import scene_adapter as sa

sys.path.insert(0, str(Path(__file__).resolve().parent))  # importlib mode: shared RL fakes
from test_rl_fixtures import NAMES, fake_model, scene_names  # noqa: E402


def test_base_body_and_table(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    ep = Episode(rec)
    assert sa._base_body(ep, {}, None) == "palm"  # configs/retarget.toml
    ep.save_npz("retarget_shadow", {"step": "retarget"}, qpos=np.zeros((2, 3)), base_body="forearm")
    assert sa._base_body(ep, {}, "retarget_shadow") == "forearm"
    assert sa._base_body(ep, {"scene": {"base_body": "palm"}}, "retarget_shadow") == "palm"
    assert math.isnan(sa._table_top(ep))
    ep.save_json("sim/scene.json", {"table": {"top_z": -0.42}}, {"step": "scene"})
    assert sa._table_top(ep) == pytest.approx(-0.42)


def test_explicit_xml_scene(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    ep = Episode(rec)
    train = {"scene": {"xml": "my.xml", "root_body": "hand", "mocap_body": "target", "tip_sites": ["a", "b"]}}
    names = sa.compose_rl_scene(ep, "shadow", "right", "cup", train, tmp_path)
    assert names.xml == rec / "my.xml" and names.root_body == "hand" and names.mocap_body == "target"
    assert names.tip_sites == ["a", "b"] and names.object_body == "obj_cup"


def test_reference_clipping():
    q = np.array([[0.0, 2.0], [-1.0, 0.5]])
    out, worst = sa.clip_reference_to_limits(q, np.array([-np.inf, 0.0]), np.array([np.inf, 1.5]))
    np.testing.assert_allclose(out, [[0.0, 1.5], [-1.0, 0.5]])
    assert worst == pytest.approx(0.5)


def joints_mode_model():
    """Hand root with 3 slide + 3 hinge joints (each with a position actuator) and one finger."""
    qpos0 = np.zeros(14)
    qpos0[10] = 1.0
    n_act = 7
    biasprm = np.zeros((n_act, 10))
    biasprm[:, 1] = -10.0
    gear = np.zeros((n_act, 6))
    gear[:, 0] = 1.0
    return fake_model(
        nq=14, nv=13, nu=n_act, nbody=5, njnt=8, ngeom=2, qpos0=qpos0,
        body_parentid=np.array([0, 0, 1, 0, 0]), body_mocapid=np.array([-1, -1, -1, -1, -1]),
        jnt_type=np.array([2, 2, 2, 3, 3, 3, 3, 0]), jnt_bodyid=np.array([1, 1, 1, 1, 1, 1, 2, 4]),
        jnt_qposadr=np.array([0, 1, 2, 3, 4, 5, 6, 7]), jnt_dofadr=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        jnt_limited=np.array([0, 0, 0, 0, 0, 0, 1, 0]), jnt_range=np.array([[0, 0]] * 6 + [[0, 1.5], [0, 0]], dtype=float),
        actuator_trntype=np.zeros(n_act, dtype=int), actuator_trnid=np.array([[j, -1] for j in range(7)]),
        actuator_gear=gear, actuator_gaintype=np.zeros(n_act, dtype=int), actuator_biastype=np.ones(n_act, dtype=int),
        actuator_biasprm=biasprm, actuator_ctrllimited=np.zeros(n_act, dtype=int), actuator_ctrlrange=np.zeros((n_act, 2)),
        tendon_adr=np.zeros(0, dtype=int), tendon_num=np.zeros(0, dtype=int), wrap_type=np.zeros(0, dtype=int),
        wrap_objid=np.zeros(0, dtype=int), wrap_prm=np.zeros(0), geom_bodyid=np.array([2, 4]),
        geom_contype=np.ones(2, dtype=int), geom_conaffinity=np.ones(2, dtype=int),
    )


def test_joints_mode_base():
    names = {"body": ["world", "base", "finger", "unused", "obj"],
             "joint": ["x", "y", "z", "rx", "ry", "rz", "f", "obj_free"], "site": ["tip"],
             "actuator": [f"a{k}" for k in range(7)]}
    tm = sa.build_task_model(joints_mode_model(), names,
                             scene_names(root_body="base", mocap_body=None, tip_sites=["tip"], object_body="obj"), 0.02)
    assert tm.base_mode == "joints" and tm.mocap_id == -1
    assert tm.act_kind == ["base_pos"] * 3 + ["base_rot"] * 3 + ["finger"]
    np.testing.assert_array_equal(tm.base_v_cols, np.arange(6))
    np.testing.assert_array_equal(tm.finger_q_cols, [6])
    from glassesdata_pipeline.rl.task import ActionConfig

    np.testing.assert_allclose(tm.ctrl_scale(ActionConfig(base_pos=0.02, base_rot=0.1, finger=0.3)),
                               [0.02] * 3 + [0.1] * 3 + [0.3])


def test_free_base_without_mocap_is_rejected():
    with pytest.raises(PipelineError, match="base='mocap'"):
        sa.build_task_model(fake_model(), NAMES, scene_names(mocap_body=""), 0.02)
