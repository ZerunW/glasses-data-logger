"""Reward terms (rl/rewards.py) and observation blocks (rl/obs.py) on hand-made numpy states."""

from __future__ import annotations

import numpy as np
import pytest

from glassesdata_pipeline.rl import obs as O
from glassesdata_pipeline.rl import rewards as Rw
from glassesdata_pipeline.rl import xmath as xm
from glassesdata_pipeline.rl.task import ObsConfig, RewardConfig

N, K, NF = 3, 2, 4


def state(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(N, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return {
        "finger_q": rng.normal(size=(N, NF)), "finger_qd": rng.normal(size=(N, NF)),
        "base_pos": rng.normal(size=(N, 3)), "base_quat": q, "base_vel": rng.normal(size=(N, 6)),
        "obj_pos": rng.normal(size=(N, 3)), "obj_quat": q[::-1].copy(), "obj_linvel": rng.normal(size=(N, 3)),
        "obj_angvel": rng.normal(size=(N, 3)), "tips": rng.normal(size=(N, K, 3)),
        "hand_qfrc": rng.normal(size=(N, 7)), "hand_qvel": rng.normal(size=(N, 7)),
    }


def ref_from(s: dict, grasp: float = 1.0) -> dict:
    return {"obj_pos": s["obj_pos"].copy(), "obj_quat": s["obj_quat"].copy(), "finger_q": s["finger_q"].copy(),
            "base_pos": s["base_pos"].copy(), "base_quat": s["base_quat"].copy(), "tips": s["tips"].copy(),
            "grasp": np.full(N, grasp)}


def test_errors_zero_and_known_offsets():
    s = state()
    r = ref_from(s)
    e = Rw.tracking_errors(np, s, r)
    for key, value in e.items():
        np.testing.assert_allclose(value, 0.0, atol=1e-7, err_msg=key)
    r["obj_pos"] = r["obj_pos"] + np.array([0.03, 0.04, 0.0])
    r["finger_q"] = r["finger_q"] + 0.2
    r["obj_quat"] = xm.quat_mul(np, xm.quat_from_rotvec(np, np.tile([0, 0, 0.3], (N, 1))), r["obj_quat"])
    e = Rw.tracking_errors(np, s, r)
    np.testing.assert_allclose(e["obj_pos"], 0.05)
    np.testing.assert_allclose(e["hand_qpos"], 0.2)
    np.testing.assert_allclose(e["obj_rot"], 0.3, atol=1e-9)


def test_reward_terms_and_total():
    cfg = RewardConfig()
    s = state()
    r = ref_from(s)
    e = Rw.tracking_errors(np, s, r)
    a = np.ones((N, 5))
    prev = np.zeros((N, 5))
    terms = Rw.reward_terms(np, s, r, e, a, prev, cfg)
    for key in ("obj_pos", "obj_rot", "hand_qpos", "base_pos", "base_rot", "tips"):
        np.testing.assert_allclose(terms[key], 1.0, atol=1e-9)
    np.testing.assert_allclose(terms["action_rate"], 5.0)
    np.testing.assert_allclose(terms["energy"], np.abs(s["hand_qfrc"] * s["hand_qvel"]).sum(-1))
    w = cfg.weights
    total = Rw.total_reward(np, terms, np.array([False, True, False]), w)
    expected = sum(w[k] * terms[k] for k in Rw.POSITIVE) - w["action_rate"] * 5.0 - w["energy"] * terms["energy"]
    np.testing.assert_allclose(total, expected - w["termination"] * np.array([0, 1, 0]))
    # the kernel: at error = sigma the term is exp(-1)
    e2 = dict(e, obj_pos=np.full(N, cfg.sigmas["obj_pos"]))
    assert Rw.reward_terms(np, s, r, e2, a, prev, cfg)["obj_pos"][0] == pytest.approx(np.exp(-1.0))


def test_contact_bonus():
    obj = np.zeros((1, 3))
    near_m, sigma = 0.03, 0.02
    # demo: tips 0 and 1 on a big object (6-7 cm from its centre), tip 2 far away
    ref_tips = np.array([[[0.06, 0, 0], [0.0, 0.07, 0], [0.3, 0, 0]]])
    # simulated tips at the demo distances -> full bonus (object size does not matter); tip 2 is ignored
    sim_tips = np.array([[[0.0, 0.06, 0], [0.07, 0, 0], [0.9, 0, 0]]])
    b = Rw.contact_bonus(np, sim_tips, obj, ref_tips, obj, np.ones(1), near_m, sigma)
    assert b[0] == pytest.approx(1.0)
    # closer than in the demo is not penalised; further by sigma gives exp(-1) for that tip
    sim_tips = np.array([[[0.05, 0, 0], [0.0, 0.09, 0], [0.9, 0, 0]]])
    b = Rw.contact_bonus(np, sim_tips, obj, ref_tips, obj, np.ones(1), near_m, sigma)
    assert b[0] == pytest.approx(0.5 * (1.0 + np.exp(-1)))
    # no grasp in the demo -> 0
    assert Rw.contact_bonus(np, sim_tips, obj, ref_tips, obj, np.zeros(1), near_m, sigma)[0] == 0.0
    # partial grasp flag scales the bonus; a single closest tip is always counted
    one = np.array([[[0.05, 0, 0], [0.2, 0, 0], [0.3, 0, 0]]])
    assert Rw.contact_bonus(np, one, obj, one, obj, np.full(1, 0.5), near_m, sigma)[0] == pytest.approx(0.5)


def test_policy_observation_layout_and_content():
    s = state(1)
    r1 = ref_from(state(2))
    rk = ref_from(state(3))
    prev = np.arange(N * 5, dtype=float).reshape(N, 5)
    origin = np.array([0.3, 0.0, 0.05])
    blocks = O.policy_blocks(np, s, r1, rk, np.array([0.0, 0.5, 1.0]), prev, origin, 0.1)
    layout = O.policy_layout(NF, K, 5)
    assert [name for name, _ in blocks] == [name for name, _ in layout]
    for (name, arr), (_, size) in zip(blocks, layout):
        assert arr.shape == (N, size), name
    out = O.policy_obs(np, s, r1, rk, np.array([0.0, 0.5, 1.0]), prev, origin, ObsConfig(vel_scale=0.1, clip=5.0))
    assert out.shape == (N, sum(size for _, size in layout))
    assert out.max() <= 5.0 and out.min() >= -5.0
    d = dict(blocks)
    np.testing.assert_allclose(d["obj_pos"], s["obj_pos"] - origin)
    np.testing.assert_allclose(d["ref_obj_dpos_k"], rk["obj_pos"] - s["obj_pos"])
    np.testing.assert_allclose(d["finger_qd"], 0.1 * s["finger_qd"])
    np.testing.assert_allclose(d["tips_rel_obj"][:, :3], s["tips"][:, 0] - s["obj_pos"])
    # the relative rotation block is the identity when the object matches the reference
    same = ref_from(s)
    d2 = dict(O.policy_blocks(np, s, same, same, np.zeros(N), prev, origin, 0.1))
    np.testing.assert_allclose(d2["ref_obj_drot_1"], np.tile([1, 0, 0, 0, 1, 0], (N, 1)), atol=1e-12)


def test_observations_replace_nonfinite():
    s = state()
    s["obj_pos"][1, 0] = np.nan
    out = O.policy_obs(np, s, ref_from(state(2)), ref_from(state(3)), np.zeros(N), np.zeros((N, 5)), np.zeros(3), ObsConfig())
    assert np.isfinite(out).all()
    priv = O.privileged_obs(np, s, ref_from(state()), np.ones(N), np.ones(N), ObsConfig())
    assert priv.shape == (N, 1 + 1 + 1 + 3 * K + 1 + 1) and np.isfinite(priv).all()
