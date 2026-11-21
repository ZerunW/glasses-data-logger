"""Reward terms (pure functions of ``xp`` arrays).

With ``k(e, s) = exp(-(e / s)^2)`` (1 at zero error, 0.37 at ``e = s``; sigmas in
``[train.reward.sigma]``), the terms at demo step t (after the action) are:

* ``obj_pos``   ``k(|p_obj - p_ref|, sigma.obj_pos)`` - object position tracking
* ``obj_rot``   ``k(angle(q_obj, q_ref), sigma.obj_rot)`` - object orientation tracking
* ``hand_qpos`` ``k(rms(q_finger - q_finger_ref), sigma.hand_qpos)`` - finger imitation
* ``base_pos``  ``k(|p_base - p_base_ref|, sigma.base_pos)`` - wrist (floating base) imitation
* ``base_rot``  ``k(angle(q_base, q_base_ref), sigma.base_rot)``
* ``tips``      ``k(mean_i |tip_i - tip_ref_i|, sigma.tips)`` - fingertip positions (world)
* ``contact``   only while the demo grasps (``grasp_ref(t)`` in [0, 1]): for the demo's contact
  fingertips (reference distance to the object centre within ``contact_near_m`` of the closest
  reference fingertip's), ``mean_i k(max(d_i - d_ref_i, 0), sigma.contact)`` with ``d_i = |tip_i -
  p_obj|`` and ``d_ref_i`` the same distance in the reference - "the fingertips that touched the
  object in the demo are at least as close to it as there". Relative to the demo, so it needs no
  object shape or contact sensors (the same on MuJoCo and MuJoCo Warp) and does not depend on the
  object's size
* ``action_rate`` ``sum (a_t - a_{t-1})^2`` (penalty)
* ``energy``    ``sum |qfrc_actuator * qvel|`` over the hand dofs (penalty, watts)
* ``termination`` 1 on early termination (penalty)

``total = sum_k w_k * term_k - w_action_rate * action_rate - w_energy * energy
- w_termination * terminated`` with the weights from ``[train.reward]``.
"""

from __future__ import annotations

from typing import Any

from . import xmath as xm

POSITIVE = ("obj_pos", "obj_rot", "hand_qpos", "base_pos", "base_rot", "tips", "contact")
PENALTIES = ("action_rate", "energy")


def tracking_errors(xp: Any, s: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
    """Per-world errors between the simulated state ``s`` and the reference ``r`` at the same step."""
    nf = s["finger_q"].shape[-1]
    dq = s["finger_q"] - r["finger_q"]
    return {
        "obj_pos": xm.norm(xp, s["obj_pos"] - r["obj_pos"]),
        "obj_rot": xm.quat_angle(xp, s["obj_quat"], r["obj_quat"]),
        "hand_qpos": xp.sqrt((dq * dq).sum(-1) / max(nf, 1)),
        "base_pos": xm.norm(xp, s["base_pos"] - r["base_pos"]),
        "base_rot": xm.quat_angle(xp, s["base_quat"], r["base_quat"]),
        "tips": xm.norm(xp, s["tips"] - r["tips"]).mean(-1),
    }


def contact_bonus(xp: Any, tips: Any, obj_pos: Any, ref_tips: Any, ref_obj_pos: Any, grasp: Any,
                  near_m: float, sigma: float) -> Any:
    """See the module docstring; ``tips``/``ref_tips`` ``(N, K, 3)``, ``grasp`` ``(N,)``."""
    d = xm.norm(xp, tips - obj_pos[:, None, :])
    d_ref = xm.norm(xp, ref_tips - ref_obj_pos[:, None, :])
    closest = xm.amin(xp, d_ref, -1)[:, None]
    near = xm.to_float(xp, d_ref <= closest + near_m, d)  # at least one fingertip (the closest)
    excess = xp.maximum(d - d_ref, d * 0)
    score = (near * xm.exp_kernel(xp, excess, sigma)).sum(-1) / near.sum(-1)
    return grasp * score


def reward_terms(xp: Any, s: dict[str, Any], r: dict[str, Any], errors: dict[str, Any], action: Any,
                 prev_action: Any, cfg: Any) -> dict[str, Any]:
    """Unweighted reward terms ``(N,)`` (penalties as positive magnitudes)."""
    sg = cfg.sigmas
    da = action - prev_action
    terms = {key: xm.exp_kernel(xp, errors[key], sg[key])
             for key in ("obj_pos", "obj_rot", "hand_qpos", "base_pos", "base_rot", "tips")}
    terms["contact"] = contact_bonus(xp, s["tips"], s["obj_pos"], r["tips"], r["obj_pos"], r["grasp"],
                                     cfg.contact_near_m, sg["contact"])
    terms["action_rate"] = (da * da).sum(-1)
    terms["energy"] = xp.abs(s["hand_qfrc"] * s["hand_qvel"]).sum(-1)
    return terms


def total_reward(xp: Any, terms: dict[str, Any], terminated: Any, weights: dict[str, float]) -> Any:
    total = terms["obj_pos"] * 0
    for key in POSITIVE:
        total = total + weights.get(key, 0.0) * terms[key]
    for key in PENALTIES:
        total = total - weights.get(key, 0.0) * terms[key]
    return total - weights.get("termination", 0.0) * xm.to_float(xp, terminated, total)
