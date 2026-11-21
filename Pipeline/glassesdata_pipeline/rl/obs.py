"""Observations (pure functions of ``xp`` arrays).

Policy observation (group ``"policy"``; all positions in the world frame minus ``origin`` = the
reference object position at demo step 0, velocities times ``obs.vel_scale``):

=====================================  ==========  =============================================
block                                  size        content
=====================================  ==========  =============================================
``finger_q``                           nf          finger joint positions (rad / m)
``finger_qd``                          nf          finger joint velocities * vel_scale
``base_pos``                           3           hand root position - origin
``base_rot6d``                         6           hand root orientation (6D)
``base_vel``                           6           base dof velocities * vel_scale
``obj_pos``                            3           object position - origin
``obj_rot6d``                          6           object orientation (6D)
``obj_linvel``, ``obj_angvel``         3 + 3       object velocity (world) * vel_scale
``ref_obj_dpos_1``, ``ref_obj_rot6d_1``  3 + 6     reference object at t+1: position - object, 6D
``ref_obj_dpos_k``, ``ref_obj_rot6d_k``  3 + 6     the same at t+k (``lookahead_steps``)
``ref_obj_drot_1``                     6           6D of the rotation object -> reference at t+1
``ref_finger_dq_1``                    nf          reference finger qpos at t+1 - current
``ref_base_dpos_1``, ``ref_base_rot6d_1``  3 + 6   reference hand root at t+1
``tips_rel_obj``                       3K          fingertip - object positions
``phase``                              1           demo phase in [0, 1]
``prev_action``                        na          the action applied last step
=====================================  ==========  =============================================

Privileged observation (group ``"privileged"``, critic only): the object mass and friction
scales, the reference grasp flag at t, the reference fingertip - object offsets at t (3K), the
current object position / rotation errors.
"""

from __future__ import annotations

from typing import Any

from . import xmath as xm


def policy_blocks(xp: Any, s: dict[str, Any], r1: dict[str, Any], rk: dict[str, Any], phase: Any,
                  prev_action: Any, origin: Any, vel_scale: float) -> list[tuple[str, Any]]:
    """The named policy blocks (in order); ``s`` is the simulated state, ``r1``/``rk`` the
    reference at t+1 and t+k."""
    n = s["obj_pos"].shape[0]
    obj_q = s["obj_quat"]
    drot = xm.quat_mul(xp, xm.quat_conj(xp, obj_q), r1["obj_quat"])
    tips_rel = (s["tips"] - s["obj_pos"][:, None, :]).reshape(n, -1)
    return [
        ("finger_q", s["finger_q"]),
        ("finger_qd", s["finger_qd"] * vel_scale),
        ("base_pos", s["base_pos"] - origin),
        ("base_rot6d", xm.quat_to_rot6d(xp, s["base_quat"])),
        ("base_vel", s["base_vel"] * vel_scale),
        ("obj_pos", s["obj_pos"] - origin),
        ("obj_rot6d", xm.quat_to_rot6d(xp, obj_q)),
        ("obj_linvel", s["obj_linvel"] * vel_scale),
        ("obj_angvel", s["obj_angvel"] * vel_scale),
        ("ref_obj_dpos_1", r1["obj_pos"] - s["obj_pos"]),
        ("ref_obj_rot6d_1", xm.quat_to_rot6d(xp, r1["obj_quat"])),
        ("ref_obj_dpos_k", rk["obj_pos"] - s["obj_pos"]),
        ("ref_obj_rot6d_k", xm.quat_to_rot6d(xp, rk["obj_quat"])),
        ("ref_obj_drot_1", xm.quat_to_rot6d(xp, drot)),
        ("ref_finger_dq_1", r1["finger_q"] - s["finger_q"]),
        ("ref_base_dpos_1", r1["base_pos"] - s["base_pos"]),
        ("ref_base_rot6d_1", xm.quat_to_rot6d(xp, r1["base_quat"])),
        ("tips_rel_obj", tips_rel),
        ("phase", phase[:, None]),
        ("prev_action", prev_action),
    ]


def finalize(xp: Any, blocks: list[tuple[str, Any]], clip: float) -> Any:
    """Concatenate, replace non-finite values by 0, clip to ``[-clip, clip]``."""
    out = xm.cat(xp, [b for _, b in blocks], -1)
    return xm.clip(xp, xm.nan_to_zero(xp, out), -clip, clip)


def policy_obs(xp: Any, s: dict[str, Any], r1: dict[str, Any], rk: dict[str, Any], phase: Any,
               prev_action: Any, origin: Any, cfg: Any) -> Any:
    return finalize(xp, policy_blocks(xp, s, r1, rk, phase, prev_action, origin, cfg.vel_scale), cfg.clip)


def privileged_blocks(xp: Any, s: dict[str, Any], r: dict[str, Any], mass_scale: Any,
                      friction_scale: Any) -> list[tuple[str, Any]]:
    n = s["obj_pos"].shape[0]
    ref_tips_rel = (r["tips"] - r["obj_pos"][:, None, :]).reshape(n, -1)
    pos_err = xm.norm(xp, s["obj_pos"] - r["obj_pos"])
    rot_err = xm.quat_angle(xp, s["obj_quat"], r["obj_quat"])
    return [
        ("mass_scale", mass_scale[:, None]),
        ("friction_scale", friction_scale[:, None]),
        ("ref_grasp", r["grasp"][:, None]),
        ("ref_tips_rel_obj", ref_tips_rel),
        ("obj_pos_err", pos_err[:, None]),
        ("obj_rot_err", rot_err[:, None]),
    ]


def privileged_obs(xp: Any, s: dict[str, Any], r: dict[str, Any], mass_scale: Any, friction_scale: Any,
                   cfg: Any) -> Any:
    return finalize(xp, privileged_blocks(xp, s, r, mass_scale, friction_scale), cfg.clip)


def policy_layout(num_fingers_q: int, num_tips: int, num_actions: int) -> list[tuple[str, int]]:
    """``[(block, size)]`` of the policy observation (documentation and tests)."""
    nf = num_fingers_q
    sizes = {"finger_q": nf, "finger_qd": nf, "base_pos": 3, "base_rot6d": 6, "base_vel": 6, "obj_pos": 3,
             "obj_rot6d": 6, "obj_linvel": 3, "obj_angvel": 3, "ref_obj_dpos_1": 3, "ref_obj_rot6d_1": 6,
             "ref_obj_dpos_k": 3, "ref_obj_rot6d_k": 6, "ref_obj_drot_1": 6, "ref_finger_dq_1": nf,
             "ref_base_dpos_1": 3, "ref_base_rot6d_1": 6, "tips_rel_obj": 3 * num_tips, "phase": 1,
             "prev_action": num_actions}
    return list(sizes.items())
