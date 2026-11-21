"""Physics plausibility check: drive robot hands through the reconstructed episode in MuJoCo and
compare the simulated objects with the reconstructed ones.

Model: :func:`mjcf.compose` of ``scene.xml`` + the retargeted hands with ``base="mocap"`` (each hand
base is pulled to the mocap body ``<side>/base_target`` by the soft weld ``<side>/base_weld``).
Per video frame the retargeted ``qpos`` gives

* the base target: the hand's free-joint pose (position lerp, quaternion slerp between frames),
* the finger controls: for every position actuator the transmission length at the target
  configuration (``mj_kinematics``, ``mj_comPos``, ``mj_tendon``, ``mj_transmission`` on a scratch
  ``MjData`` -> ``actuator_length``), clipped to ``ctrlrange``. This handles the Shadow Hand's coupled
  J1+J2 tendon actuators (their length is ``q_J1 + q_J2``) the same way as plain joint actuators;
  a position servo's force is ``kp (ctrl - length)``, so ``ctrl = length(q_target)`` holds the target.

Controls are interpolated linearly to the physics rate (the scene's ``timestep``); objects start at
their scene pose (first valid reconstructed pose), the whole system settles for ``settle_s``.

Scores (:func:`object_stats`, :func:`plausibility`), all from numpy:

* per object: position / rotation deviation (sim vs reconstruction) over the frames where the
  reconstruction is valid; the demo *lifts* an object when it rises more than ``lift_threshold_m``
  while grasped (any frame when there are no grasp flags); the sim lift *succeeds* when the
  simulated object rises at least ``lift_success_ratio`` of that at the demo's highest frame,
* max penetration of robot geoms into anything else (``-min contact.dist``),
* ``plausibility = mean_k(exp(-pos_k / pos_scale) * exp(-rot_k / rot_scale)) * exp(-max(0, pen -
  tol) / pen_scale) * (0.5 + 0.5 * lift_rate)`` (the last factor only when the demo lifts
  something) - in [0, 1], 1 = the simulated hands reproduce the reconstructed object motion.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .. import geometry as geo
from ..errors import PipelineError, require
from .frames import slerp
from .mjcf import ComposedModel, object_body, sanitize_id

log = logging.getLogger(__name__)


# =============================================================================================
# numpy scoring
# =============================================================================================


def object_stats(sim_T: np.ndarray, rec_T: np.ndarray, valid: np.ndarray, grasped: np.ndarray | None,
                 cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Deviation and lift statistics of one object. ``sim_T``/``rec_T`` ``(T, 4, 4)``,
    ``valid`` ``(T,)``, ``grasped`` ``(T, 2)`` or None."""
    sim_T = np.asarray(sim_T, dtype=np.float64)
    rec_T = np.asarray(rec_T, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool) & np.all(np.isfinite(rec_T), axis=(-1, -2)) & np.all(np.isfinite(sim_T), axis=(-1, -2))
    T = len(sim_T)
    pos = np.full(T, np.nan)
    rot = np.full(T, np.nan)
    if np.any(ok):
        pos[ok] = np.linalg.norm(sim_T[ok, :3, 3] - rec_T[ok, :3, 3], axis=-1)
        rot[ok] = geo.rotation_angle(sim_T[ok, :3, :3], rec_T[ok, :3, :3])
    out: dict[str, Any] = {"frames": int(ok.sum())}
    if not np.any(ok):
        out["status"] = "no valid reconstructed pose"
        return out

    def summary(x: np.ndarray, scale: float = 1.0) -> dict[str, float]:
        v = x[np.isfinite(x)] * scale
        return {"mean": float(v.mean()), "median": float(np.median(v)), "max": float(v.max()), "final": float(v[-1])}

    out["pos_dev_m"] = summary(pos)
    out["rot_dev_deg"] = summary(rot, 180.0 / math.pi)
    first = int(np.flatnonzero(ok)[0])
    z_rec = rec_T[:, 2, 3] - rec_T[first, 2, 3]
    z_sim = sim_T[:, 2, 3] - sim_T[first, 2, 3]
    window = ok.copy()
    if grasped is not None and np.any(grasped):
        window &= np.asarray(grasped, dtype=bool).any(axis=-1)
    demo_lift = float(np.nanmax(np.where(window, z_rec, -np.inf))) if np.any(window) else 0.0
    out["demo_lift_m"] = demo_lift if math.isfinite(demo_lift) else 0.0
    lifted = out["demo_lift_m"] > float(cfg.get("lift_threshold_m", 0.03))
    out["demo_lifts"] = bool(lifted)
    if lifted:
        f = int(np.argmax(np.where(window, z_rec, -np.inf)))
        out["lift_frame"] = f
        out["sim_lift_m"] = float(z_sim[f])
        out["lift_success"] = bool(z_sim[f] >= float(cfg.get("lift_success_ratio", 0.5)) * out["demo_lift_m"])
    out["per_frame"] = {"pos_dev_m": [None if not np.isfinite(v) else round(float(v), 5) for v in pos],
                        "rot_dev_deg": [None if not np.isfinite(v) else round(float(np.degrees(v)), 3) for v in rot]}
    return out


def plausibility(stats: Mapping[str, Mapping[str, Any]], max_penetration_m: float,
                 cfg: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    """Score in [0, 1] and its components (see module docstring)."""
    pos_scale = float(cfg.get("pos_scale_m", 0.05))
    rot_scale = float(cfg.get("rot_scale_deg", 30.0))
    tracking = []
    for s in stats.values():
        if "pos_dev_m" not in s:
            continue
        tracking.append(math.exp(-s["pos_dev_m"]["mean"] / pos_scale) * math.exp(-s["rot_dev_deg"]["mean"] / rot_scale))
    s_track = float(np.mean(tracking)) if tracking else 1.0
    pen = max(0.0, float(max_penetration_m))
    s_pen = math.exp(-max(0.0, pen - float(cfg.get("penetration_tol_m", 0.005))) / float(cfg.get("penetration_scale_m", 0.01)))
    lifts = [s for s in stats.values() if s.get("demo_lifts")]
    lift_rate = float(np.mean([bool(s.get("lift_success")) for s in lifts])) if lifts else None
    s_lift = 1.0 if lift_rate is None else 0.5 + 0.5 * lift_rate
    score = float(np.clip(s_track * s_pen * s_lift, 0.0, 1.0))
    return score, {"tracking": s_track, "penetration": s_pen, "lift": s_lift, "lift_rate": lift_rate,
                   "lifted_objects": len(lifts)}


def lerp_pose(p0: np.ndarray, q0: np.ndarray, p1: np.ndarray, q1: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    """Position lerp and quaternion slerp."""
    return (1.0 - s) * np.asarray(p0) + s * np.asarray(p1), slerp(q0, q1, s)


def column_map(src_names: Sequence[str], src_adr: Sequence[int], src_width: Sequence[int],
               dst_names: Sequence[str], dst_adr: Sequence[int]) -> list[tuple[int, int, int]]:
    """``[(src qpos adr, dst qpos adr, width)]`` for joints present in both models (by name)."""
    dst = dict(zip(dst_names, dst_adr))
    return [(int(a), int(dst[n]), int(w)) for n, a, w in zip(src_names, src_adr, src_width) if n in dst]


# =============================================================================================
# Simulation (MuJoCo)
# =============================================================================================


def simulate(composed: ComposedModel, retarget: Mapping[str, Any], objects: Sequence[Any], t: np.ndarray,
             cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Run the episode in MuJoCo. ``objects``: :class:`sim.data.ObjectData` (ids + reconstructed
    poses). Returns ``{"T_world_obj_sim" (T, K, 4, 4), "object_ids", "max_penetration_m",
    "penetration_m" (T,), "timestep", "substeps"}``."""
    mujoco = require("mujoco", "mujoco", "step physics")
    model = mujoco.MjModel.from_xml_path(str(composed.xml))
    data = mujoco.MjData(model)
    scratch = mujoco.MjData(model)
    qpos_src = np.asarray(retarget["qpos"], dtype=np.float64)
    names = [str(n) for n in retarget["joint_names"]]
    src_adr = np.asarray(retarget["joint_qposadr"], dtype=np.int64)
    widths = []
    for n in names:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        widths.append({0: 7, 1: 4}.get(int(model.jnt_type[j]), 1) if j >= 0 else 1)
    dst_names, dst_adr = [], []
    for j in range(model.njnt):
        dst_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
        dst_adr.append(int(model.jnt_qposadr[j]))
    cmap = column_map(names, src_adr, widths, dst_names, dst_adr)
    if not cmap:
        raise PipelineError(f"{composed.xml}: no joint of the retargeted model found")
    T = min(len(qpos_src), len(t))
    max_frames = int(cfg.get("max_frames", 0) or 0)
    if max_frames:
        T = min(T, max_frames)

    def physics_qpos(f: int) -> np.ndarray:
        q = model.qpos0.copy()
        for a, b, w in cmap:
            q[b:b + w] = qpos_src[f, a:a + w]
        return q

    # finger controls per frame from the actuator transmission lengths
    ctrl = np.zeros((T, model.nu))
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    limited = model.actuator_ctrllimited.astype(bool)
    for f in range(T):
        scratch.qpos[:] = physics_qpos(f)
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        mujoco.mj_tendon(model, scratch)
        mujoco.mj_transmission(model, scratch)
        c = np.array(scratch.actuator_length)
        ctrl[f] = np.where(limited, np.clip(c, lo, hi), c)
    # base targets per hand
    bases = []
    for side, hand in composed.hands.items():
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hand.mocap_body)
        joint = names.index(hand.base_joint) if hand.base_joint in names else -1
        if body < 0 or joint < 0:
            raise PipelineError(f"{composed.xml}: hand {side} lacks its mocap target or base joint")
        a = int(src_adr[joint])
        bases.append((int(model.body_mocapid[body]), qpos_src[:T, a:a + 3], qpos_src[:T, a + 3:a + 7]))
    # robot geoms (bodies under a hand prefix)
    prefixes = tuple(h.prefix for h in composed.hands.values())
    robot_body = np.array([(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(prefixes)
                           for b in range(model.nbody)])
    robot_geom = robot_body[model.geom_bodyid]
    obj_bodies = []
    for obj in objects:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body(sanitize_id(obj.id)))
        obj_bodies.append(b)

    def set_targets(f0: int, f1: int, s: float) -> None:
        for mocap, P, Q in bases:
            p, q = lerp_pose(P[f0], Q[f0], P[f1], Q[f1], s)
            data.mocap_pos[mocap] = p
            data.mocap_quat[mocap] = q
        data.ctrl[:] = (1.0 - s) * ctrl[f0] + s * ctrl[f1]

    def penetration() -> float:
        n = data.ncon
        if n == 0:
            return 0.0
        try:
            dist = np.array(data.contact.dist[:n])
            geoms = np.array(data.contact.geom[:n]).reshape(n, 2)
        except (AttributeError, TypeError):
            dist = np.array([data.contact[i].dist for i in range(n)])
            geoms = np.array([data.contact[i].geom for i in range(n)]).reshape(n, 2)
        g = np.clip(geoms, 0, model.ngeom - 1)
        involved = (robot_geom[g[:, 0]] & (geoms[:, 0] >= 0)) | (robot_geom[g[:, 1]] & (geoms[:, 1] >= 0))
        if not np.any(involved):
            return 0.0
        return float(max(0.0, -dist[involved].min()))

    # initial state
    data.qpos[:] = physics_qpos(0)
    set_targets(0, 0, 0.0)
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    for _ in range(int(round(float(cfg.get("settle_s", 0.2)) / dt))):
        mujoco.mj_step(model, data)
    K = len(objects)
    T_sim = np.full((T, K, 4, 4), np.nan)
    pen = np.zeros(T)
    substeps = []

    def record(f: int) -> None:
        for k, b in enumerate(obj_bodies):
            if b >= 0:
                T_sim[f, k] = geo.make_T(geo.quat_to_mat(np.array(data.xquat[b])), np.array(data.xpos[b]))

    record(0)
    for f in range(T - 1):
        n = max(1, int(round((t[f + 1] - t[f]) / dt)))
        substeps.append(n)
        worst = 0.0
        for k in range(1, n + 1):
            set_targets(f, f + 1, k / n)
            mujoco.mj_step(model, data)
            worst = max(worst, penetration())
        pen[f + 1] = worst
        record(f + 1)
    return {"T_world_obj_sim": T_sim, "object_ids": [o.id for o in objects], "penetration_m": pen,
            "max_penetration_m": float(pen.max()) if len(pen) else 0.0, "timestep": dt,
            "substeps": int(np.median(substeps)) if substeps else 0, "frames": T}
