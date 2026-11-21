"""Human hands -> floating robot hands with mink differential IK (needs ``mujoco`` and ``mink``).

Written against **mink 1.4.0** (``mink.Configuration(model)``, ``.update(q)``, ``.q``, ``.data``,
``.integrate_inplace(v, dt)``; ``mink.FrameTask(frame_name, frame_type, position_cost,
orientation_cost, lm_damping=)`` + ``.set_target(mink.SE3.from_matrix(T))``;
``mink.PostureTask(model, cost)`` + ``.set_target(q)``; ``mink.ConfigurationLimit(model)``;
``mink.solve_ik(configuration, tasks, dt, solver, damping=, limits=)``) and MuJoCo >= 3.10.

Per frame, warm-started from the previous solution:

* targets from :func:`robots.keypoint_targets` (palm pose from the human wrist frame; 20 keypoints
  scaled per finger),
* tasks: palm ``FrameTask`` (position + orientation), one position-only ``FrameTask`` per keypoint
  (tips ``tip_cost``, PIP/DIP ``middle_cost``, MCP ``base_cost``), a ``PostureTask`` towards the
  neutral pose and one towards the previous frame (finger joints only; the free-joint base has
  zero posture cost), ``ConfigurationLimit``,
* iterate ``solve_ik`` + ``integrate_inplace`` until the mean keypoint error stops improving.

A hand that is not valid in a frame keeps its previous solution (and its first solution before it
appears). Afterwards the trajectory is smoothed (Gaussian, ``smooth_sigma_frames``; quaternions
hemisphere-aligned) and the errors are recomputed.

The numpy helpers (:func:`choose_hands`, :func:`hold_invalid`, :func:`smooth_qpos`,
:func:`joint_layout`) are unit-tested on the Mac; :func:`retarget` itself needs MuJoCo + mink.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import geometry as geo
from ..errors import PipelineError, require
from . import robots
from .frames import align_hemisphere, smooth_quaternions, smooth_series
from .mjcf import ComposedModel

log = logging.getLogger(__name__)

MINK_PIN = "mink==1.4.0"
SLOTS = {"left": 0, "right": 1}


# =============================================================================================
# numpy helpers
# =============================================================================================


def choose_hands(valid: np.ndarray, setting: Any, min_fraction: float) -> list[str]:
    """Hands to retarget: ``"auto"`` = those valid in at least ``min_fraction`` of the frames
    (right first), or an explicit list."""
    if isinstance(setting, str) and setting != "auto":
        setting = [setting]
    if setting == "auto":
        frac = np.asarray(valid, dtype=bool).mean(axis=0) if len(valid) else np.zeros(2)
        return [side for side in ("right", "left") if frac[SLOTS[side]] >= min_fraction]
    hands = [str(s) for s in setting]
    for side in hands:
        if side not in SLOTS:
            raise PipelineError(f"retarget.hands: unknown hand {side!r}")
    return hands


@dataclass
class Joint:
    name: str
    type: str  # free, ball, slide, hinge
    qposadr: int
    dofadr: int
    range: tuple[float, float] | None


def joint_layout(model: Any) -> list[Joint]:
    """The model's joints (MuJoCo order) with qpos/dof addresses."""
    import mujoco

    kinds = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    out = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        limited = bool(model.jnt_limited[j])
        rng = (float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1])) if limited else None
        out.append(Joint(name, kinds[int(model.jnt_type[j])], int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]), rng))
    return out


def hold_invalid(qpos: np.ndarray, valid: np.ndarray, columns: Sequence[int]) -> np.ndarray:
    """Copy ``columns`` of the first valid frame backwards and hold the last valid frame through
    invalid frames (in place; returns ``qpos``)."""
    idx = np.flatnonzero(valid)
    if not len(idx):
        return qpos
    cols = np.asarray(columns, dtype=int)
    qpos[: idx[0], cols] = qpos[idx[0], cols]
    last = idx[0]
    for f in range(idx[0], len(qpos)):
        if valid[f]:
            last = f
        else:
            qpos[f, cols] = qpos[last, cols]
    return qpos


def smooth_qpos(qpos: np.ndarray, joints: Sequence[Joint], sigma: float, valid: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
    """Gaussian smoothing over time of every joint's qpos columns (free joints: position and
    hemisphere-aligned quaternion separately; hinge/slide joints clipped back to their range).
    ``valid`` maps a joint-name prefix (e.g. ``"right/"``) to a ``(T,)`` mask of frames to use."""
    if sigma <= 0:
        return qpos.copy()
    out = qpos.copy()
    for j in joints:
        mask = None
        if valid:
            for prefix, m in valid.items():
                if j.name.startswith(prefix):
                    mask = m
        if j.type == "free":
            a = j.qposadr
            out[:, a:a + 3] = _fill(smooth_series(qpos[:, a:a + 3], sigma, mask), qpos[:, a:a + 3])
            out[:, a + 3:a + 7] = _fill(smooth_quaternions(qpos[:, a + 3:a + 7], sigma, mask), align_hemisphere(qpos[:, a + 3:a + 7]))
        elif j.type == "ball":
            a = j.qposadr
            out[:, a:a + 4] = _fill(smooth_quaternions(qpos[:, a:a + 4], sigma, mask), align_hemisphere(qpos[:, a:a + 4]))
        else:
            a = j.qposadr
            col = _fill(smooth_series(qpos[:, a:a + 1], sigma, mask), qpos[:, a:a + 1])
            if j.range is not None:
                col = np.clip(col, j.range[0], j.range[1])
            out[:, a:a + 1] = col
    return out


def _fill(smoothed: np.ndarray, original: np.ndarray) -> np.ndarray:
    bad = ~np.all(np.isfinite(smoothed), axis=-1)
    smoothed[bad] = original[bad]
    return smoothed


def mean_over_hands(err: np.ndarray) -> np.ndarray:
    """``(T,)`` mean of a ``(T, H)`` error array over the finite entries (NaN when none)."""
    err = np.asarray(err, dtype=np.float64)
    if err.ndim != 2 or err.shape[1] == 0:
        return np.full(len(err), np.nan)
    ok = np.isfinite(err)
    total = np.where(ok, err, 0.0).sum(axis=1)
    count = ok.sum(axis=1)
    out = np.full(len(err), np.nan)
    out[count > 0] = total[count > 0] / count[count > 0]
    return out


def keypoint_error(positions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Euclidean distance between robot keypoints and their targets (21 x 3 each)."""
    return float(np.mean(np.linalg.norm(np.asarray(positions) - np.asarray(targets), axis=-1)))


# =============================================================================================
# IK (MuJoCo + mink)
# =============================================================================================


@dataclass
class _Hand:
    side: str
    slot: int
    prefix: str
    base_adr: int  # qpos address of the free joint
    palm_body: int
    frames: list[tuple[int, str, int, str, str]]  # (joint index, kind, id, name, role)
    mapping: robots.HandMapping
    T_base_palm: np.ndarray
    tasks: list[Any]
    palm_task: Any


def _frame_pos(data: Any, kind: str, ident: int) -> np.ndarray:
    return np.array(data.xpos[ident] if kind == "body" else data.site_xpos[ident])


def _body_T(data: Any, ident: int) -> np.ndarray:
    return geo.make_T(np.array(data.xmat[ident]).reshape(3, 3), np.array(data.xpos[ident]))


def iterate_keep_best(q_start: np.ndarray, err_start: float, step: Callable[[], tuple[np.ndarray, float]],
                      max_iters: int, tol: float) -> tuple[np.ndarray, float, int]:
    """Run ``step()`` (one IK iteration: returns the new ``(q, error)``) up to ``max_iters`` times,
    stopping when an iteration improves the error by less than ``tol``. Returns the best ``q`` seen
    (the start included), its error and the number of iterations: the last iterate of a
    differential-IK loop can be worse than an earlier one (overshoot, a limit, an active
    constraint switching)."""
    best_q, best_e = np.array(q_start, dtype=np.float64, copy=True), float(err_start)
    previous = np.inf
    n = 0
    for n in range(1, max(0, int(max_iters)) + 1):
        q, e = step()
        e = float(e)
        if e < best_e:
            best_q, best_e = np.array(q, dtype=np.float64, copy=True), e
        if previous - e < tol:
            break
        previous = e
    return best_q, best_e, n


def retarget(composed: ComposedModel, robot: robots.HandRobot, joints_world: np.ndarray, valid: np.ndarray,
             bone_lengths: np.ndarray, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Solve the episode. ``joints_world`` ``(T, 2, 21, 3)``, ``valid`` ``(T, 2)``, ``bone_lengths``
    ``(2, 20)``; ``composed`` from ``mjcf.compose(..., base="free")`` with the hands to solve.
    Returns the arrays of ``retarget_<robot>.npz``."""
    mujoco = require("mujoco", "mujoco", "step retarget")
    mink = require("mink", MINK_PIN, "step retarget")
    model = mujoco.MjModel.from_xml_path(str(composed.xml))
    configuration = mink.Configuration(model)
    q0 = model.qpos0.copy()
    configuration.update(q0)
    data = configuration.data
    layout = joint_layout(model)
    T = joints_world.shape[0]
    weights = {"wrist": 0.0, "tip": float(cfg.get("tip_cost", 1.0)), "middle": float(cfg.get("middle_cost", 0.25)),
               "base": float(cfg.get("base_cost", 0.1))}
    lm = float(cfg.get("lm_damping", 1e-3))
    hands: list[_Hand] = []
    for side, info in composed.hands.items():
        slot = SLOTS[side]
        bj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, info.base_joint)
        palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, info.palm_body)
        base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, info.base_body)
        if bj < 0 or palm < 0 or base_body < 0:
            raise PipelineError(f"{composed.xml}: missing {info.base_joint} / {info.palm_body}")
        frames = []
        for joint, kind, name, role in info.keypoints:
            obj = mujoco.mjtObj.mjOBJ_BODY if kind == "body" else mujoco.mjtObj.mjOBJ_SITE
            ident = mujoco.mj_name2id(model, obj, name)
            if ident < 0:
                raise PipelineError(f"{composed.xml}: no {kind} {name!r}")
            frames.append((joint, kind, ident, name, role))
        T_world_palm0 = _body_T(data, palm)
        rest = np.zeros((21, 3))
        for joint, kind, ident, _, _ in frames:
            rest[joint] = geo.transform(geo.inv_T(T_world_palm0), _frame_pos(data, kind, ident))
        mapping = robots.hand_mapping(rest, side, bone_lengths[slot], str(cfg.get("scaling", "wrist")),
                                      tuple(cfg.get("scale_limits", (0.5, 2.0))))
        palm_task = mink.FrameTask(info.palm_body, "body", position_cost=float(cfg.get("palm_position_cost", 1.0)),
                                   orientation_cost=float(cfg.get("palm_orientation_cost", 0.5)), lm_damping=lm)
        tasks = [palm_task]
        for joint, kind, ident, name, role in frames:
            if joint == 0 or weights[role] <= 0:
                continue
            task = mink.FrameTask(name, kind, position_cost=weights[role], orientation_cost=0.0, lm_damping=lm)
            tasks.append((joint, task))
        T_base_palm = geo.inv_T(_body_T(data, base_body)) @ T_world_palm0
        hands.append(_Hand(side, slot, info.prefix, int(model.jnt_qposadr[bj]), palm, frames, mapping, T_base_palm,
                           tasks, palm_task))
    # posture costs: zero on free joints
    finger_cost = np.zeros(model.nv)
    for j in layout:
        if j.type in ("hinge", "slide"):
            finger_cost[j.dofadr] = 1.0
    posture = mink.PostureTask(model, cost=finger_cost * float(cfg.get("posture_cost", 0.01)))
    posture.set_target(q0)
    temporal = mink.PostureTask(model, cost=finger_cost * float(cfg.get("temporal_cost", 0.05)))
    limits = [mink.ConfigurationLimit(model)]
    solver = str(cfg.get("solver", "daqp"))
    dt = float(cfg.get("dt", 0.02))
    damping = float(cfg.get("damping", 1e-6))
    max_iters = int(cfg.get("max_iters", 30))
    tol = float(cfg.get("tol_m", 1e-5))

    qpos = np.zeros((T, model.nq))
    err = np.full((T, len(hands)), np.nan)
    started = {h.side: False for h in hands}
    q = q0.copy()
    targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def current_error(h: _Hand) -> float:
        T_palm, tgt = targets[h.side]
        pos = np.zeros((21, 3))
        for joint, kind, ident, _, _ in h.frames:
            pos[joint] = _frame_pos(configuration.data, kind, ident)
        return keypoint_error(pos, tgt)

    for f in range(T):
        active = [h for h in hands if valid[f, h.slot]]
        for h in active:
            T_palm, tgt = robots.keypoint_targets(joints_world[f, h.slot], h.mapping)
            targets[h.side] = (T_palm, tgt)
            if not started[h.side]:
                T_base = T_palm @ geo.inv_T(h.T_base_palm)
                pos, quat = T_base[:3, 3], geo.mat_to_quat(T_base[:3, :3])
                q[h.base_adr:h.base_adr + 3] = pos
                q[h.base_adr + 3:h.base_adr + 7] = quat
                started[h.side] = True
        configuration.update(q)
        if active:
            task_list = [posture, temporal]
            temporal.set_target(q)
            for h in active:
                T_palm, tgt = targets[h.side]
                h.palm_task.set_target(mink.SE3.from_matrix(T_palm))
                task_list.append(h.palm_task)
                for joint, task in h.tasks[1:]:
                    task.set_target(mink.SE3.from_translation(tgt[joint]))
                    task_list.append(task)

            def step() -> tuple[np.ndarray, float]:
                velocity = mink.solve_ik(configuration, task_list, dt, solver, damping=damping, limits=limits)
                configuration.integrate_inplace(velocity, dt)
                return configuration.q.copy(), float(np.mean([current_error(h) for h in active]))

            start_error = float(np.mean([current_error(h) for h in active]))
            q, _, _ = iterate_keep_best(configuration.q.copy(), start_error, step, max_iters, tol)
            configuration.update(q)
        qpos[f] = q
        for i, h in enumerate(hands):
            if valid[f, h.slot]:
                err[f, i] = current_error(h)
    # hands that appear later: copy their first solution backwards; hold through gaps
    for h in hands:
        cols = [c for j in layout if j.name.startswith(h.prefix) for c in range(j.qposadr, j.qposadr + _width(j))]
        hold_invalid(qpos, valid[:, h.slot], cols)
    sigma = float(cfg.get("smooth_sigma_frames", 1.0))
    if sigma > 0:
        qpos = smooth_qpos(qpos, layout, sigma, {h.prefix: valid[:, h.slot] for h in hands})
        for f in range(T):
            configuration.update(qpos[f])
            for i, h in enumerate(hands):
                if valid[f, h.slot]:
                    T_palm, tgt = robots.keypoint_targets(joints_world[f, h.slot], h.mapping)
                    targets[h.side] = (T_palm, tgt)
                    err[f, i] = current_error(h)
    names = [j.name for j in layout]
    return {
        "qpos": qpos,
        "joint_names": np.array(names),
        "joint_qposadr": np.array([j.qposadr for j in layout], dtype=np.int64),
        "qpos_names": np.array(robots.qpos_column_names([(j.name, j.type, j.qposadr) for j in layout], model.nq)),
        "err_m": mean_over_hands(err),
        "err_m_hand": err,
        "valid": np.stack([valid[:, h.slot] for h in hands], axis=1) if hands else np.zeros((T, 0), bool),
        "hands": np.array([h.side for h in hands]),
        "base_joints": np.array([composed.hands[h.side].base_joint for h in hands]),
        "palm_bodies": np.array([composed.hands[h.side].palm_body for h in hands]),
        "finger_scale": np.stack([h.mapping.chain_scale for h in hands]) if hands else np.zeros((0, 5)),
    }


def _width(j: Joint) -> int:
    return {"free": 7, "ball": 4}.get(j.type, 1)
