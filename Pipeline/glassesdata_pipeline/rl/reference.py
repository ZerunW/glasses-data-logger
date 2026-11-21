"""The demonstration as a reference trajectory at the control rate (numpy only).

Inputs (PIPELINE.md section 5): ``timeline.npz`` (frame times ``t``), ``retarget_<robot>.npz``
(``qpos[T, nq]``, ``joint_names``, ``err_m[T]``), ``objects/index.json`` and
``objects/<id>/poses.npz`` (``T_world_obj[T, 4, 4]``, ``valid[T]``, ``grasped[T, 2]``).

Steps (:func:`load_demo`):

1. Choose the manipulated object: ``train.object`` or, with ``"auto"``, the object with the most
   frames where the human hand ``train.hand`` (default: the robot's side) grasps it (ties: any
   hand, then the most valid frames); the same hand's grasp flags drive the contact reward.
2. Map the retarget qpos columns onto the RL model's hand qpos columns by name (``qpos_names``
   when present, as written by ``steps/retarget.py``, else ``joint_names`` expanded with
   :func:`expand_joint_columns`; :func:`map_columns`). Hand validity: the retarget file's
   ``valid[:, hands.index(side)]`` when present, else finite ``err_m``.
3. Keep the frame range where both the hand and the object are known (optionally cut to
   ``train.segment_s``), fill gaps inside it (linear for scalars, slerp for quaternions; gaps
   longer than ``train.max_gap_s`` are still filled but reported).
4. Resample to the control rate (``train.control_dt``) on the video timestamps, optionally smooth
   (``train.smooth_sigma_s``, Gaussian in seconds), and differentiate for velocities (object linear
   and world angular velocity; hand qvel in MuJoCo's convention: free joints have world linear and
   body-frame angular velocity).

The floating-base pose, fingertip positions and actuator targets of the reference need the
robot's kinematics; :mod:`.scene_adapter` fills ``base_pos``, ``base_quat``, ``tips_pos`` and
``ctrl`` (:func:`reference_ctrl`) with MuJoCo.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import PipelineError

log = logging.getLogger(__name__)

SIDE_SLOT = {"left": 0, "right": 1}
JOINT_WIDTH = {"free": (7, 6), "ball": (4, 3), "slide": (1, 1), "hinge": (1, 1)}
FREE_SUFFIXES = ("px", "py", "pz", "qw", "qx", "qy", "qz")
BALL_SUFFIXES = ("qw", "qx", "qy", "qz")


@dataclass(frozen=True)
class JointSpec:
    """One hand joint in the RL model: ``q_off``/``v_off`` are offsets inside the hand qpos/qvel."""

    name: str
    jtype: str  # free | ball | slide | hinge
    q_off: int
    v_off: int


def column_names(joints: Sequence[JointSpec]) -> list[str]:
    """Per-column names of a hand qpos vector (``j:px ... j:qz`` for free joints, ``j:qw ...``
    for ball joints, the joint name otherwise) - the convention of ``sim.robots.qpos_column_names``."""
    out: list[str] = []
    for j in joints:
        if j.jtype == "free":
            out += [f"{j.name}:{s}" for s in FREE_SUFFIXES]
        elif j.jtype == "ball":
            out += [f"{j.name}:{s}" for s in BALL_SUFFIXES]
        else:
            out.append(j.name)
    return out


def quat_blocks(joints: Sequence[JointSpec]) -> list[int]:
    """Start columns of the quaternions (free / ball joints) inside the hand qpos."""
    return [j.q_off + (3 if j.jtype == "free" else 0) for j in joints if j.jtype in ("free", "ball")]


# =============================================================================================
# the reference
# =============================================================================================


@dataclass
class DemoReference:
    """Reference at the control rate; arrays have first dimension ``T`` (control steps)."""

    dt: float
    t: np.ndarray  # (T,) episode seconds
    hand_qpos: np.ndarray  # (T, nhq) RL-model hand qpos columns
    hand_qvel: np.ndarray  # (T, nhv)
    obj_pos: np.ndarray  # (T, 3)
    obj_quat: np.ndarray  # (T, 4) [w, x, y, z]
    obj_linvel: np.ndarray  # (T, 3) world
    obj_angvel: np.ndarray  # (T, 3) world
    grasp: np.ndarray  # (T,) in [0, 1]
    observed: np.ndarray  # (T,) bool: hand and object observed (not gap-filled) near this time
    object_id: str = ""
    side: str = "right"
    base_pos: np.ndarray | None = None  # (T, 3) hand root (filled from kinematics)
    base_quat: np.ndarray | None = None  # (T, 4)
    tips_pos: np.ndarray | None = None  # (T, K, 3)
    ctrl: np.ndarray | None = None  # (T, nu_hand) actuator targets of the reference
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return int(self.t.shape[0])

    def check_complete(self) -> None:
        missing = [k for k in ("base_pos", "base_quat", "tips_pos", "ctrl") if getattr(self, k) is None]
        if missing:
            raise PipelineError(f"reference is missing {', '.join(missing)} (run scene_adapter.attach_kinematics)")
        for key in ("hand_qpos", "hand_qvel", "obj_pos", "obj_quat", "obj_linvel", "obj_angvel", "grasp",
                    "base_pos", "base_quat", "tips_pos", "ctrl"):
            value = getattr(self, key)
            if value.shape[0] != self.num_steps:
                raise PipelineError(f"reference.{key} has {value.shape[0]} rows, expected {self.num_steps}")
            if not np.all(np.isfinite(value)):
                raise PipelineError(f"reference.{key} contains non-finite values")

    def lift_profile(self) -> np.ndarray:
        """``(T,)`` object height above its height at the first step."""
        return self.obj_pos[:, 2] - self.obj_pos[0, 2]


def reference_ctrl(hand_qpos: np.ndarray, ctrl_map: np.ndarray) -> np.ndarray:
    """Actuator targets of the reference: ``hand_qpos @ ctrl_map.T`` (``ctrl_map`` (nu, nhq))."""
    return np.asarray(hand_qpos, dtype=np.float64) @ np.asarray(ctrl_map, dtype=np.float64).T


# =============================================================================================
# quaternions (numpy)
# =============================================================================================


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., :1], -q[..., 1:]], -1)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, q[..., :1])
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(s < 1e-12, 2.0 / np.maximum(q[..., :1], 1e-12), angle / np.where(s < 1e-12, 1.0, s))
    return v * k


def quat_rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vectors into the frame of ``q`` (``R(q)^T v``)."""
    qc = quat_conj(q)
    vq = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], -1)
    return quat_mul(quat_mul(qc, vq), q)[..., 1:]


def continuous_quats(q: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive quaternions have a non-negative dot product (rows; NaN rows are
    skipped and do not break the chain)."""
    q = np.array(q, dtype=np.float64)
    last = None
    for k in range(q.shape[0]):
        if not np.all(np.isfinite(q[k])):
            continue
        if last is not None and float(np.dot(q[k], last)) < 0:
            q[k] = -q[k]
        last = q[k]
    return q


def slerp(q0: np.ndarray, q1: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Spherical interpolation, ``s`` in [0, 1] (broadcasting over leading dimensions)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin = np.sin(theta)
    small = sin < 1e-6
    safe = np.where(small, 1.0, sin)
    w0 = np.where(small, 1.0 - s, np.sin((1.0 - s) * theta) / safe)
    w1 = np.where(small, s, np.sin(s * theta) / safe)
    out = w0 * q0 + w1 * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


# =============================================================================================
# gaps, resampling, smoothing, velocities
# =============================================================================================


def fill_gaps(t: np.ndarray, x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linear interpolation over invalid rows of ``x`` ``(T, ...)``; rows before the first / after
    the last valid row hold the nearest valid value. Raises when no row is valid."""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        raise PipelineError("fill_gaps: no valid samples")
    flat = x.reshape(x.shape[0], -1)
    out = np.empty_like(flat)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(t, t[valid], flat[valid, c])
    return out.reshape(x.shape)


def fill_quat_gaps(t: np.ndarray, q: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Slerp over invalid rows of unit quaternions ``q`` ``(T, 4)`` (ends hold)."""
    t = np.asarray(t, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        raise PipelineError("fill_quat_gaps: no valid samples")
    tv = t[valid]
    qv = continuous_quats(np.asarray(q, dtype=np.float64)[valid])
    return resample_quat(tv, qv, t)


def resample_linear(t_src: np.ndarray, x: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """``x`` ``(T, ...)`` sampled at ``t_src`` (increasing) -> ``(len(t_dst), ...)`` (ends hold)."""
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(x.shape[0], -1)
    out = np.stack([np.interp(t_dst, t_src, flat[:, c]) for c in range(flat.shape[1])], -1)
    return out.reshape((len(t_dst),) + x.shape[1:])


def resample_quat(t_src: np.ndarray, q: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Slerp resampling of quaternions ``(T, 4)`` (ends hold)."""
    t_src = np.asarray(t_src, dtype=np.float64)
    t_dst = np.asarray(t_dst, dtype=np.float64)
    q = continuous_quats(q)
    if len(t_src) == 1:
        return np.repeat(q[:1], len(t_dst), axis=0)
    j = np.clip(np.searchsorted(t_src, t_dst, side="right") - 1, 0, len(t_src) - 2)
    span = t_src[j + 1] - t_src[j]
    s = np.clip((t_dst - t_src[j]) / np.where(span > 0, span, 1.0), 0.0, 1.0)
    return continuous_quats(slerp(q[j], q[j + 1], s))


def gaussian_smooth(x: np.ndarray, sigma_steps: float) -> np.ndarray:
    """Gaussian smoothing along axis 0 with edge renormalisation (no-op for ``sigma_steps <= 0``)."""
    x = np.asarray(x, dtype=np.float64)
    if sigma_steps <= 0 or x.shape[0] < 3:
        return x.copy()
    # the kernel must not be longer than the series (np.convolve "same" returns the longer length)
    radius = min(max(1, int(math.ceil(3 * sigma_steps))), (x.shape[0] - 1) // 2)
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma_steps) ** 2)
    flat = x.reshape(x.shape[0], -1)
    ones = np.ones(flat.shape[0])
    norm = np.convolve(ones, k, mode="same")
    out = np.stack([np.convolve(flat[:, c], k, mode="same") / norm for c in range(flat.shape[1])], -1)
    return out.reshape(x.shape)


def smooth_quats(q: np.ndarray, sigma_steps: float) -> np.ndarray:
    q = continuous_quats(q)
    out = gaussian_smooth(q, sigma_steps)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def finite_difference(x: np.ndarray, dt: float) -> np.ndarray:
    """Central differences along axis 0 (one-sided at the ends); zeros for a single sample."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        return np.zeros_like(x)
    return np.gradient(x, dt, axis=0)


def angular_velocity_world(q: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity ``(T, 3)`` of a unit-quaternion sequence ``(T, 4)``:
    ``rotvec(q[k+1] q[k-1]^*) / (2 dt)`` inside, one-sided at the ends."""
    q = continuous_quats(q)
    T = q.shape[0]
    if T < 2:
        return np.zeros((T, 3))
    w = np.zeros((T, 3))
    if T > 2:
        w[1:-1] = quat_to_rotvec(quat_mul(q[2:], quat_conj(q[:-2]))) / (2.0 * dt)
    w[0] = quat_to_rotvec(quat_mul(q[1], quat_conj(q[0]))) / dt
    w[-1] = quat_to_rotvec(quat_mul(q[-1], quat_conj(q[-2]))) / dt
    return w


def hand_qvel_from_qpos(qpos: np.ndarray, dt: float, joints: Sequence[JointSpec], nv: int) -> np.ndarray:
    """MuJoCo-convention velocities ``(T, nv)`` of a hand qpos sequence: hinge/slide central
    differences; free joints world linear + body-frame angular velocity; ball joints body-frame
    angular velocity."""
    qpos = np.asarray(qpos, dtype=np.float64)
    out = np.zeros((qpos.shape[0], nv))
    for j in joints:
        if j.jtype in ("hinge", "slide"):
            out[:, j.v_off] = finite_difference(qpos[:, j.q_off], dt)
        elif j.jtype == "free":
            out[:, j.v_off:j.v_off + 3] = finite_difference(qpos[:, j.q_off:j.q_off + 3], dt)
            q = qpos[:, j.q_off + 3:j.q_off + 7]
            out[:, j.v_off + 3:j.v_off + 6] = quat_rotate_inv(q, angular_velocity_world(q, dt))
        elif j.jtype == "ball":
            q = qpos[:, j.q_off:j.q_off + 4]
            out[:, j.v_off:j.v_off + 3] = quat_rotate_inv(q, angular_velocity_world(q, dt))
        else:
            raise PipelineError(f"joint {j.name}: unknown type {j.jtype!r}")
    return out


def control_times(t0: float, t1: float, dt: float) -> np.ndarray:
    """``t0, t0 + dt, ...`` up to and including ``t1`` (at least one sample)."""
    n = int(math.floor((t1 - t0) / dt + 1e-9)) + 1
    return t0 + dt * np.arange(max(n, 1))


def resample_hand(t_src: np.ndarray, qpos: np.ndarray, t_dst: np.ndarray, joints: Sequence[JointSpec]) -> np.ndarray:
    """Hand qpos resampling: linear for scalar columns, slerp for quaternion blocks."""
    out = resample_linear(t_src, qpos, t_dst)
    for start in quat_blocks(joints):
        out[:, start:start + 4] = resample_quat(t_src, qpos[:, start:start + 4], t_dst)
    return out


def fill_hand_gaps(t: np.ndarray, qpos: np.ndarray, valid: np.ndarray, joints: Sequence[JointSpec]) -> np.ndarray:
    out = fill_gaps(t, qpos, valid)
    for start in quat_blocks(joints):
        out[:, start:start + 4] = fill_quat_gaps(t, qpos[:, start:start + 4], valid)
    return out


def smooth_hand(qpos: np.ndarray, sigma_steps: float, joints: Sequence[JointSpec]) -> np.ndarray:
    out = gaussian_smooth(qpos, sigma_steps)
    for start in quat_blocks(joints):
        out[:, start:start + 4] = smooth_quats(qpos[:, start:start + 4], sigma_steps)
    return out


def longest_gap_s(t: np.ndarray, valid: np.ndarray) -> float:
    """Longest run of invalid samples between valid ones, in seconds (0 when none)."""
    t = np.asarray(t, dtype=np.float64)
    idx = np.nonzero(np.asarray(valid, dtype=bool))[0]
    if len(idx) < 2:
        return 0.0
    return float(np.max(t[idx[1:]] - t[idx[:-1]]) - (np.median(np.diff(t)) if len(t) > 1 else 0.0))


# =============================================================================================
# column bookkeeping and object choice
# =============================================================================================


def hand_slot(side: str, hand: Any = "auto") -> int:
    """Slot (0 left, 1 right) of the human hand whose grasp flags count: ``train.hand`` or, with
    ``"auto"``, the robot's side."""
    if hand in ("auto", None, ""):
        hand = side
    if hand not in SIDE_SLOT:
        raise PipelineError(f"train.hand must be 'auto', 'left' or 'right', got {hand!r}")
    return SIDE_SLOT[str(hand)]


def parse_robot(name: str) -> tuple[str, str]:
    """``"shadow_right"`` -> ``("shadow", "right")``; a name without a side means the right hand."""
    base, _, side = name.rpartition("_")
    if base and side in SIDE_SLOT:
        return base, side
    return name, "right"


def expand_joint_columns(joint_names: Sequence[str], nq: int, widths: Mapping[str, int]) -> list[str]:
    """Per-column names of a retarget ``qpos[T, nq]``.

    ``joint_names`` is either already per column (``len == nq``, e.g. ``"j:px"`` for free joints) or
    per joint; per-joint names are expanded with the qpos widths of the RL model's joints
    (``widths``: name -> 7 / 4 / 1). At most one joint of unknown width is allowed (it takes the
    remaining columns)."""
    names = [str(n) for n in joint_names]
    if len(names) == nq:
        return names
    known = sum(widths.get(n, 0) for n in names)
    unknown = [n for n in names if n not in widths]
    if len(unknown) > 1 or (not unknown and known != nq) or (unknown and nq - known < 1):
        raise PipelineError(f"retarget joint_names ({len(names)} names) do not describe qpos with {nq} columns; "
                            f"unknown joints: {unknown[:5]}")
    out: list[str] = []
    for n in names:
        w = widths.get(n, nq - known)
        if w == 7:
            out += [f"{n}:{s}" for s in FREE_SUFFIXES]
        elif w == 4:
            out += [f"{n}:{s}" for s in BALL_SUFFIXES]
        elif w == 1:
            out.append(n)
        else:
            out += [f"{n}:{k}" for k in range(w)]
    return out


def map_columns(src: Sequence[str], dst: Sequence[str]) -> np.ndarray:
    """Index into ``src`` for every name of ``dst``; raises listing the missing names."""
    lookup = {name: k for k, name in enumerate(src)}
    missing = [name for name in dst if name not in lookup]
    if missing:
        raise PipelineError(f"{len(missing)} RL hand qpos columns are not in the retarget output: "
                            f"{missing[:8]}{' ...' if len(missing) > 8 else ''}; retarget has {list(src)[:8]} ...")
    return np.array([lookup[name] for name in dst], dtype=np.int64)


def choose_object(candidates: Mapping[str, Mapping[str, np.ndarray]], slot: int, requested: str = "auto") -> str:
    """The manipulated object: ``requested`` if not ``"auto"``, else the one grasped by hand
    ``slot`` in the most frames (then by any hand, then the most valid frames)."""
    if not candidates:
        raise PipelineError("no objects (objects/index.json is empty)")
    if requested and requested != "auto":
        if requested not in candidates:
            raise PipelineError(f"train.object = {requested!r} is not one of {sorted(candidates)}")
        return requested

    def score(item: tuple[str, Mapping[str, np.ndarray]]) -> tuple[int, int, int]:
        g = np.asarray(item[1]["grasped"], dtype=bool)
        valid = np.asarray(item[1]["valid"], dtype=bool)
        return int(g[:, slot].sum()), int(g.any(axis=1).sum()), int(valid.sum())

    best = max(candidates.items(), key=score)
    if score(best)[1] == 0:
        log.warning("no object is ever grasped; using %s (most valid frames)", best[0])
    return best[0]


# =============================================================================================
# loading from an episode
# =============================================================================================


def find_retarget(ep: Any, robot: str, side: str) -> str:
    """Name (relative to ``derived/``) of the retarget output: ``retarget_<robot>_<side>`` or
    ``retarget_<robot>``."""
    for name in (f"retarget_{robot}_{side}", f"retarget_{robot}"):
        if ep.has(f"{name}.npz"):
            return name
    raise PipelineError(f"{ep.path}: no derived/retarget_{robot}_{side}.npz or retarget_{robot}.npz - run the retarget step")


def load_objects(ep: Any) -> dict[str, dict[str, np.ndarray]]:
    index = ep.load_json("objects/index.json")
    if isinstance(index, Mapping):  # tolerate {"objects": [...]}
        index = index.get("objects", [])
    out: dict[str, dict[str, np.ndarray]] = {}
    for entry in index:
        oid = str(entry["id"])
        if ep.has(f"objects/{oid}/poses.npz"):
            out[oid] = ep.load_npz(f"objects/{oid}/poses")
    return out


def load_demo(ep: Any, robot: str, side: str, joints: Sequence[JointSpec], nv_hand: int,
              train: Mapping[str, Any]) -> DemoReference:
    """Build the :class:`DemoReference` (without kinematics) for the RL model's hand ``joints``."""
    timeline = ep.timeline()
    t_frames = np.asarray(timeline["t"], dtype=np.float64)
    rt = ep.load_npz(find_retarget(ep, robot, side))
    qpos = np.asarray(rt["qpos"], dtype=np.float64)
    T = qpos.shape[0]
    if T != len(t_frames):
        raise PipelineError(f"retarget qpos has {T} frames but timeline.npz has {len(t_frames)}")
    if "qpos_names" in rt:  # per-column names (steps/retarget.py)
        src_cols = [str(n) for n in np.asarray(rt["qpos_names"]).reshape(-1)]
    else:
        names = rt["joint_names"]
        names = [names] if isinstance(names, str) else [str(n) for n in np.asarray(names).reshape(-1)]
        widths = {j.name: JOINT_WIDTH[j.jtype][0] for j in joints}
        src_cols = expand_joint_columns(names, qpos.shape[1], widths)
    hand_q = qpos[:, map_columns(src_cols, column_names(joints))]
    hand_valid = np.all(np.isfinite(hand_q), axis=1)
    err = rt.get("err_m")
    if "valid" in rt and "hands" in rt:  # per-hand validity (steps/retarget.py)
        hands = [str(h) for h in np.asarray(rt["hands"]).reshape(-1)]
        if side not in hands:
            raise PipelineError(f"the retarget output has the hands {hands}, not {side!r}")
        h = hands.index(side)
        hand_valid &= np.asarray(rt["valid"], dtype=bool).reshape(T, -1)[:, h]
        if "err_m_hand" in rt:
            err = np.asarray(rt["err_m_hand"], dtype=np.float64).reshape(T, -1)[:, h]
    elif err is not None:
        hand_valid &= np.isfinite(np.asarray(err, dtype=np.float64))

    objects = load_objects(ep)
    slot = hand_slot(side, train.get("hand", "auto"))
    oid = choose_object(objects, slot, str(train.get("object", "auto")))
    poses = objects[oid]
    T_wo = np.asarray(poses["T_world_obj"], dtype=np.float64)
    obj_valid = np.asarray(poses["valid"], dtype=bool) & np.all(np.isfinite(T_wo.reshape(T_wo.shape[0], -1)), axis=1)
    grasped = np.asarray(poses["grasped"], dtype=bool)
    if T_wo.shape[0] != T:
        raise PipelineError(f"objects/{oid}/poses.npz has {T_wo.shape[0]} frames, retarget has {T}")

    both = hand_valid & obj_valid
    seg = train.get("segment_s", [0.0, 0.0]) or [0.0, 0.0]
    in_seg = np.ones(T, dtype=bool)
    if len(seg) == 2 and float(seg[1]) > float(seg[0]):
        in_seg = (t_frames >= float(seg[0])) & (t_frames <= float(seg[1]))
    idx = np.nonzero(both & in_seg)[0]
    if len(idx) < 2:
        raise PipelineError(f"fewer than 2 frames where both the hand and object {oid} are known")
    f0, f1 = int(idx[0]), int(idx[-1])
    sl = slice(f0, f1 + 1)
    t = t_frames[sl]
    hv, ov = hand_valid[sl], obj_valid[sl]
    hand_q = fill_hand_gaps(t, hand_q[sl], hv, joints)
    obj_pos = fill_gaps(t, T_wo[sl, :3, 3], ov)
    obj_quat = fill_quat_gaps(t, _mat_to_quat(T_wo[sl, :3, :3]), ov)
    grasp = grasped[sl, slot].astype(np.float64)

    dt = float(train.get("control_dt", 0.02))
    t_ctrl = control_times(float(t[0]), float(t[-1]), dt)
    sigma = float(train.get("smooth_sigma_s", 0.0)) / dt
    hand_q = smooth_hand(resample_hand(t, hand_q, t_ctrl, joints), sigma, joints)
    obj_pos = gaussian_smooth(resample_linear(t, obj_pos, t_ctrl), sigma)
    obj_quat = smooth_quats(resample_quat(t, obj_quat, t_ctrl), sigma)
    grasp = np.clip(resample_linear(t, grasp, t_ctrl), 0.0, 1.0)
    nearest = np.clip(np.searchsorted(t, t_ctrl), 0, len(t) - 1)
    observed = (hv & ov)[nearest]
    gap = max(longest_gap_s(t, hv), longest_gap_s(t, ov))
    max_gap = float(train.get("max_gap_s", 0.5))
    if gap > max_gap:
        log.warning("reference has a gap of %.2f s (> train.max_gap_s = %.2f s) filled by interpolation", gap, max_gap)
    return DemoReference(
        dt=dt, t=t_ctrl, hand_qpos=hand_q, hand_qvel=hand_qvel_from_qpos(hand_q, dt, joints, nv_hand),
        obj_pos=obj_pos, obj_quat=obj_quat, obj_linvel=finite_difference(obj_pos, dt),
        obj_angvel=angular_velocity_world(obj_quat, dt), grasp=grasp, observed=observed, object_id=oid,
        side=side, info={"frames": [f0, f1], "longest_gap_s": gap, "retarget_err_m": _nanmean(err, sl),
                         "observed_fraction": float(observed.mean())},
    )


def _nanmean(x: Any, sl: slice) -> float | None:
    if x is None:
        return None
    x = np.asarray(x, dtype=np.float64)[sl]
    return float(np.nanmean(x)) if np.isfinite(x).any() else None


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    from ..geometry import mat_to_quat

    return mat_to_quat(np.where(np.isfinite(R), R, np.eye(3)))
