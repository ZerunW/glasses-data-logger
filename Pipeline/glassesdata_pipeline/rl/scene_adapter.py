"""The interface between the RL task and the simulation package (:mod:`glassesdata_pipeline.sim`).

What the RL code needs from a MuJoCo model is collected in :class:`TaskModel` (plain numpy, so
the task math is testable without MuJoCo). It is built from a compiled ``mujoco.MjModel`` by
:func:`build_task_model`, which reads only standard ``mjModel`` fields (any object with the same
attribute names works - the tests use a fake one).

Where the model comes from (:func:`compose_rl_scene`):

1. ``train.scene.xml`` set: that file, with the names in ``[train.scene]`` (``root_body``,
   ``mocap_body``, ``tip_sites``, ``object_body``).
2. Otherwise ``derived/sim/scene.xml`` (step ``scene``) plus one floating robot hand added by
   :func:`glassesdata_pipeline.sim.mjcf.compose` with ``base="mocap"``: every robot name is
   prefixed with ``"<side>/"``, the base body (``base_body``: palm or forearm) carries the free
   joint ``<side>/base_free``, the mocap body ``<side>/base_target`` drives it through the weld
   ``<side>/base_weld`` (``relpose`` identity), fingertip sites come from the robot table's
   keypoints with role ``"tip"``; objects are bodies ``obj_<id>`` with free joints
   ``obj_<id>_free``; the table top height is ``table.top_z`` in ``derived/sim/scene.json``.
   ``base_body`` must match the retargeting model (``train.scene.base_body``, ``"auto"`` = the
   retarget file's ``base_body`` entry or config ``retarget.base_body``, else ``"palm"``).

Base modes:

* ``"mocap"``: the hand root has a free joint welded to a mocap body; the policy's first 6
  actions move the mocap target (position / rotation-vector residuals on the reference).
* ``"joints"``: the hand root has 3 slide + 3 hinge joints driven by position actuators (only for
  user-supplied models via ``train.scene.xml``); base residuals go through those actuators.

A free base without a weld or actuators cannot be controlled and is rejected.

Actuators must be position servos (``gaintype fixed``, ``biastype affine``, ``biasprm[1] < 0``) on
joints or fixed tendons; the reference targets are ``ctrl = ctrl_map @ hand_qpos`` (a tendon target
is ``gear * sum(coef_i * q_i)``, which covers the Shadow Hand's coupled J1+J2 tendons).
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import MissingDependency, PipelineError
from .reference import JOINT_WIDTH, DemoReference, JointSpec, quat_blocks, reference_ctrl

log = logging.getLogger(__name__)

# MuJoCo enum values (mjtJoint, mjtTrn, mjtWrap, mjtGain, mjtBias); checked against the installed
# mujoco in :func:`check_enums`.
JNT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
TRN_JOINT, TRN_JOINTINPARENT, TRN_TENDON = 0, 1, 3
WRAP_JOINT = 1
GAIN_FIXED, BIAS_AFFINE = 0, 1


@dataclass
class SceneNames:
    """Names inside the composed MJCF that the RL task uses."""

    xml: Path
    robot: str
    side: str
    root_body: str  # the floating hand root (carries the free joint or the 6 base joints)
    mocap_body: str | None  # mocap target welded to the root (base mode "mocap")
    tip_sites: list[str]
    object_body: str
    table_z: float = math.nan  # table top height (fall detection); NaN = unknown


@dataclass
class TaskModel:
    """Model facts for :class:`~glassesdata_pipeline.rl.task.TrackingTask` (numpy only)."""

    nq: int
    nv: int
    nu: int
    qpos0: np.ndarray
    hand_joints: list[JointSpec]
    hand_qadr: np.ndarray  # (nhq,) sim qpos address of every hand qpos column
    hand_vadr: np.ndarray  # (nhv,)
    hand_q_kind: list[str]  # per column: finger | base_pos | base_rot | base_quat
    hand_q_lo: np.ndarray
    hand_q_hi: np.ndarray
    finger_q_cols: np.ndarray
    finger_v_cols: np.ndarray
    base_v_cols: np.ndarray  # (6,)
    obj_qadr: np.ndarray  # (7,)
    obj_vadr: np.ndarray  # (6,)
    act_ids: np.ndarray  # (nu_h,) model actuator ids driven by the policy
    act_kind: list[str]  # finger | base_pos | base_rot
    ctrl_map: np.ndarray  # (nu_h, nhq)
    ctrl_lo: np.ndarray
    ctrl_hi: np.ndarray
    base_mode: str  # mocap | joints
    mocap_id: int = -1
    mocap_offset_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mocap_offset_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    root_body_id: int = -1
    object_body_id: int = -1
    tip_site_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    object_geom_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    hand_geom_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    table_z: float = math.nan
    timestep: float = 0.002
    decimation: int = 10
    names: dict[str, Any] = field(default_factory=dict)

    @property
    def quat_blocks(self) -> list[int]:
        return quat_blocks(self.hand_joints)

    @property
    def num_hand_q(self) -> int:
        return int(len(self.hand_qadr))

    def ctrl_scale(self, action: Any) -> np.ndarray:
        """Residual scale per policy-driven actuator: ``finger`` rad times the sum of the
        transmission coefficients (2x for a coupled J1+J2 tendon), ``base_pos`` m / ``base_rot`` rad
        for base actuators."""
        coef = np.abs(self.ctrl_map).sum(axis=1)
        per_kind = {"finger": action.finger, "base_pos": action.base_pos, "base_rot": action.base_rot}
        return np.array([per_kind[k] * max(c, 1e-6) for k, c in zip(self.act_kind, coef)], dtype=np.float64)

    def qpos_noise_std(self, noise: Any) -> np.ndarray:
        per_kind = {"finger": noise.finger_qpos, "base_pos": noise.base_pos, "base_rot": noise.base_rot,
                    "base_quat": 0.5 * noise.base_rot}
        return np.array([per_kind[k] for k in self.hand_q_kind], dtype=np.float64)


# =============================================================================================
# TaskModel from mjModel fields
# =============================================================================================


def resolve_name(names: Sequence[str], wanted: str, what: str) -> int:
    """Index of ``wanted`` in ``names``: exact match, else the unique name ending in ``/wanted``
    or ``wanted`` (prefixed models)."""
    names = list(names)
    if wanted in names:
        return names.index(wanted)
    hits = [k for k, n in enumerate(names) if n.endswith("/" + wanted)] or \
           [k for k, n in enumerate(names) if n.endswith(wanted)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise PipelineError(f"no {what} named {wanted!r} in the model")
    raise PipelineError(f"{what} {wanted!r} is ambiguous: {[names[k] for k in hits]}")


def subtree_bodies(parent: Sequence[int], root: int) -> set[int]:
    """Body ids in the subtree of ``root`` (``parent``: ``body_parentid``)."""
    out = {root}
    changed = True
    parent = [int(p) for p in parent]
    while changed:
        changed = False
        for b, p in enumerate(parent):
            if b not in out and p in out and b != p:
                out.add(b)
                changed = True
    return out


def actuator_ctrl_map(m: Any, act_ids: Sequence[int], col_of_qadr: Mapping[int, int], nhq: int) -> np.ndarray:
    """``(len(act_ids), nhq)``: actuator target as a linear function of the hand qpos."""
    A = np.zeros((len(act_ids), nhq))
    for row, a in enumerate(act_ids):
        trn = int(m.actuator_trntype[a])
        gear = float(np.asarray(m.actuator_gear)[a][0])
        target = int(np.asarray(m.actuator_trnid)[a][0])
        if trn in (TRN_JOINT, TRN_JOINTINPARENT):
            jt = JNT_TYPES[int(m.jnt_type[target])]
            if jt not in ("hinge", "slide"):
                raise PipelineError(f"actuator {a}: joint transmission on a {jt} joint is not supported")
            A[row, col_of_qadr[int(m.jnt_qposadr[target])]] = gear
        elif trn == TRN_TENDON:
            adr, num = int(m.tendon_adr[target]), int(m.tendon_num[target])
            for w in range(adr, adr + num):
                if int(m.wrap_type[w]) != WRAP_JOINT:
                    raise PipelineError(f"actuator {a}: tendon {target} is spatial; only fixed tendons (joint "
                                        "couplings) are supported - replace it with joint position actuators")
                j = int(m.wrap_objid[w])
                A[row, col_of_qadr[int(m.jnt_qposadr[j])]] += gear * float(m.wrap_prm[w])
        else:
            raise PipelineError(f"actuator {a}: transmission type {trn} is not supported (joint or fixed tendon only)")
    return A


def _actuator_targets_hand(m: Any, a: int, hand_jnts: set[int]) -> bool | None:
    """True/False whether actuator ``a`` acts on the hand; None for transmissions we cannot read."""
    trn = int(m.actuator_trntype[a])
    target = int(np.asarray(m.actuator_trnid)[a][0])
    if trn in (TRN_JOINT, TRN_JOINTINPARENT):
        return target in hand_jnts
    if trn == TRN_TENDON:
        adr, num = int(m.tendon_adr[target]), int(m.tendon_num[target])
        joints = {int(m.wrap_objid[w]) for w in range(adr, adr + num) if int(m.wrap_type[w]) == WRAP_JOINT}
        if not joints:
            return None
        return bool(joints & hand_jnts)
    return None


def build_task_model(m: Any, names: Mapping[str, Sequence[str]], scene: SceneNames, control_dt: float,
                     mocap_offset: tuple[np.ndarray, np.ndarray] | None = None) -> TaskModel:
    """:class:`TaskModel` from ``mjModel`` fields.

    Args:
        m: a ``mujoco.MjModel`` (or an object with the same fields).
        names: ``{"body": [...], "joint": [...], "site": [...], "actuator": [...]}`` by id.
        scene: the names to look up.
        control_dt: policy period; ``decimation = round(control_dt / timestep)``.
        mocap_offset: ``T_mocap_root`` as ``(pos, quat)`` (identity when None).
    """
    root = resolve_name(names["body"], scene.root_body, "body")
    hand_bodies = subtree_bodies(m.body_parentid, root)
    njnt = int(m.njnt)
    hand_jnt_ids = sorted((j for j in range(njnt) if int(m.jnt_bodyid[j]) in hand_bodies),
                          key=lambda j: int(m.jnt_qposadr[j]))
    if not hand_jnt_ids:
        raise PipelineError(f"the hand root {scene.root_body!r} has no joints in its subtree")
    joints: list[JointSpec] = []
    qadr: list[int] = []
    vadr: list[int] = []
    kinds: list[str] = []
    lo: list[float] = []
    hi: list[float] = []
    finger_q: list[int] = []
    finger_v: list[int] = []
    base_v: list[int] = []
    col_of_qadr: dict[int, int] = {}
    root_types: list[str] = []
    for j in hand_jnt_ids:
        jt = JNT_TYPES[int(m.jnt_type[j])]
        nq_j, nv_j = JOINT_WIDTH[jt]
        q0, v0 = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        on_root = int(m.jnt_bodyid[j]) == root
        spec = JointSpec(str(names["joint"][j]), jt, len(qadr), len(vadr))
        joints.append(spec)
        for k in range(nq_j):
            col_of_qadr[q0 + k] = len(qadr)
            qadr.append(q0 + k)
        if jt == "free":
            if not on_root:
                raise PipelineError(f"free joint {spec.name!r} inside the hand (only the root may be free)")
            kinds += ["base_pos"] * 3 + ["base_quat"] * 4
            lo += [-math.inf] * 7
            hi += [math.inf] * 7
        elif jt == "ball":
            raise PipelineError(f"ball joint {spec.name!r}: not supported in the hand")
        else:
            limited = bool(int(np.asarray(m.jnt_limited)[j]))
            rng = np.asarray(m.jnt_range)[j]
            lo.append(float(rng[0]) if limited else -math.inf)
            hi.append(float(rng[1]) if limited else math.inf)
            if on_root:
                kinds.append("base_pos" if jt == "slide" else "base_rot")
            else:
                kinds.append("finger")
                finger_q.append(spec.q_off)
                finger_v.append(spec.v_off)
        for k in range(nv_j):
            if on_root:
                base_v.append(len(vadr))
            vadr.append(v0 + k)
        if on_root:
            root_types.append(jt)

    if root_types == ["free"]:
        base_mode = "mocap"
    elif sorted(root_types) == ["hinge"] * 3 + ["slide"] * 3:
        base_mode = "joints"
    else:
        raise PipelineError(f"hand root {scene.root_body!r} has joints {root_types}; expected one free joint "
                            "(welded to a mocap body) or 3 slide + 3 hinge base joints")
    mocap_id = -1
    if base_mode == "mocap":
        if not scene.mocap_body:
            raise PipelineError("the hand base is a free joint but no mocap body is given: compose the model with "
                                "base='mocap' (a free base without a weld or actuators cannot be controlled)")
        mb = resolve_name(names["body"], scene.mocap_body, "body")
        mocap_id = int(m.body_mocapid[mb])
        if mocap_id < 0:
            raise PipelineError(f"body {scene.mocap_body!r} is not a mocap body")

    hand_jnt_set = set(hand_jnt_ids)
    act_ids: list[int] = []
    for a in range(int(m.nu)):
        hit = _actuator_targets_hand(m, a, hand_jnt_set)
        if hit is None:
            raise PipelineError(f"actuator {names['actuator'][a]!r}: unsupported transmission "
                                f"(type {int(m.actuator_trntype[a])}); use joint or fixed-tendon position actuators")
        if hit:
            act_ids.append(a)
    for a in act_ids:
        gain = int(m.actuator_gaintype[a])
        bias = int(m.actuator_biastype[a])
        if gain != GAIN_FIXED or bias != BIAS_AFFINE or not float(np.asarray(m.actuator_biasprm)[a][1]) < 0:
            raise PipelineError(f"actuator {names['actuator'][a]!r} is not a position servo "
                                "(gaintype fixed, biastype affine, biasprm[1] < 0)")
    A = actuator_ctrl_map(m, act_ids, col_of_qadr, len(qadr))
    base_q_set = {c for c, k in enumerate(kinds) if k != "finger"}
    act_kind = []
    for row in range(len(act_ids)):
        cols = set(np.nonzero(A[row])[0].tolist())
        if cols & base_q_set:
            if not cols <= base_q_set or len(cols) != 1:
                raise PipelineError(f"actuator {names['actuator'][act_ids[row]]!r} couples base and finger joints")
            act_kind.append(kinds[next(iter(cols))])
        else:
            act_kind.append("finger")
    ctrl_lo = np.full(len(act_ids), -math.inf)
    ctrl_hi = np.full(len(act_ids), math.inf)
    for row, a in enumerate(act_ids):
        if bool(int(np.asarray(m.actuator_ctrllimited)[a])):
            ctrl_lo[row], ctrl_hi[row] = (float(v) for v in np.asarray(m.actuator_ctrlrange)[a])
    if base_mode == "joints" and sorted(k for k in act_kind if k != "finger") != ["base_pos"] * 3 + ["base_rot"] * 3:
        raise PipelineError("base mode 'joints' needs one position actuator per base joint")

    ob = resolve_name(names["body"], scene.object_body, "body")
    oj = [j for j in range(njnt) if int(m.jnt_bodyid[j]) == ob]
    if len(oj) != 1 or JNT_TYPES[int(m.jnt_type[oj[0]])] != "free":
        raise PipelineError(f"object body {scene.object_body!r} must have exactly one free joint")
    oq, ov = int(m.jnt_qposadr[oj[0]]), int(m.jnt_dofadr[oj[0]])
    tips = np.array([resolve_name(names["site"], s, "site") for s in scene.tip_sites], dtype=np.int64)
    if len(tips) == 0:
        raise PipelineError("no fingertip sites")
    geom_body = np.asarray(m.geom_bodyid)
    collides = (np.asarray(m.geom_contype) != 0) | (np.asarray(m.geom_conaffinity) != 0)
    hand_geoms = np.array([g for g in range(int(m.ngeom)) if int(geom_body[g]) in hand_bodies and collides[g]], dtype=np.int64)
    obj_geoms = np.array([g for g in range(int(m.ngeom)) if int(geom_body[g]) == ob and collides[g]], dtype=np.int64)
    timestep = float(m.opt.timestep)
    decimation = max(1, int(round(control_dt / timestep)))
    if abs(decimation * timestep - control_dt) > 1e-6:
        log.warning("control_dt %.4f s is not a multiple of the timestep %.4f s; using %d substeps (%.4f s)",
                    control_dt, timestep, decimation, decimation * timestep)
    off_pos, off_quat = mocap_offset if mocap_offset is not None else (np.zeros(3), np.array([1.0, 0, 0, 0]))
    return TaskModel(
        nq=int(m.nq), nv=int(m.nv), nu=int(m.nu), qpos0=np.asarray(m.qpos0, dtype=np.float64).copy(),
        hand_joints=joints, hand_qadr=np.array(qadr, dtype=np.int64), hand_vadr=np.array(vadr, dtype=np.int64),
        hand_q_kind=kinds, hand_q_lo=np.array(lo), hand_q_hi=np.array(hi),
        finger_q_cols=np.array(finger_q, dtype=np.int64), finger_v_cols=np.array(finger_v, dtype=np.int64),
        base_v_cols=np.array(base_v, dtype=np.int64), obj_qadr=np.arange(oq, oq + 7), obj_vadr=np.arange(ov, ov + 6),
        act_ids=np.array(act_ids, dtype=np.int64), act_kind=act_kind, ctrl_map=A, ctrl_lo=ctrl_lo, ctrl_hi=ctrl_hi,
        base_mode=base_mode, mocap_id=mocap_id, mocap_offset_pos=np.asarray(off_pos, dtype=np.float64),
        mocap_offset_quat=np.asarray(off_quat, dtype=np.float64), root_body_id=root, object_body_id=ob,
        tip_site_ids=tips, object_geom_ids=obj_geoms, hand_geom_ids=hand_geoms, table_z=float(scene.table_z),
        timestep=timestep, decimation=decimation,
        names={"root_body": names["body"][root], "object_body": names["body"][ob],
               "tip_sites": [names["site"][s] for s in tips], "actuators": [names["actuator"][a] for a in act_ids],
               "hand_joints": [j.name for j in joints]},
    )


def clip_reference_to_limits(hand_qpos: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, float]:
    """Reference hand qpos clipped to the joint limits; also returns the largest violation (rad/m)."""
    q = np.asarray(hand_qpos, dtype=np.float64)
    clipped = np.minimum(np.maximum(q, lo), hi)
    return clipped, float(np.max(np.abs(clipped - q))) if q.size else 0.0


# =============================================================================================
# MuJoCo-dependent parts (Linux machine)
# =============================================================================================


def require_mujoco() -> Any:
    try:
        import mujoco
    except ImportError as error:
        raise MissingDependency("pip install mujoco (needed for RL training)") from error
    return mujoco


def check_enums(mujoco: Any) -> None:
    """Fail loudly if the installed MuJoCo numbers its enums differently from the constants here."""
    expected = {
        "mjJNT_FREE": (mujoco.mjtJoint, 0), "mjJNT_BALL": (mujoco.mjtJoint, 1),
        "mjJNT_SLIDE": (mujoco.mjtJoint, 2), "mjJNT_HINGE": (mujoco.mjtJoint, 3),
        "mjTRN_JOINT": (mujoco.mjtTrn, TRN_JOINT), "mjTRN_JOINTINPARENT": (mujoco.mjtTrn, TRN_JOINTINPARENT),
        "mjTRN_TENDON": (mujoco.mjtTrn, TRN_TENDON), "mjWRAP_JOINT": (mujoco.mjtWrap, WRAP_JOINT),
        "mjGAIN_FIXED": (mujoco.mjtGain, GAIN_FIXED), "mjBIAS_AFFINE": (mujoco.mjtBias, BIAS_AFFINE),
    }
    for name, (enum, value) in expected.items():
        if int(getattr(enum, name)) != value:
            raise PipelineError(f"MuJoCo enum {name} = {int(getattr(enum, name))}, expected {value}")


def model_names(mujoco: Any, m: Any) -> dict[str, list[str]]:
    def names(obj: Any, n: int) -> list[str]:
        return [mujoco.mj_id2name(m, obj, i) or "" for i in range(n)]

    return {"body": names(mujoco.mjtObj.mjOBJ_BODY, m.nbody), "joint": names(mujoco.mjtObj.mjOBJ_JOINT, m.njnt),
            "site": names(mujoco.mjtObj.mjOBJ_SITE, m.nsite), "actuator": names(mujoco.mjtObj.mjOBJ_ACTUATOR, m.nu),
            "geom": names(mujoco.mjtObj.mjOBJ_GEOM, m.ngeom)}


def mocap_offset(mujoco: Any, m: Any, root_body: int, mocap_body: int) -> tuple[np.ndarray, np.ndarray]:
    """``T_mocap_root`` at ``qpos0`` with the mocap body at its model pose (the weld's rest pose
    when ``relpose`` is computed from ``qpos0`` or is the identity with coincident initial poses)."""
    from .reference import quat_conj, quat_mul, quat_rotate_inv

    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mid = int(m.body_mocapid[mocap_body])
    pm, qm = np.array(d.mocap_pos[mid]), np.array(d.mocap_quat[mid])
    pr, qr = np.array(d.xpos[root_body]), np.array(d.xquat[root_body])
    return quat_rotate_inv(qm, pr - pm), quat_mul(quat_conj(qm), qr)


def attach_kinematics(mujoco: Any, m: Any, tm: TaskModel, ref: DemoReference) -> DemoReference:
    """Fill ``base_pos``, ``base_quat``, ``tips_pos`` (forward kinematics of the reference) and
    ``ctrl`` (actuator targets); clips the reference hand qpos to the joint limits first."""
    ref.hand_qpos, violation = clip_reference_to_limits(ref.hand_qpos, tm.hand_q_lo, tm.hand_q_hi)
    if violation > 0.05:
        log.warning("reference hand qpos exceeds joint limits by up to %.3f (clipped)", violation)
    ref.info["limit_violation"] = violation
    d = mujoco.MjData(m)
    T = ref.num_steps
    base_pos, base_quat = np.zeros((T, 3)), np.zeros((T, 4))
    tips = np.zeros((T, len(tm.tip_site_ids), 3))
    for k in range(T):
        d.qpos[:] = tm.qpos0
        d.qpos[tm.hand_qadr] = ref.hand_qpos[k]
        d.qpos[tm.obj_qadr] = np.concatenate([ref.obj_pos[k], ref.obj_quat[k]])
        mujoco.mj_kinematics(m, d)
        base_pos[k] = d.xpos[tm.root_body_id]
        base_quat[k] = d.xquat[tm.root_body_id]
        tips[k] = d.site_xpos[tm.tip_site_ids]
    ref.base_pos, ref.base_quat, ref.tips_pos = base_pos, base_quat, tips
    ref.ctrl = reference_ctrl(ref.hand_qpos, tm.ctrl_map)
    return ref


def warp_compat(mujoco: Any, m: Any) -> list[str]:
    """Adjust ``m`` (in place) for MuJoCo Warp 4.0.1 and return notes. Raises for what has no
    fallback. Unsupported per the MJWarp docs / enums: the PGS solver (-> Newton), plugins,
    DISTANCE equalities, the MIDPHASE / AUTORESET disable flags and the OVERRIDE / FWDINV / ISLAND
    enable flags (cleared)."""
    notes: list[str] = []
    if int(getattr(m, "nplugin", 0)) > 0:
        raise PipelineError("the model uses plugins, which MuJoCo Warp does not support")
    dist = getattr(mujoco.mjtEq, "mjEQ_DISTANCE", None)
    if dist is not None and int(m.neq) and np.any(np.asarray(m.eq_type) == int(dist)):
        raise PipelineError("DISTANCE equality constraints are not supported by MuJoCo Warp")
    if int(m.opt.solver) == int(mujoco.mjtSolver.mjSOL_PGS):
        m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        notes.append("solver PGS -> Newton")
    for flag in ("mjDSBL_MIDPHASE", "mjDSBL_AUTORESET"):
        bit = getattr(mujoco.mjtDisableBit, flag, None)
        if bit is not None and int(m.opt.disableflags) & int(bit):
            m.opt.disableflags = int(m.opt.disableflags) & ~int(bit)
            notes.append(f"cleared {flag}")
    for flag in ("mjENBL_OVERRIDE", "mjENBL_FWDINV", "mjENBL_ISLAND"):
        bit = getattr(mujoco.mjtEnableBit, flag, None)
        if bit is not None and int(m.opt.enableflags) & int(bit):
            m.opt.enableflags = int(m.opt.enableflags) & ~int(bit)
            notes.append(f"cleared {flag}")
    mesh = int(mujoco.mjtGeom.mjGEOM_MESH)
    colliding_mesh = int(np.sum((np.asarray(m.geom_type) == mesh) &
                                ((np.asarray(m.geom_contype) != 0) | (np.asarray(m.geom_conaffinity) != 0))))
    if colliding_mesh:
        notes.append(f"{colliding_mesh} colliding mesh geoms (convex-hull GJK/EPA in MJWarp; mesh-mesh pairs are "
                     "the slowest - prefer primitive hand collision geoms)")
    return notes


@dataclass
class LoadedScene:
    mj_model: Any
    task_model: TaskModel
    reference: DemoReference
    names: SceneNames
    notes: list[str] = field(default_factory=list)


def _table_top(ep: Any) -> float:
    try:
        info = ep.load_json("sim/scene.json")
        table = info.get("table") or {}
        if table.get("top_z") is not None:
            return float(table["top_z"])
    except PipelineError:
        pass
    log.warning("table top height unknown (derived/sim/scene.json has no table); fall detection is off")
    return math.nan


def _base_body(ep: Any, train: Mapping[str, Any], retarget_name: str | None) -> str:
    wanted = str((train.get("scene") or {}).get("base_body", "auto"))
    if wanted != "auto":
        return wanted
    if retarget_name is not None:
        rt = ep.load_npz(retarget_name)
        if isinstance(rt.get("base_body"), str):
            return rt["base_body"]
        params = (rt.get("meta") or {}).get("params") or {}
        for key in ("base_body", "retarget.base_body"):
            if isinstance(params.get(key), str):
                return params[key]
        if isinstance((params.get("retarget") or {}).get("base_body"), str):
            return params["retarget"]["base_body"]
    value = ep.config.get("retarget.base_body")
    return str(value) if isinstance(value, str) else "palm"


def compose_rl_scene(ep: Any, robot_name: str, side: str, object_id: str, train: Mapping[str, Any],
                     out_dir: Path) -> SceneNames:
    """Write (or locate) the RL model and return the names in it (see the module docstring)."""
    scfg = dict(train.get("scene") or {})
    from ..sim import mjcf, robots

    obj_body = mjcf.object_body(mjcf.sanitize_id(object_id))
    if scfg.get("xml"):
        xml = Path(str(scfg["xml"])).expanduser()
        xml = xml if xml.is_absolute() else ep.path / xml
        return SceneNames(xml=xml, robot=robot_name, side=side, root_body=str(scfg.get("root_body") or f"{side}/base"),
                          mocap_body=scfg.get("mocap_body") or None, tip_sites=list(scfg.get("tip_sites") or []),
                          object_body=str(scfg.get("object_body") or obj_body), table_z=_table_top(ep))
    if not ep.has("sim/scene.xml"):
        raise PipelineError(f"{ep.path}: no derived/sim/scene.xml - run the scene step (or set train.scene.xml)")
    robot = robots.get_robot(robot_name)
    menagerie = Path(str(scfg.get("menagerie_path") or ep.config.get("retarget.menagerie_path") or "~/mujoco_menagerie")).expanduser()
    if not menagerie.is_dir() and os.environ.get("MUJOCO_MENAGERIE"):  # same fallback as steps/retarget.py
        menagerie = Path(os.environ["MUJOCO_MENAGERIE"]).expanduser()
    if not menagerie.is_dir():
        raise PipelineError(f"MuJoCo Menagerie not found at {menagerie}: set retarget.menagerie_path "
                            "(or train.scene.menagerie_path, or $MUJOCO_MENAGERIE)")
    from .reference import find_retarget

    try:
        rt_name = find_retarget(ep, robot_name, side)
    except PipelineError:
        rt_name = None
    base_body = _base_body(ep, train, rt_name)
    out = Path(out_dir) / f"scene_{robot_name}_{side}.xml"
    composed = mjcf.compose(out, robot, [side], menagerie, scene_xml=ep.derived / "sim" / "scene.xml",
                            base="mocap", base_body=base_body, model_name=f"rl_{robot_name}_{side}")
    hand = composed.hands[side]
    tips = [name for _, kind, name, role in hand.keypoints if role == "tip" and kind == "site"]
    return SceneNames(xml=composed.xml, robot=robot_name, side=side, root_body=hand.base_body,
                      mocap_body=hand.mocap_body, tip_sites=tips, object_body=obj_body, table_z=_table_top(ep))


def load_scene(ep: Any, robot_name: str, side: str, train: Mapping[str, Any], out_dir: Path,
               for_warp: bool = False) -> LoadedScene:
    """Compose the model, build the :class:`TaskModel` and the reference with kinematics."""
    mujoco = require_mujoco()
    check_enums(mujoco)
    from . import reference as refmod
    from .task import parse_task_config

    cfg = parse_task_config(train)
    objects = refmod.load_objects(ep)
    oid = refmod.choose_object(objects, refmod.hand_slot(side, train.get("hand", "auto")), str(train.get("object", "auto")))
    names = compose_rl_scene(ep, robot_name, side, oid, train, out_dir)
    m = mujoco.MjModel.from_xml_path(str(names.xml))
    notes = warp_compat(mujoco, m) if for_warp else []
    nm = model_names(mujoco, m)
    offset = None
    if names.mocap_body:
        offset = mocap_offset(mujoco, m, resolve_name(nm["body"], names.root_body, "body"),
                              resolve_name(nm["body"], names.mocap_body, "body"))
    tm = build_task_model(m, nm, names, cfg.control_dt, offset)
    train_ref = dict(train)
    train_ref["object"] = oid
    ref = refmod.load_demo(ep, robot_name, side, tm.hand_joints, len(tm.hand_vadr), train_ref)
    attach_kinematics(mujoco, m, tm, ref)
    (Path(out_dir) / "scene_names.json").write_text(json.dumps({
        "xml": str(names.xml), "robot": robot_name, "side": side, "object": oid, "base_mode": tm.base_mode,
        "decimation": tm.decimation, "timestep": tm.timestep, "table_z": tm.table_z, **tm.names,
        "reference": {k: v for k, v in ref.info.items()}, "steps": ref.num_steps, "notes": notes,
    }, indent=2, default=str))
    return LoadedScene(m, tm, ref, names, notes)
