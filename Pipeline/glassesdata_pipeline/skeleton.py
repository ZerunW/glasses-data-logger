"""Hand topology, body profile, hand rest pose / forward kinematics and the arm model.

Joint order and parents follow MediaPipe (PIPELINE.md section 1)::

    0 wrist; thumb 1 CMC, 2 MCP, 3 IP, 4 tip; index 5 MCP, 6 PIP, 7 DIP, 8 tip;
    middle 9-12; ring 13-16; pinky 17-20.   Bone i (1..20) joins joint i to PARENTS[i].

Canonical hand frame (used by :func:`rest_pose` and as the wrist frame ``T_world_wrist`` in
:func:`pose_hand`), wrist joint at the origin:

* +y: distal, from the wrist towards the middle-finger MCP joint (exactly through joint 9),
* +z: palmar, the normal pointing out of the palm (fingers flex towards +z),
* +x = y x z: radial, towards the thumb, for the **right** hand.

The **left** hand is the mirror image: its canonical coordinates are the right hand's with x
negated (the thumb is on -x). The frame stays right-handed, so ``T_world_wrist`` is a proper
rigid transform for both hands; :func:`wrist_frame` recovers it from 21 joints.

Bone lengths (section 4): from ``hand_length_m`` (L, wrist crease to middle fingertip) with the
Buchholz et al. (1992) ratios. The phalanges use the ratios directly. The wrist-to-MCP bones
(5, 9, 13, 17) are metacarpal ratio + a carpal offset ``c = 1 - sum(middle-finger ratios)``
(= 0.072 L), so that a straight middle finger spans exactly L from the wrist crease (the joint-0
position of MediaPipe); the raw metacarpal ratios measure CMC to MCP and would leave the middle
chain at 0.928 L. The thumb CMC is placed by a fixed template offset (scaled by L). Palm width
rescales the lateral (x) spacing of the MCP joints.

Arm model (used by the synthetic generator; the ``arms`` step should use the same):

* :func:`shoulders_from_head`: shoulder = eye (camera centre) + ``Rz(yaw) @ [-back, +-width/2, -drop]``
  in the gravity-aligned world (z up), yaw = heading of the optical axis (:func:`geometry.yaw_of`).
* :func:`elbow_from_swivel`: the elbow lies on the circle of the two spheres (shoulder, upper arm)
  and (wrist, forearm). Swivel 0 = lowest point of the circle; positive swivel turns the elbow
  outward (to the person's right for the right arm, to the left for the left arm).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import geometry as geo

# =============================================================================================
# Topology
# =============================================================================================

N_JOINTS = 21
PARENTS: list[int] = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
JOINT_NAMES: list[str] = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
FINGER_NAMES: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
FINGERS: dict[str, list[int]] = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}
"""Joint indices of each finger from its base joint (thumb CMC / finger MCP) to the tip."""
BONES: list[tuple[int, int]] = [(PARENTS[i], i) for i in range(1, N_JOINTS)]
"""Bone i-1 (0-based array index) joins ``(PARENTS[i], i)``."""
SLOTS = {"left": 0, "right": 1}

# =============================================================================================
# Profile
# =============================================================================================

DEFAULT_PROFILE: dict[str, float] = {
    "hand_length_m": 0.187,
    "palm_length_m": 0.107,
    "palm_width_m": 0.084,
    "forearm_length_m": 0.26,
    "upper_arm_length_m": 0.32,
    "shoulder_width_m": 0.39,
    "eye_to_shoulder_drop_m": 0.20,
    "eye_to_shoulder_back_m": 0.07,
}
PROFILE_KEYS = tuple(DEFAULT_PROFILE)

#: Buchholz et al. (1992) segment lengths as fractions of hand length. Thumb: MC, PP, DP;
#: fingers: MC, PP, MP, DP. (PIPELINE.md asks to verify these against the paper.)
BUCHHOLZ: dict[str, tuple[float, ...]] = {
    "thumb": (0.251, 0.196, 0.158),
    "index": (0.374, 0.265, 0.143, 0.097),
    "middle": (0.373, 0.277, 0.170, 0.108),
    "ring": (0.336, 0.259, 0.165, 0.107),
    "pinky": (0.295, 0.206, 0.117, 0.093),
}
CARPAL_FRACTION = 1.0 - sum(BUCHHOLZ["middle"])  # 0.072: wrist crease to the CMC line

# Template at the default hand (L = 0.187 m, palm width 0.084 m), right hand, canonical frame.
_REF_L = DEFAULT_PROFILE["hand_length_m"]
_REF_W = DEFAULT_PROFILE["palm_width_m"]
_MCP_SPACING = 0.019  # metres between neighbouring MCP joints at the reference palm width
_MCP_LATERAL = {"index": 1.0, "middle": 0.0, "ring": -1.0, "pinky": -2.0}  # in units of spacing
_FAN = {"index": -0.08, "middle": 0.0, "ring": 0.06, "pinky": 0.14}  # rest direction, rad about z
_ABDUCTION_SIGN = {"thumb": -1.0, "index": -1.0, "middle": -1.0, "ring": 1.0, "pinky": 1.0}
_THUMB_CMC = np.array([0.022, 0.028, 0.008])  # metres at the reference hand length
_THUMB_AXIS = np.array([math.sin(math.radians(40)) * math.cos(math.radians(15)),
                        math.cos(math.radians(40)) * math.cos(math.radians(15)),
                        math.sin(math.radians(15))])
_THUMB_FLEX = np.array([-0.5, -0.1, 1.0])  # direction the thumb tip moves when it flexes


def complete_profile(profile: Mapping[str, Any] | None) -> dict[str, Any]:
    """All profile keys, defaults (section 4) filled in for missing or invalid values.

    The result keeps ``measured`` (as given, or ``{}``) and adds ``defaulted``: the keys that
    took their default value here.
    """
    profile = dict(profile or {})
    out: dict[str, Any] = {"version": int(profile.get("version", 1))}
    defaulted: list[str] = []
    for key, default in DEFAULT_PROFILE.items():
        value = profile.get(key)
        try:
            value = float(value)
            ok = math.isfinite(value) and value > 0
        except (TypeError, ValueError):
            ok = False
        if ok:
            out[key] = value
        else:
            out[key] = default
            defaulted.append(key)
    measured = profile.get("measured")
    out["measured"] = dict(measured) if isinstance(measured, Mapping) else {}
    out["defaulted"] = defaulted
    return out


def _profile_values(profile: Mapping[str, Any] | None) -> tuple[float, float]:
    p = complete_profile(profile)
    return p["hand_length_m"], p["palm_width_m"]


# =============================================================================================
# Hand model
# =============================================================================================


def _finger_bases(L: float, W: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Base joint position and base rotation (columns x, y, z) of each finger, right hand."""
    bases: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    spacing = _MCP_SPACING * W / _REF_W
    for name in ("index", "middle", "ring", "pinky"):
        reach = (BUCHHOLZ[name][0] + CARPAL_FRACTION) * L  # wrist crease to MCP
        x = _MCP_LATERAL[name] * spacing
        x = float(np.clip(x, -0.9 * reach, 0.9 * reach))
        position = np.array([x, math.sqrt(reach * reach - x * x), 0.0])
        R = geo.rot_z(_FAN[name])  # columns: x radial, y along the finger, z palmar
        bases[name] = (position, R)
    y = _THUMB_AXIS / np.linalg.norm(_THUMB_AXIS)
    z = _THUMB_FLEX - y * float(_THUMB_FLEX @ y)
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    bases["thumb"] = (_THUMB_CMC * (L / _REF_L), np.stack([x, y, z], axis=-1))
    return bases


def _segment_lengths(L: float) -> dict[str, np.ndarray]:
    """Lengths of the three moving segments of each finger (thumb: MC, PP, DP; others PP, MP, DP)."""
    return {
        name: np.array(BUCHHOLZ[name][:3] if name == "thumb" else BUCHHOLZ[name][1:]) * L
        for name in FINGER_NAMES
    }


def _pose_local(L: float, W: float, flexion: np.ndarray, abduction: np.ndarray) -> np.ndarray:
    """21 joints of the right hand in the canonical frame."""
    joints = np.zeros((N_JOINTS, 3))
    bases = _finger_bases(L, W)
    lengths = _segment_lengths(L)
    for f, name in enumerate(FINGER_NAMES):
        index = FINGERS[name]
        position, R = bases[name]
        joints[index[0]] = position
        R = R @ geo.rot_z(_ABDUCTION_SIGN[name] * abduction[f]) @ geo.rot_x(flexion[f, 0])
        p = position
        for k in range(3):
            if k:
                R = R @ geo.rot_x(flexion[f, k])
            p = p + R[:, 1] * lengths[name][k]
            joints[index[k + 1]] = p
    return joints


def _mirror(joints: np.ndarray, side: str) -> np.ndarray:
    if side not in SLOTS:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if side == "left":
        joints = joints.copy()
        joints[..., 0] *= -1.0
    return joints


def rest_pose(profile: Mapping[str, Any] | None = None, side: str = "right") -> np.ndarray:
    """``(21, 3)`` flat open hand (all angles zero) in the canonical frame (wrist at the origin)."""
    L, W = _profile_values(profile)
    return _mirror(_pose_local(L, W, np.zeros((5, 3)), np.zeros(5)), side)


def bone_lengths(profile: Mapping[str, Any] | None = None) -> np.ndarray:
    """``(20,)`` lengths of bones 1..20 (bone i joins joint i to ``PARENTS[i]``), metres."""
    joints = rest_pose(profile)
    return np.array([np.linalg.norm(joints[i] - joints[PARENTS[i]]) for i in range(1, N_JOINTS)])


def pose_hand(
    profile: Mapping[str, Any] | None,
    side: str,
    T_world_wrist: Any,
    flexion: Any = None,
    abduction: Any = None,
) -> np.ndarray:
    """Forward kinematics: ``(21, 3)`` joints in the world.

    Args:
        profile: body profile (missing keys take defaults).
        side: ``"left"`` or ``"right"``.
        T_world_wrist: 4x4 pose of the canonical hand frame (see module docstring).
        flexion: ``(5, 3)`` radians per finger (thumb, index, middle, ring, pinky); thumb: CMC, MCP,
            IP; fingers: MCP, PIP, DIP. Positive flexes towards the palm. Default zeros.
        abduction: ``(5,)`` radians at the base joint; positive spreads the finger away from the
            middle finger (index and middle radially, ring and pinky ulnarly, the thumb away from
            the index). Default zeros.
    """
    L, W = _profile_values(profile)
    flexion = np.zeros((5, 3)) if flexion is None else np.asarray(flexion, dtype=np.float64).reshape(5, 3)
    abduction = np.zeros(5) if abduction is None else np.asarray(abduction, dtype=np.float64).reshape(5)
    local = _mirror(_pose_local(L, W, flexion, abduction), side)
    return geo.transform(np.asarray(T_world_wrist, dtype=np.float64), local)


def wrist_frame(joints: Any, side: str) -> np.ndarray:
    """``T_world_wrist`` ``(..., 4, 4)`` from joints ``(..., 21, 3)`` (inverse of :func:`pose_hand`
    for the palm: uses joints 0, 5, 9, 17, which finger angles do not move)."""
    J = np.asarray(joints, dtype=np.float64)
    origin = J[..., 0, :]
    y = J[..., 9, :] - origin
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    lateral = J[..., 5, :] - J[..., 17, :]
    if side == "left":
        lateral = -lateral
    x = lateral - y * np.sum(lateral * y, axis=-1, keepdims=True)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    z = np.cross(x, y)
    return geo.make_T(np.stack([x, y, z], axis=-1), origin)


# =============================================================================================
# Arm model
# =============================================================================================


def shoulders_from_head(T_world_cam: Any, profile: Mapping[str, Any] | None = None) -> np.ndarray:
    """Shoulder positions ``(..., 2, 3)`` (slot 0 left, 1 right) from camera poses ``(..., 4, 4)``
    in a gravity-aligned world (z up)."""
    p = complete_profile(profile)
    T = np.asarray(T_world_cam, dtype=np.float64)
    yaw = geo.yaw_of(T[..., :3, :3])
    Rz = geo.rot_z(yaw)
    half = p["shoulder_width_m"] / 2.0
    offsets = np.array([
        [-p["eye_to_shoulder_back_m"], half, -p["eye_to_shoulder_drop_m"]],
        [-p["eye_to_shoulder_back_m"], -half, -p["eye_to_shoulder_drop_m"]],
    ])
    return T[..., None, :3, 3] + np.einsum("...ij,kj->...ki", Rz, offsets)


def _elbow_circle(shoulder: np.ndarray, wrist: np.ndarray, upper: float, fore: float, side: str):
    d_vec = wrist - shoulder
    d = np.linalg.norm(d_vec, axis=-1, keepdims=True)
    n = d_vec / d
    along = (d * d + upper * upper - fore * fore) / (2.0 * d)  # distance from shoulder to circle centre
    centre = shoulder + n * along
    radius = np.sqrt(np.clip(upper * upper - along * along, 0.0, None))
    down = np.array([0.0, 0.0, -1.0])
    u = down - n * np.sum(down * n, axis=-1, keepdims=True)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    v = np.cross(n, u)  # for an arm pointing forward and u down, n x u points to the person's left
    if side == "right":
        v = -v
    return centre, radius, u, v


def elbow_from_swivel(
    shoulder: Any, wrist: Any, upper_arm: float, forearm: float, swivel: Any, side: str
) -> np.ndarray:
    """Elbow ``(..., 3)`` on the two-sphere circle at swivel angle ``swivel`` (radians; 0 = lowest
    point, positive = outward). When the wrist is out of reach the circle collapses (radius 0)."""
    shoulder = np.asarray(shoulder, dtype=np.float64)
    wrist = np.asarray(wrist, dtype=np.float64)
    centre, radius, u, v = _elbow_circle(shoulder, wrist, upper_arm, forearm, side)
    phi = np.asarray(swivel, dtype=np.float64)[..., None]
    return centre + radius * (np.cos(phi) * u + np.sin(phi) * v)


def swivel_of_elbow(shoulder: Any, wrist: Any, elbow: Any, side: str) -> np.ndarray:
    """Inverse of :func:`elbow_from_swivel`: the swivel angle (radians) of a given elbow."""
    shoulder = np.asarray(shoulder, dtype=np.float64)
    wrist = np.asarray(wrist, dtype=np.float64)
    elbow = np.asarray(elbow, dtype=np.float64)
    d_vec = wrist - shoulder
    n = d_vec / np.linalg.norm(d_vec, axis=-1, keepdims=True)
    down = np.array([0.0, 0.0, -1.0])
    u = down - n * np.sum(down * n, axis=-1, keepdims=True)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    v = np.cross(n, u)
    if side == "right":
        v = -v
    e = elbow - shoulder
    return np.arctan2(np.sum(e * v, axis=-1), np.sum(e * u, axis=-1))


def reach_ratio(shoulder: Any, wrist: Any, upper_arm: float, forearm: float) -> np.ndarray:
    """``|wrist - shoulder| / (upper + forearm)``; must stay below 1 for a reachable wrist."""
    d = np.linalg.norm(np.asarray(wrist, dtype=np.float64) - np.asarray(shoulder, dtype=np.float64), axis=-1)
    return d / (upper_arm + forearm)
