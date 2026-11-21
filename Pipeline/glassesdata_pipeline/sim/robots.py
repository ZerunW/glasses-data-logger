"""Robot hand tables and the human -> robot keypoint mapping (numpy only).

Shadow Hand (MuJoCo Menagerie ``shadow_hand/right_hand.xml`` and ``left_hand.xml``; get them on
the Linux machine with ``git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie``
and set ``retarget.menagerie_path``). Facts read from those files (Menagerie ``main``, Nov 2026):

* Names: right hand prefix ``rh_``, left ``lh_``. Bodies ``{p}forearm`` (root, ``quat="0 1 0 1"``)
  -> ``{p}wrist`` (joint ``{p}WRJ2``) -> ``{p}palm`` (joint ``{p}WRJ1``, ``pos="0 0 0.034"``) -> per
  finger ``{p}ffknuckle`` (FFJ4) -> ``{p}ffproximal`` (FFJ3) -> ``{p}ffmiddle`` (FFJ2, +0.045 m)
  -> ``{p}ffdistal`` (FFJ1, +0.025 m); same for ``mf``/``rf``; the little finger starts with
  ``{p}lfmetacarpal`` (LFJ5); thumb ``{p}thbase`` (THJ5) -> ``{p}thproximal`` (THJ4) ->
  ``{p}thhub`` (THJ3, +0.038 m) -> ``{p}thmiddle`` (THJ2) -> ``{p}thdistal`` (THJ1, +0.032 m).
  24 hinge joints, 20 position actuators ``{p}A_*``; J1+J2 of FF/MF/RF/LF are driven together
  through the fixed tendons ``{p}FFJ0`` ... (actuators ``{p}A_FFJ0`` ..., ``ctrlrange 0..pi``).
  The files have no fingertip sites; :data:`SHADOW` adds ``{p}fftip`` ... at the distal tip
  (26 mm along the distal phalanx, 27.5 mm for the thumb: Shadow Hand segment lengths, which
  match the inertial centres 13.1 / 14.6 mm in the files). The right hand has a site
  ``grasp_site`` without prefix (the composer prefixes every name, so two hands do not clash).
* Palm frame ``{p}palm`` (origin on the WRJ1 axis): +z distal (knuckles at z = 0.095..0.099),
  -y palmar (the flexion axes are +x, so flexing moves the fingertips to -y; ``grasp_site`` is at
  y = -0.035), +x towards the index finger and thumb for the right hand (FF knuckle x = +0.033,
  thumb base x = +0.034) and towards the little finger for the left hand (FF at x = -0.033): the
  left palm is the mirror image in x. Relative to the canonical hand frame of
  :mod:`skeleton` (x radial for the right hand, y distal, z palmar; left = mirror) the palm frame
  is therefore ``x_c = x_p, y_c ~ z_p, z_c ~ -y_p`` for both hands. The exact rotation is taken
  as ``skeleton.wrist_frame`` of the robot's own rest keypoints (:func:`canonical_in_palm`), so a
  human hand shaped like the robot maps to the robot's rest pose exactly.

Keypoints (MediaPipe index -> robot frame): wrist -> palm body (position and orientation), thumb
CMC/MCP/IP -> ``thbase``/``thhub``/``thdistal`` origins, finger MCP/PIP/DIP -> ``*knuckle``
(``lfknuckle``) / ``*middle`` / ``*distal`` origins, tips -> the added tip sites.

Scaling (:func:`keypoint_targets`): ``mode = "wrist"`` scales each finger's keypoints, expressed in
the wrist frame, by ``s_f = L_robot,f / L_human,f`` where ``L`` is the rest chain length wrist ->
finger base -> ... -> tip; ``"base"`` anchors each finger at the robot's base joint and scales the
finger's own segments; ``"none"`` keeps human positions.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import geometry as geo
from .. import skeleton as sk
from ..errors import PipelineError

SIDES = ("left", "right")


@dataclass(frozen=True)
class Keypoint:
    """One human joint and the robot frame it maps to (``name`` may contain ``{p}``)."""

    joint: int
    kind: str  # "body" or "site"
    name: str
    role: str  # "wrist", "base", "middle", "tip"


@dataclass(frozen=True)
class AddedSite:
    """A site the composer adds to a robot body (e.g. fingertips)."""

    name: str
    body: str
    pos: tuple[float, float, float]
    size: float = 0.004


@dataclass(frozen=True)
class HandRobot:
    """A floating dexterous hand from the MuJoCo Menagerie."""

    name: str
    files: Mapping[str, str]  # side -> path relative to the Menagerie root
    prefixes: Mapping[str, str]  # side -> name prefix used inside the robot file
    palm_body: str
    forearm_body: str
    keypoints: tuple[Keypoint, ...]
    added_sites: tuple[AddedSite, ...]
    rest_keypoints_right: np.ndarray = field(repr=False)  # (21, 3) palm frame, from the XML
    angle_unit: str = "radian"

    def resolve(self, template: str, side: str) -> str:
        """``"{p}palm"`` -> ``"rh_palm"`` for the right hand."""
        if side not in self.prefixes:
            raise PipelineError(f"robot {self.name} has no {side!r} hand")
        return template.format(p=self.prefixes[side])

    def xml_path(self, menagerie: str | Path, side: str) -> Path:
        path = Path(menagerie).expanduser() / self.files[side]
        if not path.is_file():
            raise PipelineError(
                f"{path} not found: set retarget.menagerie_path to a clone of "
                "https://github.com/google-deepmind/mujoco_menagerie (git clone --depth 1 ...)"
            )
        return path

    def rest_keypoints(self, side: str) -> np.ndarray:
        """``(21, 3)`` robot keypoints at q = 0 in the palm frame (the left hand mirrors x)."""
        kp = np.array(self.rest_keypoints_right, dtype=np.float64)
        if side == "left":
            kp[:, 0] *= -1.0
        return kp

    def keypoint_frames(self, side: str) -> list[tuple[int, str, str, str]]:
        """``[(joint, kind, resolved_name, role)]`` for the 21 keypoints."""
        return [(k.joint, k.kind, self.resolve(k.name, side), k.role) for k in self.keypoints]


def _shadow_keypoints() -> tuple[Keypoint, ...]:
    kps = [Keypoint(0, "body", "{p}palm", "wrist")]
    kps += [Keypoint(1, "body", "{p}thbase", "base"), Keypoint(2, "body", "{p}thhub", "middle"),
            Keypoint(3, "body", "{p}thdistal", "middle"), Keypoint(4, "site", "{p}thtip", "tip")]
    for finger, (base, short) in zip(("index", "middle", "ring", "pinky"),
                                     (("ffknuckle", "ff"), ("mfknuckle", "mf"), ("rfknuckle", "rf"), ("lfknuckle", "lf"))):
        j = sk.FINGERS[finger]
        kps += [Keypoint(j[0], "body", "{p}" + base, "base"), Keypoint(j[1], "body", "{p}" + short + "middle", "middle"),
                Keypoint(j[2], "body", "{p}" + short + "distal", "middle"), Keypoint(j[3], "site", "{p}" + short + "tip", "tip")]
    return tuple(sorted(kps, key=lambda k: k.joint))


_S45 = math.sqrt(0.5)
_TH_DIR = np.array([_S45, 0.0, _S45])  # thumb chain direction in the right palm (thbase quat: 45 deg about y)
_TH0 = np.array([0.034, -0.00858, 0.029])

SHADOW_REST_RIGHT = np.array([
    [0.0, 0.0, 0.0],                                   # 0 palm origin (WRJ1 axis)
    _TH0, _TH0 + 0.038 * _TH_DIR, _TH0 + 0.070 * _TH_DIR, _TH0 + 0.0975 * _TH_DIR,  # thumb 1-4
    [0.033, 0.0, 0.095], [0.033, 0.0, 0.140], [0.033, 0.0, 0.165], [0.033, 0.0, 0.191],    # FF 5-8
    [0.011, 0.0, 0.099], [0.011, 0.0, 0.144], [0.011, 0.0, 0.169], [0.011, 0.0, 0.195],    # MF 9-12
    [-0.011, 0.0, 0.095], [-0.011, 0.0, 0.140], [-0.011, 0.0, 0.165], [-0.011, 0.0, 0.191],  # RF 13-16
    [-0.033, 0.0, 0.0865], [-0.033, 0.0, 0.1315], [-0.033, 0.0, 0.1565], [-0.033, 0.0, 0.1825],  # LF 17-20
])
"""Right Shadow Hand keypoints at q = 0 in the ``rh_palm`` frame, from the body ``pos``/``quat``
chain of ``right_hand.xml`` plus the tip sites (checked against the file with
:func:`rest_keypoints_from_xml`). At run time the retargeter recomputes them from the loaded model."""

SHADOW = HandRobot(
    name="shadow",
    files={"right": "shadow_hand/right_hand.xml", "left": "shadow_hand/left_hand.xml"},
    prefixes={"right": "rh_", "left": "lh_"},
    palm_body="{p}palm",
    forearm_body="{p}forearm",
    keypoints=_shadow_keypoints(),
    added_sites=(
        AddedSite("{p}thtip", "{p}thdistal", (0.0, 0.0, 0.0275)),
        AddedSite("{p}fftip", "{p}ffdistal", (0.0, 0.0, 0.026)),
        AddedSite("{p}mftip", "{p}mfdistal", (0.0, 0.0, 0.026)),
        AddedSite("{p}rftip", "{p}rfdistal", (0.0, 0.0, 0.026)),
        AddedSite("{p}lftip", "{p}lfdistal", (0.0, 0.0, 0.026)),
    ),
    rest_keypoints_right=SHADOW_REST_RIGHT,
)

ROBOTS: dict[str, HandRobot] = {"shadow": SHADOW}


def get_robot(name: str) -> HandRobot:
    try:
        return ROBOTS[name]
    except KeyError:
        raise PipelineError(f"unknown robot {name!r}; available: {', '.join(ROBOTS)}") from None


# =============================================================================================
# Rest keypoints from an MJCF file (numpy forward kinematics at q = 0)
# =============================================================================================


def _body_local(element: ET.Element) -> np.ndarray:
    pos = np.array([float(v) for v in element.get("pos", "0 0 0").split()])
    if element.get("quat") is not None:
        q = np.array([float(v) for v in element.get("quat").split()])
        R = geo.quat_to_mat(q)
    elif any(element.get(a) is not None for a in ("euler", "axisangle", "xyaxes", "zaxis")):
        raise PipelineError(f"body {element.get('name')!r}: only pos/quat orientations are supported here")
    else:
        R = np.eye(3)
    return geo.make_T(R, pos)


def rest_frames_from_xml(xml: str | Path) -> dict[str, np.ndarray]:
    """``{body name: T_world_body}`` of an MJCF model at q = 0 (``pos``/``quat`` chain only)."""
    text = Path(xml).read_text() if isinstance(xml, Path) or (isinstance(xml, str) and "<" not in xml) else xml
    root = ET.fromstring(text)
    world = root.find("worldbody")
    if world is None:
        raise PipelineError("MJCF has no <worldbody>")
    frames: dict[str, np.ndarray] = {}

    def walk(parent: ET.Element, T_parent: np.ndarray) -> None:
        for child in parent.findall("body"):
            T = T_parent @ _body_local(child)
            name = child.get("name")
            if name:
                frames[name] = T
            walk(child, T)

    walk(world, np.eye(4))
    return frames


def rest_keypoints_from_xml(robot: HandRobot, side: str, xml: str | Path) -> np.ndarray:
    """``(21, 3)`` keypoints at q = 0 in the palm frame, computed from the robot's MJCF file."""
    frames = rest_frames_from_xml(xml)
    sites = {robot.resolve(s.name, side): s for s in robot.added_sites}
    palm = frames[robot.resolve(robot.palm_body, side)]
    inv = geo.inv_T(palm)
    out = np.zeros((sk.N_JOINTS, 3))
    for joint, kind, name, _ in robot.keypoint_frames(side):
        if kind == "body":
            p = frames[name][:3, 3]
        else:
            site = sites[name]
            p = geo.transform(frames[robot.resolve(site.body, side)], np.array(site.pos))
        out[joint] = geo.transform(inv, p)
    return out


# =============================================================================================
# Keypoint mapping and scaling
# =============================================================================================


def chain_lengths(keypoints: Any) -> np.ndarray:
    """``(5,)`` rest chain lengths wrist -> base -> ... -> tip of the five fingers from ``(21, 3)``
    keypoints (thumb, index, middle, ring, pinky)."""
    kp = np.asarray(keypoints, dtype=np.float64)
    out = np.zeros(5)
    for f, name in enumerate(sk.FINGER_NAMES):
        idx = [0] + sk.FINGERS[name]
        out[f] = float(np.sum(np.linalg.norm(np.diff(kp[idx], axis=0), axis=-1)))
    return out


def finger_lengths(keypoints: Any) -> np.ndarray:
    """``(5,)`` rest lengths base -> tip (without the wrist -> base segment)."""
    kp = np.asarray(keypoints, dtype=np.float64)
    out = np.zeros(5)
    for f, name in enumerate(sk.FINGER_NAMES):
        idx = sk.FINGERS[name]
        out[f] = float(np.sum(np.linalg.norm(np.diff(kp[idx], axis=0), axis=-1)))
    return out


def human_chain_lengths(bone_lengths: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(chain, finger)`` lengths ``(5,)`` each from the 20 bone lengths (bone i-1 joins joint i to
    ``PARENTS[i]``)."""
    b = np.asarray(bone_lengths, dtype=np.float64).reshape(20)
    chain = np.zeros(5)
    finger = np.zeros(5)
    for f, name in enumerate(sk.FINGER_NAMES):
        idx = sk.FINGERS[name]
        bones = [b[i - 1] for i in idx]  # base bone (from the wrist) then the three segments
        chain[f] = float(np.sum(bones))
        finger[f] = float(np.sum(bones[1:]))
    return chain, finger


def canonical_in_palm(rest_palm: Any, side: str) -> np.ndarray:
    """``T_palm_canon``: the canonical hand frame (``skeleton.wrist_frame``) of the robot's rest
    keypoints, expressed in the palm frame."""
    return skeleton_wrist_frame(np.asarray(rest_palm, dtype=np.float64), side)


def skeleton_wrist_frame(joints: Any, side: str) -> np.ndarray:
    return sk.wrist_frame(joints, side)


@dataclass
class HandMapping:
    """Everything needed to turn human joints into robot targets for one hand."""

    side: str
    T_palm_canon: np.ndarray  # (4, 4)
    rest_canon: np.ndarray  # (21, 3) robot rest keypoints in its canonical frame
    chain_scale: np.ndarray  # (5,) L_robot / L_human (wrist -> tip)
    finger_scale: np.ndarray  # (5,) robot / human (base -> tip)
    mode: str = "wrist"

    @property
    def T_canon_palm(self) -> np.ndarray:
        return geo.inv_T(self.T_palm_canon)


def hand_mapping(rest_palm: Any, side: str, human_bone_lengths: Any, mode: str = "wrist",
                 scale_limits: tuple[float, float] = (0.5, 2.0)) -> HandMapping:
    """Build the mapping from the robot rest keypoints (palm frame) and the person's bone lengths."""
    if mode not in ("wrist", "base", "none"):
        raise PipelineError(f"retarget.scaling must be wrist, base or none, got {mode!r}")
    rest_palm = np.asarray(rest_palm, dtype=np.float64)
    T_palm_canon = canonical_in_palm(rest_palm, side)
    rest_canon = geo.transform(geo.inv_T(T_palm_canon), rest_palm)
    human_chain, human_finger = human_chain_lengths(human_bone_lengths)
    lo, hi = scale_limits
    with np.errstate(divide="ignore", invalid="ignore"):
        chain_scale = np.clip(chain_lengths(rest_canon) / human_chain, lo, hi)
        finger_scale = np.clip(finger_lengths(rest_canon) / human_finger, lo, hi)
    chain_scale = np.where(np.isfinite(chain_scale), chain_scale, 1.0)
    finger_scale = np.where(np.isfinite(finger_scale), finger_scale, 1.0)
    return HandMapping(side, T_palm_canon, rest_canon, chain_scale, finger_scale, mode)


def keypoint_targets(joints_world: Any, mapping: HandMapping) -> tuple[np.ndarray, np.ndarray]:
    """Robot targets for one frame: ``(T_world_palm (4, 4), targets_world (21, 3))``.

    ``joints_world`` ``(21, 3)``: the human hand. The palm target is the human canonical wrist
    frame composed with ``T_canon_palm``; keypoint targets are the human keypoints in the wrist
    frame, scaled per finger (see module docstring), mapped back to the world.
    """
    J = np.asarray(joints_world, dtype=np.float64)
    T_world_canon = sk.wrist_frame(J, mapping.side)
    local = geo.transform(geo.inv_T(T_world_canon), J)
    out = local.copy()
    out[0] = 0.0
    for f, name in enumerate(sk.FINGER_NAMES):
        idx = sk.FINGERS[name]
        if mapping.mode == "wrist":
            out[idx] = mapping.chain_scale[f] * local[idx]
        elif mapping.mode == "base":
            out[idx] = mapping.rest_canon[idx[0]] + mapping.finger_scale[f] * (local[idx] - local[idx[0]])
    T_world_palm = T_world_canon @ mapping.T_canon_palm
    return T_world_palm, geo.transform(T_world_canon, out)


def keypoint_weights(robot: HandRobot, weights: Mapping[str, float]) -> np.ndarray:
    """``(21,)`` position weights from role weights ``{"wrist", "base", "middle", "tip"}``."""
    w = np.zeros(sk.N_JOINTS)
    for k in robot.keypoints:
        w[k.joint] = float(weights.get(k.role, 0.0))
    return w


# =============================================================================================
# qpos bookkeeping
# =============================================================================================

FREE_COLUMNS = ("px", "py", "pz", "qw", "qx", "qy", "qz")
BALL_COLUMNS = ("qw", "qx", "qy", "qz")


def qpos_column_names(joints: Sequence[tuple[str, str, int]], nq: int) -> list[str]:
    """Per-column names of a qpos vector from ``[(joint name, type, qposadr)]`` where type is
    ``free``/``ball``/``slide``/``hinge``: a free joint ``j`` gives ``j:px ... j:qz``."""
    names = [""] * nq
    for name, jtype, adr in joints:
        if jtype == "free":
            for k, suffix in enumerate(FREE_COLUMNS):
                names[adr + k] = f"{name}:{suffix}"
        elif jtype == "ball":
            for k, suffix in enumerate(BALL_COLUMNS):
                names[adr + k] = f"{name}:{suffix}"
        else:
            names[adr] = name
    return names
