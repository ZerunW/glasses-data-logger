"""Step ``retarget``: ``hands_world.npz`` -> ``derived/retarget_<robot>.npz`` (floating robot hands).

Needs ``mujoco`` + ``mink`` and a MuJoCo Menagerie clone (:func:`menagerie_path`: by default
``local/src/mujoco_menagerie`` in the repository, where ``scripts/setup_ubuntu.sh`` clones it).
The robot model used is written to ``derived/sim/robot_<robot>.xml`` (:func:`sim.mjcf.compose` with
``base="free"``: every robot name prefixed with ``<side>/``, a free joint ``<side>/base_free`` on
the palm, fingertip sites added).

Output arrays (PIPELINE.md section 5 plus extras):

* ``qpos[T, nq]`` - qpos of ``sim/robot_<robot>.xml``; ``joint_names[njnt]`` (MuJoCo joint order,
  prefixed), ``joint_qposadr[njnt]``, ``qpos_names[nq]`` (per column; a free joint ``j`` gives
  ``j:px .. j:qz``) - map columns into any composed model (scene + robot) by name,
* ``err_m[T]`` - mean keypoint distance robot vs scaled human targets over the valid hands
  (NaN when none), ``err_m_hand[T, H]``, ``valid[T, H]``, ``hands[H]``, ``robot``,
* ``model_xml`` (relative to ``derived/``), ``base_body`` (palm/forearm), ``base_joints[H]``, ``palm_bodies[H]``,
  ``finger_scale[H, 5]``, ``t[T]``, ``fps``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .. import paths
from ..episode import Episode
from ..errors import PipelineError, require
from ..sim import mjcf, robots
from ..sim.retarget_mink import MINK_PIN, choose_hands, retarget

NAME = "retarget"
VERSION = 2  # 2: the best IK iterate per frame
REQUIRES = ["world"]
INPUTS = ["derived/hands_world.npz", "derived/profile.json", "derived/timeline.npz"]
DEFAULT_ROBOT = "shadow"
OUTPUTS = [f"retarget_{DEFAULT_ROBOT}.npz"]  # the runner needs a static list: only the default robot so far
CONFIG_SECTIONS = ["retarget"]

log = logging.getLogger(__name__)


def menagerie_path(rcfg) -> Path:
    """``retarget.menagerie_path`` (relative to the repository root), or ``$MUJOCO_MENAGERIE`` when
    that is empty or does not exist, else ``$GLASSESDATA_SRC`` or ``local/src`` + ``/mujoco_menagerie``
    (:func:`paths.menagerie_dir`)."""
    path = paths.menagerie_dir(rcfg.get("menagerie_path", ""))
    if not path.is_dir():
        raise PipelineError(f"MuJoCo Menagerie not found at {path}: git clone --depth 1 "
                            "https://github.com/google-deepmind/mujoco_menagerie there (scripts/setup_ubuntu.sh does), "
                            "or set retarget.menagerie_path or $MUJOCO_MENAGERIE")
    return path


def human_bone_lengths(hands: dict, profile: dict) -> np.ndarray:
    """``(2, 20)`` bone lengths: ``hands_world.bone_lengths`` where finite, else the profile's."""
    default = np.asarray(profile.get("bone_lengths_m"), dtype=np.float64).reshape(20)
    if "bone_lengths" in hands:
        b = np.asarray(hands["bone_lengths"], dtype=np.float64).reshape(2, 20)
        return np.where(np.isfinite(b) & (b > 0), b, default)
    return np.stack([default, default])


def run(ep: Episode, cfg) -> dict:
    rcfg = cfg.section("retarget")
    robot_name = str(rcfg.get("robot", DEFAULT_ROBOT))
    if robot_name != DEFAULT_ROBOT:
        raise PipelineError(f"retarget.robot = {robot_name!r}: only {DEFAULT_ROBOT!r} is wired into the step's OUTPUTS")
    robot = robots.get_robot(robot_name)
    require("mujoco", "mujoco", "step retarget")
    require("mink", MINK_PIN, "step retarget")
    menagerie = menagerie_path(rcfg)
    hands_npz = ep.load_npz("hands_world")
    joints = np.asarray(hands_npz["joints"], dtype=np.float64)
    valid = np.all(np.isfinite(joints), axis=(-1, -2))
    sides = choose_hands(valid, rcfg.get("hands", "auto"), float(rcfg.get("min_valid_fraction", 0.2)))
    if not sides:
        raise PipelineError("no hand is valid often enough to retarget (retarget.min_valid_fraction)")
    model_rel = f"sim/robot_{robot_name}.xml"
    composed = mjcf.compose(ep.file(model_rel), robot, sides, menagerie, base="free",
                            base_body=str(rcfg.get("base_body", "palm")), model_name=f"{robot_name}_hands")
    bones = human_bone_lengths(hands_npz, ep.profile())
    result = retarget(composed, robot, joints, valid, bones, rcfg)
    t = np.asarray(ep.timeline()["t"], dtype=np.float64)[: len(joints)]
    fps = float(1.0 / np.median(np.diff(t))) if len(t) > 1 else 0.0
    ep.save_npz(f"retarget_{robot_name}", ep.step_meta(backend=f"mink:{robot_name}"), **result, robot=robot_name,
                model_xml=model_rel, base_body=str(rcfg.get("base_body", "palm")), t=t, fps=fps)
    err = result["err_m"]
    finite = err[np.isfinite(err)]
    return {"hands": ",".join(sides), "nq": int(result["qpos"].shape[1]),
            "err_mm_median": round(float(np.median(finite)) * 1000, 2) if len(finite) else None,
            "err_mm_p95": round(float(np.percentile(finite, 95)) * 1000, 2) if len(finite) else None}
