"""Step ``physics``: ``sim/scene.xml`` + ``retarget_<robot>.npz`` -> ``derived/physics.json``.

Composes the scene with the retargeted robot hands (``sim/physics_<robot>.xml``; base driven by a
mocap body + soft weld, fingers by the robot's position actuators), replays the episode at the
scene's physics rate with free objects and compares the simulated objects with
``objects/<id>/poses.npz`` (see :mod:`glassesdata_pipeline.sim.physics_check` for the score).
Writes ``physics.json`` (per object: position/rotation deviation, lift success; max penetration;
``plausibility`` in [0, 1]) and ``sim/physics_traj.npz`` (``T_world_obj[T, K, 4, 4]``,
``object_ids``, ``penetration_m[T]``). Needs ``mujoco`` and the Menagerie (``[retarget]``).

``physics.json["status"]`` is ``"ok"`` or ``"inconclusive (...)"``: when the world is not
gravity-aligned (``camera_traj.npz`` meta, step ``world``: MuJoCo's -z gravity does not point down)
or when no object could be compared; ``plausibility`` is then None (the number computed anyway is
kept as ``plausibility_unchecked``).
"""

from __future__ import annotations

import logging

import numpy as np

from ..episode import Episode
from ..errors import PipelineError, require
from ..sim import mjcf, robots
from ..sim.data import NOT_ALIGNED, gravity_aligned, load_objects
from ..sim.physics_check import object_stats, plausibility, simulate
from .retarget import DEFAULT_ROBOT, menagerie_path

NAME = "physics"
VERSION = 2  # 2: status / inconclusive results
REQUIRES = ["scene", "retarget", "objects"]
INPUTS = ["derived/sim/scene.xml", "derived/sim/assets/*/*.obj", f"derived/retarget_{DEFAULT_ROBOT}.npz",
          "derived/objects/*/poses.npz", "derived/timeline.npz", "derived/camera_traj.npz"]
OUTPUTS = ["physics.json"]
CONFIG_SECTIONS = ["physics"]

log = logging.getLogger(__name__)


def run(ep: Episode, cfg) -> dict:
    pcfg = cfg.section("physics")
    rcfg = cfg.section("retarget")
    require("mujoco", "mujoco", "step physics")
    aligned = gravity_aligned(ep)
    if aligned is False:
        log.warning("physics: the world is not gravity-aligned (no IMU gravity): MuJoCo's gravity does not point "
                    "down in this scene; the result is marked %r", NOT_ALIGNED)
    robot_name = str(rcfg.get("robot", DEFAULT_ROBOT))
    robot = robots.get_robot(robot_name)
    if not ep.has("sim/scene.xml"):
        raise PipelineError("physics needs derived/sim/scene.xml (step scene)")
    ret = ep.load_npz(f"retarget_{robot_name}")
    sides = [str(s) for s in np.atleast_1d(ret["hands"])]
    composed = mjcf.compose(ep.file(f"sim/physics_{robot_name}.xml"), robot, sides, menagerie_path(rcfg),
                            scene_xml=ep.derived / "sim" / "scene.xml", base="mocap",
                            base_body=str(ret.get("base_body", "palm")),
                            weld_solref=list(pcfg.get("weld_solref", [0.02, 1.0])),
                            weld_solimp=list(pcfg.get("weld_solimp", [0.9, 0.95, 0.001, 0.5, 2.0])))
    t = np.asarray(ep.timeline()["t"], dtype=np.float64)
    objects = load_objects(ep, len(t))
    sim = simulate(composed, ret, objects, t, pcfg)
    T = sim["frames"]
    stats = {}
    for k, obj in enumerate(objects):
        stats[obj.id] = object_stats(sim["T_world_obj_sim"][:, k], obj.T_world_obj[:T], obj.valid[:T], obj.grasped[:T], pcfg)
    score, components = plausibility(stats, sim["max_penetration_m"], pcfg)
    status = components["status"]
    unchecked = score
    if aligned is False:
        status, score = NOT_ALIGNED, None
    elif score is None:
        log.warning("physics: %s", status)
    ep.save_npz("sim/physics_traj", ep.step_meta(backend="mujoco"), T_world_obj=sim["T_world_obj_sim"],
                object_ids=np.array(sim["object_ids"]), penetration_m=sim["penetration_m"], t=t[:T])
    ep.save_json("physics.json", {
        "robot": robot_name, "hands": sides, "model_xml": f"sim/physics_{robot_name}.xml", "frames": T,
        "timestep": sim["timestep"], "substeps_per_frame": sim["substeps"],
        "max_penetration_m": sim["max_penetration_m"], "plausibility": score, "status": status,
        "plausibility_unchecked": unchecked, "gravity_aligned": aligned, "components": components,
        "objects": stats,
        "score_formula": "mean_k exp(-pos_mean/pos_scale)*exp(-rot_mean/rot_scale) * exp(-max(0,pen-tol)/pen_scale)"
                         " * (0.5 + 0.5*lift_rate if the demo lifts an object)",
    }, ep.step_meta(backend="mujoco"))
    lifts = [s.get("lift_success") for s in stats.values() if s.get("demo_lifts")]
    return {"plausibility": None if score is None else round(score, 3), "status": status,
            "max_pen_mm": round(sim["max_penetration_m"] * 1000, 1),
            "lifts": f"{sum(bool(v) for v in lifts)}/{len(lifts)}"}
