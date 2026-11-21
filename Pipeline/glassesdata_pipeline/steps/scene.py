"""Step ``scene``: reconstructed episode -> MuJoCo scene ``derived/sim/scene.xml`` (+ ``sim/assets/``),
``sim/scene.json`` and the kinematic replay ``sim/replay.mp4`` / ``sim/replay_sheet.png``.

Inputs (only derived files, never ``gt/``): ``camera_traj.npz`` (head camera), ``hands_world.npz``
(21 mocap spheres per hand), ``arms.npz`` (shoulder/elbow/wrist mocap spheres; optional),
``objects/`` (one free body per object at its first valid pose, meshes, mass, friction) and
``objects/table.json`` (a box from the table top down to the floor; floor = top -
``scene.table_height_m``). Without a table, an inferred support box under the objects (top = their
lowest resting bottom, ``scene.json`` ``table.support = "inferred"``), so they do not fall at the
start. ``scene.json`` also records ``gravity_aligned`` (step ``world``; a warning when False).
See :mod:`glassesdata_pipeline.sim.mjcf` for the names in the file.

Robots are not part of ``scene.xml``; ``retarget``/``physics``/RL add them with
:func:`glassesdata_pipeline.sim.mjcf.compose`.

The replay runs MuJoCo's renderer when ``mujoco`` is installed (``scene.replay = "auto"``) and a
numpy wireframe renderer of the same scene otherwise (e.g. on the development Mac).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from .. import geometry as geo
from ..episode import Episode
from ..errors import PipelineError
from ..sim import mjcf, replay
from ..sim.data import EpisodeData, ObjectData, gravity_aligned, load_episode_data
from ..sim.frames import fovy_deg

NAME = "scene"
VERSION = 2  # 2: inferred support box without a table, gravity_aligned
REQUIRES = ["world", "objects"]
INPUTS = [
    "derived/camera.json", "derived/camera_traj.npz", "derived/hands_world.npz", "derived/arms.npz",
    "derived/objects/index.json", "derived/objects/table.json", "derived/objects/*/poses.npz",
    "derived/objects/*/props.json", "derived/objects/*/mesh.obj", "derived/objects/*/collision/*.obj", "video.mp4",
]
OUTPUTS = ["sim/scene.xml", "sim/scene.json"]
CONFIG_SECTIONS = ["scene"]

log = logging.getLogger(__name__)


def _friction(value: Any, default: float) -> tuple[float, float, float]:
    if value is None:
        return (float(default), 0.005, 0.0001)
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64)).reshape(-1)
    if arr.size >= 3:
        return (float(arr[0]), float(arr[1]), float(arr[2]))
    return (float(arr[0]), 0.005, 0.0001)


def object_spec(obj: ObjectData, k: int, density: float, friction: float) -> mjcf.ObjectSpec | None:
    """Mass properties and initial pose of one object (None when it never has a valid pose)."""
    f = obj.first_valid
    if f is None:
        log.warning("object %s has no valid pose; left out of the scene", obj.id)
        return None
    parts = obj.collision or ([obj.mesh] if obj.mesh is not None else [])
    volume, centroid, inertia_unit = 0.0, np.zeros(3), np.zeros((3, 3))
    moments = []
    for V, F in parts:
        v, c, I = mjcf.mesh_mass_properties(V, F)
        if v > 0:
            moments.append((v, c, I))
    if moments:
        volume = sum(v for v, _, _ in moments)
        centroid = sum(v * c for v, c, _ in moments) / volume
        inertia_unit = sum(I + v * (float((c - centroid) @ (c - centroid)) * np.eye(3) - np.outer(c - centroid, c - centroid))
                           for v, c, I in moments)
    props = obj.props or {}
    mass = props.get("mass_kg")
    if mass is None or not (float(mass) > 0):
        vol = props.get("volume_m3") or volume or float(np.prod(obj.extents()))
        mass = density * float(vol)
    mass = float(mass)
    com = np.asarray(props.get("com", centroid), dtype=np.float64).reshape(3)
    if volume > 0:
        I_c = inertia_unit * (mass / volume)
        inertia = mjcf.shift_inertia(I_c, mass, com - centroid)
    else:
        inertia = mjcf.box_inertia(mass, obj.extents())
    w = np.linalg.eigvalsh(0.5 * (inertia + inertia.T))
    if w.min() <= 1e-12:
        inertia = inertia + np.eye(3) * max(1e-9, 1e-3 * float(w.max()))
    return mjcf.ObjectSpec(
        id=obj.id, pose=obj.T_world_obj[f], visual_mesh=obj.mesh_path, collision_meshes=list(obj.collision_paths),
        mass=mass, com=com, inertia=inertia, friction=_friction(props.get("friction"), friction),
        rgba=mjcf.object_color(k), size=obj.extents(),
    )


def support_box(objects: list[mjcf.ObjectSpec], floor_z: float, margin: float) -> mjcf.TableSpec | None:
    """Without a measured table: a box whose top is the objects' lowest resting bottom (their
    first valid poses), covering their footprints plus ``margin``, down to ``floor_z`` (or 5 cm
    under the top when that is lower). None without objects."""
    if not objects:
        return None
    corners = [geo.transform(o.pose, mjcf.box_mesh(o.size)[0]) for o in objects]
    top = float(min(np.min(c[:, 2]) for c in corners))
    box = mjcf.table_from_polygon(np.concatenate([c[:, :2] for c in corners]), top, top - min(floor_z, top - 0.05))
    box.size_xy = box.size_xy + 2.0 * margin
    return box


def render_size(camera: geo.Camera | None, height: int) -> tuple[int, int]:
    if camera is None:
        return int(round(height * 9 / 16)) // 2 * 2, int(height) // 2 * 2
    w = int(round(camera.width * height / camera.height))
    return max(2, w // 2 * 2), max(2, int(height) // 2 * 2)


def scene_spec(ep: Episode, data: EpisodeData, scfg: Any) -> tuple[mjcf.SceneSpec, dict[str, Any]]:
    """The :class:`mjcf.SceneSpec` of an episode plus notes for ``scene.json``."""
    notes: dict[str, Any] = {}
    density = float(scfg.get("default_density", 500.0))
    friction = float(scfg.get("default_friction", 0.8))
    objects = [s for k, obj in enumerate(data.objects) if (s := object_spec(obj, k, density, friction)) is not None]
    f_cam = data.first_valid_camera()
    head = data.T_world_cam[f_cam] if f_cam is not None else None
    table = None
    table_height = float(scfg.get("table_height_m", 0.75))
    measured = (data.table or {})
    if measured.get("height_source") == "floor_plane" and measured.get("height_m"):
        table_height = float(measured["height_m"])  # the objects step saw the floor: use it
    if data.table_polygon is not None and data.table_top_z is not None and len(data.table_polygon) >= 3:
        table = mjcf.table_from_polygon(data.table_polygon, data.table_top_z, table_height)
        floor_z = table.floor_z
        notes["table"] = {"center_xy": table.center_xy.tolist(), "size_xy": table.size_xy.tolist(),
                          "yaw_deg": math.degrees(table.yaw), "top_z": table.top_z, "floor_z": table.floor_z,
                          "support": "measured"}
    else:
        cam_z = float(head[2, 3]) if head is not None else 0.0
        floor_z = cam_z - float(scfg.get("floor_below_camera_m", 1.45))
        table = support_box(objects, floor_z, float(scfg.get("support_margin_m", 0.10)))
        if table is None:
            notes["table"] = None
            log.warning("no table (objects/table.json) and no object; floor at z = %.3f m", floor_z)
        else:
            floor_z = table.floor_z
            notes["table"] = {"center_xy": table.center_xy.tolist(), "size_xy": table.size_xy.tolist(),
                              "yaw_deg": math.degrees(table.yaw), "top_z": table.top_z, "floor_z": table.floor_z,
                              "support": "inferred"}
            log.warning("no table (objects/table.json): an inferred support box under the objects, top at z = "
                        "%.3f m (their lowest resting bottom)", table.top_z)
    hands: dict[str, np.ndarray | None] = {}
    arms: dict[str, np.ndarray | None] = {}
    valid = data.hand_valid()
    for side, slot in replay.SIDE_SLOT.items():
        idx = np.flatnonzero(valid[:, slot])
        hands[side] = data.joints[idx[0], slot] if len(idx) else None
        if data.shoulder is not None and data.elbow is not None:
            pts = np.stack([data.shoulder[:, slot], data.elbow[:, slot],
                            data.wrist[:, slot] if data.wrist is not None else data.joints[:, slot, 0]], axis=1)
            ok = np.flatnonzero(np.all(np.isfinite(pts), axis=(-1, -2)))
            arms[side] = pts[ok[0]] if len(ok) else None
    fovy = fovy_deg(data.camera) if data.camera is not None else 60.0
    size = render_size(data.camera, int(scfg.get("replay_height", 640)))
    spec = mjcf.SceneSpec(
        name=f"glassesdata_{mjcf.sanitize_id(ep.path.name)}",
        timestep=float(scfg.get("timestep", 0.002)),
        option={"integrator": str(scfg.get("integrator", "implicitfast")), "cone": str(scfg.get("cone", "elliptic")),
                "impratio": float(scfg.get("impratio", 10.0))},
        floor_z=floor_z, table=table, objects=objects, hands=hands, arms=arms, head_pose=head, head_fovy=fovy,
        render_size=(max(size[0], 640), max(size[1], 480)), hand_radius=float(scfg.get("hand_radius_m", 0.008)),
        tip_radius=0.75 * float(scfg.get("hand_radius_m", 0.008)),
    )
    notes.update({"fovy_deg": fovy, "render_size": list(size), "head_first_frame": f_cam,
                  "hands_present": {s: v is not None for s, v in hands.items()},
                  "arms_present": {s: v is not None for s, v in arms.items()}})
    return spec, notes


def run(ep: Episode, cfg) -> dict:
    scfg = cfg.section("scene")
    data = load_episode_data(ep, with_gt=False)
    if data.T_world_cam is None:
        raise PipelineError("scene needs derived/camera_traj.npz (step world)")
    spec, notes = scene_spec(ep, data, scfg)
    notes["gravity_aligned"] = gravity_aligned(ep)
    if notes["gravity_aligned"] is False:
        log.warning("scene: the world is not gravity-aligned (no IMU gravity): MuJoCo's gravity (-z) does not "
                    "point down here; physics on this scene is inconclusive")
    sim_dir = ep.derived / "sim"
    info = mjcf.write_scene(spec, sim_dir)
    objects = []
    for o in spec.objects:
        w = np.linalg.eigvalsh(o.inertia)
        objects.append({"id": o.id, "body": mjcf.object_body(mjcf.sanitize_id(o.id)),
                        "joint": mjcf.object_joint(mjcf.sanitize_id(o.id)), "mass_kg": o.mass, "com": o.com.tolist(),
                        "principal_inertia": w.tolist(), "friction": list(o.friction),
                        "initial_pose": o.pose.tolist(), "collision_parts": len(o.collision_meshes)})
    summary: dict[str, Any] = {"objects": len(spec.objects), **{k: v for k, v in info.stats.items() if k != "geoms"}}
    replay_info: dict[str, Any] = {"renderer": None}
    kind = str(scfg.get("replay", "auto"))
    if kind != "none":
        size = tuple(notes["render_size"])
        geometry = replay.scene_geometry(data, spec.table, spec.floor_z)
        renderer = replay.make_renderer(kind, info.xml, geometry, spec.head_fovy, size, [o.id for o in data.objects])
        try:
            replay_info = replay.write_replay(ep, data, renderer, spec.head_fovy, size, sim_dir / "replay.mp4",
                                              int(scfg.get("replay_max_frames", 0)), bool(scfg.get("replay_undistort", True)),
                                              int(scfg.get("replay_crf", 20)))
            sheet_renderer = renderer if renderer.name == "mujoco" else replay.WireframeRenderer(geometry, spec.head_fovy, *size)
            replay.replay_sheet(ep, data, sheet_renderer, spec.head_fovy, size, sim_dir / "replay_sheet.png",
                                int(scfg.get("replay_sheet_frames", 6)))
            replay_info["sheet"] = "sim/replay_sheet.png"
        finally:
            if hasattr(renderer, "close"):
                renderer.close()
        summary["replay"] = f"{replay_info['renderer']} {replay_info['frames']} frames"
    ep.save_json("sim/scene.json", {
        "xml": "sim/scene.xml", "stats": info.stats, "frames": data.T, "fps": data.fps,
        "floor_z": spec.floor_z, "objects": objects, "replay": replay_info, **notes,
        "names": {
            "head_body": mjcf.HEAD_BODY, "head_camera": mjcf.HEAD_CAMERA, "overview_camera": mjcf.OVERVIEW_CAMERA,
            "table_body": mjcf.TABLE_BODY, "floor_geom": mjcf.FLOOR_GEOM,
            "hand_body": "hand_<side>_<joint name>", "arm_body": "arm_<side>_<shoulder|elbow|wrist>",
            "object_body": "obj_<id>", "object_joint": "obj_<id>_free",
        },
        "conventions": "head body = OpenCV camera frame (T_world_cam); head_cam quat 0 1 0 0 inside it; "
                       "geom groups: 0 floor/table, 1 human mocap, 2 object visuals, 3 object collision",
    }, ep.step_meta(backend=replay_info.get("renderer") or "none"))
    return summary
