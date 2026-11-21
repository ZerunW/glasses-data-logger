"""MJCF generation in pure Python (``xml.etree``; no MuJoCo needed to write or check the files).

Scene (:func:`build_scene`, :func:`write_scene`) - ``derived/sim/scene.xml`` + ``sim/assets/``:

* ``<compiler angle="radian" meshdir="assets" autolimits="true">``, gravity -z, timestep from
  ``[scene]``; ``<visual><global offwidth/offheight>`` large enough for the replay renderer.
* ``floor`` (plane at ``floor_z``) and the static body ``table`` (one box from the table top down
  to the floor, footprint = oriented bounding rectangle of the table polygon).
* Per object ``obj_<id>``: free joint ``obj_<id>_free`` at the object's first valid pose,
  explicit ``<inertial>`` (mass/COM from ``props.json``, inertia from the collision parts), visual
  mesh geom ``obj_<id>_visual`` (class ``gd_visual``, group 2, no contacts) and collision geoms
  ``obj_<id>_col<k>`` (class ``gd_collision``, group 3) with the object's friction.
* Human (kinematic replay only, no contacts, group 1): mocap bodies ``hand_<side>_<joint name>``
  (21 per hand, :data:`skeleton.JOINT_NAMES`) and ``arm_<side>_shoulder|elbow|wrist``, each with
  a small sphere.
* ``head``: mocap body at ``T_world_cam`` (OpenCV axes) carrying the camera ``head_cam``
  (``quat = 0 1 0 0`` converts to MuJoCo's camera axes, ``fovy`` from the intrinsics) and a small
  glasses box; plus a fixed ``overview`` camera looking at the table.

Robots are not in ``scene.xml``. :func:`compose` writes a new file with the scene (or an empty
world) plus floating Menagerie hands: every robot name is prefixed with ``<side>/`` (classes,
materials, meshes, bodies, joints, geoms, sites, tendons, actuators, ...), mesh files become
absolute paths, the floating base is a ``<freejoint>`` on the palm (default; the forearm, wrist
bodies and the wrist joints/actuators are dropped) or on the forearm, fingertip sites are added,
and with ``base="mocap"`` a mocap body ``<side>/base_target`` drives the base through a soft
``<weld>`` (``<side>/base_weld``). :func:`check_references` verifies that every name a file uses
is defined (a Mac-side stand-in for MuJoCo's compiler on these files).
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import geometry as geo
from .. import skeleton as sk
from ..errors import PipelineError
from .frames import QUAT_CV_MJ, fmt, look_at_mujoco, pos_quat
from .robots import HandRobot

log = logging.getLogger(__name__)

SIDES = ("left", "right")
HEAD_BODY = "head"
HEAD_CAMERA = "head_cam"
OVERVIEW_CAMERA = "overview"
TABLE_BODY = "table"
TABLE_GEOM = "table_top"
FLOOR_GEOM = "floor"
ARM_PARTS = ("shoulder", "elbow", "wrist")
HIDDEN_Z_OFFSET = 5.0  # mocap bodies without data are parked this far below the floor


def object_body(oid: str) -> str:
    return f"obj_{oid}"


def object_joint(oid: str) -> str:
    return f"obj_{oid}_free"


def hand_body(side: str, joint: int) -> str:
    return f"hand_{side}_{sk.JOINT_NAMES[joint]}"


def arm_body(side: str, part: str) -> str:
    return f"arm_{side}_{part}"


def sanitize_id(oid: str) -> str:
    """Object ids as used in MJCF names (letters, digits, ``_``, ``-``, ``.``)."""
    out = "".join(c if (c.isalnum() or c in "_-.") else "_" for c in str(oid))
    return out or "obj"


# =============================================================================================
# OBJ meshes and mass properties
# =============================================================================================


def read_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices ``(N, 3)`` and triangles ``(M, 3)`` (0-based) of a Wavefront OBJ; polygons are
    triangulated as fans; ``v/vt/vn`` and negative indices are handled."""
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                idx = []
                for token in parts[1:]:
                    i = int(token.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    V = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    F = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if F.size and (F.min() < 0 or F.max() >= len(V)):
        raise PipelineError(f"{path}: face index out of range")
    return V, F


def write_obj(path: str | Path, V: Any, F: Any, comment: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    lines = [f"# {comment}"] if comment else []
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in V]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in F]
    path.write_text("\n".join(lines) + "\n")
    return path


def box_mesh(size: Any, center: Any = (0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Closed, outward-oriented box with full extents ``size`` (8 vertices, 12 triangles)."""
    sx, sy, sz = (float(v) / 2.0 for v in size)
    V = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)]) + np.asarray(center, dtype=np.float64)
    F = np.array([
        [0, 1, 3], [0, 3, 2],  # -x
        [4, 6, 7], [4, 7, 5],  # +x
        [0, 4, 5], [0, 5, 1],  # -y
        [2, 3, 7], [2, 7, 6],  # +y
        [0, 2, 6], [0, 6, 4],  # -z
        [1, 5, 7], [1, 7, 3],  # +z
    ])
    return V, F


def mesh_mass_properties(V: Any, F: Any) -> tuple[float, np.ndarray, np.ndarray]:
    """Volume, centroid ``(3,)`` and inertia tensor ``(3, 3)`` about the centroid **per unit
    density** of a closed triangle mesh (signed tetrahedra; the sign of the volume is fixed when
    the mesh is oriented inwards)."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))  # 6 x signed tetra volume (origin, a, b, c)
    volume = float(np.sum(vol6) / 6.0)
    if abs(volume) < 1e-15:
        return 0.0, V.mean(axis=0) if len(V) else np.zeros(3), np.zeros((3, 3))
    centroid = np.sum(vol6[:, None] * (a + b + c), axis=0) / (4.0 * 6.0 * volume)
    s = a + b + c
    # Second moment about the origin: sum v/20 (a a^T + b b^T + c c^T + s s^T)
    C = np.einsum("i,ij,ik->jk", vol6 / 120.0, a, a) + np.einsum("i,ij,ik->jk", vol6 / 120.0, b, b) \
        + np.einsum("i,ij,ik->jk", vol6 / 120.0, c, c) + np.einsum("i,ij,ik->jk", vol6 / 120.0, s, s)
    if volume < 0:
        volume, C = -volume, -C
    I_origin = np.trace(C) * np.eye(3) - C
    d = centroid
    I_c = I_origin - volume * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return volume, centroid, 0.5 * (I_c + I_c.T)


def shift_inertia(I_c: np.ndarray, mass: float, offset: Any) -> np.ndarray:
    """Parallel-axis theorem: inertia about a point ``offset`` away from the centre of mass."""
    d = np.asarray(offset, dtype=np.float64)
    return I_c + mass * (float(d @ d) * np.eye(3) - np.outer(d, d))


def box_inertia(mass: float, size: Any) -> np.ndarray:
    sx, sy, sz = (float(v) for v in size)
    return np.diag([mass * (sy * sy + sz * sz), mass * (sx * sx + sz * sz), mass * (sx * sx + sy * sy)]) / 12.0


# =============================================================================================
# Scene specification
# =============================================================================================


@dataclass
class ObjectSpec:
    """One reconstructed object (all quantities in the object frame unless stated)."""

    id: str
    pose: np.ndarray  # (4, 4) initial T_world_obj
    visual_mesh: Path | None = None
    collision_meshes: list[Path] = field(default_factory=list)
    mass: float = 0.1
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia: np.ndarray = field(default_factory=lambda: np.eye(3) * 1e-5)  # about com
    friction: tuple[float, float, float] = (0.8, 0.005, 0.0001)
    rgba: tuple[float, float, float, float] = (0.85, 0.45, 0.2, 1.0)
    size: np.ndarray | None = None  # bounding-box extents (used when there is no mesh)


@dataclass
class TableSpec:
    center_xy: np.ndarray  # (2,)
    size_xy: np.ndarray  # (2,) full extents along the box's local x/y
    yaw: float  # rotation about z (radians)
    top_z: float
    floor_z: float
    friction: tuple[float, float, float] = (0.8, 0.005, 0.0001)


@dataclass
class SceneSpec:
    name: str = "glassesdata"
    timestep: float = 0.002
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    option: Mapping[str, Any] = field(default_factory=dict)  # extra <option> attributes
    floor_z: float = -1.5
    table: TableSpec | None = None
    objects: list[ObjectSpec] = field(default_factory=list)
    hands: Mapping[str, np.ndarray | None] = field(default_factory=dict)  # side -> (21, 3) initial joints or None (hidden)
    arms: Mapping[str, np.ndarray | None] = field(default_factory=dict)  # side -> (3, 3) shoulder, elbow, wrist
    head_pose: np.ndarray | None = None  # (4, 4) initial T_world_cam (OpenCV)
    head_fovy: float = 60.0
    render_size: tuple[int, int] = (640, 480)  # offscreen buffer (width, height)
    hand_radius: float = 0.008
    tip_radius: float = 0.006
    arm_radius: float = 0.025
    hand_rgba: Mapping[str, tuple[float, float, float, float]] = field(default_factory=lambda: {
        side: hex_rgba(code) for side, code in HAND_HEX.items()})


@dataclass
class SceneInfo:
    xml: Path
    assets: list[Path]
    stats: dict[str, Any]


def table_from_polygon(polygon_xy: Any, top_z: float, height: float) -> TableSpec:
    """Oriented minimum-area rectangle of a table polygon (``cv2.minAreaRect``)."""
    import cv2

    pts = np.asarray(polygon_xy, dtype=np.float64).reshape(-1, 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 3:
        raise PipelineError("table polygon needs at least 3 points")
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts.astype(np.float32))
    return TableSpec(np.array([cx, cy]), np.array([max(w, 0.05), max(h, 0.05)]), math.radians(angle), float(top_z),
                     float(top_z - height))


def table_corners(table: TableSpec, z: float | None = None) -> np.ndarray:
    """``(4, 3)`` corners of the table top (counter-clockwise)."""
    hx, hy = table.size_xy / 2.0
    R = geo.rot_z(table.yaw)[:2, :2]
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    xy = local @ R.T + table.center_xy
    return np.column_stack([xy, np.full(4, table.top_z if z is None else z)])


# =============================================================================================
# Scene XML
# =============================================================================================


def _sub(parent: ET.Element, tag: str, **attrs: Any) -> ET.Element:
    el = ET.SubElement(parent, tag)
    for key, value in attrs.items():
        if value is None:
            continue
        key = key.rstrip("_")
        el.set(key, value if isinstance(value, str) else fmt(value))
    return el


def _pose_attrs(T: np.ndarray) -> dict[str, str]:
    p, q = pos_quat(T)
    return {"pos": fmt(p), "quat": fmt(q)}


def hex_rgba(code: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    code = code.lstrip("#")
    return (int(code[0:2], 16) / 255.0, int(code[2:4], 16) / 255.0, int(code[4:6], 16) / 255.0, alpha)


# One palette for every figure and render: hands left = blue, right = orange; objects take the
# remaining categorical slots in a fixed order (aqua, violet, magenta, yellow, green, red).
HAND_HEX = {"left": "#2a78d6", "right": "#eb6834"}
OBJECT_HEX = ["#1baf7a", "#6250d6", "#e87ba4", "#eda100", "#008300", "#e34948"]
_OBJECT_COLORS = [hex_rgba(h) for h in OBJECT_HEX]


def object_color(k: int) -> tuple[float, float, float, float]:
    return _OBJECT_COLORS[k % len(_OBJECT_COLORS)]


def build_scene(spec: SceneSpec, mesh_names: Mapping[str, tuple[str | None, list[str]]] | None = None) -> ET.ElementTree:
    """The scene as an ``ElementTree``. ``mesh_names`` maps object id -> (visual mesh file,
    collision mesh files) relative to the ``meshdir`` (written by :func:`write_scene`)."""
    mesh_names = mesh_names or {}
    root = ET.Element("mujoco", model=spec.name)
    _sub(root, "compiler", angle="radian", meshdir="assets", autolimits="true",
         boundmass="1e-5", boundinertia="1e-9", balanceinertia="true")
    option = {"timestep": fmt(spec.timestep), "gravity": fmt(spec.gravity)}
    option.update({k: (v if isinstance(v, str) else fmt(v)) for k, v in spec.option.items()})
    _sub(root, "option", **option)
    width, height = spec.render_size
    visual = _sub(root, "visual")
    _sub(visual, "global", offwidth=str(int(width)), offheight=str(int(height)))
    _sub(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.55 0.55 0.55", specular="0.1 0.1 0.1")
    _sub(visual, "quality", shadowsize="4096")
    centre = np.zeros(3)
    if spec.table is not None:
        centre = np.array([*spec.table.center_xy, spec.table.top_z])
    elif spec.objects:
        centre = np.mean([o.pose[:3, 3] for o in spec.objects], axis=0)
    _sub(root, "statistic", center=fmt(centre), extent="2")

    default = _sub(root, "default")
    d = _sub(default, "default", class_="gd_visual")
    _sub(d, "geom", contype="0", conaffinity="0", group="2")
    d = _sub(default, "default", class_="gd_collision")
    _sub(d, "geom", group="3", condim="4")
    d = _sub(default, "default", class_="gd_human")
    _sub(d, "geom", type="sphere", contype="0", conaffinity="0", group="1")
    _sub(d, "site", group="1")

    asset = _sub(root, "asset")
    _sub(asset, "texture", name="gd_grid", type="2d", builtin="checker", rgb1="0.82 0.84 0.86", rgb2="0.74 0.76 0.79",
         width="512", height="512", mark="edge", markrgb="0.6 0.62 0.66")
    _sub(asset, "material", name="gd_floor", texture="gd_grid", texrepeat="6 6", texuniform="true", reflectance="0.05")
    _sub(asset, "material", name="gd_table", rgba="0.55 0.42 0.30 1", specular="0.2")
    for k, obj in enumerate(spec.objects):
        oid = sanitize_id(obj.id)
        visual_file, collision_files = mesh_names.get(obj.id, (None, []))
        if visual_file:
            _sub(asset, "mesh", name=f"obj_{oid}_visual", file=visual_file, inertia="convex")
        for j, cf in enumerate(collision_files):
            _sub(asset, "mesh", name=f"obj_{oid}_col{j}", file=cf, inertia="convex")

    world = _sub(root, "worldbody")
    light_at = centre + np.array([0.0, 0.0, 2.2])
    _sub(world, "light", name="key", pos=fmt(light_at), dir="0 0 -1", diffuse="0.6 0.6 0.6", castshadow="true")
    _sub(world, "light", name="fill", pos=fmt(centre + np.array([-1.5, -1.5, 2.0])), dir=fmt(-np.array([-1.5, -1.5, 2.0]) / 3.2),
         diffuse="0.25 0.25 0.25", castshadow="false")
    _sub(world, "geom", name=FLOOR_GEOM, type="plane", size="6 6 0.05", pos=fmt([centre[0], centre[1], spec.floor_z]),
         material="gd_floor")
    if spec.table is not None:
        t = spec.table
        height = max(t.top_z - t.floor_z, 0.02)
        body = _sub(world, "body", name=TABLE_BODY,
                    pos=fmt([t.center_xy[0], t.center_xy[1], t.top_z - height / 2.0]),
                    quat=fmt(geo.mat_to_quat(geo.rot_z(t.yaw))))
        _sub(body, "geom", name=TABLE_GEOM, type="box", size=fmt([t.size_xy[0] / 2, t.size_xy[1] / 2, height / 2]),
             material="gd_table", friction=fmt(t.friction))
    eye = centre + np.array([-0.35, -1.25, 0.85])
    cpos, cquat = look_at_mujoco(eye, centre)
    _sub(world, "camera", name=OVERVIEW_CAMERA, pos=fmt(cpos), quat=fmt(cquat), fovy="50")

    for k, obj in enumerate(spec.objects):
        oid = sanitize_id(obj.id)
        body = _sub(world, "body", name=object_body(oid), **_pose_attrs(obj.pose))
        _sub(body, "freejoint", name=object_joint(oid))
        I = 0.5 * (obj.inertia + obj.inertia.T)
        I = np.where(np.abs(I) < 1e-12 * max(float(np.abs(I).max()), 1e-30), 0.0, I)
        _sub(body, "inertial", pos=fmt(obj.com), mass=fmt(max(obj.mass, 1e-4)),
             fullinertia=fmt([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]))
        visual_file, collision_files = mesh_names.get(obj.id, (None, []))
        rgba = fmt(obj.rgba)
        if visual_file:
            _sub(body, "geom", name=f"obj_{oid}_visual", class_="gd_visual", type="mesh", mesh=f"obj_{oid}_visual", rgba=rgba)
        for j in range(len(collision_files)):
            _sub(body, "geom", name=f"obj_{oid}_col{j}", class_="gd_collision", type="mesh", mesh=f"obj_{oid}_col{j}",
                 friction=fmt(obj.friction), rgba=rgba)
        if not visual_file and not collision_files:
            size = np.asarray(obj.size if obj.size is not None else [0.05, 0.05, 0.05], dtype=np.float64)
            _sub(body, "geom", name=f"obj_{oid}_box", type="box", size=fmt(size / 2.0), friction=fmt(obj.friction), rgba=rgba)

    hidden = np.array([centre[0], centre[1], spec.floor_z - HIDDEN_Z_OFFSET])
    for side in SIDES:
        if side not in spec.hands:
            continue
        joints = spec.hands[side]
        rgba = fmt(spec.hand_rgba.get(side, (0.9, 0.6, 0.4, 1.0)))
        for j in range(sk.N_JOINTS):
            p = hidden if joints is None or not np.all(np.isfinite(joints[j])) else joints[j]
            body = _sub(world, "body", name=hand_body(side, j), mocap="true", pos=fmt(p))
            radius = spec.tip_radius if j in (4, 8, 12, 16, 20) else spec.hand_radius
            _sub(body, "geom", class_="gd_human", size=fmt(radius), rgba=rgba)
    for side in SIDES:
        if side not in spec.arms:
            continue
        pts = spec.arms[side]
        for k, part in enumerate(ARM_PARTS):
            p = hidden if pts is None or not np.all(np.isfinite(pts[k])) else pts[k]
            body = _sub(world, "body", name=arm_body(side, part), mocap="true", pos=fmt(p))
            _sub(body, "geom", class_="gd_human", size=fmt(spec.arm_radius if part != "wrist" else spec.arm_radius * 0.7),
                 rgba="0.55 0.58 0.62 0.9")
    head_T = spec.head_pose if spec.head_pose is not None else np.eye(4)
    head = _sub(world, "body", name=HEAD_BODY, mocap="true", **_pose_attrs(head_T))
    _sub(head, "camera", name=HEAD_CAMERA, quat=fmt(QUAT_CV_MJ), fovy=fmt(spec.head_fovy))
    _sub(head, "geom", class_="gd_human", type="box", size="0.07 0.018 0.012", pos="0 -0.005 -0.03",
         rgba="0.12 0.12 0.14 1")
    return ET.ElementTree(root)


def write_scene(spec: SceneSpec, out_dir: str | Path) -> SceneInfo:
    """Write ``out_dir/scene.xml`` and copy object meshes to ``out_dir/assets/<id>/``."""
    out_dir = Path(out_dir)
    assets_dir = out_dir / "assets"
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    assets_dir.mkdir(parents=True)
    mesh_names: dict[str, tuple[str | None, list[str]]] = {}
    copied: list[Path] = []
    for obj in spec.objects:
        oid = sanitize_id(obj.id)
        visual = None
        if obj.visual_mesh is not None:
            dst = assets_dir / oid / "visual.obj"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(obj.visual_mesh, dst)
            copied.append(dst)
            visual = f"{oid}/visual.obj"
        collision = []
        for j, src in enumerate(obj.collision_meshes):
            dst = assets_dir / oid / f"collision_{j}.obj"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            copied.append(dst)
            collision.append(f"{oid}/collision_{j}.obj")
        mesh_names[obj.id] = (visual, collision)
    tree = build_scene(spec, mesh_names)
    ET.indent(tree, space="  ")
    xml_path = out_dir / "scene.xml"
    tree.write(xml_path, encoding="unicode", xml_declaration=False)
    problems = check_references(tree.getroot(), xml_path.parent)
    if problems:
        raise PipelineError(f"{xml_path}: " + "; ".join(problems[:5]))
    return SceneInfo(xml_path, copied, scene_stats(tree.getroot()))


def scene_stats(root: ET.Element) -> dict[str, Any]:
    """Counts of the main MJCF elements (bodies, mocap bodies, free joints, joints, geoms, ...)."""
    world = root.find("worldbody")
    bodies = world.iter("body") if world is not None else []
    bodies = list(bodies)
    return {
        "bodies": len(bodies),
        "mocap_bodies": sum(1 for b in bodies if b.get("mocap") == "true"),
        "free_joints": len(list(root.iter("freejoint"))) + sum(1 for j in root.iter("joint") if j.get("type") == "free"),
        "hinge_joints": sum(1 for j in (world.iter("joint") if world is not None else []) if j.get("type", "hinge") == "hinge"),
        "geoms": len(list(world.iter("geom"))) if world is not None else 0,
        "meshes": len(root.findall("asset/mesh")),
        "cameras": len(list(root.iter("camera"))),
        "actuators": sum(len(list(a)) for a in root.findall("actuator")),
        "equalities": sum(len(list(e)) for e in root.findall("equality")),
    }


# =============================================================================================
# Reference checking
# =============================================================================================

# attribute -> namespace of the name it refers to
_REF_ATTRS: dict[str, str] = {
    "class": "class", "childclass": "class",
    "material": "material", "mesh": "mesh", "texture": "texture", "hfield": "hfield", "skin": "skin",
    "joint": "joint", "joint1": "joint", "joint2": "joint", "jointinparent": "joint",
    "tendon": "tendon", "tendon1": "tendon", "tendon2": "tendon",
    "body": "body", "body1": "body", "body2": "body", "target": "body",
    "site": "site", "site1": "site", "site2": "site", "cranksite": "site", "slidersite": "site", "refsite": "site",
    "geom": "geom", "geom1": "geom", "geom2": "geom",
    "actuator": "actuator",
}
_DEFINING_TAGS: dict[str, str] = {
    "body": "body", "joint": "joint", "freejoint": "joint", "geom": "geom", "site": "site", "camera": "camera",
    "light": "light", "material": "material", "mesh": "mesh", "texture": "texture", "hfield": "hfield", "skin": "skin",
}
_SECTION_NAMESPACES = {"tendon": "tendon", "actuator": "actuator", "sensor": "sensor", "equality": "equality",
                       "contact": "contact", "keyframe": "key"}


def _asset_name(el: ET.Element) -> str | None:
    name = el.get("name")
    if name is None and el.get("file"):
        name = Path(el.get("file")).stem
    return name


def defined_names(root: ET.Element) -> dict[str, set[str]]:
    """Names defined in an MJCF tree per namespace (mesh/texture names default to the file stem)."""
    names: dict[str, set[str]] = {}

    def add(ns: str, name: str | None) -> None:
        if name:
            names.setdefault(ns, set()).add(name)

    for el in root.iter("default"):
        add("class", el.get("class"))
    for section in root.findall("asset"):
        for el in section:
            add(_DEFINING_TAGS.get(el.tag, el.tag), _asset_name(el))
    world = root.find("worldbody")
    if world is not None:
        for el in world.iter():
            if el.tag in _DEFINING_TAGS:
                add(_DEFINING_TAGS[el.tag], el.get("name"))
    for tag, ns in _SECTION_NAMESPACES.items():
        for section in root.findall(tag):
            for el in section:
                add(ns, el.get("name"))
    return names


def check_references(root: ET.Element, base_dir: str | Path | None = None) -> list[str]:
    """Problems found: references to undefined names, duplicate names, missing mesh/texture files.
    Empty list = the file is consistent (MuJoCo may still reject physics values)."""
    problems: list[str] = []
    names = defined_names(root)
    # duplicates
    seen: dict[tuple[str, str], int] = {}
    world = root.find("worldbody")
    if world is not None:
        for el in world.iter():
            if el.tag in _DEFINING_TAGS and el.get("name"):
                key = (_DEFINING_TAGS[el.tag], el.get("name"))
                seen[key] = seen.get(key, 0) + 1
    for section in root.findall("asset"):
        for el in section:
            n = _asset_name(el)
            if n:
                key = (el.tag, n)
                seen[key] = seen.get(key, 0) + 1
    for el in root.iter("default"):
        if el.get("class"):
            key = ("class", el.get("class"))
            seen[key] = seen.get(key, 0) + 1
    for tag, ns in _SECTION_NAMESPACES.items():
        for section in root.findall(tag):
            for el in section:
                if el.get("name"):
                    key = (ns, el.get("name"))
                    seen[key] = seen.get(key, 0) + 1
    problems += [f"duplicate {ns} name {n!r}" for (ns, n), c in seen.items() if c > 1]
    # references
    for el in root.iter():
        if el.tag == "default":
            continue  # its "class" attribute defines the class
        for attr, ns in _REF_ATTRS.items():
            value = el.get(attr)
            if value is None:
                continue
            if attr == "target" and el.tag not in ("camera", "light"):
                continue
            if value not in names.get(ns, set()) and not (ns == "body" and value == "world"):
                problems.append(f"<{el.tag}> {attr}={value!r} is not defined")
    # files
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    texturedir = compiler.get("texturedir", "") if compiler is not None else ""
    assetdir = compiler.get("assetdir", "") if compiler is not None else ""
    if base_dir is not None:
        for section in root.findall("asset"):
            for el in section:
                f = el.get("file")
                if not f:
                    continue
                sub = meshdir if el.tag in ("mesh", "hfield", "skin") else texturedir
                path = resolve_asset_path(f, Path(base_dir), sub or assetdir)
                if not path.is_file():
                    problems.append(f"<{el.tag}> file {f!r} not found ({path})")
    return problems


def resolve_asset_path(file: str, xml_dir: Path, subdir: str) -> Path:
    """MuJoCo's rule (MJCF reference, ``compiler/meshdir``): an absolute file name is used as is;
    else an absolute ``meshdir`` is prepended; else model directory / meshdir / file."""
    path = Path(file)
    if path.is_absolute():
        return path
    if subdir and Path(subdir).is_absolute():
        return Path(subdir) / path
    return xml_dir / (subdir or "") / path


# =============================================================================================
# Composition with robot hands
# =============================================================================================


@dataclass
class ComposedHand:
    side: str
    prefix: str
    base_body: str  # body carrying the free joint
    base_joint: str
    palm_body: str
    mocap_body: str | None
    weld: str | None
    keypoints: list[tuple[int, str, str, str]]  # (joint, kind, prefixed name, role)
    removed_joints: list[str]


@dataclass
class ComposedModel:
    xml: Path
    robot: str
    base: str
    hands: dict[str, ComposedHand]


def _absolutize_assets(root: ET.Element, xml_dir: Path) -> None:
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    texturedir = compiler.get("texturedir", "") if compiler is not None else ""
    assetdir = compiler.get("assetdir", "") if compiler is not None else ""
    for section in root.findall("asset"):
        for el in section:
            f = el.get("file")
            if not f:
                continue
            if el.tag in ("mesh", "hfield", "skin") and el.get("name") is None:
                el.set("name", Path(f).stem)  # keep the implicit name before the path changes
            if el.tag == "texture" and el.get("name") is None:
                el.set("name", Path(f).stem)
            sub = meshdir if el.tag in ("mesh", "hfield", "skin") else texturedir
            el.set("file", str(resolve_asset_path(f, xml_dir, sub or assetdir).resolve()))
    if compiler is not None:
        for attr in ("meshdir", "texturedir", "assetdir"):
            compiler.attrib.pop(attr, None)


def _prefix_names(root: ET.Element, prefix: str) -> None:
    """Prefix every defined name and every reference in a robot MJCF tree (in place)."""
    names = defined_names(root)
    for el in root.iter():
        if el.tag == "default":
            if el.get("class"):
                el.set("class", prefix + el.get("class"))
            continue
        name = el.get("name")
        if name and el.tag != "mujoco":
            el.set("name", prefix + name)
        for attr, ns in _REF_ATTRS.items():
            value = el.get(attr)
            if value is not None and value in names.get(ns, set()):
                el.set(attr, prefix + value)


def _find_body(root: ET.Element, name: str) -> tuple[ET.Element | None, list[ET.Element]]:
    """The body named ``name`` and its ancestors (outermost first)."""
    world = root.find("worldbody")

    def walk(node: ET.Element, path: list[ET.Element]):
        for child in node.findall("body"):
            if child.get("name") == name:
                return child, path
            found = walk(child, path + [child])
            if found[0] is not None:
                return found
        return None, []

    return walk(world, []) if world is not None else (None, [])


def _names_in_subtree(body: ET.Element) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {"body": set(), "joint": set(), "geom": set(), "site": set()}
    for el in body.iter():
        ns = _DEFINING_TAGS.get(el.tag)
        if ns in out and el.get("name"):
            out[ns].add(el.get("name"))
    return out


def _drop_dangling(robot: ET.Element, kept: dict[str, set[str]]) -> list[str]:
    """Remove tendons/actuators/sensors/equalities/contact pairs that reference removed bodies,
    joints, geoms or sites (and actuators of removed tendons). Returns what was removed."""
    removed: list[str] = []
    ref_ns = {"joint": "joint", "joint1": "joint", "joint2": "joint", "body": "body", "body1": "body", "body2": "body",
              "geom": "geom", "geom1": "geom", "geom2": "geom", "site": "site", "site1": "site", "site2": "site"}

    def dangling(el: ET.Element) -> bool:
        for sub in el.iter():
            for attr, ns in ref_ns.items():
                v = sub.get(attr)
                if v is not None and v not in kept[ns]:
                    return True
        return False

    removed_tendons: set[str] = set()
    for section in robot.findall("tendon"):
        for el in list(section):
            if dangling(el):
                section.remove(el)
                removed.append(f"tendon {el.get('name')}")
                if el.get("name"):
                    removed_tendons.add(el.get("name"))
    for tag in ("actuator", "sensor", "equality", "contact"):
        for section in robot.findall(tag):
            for el in list(section):
                if dangling(el) or el.get("tendon") in removed_tendons:
                    section.remove(el)
                    removed.append(f"{tag} {el.get('name') or el.tag}")
    return removed


def _ensure(root: ET.Element, tag: str, before: Sequence[str] = ()) -> ET.Element:
    el = root.find(tag)
    if el is None:
        el = ET.Element(tag)
        index = len(root)
        for i, child in enumerate(root):
            if child.tag in before:
                index = i
                break
        root.insert(index, el)
    return el


def load_robot_tree(robot: HandRobot, menagerie: str | Path, side: str) -> tuple[ET.Element, Path]:
    """Parse the robot's MJCF file for ``side``: ``(root element, file path)``."""
    path = robot.xml_path(menagerie, side)
    return ET.parse(path).getroot(), path


def compose(
    out_path: str | Path,
    robot: HandRobot,
    sides: Sequence[str],
    menagerie: str | Path,
    scene_xml: str | Path | None = None,
    base: str = "free",
    base_body: str = "palm",
    initial_base: Mapping[str, np.ndarray] | None = None,
    weld_solref: Sequence[float] = (0.02, 1.0),
    weld_solimp: Sequence[float] = (0.9, 0.95, 0.001, 0.5, 2.0),
    model_name: str | None = None,
) -> ComposedModel:
    """Write a model with the scene (or an empty world when ``scene_xml`` is None) plus floating
    robot hands for ``sides``.

    Args:
        robot: e.g. :data:`robots.SHADOW`; files are read from ``menagerie``.
        base: ``"free"`` - the base body only has a free joint (retargeting, RL with direct base
            control); ``"mocap"`` - plus a mocap body ``<side>/base_target`` and a soft weld
            ``<side>/base_weld`` (``solref``/``solimp`` below) that pulls the base to it (physics).
        base_body: ``"palm"`` (free joint on the palm; forearm/wrist bodies and wrist joints are
            dropped) or ``"forearm"`` (the whole Menagerie model with its wrist joints).
        initial_base: side -> 4x4 initial pose of the base body (default: 0.3 m apart, 1 m up).
    """
    if base not in ("free", "mocap"):
        raise PipelineError(f"base must be 'free' or 'mocap', got {base!r}")
    if base_body not in ("palm", "forearm"):
        raise PipelineError(f"base_body must be 'palm' or 'forearm', got {base_body!r}")
    out_path = Path(out_path)
    if scene_xml is not None:
        scene_xml = Path(scene_xml)
        root = ET.parse(scene_xml).getroot()
        _absolutize_assets(root, scene_xml.parent.resolve())
        if model_name:
            root.set("model", model_name)
    else:
        root = ET.Element("mujoco", model=model_name or f"{robot.name}_hands")
        ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    compiler = _ensure(root, "compiler")
    if compiler.get("angle", "degree") != "radian":
        raise PipelineError("the scene must use <compiler angle='radian'>")
    hands: dict[str, ComposedHand] = {}
    for n, side in enumerate(sides):
        prefix = f"{side}/"
        rtree, path = load_robot_tree(robot, menagerie, side)
        rcompiler = rtree.find("compiler")
        if rcompiler is None or rcompiler.get("angle", "degree") != robot.angle_unit:
            raise PipelineError(f"{path}: expected <compiler angle='{robot.angle_unit}'>")
        _absolutize_assets(rtree, path.parent.resolve())
        _prefix_names(rtree, prefix)
        # ---- the floating base body
        palm_name = prefix + robot.resolve(robot.palm_body, side)
        root_name = prefix + robot.resolve(robot.forearm_body, side) if base_body == "forearm" else palm_name
        body, ancestors = _find_body(rtree, root_name)
        if body is None:
            raise PipelineError(f"{path}: no body {root_name[len(prefix):]!r}")
        removed_joints: list[str] = []
        if ancestors:
            parent = ancestors[-1]
            parent.remove(body)
            inherited = next((a.get("childclass") for a in reversed(ancestors) if a.get("childclass")), None)
            if inherited and not body.get("childclass"):
                body.set("childclass", inherited)
            for j in list(body.findall("joint")):
                body.remove(j)
                removed_joints.append(j.get("name", ""))
        else:
            rworld = rtree.find("worldbody")
            rworld.remove(body)
        for j in body.findall("freejoint"):
            body.remove(j)
        for attr in ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis"):
            body.attrib.pop(attr, None)
        T0 = None if initial_base is None else initial_base.get(side)
        if T0 is None:
            T0 = geo.make_T(None, [0.0, (0.3 if side == "left" else -0.3) * (1 if len(sides) > 1 else 0), 1.0])
        p0, q0 = pos_quat(T0)
        body.set("pos", fmt(p0))
        body.set("quat", fmt(q0))
        free = ET.Element("freejoint", name=f"{prefix}base_free")
        body.insert(0, free)
        kept = _names_in_subtree(body)
        kept["body"].add("world")
        # ---- fingertip (and other) sites
        for site in robot.added_sites:
            parent_name = prefix + robot.resolve(site.body, side)
            parent = next((b for b in body.iter("body") if b.get("name") == parent_name), None)
            if parent is None:
                raise PipelineError(f"{path}: no body {parent_name!r} for site {site.name}")
            sname = prefix + robot.resolve(site.name, side)
            ET.SubElement(parent, "site", name=sname, pos=fmt(site.pos), size=fmt(site.size), group="4",
                          rgba="0.9 0.2 0.2 0.6")
            kept["site"].add(sname)
        dropped = _drop_dangling(rtree, kept)
        if dropped:
            log.debug("%s: dropped %s", side, ", ".join(dropped))
        # ---- merge into the scene
        rdefault = rtree.find("default")
        if rdefault is not None:
            sdefault = _ensure(root, "default", before=("asset", "worldbody"))
            wrapper = ET.SubElement(sdefault, "default")
            wrapper.set("class", f"{prefix}robot_root")
            for child in list(rdefault):
                wrapper.append(child)
            if not body.get("childclass"):
                body.set("childclass", f"{prefix}robot_root")
        for section in rtree.findall("asset"):
            sasset = _ensure(root, "asset", before=("worldbody",))
            for child in list(section):
                sasset.append(child)
        world = _ensure(root, "worldbody")
        world.append(body)
        mocap = weld = None
        if base == "mocap":
            mocap = f"{prefix}base_target"
            mbody = ET.SubElement(world, "body", name=mocap, mocap="true", pos=fmt(p0), quat=fmt(q0))
            ET.SubElement(mbody, "site", name=f"{prefix}base_target_site", size="0.01", group="4", rgba="0.2 0.9 0.2 0.5")
            equality = _ensure(root, "equality")
            weld = f"{prefix}base_weld"
            ET.SubElement(equality, "weld", name=weld, body1=mocap, body2=root_name, relpose="0 0 0 1 0 0 0",
                          solref=fmt(weld_solref), solimp=fmt(weld_solimp))
        for tag in ("contact", "tendon", "equality", "actuator", "sensor"):
            for section in rtree.findall(tag):
                if len(section) == 0:
                    continue
                target = _ensure(root, tag)
                for child in list(section):
                    target.append(child)
        roption = rtree.find("option")
        if roption is not None:
            soption = _ensure(root, "option", before=("visual", "default", "asset", "worldbody"))
            for k, v in roption.attrib.items():
                soption.attrib.setdefault(k, v)
        keypoints = [(j, kind, prefix + name, role) for j, kind, name, role in robot.keypoint_frames(side)]
        hands[side] = ComposedHand(side, prefix, root_name, f"{prefix}base_free", palm_name, mocap, weld, keypoints,
                                   removed_joints)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="unicode")
    problems = check_references(root, out_path.parent)
    if problems:
        raise PipelineError(f"{out_path}: " + "; ".join(problems[:5]))
    return ComposedModel(out_path, robot.name, base, hands)


def relative_to(path: str | Path, start: str | Path) -> str:
    return os.path.relpath(Path(path), Path(start))
