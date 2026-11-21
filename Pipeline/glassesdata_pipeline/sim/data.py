"""Load whatever derived files of an episode exist into one object (numpy only).

Used by the ``scene`` step (never with ground truth), the replay renderers, the ``viz`` figures
and the physics check. Every field is optional; missing files give ``None``.

Table height: ``objects/table.json`` has ``normal``, ``offset``, ``height_m`` and ``polygon``
(PIPELINE.md section 5) without a stated plane convention. :func:`resolve_table_top` evaluates the
plane at the polygon centre with both conventions (``n . x + offset = 0`` and ``n . x = offset``)
and also considers ``height_m`` as a world z, then picks the candidate closest to the bottoms of
the objects at their first pose (or the first convention when there are no objects).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import geometry as geo
from ..episode import Episode
from . import mjcf

log = logging.getLogger(__name__)


@dataclass
class ObjectData:
    id: str
    T_world_obj: np.ndarray  # (T, 4, 4), NaN where unknown
    valid: np.ndarray  # (T,)
    grasped: np.ndarray  # (T, 2)
    mesh: tuple[np.ndarray, np.ndarray] | None = None  # (V, F) object frame
    collision: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    mesh_path: Path | None = None
    collision_paths: list[Path] = field(default_factory=list)
    props: dict[str, Any] = field(default_factory=dict)

    @property
    def first_valid(self) -> int | None:
        idx = np.flatnonzero(self.valid)
        return int(idx[0]) if len(idx) else None

    def extents(self) -> np.ndarray:
        """Bounding-box size in the object frame."""
        for V, _ in ([self.mesh] if self.mesh is not None else []) + self.collision:
            if len(V):
                return V.max(axis=0) - V.min(axis=0)
        return np.full(3, 0.05)

    def held_poses(self) -> np.ndarray:
        """``(T, 4, 4)``: the last valid pose held through gaps (the first valid one before it)."""
        out = np.array(self.T_world_obj, dtype=np.float64, copy=True)
        idx = np.flatnonzero(self.valid)
        if not len(idx):
            return out
        last = out[idx[0]].copy()
        for f in range(len(out)):
            if self.valid[f]:
                last = out[f].copy()
            else:
                out[f] = last
        return out


@dataclass
class GroundTruthData:
    T_world_cam: np.ndarray
    joints: np.ndarray
    shoulder: np.ndarray
    elbow: np.ndarray
    T_world_obj: np.ndarray  # (T, K, 4, 4)
    object_ids: list[str]
    object_sizes: np.ndarray
    grasped: np.ndarray  # (T, K, 2)
    table_polygon: np.ndarray | None
    table_top_z: float | None
    floor_z: float | None


@dataclass
class EpisodeData:
    name: str
    T: int
    t: np.ndarray
    fps: float
    camera: geo.Camera | None = None
    T_world_cam: np.ndarray | None = None
    cam_valid: np.ndarray | None = None
    joints: np.ndarray | None = None  # (T, 2, 21, 3)
    observed: np.ndarray | None = None
    filled: np.ndarray | None = None
    bone_lengths: np.ndarray | None = None  # (2, 20)
    shoulder: np.ndarray | None = None
    elbow: np.ndarray | None = None
    wrist: np.ndarray | None = None
    elbow_conf: np.ndarray | None = None
    arm_method: np.ndarray | None = None
    objects: list[ObjectData] = field(default_factory=list)
    table: dict[str, Any] | None = None
    table_top_z: float | None = None
    table_polygon: np.ndarray | None = None
    gt: GroundTruthData | None = None
    retarget: dict[str, Any] | None = None
    physics: dict[str, Any] | None = None
    physics_traj: dict[str, Any] | None = None
    imu: dict[str, np.ndarray] | None = None
    profile: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)

    def hand_valid(self) -> np.ndarray:
        """``(T, 2)``: all 21 joints are finite."""
        if self.joints is None:
            return np.zeros((self.T, 2), dtype=bool)
        return np.all(np.isfinite(self.joints), axis=(-1, -2))

    def first_valid_camera(self) -> int | None:
        if self.T_world_cam is None:
            return None
        ok = np.all(np.isfinite(self.T_world_cam), axis=(-1, -2))
        if self.cam_valid is not None:
            ok &= self.cam_valid
        idx = np.flatnonzero(ok)
        return int(idx[0]) if len(idx) else None


def _load(ep: Episode, name: str) -> dict[str, Any] | None:
    try:
        return ep.load_npz(name) if ep.has(name) else None
    except Exception as error:  # noqa: BLE001 - a broken file must not stop the others
        log.warning("cannot read derived/%s: %s", name, error)
        return None


def _object_index(ep: Episode) -> list[dict[str, Any]]:
    path = ep.derived / "objects" / "index.json"
    if not path.is_file():
        folder = ep.derived / "objects"
        if folder.is_dir():
            return [{"id": p.parent.name} for p in sorted(folder.glob("*/poses.npz"))]
        return []
    data = json.loads(path.read_text())
    items = data.get("items", []) if isinstance(data, dict) else data
    return [dict(item) for item in items if isinstance(item, dict) and "id" in item]


def load_objects(ep: Episode, T: int) -> list[ObjectData]:
    out: list[ObjectData] = []
    for item in _object_index(ep):
        oid = str(item["id"])
        folder = ep.derived / "objects" / oid
        poses = _load(ep, f"objects/{oid}/poses.npz")
        if poses is None:
            log.warning("object %s has no poses.npz; skipped", oid)
            continue
        Tw = np.asarray(poses["T_world_obj"], dtype=np.float64)
        n = min(len(Tw), T)
        T_full = np.full((T, 4, 4), np.nan)
        T_full[:n] = Tw[:n]
        valid = np.zeros(T, dtype=bool)
        v = np.asarray(poses.get("valid", np.ones(len(Tw), bool)), dtype=bool)
        valid[:n] = v[:n]
        valid &= np.all(np.isfinite(T_full), axis=(-1, -2))
        grasped = np.zeros((T, 2), dtype=bool)
        if "grasped" in poses:
            g = np.asarray(poses["grasped"], dtype=bool).reshape(len(Tw), -1)
            grasped[:n, : g.shape[1]] = g[:n, :2]
        obj = ObjectData(oid, T_full, valid, grasped)
        mesh_path = folder / "mesh.obj"
        if mesh_path.is_file():
            obj.mesh = mjcf.read_obj(mesh_path)
            obj.mesh_path = mesh_path
        for part in sorted((folder / "collision").glob("part_*.obj")):
            obj.collision.append(mjcf.read_obj(part))
            obj.collision_paths.append(part)
        props_path = folder / "props.json"
        if props_path.is_file():
            obj.props = json.loads(props_path.read_text())
        out.append(obj)
    return out


def resolve_table_top(table: dict[str, Any], polygon: np.ndarray | None, objects: list[ObjectData]) -> float | None:
    """World z of the table top (see the module docstring).

    The contract (PIPELINE.md §5) now fixes ``top_z`` as the world z of the top surface; it wins
    whenever present. The guessing below only remains for older files without it.
    """
    if table.get("top_z") is not None and np.isfinite(float(table["top_z"])):
        return float(table["top_z"])
    candidates: list[tuple[str, float]] = []
    centre = np.nanmean(polygon, axis=0) if polygon is not None and len(polygon) else np.zeros(2)
    n = table.get("normal")
    d = table.get("offset")
    if n is not None and d is not None:
        n = np.asarray(n, dtype=np.float64).reshape(3)
        if abs(n[2]) > 0.3:
            xy = float(n[0] * centre[0] + n[1] * centre[1])
            candidates.append(("n.x + offset = 0", -(float(d) + xy) / n[2]))
            candidates.append(("n.x = offset", (float(d) - xy) / n[2]))
    if table.get("height_m") is not None:
        candidates.append(("height_m", float(table["height_m"])))
    if not candidates:
        return None
    bottoms = []
    for obj in objects:
        f = obj.first_valid
        if f is None:
            continue
        V = obj.mesh[0] if obj.mesh is not None else (obj.collision[0][0] if obj.collision else None)
        if V is None or not len(V):
            continue
        bottoms.append(float(np.min(geo.transform(obj.T_world_obj[f], V)[:, 2])))
    if not bottoms:
        return candidates[0][1]
    ref = float(np.median(bottoms))
    label, z = min(candidates, key=lambda c: abs(c[1] - ref))
    if label != candidates[0][0]:
        log.info("table top z: using %s (%.3f m; object bottoms at %.3f m)", label, z, ref)
    return z


def load_episode_data(ep: Episode, with_gt: bool = True) -> EpisodeData:
    """Everything available for ``ep``; ``with_gt`` adds ``gt/`` (synthetic recordings)."""
    tl = _load(ep, "timeline.npz")
    gt = ep.gt if with_gt else None
    if tl is not None:
        t = np.asarray(tl["t"], dtype=np.float64)
    elif gt is not None and "t" in gt:
        t = np.asarray(gt["t"], dtype=np.float64)
    else:
        traj = _load(ep, "camera_traj.npz")
        n = len(traj["T_world_cam"]) if traj is not None else 0
        t = np.arange(n) / 24.0
    T = len(t)
    fps = float(1.0 / np.median(np.diff(t))) if T > 1 else 24.0
    data = EpisodeData(ep.path.name, T, t, fps)
    try:
        data.camera = ep.camera()
    except Exception as error:  # noqa: BLE001
        log.warning("no camera: %s", error)
    try:
        data.profile = ep.profile()
    except Exception:  # noqa: BLE001
        data.profile = {}

    def fit(a: np.ndarray, fill: float = np.nan) -> np.ndarray:
        a = np.asarray(a)
        out = np.full((T,) + a.shape[1:], fill, dtype=np.float64 if a.dtype.kind == "f" else a.dtype)
        n = min(len(a), T)
        out[:n] = a[:n]
        return out

    traj = _load(ep, "camera_traj.npz")
    if traj is not None:
        data.T_world_cam = fit(traj["T_world_cam"])
        data.cam_valid = fit(np.asarray(traj.get("valid", np.ones(len(traj["T_world_cam"]), bool)), dtype=bool), False).astype(bool)
        data.sources["camera"] = "camera_traj.npz"
    hands = _load(ep, "hands_world.npz")
    if hands is not None:
        data.joints = fit(hands["joints"])
        if "observed" in hands:
            data.observed = fit(np.asarray(hands["observed"], dtype=bool), False).astype(bool)
        if "filled" in hands:
            data.filled = fit(np.asarray(hands["filled"], dtype=bool), False).astype(bool)
        if "bone_lengths" in hands:
            data.bone_lengths = np.asarray(hands["bone_lengths"], dtype=np.float64)
        data.sources["hands"] = "hands_world.npz"
    arms = _load(ep, "arms.npz")
    if arms is not None:
        data.shoulder = fit(arms["shoulder"])
        data.elbow = fit(arms["elbow"])
        data.wrist = fit(arms["wrist"]) if "wrist" in arms else None
        if "elbow_conf" in arms:
            data.elbow_conf = fit(arms["elbow_conf"])
        if "method" in arms:
            data.arm_method = fit(np.asarray(arms["method"]).astype(np.int8), 0).astype(np.int8)
        data.sources["arms"] = "arms.npz"
    data.objects = load_objects(ep, T)
    if ep.has("objects/table.json"):
        table = ep.load_json("objects/table.json")
        data.table = table
        poly = table.get("polygon")
        if poly is not None and len(poly):
            data.table_polygon = np.asarray(poly, dtype=np.float64)[:, :2]
        data.table_top_z = resolve_table_top(table, data.table_polygon, data.objects)
    retargets = sorted(ep.derived.glob("retarget_*.npz"))
    if retargets:
        data.retarget = _load(ep, retargets[0].name)
        if data.retarget is not None:
            data.retarget["file"] = retargets[0].name
    if ep.has("physics.json"):
        data.physics = ep.load_json("physics.json")
    if ep.has("sim/physics_traj.npz"):
        data.physics_traj = _load(ep, "sim/physics_traj.npz")
    if tl is not None and len(tl.get("imu_t", [])):
        data.imu = {k: np.asarray(tl[k]) for k in ("imu_t", "acc", "gyro") if k in tl}
    if gt is not None:
        g = {k: gt.get(k) for k in gt.keys()}
        data.gt = GroundTruthData(
            T_world_cam=fit(g["T_world_cam"]),
            joints=fit(g["joints_world"]),
            shoulder=fit(g["shoulder_world"]),
            elbow=fit(g["elbow_world"]),
            T_world_obj=fit(g["T_world_obj"]),
            object_ids=gt.object_ids,
            object_sizes=np.asarray(g["object_sizes"], dtype=np.float64),
            grasped=fit(np.asarray(g["grasped"], dtype=bool), False).astype(bool) if g.get("grasped") is not None
            else np.zeros((T, len(gt.object_ids), 2), bool),
            table_polygon=np.asarray(g["table_polygon"], dtype=np.float64) if g.get("table_polygon") is not None else None,
            table_top_z=float(g["table_top_z"]) if g.get("table_top_z") is not None else None,
            floor_z=float(g["floor_z"]) if g.get("floor_z") is not None else None,
        )
    return data
