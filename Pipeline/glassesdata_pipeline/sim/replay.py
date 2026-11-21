"""Kinematic replay of an episode in its MuJoCo scene: ``sim/replay.mp4`` and ``sim/replay_sheet.png``.

Every frame sets the scene's mocap bodies (head camera, 21 joints per hand, shoulder/elbow/wrist)
and the objects' free joints to the reconstruction, renders the head camera and writes the source
frame (undistorted onto the sim camera's pinhole grid, so both halves line up) next to the sim
view.

Renderers:

* :class:`MujocoRenderer` (Linux; needs ``mujoco``): ``mujoco.Renderer`` on ``scene.xml``,
  ``mj_forward`` per frame, hand/arm bones added to the ``MjvScene`` with ``mjv_initGeom`` +
  ``mjv_connector`` before ``render()``.
* :class:`WireframeRenderer` (numpy + OpenCV, runs anywhere): the same scene content projected with
  :func:`frames.project_mujoco` (MuJoCo's camera model, near-plane clipping, painter's algorithm
  for the table and object triangles). On the Mac it stands in for MuJoCo, and its overlay on the
  video in the replay sheet checks the camera conversion and the scene geometry.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from .. import geometry as geo
from .. import skeleton as sk
from ..episode import Episode
from ..errors import require
from . import mjcf
from .data import EpisodeData
from .frames import mujoco_K, mujoco_camera_pose, pos_quat

log = logging.getLogger(__name__)

HAND_BGR = {side: (int(code[5:7], 16), int(code[3:5], 16), int(code[1:3], 16)) for side, code in mjcf.HAND_HEX.items()}
ARM_BGR = (150, 150, 150)
TABLE_BGR = (95, 125, 160)
FLOOR_BGR = (200, 200, 200)
SIDE_SLOT = {"left": 0, "right": 1}


# =============================================================================================
# Per-frame state
# =============================================================================================


@dataclass
class SceneGeometry:
    """What the wireframe renderer draws (world frame)."""

    floor_z: float | None
    table: mjcf.TableSpec | None
    objects: list[tuple[str, np.ndarray, np.ndarray, tuple[int, int, int]]] = field(default_factory=list)  # id, V, F, BGR


@dataclass
class FrameState:
    T_world_cam: np.ndarray | None
    hands: np.ndarray  # (2, 21, 3) NaN when missing
    arms: np.ndarray  # (2, 3, 3) shoulder, elbow, wrist
    objects: dict[str, np.ndarray]  # id -> (4, 4)


class EpisodeStates:
    """Frame-by-frame scene state from :class:`EpisodeData` (objects hold their last valid pose)."""

    def __init__(self, data: EpisodeData) -> None:
        self.data = data
        self.held = {obj.id: obj.held_poses() for obj in data.objects}

    def __call__(self, f: int) -> FrameState:
        d = self.data
        cam = None
        if d.T_world_cam is not None and np.all(np.isfinite(d.T_world_cam[f])) and (d.cam_valid is None or d.cam_valid[f]):
            cam = d.T_world_cam[f]
        hands = d.joints[f] if d.joints is not None else np.full((2, 21, 3), np.nan)
        arms = np.full((2, 3, 3), np.nan)
        if d.shoulder is not None:
            arms[:, 0] = d.shoulder[f]
        if d.elbow is not None:
            arms[:, 1] = d.elbow[f]
        if d.wrist is not None:
            arms[:, 2] = d.wrist[f]
        elif d.joints is not None:
            arms[:, 2] = d.joints[f, :, 0]
        objects = {oid: T[f] for oid, T in self.held.items() if np.all(np.isfinite(T[f]))}
        return FrameState(cam, hands, arms, objects)


def render_meshes(obj: Any, max_triangles: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    """The mesh to draw for an object: its visual mesh, else the collision parts, else its box;
    a mesh with too many triangles is replaced by its bounding box."""
    if obj.mesh is not None and 0 < len(obj.mesh[1]) <= max_triangles:
        return obj.mesh
    if obj.collision:
        V = np.concatenate([c[0] for c in obj.collision])
        offsets = np.cumsum([0] + [len(c[0]) for c in obj.collision[:-1]])
        F = np.concatenate([c[1] + o for c, o in zip(obj.collision, offsets)])
        if len(F) <= max_triangles:
            return V, F
    ext = obj.extents()
    centre = np.zeros(3)
    for V, _ in ([obj.mesh] if obj.mesh is not None else []) + obj.collision:
        if len(V):
            centre = (V.max(axis=0) + V.min(axis=0)) / 2.0
            break
    return mjcf.box_mesh(ext, centre)


def scene_geometry(data: EpisodeData, table: mjcf.TableSpec | None, floor_z: float | None) -> SceneGeometry:
    objs = []
    for k, obj in enumerate(data.objects):
        V, F = render_meshes(obj)
        r, g, b, _ = mjcf.object_color(k)
        objs.append((obj.id, V, F, (int(b * 255), int(g * 255), int(r * 255))))
    return SceneGeometry(floor_z, table, objs)


# =============================================================================================
# Wireframe renderer (numpy / OpenCV)
# =============================================================================================


class WireframeRenderer:
    """Draws the scene from the head camera with MuJoCo's camera model (see module docstring)."""

    name = "wireframe"

    def __init__(self, geometry: SceneGeometry, fovy: float, width: int, height: int, near: float = 0.02) -> None:
        self.geometry = geometry
        self.fovy = float(fovy)
        self.width = int(width)
        self.height = int(height)
        self.near = near
        self.f = (height / 2.0) / math.tan(math.radians(fovy) / 2.0)
        self._background = self._make_background()
        self._edges: dict[tuple[str, int], np.ndarray] = {}
        self._table = None
        if geometry.table is not None:
            V, F = _table_mesh(geometry.table)
            self._table = (V, F, feature_edges(V, F))

    def _make_background(self) -> np.ndarray:
        ramp = np.linspace(238, 214, self.height).astype(np.uint8)
        img = np.repeat(ramp[:, None], self.width, axis=1)
        return np.dstack([img, img, img])

    # -- projection helpers (MuJoCo camera: looks along -z, +y up)
    def _local(self, P: np.ndarray, pos: np.ndarray, R: np.ndarray) -> np.ndarray:
        return (np.asarray(P, dtype=np.float64) - pos) @ R

    def _pix(self, local: np.ndarray) -> np.ndarray:
        depth = -local[..., 2]
        u = (self.width - 1) / 2.0 + self.f * local[..., 0] / depth
        v = (self.height - 1) / 2.0 - self.f * local[..., 1] / depth
        return np.stack([u, v], axis=-1)

    def _segments(self, A: np.ndarray, B: np.ndarray, pos: np.ndarray, R: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Clip world segments against the near plane and project them."""
        a, b = self._local(A, pos, R), self._local(B, pos, R)
        da, db = -a[:, 2], -b[:, 2]
        out = []
        for i in range(len(a)):
            if not (np.all(np.isfinite(a[i])) and np.all(np.isfinite(b[i]))):
                continue
            pa, pb, za, zb = a[i], b[i], da[i], db[i]
            if za < self.near and zb < self.near:
                continue
            if za < self.near:
                s = (self.near - za) / (zb - za)
                pa = pa + s * (pb - pa)
            elif zb < self.near:
                s = (self.near - zb) / (za - zb)
                pb = pb + s * (pa - pb)
            out.append((self._pix(pa), self._pix(pb)))
        return out

    @staticmethod
    def _pt(p: np.ndarray) -> tuple[int, int]:
        return int(round(float(p[0]) * 4)), int(round(float(p[1]) * 4))  # 2 bits of sub-pixel precision

    def render(self, state: FrameState, background: np.ndarray | None = None) -> np.ndarray:
        """BGR image of the scene; with ``background`` (an image of the same size) the scene is
        drawn over it (no floor grid) - an overlay for checking alignment with the video."""
        import cv2

        img = self._background.copy() if background is None else np.array(background, copy=True)
        if state.T_world_cam is None:
            cv2.putText(img, "no camera pose", (12, self.height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1, cv2.LINE_AA)
            return img
        pos, quat = mujoco_camera_pose(state.T_world_cam)
        R = geo.quat_to_mat(quat)
        g = self.geometry
        line = lambda p, q, color, th: cv2.line(img, self._pt(p), self._pt(q), color, th, cv2.LINE_AA, shift=2)  # noqa: E731
        # floor grid
        if g.floor_z is not None and background is None:
            c = np.zeros(2) if g.table is None else g.table.center_xy
            ticks = np.arange(-8, 9) * 0.25
            A = np.concatenate([np.column_stack([c[0] + ticks, np.full(17, c[1] - 2.0), np.full(17, g.floor_z)]),
                                np.column_stack([np.full(17, c[0] - 2.0), c[1] + ticks, np.full(17, g.floor_z)])])
            B = np.concatenate([np.column_stack([c[0] + ticks, np.full(17, c[1] + 2.0), np.full(17, g.floor_z)]),
                                np.column_stack([np.full(17, c[0] + 2.0), c[1] + ticks, np.full(17, g.floor_z)])])
            for p, q in self._segments(A, B, pos, R):
                line(p, q, FLOOR_BGR, 1)
        # triangles: table and objects (painter's algorithm)
        tris: list[tuple[float, np.ndarray, tuple[int, int, int], bool]] = []
        view_dir_world = -R[:, 2]
        if self._table is not None:
            V, F, E = self._table
            tris += self._triangles(V, F, pos, R, TABLE_BGR, view_dir_world, E)
        for oid, V, F, color in g.objects:
            T = state.objects.get(oid)
            if T is None:
                continue
            key = (oid, len(F))
            if key not in self._edges:
                self._edges[key] = feature_edges(V, F) if len(F) <= 400 else np.zeros((len(F), 3), bool)
            tris += self._triangles(geo.transform(T, V), F, pos, R, color, view_dir_world, self._edges[key])
        tris.sort(key=lambda item: -item[0])
        for _, pts, color, edges in tris:
            poly = np.round(pts * 4).astype(np.int32)
            cv2.fillConvexPoly(img, poly, color, cv2.LINE_AA, shift=2)
            if np.any(edges):
                dark = tuple(int(c * 0.55) for c in color)
                for e in np.flatnonzero(edges):
                    cv2.line(img, tuple(poly[e]), tuple(poly[(e + 1) % 3]), dark, 1, cv2.LINE_AA, shift=2)
        # arms then hands
        for slot in range(2):
            a = state.arms[slot]
            for p, q in self._segments(a[:-1], a[1:], pos, R):
                line(p, q, ARM_BGR, 6)
        for side, slot in SIDE_SLOT.items():
            J = state.hands[slot]
            if not np.any(np.all(np.isfinite(J), axis=-1)):
                continue
            color = HAND_BGR[side]
            A = np.array([J[p] for p, _ in sk.BONES])
            B = np.array([J[c] for _, c in sk.BONES])
            for p, q in self._segments(A, B, pos, R):
                line(p, q, color, 2)
            local = self._local(J, pos, R)
            for k in range(sk.N_JOINTS):
                if np.all(np.isfinite(local[k])) and -local[k, 2] > self.near:
                    cv2.circle(img, self._pt(self._pix(local[k])), 3 * 4, color, -1, cv2.LINE_AA, shift=2)
        return img

    def _triangles(self, V: np.ndarray, F: np.ndarray, pos: np.ndarray, R: np.ndarray, color: tuple[int, int, int],
                   view_dir: np.ndarray, edges: np.ndarray) -> list[tuple[float, np.ndarray, tuple[int, int, int], np.ndarray]]:
        local = self._local(V, pos, R)
        depth = -local[:, 2]
        out = []
        tri_depth = depth[F]
        ok = np.all(tri_depth > self.near, axis=1)
        if not np.any(ok):
            return out
        pix = self._pix(local)
        P = V[F]
        normals = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        shade = 0.45 + 0.55 * np.abs(normals @ view_dir)
        for i in np.flatnonzero(ok):
            c = tuple(int(min(255, ch * shade[i])) for ch in color)
            out.append((float(tri_depth[i].mean()), pix[F[i]], c, edges[i]))
        return out


def feature_edges(V: np.ndarray, F: np.ndarray, angle_deg: float = 20.0) -> np.ndarray:
    """``(M, 3)`` bool: edge k of triangle i (vertices k -> k+1) is a crease (adjacent face normals
    differ by more than ``angle_deg``) or a boundary - box diagonals are not."""
    F = np.asarray(F, dtype=np.int64)
    P = np.asarray(V, dtype=np.float64)[F]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    faces: dict[tuple[int, int], list[int]] = {}
    for i, tri in enumerate(F):
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            faces.setdefault((min(a, b), max(a, b)), []).append(i)
    cos_limit = math.cos(math.radians(angle_deg))
    out = np.zeros(F.shape, dtype=bool)
    for i, tri in enumerate(F):
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            adj = faces[(min(a, b), max(a, b))]
            out[i, k] = len(adj) != 2 or float(n[adj[0]] @ n[adj[1]]) < cos_limit
    return out


def _table_mesh(table: mjcf.TableSpec) -> tuple[np.ndarray, np.ndarray]:
    height = max(table.top_z - table.floor_z, 0.02)
    V, F = mjcf.box_mesh([table.size_xy[0], table.size_xy[1], height])
    T = geo.make_T(geo.rot_z(table.yaw), [table.center_xy[0], table.center_xy[1], table.top_z - height / 2])
    return geo.transform(T, V), F


# =============================================================================================
# MuJoCo renderer (Linux)
# =============================================================================================


class MujocoRenderer:
    """``mujoco.Renderer`` on ``scene.xml`` from the head camera. Written against the MuJoCo
    3.17 / 4.0 Python API (``mujoco.Renderer(model, height, width)``, ``update_scene(data, camera)``,
    ``render()`` -> RGB uint8; ``mjv_initGeom``; ``mjv_connector(geom, type, width, from, to)``)."""

    name = "mujoco"

    def __init__(self, scene_xml: str | Path, width: int, height: int, bone_radius: float = 0.004) -> None:
        mujoco = require("mujoco", "mujoco", "the MuJoCo replay renderer (step scene)")
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)
        self.width, self.height = int(width), int(height)
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.bone_radius = bone_radius
        obj_body = mujoco.mjtObj.mjOBJ_BODY

        def mocap(name: str) -> int:
            body = mujoco.mj_name2id(self.model, obj_body, name)
            return int(self.model.body_mocapid[body]) if body >= 0 else -1

        self.head = mocap(mjcf.HEAD_BODY)
        self.hand_ids = {side: [mocap(mjcf.hand_body(side, j)) for j in range(sk.N_JOINTS)] for side in SIDE_SLOT}
        self.arm_ids = {side: [mocap(mjcf.arm_body(side, p)) for p in mjcf.ARM_PARTS] for side in SIDE_SLOT}
        self.hidden = np.array([0.0, 0.0, -1e3])  # mocap bodies without data are parked far below
        self.obj_qpos: dict[str, int] = {}

    def bind_objects(self, ids: Sequence[str]) -> None:
        for oid in ids:
            j = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT, mjcf.object_joint(mjcf.sanitize_id(oid)))
            if j >= 0:
                self.obj_qpos[oid] = int(self.model.jnt_qposadr[j])

    def render(self, state: FrameState) -> np.ndarray:
        import cv2

        mj, m, d = self.mj, self.model, self.data
        if state.T_world_cam is not None and self.head >= 0:
            p, q = pos_quat(state.T_world_cam)  # the head body is the OpenCV camera frame
            d.mocap_pos[self.head] = p
            d.mocap_quat[self.head] = q
        for side, slot in SIDE_SLOT.items():
            for j, mid in enumerate(self.hand_ids[side]):
                if mid >= 0:
                    pt = state.hands[slot, j]
                    d.mocap_pos[mid] = pt if np.all(np.isfinite(pt)) else self.hidden
            for k, mid in enumerate(self.arm_ids[side]):
                if mid >= 0:
                    pt = state.arms[slot, k]
                    d.mocap_pos[mid] = pt if np.all(np.isfinite(pt)) else self.hidden
        for oid, adr in self.obj_qpos.items():
            T = state.objects.get(oid)
            if T is not None:
                p, q = pos_quat(T)
                d.qpos[adr:adr + 3] = p
                d.qpos[adr + 3:adr + 7] = q
        mj.mj_forward(m, d)
        self.renderer.update_scene(d, camera=mjcf.HEAD_CAMERA)
        scene = self.renderer.scene
        for side, slot in SIDE_SLOT.items():
            J = state.hands[slot]
            rgba = np.array([*(np.array(HAND_BGR[side][::-1]) / 255.0), 1.0], dtype=np.float32)
            for a, b in sk.BONES:
                self._connector(scene, J[a], J[b], self.bone_radius, rgba)
            arm = state.arms[slot]
            for k in range(2):
                self._connector(scene, arm[k], arm[k + 1], 0.02, np.array([0.6, 0.6, 0.6, 0.9], dtype=np.float32))
        rgb = self.renderer.render()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _connector(self, scene: Any, a: np.ndarray, b: np.ndarray, radius: float, rgba: np.ndarray) -> None:
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))) or scene.ngeom >= scene.maxgeom:
            return
        mj = self.mj
        geom = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
        mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_CAPSULE, radius, np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
        scene.ngeom += 1

    def close(self) -> None:
        self.renderer.close()


# =============================================================================================
# Source frames on the sim grid, video output, composition
# =============================================================================================


class Undistorter:
    """Maps decoded source frames onto the pinhole grid of the sim camera (``mujoco_K``)."""

    def __init__(self, camera: geo.Camera, fovy: float, out_size: tuple[int, int]) -> None:
        import cv2

        w, h = out_size
        K_new = mujoco_K(fovy, camera.width, camera.height)
        if camera.model == "fisheye":
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(camera.K, camera.dist, np.eye(3), K_new,
                                                         (camera.width, camera.height), cv2.CV_32FC1)
        else:
            dist = camera.dist if camera.dist.size else np.zeros(5)
            m1, m2 = cv2.initUndistortRectifyMap(camera.K, dist, np.eye(3), K_new, (camera.width, camera.height), cv2.CV_32FC1)
        self.maps = (m1, m2)
        self.size = (int(w), int(h))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        import cv2

        out = cv2.remap(image, self.maps[0], self.maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if (out.shape[1], out.shape[0]) != self.size:
            out = cv2.resize(out, self.size, interpolation=cv2.INTER_AREA)
        return out


class VideoOut:
    """H.264 MP4 writer (PyAV); falls back through ``codecs``. Frames are BGR uint8."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float,
                 codecs: Sequence[str] = ("libx264", "h264", "mpeg4"), crf: int = 20) -> None:
        import av

        self._av = av
        self.width, self.height = int(width) // 2 * 2, int(height) // 2 * 2
        self.container = av.open(str(path), "w")
        self.stream = None
        errors = []
        for codec in codecs:
            try:
                self.stream = self.container.add_stream(codec, rate=Fraction(fps).limit_denominator(1001))
                self.codec = codec
                break
            except Exception as error:  # noqa: BLE001
                errors.append(f"{codec}: {error}")
        if self.stream is None:
            self.container.close()
            raise RuntimeError("no usable video encoder: " + "; ".join(errors))
        self.stream.width, self.stream.height, self.stream.pix_fmt = self.width, self.height, "yuv420p"
        if self.codec == "libx264":
            self.stream.options = {"crf": str(crf), "preset": "veryfast"}
        self.frames = 0

    def write(self, image_bgr: np.ndarray) -> None:
        img = np.ascontiguousarray(image_bgr[: self.height, : self.width])
        frame = self._av.VideoFrame.from_ndarray(img, format="bgr24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def _label(img: np.ndarray, text: str, org: tuple[int, int]) -> None:
    import cv2

    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    x, y = org
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 6, y - h - 6), (x + w + 6, y + base + 4), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)


def side_by_side(source: np.ndarray | None, sim: np.ndarray, t: float, frame: int, renderer: str, gap: int = 4) -> np.ndarray:
    h, w = sim.shape[:2]
    left = np.full_like(sim, 30) if source is None else source
    canvas = np.full((h, 2 * w + gap, 3), 24, dtype=np.uint8)
    canvas[:, :w] = left
    canvas[:, w + gap:] = sim
    _label(canvas, f"video  t={t:6.2f}s  #{frame}", (10, 22))
    _label(canvas, f"sim ({renderer})", (w + gap + 10, 22))
    return canvas


def make_renderer(kind: str, scene_xml: Path, geometry: SceneGeometry, fovy: float, size: tuple[int, int],
                  object_ids: Sequence[str]) -> WireframeRenderer | MujocoRenderer:
    """``kind``: ``"mujoco"``, ``"wireframe"`` or ``"auto"`` (MuJoCo when importable)."""
    if kind not in ("auto", "mujoco", "wireframe"):
        raise ValueError(f"scene.replay must be auto, mujoco, wireframe or none, got {kind!r}")
    if kind in ("auto", "mujoco"):
        try:
            renderer = MujocoRenderer(scene_xml, size[0], size[1])
            renderer.bind_objects(object_ids)
            return renderer
        except ImportError:
            if kind == "mujoco":
                raise
            log.info("mujoco is not installed; the replay uses the wireframe renderer")
        except Exception as error:  # noqa: BLE001 - e.g. no OpenGL context (headless: export MUJOCO_GL=egl)
            if kind == "mujoco":
                raise
            log.warning("MuJoCo renderer unavailable (%s: %s); the replay uses the wireframe renderer "
                        "(headless Linux: export MUJOCO_GL=egl)", type(error).__name__, error)
    return WireframeRenderer(geometry, fovy, size[0], size[1])


def write_replay(ep: Episode, data: EpisodeData, renderer: Any, fovy: float, size: tuple[int, int], out_path: Path,
                 max_frames: int = 0, undistort: bool = True, crf: int = 20) -> dict[str, Any]:
    """Render every frame (or ``max_frames`` evenly spaced) and write the side-by-side MP4."""
    states = EpisodeStates(data)
    frames = np.arange(data.T)
    if max_frames and data.T > max_frames:
        frames = np.unique(np.linspace(0, data.T - 1, max_frames).round().astype(int))
    fix = Undistorter(data.camera, fovy, size) if (undistort and data.camera is not None) else None
    w, h = size
    fps = data.fps if len(frames) == data.T else max(1.0, len(frames) / max(float(data.t[-1] - data.t[0]), 1e-3))
    out = VideoOut(out_path, 2 * w + 4, h, fps, crf=crf)
    import cv2

    written = 0
    try:
        source_iter = ep.frames(frames.tolist()) if ep.video_path.is_file() else iter(())
        for f in frames:
            src = None
            for index, image in source_iter:
                if index == f:
                    src = image
                    break
            if src is not None:
                src = fix(src) if fix is not None else cv2.resize(src, size, interpolation=cv2.INTER_AREA)
            sim = renderer.render(states(int(f)))
            out.write(side_by_side(src, sim, float(data.t[f]), int(f), renderer.name))
            written += 1
    finally:
        out.close()
    return {"frames": written, "codec": out.codec, "size": [out.width, out.height], "renderer": renderer.name}


def replay_sheet(ep: Episode, data: EpisodeData, renderer: Any, fovy: float, size: tuple[int, int], out_png: Path,
                 n: int = 6, title: str = "") -> Path:
    """Matplotlib sheet: for ``n`` frames, the (undistorted) video with the sim overlaid, and the
    sim view alone. Alignment of the overlay checks the head-camera conversion."""
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    states = EpisodeStates(data)
    ok = np.ones(data.T, bool) if data.T_world_cam is None else np.all(np.isfinite(data.T_world_cam), axis=(-1, -2))
    candidates = np.flatnonzero(ok)
    if not len(candidates):
        candidates = np.arange(data.T)
    picks = np.unique(candidates[np.linspace(0, len(candidates) - 1, min(n, len(candidates))).round().astype(int)])
    fix = Undistorter(data.camera, fovy, size) if data.camera is not None else None
    images = {}
    if ep.video_path.is_file():
        for index, image in ep.frames(picks.tolist()):
            images[index] = fix(image) if fix is not None else cv2.resize(image, size)
    cols = len(picks)
    fig, axes = plt.subplots(2, cols, figsize=(2.3 * cols, 2.3 * 2 * size[1] / size[0] + 0.6), squeeze=False)
    for c, f in enumerate(picks):
        state = states(int(f))
        sim = renderer.render(state)
        src = images.get(int(f))
        if src is None:
            top = sim
        elif isinstance(renderer, WireframeRenderer):
            top = cv2.addWeighted(src, 0.35, renderer.render(state, background=src), 0.65, 0)
        else:
            top = cv2.addWeighted(src, 0.55, sim, 0.45, 0)
        axes[0, c].imshow(cv2.cvtColor(top, cv2.COLOR_BGR2RGB))
        axes[1, c].imshow(cv2.cvtColor(sim, cv2.COLOR_BGR2RGB))
        axes[0, c].set_title(f"t = {data.t[f]:.2f} s  (#{f})", fontsize=8, color="#333333")
        for r in range(2):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for spine in axes[r, c].spines.values():
                spine.set_color("#dddddd")
    axes[0, 0].set_ylabel("video + sim overlay", fontsize=8, color="#555555")
    axes[1, 0].set_ylabel(f"sim view ({renderer.name})", fontsize=8, color="#555555")
    fig.suptitle(title or f"{data.name}: kinematic replay from the head camera", fontsize=10, color="#222222")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, facecolor="white")
    plt.close(fig)
    return out_png
