"""Analytic ray casting against textured oriented boxes (room, table, objects) and capsules (hands).

Everything here is in the *room* frame: floor at z = 0, z up. Rays are given per pixel in the
camera frame as ``(x, y, 1)`` so the hit parameter is the z-depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import geometry as geo
from .textures import Texture, make_texture

LIGHT = np.array([0.35, -0.25, 1.0]) / np.linalg.norm([0.35, -0.25, 1.0])
SKIN_BGR = np.array([120.0, 150.0, 205.0], np.float32)
SLEEVE_BGR = np.array([150.0, 95.0, 60.0], np.float32)
HAND_LABEL = -2

# face id = 2 * axis + (0 for the -axis side, 1 for +axis); (u axis, v axis, v sign)
_FACE_UV = {0: (1, 2, -1.0), 1: (0, 2, -1.0), 2: (0, 1, 1.0)}


@dataclass
class Box:
    """An oriented box. ``R`` maps box coordinates to the room; ``half`` are half extents."""

    name: str
    half: np.ndarray
    center: np.ndarray
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    textures: list[Texture] = field(default_factory=list)
    inside: bool = False  # the room: rays start inside and hit the far side
    object_id: int = -1  # index of a ground-truth object, -1 for scenery

    def set_pose(self, T_room_box: np.ndarray) -> None:
        self.R = np.asarray(T_room_box[:3, :3], dtype=np.float64)
        self.center = np.asarray(T_room_box[:3, 3], dtype=np.float64)

    def face_extent(self, face: int) -> tuple[float, float]:
        ua, va, _ = _FACE_UV[face // 2]
        return 2 * self.half[ua], 2 * self.half[va]

    def intersect(self, o: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Hit parameter ``(N,)`` (inf when missed) and face id ``(N,)`` for rays ``o + s d``."""
        ol = (o - self.center) @ self.R  # (3,)
        dl = d @ self.R  # (N, 3)
        dl = np.where(np.abs(dl) < 1e-12, 1e-12, dl)
        inv = 1.0 / dl
        positive = dl > 0
        h = self.half
        if self.inside:
            tfar = (np.where(positive, h, -h) - ol) * inv
            axis = np.argmin(tfar, axis=1)
            rows = np.arange(len(d))
            t = tfar[rows, axis]
            sign_pos = positive[rows, axis]
            hit = t > 1e-6
        else:
            tnear = (np.where(positive, -h, h) - ol) * inv
            tfar = (np.where(positive, h, -h) - ol) * inv
            axis = np.argmax(tnear, axis=1)
            rows = np.arange(len(d))
            t = tnear[rows, axis]
            exit_ = tfar.min(axis=1)
            sign_pos = ~positive[rows, axis]
            hit = (t <= exit_) & (t > 1e-6)
        face = (2 * axis + sign_pos).astype(np.int8)
        return np.where(hit, t, np.inf), face

    def corners(self) -> np.ndarray:
        """``(8, 3)`` corners in the room frame."""
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
        return self.center + (signs * self.half) @ self.R.T

    def face_coords(self, p_room: np.ndarray, face: int) -> tuple[np.ndarray, np.ndarray]:
        """Texture coordinates (metres from the face corner) of room points on ``face``."""
        pl = (p_room - self.center) @ self.R
        ua, va, vsign = _FACE_UV[face // 2]
        s = pl[:, ua] + self.half[ua]
        t = (self.half[va] - pl[:, va]) if vsign < 0 else (pl[:, va] + self.half[va])
        return s, t

    def face_normal(self, face: int) -> np.ndarray:
        """Outward normal in the room frame (inward, i.e. visible, for the room box)."""
        n = np.zeros(3)
        n[face // 2] = 1.0 if face % 2 else -1.0
        n = self.R @ n
        return -n if self.inside else n


def textured_box(
    rng: np.random.Generator,
    name: str,
    size: tuple[float, float, float],
    center: tuple[float, float, float],
    base_bgr: tuple[int, int, int],
    texel_m: float,
    yaw: float = 0.0,
    inside: bool = False,
    object_id: int = -1,
    detail: float = 1.0,
    max_texture: int = 1024,
    style: str = "muted",
) -> Box:
    """A box with a distinct procedural texture on each face (``texel_m`` metres per texel)."""
    half = np.asarray(size, dtype=np.float64) / 2
    box = Box(name, half, np.asarray(center, dtype=np.float64), geo.rot_z(yaw), [], inside, object_id)
    for face in range(6):
        eu, ev = box.face_extent(face)
        w = int(np.clip(2 ** np.ceil(np.log2(max(eu / texel_m, 8))), 16, max_texture))
        h = int(np.clip(2 ** np.ceil(np.log2(max(ev / texel_m, 8))), 16, max_texture))
        tint = np.clip(np.array(base_bgr) + rng.normal(scale=12 if style != "busy" else 30, size=3), 20, 235)
        image = make_texture(rng, h, w, tuple(int(c) for c in tint), detail=detail, style=style)
        box.textures.append(Texture(image, eu, ev))
    return box


@dataclass
class Capsule:
    a: np.ndarray
    b: np.ndarray
    radius: float
    color: np.ndarray


def _capsule_hit(o: np.ndarray, rd: np.ndarray, cap: Capsule) -> tuple[np.ndarray, np.ndarray]:
    """Distance along unit rays ``rd`` (inf when missed) and unit normals."""
    a, b, r = cap.a, cap.b, cap.radius
    ba = b - a
    oa = o - a
    baba = float(ba @ ba)
    bard = rd @ ba
    baoa = float(ba @ oa)
    rdoa = rd @ oa
    oaoa = float(oa @ oa)
    qa = baba - bard * bard
    qb = baba * rdoa - baoa * bard
    qc = baba * oaoa - baoa * baoa - r * r * baba
    h = qb * qb - qa * qc
    t = np.full(len(rd), np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        tb = (-qb - np.sqrt(h)) / qa
        y = baoa + tb * bard
        body = (h >= 0) & (y > 0) & (y < baba) & (tb > 0)
        t[body] = tb[body]
        # end caps
        cap_center = np.where((y <= 0)[:, None], a, b)
        oc = o - cap_center
        cb = np.sum(rd * oc, axis=1)
        cc = np.sum(oc * oc, axis=1) - r * r
        ch = cb * cb - cc
        tc = -cb - np.sqrt(ch)
        caps = ~body & (h >= 0) & (ch > 0) & (tc > 0)
        t[caps] = tc[caps]
    p = o + rd * np.where(np.isfinite(t), t, 0)[:, None]
    along = np.clip(((p - a) @ ba) / max(baba, 1e-12), 0, 1)
    n = p - (a + along[:, None] * ba)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return t, n


def _screen_rows(camera: geo.Camera, T_cam_room: np.ndarray, box: Box) -> np.ndarray | None:
    """Flat pixel indices of the box's screen-space bounding rectangle (padded), or None when it
    cannot be bounded (a corner behind the camera)."""
    corners = geo.transform(T_cam_room, box.corners())
    if np.all(corners[:, 2] < 1e-3):
        return np.zeros(0, np.int64)
    if np.any(corners[:, 2] < 0.02):
        return None
    uv = camera.project(corners)
    if not np.all(np.isfinite(uv)):
        return None
    W, H = camera.width, camera.height
    lo = uv.min(axis=0)
    hi = uv.max(axis=0)
    pad = 3.0 + 0.05 * (hi - lo)
    x0, y0 = np.floor(lo - pad).astype(int)
    x1, y1 = np.ceil(hi + pad).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1)
    if x0 > x1 or y0 > y1:
        return np.zeros(0, np.int64)
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    return (ys * W + xs).reshape(-1)


class Scene:
    """Boxes (scenery + objects) and optional hand capsules."""

    def __init__(self, boxes: list[Box]) -> None:
        self.boxes = boxes

    def objects(self) -> list[Box]:
        return sorted((b for b in self.boxes if b.object_id >= 0), key=lambda b: b.object_id)

    def render(
        self,
        camera: geo.Camera,
        T_room_cam: np.ndarray,
        capsules: list[Capsule] | None = None,
    ) -> dict[str, np.ndarray]:
        """Render one frame: ``image`` (H,W,3) uint8 BGR, ``depth`` (H,W) float32 z-depth,
        ``scene_depth`` (without capsules), ``label`` (H,W) int16 box index (``HAND_LABEL`` for
        capsules)."""
        H, W = camera.height, camera.width
        rays = camera.pixel_rays().reshape(-1, 3)
        R = T_room_cam[:3, :3]
        o = T_room_cam[:3, 3]
        d = rays @ R.T
        n = len(d)
        best = np.full(n, np.inf)
        label = np.full(n, -1, np.int16)
        face = np.zeros(n, np.int8)
        T_cam_room = geo.inv_T(T_room_cam)
        for i, box in enumerate(self.boxes):
            idx = None if box.inside else _screen_rows(camera, T_cam_room, box)
            if idx is not None and len(idx) == 0:
                continue
            if idx is None:
                t, f = box.intersect(o, d)
                closer = t < best
                best[closer] = t[closer]
                label[closer] = i
                face[closer] = f[closer]
            else:
                t, f = box.intersect(o, d[idx])
                closer = t < best[idx]
                sel = idx[closer]
                best[sel] = t[closer]
                label[sel] = i
                face[sel] = f[closer]
        color = np.zeros((n, 3), np.float32)
        fx = camera.fx
        dnorm = np.linalg.norm(d, axis=1)
        key = label.astype(np.int32) * 6 + face
        order = np.argsort(key, kind="stable")
        sorted_keys = key[order]
        bounds = np.flatnonzero(np.diff(sorted_keys)) + 1
        for chunk in np.split(order, bounds):
            if len(chunk) == 0:
                continue
            i, f = divmod(int(key[chunk[0]]), 6)
            if i < 0:
                continue
            box = self.boxes[i]
            p = o + d[chunk] * best[chunk, None]
            s, tt = box.face_coords(p, f)
            normal = box.face_normal(f)
            cos_inc = np.abs(d[chunk] @ normal) / dnorm[chunk]
            footprint = best[chunk] / (fx * dnorm[chunk]) / np.sqrt(np.maximum(cos_inc, 0.05))
            texel = box.textures[f].sample(s, tt, footprint)
            shade = 0.55 + 0.45 * max(0.0, float(normal @ LIGHT))
            color[chunk] = texel * shade
        scene_depth = best.astype(np.float32)
        depth = scene_depth.copy()
        if capsules:
            self._render_capsules(camera, T_room_cam, rays, d, dnorm, capsules, depth, color, label)
        return {
            "image": np.clip(color, 0, 255).astype(np.uint8).reshape(H, W, 3),
            "depth": depth.reshape(H, W),
            "scene_depth": scene_depth.reshape(H, W),
            "label": label.reshape(H, W),
        }

    @staticmethod
    def _render_capsules(camera, T_room_cam, rays, d, dnorm, capsules, depth, color, label) -> None:
        H, W = camera.height, camera.width
        T_cam_room = geo.inv_T(T_room_cam)
        o = T_room_cam[:3, 3]
        for cap in capsules:
            ends = geo.transform(T_cam_room, np.stack([cap.a, cap.b]))
            if np.all(ends[:, 2] < 0.02):
                continue
            if np.any(ends[:, 2] < 0.05):
                x0, y0, x1, y1 = 0, 0, W - 1, H - 1
            else:
                uv = camera.project(ends)
                if not np.all(np.isfinite(uv)):
                    x0, y0, x1, y1 = 0, 0, W - 1, H - 1
                else:
                    pad = cap.radius * camera.fx / ends[:, 2].min() * 1.3 + 2
                    x0, y0 = np.floor(uv.min(0) - pad).astype(int)
                    x1, y1 = np.ceil(uv.max(0) + pad).astype(int)
                    x0, y0 = max(x0, 0), max(y0, 0)
                    x1, y1 = min(x1, W - 1), min(y1, H - 1)
                    if x0 > x1 or y0 > y1:
                        continue
            ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            idx = (ys * W + xs).reshape(-1)
            rd = d[idx] / dnorm[idx, None]
            t_unit, normal = _capsule_hit(o, rd, cap)
            z = t_unit / dnorm[idx]
            closer = z < depth[idx]
            if not np.any(closer):
                continue
            sel = idx[closer]
            depth[sel] = z[closer]
            label[sel] = HAND_LABEL
            shade = 0.5 + 0.5 * np.clip(normal[closer] @ LIGHT, 0, 1)
            color[sel] = cap.color[None, :] * shade[:, None].astype(np.float32)


def build_scene(rng: np.random.Generator, table_height: float = 0.75) -> tuple[Scene, dict]:
    """The default room: floor/walls/ceiling, a table (top at ``table_height``) with legs and two
    textured boxes (object 0 is grasped and moved, object 1 stays). Returns the scene and the
    layout (object sizes and initial poses in the room frame)."""
    boxes: list[Box] = []
    room_min, room_max = np.array([-1.6, -2.0, 0.0]), np.array([2.6, 2.0, 2.6])
    boxes.append(textured_box(rng, "room", tuple(room_max - room_min), tuple((room_min + room_max) / 2),
                              (165, 170, 175), texel_m=0.0045, inside=True, detail=0.8))
    # Distinct floor: overwrite the room's -z face texture with a warmer one.
    floor_w, floor_h = boxes[0].face_extent(4)
    boxes[0].textures[4] = Texture(make_texture(rng, 1024, 1024, (95, 115, 135), detail=0.8, style="grain"), floor_w, floor_h)
    top_t = 0.03
    table_x0, table_x1, table_y = 0.25, 1.05, 0.60
    cx = (table_x0 + table_x1) / 2
    boxes.append(textured_box(rng, "table_top", (table_x1 - table_x0, 2 * table_y, top_t),
                              (cx, 0.0, table_height - top_t / 2), (70, 100, 135), texel_m=0.0016, detail=0.8,
                              style="grain"))
    leg_h = table_height - top_t
    for k, (lx, ly) in enumerate([(table_x0 + 0.04, -table_y + 0.04), (table_x0 + 0.04, table_y - 0.04),
                                  (table_x1 - 0.04, -table_y + 0.04), (table_x1 - 0.04, table_y - 0.04)]):
        boxes.append(textured_box(rng, f"leg{k}", (0.04, 0.04, leg_h), (lx, ly, leg_h / 2), (40, 50, 60),
                                  texel_m=0.004, detail=0.5))
    size_a = (0.065, 0.065, 0.11)
    pose_a = geo.make_T(geo.rot_z(0.15), [0.50, -0.10, table_height + size_a[2] / 2])
    size_b = (0.16, 0.10, 0.07)
    pose_b = geo.make_T(geo.rot_z(0.35), [0.74, 0.20, table_height + size_b[2] / 2])
    obj_a = textured_box(rng, "box_a", size_a, tuple(pose_a[:3, 3]), (40, 150, 235), texel_m=0.0006,
                         object_id=0, detail=1.5, style="busy")
    obj_a.set_pose(pose_a)
    obj_b = textured_box(rng, "box_b", size_b, tuple(pose_b[:3, 3]), (210, 120, 50), texel_m=0.0008,
                         object_id=1, detail=1.5, style="busy")
    obj_b.set_pose(pose_b)
    boxes += [obj_a, obj_b]
    layout = {
        "object_ids": ["box_a", "box_b"],
        "object_sizes": np.array([size_a, size_b]),
        "object_poses": np.stack([pose_a, pose_b]),
        "table": {"x": [table_x0, table_x1], "y": [-table_y, table_y], "top_z": table_height},
        "room": {"min": room_min.tolist(), "max": room_max.tolist()},
    }
    return Scene(boxes), layout
