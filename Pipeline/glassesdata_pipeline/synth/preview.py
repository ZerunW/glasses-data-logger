"""Preview images of a synthetic recording: decoded frames, ground-truth overlay, depth.

>>> from glassesdata_pipeline.synth.preview import write_previews
>>> write_previews("/tmp/synth_demo", "/tmp/previews")   # frame_*.png, overlay_*.png, depth_*.png, contact sheet
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .. import geometry as geo
from .. import skeleton as sk
from ..episode import Episode

HAND_COLORS = {0: (255, 160, 40), 1: (40, 220, 255)}  # BGR: left blue-ish, right yellow
_BOX_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def depth_to_color(depth: np.ndarray, near: float = 0.3, far: float = 3.0) -> np.ndarray:
    """Turbo-coloured BGR image of a depth map (near = red, far = blue)."""
    d = np.clip((np.log(np.maximum(depth.astype(np.float32), 1e-3)) - np.log(near)) / (np.log(far) - np.log(near)), 0, 1)
    return cv2.applyColorMap(((1 - d) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def _line(img, camera, a, b, color, thickness=1):
    uv = camera.project(np.stack([a, b]))
    if np.all(np.isfinite(uv)):
        p, q = (tuple(int(round(v)) for v in x) for x in uv)
        cv2.line(img, p, q, color, thickness, cv2.LINE_AA)


def draw_gt_overlay(image: np.ndarray, camera: geo.Camera, gt, i: int) -> np.ndarray:
    """Hand skeletons (solid = visible joint, small dot = occluded/out of view), forearms,
    object wireframes and the grasp state of frame ``i`` drawn on a copy of ``image``."""
    out = image.copy()
    T_cam_world = geo.inv_T(gt["T_world_cam"][i])
    joints = gt["joints_cam"][i]
    vis = gt["visible"][i]
    for k in range(gt["T_world_obj"].shape[1]):
        size = gt["object_sizes"][k]
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * size / 2
        corners = geo.transform(T_cam_world @ gt["T_world_obj"][i, k], signs)
        held = bool(gt["grasped"][i, k].any()) if "grasped" in gt else False
        color = (60, 60, 255) if held else (80, 255, 80)
        for a, b in _BOX_EDGES:
            _line(out, camera, corners[a], corners[b], color, 1)
    for slot in range(2):
        color = HAND_COLORS[slot]
        elbow = geo.transform(T_cam_world, gt["elbow_world"][i, slot])
        if elbow[2] > 0.05 and joints[slot, 0, 2] > 0.05:
            _line(out, camera, elbow, joints[slot, 0], (200, 200, 200), 2)
        for j in range(1, 21):
            p = sk.PARENTS[j]
            if vis[slot, j] and vis[slot, p]:
                _line(out, camera, joints[slot, p], joints[slot, j], color, 2)
        uv = camera.project(joints[slot])
        for j in range(21):
            if np.all(np.isfinite(uv[j])):
                c = tuple(int(round(v)) for v in uv[j])
                cv2.circle(out, c, 3 if vis[slot, j] else 1, color if vis[slot, j] else (128, 128, 128), -1, cv2.LINE_AA)
    cv2.putText(out, f"frame {i}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_previews(recording: str | Path, out_dir: str | Path, frames: list[int] | None = None) -> dict[str, list[str]]:
    """Write ``frame_XXXX.png`` (decoded video), ``overlay_XXXX.png``, ``depth_XXXX.png`` and
    ``synth_contact.png`` (rows: frames, overlays, depth) for ``frames`` (default: six frames at
    the idle, reach, grasp, carry, place and retract phases)."""
    ep = Episode(recording)
    gt = ep.gt
    if gt is None:
        raise ValueError(f"{recording} has no gt/")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    T = gt.n_frames
    if frames is None:
        frames = sorted({min(T - 1, int(round(f * (T - 1)))) for f in (0.05, 0.24, 0.35, 0.52, 0.68, 0.92)})
    camera = ep.camera()
    depth = gt.depth()
    written: dict[str, list[str]] = {"frame": [], "overlay": [], "depth": []}
    rows: dict[str, list[np.ndarray]] = {"frame": [], "overlay": [], "depth": []}
    for i, image in ep.frames(frames):
        tiles = {"frame": image, "overlay": draw_gt_overlay(image, camera, gt, i),
                 "depth": depth_to_color(np.asarray(depth[i]))}
        for kind, tile in tiles.items():
            path = out_dir / f"{kind}_{i:04d}.png"
            cv2.imwrite(str(path), tile)
            written[kind].append(str(path))
            rows[kind].append(tile)
    pad = 4
    sheet = np.vstack([
        np.hstack([np.pad(t, ((pad, pad), (pad, pad), (0, 0)), constant_values=255) for t in rows[kind]])
        for kind in ("frame", "overlay", "depth")
    ])
    path = out_dir / "synth_contact.png"
    cv2.imwrite(str(path), sheet)
    written["contact"] = [str(path)]
    return written
