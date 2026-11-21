"""Step ``depth``: a metric (up to scale) z-depth map per video frame -> ``derived/depth/``.

Outputs (PIPELINE.md section 5):

* ``depth/depth_f16.npy``: float16 memmap ``[F, h, w]`` metres (z-depth, not ray distance) in the
  distorted image grid downsampled by ``depth.downsample`` (:func:`geometry.downsample_size`).
* ``depth/index.npz``: ``frames[F]`` (video frame indices with depth: every ``depth.stride``-th
  frame) and ``scale`` (= h / H).

The map is written row by row into a temporary ``.partial`` file that is renamed when complete, so
frames are never all in memory and a crash never leaves a complete-looking output. The depth of a
learned model is only metric up to an unknown, slowly varying scale; step ``hand3d`` measures it
(``depth_scale_smooth``) from the hands.

Backends (``depth.backend``): ``depth_anything_v2`` (default; :mod:`..backends.depth_anything`),
``video_depth_anything`` (:mod:`..backends.video_depth_anything`), ``gt`` (synthetic recordings,
:func:`..backends.gt.depth_frames` with the scale error and noise of ``[synth.gt_noise]``).
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .. import geometry as geo
from ..episode import Episode
from ..errors import PipelineError

NAME = "depth"
VERSION = 1
REQUIRES = ["ingest"]
OUTPUTS = ["depth/depth_f16.npy", "depth/index.npz"]
CONFIG_SECTIONS = ["depth", "synth"]
BACKENDS = ("depth_anything_v2", "video_depth_anything", "gt")

log = logging.getLogger(__name__)


def INPUTS(cfg) -> list[str]:  # noqa: N802 - the runner's name for it
    backend = str(cfg.get("depth", {}).get("backend", "depth_anything_v2"))
    common = ["derived/timeline.npz", "derived/camera.json"]
    if backend == "gt":
        return common + ["gt/depth_f16.npy"]
    return common + ["video.mp4"]


def frame_indices(n_frames: int, stride: int) -> np.ndarray:
    """Every ``stride``-th frame (0, stride, ...)."""
    if stride < 1:
        raise PipelineError(f"depth.stride must be >= 1, got {stride}")
    return np.arange(0, n_frames, stride, dtype=np.int64)


def ray_to_z(depth: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """Euclidean distance along each pixel's ray -> z-depth (``rays`` = ``(x, y, 1)`` per pixel)."""
    return depth / np.linalg.norm(rays, axis=-1)


def run(ep: Episode, cfg) -> dict:
    dcfg = cfg.get("depth", {}) or {}
    backend = str(dcfg.get("backend", "depth_anything_v2"))
    downsample = float(dcfg.get("downsample", 4))
    stride = int(dcfg.get("stride", 1))
    T = len(ep.timeline()["t"])
    camera = ep.camera()
    W, H = camera.width, camera.height
    w, h = geo.downsample_size(W, H, downsample)
    frames = frame_indices(T, stride)
    if backend == "gt":
        from ..backends import gt as gtb

        source = gtb.depth_frames(ep, cfg, frames.tolist())
    elif backend == "depth_anything_v2":
        from ..backends import depth_anything

        source = depth_anything.depth_frames(ep, cfg, frames.tolist(), (w, h))
    elif backend == "video_depth_anything":
        from ..backends import video_depth_anything

        source = video_depth_anything.depth_frames(ep, cfg, frames.tolist(), (w, h))
    else:
        raise PipelineError(f"depth.backend must be one of {', '.join(BACKENDS)}, got {backend!r}")
    rays = camera.pixel_rays(downsample) if bool(dcfg.get("ray_distance", False)) else None

    final = ep.file("depth/depth_f16.npy")
    partial = final.with_name(final.name + ".partial")
    mm = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=(len(frames), h, w))
    written = 0
    try:
        for row, (index, depth) in enumerate(source):
            if row >= len(frames) or int(index) != int(frames[row]):
                raise PipelineError(f"depth backend {backend} returned frame {index} at row {row}")
            depth = np.asarray(depth, dtype=np.float32)
            if depth.shape != (h, w):
                from ..backends.depth_anything import resize_depth

                depth = resize_depth(depth, (w, h))
            if rays is not None:
                depth = ray_to_z(depth, rays)
            mm[row] = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan).astype(np.float16)
            written += 1
        if written != len(frames):
            raise PipelineError(f"depth backend {backend} returned {written} of {len(frames)} frames")
        mm.flush()
        del mm
        os.replace(partial, final)
    except BaseException:
        try:
            del mm
        except NameError:
            pass
        partial.unlink(missing_ok=True)
        raise
    ep.save_npz("depth/index", ep.step_meta(backend=backend), frames=frames, scale=h / H)
    return {"frames": int(len(frames)), "size": f"{w}x{h}", "backend": backend, "stride": stride}
