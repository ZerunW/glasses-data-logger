"""Video Depth Anything (metric, temporally consistent) backend of step ``depth``.

Written for the Linux + CUDA machine; never run on the development Mac (no torch there).

Install (outside this repository, e.g. ``~/src``)::

    git clone https://github.com/DepthAnything/Video-Depth-Anything
    cd Video-Depth-Anything && pip install -r requirements.txt
    # metric checkpoints into ./checkpoints/ (get_weights.sh fetches the relative ones):
    wget -P checkpoints https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth
    #   ... -Base/.../metric_video_depth_anything_vitb.pth, -Large/.../metric_video_depth_anything_vitl.pth

then set ``depth.backend = "video_depth_anything"``, ``depth.vda_repo_dir = "~/src/Video-Depth-Anything"``
and ``depth.vda_encoder = "vits" | "vitb" | "vitl"``.

API used (from the repository's ``run.py`` / ``run_streaming.py``):

* ``video_depth_anything.video_depth.VideoDepthAnything(encoder, features, out_channels, metric=True)``
  and ``infer_video_depth(frames[T,H,W,3] RGB uint8, target_fps, input_size=518, device, fp32)``
  -> ``(depths[T,H,W], fps)`` at the input frame size; with ``metric=True`` no scale/shift
  alignment between windows is applied. ``depth.vda_mode = "chunks"`` (default) runs it on chunks
  of ``vda_chunk_frames`` frames so the whole video never sits in memory (metric output needs no
  alignment across chunks; temporal smoothing restarts at each chunk boundary).
* ``video_depth_anything.video_depth_stream.VideoDepthAnything(encoder, features, out_channels)``
  (no ``metric`` argument; the metric weights load into it) with
  ``infer_video_depth_one(frame RGB, input_size=518, device, fp32)`` -> ``(H, W)``: one frame at a
  time with an internal frame cache (``vda_mode = "stream"``; the README reports lower accuracy).
* ``video_depth.py`` imports ``utils.util`` from the repository root, so the root is put on
  ``sys.path``.

Units are metres. Whether the metric model predicts planar z or ray distance is not documented
(its training sets, Virtual KITTI 2 and IRS, store planar depth), so z is assumed; set
``depth.ray_distance = true`` if a check (e.g. ``eval-charuco``) shows otherwise.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import MissingDependency, PipelineError, require
from .depth_anything import pick_device, resize_depth

log = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _load(dcfg: Mapping[str, Any], stream: bool) -> tuple[Any, str, bool]:
    torch = require("torch", "torch", "depth backend video_depth_anything")
    repo = Path(str(dcfg.get("vda_repo_dir", "") or ".")).expanduser().resolve()
    if not (repo / "video_depth_anything").is_dir():
        raise MissingDependency(
            f"Video-Depth-Anything not found in {repo}: git clone https://github.com/DepthAnything/Video-Depth-Anything "
            "and set depth.vda_repo_dir (needed for depth backend video_depth_anything)"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if stream:
        from video_depth_anything.video_depth_stream import VideoDepthAnything  # type: ignore[import-not-found]
    else:
        from video_depth_anything.video_depth import VideoDepthAnything  # type: ignore[import-not-found]
    encoder = str(dcfg.get("vda_encoder", "vits"))
    if encoder not in MODEL_CONFIGS:
        raise PipelineError(f"depth.vda_encoder must be one of {sorted(MODEL_CONFIGS)}, got {encoder!r}")
    checkpoint = Path(str(dcfg.get("vda_checkpoint", "") or repo / "checkpoints" / f"metric_video_depth_anything_{encoder}.pth"))
    if not checkpoint.expanduser().is_file():
        raise PipelineError(f"metric checkpoint {checkpoint} not found (see backends/video_depth_anything.py)")
    kwargs = dict(MODEL_CONFIGS[encoder])
    if not stream:
        kwargs["metric"] = True
    model = VideoDepthAnything(**kwargs)
    model.load_state_dict(torch.load(str(checkpoint.expanduser()), map_location="cpu"), strict=True)
    device = pick_device(str(dcfg.get("device", "auto")))
    model = model.to(device).eval()
    fp32 = not (bool(dcfg.get("fp16", True)) and device == "cuda")
    return model, device, fp32


def depth_frames(
    ep: Any, cfg: Mapping[str, Any], frames: Sequence[int], size: tuple[int, int]
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, z-depth float32 (h, w))`` for ``frames`` in order."""
    dcfg = cfg.get("depth", {}) or {}
    mode = str(dcfg.get("vda_mode", "chunks"))
    input_size = int(dcfg.get("vda_input_size", 518))
    fps = float(dcfg.get("vda_fps", 24.0))
    if mode == "stream":
        model, device, fp32 = _load(dcfg, stream=True)
        for index, bgr in ep.frames(frames):
            depth = model.infer_video_depth_one(np.ascontiguousarray(bgr[:, :, ::-1]), input_size=input_size,
                                                device=device, fp32=fp32)
            yield index, resize_depth(np.asarray(depth), size)
        return
    if mode != "chunks":
        raise PipelineError(f"depth.vda_mode must be 'chunks' or 'stream', got {mode!r}")
    model, device, fp32 = _load(dcfg, stream=False)
    chunk = max(2, int(dcfg.get("vda_chunk_frames", 256)))
    indices: list[int] = []
    images: list[np.ndarray] = []

    def run_chunk() -> Iterator[tuple[int, np.ndarray]]:
        depths, _ = model.infer_video_depth(np.stack(images), fps, input_size=input_size, device=device, fp32=fp32)
        for index, depth in zip(indices, depths):
            yield index, resize_depth(np.asarray(depth), size)
        indices.clear()
        images.clear()

    for index, bgr in ep.frames(frames):
        indices.append(index)
        images.append(np.ascontiguousarray(bgr[:, :, ::-1]))
        if len(images) >= chunk:
            yield from run_chunk()
    if images:
        yield from run_chunk()
