"""Depth Anything V2 (metric) backend of step ``depth`` via Hugging Face ``transformers``.

Written for the Linux + CUDA machine; it also runs on Apple ``mps`` or the CPU (slowly). Not run on
the development Mac (no torch there).

Install: ``pip install torch transformers>=4.45`` (4.45 added the metric Depth Anything head:
``config.depth_estimation_type = "metric"``, ``max_depth``). Weights are downloaded by
``from_pretrained`` into the Hugging Face cache, ``local/cache/huggingface`` in the repository
(``HF_HOME``, set when the package is imported: :mod:`glassesdata_pipeline.paths`).

Checkpoints (``depth.model``): ``depth-anything/Depth-Anything-V2-Metric-Indoor-{Small,Base,Large}-hf``
(Hypersim, ``max_depth`` 20 m) or ``...-Metric-Outdoor-...-hf`` (Virtual KITTI 2, 80 m). Short names
``"small" | "base" | "large"`` select the indoor ones.

Facts this adapter relies on (checked in the transformers and Depth-Anything-V2 sources):

* ``outputs.predicted_depth`` is ``(B, h, w)`` metres (sigmoid x ``max_depth``) at the network input
  size; the image processor (``DPTImageProcessor``: size 518, ``keep_aspect_ratio``,
  ``ensure_multiple_of = 14``, bicubic, no padding) maps a 720x1280 portrait frame to 518x924, so
  the prediction covers the whole frame and is simply resized to the pipeline's grid.
* The metric models predict **planar z-depth**: their training loaders convert Hypersim's
  distance-to-camera to planar depth (``metric_depth/dataset/hypersim.py``,
  ``hypersim_distance_to_depth``) and Virtual KITTI 2 depth is planar. ``depth.ray_distance``
  (default false) converts a ray-distance model to z when needed.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from ..errors import MissingDependency, require

log = logging.getLogger(__name__)

MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}


def pick_device(name: str = "auto") -> str:
    """``"auto"`` -> ``"cuda"`` if available, else ``"mps"``, else ``"cpu"``."""
    torch = require("torch", "torch", "the depth backends")
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resize_depth(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a depth map to ``size = (w, h)`` (area when shrinking, bilinear when enlarging)."""
    depth = np.asarray(depth, dtype=np.float32)
    if (depth.shape[1], depth.shape[0]) == tuple(size):
        return depth
    shrink = depth.shape[1] > size[0] or depth.shape[0] > size[1]
    return cv2.resize(depth, tuple(size), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)


class DepthAnythingV2:
    """``transformers`` Depth Anything V2 metric model: ``predict(list of RGB uint8) -> list of (h, w)``."""

    def __init__(self, model: str = "small", device: str = "auto", fp16: bool = True) -> None:
        torch = require("torch", "torch", "depth backend depth_anything_v2")
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as error:
            raise MissingDependency("pip install 'transformers>=4.45' (needed for depth backend depth_anything_v2)") from error
        self.torch = torch
        self.model_id = MODELS.get(model, model)
        self.device = pick_device(device)
        self.fp16 = bool(fp16) and self.device == "cuda"
        log.info("loading %s on %s%s", self.model_id, self.device, " (fp16)" if self.fp16 else "")
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        net = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        kind = getattr(net.config, "depth_estimation_type", None)
        if kind != "metric":
            log.warning("%s has depth_estimation_type=%r: its output is not metric depth", self.model_id, kind)
        net = net.to(self.device)
        if self.fp16:
            net = net.half()
        self.net = net.eval()

    def predict(self, images_rgb: Sequence[np.ndarray]) -> list[np.ndarray]:
        torch = self.torch
        inputs = self.processor(images=list(images_rgb), return_tensors="pt")
        pixels = inputs["pixel_values"].to(self.device)
        if self.fp16:
            pixels = pixels.half()
        with torch.inference_mode():
            pred = self.net(pixel_values=pixels).predicted_depth  # (B, h, w) metres
        return [p for p in pred.float().cpu().numpy()]


def depth_frames(
    ep: Any, cfg: Mapping[str, Any], frames: Sequence[int], size: tuple[int, int]
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, z-depth float32 (h, w))`` for ``frames`` (in order), batched."""
    dcfg = cfg.get("depth", {}) or {}
    model = DepthAnythingV2(str(dcfg.get("model", "small")), str(dcfg.get("device", "auto")), bool(dcfg.get("fp16", True)))
    batch_size = max(1, int(dcfg.get("batch_size", 8)))
    batch: list[tuple[int, np.ndarray]] = []

    def flush() -> Iterator[tuple[int, np.ndarray]]:
        preds = model.predict([rgb for _, rgb in batch])
        for (index, _), pred in zip(batch, preds):
            yield index, resize_depth(pred, size)
        batch.clear()

    for index, bgr in ep.frames(frames):
        batch.append((index, np.ascontiguousarray(bgr[:, :, ::-1])))
        if len(batch) >= batch_size:
            yield from flush()
    if batch:
        yield from flush()
