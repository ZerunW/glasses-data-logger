"""SAM 2.1 video object masks for the ``objects`` step (``objects.mask_backend = "sam2"``).

Written against the official package (https://github.com/facebookresearch/sam2, ``main``):

* install (Linux + CUDA): ``pip install "git+https://github.com/facebookresearch/sam2.git"`` (the
  repo's ``pip install -e .``); weights are fetched from the Hugging Face hub by
  ``SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large")`` (also ``-base-plus``,
  ``-small``, ``-tiny``; these ids are keys of ``sam2.build_sam.HF_MODEL_ID_TO_FILENAMES``).
* ``predictor.init_state(video_path, offload_video_to_cpu=False, offload_state_to_cpu=False,
  async_loading_frames=False)`` - ``video_path`` is an ``.mp4`` (decoded with ``decord``) or a
  folder of ``<frame_index>.jpg``; we write a JPEG folder with PyAV-decoded frames (no decord
  needed, and ``max_side`` downscales them).
* ``predictor.add_new_points_or_box(state, frame_idx, obj_id, points=None, labels=None,
  clear_old_points=True, normalize_coords=True, box=None)`` - points ``(N, 2)`` xy pixels of the
  frames given to ``init_state``, labels 1 = positive, box ``[x0, y0, x1, y1]``.
* ``predictor.propagate_in_video(state, start_frame_idx=None, max_frame_num_to_track=None,
  reverse=False)`` yields ``(frame_idx, obj_ids, mask_logits (n_obj, 1, H, W))`` at the video
  resolution; we run it forward and then backward from the prompt frames (``reverse=True``) and
  threshold the logits at ``mask_threshold`` (0 = probability 0.5).

Not run on the development Mac (no torch); see ``tests/test_objects.py`` for the lazy-import test.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .. import geometry as geo
from ..errors import require
from ..groundtruth import pack_mask

log = logging.getLogger(__name__)

DEFAULTS = {"model": "facebook/sam2.1-hiera-large", "device": "cuda", "max_side": 1024,
            "offload_video_to_cpu": True, "mask_threshold": 0.0}


def load_predictor(scfg: Mapping[str, Any]) -> Any:
    """``SAM2VideoPredictor`` from the hub id in ``objects.sam2.model``."""
    p = {**DEFAULTS, **dict(scfg or {})}
    require("torch", "torch", "objects.mask_backend = sam2")
    require("sam2", "git+https://github.com/facebookresearch/sam2.git", "objects.mask_backend = sam2")
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    return SAM2VideoPredictor.from_pretrained(str(p["model"]), device=str(p["device"]))


def _write_frames(ep: Any, folder: Path, max_side: int) -> tuple[float, int]:
    """Decode every video frame to ``folder/<index>.jpg``; returns the scale applied and the count."""
    import cv2

    scale = 1.0
    n = 0
    for index, image in ep.frames():
        h, w = image.shape[:2]
        scale = min(1.0, float(max_side) / max(h, w)) if max_side else 1.0
        if scale < 1.0:
            size = geo.downsample_size(w, h, 1.0 / scale)
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(folder / f"{index:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    return scale, n


def video_masks(ep: Any, cfg: Any, prompts: list[dict[str, Any]], shape: tuple[int, int]) -> list[dict[str, Any] | None]:
    """One mask track per prompt (``obj_id`` = prompt index), packed at the depth grid ``shape``
    (``(h, w)``; a pixel is set when at least half of its block is the object), in the format of
    ``steps.objects.prompt_masks``: ``{"frames", "packed", "shape", "hit": None, "label": None}``."""
    import cv2

    scfg = {**DEFAULTS, **dict(cfg.get("objects", {}).get("sam2", {}) or {})}
    torch = require("torch", "torch", "objects.mask_backend = sam2")
    predictor = load_predictor(scfg)
    h, w = shape
    tmp = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
    try:
        scale, n_frames = _write_frames(ep, tmp, int(scfg["max_side"]))
        tracks: dict[int, dict[int, np.ndarray]] = {i: {} for i in range(len(prompts))}
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if str(scfg["device"]).startswith("cuda") else _null()
        with torch.inference_mode(), autocast:
            state = predictor.init_state(str(tmp), offload_video_to_cpu=bool(scfg["offload_video_to_cpu"]))
            for i, pr in enumerate(prompts):
                kwargs: dict[str, Any] = {
                    "points": np.array([[pr["x"] * scale, pr["y"] * scale]], dtype=np.float32),
                    "labels": np.array([1], dtype=np.int32),
                }
                if pr.get("box") is not None:
                    kwargs["box"] = np.asarray(pr["box"], dtype=np.float32) * scale
                predictor.add_new_points_or_box(state, frame_idx=int(pr["frame"]), obj_id=i, **kwargs)
            # forward from the first prompt, then backward from the last one (frames before an
            # object's own prompt are covered by the backward pass)
            last = max(int(pr["frame"]) for pr in prompts)
            for reverse in (False, True):
                for frame_idx, obj_ids, logits in predictor.propagate_in_video(state, start_frame_idx=last if reverse else None,
                                                                                reverse=reverse):
                    masks = (logits[:, 0] > float(scfg["mask_threshold"])).float().cpu().numpy()
                    for k, oid in enumerate(obj_ids):
                        m = cv2.resize(masks[k], (w, h), interpolation=cv2.INTER_AREA) >= 0.5
                        if m.any():
                            prev = tracks[int(oid)].get(int(frame_idx))
                            packed = pack_mask(m)
                            tracks[int(oid)][int(frame_idx)] = packed if prev is None else np.bitwise_or(prev, packed)
            predictor.reset_state(state)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out: list[dict[str, Any] | None] = []
    for i in range(len(prompts)):
        frames = sorted(tracks[i])
        if not frames:
            log.warning("sam2: prompt %d (frame %d) gave no mask", i, prompts[i]["frame"])
            out.append(None)
            continue
        out.append({"frames": np.array(frames, dtype=np.int64), "packed": np.stack([tracks[i][f] for f in frames]),
                    "shape": (h, w), "hit": None, "label": None})
    log.info("sam2: %d prompts over %d frames", len(prompts), n_frames)
    return out


class _null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None
