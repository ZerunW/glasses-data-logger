"""Ground-truth backends for synthetic recordings (``backend = "gt"``; PIPELINE.md section 6).

They read ``<recording>/gt/`` and add configurable noise from ``[synth.gt_noise]``
(``configs/synth.toml``). They return arrays in the section-5 formats and write nothing; the
calling step saves them. All noise is reproducible: it depends on ``synth.gt_noise.seed`` and the
frame index only (so asking for a subset of frames gives the same values).

Steps that use them should list ``"synth"`` in ``CONFIG_SECTIONS`` and ``"gt/gt.npz"`` (or the
file named below) in ``INPUTS`` so the cache notices noise changes.

* :func:`hands2d` -> ``{"kp" [T,2,21,3], "hand_score" [T,2], "mp_world" [T,2,21,3]}``
* :func:`depth` / :func:`depth_frames` -> depth maps at ``depth.downsample`` (``gt/depth_f16.npy``)
* :func:`masks` -> per-object packed masks at ``depth.downsample`` (``gt/masks.npz``)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import cv2
import numpy as np

from .. import geometry as geo
from ..errors import PipelineError
from ..groundtruth import GroundTruth, pack_mask, unpack_mask

DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "hands2d_sigma_px": 2.0,
    "hands2d_dropout": 0.05,
    "hands2d_score": 0.9,
    "hands2d_min_visible": 6,
    "mp_world_sigma_m": 0.003,
    "depth_scale": 1.10,
    "depth_drift": 0.05,
    "depth_drift_period_s": 6.0,
    "depth_noise": 0.03,
    "mask_min_pixels": 1,
}


def noise_params(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """``[synth.gt_noise]`` with defaults for missing keys."""
    section: Mapping[str, Any] = {}
    if cfg is not None:
        synth = cfg.get("synth", {}) or {}
        section = synth.get("gt_noise", {}) or {}
    return {**DEFAULTS, **dict(section)}


def _gt(ep: Any) -> GroundTruth:
    gt = ep.gt
    if gt is None:
        raise PipelineError(f"{ep.path} has no gt/ folder: the 'gt' backends only work on synthetic recordings")
    return gt


def _rng(seed: int, *salt: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), *[int(s) for s in salt]])


def _downsample(cfg: Mapping[str, Any] | None) -> float:
    value = (cfg or {}).get("depth", {}).get("downsample", 4) if cfg is not None else 4
    return float(value)


# =============================================================================================
# hands2d
# =============================================================================================


def hands2d(ep: Any, cfg: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
    """2D keypoints from the ground truth.

    Joints are projected with ``ep.camera()`` (the calibration in use), get Gaussian pixel noise
    (``hands2d_sigma_px``); joints the ground truth marks invisible are NaN; a hand with fewer than
    ``hands2d_min_visible`` visible joints, or dropped at random (``hands2d_dropout`` per frame and
    hand), is entirely NaN. Scores are ``hands2d_score`` (NaN when missing). ``mp_world`` is the
    camera-aligned hand-centred skeleton (origin = mean of joints 0, 5, 9, 13, 17) with Gaussian
    noise ``mp_world_sigma_m`` (all NaN when that is negative).
    """
    p = noise_params(cfg)
    gt = _gt(ep)
    camera = ep.camera()
    joints_cam = np.asarray(gt["joints_cam"], dtype=np.float64)
    visible = np.asarray(gt["visible"], dtype=bool)
    T = joints_cam.shape[0]
    kp = np.full((T, 2, 21, 3), np.nan)
    hand_score = np.full((T, 2), np.nan)
    mp_world = np.full((T, 2, 21, 3), np.nan)
    uv = camera.project(joints_cam)
    sigma = float(p["hands2d_sigma_px"])
    for i in range(T):
        rng = _rng(p["seed"], 1, i)
        noise = rng.normal(scale=sigma, size=(2, 21, 2))
        drop = rng.random(2) < float(p["hands2d_dropout"])
        world_noise = rng.normal(scale=max(float(p["mp_world_sigma_m"]), 0.0), size=(2, 21, 3))
        for h in range(2):
            vis = visible[i, h] & np.all(np.isfinite(uv[i, h]), axis=-1)
            if drop[h] or vis.sum() < int(p["hands2d_min_visible"]):
                continue
            kp[i, h, vis, :2] = uv[i, h, vis] + noise[h, vis]
            kp[i, h, vis, 2] = float(p["hands2d_score"])
            hand_score[i, h] = float(p["hands2d_score"])
            if float(p["mp_world_sigma_m"]) >= 0:
                J = joints_cam[i, h]
                centre = J[[0, 5, 9, 13, 17]].mean(axis=0)
                mp_world[i, h] = J - centre + world_noise[h]
    return {"kp": kp, "hand_score": hand_score, "mp_world": mp_world}


# =============================================================================================
# depth
# =============================================================================================


def depth_frames(
    ep: Any, cfg: Mapping[str, Any] | None = None, frames: Iterable[int] | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, depth)`` with ``depth`` float16 ``(h, w)`` metres at
    ``depth.downsample`` (block mean of the full-resolution ground truth, ``cv2.INTER_AREA``),
    multiplied by ``depth_scale`` x (1 + ``depth_drift`` sin(2 pi t / period)) x (1 + N(0,
    ``depth_noise``)) per pixel."""
    p = noise_params(cfg)
    gt = _gt(ep)
    full = gt.depth()
    T, H, W = full.shape
    w, h = geo.downsample_size(W, H, _downsample(cfg))
    times = np.asarray(gt.get("t", np.arange(T) / 24.0), dtype=np.float64)
    period = float(p["depth_drift_period_s"])
    indices = range(T) if frames is None else [int(i) for i in frames]
    for i in indices:
        if not 0 <= i < T:
            raise IndexError(f"frame {i} out of range 0..{T - 1}")
        d = np.asarray(full[i], dtype=np.float32)
        if (w, h) != (W, H):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_AREA)
        drift = 1.0 + float(p["depth_drift"]) * math.sin(2 * math.pi * times[i] / period) if period > 0 else 1.0
        noise = 1.0 + _rng(p["seed"], 2, i).normal(scale=float(p["depth_noise"]), size=d.shape)
        yield i, (d * float(p["depth_scale"]) * drift * noise).astype(np.float16)


def depth(ep: Any, cfg: Mapping[str, Any] | None = None, frames: Iterable[int] | None = None) -> dict[str, Any]:
    """All requested frames at once: ``{"depth" [F,h,w] float16, "frames" [F] int64, "scale" h/H}``.
    (For long clips iterate :func:`depth_frames` and write a memmap instead.)"""
    gt = _gt(ep)
    H = gt.depth().shape[1]
    items = list(depth_frames(ep, cfg, frames))
    stack = np.stack([d for _, d in items]) if items else np.zeros((0, 0, 0), np.float16)
    return {
        "depth": stack,
        "frames": np.array([i for i, _ in items], dtype=np.int64),
        "scale": (stack.shape[1] / H) if items else float("nan"),
    }


def depth_scale_truth(ep: Any, cfg: Mapping[str, Any] | None = None) -> np.ndarray:
    """``[T]`` true multiplicative error of :func:`depth` per frame (without the pixel noise):
    metric depth = backend depth / this. Useful to test scale estimation."""
    p = noise_params(cfg)
    gt = _gt(ep)
    T = gt.n_frames
    times = np.asarray(gt.get("t", np.arange(T) / 24.0), dtype=np.float64)
    period = float(p["depth_drift_period_s"])
    drift = 1.0 + float(p["depth_drift"]) * np.sin(2 * np.pi * times / period) if period > 0 else np.ones(T)
    return float(p["depth_scale"]) * drift


# =============================================================================================
# masks
# =============================================================================================


def masks(ep: Any, cfg: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Per object id: ``{"frames" [F] int64, "packed" [F, ceil(h*w/8)] uint8 (np.packbits of the
    flattened bool mask), "shape" (h, w), "prompt": {"frame", "bbox" [x0, y0, x1, y1]}}`` at
    ``depth.downsample``; a mask pixel is set when at least half of its full-resolution block is
    the object. Frames with fewer than ``mask_min_pixels`` pixels are left out."""
    p = noise_params(cfg)
    gt = _gt(ep)
    data = gt.masks()
    H, W = (int(v) for v in data["shape"])
    w, h = geo.downsample_size(W, H, _downsample(cfg))
    ids = [str(v) for v in data["object_ids"]]
    out: dict[str, dict[str, Any]] = {}
    for k, obj in enumerate(ids):
        frames, packed = [], []
        prompt = None
        for row, frame in enumerate(data["frames"]):
            m = unpack_mask(data["packed"][row, k], (H, W))
            if (w, h) != (W, H):
                m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= 0.5
            if m.sum() < int(p["mask_min_pixels"]):
                continue
            frames.append(int(frame))
            packed.append(pack_mask(m))
            if prompt is None:
                ys, xs = np.nonzero(m)
                prompt = {"frame": int(frame), "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]}
        out[obj] = {
            "frames": np.array(frames, dtype=np.int64),
            "packed": np.stack(packed) if packed else np.zeros((0, (h * w + 7) // 8), np.uint8),
            "shape": (h, w),
            "prompt": prompt,
        }
    return out
