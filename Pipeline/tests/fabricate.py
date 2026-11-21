"""Fabricate the upstream outputs (``hands2d``, ``depth``, ``hand3d``) of a synthetic recording
from its ground truth, in the section-5 formats, so that ``slam``/``imu``/``world``/``arms`` can be
developed and tested without the learned models.

>>> import sys; sys.path.insert(0, "Pipeline/tests")       # the test modules do the same
>>> from fabricate import fabricate_upstream
>>> info = fabricate_upstream("local/tmp/synth_demo")         # runs ingest first when needed
>>> # then: python -m glassesdata_pipeline run local/tmp/synth_demo --steps slam,imu,world,arms --no-deps

* ``hands2d.npz`` - :func:`backends.gt.hands2d` (pixel noise, dropouts, invisible joints NaN).
* ``depth/depth_f16.npy`` + ``depth/index.npz`` - :func:`backends.gt.depth_frames` (global scale
  error ``depth_scale``, slow drift, per-pixel noise; all ``[synth.gt_noise]``).
* ``hand3d.npz`` - ground-truth ``joints_cam`` plus noise: a per-frame, per-hand offset
  (``offset_sigma_m``, larger along the optical axis, like a depth error) and per-joint noise
  (``joint_sigma_m``); mean joint error about 1 cm with the defaults. ``valid`` = the hand is in
  ``hands2d`` and not dropped (``dropout`` per frame plus one burst gap of ``gap_s`` per hand);
  a fraction ``outlier_rate`` of the valid hands get an extra ``outlier_m`` offset (gross error).
  ``depth_scale_smooth`` = the true factor that makes the depth metric (``1 / depth_scale_truth``,
  so that ``metric = depth * depth_scale_smooth`` as in PIPELINE.md section 5 and ``eval``) times
  ``(1 + scale_bias) (1 + scale_error * sin(2 pi t / scale_period_s + 1))``; ``depth_scale`` is
  that with 3 % noise.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from glassesdata_pipeline.backends import gt as gtb
from glassesdata_pipeline.episode import Episode
from glassesdata_pipeline.runner import Runner


def fabricate_upstream(
    recording: str | Path | Episode,
    *,
    joint_sigma_m: float = 0.004,
    offset_sigma_m: tuple[float, float, float] = (0.003, 0.003, 0.007),
    dropout: float = 0.05,
    gap_s: float = 0.3,
    outlier_rate: float = 0.01,
    outlier_m: float = 0.08,
    scale_error: float = 0.02,
    scale_period_s: float = 5.0,
    scale_bias: float = 0.0,
    seed: int = 0,
    write: tuple[str, ...] = ("hands2d", "depth", "hand3d"),
) -> dict[str, Any]:
    """Write the fabricated upstream files into ``derived/``; returns a summary (and the true
    depth factor ``depth_scale_true`` for checks)."""
    ep = recording if isinstance(recording, Episode) else Episode(recording)
    if ep.gt is None:
        raise ValueError(f"{ep.path} is not a synthetic recording (no gt/)")
    if not ep.has("timeline.npz"):
        report = Runner(ep).run(steps=["ingest"], deps=False)
        if not report.ok:
            raise RuntimeError(report.table())
    cfg = ep.config
    gt = ep.gt
    T = gt.n_frames
    t = np.asarray(gt.get("t", np.arange(T) / 24.0), dtype=np.float64)
    rng = np.random.default_rng([int(seed), 77])
    meta = ep.step_meta(backend="fabricated")
    out: dict[str, Any] = {"frames": T}

    h2 = gtb.hands2d(ep, cfg)
    if "hands2d" in write:
        ep.save_npz("hands2d", meta, **h2)

    if "depth" in write:
        H, W = gt.depth().shape[1:]
        first = next(iter(gtb.depth_frames(ep, cfg, [0])))[1]
        h, w = first.shape
        arr = np.lib.format.open_memmap(ep.file("depth/depth_f16.npy"), mode="w+", dtype=np.float16, shape=(T, h, w))
        for i, d in gtb.depth_frames(ep, cfg):
            arr[i] = d
        arr.flush()
        del arr
        ep.save_npz("depth/index", meta, frames=np.arange(T, dtype=np.int64), scale=np.float64(h / H))
        out["depth_grid"] = [w, h]

    true_factor = 1.0 / gtb.depth_scale_truth(ep, cfg)  # metric = depth * factor
    out["depth_scale_true"] = true_factor
    if "hand3d" in write:
        J = np.asarray(gt["joints_cam"], dtype=np.float64)
        valid = np.isfinite(h2["hand_score"]) & (rng.random((T, 2)) >= dropout)
        n_gap = int(round(gap_s * 24.0))
        for hand in range(2):
            frames = np.flatnonzero(valid[:, hand])
            if n_gap > 0 and len(frames) > 3 * n_gap:
                start = int(frames[len(frames) // 2])
                valid[start:start + n_gap, hand] = False
        offset = rng.normal(size=(T, 2, 1, 3)) * np.asarray(offset_sigma_m)
        noise = rng.normal(scale=joint_sigma_m, size=J.shape)
        outlier = valid & (rng.random((T, 2)) < outlier_rate)
        direction = rng.normal(size=(T, 2, 1, 3))
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        joints = J + offset + noise + np.where(outlier[..., None, None], outlier_m * direction, 0.0)
        joints[~valid] = np.nan
        smooth = true_factor * (1.0 + scale_bias) * (1.0 + scale_error * np.sin(2 * math.pi * t / scale_period_s + 1.0))
        raw = smooth * (1.0 + rng.normal(scale=0.03, size=T))
        has_hand = np.any(valid, axis=1)
        ep.save_npz(
            "hand3d", meta,
            joints_cam=joints,
            valid=valid,
            reproj_px=np.where(valid, 2.0, np.nan),
            depth_scale=np.where(has_hand, raw, np.nan),
            depth_scale_smooth=smooth,
        )
        out.update({"hand3d_valid": valid.mean(axis=0).round(3).tolist(), "outliers": int(outlier.sum())})
    return out


def write_fake_hand3d_only(ep: Episode, **kwargs: Any) -> dict[str, Any]:
    """Only ``hand3d.npz`` (assumes hands2d/depth exist)."""
    return fabricate_upstream(ep, write=("hand3d",), **kwargs)
