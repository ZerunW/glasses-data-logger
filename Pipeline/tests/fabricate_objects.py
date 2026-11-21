"""Fabricated upstream outputs for testing the ``objects`` step on a synthetic recording.

The steps that normally produce ``objects``' inputs (hands2d, depth, hand3d, slam/imu/world) are
written by other modules; this helper writes their section-5 files from the ground truth with
realistic errors, so ``objects`` can be developed and evaluated on its own:

* ``timeline.npz``, ``camera.json``, ``profile.json``: the real ``ingest`` step.
* ``depth/depth_f16.npy`` + ``depth/index.npz``: ``backends.gt.depth`` (global scale error 1.10,
  slow drift 5 %, 3 % per-pixel noise by default: ``[synth.gt_noise]``).
* ``hand3d.npz``: ``depth_scale_smooth`` = 1 / ``backends.gt.depth_scale_truth`` x (1 + a slow
  ``scale_error`` wobble), i.e. the metric correction a good hand3d step would find;
  ``joints_cam`` = ground truth + noise.
* ``camera_traj.npz``: ground-truth ``T_world_cam`` with a slow random-walk drift
  (``cam_drift_m`` over the clip, ``cam_drift_deg``) plus white noise (``cam_noise_m``, ``cam_noise_deg``).
* ``hands_world.npz``: ground-truth joints + temporally correlated noise of ``hand_noise_m`` per
  axis (Gaussian-smoothed white noise, correlation ``hand_noise_corr_s``), ``raw`` with extra white noise.
* ``hands2d.npz``: ``backends.gt.hands2d``.

Every file's meta has ``"fabricated": true`` and ``step`` set to the producing step's name.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from glassesdata_pipeline import geometry as geo
from glassesdata_pipeline import skeleton as sk
from glassesdata_pipeline.backends import gt as gtb
from glassesdata_pipeline.episode import Episode

DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "scale_error": 0.01,       # amplitude of the relative error of depth_scale_smooth
    "cam_noise_m": 0.002,      # white noise on camera positions
    "cam_noise_deg": 0.15,     # white noise on camera rotations
    "cam_drift_m": 0.01,       # random-walk drift of the camera position over the clip (end-point std)
    "cam_drift_deg": 0.3,      # random-walk drift of the camera rotation over the clip
    "hand_noise_m": 0.01,      # per-axis std of the (smoothed) hands_world joint noise
    "hand_noise_corr_s": 0.08, # temporal correlation of that noise (Gaussian sigma, seconds)
    "hand_raw_noise_m": 0.005, # extra white noise in hands_world.raw
}


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, ...], sigma_frames: float, std: float) -> np.ndarray:
    white = rng.normal(size=shape)
    if sigma_frames <= 0.3:
        return white * std
    r = int(math.ceil(3 * sigma_frames))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_frames) ** 2)
    k /= k.sum()
    pad = np.concatenate([white[r:0:-1], white, white[-2:-r - 2:-1]], axis=0)
    out = np.zeros_like(white)
    for i, w in enumerate(k):
        out += w * pad[i:i + shape[0]]
    return out / out.std() * std


def _random_walk(rng: np.random.Generator, T: int, dim: int, end_std: float) -> np.ndarray:
    steps = rng.normal(size=(T, dim)) * (end_std / math.sqrt(max(T, 1)))
    walk = np.cumsum(steps, axis=0)
    return walk - walk[0]


def fabricate_upstream(ep: Episode, **overrides: Any) -> dict[str, Any]:
    """Write the fabricated upstream files into ``ep.derived``; returns the parameters used."""
    from glassesdata_pipeline.runner import Runner

    p = {**DEFAULTS, **overrides}
    report = Runner(ep).run(["ingest"], deps=False)
    if not report.ok:
        raise RuntimeError(f"ingest failed: {report.table()}")
    cfg = ep.config
    gt = ep.gt
    T = gt.n_frames
    fps = 1.0 / float(np.median(np.diff(gt["t"]))) if T > 1 else 24.0
    rng = np.random.default_rng(int(p["seed"]) + 7)

    def meta(step: str) -> dict[str, Any]:
        return {"step": step, "version": 0, "fabricated": True, "params": {"fabricate": p}, "backend": "fabricated"}

    # depth (gt backend, memmap)
    first = next(gtb.depth_frames(ep, cfg, [0]))[1]
    h, w = first.shape
    depth = np.lib.format.open_memmap(ep.file("depth/depth_f16.npy"), mode="w+", dtype=np.float16, shape=(T, h, w))
    for i, d in gtb.depth_frames(ep, cfg):
        depth[i] = d
    depth.flush()
    del depth
    H = gt.depth().shape[1]
    ep.save_npz("depth/index", meta("depth"), frames=np.arange(T, dtype=np.int64), scale=h / H)

    # hand3d: the metric correction of the depth
    truth = gtb.depth_scale_truth(ep, cfg)
    t = np.asarray(gt["t"], dtype=np.float64)
    wobble = 1.0 + float(p["scale_error"]) * np.sin(2 * np.pi * t / 3.7 + 0.4)
    scale_smooth = wobble / truth
    raw_scale = scale_smooth * (1.0 + rng.normal(scale=0.03, size=T))
    jc = np.asarray(gt["joints_cam"], dtype=np.float64) + rng.normal(scale=0.008, size=gt["joints_cam"].shape)
    ep.save_npz("hand3d", meta("hand3d"), joints_cam=jc, valid=np.ones((T, 2), bool), reproj_px=np.full((T, 2), 2.0),
                depth_scale=raw_scale, depth_scale_smooth=scale_smooth)

    # camera trajectory
    T_gt = np.asarray(gt["T_world_cam"], dtype=np.float64)
    drift_t = _random_walk(rng, T, 3, float(p["cam_drift_m"]))
    drift_r = _random_walk(rng, T, 3, math.radians(float(p["cam_drift_deg"])))
    noise_t = rng.normal(scale=float(p["cam_noise_m"]), size=(T, 3))
    noise_r = rng.normal(scale=math.radians(float(p["cam_noise_deg"])), size=(T, 3))
    T_est = T_gt.copy()
    T_est[:, :3, :3] = geo.so3_exp(drift_r + noise_r) @ T_gt[:, :3, :3]
    T_est[:, :3, 3] = T_gt[:, :3, 3] + drift_t + noise_t
    ep.save_npz("camera_traj", meta("world"), T_world_cam=T_est, valid=np.ones(T, bool))

    # hands in the world
    J = np.asarray(gt["joints_world"], dtype=np.float64)
    noise = _smooth_noise(rng, J.shape, float(p["hand_noise_corr_s"]) * fps, float(p["hand_noise_m"]))
    joints = J + noise
    raw = joints + rng.normal(scale=float(p["hand_raw_noise_m"]), size=J.shape)
    bones = np.stack([np.linalg.norm(J[0][:, i] - J[0][:, sk.PARENTS[i]], axis=-1) for i in range(1, 21)], axis=-1)
    ep.save_npz("hands_world", meta("world"), joints=joints, raw=raw, observed=np.ones((T, 2), bool),
                filled=np.zeros((T, 2), bool), bone_lengths=bones)

    # hands2d
    h2 = gtb.hands2d(ep, cfg)
    ep.save_npz("hands2d", meta("hands2d"), **h2)
    return p
