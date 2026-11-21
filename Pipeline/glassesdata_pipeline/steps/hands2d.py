"""Step ``hands2d``: 2D hand keypoints for every video frame -> ``derived/hands2d.npz``.

Arrays (PIPELINE.md section 5): ``kp[T,2,21,3]`` (u, v in full-resolution pixels with centres at
integers, score), ``hand_score[T,2]``, ``mp_world[T,2,21,3]`` (hand-centred metric landmarks, NaN
when the backend has none). Slot 0 = left, 1 = right (anatomical). Missing = NaN.

Extra arrays (not in the contract, for diagnostics and later steps):

* ``slot_source[T,2]`` int8 (mediapipe): how the slot was chosen: 0 none, 1 label, 2 temporal
  continuity, 3 handedness score.
* ``joints_cam_model[T,2,21,3]`` (wilor, hamer): the model's own camera-frame 3D joints (metres,
  OpenCV axes, our focal length); ``hand3d`` does not need them.

Backends (``hands2d.backend``): ``mediapipe`` (default; :mod:`..backends.mediapipe_hands`),
``gt`` (synthetic recordings; :func:`..backends.gt.hands2d`), ``wilor`` / ``hamer`` (Linux + CUDA;
:mod:`..backends.wilor`, :mod:`..backends.hamer`).
"""

from __future__ import annotations

import logging

import numpy as np

from ..episode import Episode
from ..errors import PipelineError

NAME = "hands2d"
VERSION = 1
REQUIRES = ["ingest"]
OUTPUTS = ["hands2d.npz"]
CONFIG_SECTIONS = ["hands2d", "synth"]
BACKENDS = ("mediapipe", "gt", "wilor", "hamer")

log = logging.getLogger(__name__)


def INPUTS(cfg) -> list[str]:  # noqa: N802 - the runner's name for it
    backend = str(cfg.get("hands2d", {}).get("backend", "mediapipe"))
    common = ["derived/timeline.npz", "derived/camera.json"]
    if backend == "gt":
        return common + ["gt/gt.npz"]
    return common + ["video.mp4"]


def run(ep: Episode, cfg) -> dict:
    hcfg = cfg.get("hands2d", {}) or {}
    backend = str(hcfg.get("backend", "mediapipe"))
    T = len(ep.timeline()["t"])
    extra_meta: dict = {}
    if backend == "gt":
        from ..backends import gt as gtb

        out = gtb.hands2d(ep, cfg)
    elif backend == "mediapipe":
        from ..backends import mediapipe_hands

        out = mediapipe_hands.hands2d(ep, cfg)
    elif backend == "wilor":
        from ..backends import wilor

        out = wilor.hands2d(ep, cfg)
    elif backend == "hamer":
        from ..backends import hamer

        out = hamer.hands2d(ep, cfg)
    else:
        raise PipelineError(f"hands2d.backend must be one of {', '.join(BACKENDS)}, got {backend!r}")
    if "model" in out:
        extra_meta["model"] = str(out.pop("model"))
    out = {k: np.asarray(v) for k, v in out.items()}
    for key in ("kp", "hand_score", "mp_world"):
        if key not in out:
            raise PipelineError(f"hands2d backend {backend} returned no {key!r}")
    if out["kp"].shape[0] != T:
        raise PipelineError(f"hands2d backend {backend} returned {out['kp'].shape[0]} frames, the timeline has {T}")
    ep.save_npz("hands2d", ep.step_meta(backend=backend, **extra_meta), **out)
    found = np.isfinite(out["hand_score"])
    return {
        "frames": T,
        "backend": backend,
        "left": int(found[:, 0].sum()),
        "right": int(found[:, 1].sum()),
        "both": int(np.all(found, axis=1).sum()),
    }
