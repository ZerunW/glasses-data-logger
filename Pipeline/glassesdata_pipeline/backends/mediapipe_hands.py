"""MediaPipe Tasks HandLandmarker backend of step ``hands2d`` (Mac and Linux, CPU).

* Model file ``hand_landmarker.task``: ``hands2d.model_path`` when set (relative to the
  repository root), else the repository's ``local/cache/`` (the receiver's copy: same folder as
  ``Receiver/glassesdata_receiver/paths.py``). When the file is missing it is downloaded there
  from Google's official URL only if ``hands2d.allow_download`` is true and the environment
  variable ``GLASSESDATA_NO_DOWNLOAD`` is not set (tests set it).
* VIDEO running mode over every frame (MediaPipe tracks between frames, so timestamps, taken from
  the timeline, must increase), ``num_hands = 2``.
* Coordinates: MediaPipe's normalized ``x, y`` are fractions of the image width/height measured
  from the image edge, so the pipeline's pixel (centres at integers) is ``u = x * W - 0.5``,
  ``v = y * H - 0.5``. ``mp_world`` are the world landmarks (metres, origin near the hand centre;
  their x/y follow the image axes, see ``steps/hand3d.py`` for how they are used).
* Handedness: the glasses video is not mirrored. The receiver measured that the Tasks
  HandLandmarker labels on a non-mirrored image already name the person's own hand
  (``Receiver/glassesdata_receiver/hands.py``, :func:`anatomical_label`), so labels are kept and
  only swapped when ``hands2d.mirrored_input`` is true.
* Slots (0 = left, 1 = right): by label; when both detections claim the same side, or a single
  detection's label flickers, temporal continuity (distance to the slot's last wrist) and the
  handedness score decide (:func:`assign_slots`).
* ``hand_score`` is MediaPipe's handedness score (the Tasks API exposes no per-hand presence
  score); keypoint scores are the landmark ``presence`` when MediaPipe fills it, else the hand
  score.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import tempfile
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import paths
from ..errors import MissingDependency, PipelineError

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_NAME = "hand_landmarker.task"
# SHA-256 of hand_landmarker.task (float16, 7 819 105 bytes) as served at MODEL_URL in November 2026
# (the receiver's cached copy). A download with another hash is refused (hands2d.model_sha256).
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
NO_DOWNLOAD_ENV = "GLASSESDATA_NO_DOWNLOAD"
LEFT, RIGHT = 0, 1
# slot_source codes stored in hands2d.npz
SOURCE_NONE, SOURCE_LABEL, SOURCE_CONTINUITY, SOURCE_SCORE = 0, 1, 2, 3


# =============================================================================================
# Model file
# =============================================================================================


def cache_dir() -> Path:
    """``local/cache`` in the repository, shared with the receiver (``glassesdata_receiver.paths.cache_dir``)."""
    return paths.cache_dir()


def default_model_path() -> Path:
    return cache_dir() / MODEL_NAME


def resolve_model(model_path: str | os.PathLike[str] | None = None, allow_download: bool = True,
                  sha256: str | None = MODEL_SHA256) -> Path:
    """The model file to use; downloads it to the cache when missing and allowed. A download is
    saved only when its SHA-256 equals ``sha256`` (``""`` / None: no check)."""
    if model_path:
        path = paths.resolve(model_path)
        if not path.is_file():
            raise PipelineError(f"hands2d.model_path {path} does not exist")
        return path
    path = default_model_path()
    if path.is_file() and path.stat().st_size > 0:
        return path
    if not allow_download or os.environ.get(NO_DOWNLOAD_ENV):
        raise PipelineError(
            f"MediaPipe hand model not found at {path} and downloading is disabled "
            f"(hands2d.allow_download / {NO_DOWNLOAD_ENV}); download {MODEL_URL} there or set hands2d.model_path"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading the MediaPipe hand model to %s", path)
    with urllib.request.urlopen(MODEL_URL, timeout=60) as response:  # noqa: S310 - fixed https URL
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if sha256 and digest != str(sha256).strip().lower():
        raise PipelineError(
            f"the downloaded MediaPipe hand model ({MODEL_URL}, {len(data)} bytes) has SHA-256 {digest}, expected "
            f"{sha256}; nothing was saved. If Google has published a new model, check it and set "
            'hands2d.model_sha256 to the new value (or "" to skip the check)'
        )
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(data)
    os.replace(handle.name, path)
    return path


# =============================================================================================
# Handedness and slots
# =============================================================================================


def anatomical_label(mediapipe_label: str, mirrored_input: bool) -> str:
    """The person's own hand, ``"Left"`` or ``"Right"`` (same rule as the receiver): a
    non-mirrored camera keeps MediaPipe's label, a mirrored source swaps it."""
    if not mirrored_input:
        return mediapipe_label
    return {"Left": "Right", "Right": "Left"}.get(mediapipe_label, mediapipe_label)


@dataclass
class Detection:
    """One detected hand in pipeline units."""

    side: int  # anatomical label: 0 left, 1 right (-1 unknown)
    score: float  # handedness score
    uv: np.ndarray  # (21, 2) pixels, centres at integers
    kp_score: np.ndarray  # (21,)
    world: np.ndarray  # (21, 3) metres (NaN if absent)

    @property
    def wrist(self) -> np.ndarray:
        return self.uv[0]


def assign_slots(
    detections: Sequence[Detection],
    last_wrist: np.ndarray,
    keep_slot_px: float = 60.0,
    trust_label_score: float = 0.9,
) -> list[tuple[int, Detection, int]]:
    """Slots for the detections of one frame: ``[(slot, detection, source)]``.

    Args:
        detections: at most two hands.
        last_wrist: ``(2, 2)`` wrist pixel of each slot in a recent frame (NaN when unknown).
        keep_slot_px: a single detection whose label points at an empty slot but whose wrist is
            within this distance of the other slot's last wrist keeps that other slot (label
            flicker), unless its handedness score is at least ``trust_label_score``.

    Rules: distinct labels -> by label (source 1). Same label twice -> the assignment with the
    smaller summed wrist distance to the slots' last wrists (source 2); without any history the
    higher score keeps its label and the other hand takes the other slot (source 3).
    """
    dets = list(detections)[:2]
    last = np.asarray(last_wrist, dtype=np.float64).reshape(2, 2)
    known = np.all(np.isfinite(last), axis=-1)
    if not dets:
        return []
    if len(dets) == 1:
        d = dets[0]
        slot = d.side if d.side in (LEFT, RIGHT) else RIGHT
        other = 1 - slot
        if (not known[slot]) and known[other] and d.score < trust_label_score:
            if np.linalg.norm(d.wrist - last[other]) < keep_slot_px:
                return [(other, d, SOURCE_CONTINUITY)]
        return [(slot, d, SOURCE_LABEL)]
    a, b = dets
    if a.side in (LEFT, RIGHT) and b.side in (LEFT, RIGHT) and a.side != b.side:
        return [(a.side, a, SOURCE_LABEL), (b.side, b, SOURCE_LABEL)]
    side = a.side if a.side in (LEFT, RIGHT) else (b.side if b.side in (LEFT, RIGHT) else RIGHT)
    other = 1 - side

    def dist(det: Detection, slot: int) -> float:
        return float(np.linalg.norm(det.wrist - last[slot])) if known[slot] else math.nan

    if known[side] or known[other]:
        # cost of "a keeps `side`, b takes `other`" vs the swap, over the slots with history
        keep = np.nansum([dist(a, side), dist(b, other)])
        swap = np.nansum([dist(b, side), dist(a, other)])
        first, second = (a, b) if keep <= swap else (b, a)
        return [(side, first, SOURCE_CONTINUITY), (other, second, SOURCE_CONTINUITY)]
    first, second = (a, b) if a.score >= b.score else (b, a)
    return [(side, first, SOURCE_SCORE), (other, second, SOURCE_SCORE)]


# =============================================================================================
# Landmarker
# =============================================================================================


class HandLandmarker:
    """MediaPipe Tasks HandLandmarker in VIDEO mode returning :class:`Detection` objects."""

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        num_hands: int = 2,
        min_detection: float = 0.5,
        min_presence: float = 0.5,
        min_tracking: float = 0.5,
        mirrored_input: bool = False,
    ) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
        except ImportError as error:  # pragma: no cover - mediapipe is a core dependency
            raise MissingDependency("pip install mediapipe (needed for hands2d backend mediapipe)") from error
        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=int(num_hands),
            min_hand_detection_confidence=float(min_detection),
            min_hand_presence_confidence=float(min_presence),
            min_tracking_confidence=float(min_tracking),
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._mirrored = bool(mirrored_input)
        self._last_ms = -1

    def __enter__(self) -> "HandLandmarker":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._landmarker.close()

    def detect(self, image_bgr: np.ndarray, timestamp_ms: int) -> list[Detection]:
        """Hands in one BGR frame; ``timestamp_ms`` is forced to increase strictly."""
        import cv2

        timestamp_ms = max(int(timestamp_ms), self._last_ms + 1)
        self._last_ms = timestamp_ms
        H, W = image_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        return convert_result(result, W, H, self._mirrored)


def convert_result(result: Any, width: int, height: int, mirrored_input: bool = False) -> list[Detection]:
    """A ``HandLandmarkerResult`` (or anything with ``hand_landmarks``, ``hand_world_landmarks``,
    ``handedness``) -> :class:`Detection` list in pixels."""
    out: list[Detection] = []
    worlds = list(getattr(result, "hand_world_landmarks", None) or [])
    for k, (points, categories) in enumerate(zip(result.hand_landmarks, result.handedness)):
        best = max(categories, key=lambda c: c.score)
        label = anatomical_label(str(best.category_name), mirrored_input)
        side = {"Left": LEFT, "Right": RIGHT}.get(label, -1)
        score = float(best.score)
        uv = np.array([[p.x * width - 0.5, p.y * height - 0.5] for p in points], dtype=np.float64)
        presence = np.array([_finite(getattr(p, "presence", None)) for p in points])
        kp_score = np.where(np.isfinite(presence) & (presence > 0), presence, score)
        world = np.full((21, 3), np.nan)
        if k < len(worlds) and worlds[k]:
            world = np.array([[p.x, p.y, p.z] for p in worlds[k]], dtype=np.float64)
        out.append(Detection(side, score, uv, kp_score, world))
    return out


def _finite(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


# =============================================================================================
# Whole video
# =============================================================================================


def track_frames(
    frames: Iterable[tuple[int, np.ndarray]],
    timestamps_ms: np.ndarray,
    n_frames: int,
    landmarker: Any,
    keep_slot_px: float = 60.0,
    trust_label_score: float = 0.9,
    continuity_frames: int = 12,
) -> dict[str, np.ndarray]:
    """Run ``landmarker.detect(image, ms)`` over ``(index, image)`` pairs and fill the section-5
    arrays (``kp``, ``hand_score``, ``mp_world``) plus ``slot_source[T,2]`` int8 (0 none, 1 label,
    2 continuity, 3 score)."""
    kp = np.full((n_frames, 2, 21, 3), np.nan)
    hand_score = np.full((n_frames, 2), np.nan)
    mp_world = np.full((n_frames, 2, 21, 3), np.nan)
    source = np.zeros((n_frames, 2), np.int8)
    last_wrist = np.full((2, 2), np.nan)
    last_seen = np.full(2, -10**9)
    for index, image in frames:
        if not 0 <= index < n_frames:
            continue
        stale = index - last_seen > continuity_frames
        last_wrist[stale] = np.nan
        dets = landmarker.detect(image, int(timestamps_ms[index]))
        for slot, det, how in assign_slots(dets, last_wrist, keep_slot_px, trust_label_score):
            kp[index, slot, :, :2] = det.uv
            kp[index, slot, :, 2] = det.kp_score
            hand_score[index, slot] = det.score
            mp_world[index, slot] = det.world
            source[index, slot] = how
            last_wrist[slot] = det.wrist
            last_seen[slot] = index
    return {"kp": kp, "hand_score": hand_score, "mp_world": mp_world, "slot_source": source}


def timestamps_ms(t: np.ndarray) -> np.ndarray:
    """Strictly increasing integer milliseconds from episode seconds."""
    ms = np.round(np.asarray(t, dtype=np.float64) * 1000.0).astype(np.int64)
    for i in range(1, len(ms)):
        if ms[i] <= ms[i - 1]:
            ms[i] = ms[i - 1] + 1
    return ms


def hands2d(ep: Any, cfg: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The ``hands2d`` arrays for a recording (every frame of ``video.mp4``)."""
    hcfg = cfg.get("hands2d", {}) or {}
    model = resolve_model(hcfg.get("model_path") or None, bool(hcfg.get("allow_download", True)),
                          str(hcfg.get("model_sha256", MODEL_SHA256)))
    timeline = ep.timeline()
    T = len(timeline["t"])
    camera = ep.camera()
    width = camera.width
    keep_px = float(hcfg.get("keep_slot_frac", 0.08)) * width
    with HandLandmarker(
        model,
        num_hands=int(hcfg.get("num_hands", 2)),
        min_detection=float(hcfg.get("min_detection_confidence", 0.5)),
        min_presence=float(hcfg.get("min_presence_confidence", 0.5)),
        min_tracking=float(hcfg.get("min_tracking_confidence", 0.5)),
        mirrored_input=bool(hcfg.get("mirrored_input", False)),
    ) as landmarker:
        out = track_frames(
            ep.frames(), timestamps_ms(timeline["t"]), T, landmarker,
            keep_slot_px=keep_px,
            trust_label_score=float(hcfg.get("trust_label_score", 0.9)),
            continuity_frames=int(hcfg.get("continuity_frames", 12)),
        )
    out["model"] = str(model)
    return out
