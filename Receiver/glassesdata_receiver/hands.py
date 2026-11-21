"""Hand skeletons with MediaPipe HandLandmarker (21 joints per hand, 2D image + 3D world)."""

from __future__ import annotations

import logging
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from .paths import cache_dir

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL = cache_dir() / "hand_landmarker.task"
CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


@dataclass(frozen=True)
class Hand:
    handedness: str  # "Left" / "Right": the person's own hand (see anatomical_label)
    score: float
    landmarks: list[tuple[float, float, float]]  # x, y normalised to the image; z relative depth
    world_landmarks: list[tuple[float, float, float]]  # metres, origin near the hand's centre


def ensure_model(path: Path = DEFAULT_MODEL) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading the hand landmark model to %s", path)
    with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
        data = response.read()
    # Write atomically so an interrupted download never leaves a broken model behind.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(data)
    os.replace(handle.name, path)
    return path


def anatomical_label(mediapipe_label: str, mirrored_input: bool) -> str:
    """The person's own hand, "Left" or "Right".

    MediaPipe's documentation says its labels assume a mirrored (selfie) image. Measured
    with the Tasks HandLandmarker on Google's woman_hands.jpg, which is not mirrored, the
    labels already name the person's own hands (arm, palm and thumb agree), and flipping
    the image swaps them. So a normal camera, like the glasses', keeps the label, and only
    a mirrored source needs the swap.
    """
    if not mirrored_input:
        return mediapipe_label
    return {"Left": "Right", "Right": "Left"}.get(mediapipe_label, mediapipe_label)


def handedness_note(mirrored_input: bool) -> str:
    """The convention, as recorded in info.json."""
    source = "mirrored source, MediaPipe labels swapped" if mirrored_input else "non-mirrored camera, MediaPipe labels kept"
    return f"anatomical: Left is the person's own left hand ({source})"


class HandTracker:
    """VIDEO running mode: tracks between frames, so timestamps must increase."""

    def __init__(self, model_path: Path | None = None, max_hands: int = 2, mirrored_input: bool = False) -> None:
        self._mirrored_input = mirrored_input
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path or ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, image_bgr: np.ndarray, timestamp_ms: int) -> list[Hand]:
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        hands: list[Hand] = []
        for points, world, categories in zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness):
            best = max(categories, key=lambda c: c.score)
            hands.append(
                Hand(
                    handedness=anatomical_label(best.category_name, self._mirrored_input),
                    score=float(best.score),
                    landmarks=[(p.x, p.y, p.z) for p in points],
                    world_landmarks=[(p.x, p.y, p.z) for p in world],
                )
            )
        return hands

    def close(self) -> None:
        self._landmarker.close()


# The same colours as the web page draws (BGR): cyan for the left hand, pink for the right.
HAND_COLORS = {"Left": (245, 214, 61), "Right": (182, 114, 244)}
_SHADOW = (18, 18, 18)
_SHIFT = 4  # sub-pixel precision for OpenCV's anti-aliased drawing
_ONE = 1 << _SHIFT


def _fixed(point: tuple[float, float]) -> tuple[int, int]:
    return round(point[0] * _ONE), round(point[1] * _ONE)


def draw_hands(image: np.ndarray, hands: list[Hand]) -> None:
    """Draw skeletons in place: thin anti-aliased bones over a soft dark outline, small joints,
    and an "L" / "R" tag beyond the wrist. Sizes follow the image, so it reads at any size."""
    height, width = image.shape[:2]
    unit = max(1.0, min(width, height) / 360)  # 1 at 640x360, 2 at 1280x720
    bone = max(1, round(1.5 * unit))
    joint = max(2.0, 2.2 * unit)
    for hand in hands:
        color = HAND_COLORS.get(hand.handedness, (235, 235, 235))
        points = [(x * width, y * height) for x, y, *_ in hand.landmarks]
        fixed = [_fixed(point) for point in points]
        for start, end in CONNECTIONS:
            cv2.line(image, fixed[start], fixed[end], _SHADOW, bone + max(2, round(unit * 1.5)), cv2.LINE_AA, _SHIFT)
        for start, end in CONNECTIONS:
            cv2.line(image, fixed[start], fixed[end], color, bone, cv2.LINE_AA, _SHIFT)
        for index, point in enumerate(fixed):
            radius = joint * (1.35 if index == 0 else 1.0)
            cv2.circle(image, point, round((radius + unit) * _ONE), _SHADOW, -1, cv2.LINE_AA, _SHIFT)
            cv2.circle(image, point, round(radius * _ONE), (255, 255, 255), -1, cv2.LINE_AA, _SHIFT)
            cv2.circle(image, point, round(radius * 0.55 * _ONE), color, -1, cv2.LINE_AA, _SHIFT)
        _tag(image, hand.handedness[:1].upper() or "?", points, color, unit)


def _tag(image: np.ndarray, text: str, points: list[tuple[float, float]], color: tuple[int, int, int], unit: float) -> None:
    """A small round tag beyond the wrist, on the side away from the fingers, so it never covers the hand."""
    height, width = image.shape[:2]
    radius = 7.5 * unit
    wrist = np.array(points[0], float)
    away = wrist - np.mean(points, axis=0)
    away /= max(np.linalg.norm(away), 1e-6)
    centre = wrist + away * (radius + 6 * unit)
    cx = float(np.clip(centre[0], radius + 2, max(radius + 2, width - radius - 2)))
    cy = float(np.clip(centre[1], radius + 2, max(radius + 2, height - radius - 2)))
    cv2.circle(image, _fixed((cx, cy)), round((radius + unit) * _ONE), _SHADOW, -1, cv2.LINE_AA, _SHIFT)
    cv2.circle(image, _fixed((cx, cy)), round(radius * _ONE), color, -1, cv2.LINE_AA, _SHIFT)
    scale = 0.36 * unit
    weight = max(1, round(unit))
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, weight)
    origin = (round(cx - text_w / 2), round(cy + text_h / 2))
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_DUPLEX, scale, _SHADOW, weight, cv2.LINE_AA)
