"""The live picture: the decoded frame with the hand skeleton drawn on it.

Used by the web page's MJPEG stream, the virtual camera (``--overlay``) and
``--snapshot``. Nothing else is drawn over the video: status lives on the web page.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .decoder import DecodedFrame
from .receiver import HandsResult

log = logging.getLogger(__name__)

HANDS_STALE_FRAMES = 8  # hide a skeleton that lags the picture by more than this
PLACEHOLDER_SIZE = (720, 405)  # height, width: 9:16, the glasses' portrait aspect


def hands_for(frame_index: int, hands: HandsResult | None) -> list:
    """The hands to draw on ``frame_index``: none if the result is missing or too old."""
    if hands is None or not 0 <= frame_index - hands.frame_index <= HANDS_STALE_FRAMES:
        return []
    return hands.hands


def annotate(
    frame: DecodedFrame, hands: HandsResult | None, draw: bool = True, max_height: int | None = None
) -> np.ndarray:
    """A new image: ``frame`` scaled down to ``max_height`` (never up), with the skeleton if ``draw``.

    The caller's frame is never modified. The skeleton is drawn after scaling, so its
    lines keep a visible thickness at the output size.
    """
    image = frame.image
    height, width = image.shape[:2]
    if max_height is not None and height > max_height:
        scale = max_height / height
        image = cv2.resize(image, (max(2, round(width * scale)), max_height), interpolation=cv2.INTER_AREA)
    else:
        image = image.copy()
    visible = hands_for(frame.index, hands) if draw else []
    if visible:
        from .hands import draw_hands  # imports MediaPipe; only needed when there are hands

        draw_hands(image, visible)
    return image


def placeholder(text: str = "Waiting for video") -> np.ndarray:
    image = np.zeros((*PLACEHOLDER_SIZE, 3), np.uint8)
    cv2.putText(image, text, (110, PLACEHOLDER_SIZE[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
    return image


def save_snapshot(path: Path, frame: DecodedFrame | None, hands: HandsResult | None, draw: bool) -> None:
    """``--snapshot``: the last frame at full size with the skeleton, or a placeholder without video."""
    image = annotate(frame, hands, draw) if frame is not None else placeholder()
    if not cv2.imwrite(str(path), image):
        log.error("Could not write the snapshot to %s", path)
