from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from glassesdata_receiver import viewer
from glassesdata_receiver.decoder import DecodedFrame
from glassesdata_receiver.receiver import HandsResult

pytest.importorskip("glassesdata_receiver.hands")  # draw_hands needs MediaPipe's joint list


@dataclass(frozen=True)
class FakeHand:
    handedness: str
    score: float
    landmarks: list
    world_landmarks: list


def hands_result(frame_index: int = 10) -> HandsResult:
    landmarks = [(0.3 + 0.01 * i, 0.4 + 0.01 * i, 0.0) for i in range(21)]
    return HandsResult(frame_index, 0, [FakeHand("Left", 0.9, landmarks, [(0.0, 0.0, 0.0)] * 21)], 12.0)


def frame(height: int = 1280, width: int = 720, index: int = 10, value: int = 0) -> DecodedFrame:
    return DecodedFrame(np.full((height, width, 3), value, np.uint8), 0, 0, index)


@pytest.mark.parametrize("size", [(1280, 720), (960, 540), (640, 480)])
def test_annotate_scales_down_to_the_stream_height_and_never_up(size):
    height, width = size
    source = frame(height, width, value=128)
    image = viewer.annotate(source, None, max_height=720)
    expected_height = min(height, 720)
    assert image.shape == (expected_height, round(width * expected_height / height), 3)
    assert source.image.max() == source.image.min() == 128  # the caller's frame is not drawn on
    assert image is not source.image


def test_annotate_draws_the_skeleton_only_when_asked_and_fresh():
    plain = viewer.annotate(frame(), None, max_height=720)
    drawn = viewer.annotate(frame(), hands_result(10), max_height=720)
    off = viewer.annotate(frame(), hands_result(10), draw=False, max_height=720)
    stale = viewer.annotate(frame(index=10 + viewer.HANDS_STALE_FRAMES + 1), hands_result(10), max_height=720)
    future = viewer.annotate(frame(index=5), hands_result(10), max_height=720)  # results never run ahead
    assert drawn.any() and not plain.any() and not off.any() and not stale.any() and not future.any()


def test_snapshot_writes_the_frame_or_a_placeholder(tmp_path):
    with_video = tmp_path / "video.png"
    viewer.save_snapshot(with_video, frame(), hands_result(10), draw=True)
    image = cv2.imread(str(with_video))
    assert image.shape == (1280, 720, 3) and image.any()  # full size, skeleton drawn
    without = tmp_path / "empty.png"
    viewer.save_snapshot(without, None, None, draw=True)
    assert cv2.imread(str(without)).shape == (*viewer.PLACEHOLDER_SIZE, 3)
