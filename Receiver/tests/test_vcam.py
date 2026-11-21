"""The virtual camera: landscape composition, the sender thread (with a fake camera) and the start-up check."""

import os
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from glassesdata_receiver import vcam
from glassesdata_receiver.decoder import DecodedFrame
from glassesdata_receiver.receiver import HandsResult

SOURCES = [(720, 1280), (540, 960), (1280, 720), (1920, 1080), (1280, 960), (2000, 1000), (100, 1000)]


def stripes(width: int, height: int, bands: int = 8) -> np.ndarray:
    """Horizontal bands of different grey levels, so a crop's vertical position can be read off the canvas."""
    image = np.zeros((height, width, 3), np.uint8)
    for band in range(bands):
        image[band * height // bands : (band + 1) * height // bands] = 20 + band * 30
    return image


# ---------------------------------------------------------------- canvas and layout


@pytest.mark.parametrize("canvas", vcam.CANVAS_SIZES)
@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("mode, position", [("fit", 0.5), ("fill", 0.0), ("fill", 0.5), ("fill", 1.0)])
def test_the_canvas_is_always_the_configured_landscape_size(canvas, source, mode, position):
    picture = vcam.compose(np.full((source[1], source[0], 3), 90, np.uint8), canvas, mode, position)
    assert picture.shape == (canvas[1], canvas[0], 3) and picture.dtype == np.uint8
    assert canvas[0] > canvas[1]
    plan = vcam.layout(source, canvas, mode, position)
    x, y, w, h = plan.video
    assert 0 <= x and 0 <= y and x + w <= canvas[0] and y + h <= canvas[1]
    cx, cy, cw, ch = plan.crop
    assert 0 <= cx and 0 <= cy and cx + cw <= source[0] and cy + ch <= source[1]


def test_fit_places_a_portrait_frame_at_full_height_in_the_middle():
    assert vcam.layout((720, 1280), (1920, 1080), "fit") == vcam.Layout((1920, 1080), (656, 0, 608, 1080), (0, 0, 720, 1280), True)
    assert vcam.layout((720, 1280), (1280, 720), "fit").video == (437, 0, 405, 720)


def test_fit_with_landscape_sources_is_exact_or_letterboxed():
    assert vcam.layout((1280, 720), (1920, 1080), "fit") == vcam.Layout((1920, 1080), (0, 0, 1920, 1080), (0, 0, 1280, 720), False)
    assert vcam.layout((1280, 960), (1920, 1080), "fit").video == (240, 0, 1440, 1080)  # 4:3: narrow sides
    assert vcam.layout((2000, 1000), (1920, 1080), "fit").video == (0, 60, 1920, 960)  # wider: bars above and below


def test_fill_covers_the_canvas_with_a_band_placed_by_position():
    bands = {position: vcam.layout((720, 1280), (1920, 1080), "fill", position) for position in (0.0, 0.5, 1.0)}
    for plan in bands.values():
        assert plan.video == (0, 0, 1920, 1080) and not plan.backdrop
        assert plan.crop[2:] == (720, 405)  # 16:9 of the full width
    assert [bands[p].crop[1] for p in (0.0, 0.5, 1.0)] == [0, 438, 875]
    # a wider source is cropped at the sides, centred
    assert vcam.layout((2000, 1000), (1920, 1080), "fill").crop == (111, 0, 1778, 1000)
    assert vcam.layout((1280, 720), (1280, 720), "fill").crop == (0, 0, 1280, 720)


def test_fill_position_picks_different_bands_of_the_picture():
    source = stripes(720, 1280)
    centre = {p: int(vcam.compose(source, (1920, 1080), "fill", p)[540, 960, 0]) for p in (0.0, 0.5, 1.0)}
    assert centre[0.0] < centre[0.5] < centre[1.0]
    top = vcam.compose(source, (1920, 1080), "fill", 0.0)
    assert int(top[5, 960, 0]) == 20  # the first band of the frame is at the top of the canvas
    bottom = vcam.compose(source, (1920, 1080), "fill", 1.0)
    assert int(bottom[-5, 960, 0]) == 20 + 7 * 30  # the last band at the bottom


def test_fit_shows_the_whole_frame_and_blurred_darker_sides():
    source = np.zeros((1280, 720, 3), np.uint8)
    source[::2, ::2] = 255  # fine detail that a blur removes
    picture = vcam.compose(source, (1920, 1080), "fit")
    x, y, w, h = vcam.layout((720, 1280), (1920, 1080), "fit").video
    middle = picture[:, x : x + w].astype(float)
    sides = np.concatenate([picture[:, : x - 4], picture[:, x + w + 4 :]], axis=1).astype(float)
    assert sides.std() < middle.std() / 4  # blurred
    assert sides.mean() < middle.mean() * 0.75  # darkened
    assert sides.mean() > 10  # but not black: it is the frame, enlarged


def test_the_card_has_the_canvas_size_and_is_calm():
    for size in vcam.CANVAS_SIZES:
        card = vcam.placeholder_card(size, "No iPhone connected")
        assert card.shape == (size[1], size[0], 3)
        assert card.mean() < 40 and card.max() > 200  # dark, with readable text


def test_parse_size():
    assert vcam.parse_size("1920x1080") == (1920, 1080) and vcam.parse_size("1280X720") == (1280, 720)
    for bad in ("720x1280", "1080x1920", "640x480", "big", "1920"):
        with pytest.raises(ValueError):
            vcam.parse_size(bad)


# ---------------------------------------------------------------- the skeleton overlay


@dataclass(frozen=True)
class FakeHand:
    handedness: str
    score: float
    landmarks: list
    world_landmarks: list = ()


def hand_at(cx: float, cy: float, spread: float = 0.04) -> FakeHand:
    points = [(cx + spread * np.cos(i / 3), cy + spread * np.sin(i / 3), 0.0) for i in range(21)]
    return FakeHand("Left", 0.9, points)


def test_the_overlay_is_drawn_inside_the_video_region_only():
    pytest.importorskip("glassesdata_receiver.hands")  # draw_hands needs MediaPipe's joint list
    black = np.zeros((1280, 720, 3), np.uint8)
    edge_hands = [hand_at(0.03, 0.5), hand_at(0.97, 0.2), hand_at(0.5, 0.97)]  # touching the frame's edges
    picture = vcam.compose(black, (1920, 1080), "fit", hands=edge_hands)
    x, y, w, h = vcam.layout((720, 1280), (1920, 1080), "fit").video
    drawn = np.argwhere(picture.max(axis=2) > 0)
    assert len(drawn) > 100
    assert drawn[:, 1].min() >= x and drawn[:, 1].max() < x + w  # nothing on the blurred sides


def test_the_overlay_follows_the_fill_band():
    pytest.importorskip("glassesdata_receiver.hands")
    black = np.zeros((1280, 720, 3), np.uint8)
    low = [hand_at(0.5, 0.85)]  # a hand low in the frame
    at_bottom = vcam.compose(black, (1920, 1080), "fill", 1.0, low)
    at_top = vcam.compose(black, (1920, 1080), "fill", 0.0, low)
    assert not at_top.any()  # outside the band: not drawn at all
    rows, cols = np.nonzero(at_bottom.max(axis=2))
    # crop (0, 875, 720, 405): y 0.85 * 1280 = 1088 -> (1088 - 875) / 405 of the canvas height
    assert abs(rows.mean() - (1088 - 875) / 405 * 1080) < 40 and abs(cols.mean() - 960) < 40


# ---------------------------------------------------------------- the sender thread


class FakeCamera:
    opened: list = []

    def __init__(self, width, height, fps, fail_after=None):
        self.width, self.height, self.fps = width, height, fps
        self.device, self.backend = "Fake Camera", "fake"
        self.sent: list = []  # (time, picture)
        self.closed = False
        self.fail_after = fail_after
        FakeCamera.opened.append(self)

    def send(self, picture):
        assert picture.shape == (self.height, self.width, 3) and picture.dtype == np.uint8
        if self.fail_after is not None and len(self.sent) >= self.fail_after:
            raise OSError("device unplugged")
        self.sent.append((time.monotonic(), picture))

    def close(self):
        self.closed = True


class FakeReceiver:
    def __init__(self, frame=None, busy=True):
        self.busy = busy
        self.frame = frame
        self.hands = None
        self.hands_active = True

    def latest_frame(self):
        return self.frame

    def latest_hands(self):
        return self.hands


def frame(index=1, width=720, height=1280, value=None):
    image = stripes(width, height) if value is None else np.full((height, width, 3), value, np.uint8)
    return DecodedFrame(image, index * 41_667, 0, index)


@pytest.fixture
def cameras():
    FakeCamera.opened = []
    return FakeCamera.opened


def wait_for(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.01)


def test_sends_at_a_steady_rate_repeating_the_picture_of_a_stalled_source(cameras):
    out = vcam.VirtualCamera(FakeReceiver(frame()), fps=40, open_camera=FakeCamera)
    assert out.start()[0]
    time.sleep(0.6)
    out.stop()
    (camera,) = cameras
    times = [t for t, _ in camera.sent]
    assert 20 <= len(times) <= 27  # 40 fps for 0.6 s
    gaps = np.diff(times)
    assert abs(np.median(gaps) - 1 / 40) < 0.006 and gaps.max() < 0.08
    assert len({id(picture) for _, picture in camera.sent}) == 1  # composed once, then repeated
    assert camera.closed and not out.running
    assert out.status()["frames_sent"] == len(times)


def test_a_new_frame_is_composed_once(cameras):
    receiver = FakeReceiver(frame(1))
    out = vcam.VirtualCamera(receiver, fps=50, open_camera=FakeCamera)
    out.start()
    wait_for(lambda: len(cameras[0].sent) >= 3)
    receiver.frame = frame(2, value=200)
    wait_for(lambda: len(cameras[0].sent) >= 10)
    out.stop()
    pictures = [picture for _, picture in cameras[0].sent]
    assert len({id(p) for p in pictures}) == 2
    assert pictures[-1][540, 960, 0] == 200


@pytest.mark.parametrize(
    "receiver, detail",
    [(FakeReceiver(None, busy=False), "No iPhone connected"), (FakeReceiver(None, busy=True), "Waiting for the glasses video"),
     (FakeReceiver(frame(), busy=False), "No iPhone connected")],
)
def test_the_card_is_sent_without_video(cameras, receiver, detail):
    out = vcam.VirtualCamera(receiver, size=(1280, 720), fps=50, open_camera=FakeCamera)
    out.start()
    wait_for(lambda: len(cameras[0].sent) >= 3)
    assert out.status()["showing"] == "card"
    out.stop()
    assert np.array_equal(cameras[0].sent[-1][1], vcam.placeholder_card((1280, 720), detail))


def test_a_video_that_stopped_turns_into_the_card(cameras, monkeypatch):
    monkeypatch.setattr(vcam, "STALE_AFTER_S", 0.2)
    out = vcam.VirtualCamera(FakeReceiver(frame()), fps=50, open_camera=FakeCamera)
    out.start()
    wait_for(lambda: out.status()["showing"] == "video")
    wait_for(lambda: out.status()["showing"] == "card", timeout=2)
    out.stop()
    assert np.array_equal(cameras[0].sent[-1][1], vcam.placeholder_card((1920, 1080), "The video stopped"))


def test_mode_position_and_overlay_changes_apply_live(cameras):
    receiver = FakeReceiver(frame())
    out = vcam.VirtualCamera(receiver, fps=50, open_camera=FakeCamera)
    out.start()
    wait_for(lambda: len(cameras[0].sent) >= 2)
    for mode, position in (("fill", 0.0), ("fill", 1.0), ("fit", 0.5)):
        out.set(mode=mode, position=position)
        expected = vcam.compose(receiver.frame.image, (1920, 1080), mode, position)
        wait_for(lambda: np.array_equal(cameras[0].sent[-1][1], expected))
    out.stop()
    status = out.status()
    assert (status["mode"], status["position"], status["running"]) == ("fit", 0.5, False)


def test_overlay_draws_the_hands_of_this_frame(cameras):
    pytest.importorskip("glassesdata_receiver.hands")
    receiver = FakeReceiver(frame(5, value=0))
    receiver.hands = HandsResult(5, 0, [hand_at(0.5, 0.5)], 10.0)
    out = vcam.VirtualCamera(receiver, fps=50, open_camera=FakeCamera)
    out.start()
    wait_for(lambda: len(cameras[0].sent) >= 2)
    assert not cameras[0].sent[-1][1][300:780, 700:1220].any()  # overlay off
    out.set(overlay=True)
    wait_for(lambda: cameras[0].sent[-1][1][300:780, 700:1220].any())
    receiver.hands_active = False  # hand tracking turned off on the page: no skeleton
    wait_for(lambda: not cameras[0].sent[-1][1][300:780, 700:1220].any())
    out.stop()


def test_restart_after_stop_opens_the_device_again(cameras):
    out = vcam.VirtualCamera(FakeReceiver(frame()), fps=50, open_camera=FakeCamera)
    assert out.stop() == (True, "The virtual camera is already off")
    for _ in range(2):
        ok, message = out.start()
        assert ok and message == "Virtual camera on: Fake Camera"
        assert out.start()[1] == "The virtual camera is already on (Fake Camera)"
        wait_for(lambda: out.status()["frames_sent"] >= 3)
        assert out.stop() == (True, "Virtual camera off")
    assert len(cameras) == 2 and all(camera.closed for camera in cameras)
    status = out.status()
    assert status["running"] is False and status["device"] == "Fake Camera" and status["frames_sent"] >= 3


def test_after_close_a_start_is_refused(cameras):
    out = vcam.VirtualCamera(FakeReceiver(frame()), fps=50, open_camera=FakeCamera)
    out.start()
    out.close()
    assert not out.running and cameras[0].closed
    assert out.start() == (False, "The receiver is shutting down") and len(cameras) == 1


def test_a_device_that_cannot_open_leaves_a_reason_and_a_hint():
    def refuse(width, height, fps):
        raise vcam.Unavailable("the virtual camera device could not be opened (no device)", "Do this.")

    out = vcam.VirtualCamera(FakeReceiver(frame()), open_camera=refuse)
    ok, message = out.start()
    assert not ok and message == "Virtual camera unavailable: the virtual camera device could not be opened (no device)"
    status = out.status()
    assert status["running"] is False and status["last_error"].startswith("the virtual camera device") and status["hint"] == "Do this."


def test_missing_pyvirtualcam_is_reported_in_words(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyvirtualcam", None)  # makes "import pyvirtualcam" fail
    out = vcam.VirtualCamera(FakeReceiver(frame()))
    assert out.status()["available"] is False
    ok, _ = out.start()
    status = out.status()
    assert not ok and status["last_error"] == "pyvirtualcam is not installed" and "pip install pyvirtualcam" in status["hint"]


@pytest.mark.parametrize("platform, words", [("darwin", "OBS"), ("linux", "modprobe v4l2loopback"), ("win32", "OBS")])
def test_setup_hints_per_platform(platform, words):
    assert words in vcam.setup_hint(platform)


def test_a_send_failure_stops_the_thread_and_says_why(cameras):
    out = vcam.VirtualCamera(FakeReceiver(frame()), fps=50, open_camera=lambda w, h, f: FakeCamera(w, h, f, fail_after=3))
    out.start()
    wait_for(lambda: not out.running)
    status = out.status()
    assert status["last_error"] == "sending failed: device unplugged" and status["frames_sent"] == 3
    assert cameras[0].closed


def test_settings_are_validated():
    out = vcam.VirtualCamera(FakeReceiver(), open_camera=FakeCamera)
    for bad in ({"mode": "zoom"}, {"position": -0.1}, {"position": 1.5}, {"position": True}, {"position": "0.5"},
                {"position": float("nan")}, {"overlay": "yes"}, {"overlay": 1}):
        with pytest.raises(ValueError):
            out.set(**bad)
    out.set(mode="fill", position=1, overlay=True)
    assert {k: out.status()[k] for k in ("mode", "position", "overlay")} == {"mode": "fill", "position": 1.0, "overlay": True}
    with pytest.raises(ValueError):
        vcam.VirtualCamera(FakeReceiver(), size=(720, 1280), open_camera=FakeCamera)  # never the portrait size


def test_the_preview_is_the_picture_being_sent(cameras):
    out = vcam.VirtualCamera(FakeReceiver(frame()), fps=50, open_camera=FakeCamera)
    assert out.preview_jpeg() is None
    out.start()
    wait_for(lambda: len(cameras[0].sent) >= 2)
    jpeg = out.preview_jpeg()
    assert out.preview_jpeg() is jpeg  # encoded once per picture
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (270, 480, 3)
    out.stop()
    assert out.preview_jpeg() is None


# ---------------------------------------------------------------- the start-up check


def test_the_stderr_watch_notices_obs_rejecting_frames_and_passes_everything_on(capfd):
    watch = vcam.StderrWatch(vcam.OBS_REJECTION)
    assert watch.start()
    os.write(2, b"before ")
    os.write(2, b"Pixel buffer size ")  # split across writes, as a native library may do
    os.write(2, b"mismatch after\n")
    watch.stop()
    assert watch.found
    assert capfd.readouterr().err == "before Pixel buffer size mismatch after\n"
    clean = vcam.StderrWatch(vcam.OBS_REJECTION)
    assert clean.start()
    os.write(2, b"all good\n")
    clean.stop()
    assert not clean.found and capfd.readouterr().err == "all good\n"


def test_the_check_reports_a_rejection(cameras, monkeypatch):
    monkeypatch.setattr(vcam, "SELF_CHECK_S", 0.2)

    class Rejecting(FakeCamera):
        def send(self, picture):
            super().send(picture)
            os.write(2, vcam.OBS_REJECTION)

    out = vcam.VirtualCamera(FakeReceiver(), fps=50, open_camera=Rejecting, watch_stderr=True)
    out.start()
    wait_for(lambda: out.status()["check"] is not None)
    out.stop()
    status = out.status()
    assert status["check"].startswith("OBS rejected the 1920x1080 frames") and status["last_error"] == status["check"]


def test_the_check_without_a_rejection(cameras, monkeypatch):
    monkeypatch.setattr(vcam, "SELF_CHECK_S", 0.2)
    out = vcam.VirtualCamera(FakeReceiver(), fps=50, open_camera=FakeCamera, watch_stderr=True)
    out.start()
    wait_for(lambda: out.status()["check"] is not None)
    out.stop()
    assert out.status()["check"] == "OBS took 1920x1080 frames without complaint for 0.2 s" and out.status()["last_error"] is None
    assert threading.active_count() < 50
