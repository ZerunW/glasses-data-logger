"""Camera calibration: board design, view selection, the solver, files, and a run through the HTTP API."""

import json
import math
import threading
import time

import cv2
import numpy as np
import pytest

import charuco_synth as synth
from glassesdata_receiver import calibration as cal
from glassesdata_receiver.decoder import DecodedFrame

LAPTOP = (4.4, 2.0, (1240.0, 940.0))  # CSS px per mm, device pixel ratio, area for the board in CSS px
SIZE = (720, 1280)  # the glasses' portrait video
TRUE_K = np.array([[600.0, 0.0, 361.5], [0.0, 603.0, 636.0], [0.0, 0.0, 1.0]])
TRUE_DIST = np.array([-0.06, 0.025, 0.0005, -0.0003, -0.004])


@pytest.fixture(scope="module")
def spec() -> cal.BoardSpec:
    return cal.design_board(*LAPTOP)


@pytest.fixture(scope="module")
def camera() -> synth.SyntheticCamera:
    return synth.SyntheticCamera(TRUE_K, TRUE_DIST, SIZE)


def rendered_samples(spec, camera, count, seed, **pose_options):
    rng = np.random.default_rng(seed)
    detector = cal.make_detector(spec)
    samples = []
    for rvec, tvec in synth.poses(spec, camera, count, rng, **pose_options):
        image = camera.render(spec, rvec, tvec, rng)
        corners, ids, _ = cal.detect(detector, image)
        if len(ids) >= cal.MIN_CORNERS:
            samples.append(cal.Sample(corners, ids, None, 0.0, cal._jpeg(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 90)))
    return samples


def relative_errors(K, truth):
    return [abs(K[0, 0] / truth[0, 0] - 1), abs(K[1, 1] / truth[1, 1] - 1), abs(K[0, 2] / truth[0, 2] - 1), abs(K[1, 2] / truth[1, 2] - 1)]


# ---------------------------------------------------------------- the board


@pytest.mark.parametrize("scale, ratio, area", [LAPTOP, (3.6, 1.0, (2200.0, 1300.0)), (5.0, 3.0, (1100.0, 700.0)), (3.78, 1.25, (1500.0, 800.0))])
def test_the_board_fills_the_screen_with_sharp_squares_of_known_size(scale, ratio, area):
    spec = cal.design_board(scale, ratio, area)
    square_mm, marker_mm = spec.square_m * 1000, spec.marker_m * 1000
    assert cal.MIN_SQUARE_MM <= square_mm <= cal.MAX_SQUARE_MM
    # one image pixel per device pixel: the physical size follows from whole pixels
    assert spec.square_px == pytest.approx(square_mm * scale * ratio, abs=1e-6)
    assert spec.marker_px % cal.MARKER_MODULES == 0 and 0.6 <= spec.marker_px / spec.square_px <= 0.82
    area_mm = (area[0] / scale, area[1] / scale)
    fill = max(spec.squares_x * square_mm / area_mm[0], spec.squares_y * square_mm / area_mm[1])
    assert 0.8 < fill <= 1.0  # it fills the area in at least one direction, and fits
    assert spec.squares_x * square_mm <= area_mm[0] and spec.squares_y * square_mm <= area_mm[1]
    image = spec.image()
    assert image.shape == (spec.height_px, spec.width_px) and set(np.unique(image)) == {0, 255}  # crisp: no grey
    flat = cv2.copyMakeBorder(image, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    corners, ids, _ = cal.detect(cal.make_detector(spec), flat)
    assert len(ids) == (spec.squares_x - 1) * (spec.squares_y - 1)
    # the drawing matches the model: corner k of the board sits at its object point
    objects = np.asarray(spec.board().getChessboardCorners())[ids][:, :2]
    expected = objects / spec.square_m * spec.square_px - 0.5 + 40
    assert np.abs(corners - expected).max() < 0.3
    png = cv2.imdecode(np.frombuffer(spec.png(), np.uint8), cv2.IMREAD_UNCHANGED)
    assert png.shape == image.shape


def test_board_json_is_the_pipeline_format(spec):
    board = spec.to_json()
    assert set(board) == {"dictionary", "squares_x", "squares_y", "square_m", "marker_m", "css_px_per_mm"}
    assert board["dictionary"] == "DICT_5X5_100" and board["css_px_per_mm"] == 4.4
    described = spec.describe()
    assert described["css_width"] * spec.device_pixel_ratio == spec.width_px


@pytest.mark.parametrize("args", [(float("nan"), 2, (1000, 800)), (0, 2, (1000, 800)), (True, 2, (1000, 800)), (4, 2, (50, 800)),
                                  (4, 0, (1000, 800)), (4, 2, (1000, float("inf")))])
def test_board_design_refuses_nonsense(args):
    with pytest.raises(ValueError):
        cal.design_board(*args)


# ---------------------------------------------------------------- choosing views


def projected(spec, rvec, tvec, K=TRUE_K, dist=TRUE_DIST, size=SIZE):
    objects = np.asarray(spec.board().getChessboardCorners(), dtype=np.float64)
    image, _ = cv2.projectPoints(objects, rvec, tvec, K, dist)
    image = image.reshape(-1, 2).astype(np.float32)
    inside = (image[:, 0] >= 0) & (image[:, 1] >= 0) & (image[:, 0] < size[0]) & (image[:, 1] < size[1])
    return image[inside], np.flatnonzero(inside).astype(np.int32)


def board_centre(spec):
    return np.array([spec.squares_x * spec.square_m / 2, spec.squares_y * spec.square_m / 2, 0.0])


def test_pose_from_the_provisional_camera(spec):
    centre = board_centre(spec)
    K = cal.guess_camera(*SIZE)
    rvec, tvec = synth.look_at(centre + np.array([0, 0, -0.4]), centre)
    corners, ids = projected(spec, rvec, tvec, K, np.zeros(5))
    objects = np.asarray(spec.board().getChessboardCorners(), dtype=np.float64)[ids]
    pose = cal.estimate_pose(objects, corners, K, np.zeros(0), (spec.squares_x * spec.square_m, spec.squares_y * spec.square_m), SIZE[0])
    assert pose.distance_m == pytest.approx(0.4, rel=1e-3) and pose.tilt_deg < 1 and pose.side == "" and pose.key[1:] == (0, "")
    tilt = math.radians(35)
    rvec, tvec = synth.look_at(centre + 0.45 * np.array([-math.sin(tilt), 0, -math.cos(tilt)]), centre)  # from the left
    corners, ids = projected(spec, rvec, tvec, K, np.zeros(5))
    objects = np.asarray(spec.board().getChessboardCorners(), dtype=np.float64)[ids]
    pose = cal.estimate_pose(objects, corners, K, np.zeros(0), (spec.squares_x * spec.square_m, spec.squares_y * spec.square_m), SIZE[0])
    assert pose.tilt_deg == pytest.approx(35, abs=0.5) and pose.side == "left" and pose.tilt_bin == 2
    assert pose.distance_bin == "far"  # 0.45 m is over twice the distance at which this board spans the image width


def test_the_selector_keeps_new_views_and_refuses_repeats_blur_and_partial_boards(spec):
    K, dist = cal.guess_camera(*SIZE), np.zeros(0)
    selector = cal.SampleSelector(spec, SIZE[0], SIZE[1], K, dist)
    centre = board_centre(spec)
    rvec, tvec = synth.look_at(centre + np.array([0, 0, -0.35]), centre)
    corners, ids = projected(spec, rvec, tvec, K, np.zeros(5))
    look = selector.consider(corners, ids, 10.0)
    assert not look.accepted and look.reason == "moving"  # the first look has nothing to compare with
    look = selector.consider(corners, ids, 10.2)
    assert look.accepted and look.steady and len(selector.samples) == 1 and look.corners == len(ids)
    first_coverage = selector.coverage
    assert selector.consider(corners, ids, 10.4).reason == "just kept one"
    assert selector.consider(corners + 3, ids, 11.0).reason == "already have this view"  # steady, but nothing new
    assert selector.consider(corners + 60, ids, 11.2).reason == "moving"  # moved too far since the last look: blur
    few = selector.consider(corners[:8], ids[:8], 11.4)
    assert few.reason == "too few corners" and not few.accepted
    # the board in another part of the image: new cells
    other_r, other_t = synth.aim(rvec, tvec, K, (150, 1050))
    corners2, ids2 = projected(spec, other_r, other_t, K, np.zeros(5))
    selector.consider(corners2, ids2, 12.0)
    look = selector.consider(corners2, ids2, 12.2)
    assert look.accepted and look.reason.startswith("kept: ") and len(selector.samples) == 2
    assert 0 < first_coverage < selector.coverage  # cells count the views that covered them
    assert selector.cells.sum() == len(selector._cells_of(corners)) + len(selector._cells_of(corners2))
    # a frame with an id outside the board is simply out of range for nothing; empty frames are no board
    assert selector.consider(np.zeros((0, 2), np.float32), np.zeros(0, np.int32), 13.0).reason == "no board"


def test_the_selector_stops_at_the_maximum(spec, monkeypatch):
    monkeypatch.setattr(cal, "MAX_SAMPLES", 2)
    K = cal.guess_camera(*SIZE)
    selector = cal.SampleSelector(spec, SIZE[0], SIZE[1], K, np.zeros(0))
    centre = board_centre(spec)
    base = synth.look_at(centre + np.array([0, 0, -0.35]), centre)
    now = 0.0
    for target in ((360, 300), (360, 1000), (200, 640)):
        corners, ids = projected(spec, *synth.aim(*base, K, target), K, np.zeros(5))
        selector.consider(corners, ids, now)
        selector.consider(corners, ids, now + 0.2)
        now += 1.0
    assert len(selector.samples) == 2
    assert cal.guidance(cal.Look(corners=30, steady=True), selector, True) == "Enough views: press Solve"


def test_guidance_says_what_to_do_next(spec):
    K = cal.guess_camera(*SIZE)
    selector = cal.SampleSelector(spec, SIZE[0], SIZE[1], K, np.zeros(0))
    assert cal.guidance(None, None, False) == "No video from the glasses"
    assert cal.guidance(cal.Look(corners=0), selector, True) == "Look at the board"
    assert cal.guidance(cal.Look(corners=0, markers=3), selector, True) == "Show more of the board"
    assert cal.guidance(cal.Look(corners=6, near_edge=True), selector, True) == "Show the whole board"
    assert cal.guidance(cal.Look(corners=6), selector, True) == "Move closer"
    assert cal.guidance(cal.Look(corners=30, steady=False), selector, True) == "Hold still"
    assert cal.guidance(cal.Look(corners=30, steady=True), selector, True).startswith("Next: board in the ")
    selector.cells[:] = 1  # every part of the image covered: distance and angle come next
    assert cal.guidance(cal.Look(corners=30, steady=True), selector, True) == "Next: move closer"
    selector.bins[("near", 0, "")] = 1
    selector.bins[("far", 0, "")] = 1
    assert cal.guidance(cal.Look(corners=30, steady=True), selector, True) == "Next: look at the board from the left"


# ---------------------------------------------------------------- solving


def test_the_solver_recovers_a_known_camera(spec, camera):
    samples = rendered_samples(spec, camera, 30, seed=3)
    assert len(samples) >= 25
    solution = cal.solve(spec, samples, SIZE)
    assert solution.chosen.model == "pinhole"
    assert max(relative_errors(solution.chosen.K, TRUE_K)) < 0.01  # fx, fy, cx, cy within 1 %
    assert solution.chosen.rms_px < 0.5
    assert np.allclose(solution.chosen.dist[:2], TRUE_DIST[:2], atol=0.01)
    described = solution.describe()
    assert described["views"] == len(samples) and len(described["per_view_px"]) == len(samples)
    assert described["held_out_px"]["pinhole"] < 0.5
    fov = described["fov_deg"]
    assert fov["horizontal"] == pytest.approx(2 * math.degrees(math.atan(360 / 600)), abs=1.5)  # distortion widens it a little
    picture = cv2.imdecode(np.frombuffer(solution.preview, np.uint8), cv2.IMREAD_COLOR)
    assert picture.shape[0] == cal.UNDISTORTED_HEIGHT and picture.shape[1] > picture.shape[0]  # captured | undistorted


def test_the_solver_chooses_fisheye_for_a_fisheye_lens():
    K = np.array([[420.0, 0.0, 359.5], [0.0, 421.0, 639.5], [0.0, 0.0, 1.0]])  # nearly equidistant: 98 degrees across
    D = np.array([0.03, -0.01, 0.002, 0.0])
    fisheye = synth.SyntheticCamera(K, D, SIZE, model="fisheye")
    spec = cal.design_board(*LAPTOP)
    samples = rendered_samples(spec, fisheye, 30, seed=5, distance=(0.18, 0.4))
    assert len(samples) >= 20
    solution = cal.solve(spec, samples, SIZE)
    assert solution.chosen.model == "fisheye", solution.why
    assert solution.held_out_px["fisheye"] < cal.FISHEYE_MARGIN * solution.held_out_px["pinhole"]
    assert max(relative_errors(solution.chosen.K, K)) < 0.01 and solution.chosen.rms_px < 0.5
    calibration = solution.calibration()
    assert calibration["model"] == "fisheye" and len(calibration["dist"]) == 4


def test_the_solver_drops_bad_views_and_refuses_too_few(spec, camera):
    samples = rendered_samples(spec, camera, 16, seed=11)
    rng = np.random.default_rng(0)
    bad = cal.Sample(samples[0].corners + rng.normal(0, 6, samples[0].corners.shape).astype(np.float32), samples[0].ids, None, 0.0)
    samples.append(bad)
    solution = cal.solve(spec, samples, SIZE)
    assert len(samples) - 1 in solution.describe()["dropped"]
    assert solution.describe()["per_view_px"][-1] is None
    assert max(relative_errors(solution.chosen.K, TRUE_K)) < 0.01
    with pytest.raises(ValueError, match="at least 6 views"):
        cal.solve(spec, samples[:5], SIZE)


def test_calibration_json_is_exactly_the_pipeline_format(spec, camera, tmp_path):
    solution = cal.solve(spec, rendered_samples(spec, camera, 12, seed=2), SIZE)
    data = solution.calibration()
    assert set(data) == {"version", "model", "width", "height", "K", "dist", "rms_px", "created", "device", "source", "board"}
    assert (data["version"], data["width"], data["height"], data["source"], data["device"]) == (1, 720, 1280, "charuco-web", "Ray-Ban Meta")
    assert len(data["K"]) == 3 and data["K"][2] == [0.0, 0.0, 1.0] and len(data["dist"]) == 5
    assert data["board"] == spec.to_json()
    assert time.strptime(data["created"], "%Y-%m-%dT%H:%M:%SZ")
    store = cal.CalibrationStore(tmp_path)
    assert store.current() is None
    written = store.save(data)
    assert written.parent == tmp_path / "calibration" and written.name.startswith("calibration_") and written.name.endswith("Z.json")
    assert json.loads(written.read_text()) == json.loads((tmp_path / "calibration" / "current.json").read_text()) == data
    assert store.current() == data
    json.dumps(data, allow_nan=False)  # plain JSON


# ---------------------------------------------------------------- the capture thread


class StubReceiver:
    """Serves rendered frames like Receiver.wait_frame: always the newest, a new one every call."""

    def __init__(self, images):
        self.images = images
        self.busy = True
        self.calls = 0
        self.index = 0
        self.lock = threading.Lock()

    def wait_frame(self, after_index, timeout):
        with self.lock:
            self.calls += 1
            self.index += 1
            image = self.images[(self.index // 3) % len(self.images)]  # each view for three frames
            return DecodedFrame(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), self.index * 33_000, self.index, self.index)


def test_the_capture_thread_looks_at_most_five_times_a_second_and_keeps_views(spec, camera):
    rng = np.random.default_rng(9)
    images = [camera.render(spec, r, t, rng) for r, t in synth.poses(spec, camera, 6, rng)]
    receiver = StubReceiver(images)
    session = cal.CaptureSession(receiver, spec)
    session.start()
    try:
        time.sleep(2.1)
        status = session.status()
    finally:
        session.close()
    assert receiver.calls <= 12  # 5 Hz at most (2.1 s), never a busy loop on the frames
    assert status["sequence"] >= 8 and status["samples"] >= 2
    assert status["video"] == {"ok": True, "width": 720, "height": 1280}
    assert status["last"]["corners"] >= cal.MIN_CORNERS and status["coverage"]["fraction"] > 0
    assert cv2.imdecode(np.frombuffer(session.preview_jpeg(), np.uint8), cv2.IMREAD_COLOR).shape[0] == cal.PREVIEW_HEIGHT


def test_a_paused_capture_looks_at_nothing(spec, camera):
    receiver = StubReceiver([camera.render(spec, *synth.poses(spec, camera, 1, np.random.default_rng(1))[0])])
    session = cal.CaptureSession(receiver, spec)
    session.start()
    session.set_capturing(False)
    time.sleep(0.3)
    calls = receiver.calls
    time.sleep(0.6)
    assert receiver.calls == calls and session.status()["guidance"] == "Paused"
    session.close()


# ---------------------------------------------------------------- end to end: the CLI, a streamed board video, the HTTP API


def test_end_to_end_calibration_over_http_and_a_recording_that_carries_both_files(tmp_path, spec, camera):
    """Synthetic views of the board the receiver designs for a laptop screen, encoded as HEVC and
    streamed by tools/fake_sender.py into ``python -m glassesdata_receiver --tcp``; the calibration
    is driven through the HTTP API like the page does, solved, saved and compared with the true
    camera. Then a 3 s recording must hold calibration.json and profile.json."""
    import os
    import re
    import signal
    import subprocess
    import sys
    import urllib.request

    from test_web import FAKE_SENDER, RECEIVER_DIR, wait_until

    video = tmp_path / "board.mp4"
    synth.calibration_video(video, spec, camera, 32, np.random.default_rng(21), hold_frames=16)
    sender = subprocess.Popen(
        [sys.executable, "-u", str(FAKE_SENDER), str(video), "--listen", "0", "--loop", "--seconds", "150"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=RECEIVER_DIR,
    )
    port = None
    for line in sender.stdout:
        match = re.search(r"waiting for the receiver on 127\.0\.0\.1:(\d+)", line)
        if match:
            port = int(match.group(1))
            break
    assert port
    threading.Thread(target=sender.stdout.read, daemon=True).start()  # keep draining its output
    state, out = tmp_path / "state", tmp_path / "recordings"
    receiver = subprocess.Popen(
        [sys.executable, "-m", "glassesdata_receiver", "--tcp", f"127.0.0.1:{port}", "--no-hands", "--web-port", "0",
         "--out", str(out), "--state-dir", str(state), "--duration", "150"],
        cwd=RECEIVER_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={**os.environ},
    )
    threading.Thread(target=receiver.stderr.read, daemon=True).start()

    def call(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        method = "GET" if payload is None and not path.startswith("POST ") else "POST"
        path = path.removeprefix("POST ")
        request = urllib.request.Request(base + path, data=data if method == "POST" else None, method=method,
                                         headers={"X-GlassesData": "1", "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
            return json.loads(body) if response.headers["Content-Type"].startswith("application/json") else body

    try:
        line = receiver.stdout.readline()
        assert line.startswith("WEB_URL="), line
        base = line.strip().removeprefix("WEB_URL=").rstrip("/")
        wait_until(lambda: call("/api/status")["video"]["has_frame"], timeout=15)
        assert call("POST /api/profile", {"hand_length_m": 0.2, "shoulder_width_m": 0.41})["ok"]
        assert call("POST /api/calibration/screen", {"css_px_per_mm": LAPTOP[0], "device_pixel_ratio": LAPTOP[1], "screen_css": [1512, 982]})["ok"]
        started = call("POST /api/calibration/start", {"css_px_per_mm": LAPTOP[0], "device_pixel_ratio": LAPTOP[1], "area_css": list(LAPTOP[2])})
        assert started["ok"] and started["board"] == spec.describe()  # the same board the video shows
        board = cv2.imdecode(np.frombuffer(call("/api/calibration/board.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(board, spec.image())
        began = time.monotonic()
        wait_until(lambda: call("/api/calibration")["session"]["samples"] >= cal.TARGET_SAMPLES, timeout=60)
        capture_s = time.monotonic() - began
        session = call("/api/calibration")["session"]
        assert session["video"] == {"ok": True, "width": 720, "height": 1280} and session["coverage"]["fraction"] > 0.5
        assert call("/api/calibration/preview.jpg")[:2] == b"\xff\xd8"
        assert call("POST /api/calibration/capture", {"active": False})["ok"]
        solved = call("POST /api/calibration/solve", {})
        assert solved["ok"], solved
        saved = call("POST /api/calibration/save", {})
        assert saved["ok"]
        current = json.loads((state / "calibration" / "current.json").read_text())
        K = np.array(current["K"])
        errors = relative_errors(K, TRUE_K)
        print(f"\nE2E calibration: {session['samples']} views kept in {capture_s:.1f} s, model {current['model']}, rms {current['rms_px']:.3f} px")
        print(f"  true      fx {TRUE_K[0, 0]:.2f} fy {TRUE_K[1, 1]:.2f} cx {TRUE_K[0, 2]:.2f} cy {TRUE_K[1, 2]:.2f}")
        print(f"  recovered fx {K[0, 0]:.2f} fy {K[1, 1]:.2f} cx {K[0, 2]:.2f} cy {K[1, 2]:.2f}  (max error {max(errors) * 100:.3f} %)")
        print(f"  dist true {TRUE_DIST.round(4).tolist()} recovered {np.round(current['dist'], 4).tolist()}")
        print(f"  held out {solved['solution']['held_out_px']}, why: {solved['solution']['why']}")
        assert current["model"] == "pinhole" and max(errors) < 0.01 and current["rms_px"] < 0.5
        assert current["source"] == "charuco-web" and current["board"] == spec.to_json()
        state_json = call("/api/calibration")
        assert state_json["current"]["fx"] == round(K[0, 0], 2) and state_json["screen"]["css_px_per_mm"] == LAPTOP[0]
        assert call("/api/calibration/undistorted.jpg")[:2] == b"\xff\xd8"

        name = call("POST /api/record/start", {})["name"]
        time.sleep(3)
        assert call("POST /api/record/stop", {})["name"] == name
        folder = out / name
        assert json.loads((folder / "calibration.json").read_text()) == current
        profile = json.loads((folder / "profile.json").read_text())
        assert profile["hand_length_m"] == 0.2 and profile["measured"]["hand_length_m"] is True
        assert profile["measured"]["palm_width_m"] is False and profile["palm_width_m"] == 0.084
        info = json.loads((folder / "info.json").read_text())
        assert info["calibration"]["file"] == "calibration.json" and "note" not in info["calibration"]
        assert info["profile"] == {"file": "profile.json", "measured": ["hand_length_m", "shoulder_width_m"]}
        assert info["framesWritten"] > 60
    finally:
        if receiver.poll() is None:
            receiver.send_signal(signal.SIGINT)
            try:
                receiver.wait(15)
            except subprocess.TimeoutExpired:
                receiver.kill()
        if sender.poll() is None:
            sender.terminate()
            sender.wait(5)
    assert receiver.returncode == 0
