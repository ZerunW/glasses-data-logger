"""Body measurements and the calibration endpoints: validation, files, and the same access rules as every action."""

import http.client
import json
import threading

import cv2
import numpy as np
import pytest

from glassesdata_receiver import bodyprofile, web
from glassesdata_receiver.link import TcpSource
from glassesdata_receiver.receiver import Receiver, preview_lag_s
from test_web import FakeSender, post, request, wait_until

PIPELINE_KEYS = ["hand_length_m", "palm_length_m", "palm_width_m", "forearm_length_m", "upper_arm_length_m",
                 "shoulder_width_m", "eye_to_shoulder_drop_m", "eye_to_shoulder_back_m"]


# ---------------------------------------------------------------- the profile module


def test_defaults_and_definitions_are_pipeline_section_4():
    profile = bodyprofile.defaults()
    assert list(bodyprofile.KEYS) == PIPELINE_KEYS
    assert profile == {"version": 1, "hand_length_m": 0.187, "palm_length_m": 0.107, "palm_width_m": 0.084, "forearm_length_m": 0.26,
                       "upper_arm_length_m": 0.32, "shoulder_width_m": 0.39, "eye_to_shoulder_drop_m": 0.20, "eye_to_shoulder_back_m": 0.07,
                       "measured": {key: False for key in PIPELINE_KEYS}}
    definitions = {field["key"]: field["definition"] for field in bodyprofile.fields()}
    assert definitions == {
        "hand_length_m": "Wrist crease to middle fingertip",
        "palm_length_m": "Wrist crease to the base crease of the middle finger",
        "palm_width_m": "Across the knuckles of index to little finger",
        "forearm_length_m": "Wrist crease to the tip of the elbow",
        "upper_arm_length_m": "Shoulder tip (acromion) to the tip of the elbow",
        "shoulder_width_m": "Between the two shoulder tips",
    }
    for field in bodyprofile.fields():
        assert field["min_m"] < field["default_m"] < field["max_m"]


def test_update_validates_and_marks_what_was_measured():
    profile, errors = bodyprofile.update(bodyprofile.defaults(), {"hand_length_m": 0.2, "forearm_length_m": 0.27})
    assert errors == {}
    assert profile["hand_length_m"] == 0.2 and profile["measured"]["hand_length_m"] is True
    assert profile["palm_width_m"] == 0.084 and profile["measured"]["palm_width_m"] is False  # untouched: still the default
    again, errors = bodyprofile.update(profile, {"hand_length_m": None})
    assert errors == {} and again["hand_length_m"] == 0.187 and again["measured"]["hand_length_m"] is False
    assert again["forearm_length_m"] == 0.27  # not named: kept
    for bad, key in (({"hand_length_m": 0.5}, "hand_length_m"), ({"hand_length_m": 19}, "hand_length_m"),
                     ({"palm_width_m": "8"}, "palm_width_m"), ({"palm_width_m": True}, "palm_width_m"),
                     ({"shoulder_width_m": float("nan")}, "shoulder_width_m"), ({"weight_kg": 70}, ""),
                     ({"eye_to_shoulder_drop_m": 0.2}, ""), ({"hand_length_m": 0.14, "palm_length_m": 0.13}, "palm_length_m")):
        _, errors = bodyprofile.update(bodyprofile.defaults(), bad)
        assert key in errors, bad
    assert "" in bodyprofile.update(bodyprofile.defaults(), [])[1] and "" in bodyprofile.update(bodyprofile.defaults(), None)[1]
    assert "between 13 and 25 cm" in bodyprofile.update(bodyprofile.defaults(), {"hand_length_m": 0.3})[1]["hand_length_m"]


def test_the_file_round_trips_and_a_broken_file_shows_the_defaults(tmp_path):
    path = tmp_path / "profile.json"
    assert bodyprofile.load(path) == (bodyprofile.defaults(), None)
    profile, _ = bodyprofile.update(bodyprofile.defaults(), {"palm_width_m": 0.09})
    bodyprofile.save(path, profile)
    data = json.loads(path.read_text())
    assert list(data) == ["version", *PIPELINE_KEYS, "measured"] and data["palm_width_m"] == 0.09
    assert data["measured"] == {**{key: False for key in PIPELINE_KEYS}, "palm_width_m": True}
    assert bodyprofile.load(path) == (profile, None)
    path.write_text("{oops")
    loaded, note = bodyprofile.load(path)
    assert loaded == bodyprofile.defaults() and "could not be read" in note


# ---------------------------------------------------------------- endpoints


@pytest.fixture
def server(tmp_path):
    hub = Receiver(tmp_path / "recordings", state_dir=tmp_path / "state")
    server = web.WebServer(hub, "127.0.0.1", 0)
    server.start()
    yield server, f"http://127.0.0.1:{server.port}", tmp_path / "state"
    server.stop()
    hub.shutdown()


def body(payload) -> bytes:
    return json.dumps(payload).encode()


def test_profile_endpoints(server):
    _, base, state = server
    data = json.loads(request(base + "/api/profile")[2])
    assert data["saved"] is False and data["profile"] == bodyprofile.defaults() and len(data["fields"]) == 6
    code, answer = post(base, "/api/profile", body({"hand_length_m": 0.195, "palm_length_m": 0.11}))
    assert code == 200 and answer["ok"] and answer["message"] == "Body measurements saved: 2 of 6 measured"
    assert answer["saved"] is True and answer["profile"]["measured"]["hand_length_m"] is True
    saved = json.loads((state / "profile.json").read_text())
    assert saved["hand_length_m"] == 0.195 and saved["upper_arm_length_m"] == 0.32
    for bad in (b"", b"[]", b"{nope", b'{"hand_length_m": 1}', b'{"hand_length_m": NaN}', b'{"height_m": 1.8}', b"\xff"):
        code, answer = post(base, "/api/profile", bad)
        assert code == 400 and answer["ok"] is False, bad
    code, answer = post(base, "/api/profile", body({"hand_length_m": 0.5}))
    assert answer["errors"] == {"hand_length_m": "Hand length: between 13 and 25 cm"} and answer["message"].startswith("Not saved: ")
    assert json.loads((state / "profile.json").read_text()) == saved  # a refused change leaves the file alone


def test_calibration_endpoints_without_video(server):
    _, base, state = server
    data = json.loads(request(base + "/api/calibration")[2])
    assert data["current"] is None and data["session"] is None and data["screen"] is None and data["card_mm"] == [85.6, 53.98]
    for path in ("/api/calibration/board.png", "/api/calibration/preview.jpg", "/api/calibration/undistorted.jpg"):
        assert request(base + path)[0] == 404
    assert post(base, "/api/calibration/solve")[0] == 409
    assert post(base, "/api/calibration/save")[0] == 409
    assert post(base, "/api/calibration/capture", body({"active": True}))[0] == 409
    code, answer = post(base, "/api/calibration/screen", body({"css_px_per_mm": 4.41, "device_pixel_ratio": 2, "screen_css": [1512, 982]}))
    assert code == 200 and json.loads((state / "calibration" / "screen.json").read_text())["css_px_per_mm"] == 4.41
    for bad in (b"", b"{}", b'{"css_px_per_mm": 0.1, "device_pixel_ratio": 2}', b'{"css_px_per_mm": 4, "device_pixel_ratio": 2, "x": 1}',
                b'{"css_px_per_mm": "4", "device_pixel_ratio": 2}', b'{"css_px_per_mm": 4, "device_pixel_ratio": 2, "screen_css": [1]}'):
        assert post(base, "/api/calibration/screen", bad)[0] == 400, bad
    for bad in (b'{"css_px_per_mm": 4.4, "device_pixel_ratio": 2}', b'{"css_px_per_mm": 4.4, "device_pixel_ratio": 2, "area_css": [10, 10]}'):
        assert post(base, "/api/calibration/start", bad)[0] == 400
    code, answer = post(base, "/api/calibration/start", body({"css_px_per_mm": 4.4, "device_pixel_ratio": 2, "area_css": [1240, 940]}))
    assert code == 200 and answer["kept"] is False and answer["board"]["squares_x"] == 8
    status, headers, png = request(base + "/api/calibration/board.png")
    assert status == 200 and headers["Content-Type"] == "image/png"
    assert cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE).shape == (answer["board"]["height_px"], answer["board"]["width_px"])
    session = json.loads(request(base + "/api/calibration")[2])["session"]
    assert session["capturing"] is True and session["samples"] == 0 and session["guidance"] == "No video from the glasses"
    # the same board again resumes; another board starts over
    assert post(base, "/api/calibration/start", body({"css_px_per_mm": 4.4, "device_pixel_ratio": 2, "area_css": [1240, 940]}))[1]["kept"] is True
    assert post(base, "/api/calibration/capture", body({"active": False}))[0] == 200
    assert json.loads(request(base + "/api/calibration")[2])["session"]["capturing"] is False
    assert post(base, "/api/calibration/capture", body({"active": "yes"}))[0] == 400
    code, answer = post(base, "/api/calibration/solve")
    assert code == 409 and "No video" in answer["message"]
    assert post(base, "/api/calibration/reset")[0] == 200
    assert json.loads(request(base + "/api/calibration")[2])["session"] is None
    assert post(base, "/api/calibration/nothing")[0] == 404


def test_setup_actions_are_protected_like_the_others(server):
    app, base, state = server
    actions = ["/api/profile", "/api/calibration/screen", "/api/calibration/start", "/api/calibration/capture",
               "/api/calibration/solve", "/api/calibration/save", "/api/calibration/reset"]
    payload = body({"hand_length_m": 0.2})
    for path in actions:
        assert request(base + path, "POST")[0] == 403, path  # no X-GlassesData header
        assert post(base, path, payload, Origin="http://evil.example")[0] == 403, path
        assert post(base, path, payload, Origin="null")[0] == 403, path
        assert post(base, path, payload, **{"Sec-Fetch-Site": "cross-site"})[0] == 403, path
    assert not (state / "profile.json").exists()
    assert post(base, "/api/profile", payload, Origin=base, **{"Sec-Fetch-Site": "same-origin"})[0] == 200
    for path in ("/api/calibration", "/api/profile", "/api/calibration/board.png", "/api/calibration/preview.jpg"):
        connection = http.client.HTTPConnection("127.0.0.1", app.port, timeout=5)  # DNS rebinding
        connection.request("GET", path, headers={"Host": f"attacker.example:{app.port}"})
        assert connection.getresponse().status == 403, path
        connection.close()
    connection = http.client.HTTPConnection("127.0.0.1", app.port, timeout=5)
    connection.request("POST", "/api/profile", body=payload, headers={"Host": f"attacker.example:{app.port}", "X-GlassesData": "1"})
    assert connection.getresponse().status == 403
    connection.close()


def test_setup_endpoints_need_the_key_off_loopback(tmp_path):
    hub = Receiver(tmp_path / "recordings", state_dir=tmp_path / "state")
    server = web.WebServer(hub, "127.0.0.1", 0, require_key=True)
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        for path in ("/api/calibration", "/api/profile", "/api/calibration/board.png", "/api/calibration/preview.jpg",
                     "/api/calibration/undistorted.jpg"):
            assert request(base + path)[0] == 403, path
        for path in ("/api/profile", "/api/calibration/screen", "/api/calibration/start", "/api/calibration/solve",
                     "/api/calibration/save", "/api/calibration/reset", "/api/calibration/capture"):
            assert request(base + path, "POST", {"X-GlassesData": "1"})[0] == 403, path
        cookie = f"{server.cookie_name}={server.key}"
        assert request(base + "/api/profile", headers={"Cookie": cookie})[0] == 200
        assert post(base, "/api/profile", body({"hand_length_m": 0.2}), Cookie=cookie)[0] == 200
        assert post(base, "/api/profile", body({"hand_length_m": 0.2}), Cookie=cookie, Origin="http://evil.example")[0] == 403
    finally:
        server.stop()
        hub.shutdown()


def test_without_a_state_folder_nothing_is_saved(tmp_path):
    hub = Receiver(tmp_path / "recordings")
    server = web.WebServer(hub, "127.0.0.1", 0)
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        assert json.loads(request(base + "/api/profile")[2])["file"] is None
        assert post(base, "/api/profile", body({"hand_length_m": 0.2}))[0] == 409
        assert post(base, "/api/calibration/screen", body({"css_px_per_mm": 4, "device_pixel_ratio": 2}))[0] == 409
        assert list(tmp_path.iterdir()) == [] or [p.name for p in tmp_path.iterdir()] == ["recordings"]
    finally:
        server.stop()
        hub.shutdown()


# ---------------------------------------------------------------- preview lag


def test_preview_lag_maps_the_phone_clock_to_this_one():
    offset = 123_456_000_000  # phone clock minus Mac clock
    captured_mac = 10_000_000_000
    assert preview_lag_s(captured_mac + offset, offset, captured_mac + 1_250_000_000) == pytest.approx(1.25)
    assert preview_lag_s(captured_mac + offset, offset, captured_mac - 5) == 0.0  # clock noise never goes negative
    assert preview_lag_s(None, offset, 0) is None and preview_lag_s(5, None, 0) is None


def test_a_phone_that_queues_video_shows_as_a_late_preview(tmp_path):
    sender = FakeSender("--loop", "--seconds", "30", "--lag-ms", "1500")
    hub = Receiver(tmp_path / "recordings")
    stop = threading.Event()
    source = TcpSource(hub, "127.0.0.1", sender.port, stop)
    source.start()
    try:
        wait_until(lambda: web.status_payload(hub, {})["video"]["preview_lag_s"] is not None, timeout=10)
        wait_until(lambda: web.status_payload(hub, {})["video"]["phone_queue_frames"] is not None, timeout=5)
        video = web.status_payload(hub, {})["video"]
        assert video["preview_lag_s"] == pytest.approx(1.5, abs=0.3)
        assert video["phone_queue_frames"] == 45 and video["phone_record_priority"] is False
        # the phone buffers instead of dropping while this computer records
        assert hub.start_recording() is not None
        wait_until(lambda: web.status_payload(hub, {})["video"]["phone_record_priority"] is True, timeout=5)
        hub.stop_recording()
    finally:
        stop.set()
        hub.shutdown()
        source.join(3)
        sender.stop()


def test_stats_fields_from_an_older_phone_are_null():
    assert web._int_or_none(True) is None and web._int_or_none(-1) is None and web._int_or_none("3") is None
    assert web._int_or_none(7) == 7


def test_the_page_has_the_setup_views_and_the_lag_indicator(server):
    _, base, _ = server
    page = request(base + "/")[2].decode()
    for text in (">Camera calibration</h2>", ">Body measurements</h2>", 'id="lag-pill"', "/api/calibration/start", "/api/calibration/solve",
                 "/api/calibration/save", "/api/calibration/screen", "/api/calibration/capture", "/api/calibration/board.png",
                 "/api/calibration/preview.jpg", "/api/calibration/undistorted.jpg", "/api/profile", "requestFullscreen",
                 "image-rendering: pixelated", "localStorage", "devicePixelRatio", "preview_lag_s", "phone_queue_frames",
                 "phone_record_priority", "phone_gaps", 'id="board-view"', 'id="hud"', 'id="body-svg"'):
        assert text in page, text
    for key in PIPELINE_KEYS[:6]:  # every measurement has its place in the drawing
        assert f'data-m="{key}"' in page


# ---------------------------------------------------------------- review fixes


def test_a_malformed_current_calibration_does_not_break_the_page(server):
    _, base, state = server
    (state / "calibration").mkdir(parents=True)
    for text in ("{}", '{"K": []}', "[]", "{oops"):
        (state / "calibration" / "current.json").write_text(text)
        status, _, body = request(base + "/api/calibration")
        data = json.loads(body)
        assert status == 200 and data["current"] is None and "not used" in data["current_note"], text


def test_a_board_beyond_the_caps_is_refused(server):
    _, base, _ = server
    code, answer = post(base, "/api/calibration/start", body({"css_px_per_mm": 13, "device_pixel_ratio": 2, "area_css": [1240, 940]}))
    assert code == 400 and "device pixels per mm" in answer["message"]
    code, answer = post(base, "/api/calibration/start", body({"css_px_per_mm": 12, "device_pixel_ratio": 2, "area_css": [20000, 20000]}))
    assert code == 400 and "at most" in answer["message"]


def test_an_unexpected_error_answers_json_500(server, monkeypatch):
    app, base, _ = server

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(app.calibration, "state", broken)
    status, headers, data = request(base + "/api/calibration")
    assert status == 500 and headers["Content-Type"].startswith("application/json")
    assert json.loads(data) == {"ok": False, "message": "Internal error (RuntimeError): see the receiver's log"}
    monkeypatch.setattr(app.receiver, "stop_recording", broken)
    code, answer = post(base, "/api/record/stop")
    assert code == 500 and answer["ok"] is False
    assert json.loads(request(base + "/api/status")[2])["link"]["connected"] is False  # the server carries on


class FailingRecording:
    def __init__(self, directory):
        self.directory, self.duration_s, self.frames_written = directory, 1.0, 10

    def close(self):
        raise OSError(28, "No space left on device")


def test_a_recording_that_fails_to_close_is_not_reported_as_saved(server, tmp_path, caplog):
    app, base, _ = server
    folder = tmp_path / "recordings" / "2026-01-01_000000"
    folder.mkdir(parents=True)
    app.receiver._recording = FailingRecording(folder)
    with caplog.at_level("INFO"):
        code, answer = post(base, "/api/record/stop")
    assert code == 500 and answer["ok"] is False and "not closed cleanly" in answer["message"] and "No space" in answer["message"]
    assert not any(record.getMessage().startswith("Saved ") for record in caplog.records)
    assert app.receiver.close_error(folder) == "[Errno 28] No space left on device"
    app.receiver._recording = FailingRecording(folder)
    assert app.receiver.toggle_recording().startswith(f"Recording {folder} did not close cleanly")


def test_ignored_stored_measurements_show_a_note(server):
    _, base, state = server
    state.mkdir(parents=True, exist_ok=True)
    (state / "profile.json").write_text(json.dumps({"version": 1, "hand_length_m": 2.0, "measured": {"hand_length_m": True}}))
    data = json.loads(request(base + "/api/profile")[2])
    assert data["profile"]["measured"]["hand_length_m"] is False and data["profile"]["hand_length_m"] == 0.187
    assert "hand length" in data["note"]
