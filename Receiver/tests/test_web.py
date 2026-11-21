"""The web interface, against a live receiver fed by tools/fake_sender.py (ports are picked by the OS)."""

import http.client
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from conftest import PLANT
from glassesdata_receiver import vcam, viewer, web
from glassesdata_receiver.decoder import DecodedFrame
from glassesdata_receiver.link import TcpSource
from glassesdata_receiver.receiver import HandsResult, Receiver

RECEIVER_DIR = Path(__file__).resolve().parents[1]
FAKE_SENDER = RECEIVER_DIR / "tools" / "fake_sender.py"


class FakeSender:
    """tools/fake_sender.py in a subprocess, listening on a free port; its output is collected."""

    def __init__(self, *options: str) -> None:
        if not PLANT.exists():
            pytest.skip(f"{PLANT} is missing")
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(FAKE_SENDER), str(PLANT), "--listen", "0", *options],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=RECEIVER_DIR,
        )
        self.lines: list[str] = []
        self._port: int | None = None
        self._ready = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()
        assert self._ready.wait(20), f"fake sender did not start: {self.lines}"
        assert self._port, f"fake sender did not listen: {self.lines}"
        self.port: int = self._port

    def _read(self) -> None:
        for line in self.process.stdout:
            self.lines.append(line.rstrip("\n"))
            match = re.search(r"waiting for the receiver on 127\.0\.0\.1:(\d+)", line)
            if match:
                self._port = int(match.group(1))
                self._ready.set()
        self._ready.set()

    def wait(self, timeout: float = 20.0) -> int:
        return self.process.wait(timeout)

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(5)


def wait_until(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.02)


class Live:
    def __init__(self, root: Path) -> None:
        self.sender = FakeSender("--loop", "--seconds", "120")
        self.receiver = Receiver(root / "recordings")
        self.stop = threading.Event()
        self.source = TcpSource(self.receiver, "127.0.0.1", self.sender.port, self.stop)
        self.source.start()
        self.server = web.WebServer(self.receiver, "127.0.0.1", 0)
        self.server.start()
        self.base = f"http://127.0.0.1:{self.server.port}"
        wait_until(lambda: web.status_payload(self.receiver, {})["video"]["has_frame"], timeout=10)

    def close(self) -> None:
        self.server.stop()
        self.stop.set()
        self.receiver.shutdown()
        self.source.join(3)
        self.sender.stop()


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    running = Live(tmp_path_factory.mktemp("web"))
    yield running
    running.close()


def request(url: str, method: str = "GET", headers: dict | None = None) -> tuple[int, dict, bytes]:
    req = urllib.request.Request(url, method=method, headers=headers or {}, data=b"" if method == "POST" else None)
    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(req, timeout=5) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def post(base: str, path: str, body: bytes | None = None, **headers) -> tuple[int, dict]:
    req = urllib.request.Request(base + path, method="POST", headers={"X-GlassesData": "1", **headers}, data=body or b"")
    try:
        with urllib.request.build_opener(NoRedirect).open(req, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def read_jpegs(url: str, count: int, headers: dict | None = None, timeout: float = 5.0) -> list[bytes]:
    """The first ``count`` JPEG parts of an MJPEG stream."""
    return [jpeg for jpeg, _ in read_parts(url, count, headers, timeout)]


def read_parts(url: str, count: int, headers: dict | None = None, timeout: float = 5.0) -> list[tuple[bytes, dict]]:
    """The first ``count`` parts of an MJPEG stream: (JPEG, part headers with lower-case names)."""
    req = urllib.request.Request(url, headers=headers or {})
    frames = []
    with urllib.request.urlopen(req, timeout=timeout) as response:
        assert response.headers["Content-Type"] == f"multipart/x-mixed-replace; boundary={web.BOUNDARY}"
        while len(frames) < count:
            line = response.readline()
            assert line, "stream ended"
            if line.strip() != f"--{web.BOUNDARY}".encode():
                continue
            part_headers = {}
            while (header := response.readline().strip()):
                name, _, value = header.decode().partition(":")
                part_headers[name.lower()] = value.strip()
            assert part_headers["content-type"] == "image/jpeg"
            frames.append((response.read(int(part_headers["content-length"])), part_headers))
    return frames


# ---------------------------------------------------------------- page and status


def test_page_is_one_self_contained_file(live):
    status, headers, body = request(live.base + "/")
    page = body.decode()
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", page)  # no CDN, works offline
    for text in ("Pair over Wi-Fi", "<dialog", "/api/live.mjpg", "/api/stream.mjpg", "/api/status", "/api/history",
                 "/api/vcam/preview.jpg", "/api/vcam/settings", "/api/vcam/start", "/api/vcam/stop", 'role="switch"',
                 "prefers-color-scheme: light", "prefers-reduced-motion"):
        assert text in page
    # "waiting for the next keyframe" only when the decoder has no frame yet (see renderStage)
    assert page.count("waiting for the next keyframe") == 1 and "} else if (!v.has_frame) {" in page
    for group in ("Signal chain", "Virtual camera", "Link", "Hands", "Motion", "Glasses", "Recordings"):
        assert f">{group}</h2>" in page


def test_status_json_shape_with_a_fake_sender(live):
    status, headers, body = request(live.base + "/api/status")
    assert status == 200 and headers["Content-Type"].startswith("application/json")
    assert headers["Cache-Control"] == "no-store"
    data = json.loads(body)
    assert set(data) == {"time", "link", "video", "hands", "motion", "glasses", "recording", "wifi", "vcam"}
    assert data["link"] == {
        "connected": True, "text": "Connected over TCP", "kind": "tcp", "device": "Fake iPhone", "sources": {"tcp": "connected"},
    }
    video = data["video"]
    assert set(video) == {
        "format", "has_frame", "glasses_fps", "phone_sent_fps", "received_fps", "decoded_fps", "phone_dropped_frames",
        "phone_dropped_fps", "delay_ms", "rtt_ms", "display_skipped", "decode_errors", "decode_backlog",
    }
    assert video["format"] == {"width": 540, "height": 960, "fps": 30} and video["has_frame"] is True
    assert video["received_fps"] > 0 and video["decoded_fps"] > 0
    assert set(data["hands"]) == {"available", "enabled", "error", "fps", "process_ms", "count", "detected"}
    assert data["hands"]["available"] is False  # this receiver has no tracker
    assert set(data["motion"]) == {"accel", "gyro", "mag", "source", "samples_per_s"}
    assert len(data["motion"]["accel"]) == 3
    assert data["glasses"]["state"]["battery"] == 80 and data["glasses"]["inputs"] == []
    assert set(data["recording"]) == {"active", "name", "seconds", "frames", "note", "waiting_for_keyframe", "last", "folder"}
    assert data["wifi"] == {"enabled": False, "pairing_ready": False, "pair_at_start": False}
    assert data["vcam"] == web.VCAM_NOT_SET_UP  # this server was made without one


def test_status_survives_non_finite_numbers_from_the_phone():
    assert web._clean({"a": [1.0, float("nan")], "b": float("inf"), "c": {"d": 2}}) == {"a": [1.0, None], "b": None, "c": {"d": 2}}


# ---------------------------------------------------------------- stream


def test_stream_yields_jpeg_frames_at_most_at_the_video_rate(live):
    started = time.monotonic()
    frames = read_jpegs(live.base + "/api/stream.mjpg", 16)
    elapsed = time.monotonic() - started
    assert len(frames) == 16
    for jpeg in frames[:3]:
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert image is not None and image.shape[0] <= web.STREAM_MAX_HEIGHT
        assert image.shape[:2] == (720, 405)  # 540x960 scaled to 720 tall
    assert elapsed >= 14 / 30  # never faster than the 30 fps video


def test_live_stream_sends_the_clean_frame_with_its_landmarks_in_the_same_part(live):
    parts = read_parts(live.base + "/api/live.mjpg", 6)
    indices = [int(headers["x-frame"]) for _, headers in parts]
    assert indices == sorted(indices) and len(set(indices)) == len(indices)
    for jpeg, headers in parts:
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert image.shape[:2] == (720, 405)
        hands = json.loads(headers["x-hands"])
        assert hands == {"frame": int(headers["x-frame"]), "enabled": False, "lag": None, "hands": []}  # no tracker here
    wait_until(lambda: live.server.live_feed.viewers == 0, timeout=4)


def test_a_new_viewer_gets_the_current_picture_at_once_not_the_one_from_the_last_visit(live):
    """A page that loads while video flows must get its first part within a frame or two, and it
    must be the current frame, not the picture kept from the last time someone watched."""
    (_, first), = read_parts(live.base + "/api/live.mjpg", 1)
    wait_until(lambda: live.server.live_feed.viewers == 0, timeout=4)
    time.sleep(0.7)  # nobody watches; the video goes on (30 fps)
    started = time.monotonic()
    (_, again), = read_parts(live.base + "/api/live.mjpg", 1)
    assert time.monotonic() - started < 0.5
    assert int(again["x-frame"]) >= int(first["x-frame"]) + 10
    wait_until(lambda: live.server.live_feed.viewers == 0, timeout=4)


@dataclass(frozen=True)
class FakeHand:
    handedness: str
    score: float
    landmarks: list
    world_landmarks: list


def fake_hands(frame_index: int, process_ms: float = 40.0) -> HandsResult:
    points = [(0.1 + 0.01 * i, 0.5, -0.02) for i in range(21)]
    return HandsResult(frame_index, 0, [FakeHand("Right", 0.912, points, points)], process_ms)


def test_hands_header_is_compact_ascii_json():
    value = web.hands_header(12, fake_hands(10), enabled=True)
    value.encode("ascii")
    data = json.loads(value)
    assert data["frame"] == 12 and data["enabled"] is True and data["lag"] == 2
    (hand,) = data["hands"]
    assert hand["side"] == "Right" and hand["score"] == 0.91
    assert len(hand["points"]) == 42 and hand["points"][:4] == [0.1, 0.5, 0.11, 0.5]
    assert json.loads(web.hands_header(12, fake_hands(10), enabled=False))["hands"] == []
    stale = 10 + viewer.HANDS_STALE_FRAMES + 1
    assert json.loads(web.hands_header(stale, fake_hands(10), enabled=True)) == {"frame": stale, "enabled": True, "lag": None, "hands": []}
    assert json.loads(web.hands_header(9, fake_hands(10), enabled=True))["hands"] == []  # results never run ahead


class ScriptedHands:
    """A stand-in receiver whose hand results change at given delays after ``start``."""

    def __init__(self, first, later=()):
        self.hands_active = True
        self._first = first
        self._later = list(later)  # (seconds after start, result)
        self._started = time.monotonic()

    def latest_hands(self):
        elapsed = time.monotonic() - self._started
        current = self._first
        for at, result in self._later:
            if elapsed >= at:
                current = result
        return current


def frame_at(index: int) -> DecodedFrame:
    return DecodedFrame(np.zeros((8, 8, 3), np.uint8), 0, 0, index)


@pytest.mark.parametrize(
    "first, later, expected, waits",
    [
        (fake_hands(9), [(0.01, fake_hands(10))], 10, True),  # keeping up: waits for this frame's own result
        (fake_hands(9), [(0.01, fake_hands(11))], 9, True),  # the tracker skipped it: the earlier result
        (fake_hands(9), [], 9, True),  # nothing in time: gives up after about two processing times
        (fake_hands(4), [(0.01, fake_hands(10))], 4, False),  # far behind: no wait at all
        (fake_hands(10), [], 10, False),  # already there
        (fake_hands(11), [], None, False),  # a result for a later frame is never used
        (None, [], None, False),
    ],
)
def test_the_feed_pairs_a_frame_with_its_own_hand_result_when_the_tracker_keeps_up(first, later, expected, waits):
    feed = web.JpegFeed(ScriptedHands(first, later), threading.Event(), burn_in=False)
    started = time.monotonic()
    chosen = feed._hands_for(frame_at(10))
    elapsed = time.monotonic() - started
    assert (chosen.frame_index if chosen else None) == expected
    assert elapsed < web.HANDS_WAIT_MAX_S + 0.05
    if not waits:
        assert elapsed < 0.03  # a wait would take HANDS_WAIT_MAX_S (60 ms) here


def test_history_has_rows_for_the_charts_and_filters_by_time(live):
    wait_until(lambda: len(live.server.history.rows()) >= 5)
    status, headers, body = request(live.base + "/api/history")
    assert status == 200 and headers["Content-Type"].startswith("application/json") and headers["Cache-Control"] == "no-store"
    data = json.loads(body)
    assert data["fields"] == list(web.HISTORY_FIELDS) and data["interval_s"] == web.HISTORY_INTERVAL_S
    rows = data["rows"]
    assert len(rows) >= 5 and all(len(row) == len(data["fields"]) for row in rows)
    times = [row[0] for row in rows]
    assert times == sorted(times)
    latest = dict(zip(data["fields"], rows[-1]))
    assert latest["expected_fps"] == 30 and latest["received_fps"] > 0 and latest["decoded_fps"] > 0
    assert latest["hands_fps"] is None  # no tracker
    assert latest["ay"] == 9.81 and latest["mx"] == 20.0
    newer = json.loads(request(f"{live.base}/api/history?since={times[-2]}")[2])["rows"]
    assert newer and all(row[0] > times[-2] for row in newer)
    for junk in ("nan", "inf", "x", ""):
        assert request(f"{live.base}/api/history?since={junk}")[0] == 200
    assert web.History(live.receiver, threading.Event())._rows.maxlen == round(web.HISTORY_SECONDS / web.HISTORY_INTERVAL_S)


def test_history_rows_are_empty_while_nothing_is_connected(tmp_path):
    hub = Receiver(tmp_path)
    try:
        row = web.History(hub, threading.Event()).sample()
        assert len(row) == len(web.HISTORY_FIELDS) and row[0] > 0 and row[1:] == [None] * (len(row) - 1)
    finally:
        hub.shutdown()


def test_a_stalled_viewer_blocks_neither_other_viewers_nor_the_recording(live, monkeypatch):
    monkeypatch.setattr(web._Handler, "timeout", 1.0)  # drop a stalled viewer quickly
    stalled = socket.socket()
    stalled.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2048)
    stalled.connect(("127.0.0.1", live.server.port))
    stalled.sendall(f"GET /api/stream.mjpg HTTP/1.1\r\nHost: 127.0.0.1:{live.server.port}\r\n\r\n".encode())
    try:
        wait_until(lambda: live.server.feed.viewers == 1)
        assert post(live.base, "/api/record/start")[0] == 200
        frames_before = live.receiver.status().recording_frames
        started = time.monotonic()
        assert len(read_jpegs(live.base + "/api/stream.mjpg", 20)) == 20  # never reads the stalled one
        assert time.monotonic() - started < 3.0
        wait_until(lambda: live.receiver.status().recording_frames >= frames_before + 30)
        # The stalled viewer's own writes timed out and its connection was dropped.
        wait_until(lambda: live.server.feed.viewers == 0, timeout=6)
        assert live.receiver.status().decoded_fps > 20
    finally:
        stalled.close()
        post(live.base, "/api/record/stop")


# ---------------------------------------------------------------- actions and recordings


def test_record_start_and_stop_by_post_and_the_recordings_list(live):
    status, body = post(live.base, "/api/record/start")
    assert status == 200 and body["ok"] and body["message"].startswith("Recording to ")
    name = body["name"]
    assert post(live.base, "/api/record/start")[1]["message"] == f"Already recording to {name}"
    wait_until(lambda: live.receiver.status().recording_frames >= 15)
    listing = json.loads(request(live.base + "/api/recordings")[2])
    assert listing["folder"] == str(live.receiver.out_root)
    assert listing["recordings"][0]["name"] == name and listing["recordings"][0]["state"] == "recording"
    status, body = post(live.base, "/api/record/stop")
    assert status == 200 and body == {"ok": True, "message": f"Saved {name}", "name": name}
    assert post(live.base, "/api/record/stop")[1]["message"] == "Not recording"

    entries = json.loads(request(live.base + "/api/recordings")[2])["recordings"]
    newest = entries[0]
    assert newest["name"] == name and newest["state"] == "complete"
    assert newest["frames"] >= 15 and newest["seconds"] > 0 and newest["bytes"] > 0 and newest["started"]
    folder = live.receiver.out_root / name
    with av.open(str(folder / "video.mp4")) as container:
        assert sum(1 for _ in container.decode(video=0)) == newest["frames"]
    data = json.loads(request(live.base + "/api/status")[2])
    assert data["recording"]["active"] is False and data["recording"]["last"]["name"] == name


def test_hands_toggle_without_a_tracker_is_refused(live):
    status, body = post(live.base, "/api/hands/toggle")
    assert status == 409 and body["ok"] is False


def test_post_needs_the_custom_header_and_a_matching_origin(live):
    assert request(live.base + "/api/record/stop", "POST")[0] == 403
    assert post(live.base, "/api/record/stop", Origin="http://evil.example")[0] == 403
    assert post(live.base, "/api/record/stop", Origin="null")[0] == 403
    assert post(live.base, "/api/record/stop", **{"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert post(live.base, "/api/record/stop", Origin=live.base)[0] == 200


def test_loopback_server_refuses_foreign_host_names(live):
    # A DNS-rebinding page reaches 127.0.0.1 under its own name; the Host header gives it away.
    for path in ("/api/status", "/api/history", "/api/live.mjpg", "/api/stream.mjpg"):
        connection = http.client.HTTPConnection("127.0.0.1", live.server.port, timeout=5)
        connection.request("GET", path, headers={"Host": f"attacker.example:{live.server.port}"})
        assert connection.getresponse().status == 403, path
        connection.close()
    for host in (f"localhost:{live.server.port}", f"127.0.0.1:{live.server.port}"):
        connection = http.client.HTTPConnection("127.0.0.1", live.server.port, timeout=5)
        connection.request("GET", "/api/status", headers={"Host": host})
        assert connection.getresponse().status == 200
        connection.close()


def test_pairing_code_only_with_wifi(live, tmp_path):
    assert request(live.base + "/api/pairing.png")[0] == 404
    hub = Receiver(tmp_path)
    server = web.WebServer(hub, "127.0.0.1", 0, wifi=True)
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        assert request(base + "/api/pairing.png")[0] == 404  # listener not running yet
        from glassesdata_receiver.wifi import qr_png

        server.pairing_png = qr_png("gdpair://v1?token=secret")
        status, headers, body = request(base + "/api/pairing.png")
        assert status == 200 and headers["Content-Type"] == "image/png" and body.startswith(b"\x89PNG")
        assert json.loads(request(base + "/api/status")[2])["wifi"]["pairing_ready"] is True
    finally:
        server.stop()
        hub.shutdown()


def test_list_recordings_newest_first(tmp_path):
    def make(name, info, frames_rows=None):
        folder = tmp_path / name
        folder.mkdir()
        if info is not None:
            (folder / "info.json").write_text(json.dumps(info))
        if frames_rows is not None:
            (folder / "frames.csv").write_text("frame_index,pts_us\n" + "".join(f"{i},{i}\n" for i in range(frames_rows)))
        return folder

    make("2026-01-01_100000", {"startedAt": "2026-01-01T10:00:00+00:00", "startedMacTimeNs": 0,
                               "endedMacTimeNs": 2_500_000_000, "framesWritten": 75})
    make("2026-01-02_100000", {"startedAt": "2026-01-02T10:00:00+00:00", "startedMacTimeNs": 5}, frames_rows=12)
    active = make("2026-01-03_100000", {"startedAt": "2026-01-03T10:00:00+00:00", "startedMacTimeNs": 5})
    make("not-a-recording", None)
    (tmp_path / "loose-file.txt").write_text("x")
    entries = web.list_recordings(tmp_path, active, 4.0, 120)
    assert [entry["name"] for entry in entries] == ["2026-01-03_100000", "2026-01-02_100000", "2026-01-01_100000"]
    assert [entry["state"] for entry in entries] == ["recording", "interrupted", "complete"]
    assert [entry["frames"] for entry in entries] == [120, 12, 75]
    assert [entry["seconds"] for entry in entries] == [4.0, None, 2.5]
    assert all(entry["bytes"] > 0 for entry in entries)
    assert web.list_recordings(tmp_path / "missing") == []


# ---------------------------------------------------------------- virtual camera


class FakeCamera:
    def __init__(self, width, height, fps):
        self.width, self.height, self.device, self.backend = width, height, "Fake Camera", "fake"
        self.sent = 0

    def send(self, picture):
        assert picture.shape == (self.height, self.width, 3)
        self.sent += 1

    def close(self):
        pass


@pytest.fixture
def with_vcam(tmp_path):
    """A receiver with no iPhone and a virtual camera that sends to a fake device."""
    hub = Receiver(tmp_path / "recordings")
    camera = vcam.VirtualCamera(hub, size=(1280, 720), open_camera=FakeCamera)
    server = web.WebServer(hub, "127.0.0.1", 0, vcam=camera)
    server.start()
    yield server, f"http://127.0.0.1:{server.port}"
    camera.stop()
    server.stop()
    hub.shutdown()


def vcam_status(base: str) -> dict:
    return json.loads(request(base + "/api/status")[2])["vcam"]


def test_vcam_status_and_start_stop_by_post(with_vcam):
    server, base = with_vcam
    status = vcam_status(base)
    assert set(status) == set(web.VCAM_NOT_SET_UP)
    assert status["available"] is True and status["running"] is False and (status["width"], status["height"]) == (1280, 720)
    assert (status["mode"], status["position"], status["overlay"], status["frames_sent"]) == ("fit", 0.5, False, 0)
    assert request(base + "/api/vcam/preview.jpg")[0] == 404
    code, body = post(base, "/api/vcam/start")
    assert code == 200 and body["ok"] and body["message"] == "Virtual camera on: Fake Camera" and body["vcam"]["running"]
    wait_until(lambda: vcam_status(base)["frames_sent"] > 3)
    status = vcam_status(base)
    assert status["device"] == "Fake Camera" and status["backend"] == "fake" and status["showing"] == "card"  # no iPhone
    code, headers, jpeg = request(base + "/api/vcam/preview.jpg")
    assert code == 200 and headers["Content-Type"] == "image/jpeg" and headers["Cache-Control"] == "no-store"
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (270, 480, 3)
    code, body = post(base, "/api/vcam/stop")
    assert code == 200 and body["vcam"]["running"] is False
    assert request(base + "/api/vcam/preview.jpg")[0] == 404


def test_vcam_settings_by_post_and_refusals(with_vcam):
    server, base = with_vcam
    ok = [({"mode": "fill"}, "Virtual camera: Fill: a band of the picture"),
          ({"position": 0.25}, "Virtual camera: band at 25% from the top"),
          ({"overlay": True, "mode": "fit"}, "Virtual camera: Fit: the whole picture, hand skeleton on")]
    for change, message in ok:
        code, body = post(base, "/api/vcam/settings", json.dumps(change).encode())
        assert code == 200 and body == {"ok": True, "message": message, "vcam": body["vcam"]}
    status = vcam_status(base)
    assert (status["mode"], status["position"], status["overlay"]) == ("fit", 0.25, True)
    for bad in (b"", b"not json", b"[1]", b"{}", b'{"mode": "zoom"}', b'{"position": 2}', b'{"position": -1}',
                b'{"position": NaN}', b'{"position": "0.5"}', b'{"overlay": "on"}', b'{"size": "640x480"}', b"\xff\xfe",
                b'{"position": null}', b'{"mode": null}', b'{"overlay": null}', b'{"position": 1' + b"0" * 400 + b"}",
                b'{"mode": ["fill"]}', b'{"position": true}'):
        code, body = post(base, "/api/vcam/settings", bad)
        assert code == 400 and body["ok"] is False and body["message"].startswith("Not changed: "), bad
    assert vcam_status(base) == status  # nothing changed
    assert post(base, "/api/vcam/unknown")[0] == 404


def test_vcam_actions_are_protected_like_the_others(with_vcam):
    server, base = with_vcam
    for path in ("/api/vcam/start", "/api/vcam/stop", "/api/vcam/settings"):
        assert request(base + path, "POST")[0] == 403  # no X-GlassesData header
        assert post(base, path, b'{"mode": "fill"}', Origin="http://evil.example")[0] == 403
        assert post(base, path, b'{"mode": "fill"}', **{"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert vcam_status(base)["running"] is False and vcam_status(base)["mode"] == "fit"
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    connection.request("GET", "/api/vcam/preview.jpg", headers={"Host": f"attacker.example:{server.port}"})
    assert connection.getresponse().status == 403
    connection.close()


def test_vcam_that_cannot_open_answers_409_with_the_reason_and_the_page_shows_a_hint(tmp_path):
    def refuse(width, height, fps):
        raise vcam.Unavailable("the virtual camera device could not be opened (no OBS)", vcam.setup_hint("darwin"))

    hub = Receiver(tmp_path)
    server = web.WebServer(hub, "127.0.0.1", 0, vcam=vcam.VirtualCamera(hub, open_camera=refuse))
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        code, body = post(base, "/api/vcam/start")
        assert code == 409 and body["message"] == "Virtual camera unavailable: the virtual camera device could not be opened (no OBS)"
        status = vcam_status(base)
        assert status["running"] is False and "OBS Studio" in status["hint"]
        assert json.loads(request(base + "/api/status")[2])["link"]["connected"] is False  # the receiver carries on
    finally:
        server.stop()
        hub.shutdown()


def test_vcam_actions_without_a_virtual_camera(live):
    code, body = post(live.base, "/api/vcam/start")
    assert code == 409 and body["ok"] is False


# ---------------------------------------------------------------- access key (non-loopback hosts)


@pytest.mark.parametrize(
    "host, loopback",
    [("127.0.0.1", True), ("127.1.2.3", True), ("::1", True), ("[::1]", True), ("localhost", True), ("LOCALHOST", True),
     ("0.0.0.0", False), ("::", False), ("192.168.1.20", False), ("my-laptop.local", False), ("", False)],
)
def test_loopback_hosts(host, loopback):
    assert web.is_loopback_host(host) is loopback


@pytest.fixture
def keyed(tmp_path):
    """A server that behaves as on a non-loopback host, bound to 127.0.0.1 so tests stay local."""
    hub = Receiver(tmp_path / "recordings")
    server = web.WebServer(hub, "127.0.0.1", 0, wifi=True, require_key=True)
    from glassesdata_receiver.wifi import qr_png

    server.pairing_png = qr_png("gdpair://v1?token=secret")
    server.start()
    yield server, f"http://127.0.0.1:{server.port}"
    server.stop()
    hub.shutdown()


def test_every_endpoint_needs_the_key(keyed):
    server, base = keyed
    assert server.key and len(server.key) >= 30
    assert server.url() == f"http://127.0.0.1:{server.port}/?key={server.key}"
    gets = ["/", "/api/status", "/api/history", "/api/stream.mjpg", "/api/live.mjpg", "/api/recordings", "/api/pairing.png",
            "/api/vcam/preview.jpg", "/nothing"]
    posts = ["/api/record/start", "/api/record/stop", "/api/hands/toggle", "/api/vcam/start", "/api/vcam/stop", "/api/vcam/settings"]
    for path in gets:
        assert request(base + path)[0] == 403, path
        assert request(base + path + "?key=wrong" + server.key[5:])[0] == 403, path
    for path in posts:
        assert request(base + path, "POST", {"X-GlassesData": "1"})[0] == 403, path
    cookie = f"{server.cookie_name}={server.key}"
    assert request(base + "/api/pairing.png", headers={"Cookie": f"{server.cookie_name}=wrong"})[0] == 403
    assert request(base + "/api/pairing.png", headers={"Cookie": cookie})[0] == 200
    assert post(base, "/api/record/stop", Cookie=cookie)[0] == 200
    assert post(base, "/api/record/stop", Cookie=cookie, Origin="http://evil.example")[0] == 403  # cross-site
    assert request(base + "/api/record/stop", "POST", {"Cookie": cookie})[0] == 403  # no custom header


def test_the_key_in_the_url_sets_an_httponly_cookie(keyed):
    server, base = keyed
    status, headers, _ = request(f"{base}/?key={server.key}")
    assert status == 303 and headers["Location"] == "/"  # the key leaves the address bar
    cookie = headers["Set-Cookie"]
    assert cookie.startswith(f"{server.cookie_name}={server.key};")
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie and "Path=/" in cookie
    status, headers, body = request(f"{base}/", headers={"Cookie": cookie.split(";")[0]})
    assert status == 200 and b"<!doctype html>" in body
    status, headers, _ = request(f"{base}/api/status?key={server.key}")  # e.g. curl
    assert status == 200 and "Set-Cookie" in headers


# ---------------------------------------------------------------- the command line, end to end


FAKE_PYVIRTUALCAM = """
class PixelFormat:
    BGR = "BGR"


class Camera:  # like the OBS Virtual Camera: landscape only
    def __init__(self, width, height, fps, fmt=None, **kw):
        if (width, height) not in ((1920, 1080), (1280, 720)):
            raise RuntimeError("unsupported size")
        self.width, self.height, self.device, self.backend = width, height, "Stand-in Camera", "stand-in"

    def send(self, frame):
        assert frame.shape == (self.height, self.width, 3), frame.shape

    def close(self):
        pass
"""


def test_cli_serves_the_page_and_the_phone_controls_recording(tmp_path):
    sender = FakeSender("--loop", "--seconds", "6", "--record-after", "1.5", "--record-for", "1.5")
    out = tmp_path / "recordings"
    fake_modules = tmp_path / "modules"
    (fake_modules / "pyvirtualcam").mkdir(parents=True)
    (fake_modules / "pyvirtualcam" / "__init__.py").write_text(FAKE_PYVIRTUALCAM)
    receiver = subprocess.Popen(
        [sys.executable, "-m", "glassesdata_receiver", "--tcp", f"127.0.0.1:{sender.port}", "--no-hands",
         "--web-port", "0", "--out", str(out), "--duration", "4.5", "--snapshot", str(tmp_path / "last.png"),
         "--virtual-camera", "--virtual-camera-size", "1280x720", "--virtual-camera-mode", "fill"],
        cwd=RECEIVER_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "PYTHONPATH": str(fake_modules)},
    )
    try:
        line = receiver.stdout.readline()
        assert line.startswith("WEB_URL=http://127.0.0.1:"), line
        base = line.strip().removeprefix("WEB_URL=").rstrip("/")
        assert request(base + "/")[0] == 200
        wait_until(lambda: json.loads(request(base + "/api/status")[2])["link"]["connected"], timeout=10)
        # --virtual-camera: started at launch, landscape, sending the video (not the card) on its own thread
        wait_until(lambda: json.loads(request(base + "/api/status")[2])["vcam"]["showing"] == "video", timeout=5)
        status = json.loads(request(base + "/api/status")[2])["vcam"]
        assert status["running"] and status["device"] == "Stand-in Camera" and status["mode"] == "fill"
        assert (status["width"], status["height"], status["last_error"]) == (1280, 720, None)
        stdout, stderr = receiver.communicate(timeout=20)
        assert receiver.returncode == 0, stderr
        assert sender.wait() == 0
    finally:
        if receiver.poll() is None:
            receiver.kill()
        sender.stop()
    assert "WEB_URL" not in stdout  # printed once
    states = [json.loads(line.removeprefix("RECORDER_STATE ")) for line in sender.lines if line.startswith("RECORDER_STATE ")]
    assert states[0] == {"recording": False, "name": None, "seconds": 0.0, "frames": 0, "note": None}
    assert any(state["recording"] for state in states)
    final = states[-1]
    assert final["recording"] is False and final["frames"] > 20 and final["note"] is None
    folders = [folder for folder in out.iterdir() if folder.is_dir()]
    assert [folder.name for folder in folders] == [final["name"]]
    with av.open(str(folders[0] / "video.mp4")) as container:
        assert sum(1 for _ in container.decode(video=0)) == final["frames"]
    assert cv2.imread(str(tmp_path / "last.png")).shape == (960, 540, 3)  # --snapshot still works
    assert "Virtual camera self-check: opened 'Stand-in Camera' (backend stand-in); sending 1280x720 at 30 fps, fill" in stderr


def test_a_viewer_that_leaves_before_any_video_is_dropped(tmp_path):
    hub = Receiver(tmp_path)
    server = web.WebServer(hub, "127.0.0.1", 0)
    server.start()
    try:
        viewer = socket.create_connection(("127.0.0.1", server.port))
        viewer.sendall(f"GET /api/stream.mjpg HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\n\r\n".encode())
        assert viewer.recv(4096).startswith(b"HTTP/1.1 200")
        wait_until(lambda: server.feed.viewers == 1)
        viewer.close()  # no frame was ever written, so only the idle check can notice
        wait_until(lambda: server.feed.viewers == 0, timeout=4)
    finally:
        server.stop()
        hub.shutdown()


def test_recordings_list_ignores_a_malformed_info_json(tmp_path):
    for name, text in (("a", "[]"), ("b", "null"), ("c", '"text"'), ("d", "{not json")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "info.json").write_text(text)
    entries = web.list_recordings(tmp_path)
    assert sorted(entry["name"] for entry in entries) == ["a", "b", "c", "d"]
    assert all(entry["state"] == "interrupted" for entry in entries)
