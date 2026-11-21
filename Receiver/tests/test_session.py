"""Session handshake, clock estimation and the receiver hub, over socketpairs."""

import csv
import json
import logging
import socket
import threading
import time

import av
import pytest

from glassesdata_receiver import protocol as p
from glassesdata_receiver import session as session_module
from glassesdata_receiver.receiver import Receiver
from glassesdata_receiver.session import Session

HELLO = {"protocol": p.PROTOCOL_VERSION, "app": "test", "device": "iPhone", "name": "Test iPhone"}


class Phone:
    """The iPhone end of a socketpair."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.sock.settimeout(3.0)
        self._reader = p.FrameReader()
        self._pending: list[p.Message] = []

    def send(self, data: bytes) -> None:
        self.sock.sendall(data)

    def hello(self, **fields) -> None:
        self.send(p.encode_json(p.MessageType.HELLO, {**HELLO, **fields}))

    def receive(self, kind: p.MessageType) -> p.Message:
        while True:
            while self._pending:
                message = self._pending.pop(0)
                if message.type is kind:
                    return message
            data = self.sock.recv(65536)
            if not data:
                raise ConnectionError("receiver closed the connection")
            self._pending += [p.decode_message(t, payload) for t, payload in self._reader.feed(data)]

    def next(self) -> p.Message:
        """The next message of any type."""
        while not self._pending:
            data = self.sock.recv(65536)
            if not data:
                raise ConnectionError("receiver closed the connection")
            self._pending += [p.decode_message(t, payload) for t, payload in self._reader.feed(data)]
        return self._pending.pop(0)

    def state(self, predicate=lambda body: True) -> dict:
        """The next RECORDER_STATE body that matches ``predicate``."""
        while True:
            body = self.receive(p.MessageType.RECORDER_STATE).body
            if predicate(body):
                return body

    def control(self, command) -> None:
        self.send(p.encode_json(p.MessageType.CONTROL, {"t": time.monotonic_ns(), "cmd": command}))

    def close(self) -> None:
        self.sock.close()


class Running:
    """A Session.run on its own thread."""

    def __init__(self, receiver: Receiver, token: str | None = None) -> None:
        mac_side, phone_side = socket.socketpair()
        self.phone = Phone(phone_side)
        self.session = Session(mac_side, "tcp", "test-peer", receiver, token=token)
        self.reason: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self.reason = self.session.run()

    def wait(self, timeout: float = 3.0) -> str:
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "session did not end"
        return self.reason


@pytest.fixture
def receiver(tmp_path):
    hub = Receiver(tmp_path / "recordings")
    yield hub
    hub.shutdown()


def wait_until(condition, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.01)


def test_good_hello_is_acknowledged(receiver):
    running = Running(receiver)
    running.phone.hello()
    ack = running.phone.receive(p.MessageType.HELLO_ACK).body
    assert ack["ok"] is True and ack["protocol"] == p.PROTOCOL_VERSION
    assert ack["name"] == session_module.mac_name() and "code" not in ack and "error" not in ack
    wait_until(lambda: receiver.status().kind == "tcp")
    assert receiver.busy and receiver.status().device == "Test iPhone"
    running.phone.close()
    assert running.wait() == "iPhone closed the connection"
    assert not receiver.busy


def test_wrong_protocol_version_is_rejected(receiver):
    running = Running(receiver)
    running.phone.hello(protocol=99)
    ack = running.phone.receive(p.MessageType.HELLO_ACK).body
    assert ack["ok"] is False and "protocol" in ack["error"]
    assert ack["code"] == "protocol" and ack["name"] == session_module.mac_name()
    assert running.wait().startswith("protocol error")
    assert not receiver.busy


def test_first_message_must_be_hello(receiver):
    running = Running(receiver)
    running.phone.send(p.encode_json(p.MessageType.STATS, {"t": 1}))
    assert running.wait() == "protocol error: first message must be HELLO"


@pytest.mark.parametrize("token", [None, "wrong", "sécret", 12345])
def test_missing_or_wrong_token_is_rejected(receiver, token):
    running = Running(receiver, token="secret")
    if token is None:
        running.phone.hello()
    else:
        running.phone.hello(token=token)
    ack = running.phone.receive(p.MessageType.HELLO_ACK).body
    assert ack["ok"] is False and ack["error"] == "not paired with this Mac"
    assert ack["code"] == "pairing" and ack["name"]
    assert running.wait() == "rejected: wrong pairing token"
    assert not receiver.busy


def test_correct_token_is_accepted(receiver):
    running = Running(receiver, token="secret")
    running.phone.hello(token="secret")
    assert running.phone.receive(p.MessageType.HELLO_ACK).body["ok"] is True
    running.phone.close()
    running.wait()


def test_second_session_is_busy(receiver):
    first = Running(receiver)
    first.phone.hello()
    assert first.phone.receive(p.MessageType.HELLO_ACK).body["ok"] is True
    second = Running(receiver)
    second.phone.hello(name="Other iPhone")
    ack = second.phone.receive(p.MessageType.HELLO_ACK).body
    assert ack["ok"] is False and ack["error"].startswith("busy")
    assert ack["code"] == "busy" and ack["name"]
    assert second.wait() == "rejected: busy"
    assert receiver.status().device == "Test iPhone"  # the first session is untouched
    first.phone.close()
    first.wait()


def test_ping_pong_estimates_the_clock_offset(receiver, monkeypatch):
    monkeypatch.setattr(session_module, "PING_INTERVAL_S", 0.05)
    monkeypatch.setattr(session_module, "POLL_S", 0.02)
    skew_ns = 123_456_789_000
    running = Running(receiver)
    running.phone.hello()
    running.phone.receive(p.MessageType.HELLO_ACK)
    for _ in range(5):
        ping = running.phone.receive(p.MessageType.PING).body
        running.phone.send(p.encode_pong(p.Pong(ping.mac_time_ns, time.monotonic_ns() + skew_ns)))
    wait_until(lambda: running.session.clock.best is not None and receiver.status().rtt_ms is not None)
    best = running.session.clock.best
    assert abs(best.offset_ns - skew_ns) < 5_000_000  # within 5 ms on loopback
    assert 0 <= best.rtt_ns < 50_000_000
    phone_time = time.monotonic_ns() + skew_ns
    assert abs(running.session.clock.phone_to_mac(phone_time) - (phone_time - skew_ns)) < 5_000_000
    running.phone.close()
    running.wait()


def test_idle_connection_times_out(receiver, monkeypatch):
    monkeypatch.setattr(session_module, "IDLE_TIMEOUT_S", 0.3)
    monkeypatch.setattr(session_module, "POLL_S", 0.05)
    running = Running(receiver)
    running.phone.hello()
    running.phone.receive(p.MessageType.HELLO_ACK)
    started = time.monotonic()
    assert running.wait() == "connection timed out"
    assert time.monotonic() - started < 2.0
    assert not receiver.busy


def test_messages_sent_with_hello_are_handled_at_once(receiver):
    running = Running(receiver)
    state = {"t": 1, "battery": 55}
    running.phone.send(p.encode_json(p.MessageType.HELLO, HELLO) + p.encode_json(p.MessageType.DEVICE_STATE, state))
    wait_until(lambda: receiver.status().device_state == state, timeout=1.0)
    running.phone.close()
    running.wait()


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))[1:]


def test_recording_starts_with_the_known_device_state(receiver, plant, caplog):
    fmt, packets = plant
    running = Running(receiver)
    running.phone.hello()
    running.phone.receive(p.MessageType.HELLO_ACK)
    state = {"t": 42, "battery": 77, "charging": "no", "don": "donned", "hinge": "open", "thermal": "nominal", "link": "ok"}
    running.phone.send(p.encode_video_format(fmt) + p.encode_json(p.MessageType.DEVICE_STATE, state))
    wait_until(lambda: receiver.status().format == fmt and receiver.status().device_state == state)

    with caplog.at_level(logging.INFO, logger="glassesdata_receiver.receiver"):
        message = receiver.toggle_recording()
    assert message.startswith("Recording to ")
    assert sum("Recording to" in record.getMessage() for record in caplog.records) == 1
    assert receiver.start_recording() == receiver.status().recording  # idempotent

    for packet in packets[:24]:
        running.phone.send(p.encode_video(packet))
    wait_until(lambda: receiver.status().recording_frames == 24)
    folder = receiver.status().recording
    assert receiver.toggle_recording() == f"Saved {folder}"

    device_rows = read_rows(folder / "device_state.csv")
    assert len(device_rows) == 1 and device_rows[0][:7] == ["42", "77", "no", "donned", "open", "nominal", "ok"]
    assert len(read_rows(folder / "frames.csv")) == 24
    running.phone.close()
    running.wait()


def test_disconnect_closes_the_recording_and_forgets_the_format(receiver, plant):
    fmt, packets = plant
    running = Running(receiver)
    running.phone.hello()
    running.phone.receive(p.MessageType.HELLO_ACK)
    running.phone.send(p.encode_video_format(fmt))
    wait_until(lambda: receiver.status().format == fmt)
    assert receiver.start_recording() is not None
    running.phone.send(b"".join(p.encode_video(packet) for packet in packets[:12]))
    wait_until(lambda: receiver.status().recording_frames == 12)
    running.phone.close()
    running.wait()
    status = receiver.status()
    assert status.recording is None and status.last_recording is not None
    assert receiver.start_recording() is None  # no session

    again = Running(receiver)
    again.phone.hello()
    again.phone.receive(p.MessageType.HELLO_ACK)
    wait_until(lambda: receiver.status().kind == "tcp")
    assert receiver.status().format is None and receiver.status().device_state == {}
    again.phone.close()
    again.wait()


class FakeTracker:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def detect(self, image, timestamp_ms):
        time.sleep(0.02)
        self._log.append("detect")
        return []

    def close(self) -> None:
        self._log.append("close")


def test_shutdown_joins_workers_and_closes_the_tracker(tmp_path, plant):
    fmt, packets = plant
    calls: list[str] = []
    hub = Receiver(tmp_path, lambda: FakeTracker(calls))
    running = Running(hub)
    running.phone.hello()
    running.phone.receive(p.MessageType.HELLO_ACK)
    running.phone.send(p.encode_video_format(fmt))
    wait_until(lambda: hub.status().format == fmt)
    assert hub.start_recording() is not None
    running.phone.send(b"".join(p.encode_video(packet) for packet in packets[:24]))
    wait_until(lambda: "detect" in calls)

    hub.shutdown()
    assert not any(thread.is_alive() for thread in hub._threads)
    assert calls[-1] == "close" and calls.count("close") == 1
    assert running.wait() == "closed"
    status = hub.status()
    assert status.recording is None and status.last_recording is not None
    assert hub.start_recording() is None  # nothing restarts after shutdown


def connected(receiver, fmt) -> Running:
    running = Running(receiver)
    running.phone.hello()
    running.phone.receive(p.MessageType.HELLO_ACK)
    running.phone.send(p.encode_video_format(fmt))
    wait_until(lambda: receiver.status().format == fmt)
    return running


def test_manual_stop_is_not_undone_by_auto_record(receiver, plant):
    fmt, _ = plant
    running = connected(receiver, fmt)
    assert receiver.start_recording(auto=True) is not None
    assert receiver.toggle_recording().startswith("Saved ")
    assert receiver.start_recording(auto=True) is None  # what the --record loop calls
    assert receiver.toggle_recording().startswith("Recording to ")  # the r key still works
    receiver.toggle_recording()
    running.phone.close()
    running.wait()


def test_a_failing_recording_write_keeps_the_link(receiver, plant, monkeypatch):
    fmt, packets = plant
    running = connected(receiver, fmt)
    folder = receiver.start_recording()

    def disk_full(self, body, mac_time_ns):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("glassesdata_receiver.recorder.Recording.write_motion", disk_full)
    running.phone.send(p.encode_json(p.MessageType.MOTION, {"t": 1}))
    wait_until(lambda: receiver.status().recording is None)
    status = receiver.status()
    assert status.kind == "tcp" and receiver.busy  # still connected
    assert status.last_recording == folder and "No space left" in status.link
    assert receiver.start_recording(auto=True) is None  # --record does not retry in a loop
    running.phone.send(b"".join(p.encode_video(packet) for packet in packets[:12]))
    wait_until(lambda: receiver.status().received_fps > 0)
    running.phone.close()
    assert running.wait() == "iPhone closed the connection"


def test_ack_carries_the_given_mac_name(receiver):
    mac_side, phone_side = socket.socketpair()
    phone = Phone(phone_side)
    session = Session(mac_side, "wifi", "peer", receiver, token="t", name="Studio Mac")
    thread = threading.Thread(target=session.run, daemon=True)
    thread.start()
    phone.hello(token="t")
    assert phone.receive(p.MessageType.HELLO_ACK).body["name"] == "Studio Mac"
    phone.close()
    thread.join(3.0)


def test_timestamp_reset_starts_a_new_recording(receiver, plant):
    fmt, packets = plant
    running = connected(receiver, fmt)
    first = receiver.start_recording()
    running.phone.send(b"".join(p.encode_video(packet) for packet in packets))
    wait_until(lambda: receiver.status().recording_frames == len(packets))
    # The glasses stream restarts: same format, timestamps from 0 again.
    running.phone.send(b"".join(p.encode_video(packet) for packet in packets))
    wait_until(lambda: receiver.status().recording not in (None, first) and receiver.status().recording_frames == len(packets))
    status = receiver.status()
    second = status.recording
    assert "timestamps went back" in status.recording_note
    receiver.toggle_recording()
    running.phone.close()
    running.wait()

    for folder in (first, second):
        assert len(read_rows(folder / "frames.csv")) == len(packets)  # nothing lost from either pass
        with av.open(str(folder / "video.mp4")) as container:
            assert sum(1 for _ in container.decode(video=0)) == len(packets)
    assert [int(row[1]) for row in read_rows(second / "frames.csv")] == [packet.pts_us for packet in packets]


def test_a_non_keyframe_with_an_old_timestamp_is_skipped(receiver, plant):
    fmt, packets = plant
    running = connected(receiver, fmt)
    folder = receiver.start_recording()
    running.phone.send(b"".join(p.encode_video(packet) for packet in packets[:24] + [packets[13]] + packets[24:30]))
    wait_until(lambda: receiver.status().recording_frames == 30)
    assert receiver.status().recording == folder and receiver.status().recording_note == ""
    running.phone.close()
    running.wait()


def test_recording_metadata_reaches_info_json(tmp_path, plant):
    fmt, _ = plant
    hub = Receiver(tmp_path, recording_metadata={"handedness": "anatomical"})
    try:
        running = connected(hub, fmt)
        folder = hub.start_recording()
        hub.toggle_recording()
        info = json.loads((folder / "info.json").read_text())
        assert info["handedness"] == "anatomical" and info["link"] == "tcp"
        running.phone.close()
        running.wait()
    finally:
        hub.shutdown()


# ---------------------------------------------------------------- remote record control and the outbox

STATE_KEYS = {"recording", "name", "seconds", "frames", "note"}


def test_recorder_state_comes_right_after_hello_ack(receiver):
    running = Running(receiver)
    running.phone.hello()
    assert running.phone.next().type is p.MessageType.HELLO_ACK
    first = running.phone.next()
    assert first.type is p.MessageType.RECORDER_STATE
    assert first.body == {"recording": False, "name": None, "seconds": 0.0, "frames": 0, "note": None}
    running.phone.close()
    running.wait()


def test_control_starts_and_stops_a_recording(receiver, plant):
    fmt, packets = plant
    running = connected(receiver, fmt)
    running.phone.control("record_start")
    started = running.phone.state(lambda body: body["recording"])
    assert set(started) == STATE_KEYS and started["note"] is None and started["name"]
    folder = receiver.status().recording
    assert folder.name == started["name"]
    running.phone.send(b"".join(p.encode_video(packet) for packet in packets[:24]))
    wait_until(lambda: receiver.status().recording_frames == 24)
    running.phone.control("record_stop")
    stopped = running.phone.state(lambda body: not body["recording"])
    assert stopped["name"] == folder.name and stopped["frames"] == 24 and stopped["seconds"] > 0
    assert receiver.status().recording is None and receiver.start_recording(auto=True) is None  # a manual stop
    with av.open(str(folder / "video.mp4")) as container:
        assert sum(1 for _ in container.decode(video=0)) == 24
    running.phone.close()
    running.wait()


def test_refused_controls_answer_with_a_note_and_keep_the_session(receiver, plant):
    fmt, _ = plant
    running = Running(receiver)
    running.phone.hello()
    running.phone.state()  # the one after HELLO_ACK
    running.phone.control("record_start")
    refused = running.phone.state()
    assert refused["recording"] is False and "no video yet" in refused["note"]
    running.phone.control("record_stop")
    assert running.phone.state()["note"] == "not recording"
    running.phone.send(p.encode_video_format(fmt))
    wait_until(lambda: receiver.status().format == fmt)
    running.phone.send(p.encode_json(p.MessageType.CONTROL, {"t": 1, "cmd": "make_coffee"}))  # ignored, no answer
    running.phone.send(p.encode_json(p.MessageType.CONTROL, {"t": 1, "cmd": 7}))  # ignored too
    running.phone.control("record_start")
    started = running.phone.state()  # the first answer is to record_start: the others got none
    assert started["recording"] is True and started["note"] is None
    running.phone.control("record_start")
    again = running.phone.state(lambda body: body["note"] is not None)
    assert again["recording"] is True and again["note"] == "already recording"
    assert receiver.status().kind == "tcp" and receiver.busy
    receiver.stop_recording()
    assert running.phone.state(lambda body: not body["recording"])["note"] is None
    running.phone.close()
    assert running.wait() == "iPhone closed the connection"


def test_state_is_sent_every_tick_while_recording(receiver, plant, monkeypatch):
    monkeypatch.setattr(session_module, "TICK_INTERVAL_S", 0.05)
    fmt, _ = plant
    running = connected(receiver, fmt)
    receiver.start_recording()  # from another thread, like the web page
    states = [running.phone.state(lambda body: body["recording"]) for _ in range(4)]
    assert len({state["name"] for state in states}) == 1
    assert [state["seconds"] for state in states] == sorted(state["seconds"] for state in states)
    receiver.stop_recording()
    running.phone.state(lambda body: not body["recording"])
    running.phone.close()
    running.wait()


def test_rotation_and_failure_are_reported_with_a_note(receiver, plant, monkeypatch):
    fmt, packets = plant
    running = connected(receiver, fmt)
    first = receiver.start_recording()
    running.phone.send(b"".join(p.encode_video(packet) for packet in packets[:24]))
    wait_until(lambda: receiver.status().recording_frames == 24)
    running.phone.send(p.encode_video(packets[0]))  # the stream restarts: a keyframe at pts 0
    rotated = running.phone.state(lambda body: body["note"] is not None)
    assert rotated["recording"] is True and rotated["name"] != first.name and "timestamps went back" in rotated["note"]

    def disk_full(self, body, mac_time_ns):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("glassesdata_receiver.recorder.Recording.write_motion", disk_full)
    running.phone.send(p.encode_json(p.MessageType.MOTION, {"t": 1}))
    failed = running.phone.state(lambda body: not body["recording"])
    assert failed["name"] == rotated["name"] and "No space left" in failed["note"]
    running.phone.close()
    running.wait()


class ThreadCheckingSocket:
    """Wraps the receiver's end of the socketpair and records which threads write to it."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self.writers: set[str] = set()

    def sendall(self, data: bytes) -> None:
        self.writers.add(threading.current_thread().name)
        self._sock.sendall(data)

    def __getattr__(self, name):
        return getattr(self._sock, name)


def test_outbox_sends_posts_from_many_threads_on_the_session_thread(receiver):
    mac_side, phone_side = socket.socketpair()
    checked = ThreadCheckingSocket(mac_side)
    phone = Phone(phone_side)
    session = Session(checked, "tcp", "peer", receiver)
    thread = threading.Thread(target=session.run, name="session-thread", daemon=True)
    thread.start()
    phone.hello()
    phone.receive(p.MessageType.HELLO_ACK)

    def poster(worker: int) -> None:
        for index in range(50):
            session.post(p.encode_json(p.MessageType.CONTROL, {"worker": worker, "index": index}))

    posters = [threading.Thread(target=poster, args=(worker,), name=f"poster-{worker}") for worker in range(4)]
    for each in posters:
        each.start()
    for each in posters:
        each.join()
    seen: dict[int, list[int]] = {worker: [] for worker in range(4)}
    while sum(len(values) for values in seen.values()) < 200:
        body = phone.receive(p.MessageType.CONTROL).body
        seen[body["worker"]].append(body["index"])
    assert all(values == list(range(50)) for values in seen.values())  # each thread's order is kept
    assert checked.writers == {"session-thread"}
    phone.close()
    thread.join(3.0)
    session.post(b"ignored")  # after the end: dropped, no error


def test_close_wakes_the_session_at_once(receiver, monkeypatch):
    monkeypatch.setattr(session_module, "POLL_S", 5.0)  # without the wake-up, close would take this long
    running = Running(receiver)
    running.phone.hello()
    running.phone.receive(p.MessageType.HELLO_ACK)
    wait_until(lambda: receiver.busy)
    started = time.monotonic()
    running.session.close()
    assert running.wait() == "closed"
    assert time.monotonic() - started < 1.0


def test_video_queued_before_a_reconnect_is_never_shown(receiver, plant):
    fmt, packets = plant
    first = connected(receiver, fmt)
    first.phone.close()
    first.wait()
    again = Running(receiver)
    again.phone.hello()
    again.phone.receive(p.MessageType.HELLO_ACK)
    wait_until(lambda: receiver.status().kind == "tcp")
    # What the decoder could still hold from the first connection: dropped, never published.
    stale = receiver._epoch - 1
    receiver._decode_queue.put((stale, fmt))
    for packet in packets[:24]:
        receiver._decode_queue.put((stale, packet))
    time.sleep(0.5)
    assert receiver.latest_frame() is None and receiver._decode_queue.qsize() == 0
    again.phone.send(p.encode_video_format(fmt) + b"".join(p.encode_video(packet) for packet in packets[:24]))
    wait_until(lambda: receiver.latest_frame() is not None)
    again.phone.close()
    again.wait()
