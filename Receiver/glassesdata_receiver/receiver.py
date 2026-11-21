"""The receiver hub: owns the active session, decoding, hand tracking, recording and status.

Threads and the state each one touches:

* session thread (one per connection): ``claim``/``session_started``/``session_ended``,
  ``handle``, ``clock_sample`` and ``session_tick``. Writes every video packet to the
  recording and queues it for decoding; it never waits for the decoder or the hand tracker.
* ``gd-decode``: the only user of the decoder; publishes ``_latest_frame``.
* ``gd-hands``: the only user of the hand tracker; publishes ``_latest_hands``.
* the web server's threads, the main thread and the auto-record thread: ``status``,
  ``latest_*``, ``wait_frame``, ``start_recording``, ``stop_recording``,
  ``toggle_recording``, ``set_hands``, ``shutdown``.

Messages to the phone (RECORDER_STATE) are only ever queued with ``Session.post``;
the session's own thread sends them, so a TLS socket is never written from two threads.

Locks:

* ``_lock`` guards ``_session``, ``_format``, ``_recording``, ``_status``, ``_latest_hands``,
  ``_device_state``, ``_phone_previous``, ``_inputs`` and ``_latencies``.
* ``_frame_ready`` (a Condition with its own lock) guards ``_latest_frame`` and ``_latest_epoch``.

Stream epochs: ``_epoch`` grows (under ``_lock``) when a connection starts and when the
recording rotates because the glasses stream restarted. Every item on the decode queue
carries the epoch it was queued in; the decoder drops items from an older epoch, so
video still queued from the previous connection or stream is never shown, and hand
rows are only recorded for frames of the current epoch.
* ``Recording`` and ``RateMeter`` lock themselves.

Every recording write goes through ``_record`` with ``_lock`` held, so nothing lands
in a closed or replaced recording (``Recording`` also drops writes after ``close``),
and a write that fails stops the recording instead of the link or a worker thread.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import protocol as p
from .decoder import DecodedFrame, HevcDecoder
from .recorder import Recording
from .session import ClockSample, Session

log = logging.getLogger(__name__)

DECODE_BACKLOG_LIMIT = 48  # about two seconds at 24 fps
SHUTDOWN_TIMEOUT_S = 5.0
_STOP = object()  # wakes the decode thread at shutdown


class RateMeter:
    def __init__(self, window_s: float = 1.0) -> None:
        self._window_ns = int(window_s * 1e9)
        self._ticks: deque[int] = deque()
        self._lock = threading.Lock()

    def tick(self, now_ns: int | None = None) -> None:
        with self._lock:
            self._ticks.append(now_ns or time.monotonic_ns())

    def rate(self) -> float:
        now = time.monotonic_ns()
        with self._lock:
            while self._ticks and now - self._ticks[0] > self._window_ns:
                self._ticks.popleft()
            return len(self._ticks) * 1e9 / self._window_ns


@dataclass
class HandsResult:
    frame_index: int
    pts_us: int
    hands: list[Any]
    process_ms: float


@dataclass
class Status:
    """A snapshot for the viewer, safe to read without locks.

    The dict fields hold message bodies that are replaced, never mutated in place,
    so sharing them with the snapshot is safe.
    """

    link: str = "Waiting for an iPhone"
    sources: dict[str, str] = field(default_factory=dict)
    device: str = ""
    kind: str = ""
    format: p.VideoFormat | None = None
    received_fps: float = 0.0
    decoded_fps: float = 0.0
    hands_fps: float = 0.0
    phone: dict[str, Any] = field(default_factory=dict)
    phone_rates: dict[str, float] = field(default_factory=dict)
    latency_ms: float | None = None
    rtt_ms: float | None = None
    decode_backlog: int = 0
    skipped_for_backlog: int = 0
    decode_errors: int = 0
    motion: dict[str, Any] = field(default_factory=dict)
    device_state: dict[str, Any] = field(default_factory=dict)
    inputs: list[dict[str, Any]] = field(default_factory=list)
    recording: Path | None = None
    recording_s: float = 0.0
    recording_frames: int = 0
    recording_note: str = ""  # why the recording rotated or failed, if it did
    last_recording: Path | None = None
    last_recording_s: float = 0.0
    last_recording_frames: int = 0
    hands_enabled: bool = False
    hands_error: str | None = None


class Receiver:
    def __init__(
        self,
        out_root: Path,
        hand_tracker_factory: Callable[[], Any] | None = None,
        recording_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.out_root = out_root
        self._recording_metadata = dict(recording_metadata or {})  # added to every info.json
        self._lock = threading.RLock()
        self._session_gone = threading.Condition(self._lock)
        self._session: Session | None = None
        self._status = Status(hands_enabled=hand_tracker_factory is not None)
        self._format: p.VideoFormat | None = None
        self._recording: Recording | None = None
        self._auto_record_blocked = False  # after a manual stop or a failure, until the next session
        self._device_state: tuple[dict[str, Any], int] | None = None  # latest body and its mac_time_ns
        self._decode_queue: queue.Queue[Any] = queue.Queue()
        self._latest_frame: DecodedFrame | None = None
        self._latest_hands: HandsResult | None = None
        self._frame_ready = threading.Condition()
        self._epoch = 0
        self._latest_epoch = 0
        self._received = RateMeter()
        self._decoded = RateMeter()
        self._hands_meter = RateMeter()
        self._latencies: deque[float] = deque(maxlen=48)
        self._phone_previous: tuple[int, dict[str, Any]] | None = None
        self._inputs: deque[dict[str, Any]] = deque(maxlen=8)
        self._stop = threading.Event()
        self._decode_errors = 0  # written by the decode thread only
        self.hands_active = hand_tracker_factory is not None  # a plain bool flag, toggled from the web page
        self._hands_possible = hand_tracker_factory is not None
        # Daemon threads only as a last resort: shutdown() joins them, because MediaPipe and
        # PyAV must not run while the interpreter tears down (see shutdown).
        self._threads = [threading.Thread(target=self._decode_loop, name="gd-decode", daemon=True)]
        if hand_tracker_factory is not None:
            self._threads.append(
                threading.Thread(target=self._hands_loop, args=(hand_tracker_factory,), name="gd-hands", daemon=True)
            )
        for thread in self._threads:
            thread.start()

    # ------------------------------------------------------------ session lifecycle (network threads)

    def claim(self, session: Session) -> bool:
        with self._lock:
            if self._session is not None or self._stop.is_set():
                return False
            self._session = session
            return True

    @property
    def busy(self) -> bool:
        return self._session is not None

    def session_started(self, session: Session, hello: dict[str, Any]) -> None:
        with self._lock:
            self._status.link = f"Connected over {session.kind.upper()}"
            self._status.device = session.device_name
            self._status.kind = session.kind
            # Nothing from an earlier connection may leak into this one: the phone sends
            # VIDEO_FORMAT again, and its sensor state is unknown until it arrives.
            self._format = None
            self._status.format = None
            self._status.motion = {}
            self._status.device_state = {}
            self._device_state = None
            self._inputs.clear()
            self._latencies.clear()
            self._phone_previous = None
            self._status.rtt_ms = None
            self._latest_hands = None
            self._auto_record_blocked = False
            self._status.recording_note = ""
            self._epoch += 1  # drop video still queued from an earlier connection
            self._publish_state()  # the phone's first RECORDER_STATE, right after HELLO_ACK
        with self._frame_ready:
            self._latest_frame = None  # the page must not show the previous connection's last frame
        log.info("Connected: %s over %s (app %s)", session.device_name, session.kind, hello.get("app"))

    def session_ended(self, session: Session, reason: str) -> None:
        with self._lock:
            if self._session is not session:
                return
            self._session = None
            self._status.link = f"Disconnected: {reason}"
            self._status.device = ""
            self._status.kind = ""
            self._status.phone = {}
            self._status.phone_rates = {}
            self._close_recording()
            self._session_gone.notify_all()
        log.info("Disconnected: %s", reason)

    def set_source_status(self, kind: str, text: str) -> None:
        with self._lock:
            self._status.sources[kind] = text

    def session_tick(self, session: Session) -> None:
        """Called by the session loop once a second: RECORDER_STATE while recording."""
        with self._lock:
            if session is self._session and self._recording is not None:
                self._publish_state()

    # ------------------------------------------------------------ messages (session thread)

    def handle(self, session: Session, message: p.Message, mac_time_ns: int) -> None:
        kind = message.type
        if kind is p.MessageType.VIDEO:
            self._on_video(session, message.body, mac_time_ns)
        elif kind is p.MessageType.VIDEO_FORMAT:
            self._on_format(message.body)
        elif kind is p.MessageType.MOTION:
            with self._lock:
                self._status.motion = message.body
                self._record(lambda recording: recording.write_motion(message.body, mac_time_ns))
        elif kind is p.MessageType.INPUT:
            with self._lock:
                self._inputs.appendleft(message.body)
                self._record(lambda recording: recording.write_input(message.body, mac_time_ns))
        elif kind is p.MessageType.DEVICE_STATE:
            with self._lock:
                self._status.device_state = message.body
                self._device_state = (message.body, mac_time_ns)
                self._record(lambda recording: recording.write_device_state(message.body, mac_time_ns))
        elif kind is p.MessageType.STATS:
            self._on_stats(message.body, mac_time_ns)
        elif kind is p.MessageType.CONTROL:
            self._on_control(message.body)

    def _on_control(self, body: dict[str, Any]) -> None:
        """Remote record control from the phone. Never ends the session."""
        command = body.get("cmd")
        with self._lock:
            if command == p.CONTROL_RECORD_START:
                if self._recording is not None:
                    self._publish_state(note="already recording")
                elif self._format is None:
                    self._publish_state(note=self._why_not_recording())
                else:
                    log.info("Recording requested by the iPhone")
                    if self.start_recording() is None and self._stop.is_set():
                        # (a failed start already sent its state with the reason)
                        self._publish_state(note=self._why_not_recording())
            elif command == p.CONTROL_RECORD_STOP:
                if self._recording is None:
                    self._publish_state(note="not recording")
                else:
                    log.info("Recording stop requested by the iPhone")
                    self.stop_recording()
            else:
                log.debug("Ignoring CONTROL command %r", command)

    def clock_sample(self, session: Session, sample: ClockSample) -> None:
        with self._lock:
            self._status.rtt_ms = sample.rtt_ns / 1e6
            self._record(
                lambda recording: recording.write_clock(
                    sample.mac_send_ns, sample.mac_receive_ns, sample.phone_time_ns, sample.offset_ns
                )
            )

    def _on_format(self, fmt: p.VideoFormat) -> None:
        with self._lock:
            if fmt == self._format:
                return
            changed = self._format is not None
            self._format = fmt
            self._status.format = fmt
            if changed and self._recording is not None:
                self._rotate_recording("video format changed")
            # Queued under the lock so the decoder sees formats in the order they were accepted.
            self._decode_queue.put((self._epoch, fmt))

    def _on_video(self, session: Session, packet: p.VideoPacket, mac_time_ns: int) -> None:
        self._received.tick(mac_time_ns)
        mapped = session.clock.phone_to_mac(packet.phone_time_ns)  # the clock is only used on this thread
        with self._lock:
            if mapped is not None:
                self._latencies.append((mac_time_ns - mapped) / 1e6)
            if self._recording is not None and packet.keyframe:
                last_pts = self._recording.last_pts_us
                if last_pts is not None and packet.pts_us <= last_pts:
                    # The glasses stream restarted with the same format. MP4 timestamps must
                    # increase, so continue in a new recording that starts at this keyframe.
                    self._rotate_recording(f"video timestamps went back from {last_pts} to {packet.pts_us} us")
                    self._epoch += 1  # frames of the old stream must not add hand rows to the new recording
            self._record(lambda recording: recording.write_video(packet, mac_time_ns))
            epoch = self._epoch
        self._decode_queue.put((epoch, packet))

    def _on_stats(self, body: dict[str, Any], mac_time_ns: int) -> None:
        with self._lock:
            rates: dict[str, float] = {}
            if self._phone_previous is not None:
                previous_time, previous = self._phone_previous
                seconds = (mac_time_ns - previous_time) / 1e9
                if seconds > 0:
                    for key in ("glasses_frames", "sent_frames", "dropped_frames", "motion_samples"):
                        if isinstance(body.get(key), (int, float)) and isinstance(previous.get(key), (int, float)):
                            rates[key] = max(0.0, (body[key] - previous[key]) / seconds)
            self._phone_previous = (mac_time_ns, body)
            self._status.phone = body
            self._status.phone_rates = rates

    # ------------------------------------------------------------ decode and hands (worker threads)

    def _decode_loop(self) -> None:
        decoder: HevcDecoder | None = None
        skip_until_keyframe = False
        next_index = 0  # frame indices keep increasing across format changes
        epoch = 0
        while not self._stop.is_set():
            try:
                entry = self._decode_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if entry is _STOP:
                break
            item_epoch, item = entry
            if item_epoch < self._epoch:
                continue  # queued before a reconnect or a stream restart: stale
            if item_epoch != epoch:
                epoch = item_epoch
                skip_until_keyframe = True
            try:
                if isinstance(item, p.VideoFormat):
                    if decoder is not None:
                        next_index = decoder.next_index
                    decoder = HevcDecoder(item, next_index)
                    skip_until_keyframe = True
                    continue
                if decoder is None:
                    continue
                if self._decode_queue.qsize() > DECODE_BACKLOG_LIMIT:
                    # Too far behind to show live video: skip to the next keyframe. Recording is unaffected.
                    skip_until_keyframe = True
                    with self._lock:
                        self._status.skipped_for_backlog += 1
                if skip_until_keyframe:
                    if not item.keyframe:
                        continue
                    skip_until_keyframe = False
                errors_before = decoder.errors
                frames = decoder.decode(item)
                self._decode_errors += decoder.errors - errors_before
            except Exception:
                log.exception("Decoding failed")
                self._decode_errors += 1
                skip_until_keyframe = True
                continue
            for frame in frames:
                self._decoded.tick()
                with self._frame_ready:
                    if epoch != self._epoch:
                        break  # a reconnect happened while decoding
                    self._latest_frame = frame
                    self._latest_epoch = epoch
                    self._frame_ready.notify_all()

    def _hands_loop(self, tracker_factory: Callable[[], Any]) -> None:
        try:
            tracker = tracker_factory()
        except Exception as error:
            log.exception("Hand tracking unavailable")
            with self._lock:
                self._status.hands_error = str(error)
            self.hands_active = False
            return
        failures = 0
        last_index = -1
        try:
            while True:
                with self._frame_ready:
                    self._frame_ready.wait_for(
                        lambda: self._stop.is_set()
                        or (self._latest_frame is not None and self._latest_frame.index != last_index),
                        timeout=0.5,
                    )
                    frame = self._latest_frame
                    frame_epoch = self._latest_epoch
                if self._stop.is_set():
                    break
                if frame is None or frame.index == last_index:
                    continue
                last_index = frame.index
                if not self.hands_active:
                    continue
                started = time.perf_counter()
                try:
                    hands = tracker.detect(frame.image, frame.pts_us // 1000)
                except Exception:
                    failures += 1
                    if failures == 1:
                        log.exception("Hand tracking failed on a frame")
                    continue
                result = HandsResult(frame.index, frame.pts_us, hands, (time.perf_counter() - started) * 1000)
                self._hands_meter.tick()
                with self._lock:
                    if frame_epoch != self._epoch:
                        continue  # the frame belongs to an earlier connection or stream
                    self._latest_hands = result
                    self._record(lambda recording: recording.write_hands(frame.pts_us, frame.phone_time_ns, hands))
        finally:
            tracker.close()

    # ------------------------------------------------------------ viewer API (main thread)

    def latest_frame(self) -> DecodedFrame | None:
        with self._frame_ready:
            return self._latest_frame

    def wait_frame(self, after_index: int, timeout: float) -> DecodedFrame | None:
        """The latest frame once its index differs from ``after_index``; None on timeout or shutdown."""
        with self._frame_ready:
            self._frame_ready.wait_for(
                lambda: self._stop.is_set() or (self._latest_frame is not None and self._latest_frame.index != after_index),
                timeout=timeout,
            )
            frame = self._latest_frame
        if self._stop.is_set() or frame is None or frame.index == after_index:
            return None
        return frame

    def video_format(self) -> p.VideoFormat | None:
        with self._lock:
            return self._format

    @property
    def hands_available(self) -> bool:
        """A hand tracker exists and has not failed."""
        with self._lock:
            return self._hands_possible and self._status.hands_error is None

    def set_hands(self, enabled: bool) -> bool:
        """Turn live hand tracking on or off. Returns the new state (False when unavailable)."""
        self.hands_active = bool(enabled) and self.hands_available
        return self.hands_active

    def latest_hands(self) -> HandsResult | None:
        with self._lock:
            return self._latest_hands

    def status(self) -> Status:
        with self._lock:
            status = Status(**vars(self._status))
            status.sources = dict(self._status.sources)
            status.inputs = list(self._inputs)
            latencies = sorted(self._latencies)
            if self._recording is not None:
                status.recording = self._recording.directory
                status.recording_s = self._recording.duration_s
                status.recording_frames = self._recording.frames_written
        status.received_fps = self._received.rate()
        status.decoded_fps = self._decoded.rate()
        status.hands_fps = self._hands_meter.rate()
        status.latency_ms = latencies[len(latencies) // 2] if latencies else None
        status.decode_backlog = self._decode_queue.qsize()
        status.decode_errors = self._decode_errors
        status.hands_enabled = self.hands_active
        return status

    def toggle_recording(self) -> str:
        """Start or stop. Returns a message for people; logging is left to the helpers."""
        with self._lock:
            if self._recording is not None:
                return f"Saved {self.stop_recording()}"
            path = self.start_recording()
            if path is None:
                return f"Could not start recording: {self._why_not_recording()}"
            return f"Recording to {path}"

    def start_recording(self, auto: bool = False) -> Path | None:
        """Start a recording unless one is running. Returns its folder, or None if it cannot start
        (``why_not_recording`` says why).

        ``auto`` (the --record loop) also respects a manual stop or an earlier failure; a
        manual start lifts that block.
        """
        with self._lock:
            if self._recording is not None:
                return self._recording.directory
            if self._session is None or self._format is None or self._stop.is_set():
                return None
            if auto and self._auto_record_blocked:
                return None
            if not auto:
                self._auto_record_blocked = False
            self._start_recording()
            self._publish_state()  # started, or failed with the reason in the note
            return self._recording.directory if self._recording is not None else None

    def stop_recording(self) -> Path | None:
        """A manual stop: --record will not start again until the next connection or a manual start.
        Returns the saved folder, or None if nothing was recording."""
        with self._lock:
            if self._recording is None:
                return None
            self._auto_record_blocked = True
            self._status.recording_note = ""
            path = self._close_recording()
            self._publish_state()
            return path

    def why_not_recording(self) -> str:
        with self._lock:
            return self._why_not_recording()

    def shutdown(self) -> None:
        """Stop everything and wait for it, so no worker runs during interpreter teardown.

        MediaPipe dispatches every call through a ThreadPoolExecutor, and at exit
        ``concurrent.futures`` refuses new work, so a hand-tracking thread still running
        then raises; PyAV in a dying daemon thread can crash outright.
        """
        self._stop.set()
        with self._lock:
            session = self._session
        if session is not None:
            session.close()
        self._decode_queue.put(_STOP)
        with self._frame_ready:
            self._frame_ready.notify_all()
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                log.warning("%s did not stop within %.0f s", thread.name, SHUTDOWN_TIMEOUT_S)
        with self._lock:
            # The session thread notices close() within one poll interval and ends the session.
            self._session_gone.wait_for(lambda: self._session is None, max(0.0, deadline - time.monotonic()))
            self._close_recording()

    # ------------------------------------------------------------ recording helpers (lock held)

    def _start_recording(self) -> None:
        session = self._session
        metadata = {
            **self._recording_metadata,
            "link": session.kind if session else None,
            "device": session.device_name if session else None,
            "app": session.hello.get("app") if session else None,
        }
        try:
            self._recording = Recording(self.out_root, self._format, metadata)
        except Exception as error:
            self._recording_failed(error)
            return
        self._status.recording_note = ""
        log.info("Recording to %s", self._recording.directory)
        if self._device_state is not None:
            # The phone need not resend DEVICE_STATE, so start with the state already known.
            body, mac_time_ns = self._device_state
            self._record(lambda recording: recording.write_device_state(body, mac_time_ns))

    def _rotate_recording(self, reason: str) -> None:
        self._close_recording()
        self._start_recording()
        if self._recording is not None:
            log.info("%s; continuing in %s", reason[0].upper() + reason[1:], self._recording.directory.name)
            self._status.recording_note = f"new recording: {reason}"
        self._publish_state()

    def _record(self, write: Callable[[Recording], None]) -> None:
        """Every recording write goes through here: a failing disk stops the recording,
        never the link, the viewer or a worker thread."""
        if self._recording is None:
            return
        try:
            write(self._recording)
        except Exception as error:
            self._recording_failed(error)
            self._close_recording()
            self._publish_state()

    def _recording_failed(self, error: Exception) -> None:
        log.error("Recording failed: %s", error, exc_info=error)
        self._status.link = f"Recording stopped: {error}"
        self._status.recording_note = f"recording stopped: {error}"
        self._auto_record_blocked = True  # --record must not retry in a loop

    def _why_not_recording(self) -> str:
        if self._recording is not None:
            return "already recording"
        if self._stop.is_set():
            return "the receiver is shutting down"
        if self._session is None:
            return "no iPhone connected"
        if self._format is None:
            return "no video yet: start the glasses preview in the app"
        return self._status.recording_note or "the recording could not start"

    def _recorder_state(self, note: str | None = None) -> dict[str, Any]:
        """The RECORDER_STATE body (see protocol.py)."""
        recording = self._recording
        if recording is not None:
            name, seconds, frames = recording.directory.name, recording.duration_s, recording.frames_written
        else:
            last = self._status.last_recording
            name = last.name if last is not None else None
            seconds, frames = self._status.last_recording_s, self._status.last_recording_frames
        return {
            "recording": recording is not None,
            "name": name,
            "seconds": round(float(seconds), 3),
            "frames": int(frames),
            "note": note or self._status.recording_note or None,
        }

    def _publish_state(self, note: str | None = None) -> None:
        """Queue a RECORDER_STATE for the connected phone, if any (lock held)."""
        session = self._session
        if session is None:
            return
        try:
            session.post(p.encode_json(p.MessageType.RECORDER_STATE, self._recorder_state(note)))
        except p.ProtocolError as error:  # pragma: no cover - the body is always small, valid JSON
            log.error("Cannot encode RECORDER_STATE: %s", error)

    def _close_recording(self) -> Path | None:
        recording, self._recording = self._recording, None
        if recording is None:
            return None
        try:
            recording.close()
        except Exception as error:
            self._recording_failed(error)
        self._status.last_recording = recording.directory
        self._status.last_recording_s = recording.duration_s
        self._status.last_recording_frames = recording.frames_written
        log.info("Saved %s", recording.directory)
        return recording.directory
