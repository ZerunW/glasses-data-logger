"""The local web interface: one page with the live video, the status and the record button.

Standard library only (``http.server``). Endpoints (all JSON is UTF-8):

==========================  ======  ==============================================================
``/``                       GET     the page (``page.html``: inline CSS and JS, works offline)
``/api/status``             GET     status JSON; the page polls it 4 times a second
``/api/history``            GET     the last ``HISTORY_SECONDS`` of rates, delay and motion, sampled
                                    every ``HISTORY_INTERVAL_S`` (``?since=T``: only rows after T)
``/api/live.mjpg``          GET     MJPEG of the clean decoded frame; every part carries its hand
                                    landmarks in an ``X-Hands`` header (the page draws them)
``/api/stream.mjpg``        GET     MJPEG (``multipart/x-mixed-replace``): the decoded frame with the
                                    hand skeleton, at most the video's frame rate, at most 720 px tall
``/api/recordings``         GET     recordings in the output folder, newest first
``/api/pairing.png``        GET     the Wi-Fi pairing QR code (404 without ``--wifi``)
``/api/vcam/preview.jpg``   GET     the picture the virtual camera is sending, 480 px wide (404 while off)
``/api/record/start``       POST    start recording
``/api/record/stop``        POST    stop recording
``/api/hands/toggle``       POST    hand tracking on/off
``/api/vcam/start``         POST    start the virtual camera
``/api/vcam/stop``          POST    stop the virtual camera
``/api/vcam/settings``      POST    JSON body with any of ``mode`` ("fit"/"fill"), ``position`` (0..1),
                                    ``overlay`` (bool)
``/api/calibration``        GET     the camera calibration: saved screen scale, the current
                                    calibration, the capture session (views, coverage, guidance)
``/api/calibration/...``    GET     ``board.png`` (the board to show), ``preview.jpg`` (the last look,
                                    corners drawn), ``undistorted.jpg`` (the solution's check picture)
``/api/calibration/...``    POST    ``screen``, ``start``, ``capture``, ``solve``, ``save``, ``reset``
                                    (see ``calibration.CalibrationService``)
``/api/profile``            GET     body measurements, the form's fields and defaults
``/api/profile``            POST    JSON body ``{key: metres | null}``: save body measurements
==========================  ======  ==============================================================

POST requests must carry the header ``X-GlassesData: 1`` (a cross-site page cannot add
it without a CORS preflight, which this server never grants) and, if the browser
sends ``Origin``, an origin that matches ``Host``. They answer
``{"ok": bool, "message": str, ...}``.

Access: on a loopback host (the default, 127.0.0.1) there is no key, but the ``Host``
header must name a loopback address, which stops DNS-rebinding pages from reading the
status or the pairing code. On any other host every request needs the random key
printed at start (``?key=...`` once, then an HttpOnly, SameSite=Strict cookie).

A slow viewer never slows the receiver: one ``JpegFeed`` thread per stream kind encodes
the newest frame while someone watches, and each stream connection only ever takes the
latest JPEG (it skips the ones it was too slow for). A stalled connection blocks only its
own thread, and is dropped after ``REQUEST_TIMEOUT_S``.

Skeleton sync: a frame and the landmarks drawn on it travel in the same multipart part,
so they can never drift apart in the browser. When the hand tracker keeps up with the
video, the feed waits briefly (at most ``HANDS_WAIT_MAX_S``, only in its own thread) for
the result of exactly that frame; otherwise it pairs the frame with the newest earlier
result, as before, and ``X-Hands`` says how many frames the skeleton lags.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import math
import secrets
import select
import socket
import socketserver
import sys
import threading
import time
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

import cv2

from . import bodyprofile, paths
from .calibration import CalibrationService
from .decoder import DecodedFrame
from .receiver import HandsResult, Receiver
from .viewer import annotate, hands_for

if TYPE_CHECKING:
    from .vcam import VirtualCamera

log = logging.getLogger(__name__)

DEFAULT_PORT = 8765
STREAM_MAX_HEIGHT = 720
JPEG_QUALITY = 80
BOUNDARY = "gdframe"
REQUEST_TIMEOUT_S = 10.0  # idle keep-alive connections and stalled stream writes
STREAM_SEND_BUFFER = 256 * 1024  # bytes: a few frames, so a slow viewer sees little stale video
POST_HEADER = "X-GlassesData"
PAGE = Path(__file__).with_name("page.html")
MAX_RECORDINGS = 200
MAX_POST_BODY = 65536
HANDS_WAIT_MAX_S = 0.06  # longest a feed waits for the hand result of the frame it is about to send
HANDS_KEEPING_UP = 2  # frames: a tracker this close behind will reach the newest frame next
HISTORY_INTERVAL_S = 0.1
HISTORY_SECONDS = 30.0
HISTORY_FIELDS = (
    "t", "expected_fps", "glasses_fps", "sent_fps", "received_fps", "decoded_fps", "hands_fps",
    "dropped_frames", "display_skipped", "decode_errors", "delay_ms",
    "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz",
)


def is_loopback_host(host: str) -> bool:
    """True for addresses only this computer can reach: 127.0.0.0/8, ::1 and "localhost"."""
    host = host.strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # a name: it may resolve to anything, so treat it as reachable from outside


def _host_only(header: str) -> str:
    """The host name of a Host header value ("[::1]:8765" -> "::1", "a:80" -> "a")."""
    header = header.strip()
    if header.startswith("["):
        return header[1 : header.find("]")] if "]" in header else header
    return header.rsplit(":", 1)[0] if header.count(":") == 1 else header


def _settings_message(changes: dict[str, Any]) -> str:
    parts = []
    if "mode" in changes:
        parts.append("Fit: the whole picture" if changes["mode"] == "fit" else "Fill: a band of the picture")
    if "position" in changes:
        parts.append(f"band at {round(float(changes['position']) * 100)}% from the top")
    if "overlay" in changes:
        parts.append("hand skeleton " + ("on" if changes["overlay"] else "off"))
    return "Virtual camera: " + ", ".join(parts)


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a number")


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _clean(value: Any) -> Any:
    """JSON from the phone may hold NaN or Infinity (Python accepts them); browsers do not."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


# ---------------------------------------------------------------- data for the page


VCAM_NOT_SET_UP = {
    "available": False, "running": False, "device": None, "device_label": None, "requested_device": None,
    "loopback": None, "backend": None, "width": None, "height": None, "fps": None,
    "mode": None, "position": None, "overlay": False, "frames_sent": 0, "showing": None, "check": None,
    "last_error": "this receiver has no virtual camera output", "hint": None,
}


def status_payload(receiver: Receiver, wifi: dict[str, Any], vcam: "VirtualCamera | None" = None) -> dict[str, Any]:
    status = receiver.status()
    frame = receiver.latest_frame()
    hands = receiver.latest_hands()
    connected = bool(status.kind)
    visible = hands_for(frame.index, hands) if (frame is not None and status.hands_enabled) else []
    rates = status.phone_rates
    fmt = status.format
    recording = status.recording
    last = status.last_recording
    motion = status.motion
    return _clean({
        "time": time.time(),
        "link": {
            "connected": connected,
            "text": status.link,
            "kind": status.kind or None,
            "device": status.device or None,
            "sources": dict(status.sources),
        },
        "video": {
            "format": {"width": fmt.width, "height": fmt.height, "fps": fmt.fps} if fmt else None,
            "has_frame": connected and frame is not None,
            "glasses_fps": rates.get("glasses_frames"),
            "phone_sent_fps": rates.get("sent_frames"),
            "received_fps": status.received_fps,
            "decoded_fps": status.decoded_fps,
            "phone_dropped_frames": status.phone.get("dropped_frames"),
            "phone_dropped_fps": rates.get("dropped_frames"),
            "delay_ms": status.latency_ms,
            "rtt_ms": status.rtt_ms,
            "display_skipped": status.skipped_for_backlog,
            "decode_errors": status.decode_errors,
            "decode_backlog": status.decode_backlog,
            "preview_lag_s": _rounded(status.preview_lag_s, 3),
            # Newer phones: video frames waiting to be sent, whether they buffer instead of
            # dropping while this computer records, and drop episodes since connecting.
            "phone_queue_frames": _int_or_none(status.phone.get("queue_frames")),
            "phone_record_priority": status.phone.get("record_priority") if isinstance(status.phone.get("record_priority"), bool) else None,
            "phone_gaps": _int_or_none(status.phone.get("gaps")),
        },
        "hands": {
            "available": receiver.hands_available,
            "enabled": status.hands_enabled,
            "error": status.hands_error,
            "fps": status.hands_fps,
            "process_ms": hands.process_ms if hands is not None else None,
            "count": len(visible),
            "detected": [{"handedness": hand.handedness, "score": round(float(hand.score), 2)} for hand in visible],
        },
        "motion": {
            "accel": motion.get("a"),
            "gyro": motion.get("g"),
            "mag": motion.get("m"),
            "source": motion.get("src"),
            "samples_per_s": rates.get("motion_samples"),
        },
        "glasses": {
            "state": {key: status.device_state.get(key) for key in ("battery", "charging", "don", "hinge", "thermal", "link")}
            if status.device_state else None,
            "inputs": [
                {key: event.get(key) for key in ("type", "detail", "src", "t")} for event in status.inputs[:5]
            ],
        },
        "recording": {
            "active": recording is not None,
            "name": recording.name if recording is not None else None,
            "seconds": status.recording_s if recording is not None else 0.0,
            "frames": status.recording_frames if recording is not None else 0,
            "note": status.recording_note or None,
            "waiting_for_keyframe": recording is not None and status.recording_frames == 0,
            "last": {"name": last.name, "seconds": status.last_recording_s, "frames": status.last_recording_frames}
            if last is not None else None,
            "folder": str(receiver.out_root),
        },
        "wifi": wifi,
        "vcam": vcam.status() if vcam is not None else dict(VCAM_NOT_SET_UP),
    })


_frame_counts: dict[Path, tuple[int, int, int]] = {}  # frames.csv -> (size, mtime_ns, rows)
_frame_counts_lock = threading.Lock()


def _count_rows(path: Path) -> int | None:
    """Rows in an unfinished recording's frames.csv (cached until the file changes)."""
    try:
        stat = path.stat()
    except OSError:
        return None
    with _frame_counts_lock:
        cached = _frame_counts.get(path)
        if cached is not None and cached[:2] == (stat.st_size, stat.st_mtime_ns):
            return cached[2]
    rows = -1  # the header
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                rows += block.count(b"\n")
    except OSError:
        return None
    rows = max(rows, 0)
    with _frame_counts_lock:
        _frame_counts[path] = (stat.st_size, stat.st_mtime_ns, rows)
    return rows


def list_recordings(root: Path, active: Path | None = None, active_seconds: float = 0.0, active_frames: int = 0) -> list[dict[str, Any]]:
    """Recording folders (those with an info.json) under ``root``, newest first.

    ``state`` is "recording" (the one running now), "complete", or "interrupted" (the
    receiver stopped without closing it; video.mp4 is still playable up to its last
    fragment). ``seconds`` is the wall-clock length; null when unknown.
    """
    try:
        children = [child for child in root.iterdir() if child.is_dir()]
    except OSError:
        return []
    entries = []
    for folder in children:
        info_path = folder / "info.json"
        if not info_path.is_file():
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            info = {}
        if not isinstance(info, dict):
            info = {}
        size = 0
        try:
            for item in folder.iterdir():
                if item.is_file():
                    size += item.stat().st_size
        except OSError:
            pass
        started = info.get("startedAt")
        if active is not None and folder == active:
            state, seconds, frames = "recording", active_seconds, active_frames
        elif isinstance(info.get("endedMacTimeNs"), int) and isinstance(info.get("startedMacTimeNs"), int):
            state = "complete"
            seconds = (info["endedMacTimeNs"] - info["startedMacTimeNs"]) / 1e9
            frames = info.get("framesWritten")
        else:
            state, seconds, frames = "interrupted", None, _count_rows(folder / "frames.csv")
        entries.append({
            "name": folder.name, "started": started, "seconds": seconds, "frames": frames, "bytes": size, "state": state,
            "error": info.get("closeError") if isinstance(info.get("closeError"), str) else None,
        })

    def newest_first(entry: dict[str, Any]) -> tuple[float, str]:
        try:
            stamp = datetime.fromisoformat(entry["started"]).timestamp()
        except (TypeError, ValueError):
            stamp = 0.0
        return stamp, entry["name"]

    entries.sort(key=newest_first, reverse=True)
    return entries[:MAX_RECORDINGS]


# ---------------------------------------------------------------- history for the charts


def _rounded(value: Any, digits: int) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _vector(values: Any) -> list[float | None]:
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        return [None, None, None]
    return [_rounded(value, 4) for value in values[:3]]


class History(threading.Thread):
    """Samples rates, delay and motion every ``HISTORY_INTERVAL_S`` for the page's charts.

    Rows follow ``HISTORY_FIELDS``; values are None while nothing is connected. It only
    reads ``Receiver.status()`` (a snapshot), so it never holds up the link or the decoder.
    """

    def __init__(self, receiver: Receiver, stop: threading.Event) -> None:
        super().__init__(name="gd-web-history", daemon=True)
        self._receiver = receiver
        self._stop_event = stop
        self._rows: deque[list[Any]] = deque(maxlen=round(HISTORY_SECONDS / HISTORY_INTERVAL_S))
        self._lock = threading.Lock()

    def sample(self) -> list[Any]:
        status = self._receiver.status()
        now = round(time.time(), 3)
        if not status.kind:
            return [now] + [None] * (len(HISTORY_FIELDS) - 1)
        rates = status.phone_rates
        motion = status.motion
        return [
            now,
            status.format.fps if status.format else None,
            _rounded(rates.get("glasses_frames"), 2),
            _rounded(rates.get("sent_frames"), 2),
            _rounded(status.received_fps, 2),
            _rounded(status.decoded_fps, 2),
            _rounded(status.hands_fps, 2) if status.hands_enabled else None,
            status.phone.get("dropped_frames") if isinstance(status.phone.get("dropped_frames"), int) else None,
            status.skipped_for_backlog,
            status.decode_errors,
            _rounded(status.latency_ms, 1),
            *_vector(motion.get("a")),
            *_vector(motion.get("g")),
            *_vector(motion.get("m")),
        ]

    def rows(self, since: float | None = None) -> list[list[Any]]:
        with self._lock:
            rows = list(self._rows)
        return rows if since is None else [row for row in rows if row[0] > since]

    def run(self) -> None:
        next_at = time.monotonic()
        while not self._stop_event.wait(max(0.0, next_at - time.monotonic())):
            next_at = max(next_at + HISTORY_INTERVAL_S, time.monotonic())
            try:
                row = self.sample()
            except Exception:
                log.debug("history sample failed", exc_info=True)
                continue
            with self._lock:
                self._rows.append(row)


# ---------------------------------------------------------------- MJPEG producer


def hands_header(frame_index: int, hands: HandsResult | None, enabled: bool) -> str:
    """The ``X-Hands`` value for one stream part: compact JSON, ASCII only.

    ``points`` is flat ``[x0, y0, x1, y1, ...]`` (21 joints, normalised to the image);
    ``lag`` is how many frames older than the picture the landmarks are (0: same frame).
    """
    visible = hands_for(frame_index, hands) if enabled else []
    payload: dict[str, Any] = {"frame": frame_index, "enabled": enabled, "lag": None, "hands": []}
    if visible and hands is not None:
        payload["lag"] = frame_index - hands.frame_index
        payload["hands"] = [
            {
                "side": str(hand.handedness),
                "score": _rounded(hand.score, 2),
                "points": [_rounded(value, 4) for x, y, *_ in hand.landmarks for value in (x, y)],
            }
            for hand in visible
        ]
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


class JpegFeed(threading.Thread):
    """Encodes the newest frame as JPEG while at least one viewer watches.

    ``burn_in``: draw the skeleton into the picture (``/api/stream.mjpg``, the page's
    fallback, anything else that shows the stream as is); otherwise send the clean
    picture (``/api/live.mjpg``). Both carry the landmarks in an ``X-Hands`` part header.

    Never more often than the video's frame rate; viewers read ``latest`` and skip
    whatever they were too slow for. Nothing here ever waits for a viewer.
    """

    def __init__(self, receiver: Receiver, stop: threading.Event, burn_in: bool = True) -> None:
        super().__init__(name="gd-web-jpeg" if burn_in else "gd-web-live", daemon=True)
        self._receiver = receiver
        self._stop_event = stop
        self._burn_in = burn_in
        self._changed = threading.Condition()
        self._jpeg: bytes | None = None
        self._headers = b""
        self._sequence = 0
        self._viewers = 0
        self._refresh = False  # encode the current frame now, even if it was encoded before
        self.encoded = 0

    @property
    def viewers(self) -> int:
        with self._changed:
            return self._viewers

    def add_viewer(self) -> None:
        with self._changed:
            if self._viewers == 0:
                # The picture kept from the last time someone watched may be minutes old:
                # drop it and encode the current frame at once, whether or not it is new.
                self._jpeg = None
                self._refresh = True
            self._viewers += 1
            self._changed.notify_all()

    def remove_viewer(self) -> None:
        with self._changed:
            self._viewers -= 1

    def wake(self) -> None:
        with self._changed:
            self._changed.notify_all()

    def next_jpeg(self, after: int, timeout: float) -> tuple[int, bytes] | None:
        """The newest JPEG once its sequence number passes ``after``; None on timeout or stop."""
        part = self.next_part(after, timeout)
        return None if part is None else (part[0], part[1])

    def next_part(self, after: int, timeout: float) -> tuple[int, bytes, bytes] | None:
        """Like ``next_jpeg``, plus the extra part headers (``X-Frame``, ``X-Hands``) as bytes."""
        with self._changed:
            self._changed.wait_for(
                lambda: self._stop_event.is_set() or (self._jpeg is not None and self._sequence != after), timeout
            )
            if self._stop_event.is_set() or self._jpeg is None or self._sequence == after:
                return None
            return self._sequence, self._jpeg, self._headers

    def _hands_for(self, frame: DecodedFrame) -> HandsResult | None:
        """The hand result to pair with ``frame``: its own if the tracker delivers it in time.

        Waits (in this thread only) when the tracker is keeping up, i.e. it will process
        this frame next, for about two of its processing times; never longer than
        ``HANDS_WAIT_MAX_S``. Otherwise the newest earlier result (``hands_for`` hides it
        if it is too old). A result for a later frame is never used.
        """
        receiver = self._receiver
        hands = receiver.latest_hands()
        if hands is None or hands.frame_index >= frame.index or hands.frame_index < frame.index - HANDS_KEEPING_UP:
            return hands if hands is None or hands.frame_index <= frame.index else None
        deadline = time.monotonic() + min(HANDS_WAIT_MAX_S, 2 * hands.process_ms / 1000 + 0.005)
        while time.monotonic() < deadline and not self._stop_event.wait(0.003):
            newer = receiver.latest_hands()
            if newer is None:
                return None  # a reconnect cleared it
            if newer.frame_index > frame.index:
                break  # the tracker skipped this frame; keep the earlier result
            hands = newer
            if newer.frame_index == frame.index:
                break
        return hands

    def _encode(self, frame: DecodedFrame) -> tuple[bytes, bytes] | None:
        enabled = self._receiver.hands_active
        hands = self._hands_for(frame) if enabled else None
        image = annotate(frame, hands, enabled and self._burn_in, STREAM_MAX_HEIGHT)
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return None
        headers = f"X-Frame: {frame.index}\r\nX-Hands: {hands_header(frame.index, hands, enabled)}\r\n"
        return buffer.tobytes(), headers.encode("ascii")

    def run(self) -> None:
        last_index = -1
        last_encode = 0.0
        while not self._stop_event.is_set():
            with self._changed:
                self._changed.wait_for(lambda: self._viewers > 0 or self._stop_event.is_set(), timeout=0.5)
                if self._viewers <= 0:
                    continue
            if not self._receiver.busy:
                with self._changed:
                    self._jpeg = None  # never show a previous connection's last frame
                self._stop_event.wait(0.1)
                continue
            with self._changed:
                refresh, self._refresh = self._refresh, False
            frame = self._receiver.latest_frame() if refresh else None
            if frame is None:
                frame = self._receiver.wait_frame(last_index, timeout=0.5)
            if frame is None:
                continue
            fmt = self._receiver.video_format()
            interval = 1.0 / min(max(fmt.fps if fmt and fmt.fps else 30, 1), 120)
            delay = last_encode + interval - time.monotonic()
            if delay > 0:
                if self._stop_event.wait(delay):
                    break
                frame = self._receiver.latest_frame() or frame  # the newest after waiting
            last_index = frame.index
            last_encode = time.monotonic()
            try:
                encoded = self._encode(frame)
            except Exception:
                log.exception("Could not encode a stream frame")
                continue
            if encoded is not None:
                with self._changed:
                    self._jpeg, self._headers = encoded
                    self._sequence += 1
                    self.encoded += 1
                    self._changed.notify_all()


# ---------------------------------------------------------------- HTTP server


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    app: "WebServer"

    def server_bind(self) -> None:
        # HTTPServer.server_bind does a reverse DNS lookup (getfqdn), which can stall for
        # seconds on some networks. The name is only used in CGI variables.
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name, self.server_port = str(host), port

    def handle_error(self, request: Any, client_address: Any) -> None:
        log.debug("web request from %s failed", client_address, exc_info=True)


class _Server6(_Server):
    address_family = socket.AF_INET6


class WebServer:
    def __init__(
        self,
        receiver: Receiver,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        wifi: bool = False,
        pair_at_start: bool = False,
        require_key: bool | None = None,
        vcam: "VirtualCamera | None" = None,
        state_dir: Path | None = None,
    ) -> None:
        self.receiver = receiver
        self.vcam = vcam
        # calibration and body measurements live here (default: the receiver's state folder)
        self.state_dir = state_dir if state_dir is not None else receiver.state_dir
        self.calibration = CalibrationService(receiver, self.state_dir)
        self._profile_lock = threading.Lock()
        self.host = host
        self.loopback = is_loopback_host(host)
        needs_key = (not self.loopback) if require_key is None else require_key
        self.key: str | None = secrets.token_urlsafe(24) if needs_key else None
        self.wifi = wifi
        self.pairing_png: bytes | None = None  # set once the Wi-Fi listener runs (see __main__)
        self._pair_at_start = pair_at_start
        self._stop = threading.Event()
        server_class = _Server6 if ":" in host else _Server
        self._httpd = server_class((host, port), _Handler)
        self._httpd.app = self
        self.port: int = self._httpd.server_address[1]
        self.cookie_name = f"gd_key_{self.port}"  # cookies ignore ports: one per receiver
        self.feed = JpegFeed(receiver, self._stop, burn_in=True)  # /api/stream.mjpg
        self.live_feed = JpegFeed(receiver, self._stop, burn_in=False)  # /api/live.mjpg
        self.history = History(receiver, self._stop)
        self._streams: set[socket.socket] = set()
        self._streams_lock = threading.Lock()
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.2}, name="gd-web", daemon=True)

    def start(self) -> None:
        self.feed.start()
        self.live_feed.start()
        self.history.start()
        self._thread.start()

    def url(self, for_display: str | None = None) -> str:
        """The address to open. For a wildcard host, ``for_display`` (a LAN address) replaces it."""
        host = for_display or self.host
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"
        if ":" in host:
            host = f"[{host}]"
        url = f"http://{host}:{self.port}/"
        return f"{url}?key={self.key}" if self.key else url

    def wifi_info(self) -> dict[str, Any]:
        return {"enabled": self.wifi, "pairing_ready": self.pairing_png is not None, "pair_at_start": self._pair_at_start}

    def stop(self) -> None:
        self._stop.set()
        self.calibration.close()
        self.feed.wake()
        self.live_feed.wake()
        if self._thread.is_alive():
            self._httpd.shutdown()
        with self._streams_lock:
            streams = list(self._streams)
        for connection in streams:  # wake stream writers blocked on a stalled viewer
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._httpd.server_close()
        self._thread.join(2.0)
        for thread in (self.feed, self.live_feed, self.history):
            if thread.is_alive():
                thread.join(2.0)

    # used by the handler
    def stream_opened(self, connection: socket.socket) -> None:
        with self._streams_lock:
            self._streams.add(connection)

    def stream_closed(self, connection: socket.socket) -> None:
        with self._streams_lock:
            self._streams.discard(connection)

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def page(self) -> bytes:
        return PAGE.read_bytes()

    def profile(self) -> dict[str, Any]:
        path = paths.profile_path(self.state_dir) if self.state_dir is not None else None
        profile, note = bodyprofile.load(path) if path is not None else (bodyprofile.defaults(), None)
        return {
            "profile": profile,
            "fields": bodyprofile.fields(),
            "file": str(path) if path is not None else None,
            "saved": bool(path is not None and path.is_file()),
            "note": note,
        }

    def save_profile(self, changes: Any) -> tuple[int, dict[str, Any]]:
        if self.state_dir is None:
            return HTTPStatus.CONFLICT, {"ok": False, "message": "This receiver has no state folder to save to"}
        path = paths.profile_path(self.state_dir)
        with self._profile_lock:
            current, _ = bodyprofile.load(path)
            profile, errors = bodyprofile.update(current, changes)
            if errors:
                message = "Not saved: " + "; ".join(errors.values())
                return HTTPStatus.BAD_REQUEST, {"ok": False, "message": message, "errors": errors}
            bodyprofile.save(path, profile)
        count = sum(1 for key in bodyprofile.MEASURABLE if profile["measured"][key])
        return HTTPStatus.OK, {
            "ok": True, "message": f"Body measurements saved: {count} of {len(bodyprofile.MEASURABLE)} measured",
            **self.profile(),
        }


_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",  # the key may be in the URL
    "X-Frame-Options": "DENY",
    "Cache-Control": "no-store",
}
_PAGE_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


class _Handler(BaseHTTPRequestHandler):
    server: _Server
    _cookies_to_set: list[str] = []  # replaced per request by _allowed; never mutated here
    _responded = False  # a status line was sent for the current request
    protocol_version = "HTTP/1.1"
    server_version = "GlassesData"
    sys_version = ""
    timeout = REQUEST_TIMEOUT_S

    # ------------------------------------------------------------ plumbing

    @property
    def app(self) -> WebServer:
        return self.server.app

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the base class's name
        # Never log the query string: it may hold the key.
        log.debug("web %s %s", self.address_string(), urlsplit(self.path).path)

    def _send(self, status: int, body: bytes, content_type: str, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in {**_SECURITY_HEADERS, **(headers or {})}.items():
            self.send_header(name, value)
        for cookie in self._cookies_to_set:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload, separators=(",", ":")).encode(), "application/json; charset=utf-8")

    def _text(self, status: int, text: str) -> None:
        self._send(status, (text + "\n").encode(), "text/plain; charset=utf-8")

    # ------------------------------------------------------------ access control

    def _allowed(self) -> bool:
        """Sends the refusal itself and returns False when the request may not proceed."""
        self._cookies_to_set: list[str] = []
        app = self.app
        if app.loopback and not is_loopback_host(_host_only(self.headers.get("Host", ""))):
            self._text(HTTPStatus.FORBIDDEN, "Open this page through 127.0.0.1 or localhost.")
            return False
        if app.key is not None:
            query = parse_qs(urlsplit(self.path).query).get("key", [""])[0]
            cookie = ""
            try:
                jar = SimpleCookie(self.headers.get("Cookie", ""))
                if app.cookie_name in jar:
                    cookie = jar[app.cookie_name].value
            except CookieError:
                pass
            expected = app.key.encode()
            from_query = bool(query) and hmac.compare_digest(query.encode(), expected)
            if not (from_query or (cookie and hmac.compare_digest(cookie.encode(), expected))):
                self._text(HTTPStatus.FORBIDDEN, "Access key required: open the WEB_URL the receiver printed (it ends in ?key=...).")
                return False
            if from_query:
                self._cookies_to_set.append(f"{app.cookie_name}={app.key}; Path=/; HttpOnly; SameSite=Strict")
        return True

    def _same_origin_post(self) -> bool:
        if self.headers.get(POST_HEADER) != "1":
            self._json(HTTPStatus.FORBIDDEN, {"ok": False, "message": f"missing {POST_HEADER} header"})
            return False
        origin = self.headers.get("Origin")
        if origin is not None:
            parts = urlsplit(origin)
            if parts.scheme != "http" or parts.netloc.lower() != self.headers.get("Host", "").lower():
                self._json(HTTPStatus.FORBIDDEN, {"ok": False, "message": "cross-site request refused"})
                return False
        if self.headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none"):
            self._json(HTTPStatus.FORBIDDEN, {"ok": False, "message": "cross-site request refused"})
            return False
        return True

    # ------------------------------------------------------------ routes

    def send_response(self, code: int, message: str | None = None) -> None:
        self._responded = True  # from here on an error can only close the connection
        super().send_response(code, message)

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming
        self._responded = False
        try:
            self._get()
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            self.close_connection = True
        except Exception:
            self._internal_error()

    def do_POST(self) -> None:  # noqa: N802
        self._responded = False
        try:
            self._post()
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            self.close_connection = True
        except Exception:
            self._internal_error()

    def _internal_error(self) -> None:
        """An unexpected failure answers a JSON 500 (and is logged) instead of a dropped connection."""
        log.exception("web %s %s failed", self.command, urlsplit(self.path).path)
        if self._responded:
            self.close_connection = True
            return
        error = sys.exc_info()[1]
        try:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "message": f"Internal error ({type(error).__name__}): see the receiver's log"})
        except OSError:
            self.close_connection = True

    def _get(self) -> None:
        if not self._allowed():
            return
        app = self.app
        path = urlsplit(self.path).path
        if path == "/":
            if self._cookies_to_set:  # drop the key from the address bar once the cookie is set
                self._send(HTTPStatus.SEE_OTHER, b"", "text/plain", {"Location": "/"})
                return
            self._send(HTTPStatus.OK, app.page(), "text/html; charset=utf-8", {"Content-Security-Policy": _PAGE_CSP})
        elif path == "/api/status":
            self._json(HTTPStatus.OK, status_payload(app.receiver, app.wifi_info(), app.vcam))
        elif path == "/api/history":
            self._json(HTTPStatus.OK, self._history())
        elif path == "/api/recordings":
            status = app.receiver.status()
            recordings = list_recordings(app.receiver.out_root, status.recording, status.recording_s, status.recording_frames)
            self._json(HTTPStatus.OK, {"folder": str(app.receiver.out_root), "recordings": recordings})
        elif path == "/api/pairing.png":
            png = app.pairing_png if app.wifi else None
            if png is None:
                self._text(
                    HTTPStatus.NOT_FOUND,
                    "The Wi-Fi listener is not running." if app.wifi else "Wi-Fi pairing is off: start the receiver with --wifi.",
                )
            else:
                self._send(HTTPStatus.OK, png, "image/png")
        elif path == "/api/vcam/preview.jpg":
            jpeg = app.vcam.preview_jpeg() if app.vcam is not None else None
            if jpeg is None:
                self._text(HTTPStatus.NOT_FOUND, "The virtual camera is off.")
            else:
                self._send(HTTPStatus.OK, jpeg, "image/jpeg")
        elif path == "/api/calibration":
            self._json(HTTPStatus.OK, _clean(app.calibration.state()))
        elif path in ("/api/calibration/board.png", "/api/calibration/preview.jpg", "/api/calibration/undistorted.jpg"):
            data, kind, missing = {
                "/api/calibration/board.png": (app.calibration.board_png, "image/png", "No calibration is running."),
                "/api/calibration/preview.jpg": (app.calibration.preview_jpeg, "image/jpeg", "Nothing seen yet."),
                "/api/calibration/undistorted.jpg": (app.calibration.undistorted_jpeg, "image/jpeg", "Not solved yet."),
            }[path]
            body = data()
            if body is None:
                self._text(HTTPStatus.NOT_FOUND, missing)
            else:
                self._send(HTTPStatus.OK, body, kind)
        elif path == "/api/profile":
            self._json(HTTPStatus.OK, app.profile())
        elif path == "/api/stream.mjpg":
            self._stream(app.feed)
        elif path == "/api/live.mjpg":
            self._stream(app.live_feed)
        else:
            self._text(HTTPStatus.NOT_FOUND, "Not found.")

    def _post(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        body = b""
        if 0 < length <= MAX_POST_BODY:
            body = self.rfile.read(length)  # read it in any case, so keep-alive stays in step
        elif length:
            self.close_connection = True
        if not self._allowed() or not self._same_origin_post():
            return
        if urlsplit(self.path).path.startswith("/api/vcam/"):
            self._vcam_post(urlsplit(self.path).path, body, length)
            return
        if urlsplit(self.path).path.startswith("/api/calibration/") or urlsplit(self.path).path == "/api/profile":
            self._setup_post(urlsplit(self.path).path, body, length)
            return
        receiver = self.app.receiver
        path = urlsplit(self.path).path
        if path == "/api/record/start":
            already = receiver.status().recording
            folder = receiver.start_recording()
            if folder is None:
                self._json(HTTPStatus.CONFLICT, {"ok": False, "message": f"Cannot record: {receiver.why_not_recording()}"})
            else:
                message = f"Already recording to {folder.name}" if already else f"Recording to {folder.name}"
                self._json(HTTPStatus.OK, {"ok": True, "message": message, "name": folder.name})
        elif path == "/api/record/stop":
            folder = receiver.stop_recording()
            error = receiver.close_error(folder) if folder is not None else None
            if folder is None:
                self._json(HTTPStatus.OK, {"ok": True, "message": "Not recording"})
            elif error:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "message": f"{folder.name} was not closed cleanly: {error}", "name": folder.name})
            else:
                self._json(HTTPStatus.OK, {"ok": True, "message": f"Saved {folder.name}", "name": folder.name})
        elif path == "/api/hands/toggle":
            if not receiver.hands_available:
                self._json(HTTPStatus.CONFLICT, {"ok": False, "message": "Hand tracking is unavailable", "enabled": False})
            else:
                enabled = receiver.set_hands(not receiver.hands_active)
                self._json(HTTPStatus.OK, {"ok": True, "message": f"Hand tracking {'on' if enabled else 'off'}", "enabled": enabled})
        else:
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "message": "Not found"})

    def _vcam_post(self, path: str, body: bytes, length: int) -> None:
        vcam = self.app.vcam
        if path not in ("/api/vcam/start", "/api/vcam/stop", "/api/vcam/settings"):
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "message": "Not found"})
            return
        if vcam is None:
            self._json(HTTPStatus.CONFLICT, {"ok": False, "message": "This receiver has no virtual camera output", "vcam": dict(VCAM_NOT_SET_UP)})
            return
        if path == "/api/vcam/start":
            ok, message = vcam.start()
            self._json(HTTPStatus.OK if ok else HTTPStatus.CONFLICT, {"ok": ok, "message": message, "vcam": vcam.status()})
        elif path == "/api/vcam/stop":
            ok, message = vcam.stop()
            self._json(HTTPStatus.OK, {"ok": ok, "message": message, "vcam": vcam.status()})
        else:
            try:
                changes = self._settings_body(body, length)
                vcam.set(**changes)
            except ValueError as error:
                self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "message": f"Not changed: {error}", "vcam": vcam.status()})
                return
            self._json(HTTPStatus.OK, {"ok": True, "message": _settings_message(changes), "vcam": vcam.status()})

    def _setup_post(self, path: str, body: bytes, length: int) -> None:
        """Calibration and body measurements. Every body is a JSON object (or empty)."""
        service = self.app.calibration
        actions = {
            "/api/calibration/screen": service.save_screen,
            "/api/calibration/start": service.start,
            "/api/calibration/capture": service.capture,
            "/api/calibration/solve": lambda _: service.solve(),
            "/api/calibration/save": lambda _: service.save(),
            "/api/calibration/reset": lambda _: service.reset(),
            "/api/profile": self.app.save_profile,
        }
        action = actions.get(path)
        if action is None:
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "message": "Not found"})
            return
        try:
            data = self._json_body(body, length)
            status, payload = action(data)
        except ValueError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "message": f"Not done: {error}"})
            return
        except OSError as error:
            log.error("Could not save: %s", error)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "message": f"Could not save: {error.strerror or error}"})
            return
        self._json(status, _clean(payload))

    @staticmethod
    def _json_body(body: bytes, length: int) -> Any:
        """The body as JSON (None when empty)."""
        if length < 0 or length > MAX_POST_BODY:
            raise ValueError("the body is too large")
        if not body:
            return None
        try:
            return json.loads(body.decode("utf-8"), parse_constant=_refuse_constant)
        except (UnicodeDecodeError, ValueError):
            raise ValueError("the body is not JSON") from None

    @staticmethod
    def _settings_body(body: bytes, length: int) -> dict[str, Any]:
        """``{"mode": "fit"|"fill", "position": 0..1, "overlay": bool}``, any subset, nothing else."""
        if length < 0 or length > MAX_POST_BODY:
            raise ValueError("the body is too large")
        try:
            data = json.loads(body.decode("utf-8")) if body else None
        except (UnicodeDecodeError, ValueError):
            raise ValueError("the body is not JSON") from None
        if not isinstance(data, dict) or not data:
            raise ValueError('send a JSON object such as {"mode": "fill"}')
        unknown = sorted(set(data) - {"mode", "position", "overlay"})
        if unknown:
            raise ValueError("unknown setting " + ", ".join(repr(name) for name in unknown))
        # Types are checked here, before anything else touches the values (null, huge integers...).
        if "mode" in data and not isinstance(data["mode"], str):
            raise ValueError('mode must be "fit" or "fill"')
        if "overlay" in data and not isinstance(data["overlay"], bool):
            raise ValueError("overlay must be true or false")
        if "position" in data:
            position = data["position"]
            try:
                valid = isinstance(position, (int, float)) and not isinstance(position, bool) and 0.0 <= float(position) <= 1.0
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError("position must be a number from 0 (top) to 1 (bottom)")
        return data

    def _history(self) -> dict[str, Any]:
        since: float | None = None
        try:
            since = float(parse_qs(urlsplit(self.path).query).get("since", [""])[0])
        except ValueError:
            pass
        if since is not None and not math.isfinite(since):
            since = None
        return _clean({
            "interval_s": HISTORY_INTERVAL_S,
            "seconds": HISTORY_SECONDS,
            "fields": list(HISTORY_FIELDS),
            "rows": self.app.history.rows(since),
        })

    def _stream(self, feed: JpegFeed) -> None:
        app = self.app
        connection = self.connection
        try:
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, STREAM_SEND_BUFFER)
        except OSError:
            pass
        self.close_connection = True
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Connection", "close")
        for name, value in _SECURITY_HEADERS.items():
            self.send_header(name, value)
        self.send_header("Pragma", "no-cache")
        for cookie in self._cookies_to_set:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        app.stream_opened(connection)
        feed.add_viewer()
        try:
            sequence = -1
            while not app.stopping:
                latest = feed.next_part(sequence, timeout=1.0)
                if latest is None:
                    if self._viewer_left(connection):
                        break  # nothing was written for a while, so no write could have noticed
                    continue
                sequence, jpeg, extra = latest
                head = f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n".encode()
                self.wfile.write(head + extra + b"\r\n" + jpeg + b"\r\n")
        except OSError:
            pass  # the viewer left, stalled past the timeout, or the server is stopping
        finally:
            feed.remove_viewer()
            app.stream_closed(connection)

    @staticmethod
    def _viewer_left(connection: socket.socket) -> bool:
        """True once the viewer closed the connection (it never sends anything after the request)."""
        try:
            readable, _, _ = select.select([connection], [], [], 0)
            return bool(readable) and connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True
