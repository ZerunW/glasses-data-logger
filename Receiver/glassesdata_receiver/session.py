"""One connected iPhone: handshake, message dispatch, outgoing messages and phone/Mac clock estimation."""

from __future__ import annotations

import hmac
import logging
import select
import shlex
import socket
import ssl
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import __version__
from . import protocol as p

if TYPE_CHECKING:
    from .receiver import Receiver

log = logging.getLogger(__name__)

PING_INTERVAL_S = 1.0
HELLO_TIMEOUT_S = 5.0
IDLE_TIMEOUT_S = 6.0  # the phone sends STATS every second, so silence this long means a dead link
POLL_S = 0.25
TICK_INTERVAL_S = 1.0  # Receiver.session_tick: RECORDER_STATE once a second while recording
OUTBOX_LIMIT = 256  # messages; the oldest are dropped if the phone stops reading
MACHINE_INFO = "/etc/machine-info"


@dataclass(frozen=True)
class ClockSample:
    mac_send_ns: int
    mac_receive_ns: int
    phone_time_ns: int

    @property
    def rtt_ns(self) -> int:
        return self.mac_receive_ns - self.mac_send_ns

    @property
    def offset_ns(self) -> int:
        """phone clock minus Mac clock, assuming the reply was stamped half-way."""
        return self.phone_time_ns - (self.mac_send_ns + self.mac_receive_ns) // 2


class ClockEstimator:
    """Keeps the recent sample with the smallest round trip: the least queueing error."""

    def __init__(self, window: int = 30) -> None:
        self._samples: deque[ClockSample] = deque(maxlen=window)

    def add(self, sample: ClockSample) -> None:
        if sample.rtt_ns >= 0:
            self._samples.append(sample)

    @property
    def best(self) -> ClockSample | None:
        return min(self._samples, key=lambda s: s.rtt_ns, default=None)

    def phone_to_mac(self, phone_time_ns: int) -> int | None:
        best = self.best
        return None if best is None else phone_time_ns - best.offset_ns


def computer_name(machine_info: str = MACHINE_INFO) -> str:
    """This computer's display name: in every HELLO_ACK and in the Wi-Fi pairing code.

    macOS: the host name without ".local" (unchanged from earlier versions). Linux: the
    "pretty" name set in Settings > About (systemd's PRETTY_HOSTNAME), else the host
    name without its domain.
    """
    host = socket.gethostname()
    if sys.platform == "darwin":
        return host.removesuffix(".local") or "Mac"
    pretty = _pretty_hostname(machine_info)
    if pretty:
        return pretty
    return host.split(".")[0] or "Computer"


def _pretty_hostname(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.strip().partition("=")
                if key == "PRETTY_HOSTNAME" and value:
                    words = shlex.split(value)
                    return words[0].strip() if words else ""
    except (OSError, ValueError, UnicodeDecodeError):
        pass
    return ""


mac_name = computer_name  # the earlier name


class SessionEnded(Exception):
    pass


class Session:
    """All socket I/O happens on the thread that calls ``run``.

    A TLS connection must not be read and written from two threads at once, so the
    same loop receives data, sends the periodic PING and drains the outbox. Other
    threads send to the phone with ``post``, which only queues the bytes and wakes
    the loop.
    """

    def __init__(
        self, sock: socket.socket, kind: str, peer: str, receiver: Receiver, token: str | None = None, name: str | None = None
    ) -> None:
        self.sock = sock
        self.kind = kind  # "usb", "wifi" or "tcp"
        self.peer = peer
        self.hello: dict[str, Any] = {}
        self.clock = ClockEstimator()
        self._receiver = receiver
        self._token = token
        self._name = name or mac_name()
        self._closed = threading.Event()
        self._reader = p.FrameReader()
        self._outbox: deque[bytes] = deque(maxlen=OUTBOX_LIMIT)
        self._outbox_lock = threading.Lock()
        # Wakes the loop's select() when a message is posted or the session is closed.
        self._wake_reader, self._wake_writer = socket.socketpair()
        self._wake_reader.setblocking(False)
        self._wake_writer.setblocking(False)

    @property
    def device_name(self) -> str:
        return str(self.hello.get("name") or self.hello.get("device") or self.peer)

    def close(self) -> None:
        """Ask the session to end. Safe from any thread; the run loop closes the socket."""
        self._closed.set()
        self._wake()

    def post(self, frame: bytes) -> None:
        """Queue an encoded frame for the phone. Safe from any thread, never blocks.

        The session's own loop sends it, in order, after the HELLO_ACK. Dropped
        silently once the session has ended.
        """
        if self._closed.is_set():
            return
        with self._outbox_lock:
            self._outbox.append(frame)
        self._wake()

    def _wake(self) -> None:
        try:
            self._wake_writer.send(b"\0")
        except OSError:
            pass  # buffer full (the loop is already due to wake) or closed

    def _drain_outbox(self) -> None:
        while True:
            with self._outbox_lock:
                if not self._outbox:
                    return
                frame = self._outbox.popleft()
            self._send(frame)

    def _send(self, frame: bytes) -> None:
        try:
            self.sock.sendall(frame)
        except BrokenPipeError as error:
            # The phone closed its end; the close may simply not have been read yet.
            raise SessionEnded("iPhone closed the connection") from error

    def run(self) -> str:
        """Blocks until the connection ends. Returns the reason."""
        reason = "closed"
        claimed = False
        try:
            self.sock.settimeout(HELLO_TIMEOUT_S)
            hello = self._read_hello()
            if not self._receiver.claim(self):
                self._ack(ok=False, error="busy: another iPhone is connected", code="busy")
                return "rejected: busy"
            claimed = True
            self._ack(ok=True)
            self._receiver.session_started(self, hello)
            self._loop()
        except SessionEnded as ended:
            reason = str(ended)
        except (socket.timeout, TimeoutError):
            reason = "connection timed out"
        except p.ProtocolError as error:
            reason = f"protocol error: {error}"
        except PermissionError as error:
            reason = str(error)
        except OSError as error:
            reason = "connection lost"
            log.debug("session socket error: %s", error)
        finally:
            self._closed.set()
            for sock in (self.sock, self._wake_reader, self._wake_writer):
                try:
                    sock.close()
                except OSError:
                    pass
            with self._outbox_lock:
                self._outbox.clear()
            if claimed:
                self._receiver.session_ended(self, reason)
        return reason

    def _loop(self) -> None:
        last_receive = time.monotonic()
        next_ping = time.monotonic() + 0.2  # first clock sample quickly
        next_tick = time.monotonic() + TICK_INTERVAL_S
        # Frames that arrived in the same chunk as HELLO are still buffered in the reader.
        mac_time_ns = time.monotonic_ns()
        for message_type, payload in self._reader.feed(b""):
            self._dispatch(message_type, payload, mac_time_ns)
        while not self._closed.is_set():
            self._drain_outbox()  # first round: what session_started posted, right after HELLO_ACK
            now = time.monotonic()
            if now >= next_ping:
                self._send(p.encode_ping(p.Ping(time.monotonic_ns())))
                next_ping = now + PING_INTERVAL_S
            if now >= next_tick:
                next_tick = now + TICK_INTERVAL_S
                self._receiver.session_tick(self)
                self._drain_outbox()
            if now - last_receive > IDLE_TIMEOUT_S:
                raise SessionEnded("connection timed out")
            data = self._receive(POLL_S)
            if data is None:
                continue
            if not data:
                raise SessionEnded("iPhone closed the connection")
            last_receive = time.monotonic()
            mac_time_ns = time.monotonic_ns()
            for message_type, payload in self._reader.feed(data):
                self._dispatch(message_type, payload, mac_time_ns)

    def _receive(self, timeout: float) -> bytes | None:
        """One chunk, or None when nothing arrived within ``timeout`` (or the loop was woken)."""
        pending = self.sock.pending() if isinstance(self.sock, ssl.SSLSocket) else 0
        if not pending:
            readable, _, _ = select.select([self.sock, self._wake_reader], [], [], timeout)
            if self._wake_reader in readable:
                try:
                    while self._wake_reader.recv(4096):
                        pass
                except (BlockingIOError, InterruptedError):
                    pass
            if self.sock not in readable:
                return None
        try:
            return self.sock.recv(1 << 16)
        except ssl.SSLWantReadError:
            return None  # a partial TLS record; the rest is still in flight

    def _read_hello(self) -> dict[str, Any]:
        deadline = time.monotonic() + HELLO_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._closed.is_set():
                raise SessionEnded("closed")
            data = self._receive(POLL_S)
            if data is None:
                continue
            if not data:
                raise SessionEnded("closed before HELLO")
            for message_type, payload in self._reader.feed(data):
                message = p.decode_message(message_type, payload)
                if message is None or message.type is not p.MessageType.HELLO:
                    raise p.ProtocolError("first message must be HELLO")
                hello = message.body
                if hello.get("protocol") != p.PROTOCOL_VERSION:
                    self._ack(ok=False, error=f"protocol {p.PROTOCOL_VERSION} required", code="protocol")
                    raise p.ProtocolError(f"unsupported protocol {hello.get('protocol')!r}")
                # Compare bytes: compare_digest raises TypeError on non-ASCII str.
                presented = str(hello.get("token", "")).encode()
                if self._token is not None and not hmac.compare_digest(presented, self._token.encode()):
                    self._ack(ok=False, error="not paired with this Mac", code="pairing")
                    raise PermissionError("rejected: wrong pairing token")
                self.hello = hello
                return hello
        raise SessionEnded("no HELLO received")

    def _ack(self, ok: bool, error: str | None = None, code: str | None = None) -> None:
        """``code`` is for the app's logic ("pairing", "busy", "protocol"); ``error`` is for people."""
        body: dict[str, Any] = {
            "protocol": p.PROTOCOL_VERSION, "receiver": f"glassesdata-receiver {__version__}", "ok": ok, "name": self._name,
        }
        if error:
            body["error"] = error
        if code:
            body["code"] = code
        try:
            self.sock.sendall(p.encode_json(p.MessageType.HELLO_ACK, body))
        except OSError:
            pass

    def _dispatch(self, message_type: int, payload: bytes, mac_time_ns: int) -> None:
        message = p.decode_message(message_type, payload)
        if message is None:
            return
        if message.type is p.MessageType.PONG:
            sample = ClockSample(message.body.mac_time_ns, mac_time_ns, message.body.phone_time_ns)
            self.clock.add(sample)
            self._receiver.clock_sample(self, sample)
            return
        self._receiver.handle(self, message, mac_time_ns)
