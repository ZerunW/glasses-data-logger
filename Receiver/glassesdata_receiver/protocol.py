"""GDL1: the GlassesData link protocol between the iPhone app and this receiver.

Every message is framed as::

    u32 big-endian length | u8 type | payload (length - 1 bytes)

``length`` counts the type byte and the payload. The iOS implementation lives in
``CameraAccess/Streaming/LinkProtocol.swift``; both sides are tested against the
same byte fixtures (see ``tests/test_protocol.py`` and ``LinkProtocolTests.swift``).

Phone -> receiver: HELLO, VIDEO_FORMAT, VIDEO, MOTION, INPUT, DEVICE_STATE, STATS, PONG, CONTROL.
Receiver -> phone: HELLO_ACK, PING, RECORDER_STATE.

STATS (0x30, phone -> receiver, once a second) is a JSON object of counters: ``t``,
``glasses_frames``, ``sent_frames``, ``dropped_frames``, ``motion_samples``, ``queue_bytes``
and, from newer phones, ``queue_frames`` (video frames not sent yet), ``record_priority``
(true while the phone buffers video instead of dropping it because the receiver records)
and ``gaps`` (drop episodes since connecting). The receiver ignores keys it does not know.

Remote record control (JSON bodies, like the other JSON messages):

* ``CONTROL`` (0x50, phone -> receiver)::

      {"t": <phone ns>, "cmd": "record_start" | "record_stop"}

  The receiver ignores unknown ``cmd`` values (and does not answer them), so newer
  phones stay compatible. ``record_start`` while already recording, or before the
  first VIDEO_FORMAT, does not end the session: the receiver answers with its
  current RECORDER_STATE and a ``note`` that says why nothing started. The same
  holds for ``record_stop`` while not recording.

* ``RECORDER_STATE`` (0x51, receiver -> phone)::

      {"recording": <bool>, "name": <recording folder name or null>,
       "seconds": <float>, "frames": <int>, "note": <string or null>}

  Sent right after a successful HELLO_ACK, on every change (start, stop, rotation
  to a new folder, failure), and once a second while recording. ``name`` is the
  folder of the current recording, or of the last one after a stop (``recording``
  false); ``seconds`` and ``frames`` describe that recording (0 before the first).
  ``note`` is a human-readable reason for a rotation, a failure or a refused
  command, otherwise null. The phone should treat every RECORDER_STATE as the
  complete current state.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Iterator

PROTOCOL_VERSION = 1
DEFAULT_PORT = 7900
MAX_FRAME_LENGTH = 4 * 1024 * 1024
BONJOUR_TYPE = "_glassesdata._tcp.local."

_LENGTH = struct.Struct(">I")
_VIDEO_FORMAT = struct.Struct(">HHH")  # width, height, fps; hvcC follows
_VIDEO = struct.Struct(">BqQ")  # flags, pts_us (signed), phone_time_ns; sample data follows
_PING = struct.Struct(">Q")  # mac_time_ns
_PONG = struct.Struct(">QQ")  # echoed mac_time_ns, phone_time_ns

VIDEO_FLAG_KEYFRAME = 0x01


class MessageType(IntEnum):
    HELLO = 0x01
    HELLO_ACK = 0x02
    VIDEO_FORMAT = 0x10
    VIDEO = 0x11
    MOTION = 0x20
    INPUT = 0x21
    DEVICE_STATE = 0x22
    STATS = 0x30
    PING = 0x40
    PONG = 0x41
    CONTROL = 0x50
    RECORDER_STATE = 0x51


JSON_TYPES = frozenset(
    {
        MessageType.HELLO,
        MessageType.HELLO_ACK,
        MessageType.MOTION,
        MessageType.INPUT,
        MessageType.DEVICE_STATE,
        MessageType.STATS,
        MessageType.CONTROL,
        MessageType.RECORDER_STATE,
    }
)

CONTROL_RECORD_START = "record_start"
CONTROL_RECORD_STOP = "record_stop"


class ProtocolError(Exception):
    """The peer sent bytes that are not valid GDL1."""


@dataclass(frozen=True)
class VideoFormat:
    width: int
    height: int
    fps: int
    hvcc: bytes


@dataclass(frozen=True)
class VideoPacket:
    keyframe: bool
    pts_us: int
    phone_time_ns: int
    data: bytes  # length-prefixed (4-byte big-endian) HEVC NAL units, as in an MP4 sample


@dataclass(frozen=True)
class Ping:
    mac_time_ns: int


@dataclass(frozen=True)
class Pong:
    mac_time_ns: int
    phone_time_ns: int


@dataclass(frozen=True)
class Message:
    type: MessageType
    body: Any  # dict for JSON types, otherwise one of the dataclasses above


# ---------------------------------------------------------------- framing


def encode_frame(message_type: int, payload: bytes) -> bytes:
    length = len(payload) + 1
    if length > MAX_FRAME_LENGTH:
        raise ProtocolError(f"frame of {length} bytes exceeds {MAX_FRAME_LENGTH}")
    return _LENGTH.pack(length) + bytes([message_type]) + payload


class FrameReader:
    """Incremental decoder: feed it bytes as they arrive, iterate complete frames."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> Iterator[tuple[int, bytes]]:
        self._buffer += data
        while len(self._buffer) >= 4:
            (length,) = _LENGTH.unpack_from(self._buffer)
            if length < 1 or length > MAX_FRAME_LENGTH:
                raise ProtocolError(f"invalid frame length {length}")
            if len(self._buffer) < 4 + length:
                return
            message_type = self._buffer[4]
            payload = bytes(self._buffer[5 : 4 + length])
            del self._buffer[: 4 + length]
            yield message_type, payload

    @property
    def pending_bytes(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------- encoders


def encode_json(message_type: MessageType, body: dict[str, Any]) -> bytes:
    if message_type not in JSON_TYPES:
        raise ProtocolError(f"{message_type!r} is not a JSON message")
    payload = json.dumps(body, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()
    return encode_frame(message_type, payload)


def encode_video_format(fmt: VideoFormat) -> bytes:
    return encode_frame(MessageType.VIDEO_FORMAT, _VIDEO_FORMAT.pack(fmt.width, fmt.height, fmt.fps) + fmt.hvcc)


def encode_video(packet: VideoPacket) -> bytes:
    flags = VIDEO_FLAG_KEYFRAME if packet.keyframe else 0
    header = _VIDEO.pack(flags, packet.pts_us, packet.phone_time_ns)
    return encode_frame(MessageType.VIDEO, header + packet.data)


def encode_ping(ping: Ping) -> bytes:
    return encode_frame(MessageType.PING, _PING.pack(ping.mac_time_ns))


def encode_pong(pong: Pong) -> bytes:
    return encode_frame(MessageType.PONG, _PONG.pack(pong.mac_time_ns, pong.phone_time_ns))


# ---------------------------------------------------------------- decoder


def decode_message(message_type: int, payload: bytes) -> Message | None:
    """Decode one frame. Returns None for unknown types so newer peers stay compatible."""
    try:
        kind = MessageType(message_type)
    except ValueError:
        return None
    if kind in JSON_TYPES:
        try:
            body = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProtocolError(f"invalid JSON in {kind.name}: {error}") from error
        if not isinstance(body, dict):
            raise ProtocolError(f"{kind.name} body must be a JSON object")
        return Message(kind, body)
    if kind is MessageType.VIDEO_FORMAT:
        _require(payload, _VIDEO_FORMAT.size + 1, kind)
        width, height, fps = _VIDEO_FORMAT.unpack_from(payload)
        return Message(kind, VideoFormat(width, height, fps, payload[_VIDEO_FORMAT.size :]))
    if kind is MessageType.VIDEO:
        _require(payload, _VIDEO.size + 1, kind)
        flags, pts_us, phone_time_ns = _VIDEO.unpack_from(payload)
        return Message(
            kind,
            VideoPacket(bool(flags & VIDEO_FLAG_KEYFRAME), pts_us, phone_time_ns, payload[_VIDEO.size :]),
        )
    if kind is MessageType.PING:
        _require(payload, _PING.size, kind, exact=True)
        return Message(kind, Ping(*_PING.unpack(payload)))
    if kind is MessageType.PONG:
        _require(payload, _PONG.size, kind, exact=True)
        return Message(kind, Pong(*_PONG.unpack(payload)))
    return None  # pragma: no cover - every enum member is handled above


def _require(payload: bytes, size: int, kind: MessageType, exact: bool = False) -> None:
    if (exact and len(payload) != size) or len(payload) < size:
        raise ProtocolError(f"{kind.name} payload has {len(payload)} bytes, expected {'exactly ' if exact else 'at least '}{size}")
