"""Minimal plist-protocol client for usbmuxd (built into macOS; the usbmuxd package on Linux)."""

from __future__ import annotations

import plistlib
import socket
import struct
import threading
from dataclasses import dataclass
from typing import Callable

DEFAULT_SOCKET = "/var/run/usbmuxd"


class UsbmuxError(Exception):
    """A usbmuxd transport or protocol failure."""


class UsbmuxConnectError(UsbmuxError):
    """A nonzero result from usbmuxd's Connect command."""

    def __init__(self, code: int) -> None:
        self.code = code
        description = {
            1: "bad command",
            2: "bad device",
            3: "connection refused",
            6: "bad protocol version",
        }.get(code, "unknown error")
        super().__init__(f"usbmuxd Connect failed (code {code}: {description})")


@dataclass(frozen=True)
class Device:
    device_id: int
    serial: str
    connection_type: str
    product_id: int | None


class _WatchStopped(Exception):
    """Internal signal to stop a Listen read without discarding partial frames."""


def encode_message(payload: dict, tag: int) -> bytes:
    """Frame a dictionary as a version-1, type-8 XML plist message."""
    if not 0 <= tag <= 0xFFFFFFFF:
        raise ValueError("tag must be a 32-bit unsigned integer")
    body = plistlib.dumps(payload, fmt=plistlib.FMT_XML)
    if len(body) > 1_048_576:
        raise UsbmuxError("usbmuxd message body exceeds 1 MB")
    return struct.pack("<IIII", 16 + len(body), 1, 8, tag) + body


def _recv_exact(
    sock: socket.socket,
    count: int,
    stop_event: threading.Event | None = None,
) -> bytes:
    data = bytearray()
    while len(data) < count:
        if stop_event is not None and stop_event.is_set():
            raise _WatchStopped
        try:
            chunk = sock.recv(count - len(data))
        except socket.timeout as exc:
            if stop_event is not None:
                continue
            raise UsbmuxError("timed out reading a usbmuxd message") from exc
        except OSError as exc:
            raise UsbmuxError("failed reading from usbmuxd") from exc
        if not chunk:
            raise UsbmuxError("usbmuxd closed the socket during a message")
        data.extend(chunk)
    return bytes(data)


def _read_message(
    sock: socket.socket,
    stop_event: threading.Event | None = None,
) -> tuple[int, dict]:
    """Read a frame, optionally checking for Listen cancellation."""
    header = _recv_exact(sock, 16, stop_event)
    total_length, version, message_type, tag = struct.unpack("<IIII", header)
    if version != 1:
        raise UsbmuxError(f"unsupported usbmuxd protocol version: {version}")
    if message_type != 8:
        raise UsbmuxError(f"unexpected usbmuxd message type: {message_type}")
    if total_length < 16:
        raise UsbmuxError(f"invalid usbmuxd message length: {total_length}")
    body_length = total_length - 16
    if body_length > 1_048_576:
        raise UsbmuxError("usbmuxd message body exceeds 1 MB")
    body = _recv_exact(sock, body_length, stop_event)
    try:
        payload = plistlib.loads(body)
    except Exception as exc:
        raise UsbmuxError("invalid usbmuxd plist body") from exc
    if not isinstance(payload, dict):
        raise UsbmuxError("usbmuxd plist body is not a dictionary")
    return tag, payload


def read_message(sock: socket.socket) -> tuple[int, dict]:
    """Read an entire plist frame, returning its tag and decoded dictionary."""
    return _read_message(sock)


def _open_socket(socket_path: str, timeout: float) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(socket_path)
    except OSError as exc:
        sock.close()
        raise UsbmuxError(f"could not connect to usbmuxd at {socket_path!r}") from exc
    return sock


def _request(message_type: str, **fields: object) -> dict[str, object]:
    return {
        "MessageType": message_type,
        "ProgName": "glassesdata-receiver",
        "ClientVersionString": "glassesdata-receiver-1",
        **fields,
    }


def _send_request(sock: socket.socket, payload: dict[str, object]) -> None:
    try:
        sock.sendall(encode_message(payload, tag=1))
    except OSError as exc:
        raise UsbmuxError("failed sending a request to usbmuxd") from exc


def _read_reply(
    sock: socket.socket,
    stop_event: threading.Event | None = None,
) -> dict:
    tag, payload = _read_message(sock, stop_event)
    if tag != 1:
        raise UsbmuxError(f"usbmuxd reply has unexpected tag: {tag}")
    return payload


def _parse_device(entry: object) -> Device:
    if not isinstance(entry, dict):
        raise UsbmuxError("invalid device entry from usbmuxd")
    properties = entry.get("Properties")
    if not isinstance(properties, dict):
        raise UsbmuxError("device entry lacks Properties")
    device_id = entry.get("DeviceID", properties.get("DeviceID"))
    serial = properties.get("SerialNumber", "")
    connection_type = properties.get("ConnectionType")
    product_id = properties.get("ProductID")
    if not isinstance(device_id, int):
        raise UsbmuxError("device entry has invalid DeviceID")
    if not isinstance(serial, str) or not isinstance(connection_type, str):
        raise UsbmuxError("device entry has invalid connection properties")
    if product_id is not None and not isinstance(product_id, int):
        raise UsbmuxError("device entry has invalid ProductID")
    return Device(device_id, serial, connection_type, product_id)


def _list_devices(socket_path: str, timeout: float) -> list[Device]:
    with _open_socket(socket_path, timeout) as sock:
        _send_request(sock, _request("ListDevices"))
        reply = _read_reply(sock)
    entries = reply.get("DeviceList")
    if not isinstance(entries, list):
        raise UsbmuxError("ListDevices reply lacks a DeviceList array")
    return [_parse_device(entry) for entry in entries]


def list_devices(
    socket_path: str = DEFAULT_SOCKET,
    timeout: float = 5.0,
) -> list[Device]:
    """Return both USB and Network devices reported by usbmuxd."""
    return _list_devices(socket_path, timeout)


def first_usb_device(
    socket_path: str = DEFAULT_SOCKET,
    timeout: float = 5.0,
) -> Device | None:
    """Return the first USB-connected device, if one is present."""
    return next(
        (device for device in list_devices(socket_path, timeout) if device.connection_type == "USB"),
        None,
    )


def _result_code(reply: dict, operation: str) -> int:
    if reply.get("MessageType") != "Result" or not isinstance(reply.get("Number"), int):
        raise UsbmuxError(f"malformed usbmuxd {operation} result")
    return reply["Number"]


def connect(
    device_id: int,
    port: int,
    socket_path: str = DEFAULT_SOCKET,
    timeout: float = 5.0,
) -> socket.socket:
    """Open a raw, blocking USB tunnel to an iPhone TCP port."""
    if not isinstance(device_id, int) or not 0 <= device_id <= 0xFFFFFFFF:
        raise ValueError("device_id must be a 32-bit unsigned integer")
    if not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be in the range 0..65535")
    sock = _open_socket(socket_path, timeout)
    try:
        # usbmuxd (macOS and libimobiledevice on Linux) takes the port in network byte order:
        # on little-endian machines htons(7900) is 56350, not 39966.
        _send_request(
            sock,
            _request("Connect", DeviceID=device_id, PortNumber=socket.htons(port)),
        )
        reply = _read_reply(sock)
        code = _result_code(reply, "Connect")
        if code != 0:
            raise UsbmuxConnectError(code)
        sock.settimeout(None)
        return sock
    except BaseException:
        sock.close()
        raise


def watch(
    on_attach: Callable[[Device], None],
    on_detach: Callable[[int], None],
    stop_event: threading.Event,
    socket_path: str = DEFAULT_SOCKET,
) -> None:
    """Process attach/detach notifications until stop_event is set."""
    if stop_event.is_set():
        return
    sock = _open_socket(socket_path, timeout=0.5)
    try:
        _send_request(sock, _request("Listen"))
        reply = _read_reply(sock, stop_event)
        code = _result_code(reply, "Listen")
        if code != 0:
            raise UsbmuxError(f"usbmuxd Listen failed (code {code})")
        while not stop_event.is_set():
            _, notification = _read_message(sock, stop_event)
            message_type = notification.get("MessageType")
            if message_type == "Attached":
                on_attach(_parse_device(notification))
            elif message_type == "Detached":
                device_id = notification.get("DeviceID")
                if not isinstance(device_id, int):
                    raise UsbmuxError("Detached notification lacks a valid DeviceID")
                on_detach(device_id)
    except _WatchStopped:
        return
    finally:
        sock.close()
