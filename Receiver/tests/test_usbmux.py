"""Wire-protocol tests using a threaded fake usbmuxd; no hardware required."""

from __future__ import annotations

import os
import plistlib
import socket
import struct
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from glassesdata_receiver import usbmux


@pytest.fixture
def socket_path(tmp_path: Path) -> Iterator[str]:
    path = tmp_path / "mux.sock"
    # macOS AF_UNIX paths are limited to 104 bytes, including the terminator.
    if len(os.fsencode(path)) < 100:
        yield str(path)
    else:
        with tempfile.TemporaryDirectory(prefix="usbmux-", dir="/tmp") as directory:
            yield os.path.join(directory, "mux.sock")


@contextmanager
def fake_usbmuxd(
    socket_path: str,
    handler: Callable[[socket.socket], None],
    connections: int = 1,
) -> Iterator[None]:
    errors: list[BaseException] = []
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_path)
    listener.listen(connections)
    listener.settimeout(3.0)

    def serve() -> None:
        try:
            for _ in range(connections):
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(3.0)
                    handler(connection)
        except BaseException as exc:
            errors.append(exc)
        finally:
            listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield
    finally:
        thread.join(timeout=4.0)
        if thread.is_alive():
            listener.close()
        assert not thread.is_alive(), "fake usbmuxd thread did not terminate"
        if errors:
            raise errors[0]
        os.unlink(socket_path)


def check_request(connection: socket.socket, expected_type: str) -> tuple[int, dict]:
    tag, request = usbmux.read_message(connection)
    assert request["MessageType"] == expected_type
    assert request["ProgName"] == "glassesdata-receiver"
    assert request["ClientVersionString"] == "glassesdata-receiver-1"
    return tag, request


def device_entry(device_id: int, connection_type: str) -> dict:
    return {
        "MessageType": "Attached",
        "DeviceID": device_id,
        "Properties": {
            "ConnectionType": connection_type,
            "DeviceID": device_id,
            "SerialNumber": f"0000-{device_id}",
            "ProductID": 4776,
        },
    }


def test_list_devices_and_first_usb_device(socket_path: str) -> None:
    entries = [device_entry(3, "USB"), device_entry(4, "Network")]

    def respond(connection: socket.socket) -> None:
        tag, _ = check_request(connection, "ListDevices")
        connection.sendall(usbmux.encode_message({"DeviceList": entries}, tag))

    with fake_usbmuxd(socket_path, respond, connections=2):
        devices = usbmux.list_devices(socket_path)
        first = usbmux.first_usb_device(socket_path)

    assert devices == [
        usbmux.Device(3, "0000-3", "USB", 4776),
        usbmux.Device(4, "0000-4", "Network", 4776),
    ]
    assert first == devices[0]


def test_first_usb_device_with_network_only(socket_path: str) -> None:
    def respond(connection: socket.socket) -> None:
        tag, _ = check_request(connection, "ListDevices")
        connection.sendall(
            usbmux.encode_message({"DeviceList": [device_entry(4, "Network")]}, tag)
        )

    with fake_usbmuxd(socket_path, respond):
        assert usbmux.first_usb_device(socket_path) is None


def test_connect_port_number_and_raw_echo(socket_path: str) -> None:
    def respond(connection: socket.socket) -> None:
        tag, request = check_request(connection, "Connect")
        assert request["DeviceID"] == 3
        assert request["PortNumber"] == socket.htons(7900)
        connection.sendall(
            usbmux.encode_message({"MessageType": "Result", "Number": 0}, tag)
        )
        incoming = connection.recv(5)
        assert incoming == b"hello"
        connection.sendall(incoming)

    with fake_usbmuxd(socket_path, respond):
        with usbmux.connect(3, 7900, socket_path) as tunnel:
            assert tunnel.gettimeout() is None
            tunnel.sendall(b"hello")
            assert tunnel.recv(5) == b"hello"


def test_connect_refused_raises_code_3(socket_path: str) -> None:
    def respond(connection: socket.socket) -> None:
        tag, _ = check_request(connection, "Connect")
        connection.sendall(
            usbmux.encode_message({"MessageType": "Result", "Number": 3}, tag)
        )

    with fake_usbmuxd(socket_path, respond):
        with pytest.raises(usbmux.UsbmuxConnectError, match="connection refused") as exc:
            usbmux.connect(3, 7900, socket_path)
        assert exc.value.code == 3


def test_watch_attach_detach_and_prompt_stop(socket_path: str) -> None:
    stop_event = threading.Event()
    detached = threading.Event()
    seen: list[tuple[str, usbmux.Device | int]] = []
    watch_errors: list[BaseException] = []

    def respond(connection: socket.socket) -> None:
        tag, _ = check_request(connection, "Listen")
        connection.sendall(
            usbmux.encode_message({"MessageType": "Result", "Number": 0}, tag)
            + usbmux.encode_message(device_entry(3, "USB"), tag + 1)
            + usbmux.encode_message({"MessageType": "Detached", "DeviceID": 3}, tag + 2)
        )
        # Keep the listener connection open until watch closes it on stop.
        assert connection.recv(1) == b""

    def on_attach(device: usbmux.Device) -> None:
        seen.append(("attach", device))

    def on_detach(device_id: int) -> None:
        seen.append(("detach", device_id))
        detached.set()

    def run_watch() -> None:
        try:
            usbmux.watch(on_attach, on_detach, stop_event, socket_path)
        except BaseException as exc:
            watch_errors.append(exc)

    with fake_usbmuxd(socket_path, respond):
        watcher = threading.Thread(target=run_watch, daemon=True)
        watcher.start()
        try:
            assert detached.wait(timeout=2.0)
        finally:
            start = time.monotonic()
            stop_event.set()
            watcher.join(timeout=1.5)
            elapsed = time.monotonic() - start
        assert not watcher.is_alive(), "watch did not stop promptly"
        assert elapsed < 1.5

    assert not watch_errors
    assert seen == [
        ("attach", usbmux.Device(3, "0000-3", "USB", 4776)),
        ("detach", 3),
    ]


def test_read_message_handles_one_byte_at_a_time(socket_path: str) -> None:
    payload = {"DeviceList": [device_entry(12, "USB")]}
    framed = usbmux.encode_message(payload, tag=42)

    def respond(connection: socket.socket) -> None:
        for byte in framed:
            connection.sendall(bytes((byte,)))
            time.sleep(0.0001)

    with fake_usbmuxd(socket_path, respond):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(3.0)
            client.connect(socket_path)
            assert usbmux.read_message(client) == (42, payload)


@pytest.mark.parametrize("version, message_type", [(0, 8), (1, 1)])
def test_read_message_rejects_invalid_header(
    socket_path: str, version: int, message_type: int
) -> None:
    body = plistlib.dumps({"MessageType": "Result", "Number": 0}, fmt=plistlib.FMT_XML)
    message = struct.pack("<IIII", 16 + len(body), version, message_type, 7) + body

    def respond(connection: socket.socket) -> None:
        connection.sendall(message)

    with fake_usbmuxd(socket_path, respond):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(socket_path)
            with pytest.raises(usbmux.UsbmuxError):
                usbmux.read_message(client)


def test_read_message_rejects_oversized_body(socket_path: str) -> None:
    def respond(connection: socket.socket) -> None:
        connection.sendall(struct.pack("<IIII", 16 + 1_048_577, 1, 8, 12))

    with fake_usbmuxd(socket_path, respond):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(socket_path)
            with pytest.raises(usbmux.UsbmuxError, match="exceeds 1 MB"):
                usbmux.read_message(client)


def test_read_message_rejects_truncated_body(socket_path: str) -> None:
    def respond(connection: socket.socket) -> None:
        connection.sendall(struct.pack("<IIII", 19, 1, 8, 5) + b"x")

    with fake_usbmuxd(socket_path, respond):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(socket_path)
            with pytest.raises(usbmux.UsbmuxError, match="closed"):
                usbmux.read_message(client)


def test_watch_reports_disconnect(socket_path: str) -> None:
    def respond(connection: socket.socket) -> None:
        tag, _ = check_request(connection, "Listen")
        connection.sendall(
            usbmux.encode_message({"MessageType": "Result", "Number": 0}, tag)
        )
        # Closing the daemon-side connection is an error if watch wasn't stopped.

    with fake_usbmuxd(socket_path, respond):
        with pytest.raises(usbmux.UsbmuxError, match="closed"):
            usbmux.watch(lambda device: None, lambda device_id: None, threading.Event(), socket_path)
