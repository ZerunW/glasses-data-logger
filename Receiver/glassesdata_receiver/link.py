"""Connection sources that reach the iPhone: USB (through usbmuxd) and plain TCP (for tests).

Over USB this computer connects to a port the app listens on at the phone's loopback
address. The cable and the iPhone's "Trust This Computer" pairing are the trust
boundary, so this path needs no pairing step and no encryption.

usbmuxd is part of macOS. On Linux it is the ``usbmuxd`` package, which udev starts
when an iPhone is plugged in, so its socket is often missing until then.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import sys
import threading

from . import usbmux
from .receiver import Receiver
from .session import Session

log = logging.getLogger(__name__)

RETRY_S = 1.0


def usbmuxd_unavailable(error: BaseException) -> str:
    """The USB status text when usbmuxd cannot be reached, with what to do on this platform."""
    if sys.platform == "darwin":
        return f"usbmuxd unavailable ({error})"
    installed = shutil.which("usbmuxd") is not None or any(
        os.path.exists(path) for path in ("/usr/sbin/usbmuxd", "/usr/bin/usbmuxd", "/sbin/usbmuxd")
    )
    if not installed:
        return "usbmuxd is not installed - install it with: sudo apt install usbmuxd"
    return "plug in the iPhone with a cable (usbmuxd starts when an iPhone is plugged in)"


class UsbSource(threading.Thread):
    def __init__(self, receiver: Receiver, port: int, stop: threading.Event) -> None:
        super().__init__(name="gd-usb", daemon=True)
        self._receiver = receiver
        self._port = port
        self._stop_event = stop

    def run(self) -> None:
        status = self._receiver.set_source_status
        while not self._stop_event.is_set():
            if self._receiver.busy:
                self._stop_event.wait(RETRY_S)
                continue
            try:
                device = usbmux.first_usb_device()
            except (usbmux.UsbmuxError, OSError) as error:
                status("usb", usbmuxd_unavailable(error))
                self._stop_event.wait(2 * RETRY_S)
                continue
            if device is None:
                status("usb", "plug in the iPhone with a cable")
                self._stop_event.wait(RETRY_S)
                continue
            try:
                sock = usbmux.connect(device.device_id, self._port)
            except usbmux.UsbmuxConnectError as error:
                if error.code == 3:
                    status("usb", "iPhone found - open GlassesData and turn on Computer link (USB)")
                else:
                    status("usb", f"iPhone found - cannot connect ({error})")
                self._stop_event.wait(RETRY_S)
                continue
            except (usbmux.UsbmuxError, OSError) as error:
                status("usb", f"cannot connect ({error})")
                self._stop_event.wait(RETRY_S)
                continue
            status("usb", "connected")
            reason = Session(sock, "usb", device.serial, self._receiver).run()
            status("usb", f"waiting ({reason})")
            self._stop_event.wait(RETRY_S)


class TcpSource(threading.Thread):
    """Connects to a GDL1 sender on a TCP address. Used with tools/fake_sender.py."""

    def __init__(self, receiver: Receiver, host: str, port: int, stop: threading.Event) -> None:
        super().__init__(name="gd-tcp", daemon=True)
        self._receiver = receiver
        self._address = (host, port)
        self._stop_event = stop

    def run(self) -> None:
        status = self._receiver.set_source_status
        while not self._stop_event.is_set():
            if self._receiver.busy:
                self._stop_event.wait(RETRY_S)
                continue
            try:
                sock = socket.create_connection(self._address, timeout=2)
            except OSError as error:
                status("tcp", f"waiting for {self._address[0]}:{self._address[1]} ({error.strerror or error})")
                self._stop_event.wait(RETRY_S)
                continue
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            status("tcp", "connected")
            reason = Session(sock, "tcp", f"{self._address[0]}:{self._address[1]}", self._receiver).run()
            status("tcp", f"waiting ({reason})")
            self._stop_event.wait(RETRY_S)
