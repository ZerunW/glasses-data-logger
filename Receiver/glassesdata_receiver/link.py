"""Connection sources that reach the iPhone: USB (through usbmuxd) and plain TCP (for tests).

Over USB this computer connects to a port the app listens on at the phone's loopback
address. The cable and the iPhone's "Trust This Computer" pairing are the trust
boundary, so this path needs no pairing step and no encryption.

usbmuxd is part of macOS. On Linux it is the ``usbmuxd`` package, which udev starts
when an iPhone is plugged in (and which stops again when the last one is unplugged), so
its socket is missing until then. Its socket is world-writable (mode 0666), so no group
membership is needed. Linux usbmuxd lists an iPhone even before it is trusted, and
asks the phone to show "Trust This Computer" itself; a Connect is then a plain TCP
connection to the app's port, which usbmuxd does not gate on pairing. Whether iOS
accepts it from a computer it has never trusted is not documented, so when a Connect
is refused the USB status also says when the computer is not trusted yet.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import socket
import sys
import threading

from . import usbmux
from .receiver import Receiver
from .session import Session

log = logging.getLogger(__name__)

RETRY_S = 1.0


# usbmuxd's udev rule (39-usbmuxd.rules) starts it for Apple's vendor id and these product ids.
_APPLE_VENDOR = "05ac"
_MOBILE_PRODUCT = re.compile(r"12[9a][0-9a-f]")
USB_DEVICES = "/sys/bus/usb/devices"


def apple_mobile_device_plugged(devices: str | None = None) -> bool:
    """Linux: an iPhone, iPad or iPod is on the USB bus (whether or not usbmuxd runs)."""
    devices = devices or USB_DEVICES
    try:
        entries = os.listdir(devices)
    except OSError:
        return False
    for entry in entries:
        try:
            with open(os.path.join(devices, entry, "idVendor"), encoding="ascii") as handle:
                if handle.read().strip().lower() != _APPLE_VENDOR:
                    continue
            with open(os.path.join(devices, entry, "idProduct"), encoding="ascii") as handle:
                if _MOBILE_PRODUCT.fullmatch(handle.read().strip().lower()):
                    return True
        except (OSError, UnicodeDecodeError):
            continue
    return False


def usbmuxd_unavailable(error: BaseException, devices: str | None = None) -> str:
    """The USB status text when usbmuxd cannot be reached, with what to do on this platform."""
    if sys.platform == "darwin":
        return f"usbmuxd unavailable ({error})"
    installed = shutil.which("usbmuxd") is not None or any(
        os.path.exists(path) for path in ("/usr/sbin/usbmuxd", "/usr/bin/usbmuxd", "/sbin/usbmuxd")
    )
    if not installed:
        return "usbmuxd is not installed - install it with: sudo apt install usbmuxd"
    if apple_mobile_device_plugged(devices):
        return (
            "the iPhone is plugged in but usbmuxd is not running - run: sudo systemctl restart usbmuxd "
            "(journalctl -u usbmuxd says why it stopped)"
        )
    return "plug in the iPhone with a cable (usbmuxd starts when an iPhone is plugged in)"


NOT_TRUSTED = (
    "iPhone found, but this computer is not trusted yet - unlock the iPhone and tap Trust "
    "(or run: idevicepair pair), then open GlassesData and turn on Computer link (USB)"
)


def refused_text(device: usbmux.Device) -> str:
    """The USB status when the app's port refuses the connection.

    On Linux a missing pairing record is the likelier cause on a first run, so say so.
    macOS keeps the message it has always shown (its usbmuxd is Apple's own)."""
    if sys.platform != "darwin" and usbmux.has_pair_record(device.serial) is False:
        return NOT_TRUSTED
    return "iPhone found - open GlassesData and turn on Computer link (USB)"


class UsbSource(threading.Thread):
    def __init__(self, receiver: Receiver, port: int, stop: threading.Event) -> None:
        super().__init__(name="gd-usb", daemon=True)
        self._receiver = receiver
        self._port = port
        self._stop_event = stop

    def run(self) -> None:
        status = self._receiver.set_source_status
        if isinstance(self._port, bool) or not isinstance(self._port, int) or not 1 <= self._port <= 65535:
            # The CLI refuses such a port; anything else that builds a UsbSource gets a clear error, not a dead thread.
            log.error("USB link not started: %r is not a port (1-65535)", self._port)
            status("usb", f"not started: {self._port!r} is not a port (1-65535)")
            return
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
                    status("usb", refused_text(device))
                else:
                    status("usb", f"iPhone found - cannot connect ({error})")
                self._stop_event.wait(RETRY_S)
                continue
            except (usbmux.UsbmuxError, OSError, ValueError) as error:
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
