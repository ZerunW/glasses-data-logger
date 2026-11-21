"""Command line entry point: ``python -m glassesdata_receiver``."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from . import paths, protocol
from .receiver import Receiver


def link_port(text: str) -> int:
    """``--port``: 1-65535, or 0 (any free port, only for the --wifi listener; checked in parse_args)."""
    try:
        port = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a port number") from None
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"{port} is not a port: use 1 to 65535 (0: any free port, with --wifi only)")
    return port


def tcp_address(text: str) -> tuple[str, int]:
    """``--tcp HOST:PORT`` (``:PORT`` means 127.0.0.1; an IPv6 host goes in brackets)."""
    host, separator, port = text.rpartition(":")
    if not separator:
        raise argparse.ArgumentTypeError(f"{text!r} is not HOST:PORT, such as 127.0.0.1:7901")
    host = host.strip()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        number = int(port)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{port!r} in {text!r} is not a port number") from None
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError(f"{number} in {text!r} is not a port: use 1 to 65535")
    return host or "127.0.0.1", number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="glassesdata_receiver",
        description="Receive Meta glasses video and sensors from the GlassesData iPhone app, "
        "record them, and track hand skeletons in real time.",
    )
    parser.add_argument("--usb", action="store_true", help="connect to the iPhone over a USB cable (default)")
    parser.add_argument("--wifi", action="store_true", help="accept the iPhone over encrypted Wi-Fi")
    parser.add_argument("--pair", action="store_true", help="with --wifi: show the pairing QR code at start")
    parser.add_argument("--reset-pairing", action="store_true", help="forget the paired iPhone: new certificate and token")
    parser.add_argument(
        "--state-dir", type=Path, metavar="DIR",
        help="where the receiver keeps its state: the Wi-Fi identity (certificate, key, pairing.json), the camera "
        f"calibration (calibration/) and the body measurements (profile.json) (default {paths.state_dir()})",
    )
    parser.add_argument(
        "--print-pairing", action="store_true",
        help="with --wifi: print PAIRING_URI=<uri> on stdout once the listener accepts connections",
    )
    parser.add_argument("--tcp", metavar="HOST:PORT", type=tcp_address, help="connect to a GDL1 sender over plain TCP (testing)")
    parser.add_argument(
        "--port", type=link_port, default=protocol.DEFAULT_PORT,
        help="link port, 1-65535 (default %(default)s; 0 with --wifi alone: any free port)",
    )
    parser.add_argument("--out", type=Path, default=paths.recordings_dir(), help="recordings folder (default %(default)s)")
    parser.add_argument("--record", action="store_true", help="start recording as soon as video arrives")
    parser.add_argument("--no-hands", action="store_true", help="disable hand tracking")
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument(
        "--mirrored-input", action="store_true",
        help="the video is mirrored (selfie style): swap MediaPipe's hand labels so Left stays the person's left hand",
    )
    parser.add_argument(
        "--virtual-camera", action="store_true",
        help="start with the virtual camera on (OBS Virtual Camera on macOS, v4l2loopback on Linux); "
        "without it, it can still be started from the web page",
    )
    parser.add_argument(
        "--virtual-camera-size", default="1920x1080", choices=["1920x1080", "1280x720"],
        help="the landscape picture sent to the virtual camera (default %(default)s)",
    )
    parser.add_argument(
        "--virtual-camera-mode", default="fit", choices=["fit", "fill"],
        help="fit: the whole portrait frame with blurred sides (default); fill: a landscape band that covers the picture",
    )
    parser.add_argument(
        "--virtual-camera-device", metavar="DEV",
        help="Linux: the v4l2loopback device to write to, such as /dev/video10 (default: the one labelled "
        "GlassesData, else the first v4l2loopback device). Ignored on macOS, where the camera is OBS's",
    )
    parser.add_argument("--overlay", action="store_true", help="draw the hand skeleton on the virtual camera picture")
    parser.add_argument(
        "--web-port", type=int, default=8765, metavar="PORT",
        help="port of the web page (default %(default)s; 0 picks a free port)",
    )
    parser.add_argument(
        "--web-host", default="127.0.0.1", metavar="HOST",
        help="address the web page listens on (default %(default)s: this computer only). "
        "Any other address, such as 0.0.0.0, requires the access key printed in WEB_URL",
    )
    parser.add_argument("--open", action="store_true", help="open the web page in the default browser")
    parser.add_argument("--headless", action="store_true", help="no web page (recording and logs only)")
    parser.add_argument("--duration", type=float, help="quit after this many seconds")
    parser.add_argument(
        "--snapshot", type=Path, metavar="PNG",
        help="at exit, save the last video frame (with the hand skeleton) to this image file",
    )
    parser.add_argument("--no-bonjour", action="store_true", help="with --wifi: do not advertise on the local network")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    uses_usb = args.usb or not (args.wifi or args.tcp)  # USB is the default link
    if args.port == 0 and uses_usb:
        parser.error("--port 0 (any free port) only works for the --wifi listener; over USB give the app's port (1-65535)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("glassesdata")
    if not (args.usb or args.wifi or args.tcp):
        args.usb = True
    args.web_host = args.web_host.strip("[]")  # "[::1]" as in a URL
    if not 0 <= args.web_port <= 65535:
        log.error("--web-port must be between 0 and 65535")
        return 2
    if args.headless and args.open:
        log.warning("--open does nothing with --headless (there is no web page)")

    tracker_factory = None
    metadata: dict[str, Any] = {}
    if not args.no_hands:
        from .hands import HandTracker, ensure_model, handedness_note

        try:
            model = ensure_model()
        except OSError as error:
            log.error("Could not download the hand model (%s). Continuing without hand tracking.", error)
        else:
            tracker_factory = lambda: HandTracker(model, args.max_hands, args.mirrored_input)  # noqa: E731
            metadata["handedness"] = handedness_note(args.mirrored_input)

    if (args.pair or args.print_pairing) and not args.wifi:
        log.warning("--pair and --print-pairing only apply with --wifi")

    stop = threading.Event()
    receiver = Receiver(args.out, tracker_factory, metadata, state_dir=args.state_dir or paths.state_dir())
    # Always there, off until started (by --virtual-camera or the web page); pyvirtualcam is
    # only imported when it starts, and a failure leaves the receiver running.
    from .vcam import VirtualCamera, parse_size

    device = args.virtual_camera_device
    if device and not sys.platform.startswith("linux"):
        log.warning("--virtual-camera-device only applies on Linux (v4l2loopback); ignoring %s", device)
        device = None
    virtual_camera = VirtualCamera(
        receiver, parse_size(args.virtual_camera_size), mode=args.virtual_camera_mode, overlay=args.overlay,
        device=device,
    )
    workers: list[threading.Thread] = []
    web = None
    try:
        if not args.headless:
            web = _start_web(args, receiver, virtual_camera, log)
            if web is None:
                return 2
        if args.virtual_camera:
            virtual_camera.start()  # logs the self-check, or why it cannot start and what to do
        _run(args, receiver, stop, workers, web, log)
    finally:
        # Join everything before the interpreter exits: see Receiver.shutdown.
        stop.set()
        virtual_camera.close()  # refuses a start the page may still send while the server stops
        if web is not None:
            web.stop()
        receiver.shutdown()
        for thread in workers:
            thread.join(2.0)
    return 0


def _start_web(args: argparse.Namespace, receiver: Receiver, virtual_camera: Any, log: logging.Logger) -> Any:
    from .web import WebServer, is_loopback_host

    try:
        web = WebServer(receiver, args.web_host, args.web_port, wifi=args.wifi, pair_at_start=args.pair, vcam=virtual_camera)
    except OSError as error:
        log.error(
            "Cannot serve the web page on %s:%d (%s). Pass --web-port 0 to pick a free port.",
            args.web_host, args.web_port, error.strerror or error,
        )
        return None
    display = None
    if not is_loopback_host(args.web_host):
        if args.web_host in ("0.0.0.0", "::", ""):
            from .wifi import local_ipv4_addresses

            display = next(iter(local_ipv4_addresses()), None)
        log.warning(
            "The web page listens on %s, reachable from other devices: anyone with the key in WEB_URL can watch "
            "and control recording.", args.web_host or "all interfaces",
        )
    web.start()
    url = web.url(display)
    print(f"WEB_URL={url}", flush=True)
    if args.open:
        open_browser(url, log)
    return web


def open_browser(url: str, log: logging.Logger) -> bool:
    """Open the page in the default browser; say so when that cannot work.

    On Linux there must be a graphical session: without DISPLAY or WAYLAND_DISPLAY (an SSH
    login, a text console) Python's webbrowser would fall back to a text browser such as
    lynx inside this terminal, so it is not tried. Run as root (sudo), the desktop's
    browser (Firefox is a snap on Ubuntu) refuses to start, so that is skipped too."""
    if sys.platform.startswith("linux"):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            log.warning("No graphical session (DISPLAY and WAYLAND_DISPLAY are unset): open %s in a browser yourself", url)
            return False
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            log.warning("Not opening a browser as root (the receiver needs no sudo): open %s yourself", url)
            return False
    import webbrowser

    try:
        opened = webbrowser.open(url)
    except Exception as error:  # a broken BROWSER variable, a missing helper
        log.warning("Could not open a browser (%s): open %s yourself", error, url)
        return False
    if not opened:
        log.warning("No browser could be opened: open %s yourself", url)
    return bool(opened)


def _run(
    args: argparse.Namespace,
    receiver: Receiver,
    stop: threading.Event,
    workers: list[threading.Thread],
    web: Any,
    log: logging.Logger,
) -> None:
    def start(thread: threading.Thread) -> None:
        workers.append(thread)
        thread.start()

    install_signal_handlers(stop)

    if args.usb:
        from .link import UsbSource

        start(UsbSource(receiver, args.port, stop))
    if args.tcp:
        from .link import TcpSource

        host, port = args.tcp  # checked by tcp_address
        start(TcpSource(receiver, host, port, stop))
    if args.wifi or args.reset_pairing:
        from . import wifi

        identity = wifi.load_or_create_identity(args.state_dir or paths.state_dir(), reset=args.reset_pairing)
        if args.wifi:
            wifi_source = wifi.WifiSource(receiver, identity, args.port, stop, advertise=not args.no_bonjour)
            start(wifi_source)
            if wifi_source.wait_ready(timeout=15):
                uri = wifi.pairing_uri(identity, wifi.local_ipv4_addresses(), wifi_source.port)
                if web is not None:
                    web.pairing_png = wifi.qr_png(uri)
                if args.print_pairing:
                    print(f"PAIRING_URI={uri}", flush=True)
                if args.pair:
                    print(wifi.qr_ascii(uri))
                    print("Scan this code in the GlassesData iPhone app: Computer card > Pair over Wi-Fi.\n"
                          "Treat it like a password: anyone who scans it can connect to this receiver.")
            else:
                log.error("The Wi-Fi listener did not start; pairing is unavailable.")

    deadline = time.monotonic() + args.duration if args.duration else None

    if args.record:
        start(threading.Thread(target=_auto_record, args=(receiver, stop), name="gd-auto-record", daemon=True))

    def should_stop() -> bool:
        return stop.is_set() or (deadline is not None and time.monotonic() >= deadline)

    # The main thread only waits and writes the snapshot at the end (the virtual camera has its own thread).
    while not should_stop():
        stop.wait(0.1)

    if args.snapshot is not None:
        from .viewer import save_snapshot

        frame = receiver.latest_frame() if receiver.busy else None
        save_snapshot(args.snapshot, frame, receiver.latest_hands(), receiver.hands_active)


def install_signal_handlers(stop: threading.Event) -> None:
    """Ctrl-C, SIGTERM (systemd, kill) and SIGHUP (the terminal window closed) all stop the
    receiver the same way on macOS and Linux: the recording is closed and the threads joined.
    SIGHUP only when it is not ignored already: under nohup the receiver must keep running."""
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    if hasattr(signal, "SIGHUP") and signal.getsignal(signal.SIGHUP) == signal.SIG_DFL:
        signal.signal(signal.SIGHUP, lambda *_: stop.set())


def _auto_record(receiver: Receiver, stop: threading.Event) -> None:
    """Start a recording whenever video is flowing and none is running (also after a reconnect)."""
    while not stop.wait(0.2):
        status = receiver.status()
        if status.kind and status.format is not None and status.recording is None and status.received_fps > 0:
            receiver.start_recording(auto=True)  # logs "Recording to ..." itself


if __name__ == "__main__":
    sys.exit(main())
