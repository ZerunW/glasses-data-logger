"""Linux branches, run on any computer: sys.platform, sysfs, usbmuxd and pyvirtualcam are faked.

Nothing here touches real devices: every sysfs or /lib/modules path is a folder under tmp_path,
usbmuxd is a fake Unix-socket server, and pyvirtualcam is a stand-in module.
"""

from __future__ import annotations

import enum
import logging
import signal
import socket
import sys
import threading
import types

import numpy as np
import pytest

from glassesdata_receiver import __main__ as cli
from glassesdata_receiver import link, usbmux, vcam

from test_usbmux import check_request, fake_usbmuxd, socket_path  # noqa: F401  (socket_path is a fixture)
from test_vcam import FakeReceiver, frame, wait_for

# ---------------------------------------------------------------- usbmuxd: pairing record


def test_has_pair_record_answers_true_false_or_unknown(socket_path):  # noqa: F811
    answers = [
        {"PairRecordData": b"<plist/>"},  # trusted once: usbmuxd holds the record
        {"MessageType": "Result", "Number": usbmux.ENOENT},  # never trusted
        {"MessageType": "Result", "Number": 22},  # EINVAL: not an answer about pairing
    ]
    seen = []

    def respond(connection: socket.socket) -> None:
        tag, request = check_request(connection, "ReadPairRecord")
        seen.append(request["PairRecordID"])
        connection.sendall(usbmux.encode_message(answers[len(seen) - 1], tag))

    with fake_usbmuxd(socket_path, respond, connections=3):
        results = [usbmux.has_pair_record("00008110-001A2C3D4E5F6071", socket_path) for _ in answers]
    assert results == [True, False, None]
    assert seen == ["00008110-001A2C3D4E5F6071"] * 3


def test_has_pair_record_without_usbmuxd_or_serial_is_unknown(tmp_path):
    assert usbmux.has_pair_record("abc", str(tmp_path / "missing.sock")) is None
    assert usbmux.has_pair_record("", str(tmp_path / "missing.sock")) is None


DEVICE = usbmux.Device(1, "00008110-001A2C3D4E5F6071", "USB", 0x12A8)


def test_a_refused_connection_on_linux_says_trust_when_there_is_no_pairing_record(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(usbmux, "has_pair_record", lambda serial, *a, **k: False)
    assert link.refused_text(DEVICE) == link.NOT_TRUSTED
    assert "tap Trust" in link.NOT_TRUSTED and "idevicepair pair" in link.NOT_TRUSTED
    for answer in (True, None):  # paired, or unknown: the app is probably not listening
        monkeypatch.setattr(usbmux, "has_pair_record", lambda serial, *a, answer=answer, **k: answer)
        assert link.refused_text(DEVICE) == "iPhone found - open GlassesData and turn on Computer link (USB)"


def test_a_refused_connection_on_macos_keeps_its_message(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")

    def never(*args, **kwargs):
        raise AssertionError("macOS does not ask for the pairing record")

    monkeypatch.setattr(usbmux, "has_pair_record", never)
    assert link.refused_text(DEVICE) == "iPhone found - open GlassesData and turn on Computer link (USB)"


def usb_device(root, name, vendor, product):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "idVendor").write_text(vendor + "\n")
    (folder / "idProduct").write_text(product + "\n")


def test_an_iphone_on_the_bus_without_usbmuxd_says_restart_it(monkeypatch, tmp_path):
    bus = tmp_path / "usb"
    usb_device(bus, "1-1", "046d", "c52b")  # a mouse
    usb_device(bus, "1-2", "05ac", "8600")  # an Apple device that is not a phone
    (bus / "usb1").mkdir()  # a root hub without the files
    monkeypatch.setattr(link, "USB_DEVICES", str(bus))
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(link.shutil, "which", lambda name: "/usr/sbin/usbmuxd")
    error = OSError("could not connect")
    assert not link.apple_mobile_device_plugged()
    assert link.usbmuxd_unavailable(error).startswith("plug in the iPhone")
    usb_device(bus, "1-3", "05ac", "12a8")  # an iPhone
    assert link.apple_mobile_device_plugged()
    text = link.usbmuxd_unavailable(error)
    assert "usbmuxd is not running" in text and "sudo systemctl restart usbmuxd" in text


# ---------------------------------------------------------------- v4l2loopback: sysfs


def video_device(root, name, label, loopback=True):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "name").write_text(label + "\n")
    (folder / "dev").write_text("81:0\n")
    if loopback:
        (folder / "max_openers").write_text("10\n")
        (folder / "buffers").write_text("2\n")


def unload(sysfs):
    """As after modprobe -r v4l2loopback: the module and its devices are gone, the webcam stays."""
    sysfs.module.rmdir()
    for name in ("video2", "video10"):
        for item in (sysfs.root / name).iterdir():
            item.unlink()
        (sysfs.root / name).rmdir()


@pytest.fixture
def sysfs(tmp_path, monkeypatch):
    """A fake /sys with the module loaded, one GlassesData and one OBS loopback device and a real webcam."""
    root = tmp_path / "video4linux"
    video_device(root, "video10", "GlassesData")
    video_device(root, "video2", "OBS Virtual Camera")
    video_device(root, "video0", "Integrated Camera: Integrated C", loopback=False)
    (root / "not-a-video").mkdir()
    module = tmp_path / "module" / "v4l2loopback"
    module.mkdir(parents=True)
    monkeypatch.setattr(vcam, "V4L2_SYSFS", str(root))
    monkeypatch.setattr(vcam, "V4L2LOOPBACK_MODULE", str(module))
    monkeypatch.setattr(vcam, "MODULES_ROOT", str(tmp_path / "modules"))
    monkeypatch.setattr(vcam, "SECURE_BOOT_VARIABLE", str(tmp_path / "SecureBoot"))
    return types.SimpleNamespace(root=root, module=module, tmp=tmp_path)


def test_loopback_devices_are_read_from_sysfs(sysfs):
    state = vcam.loopback_state()
    assert state.module_loaded
    assert state.devices == (vcam.LoopbackDevice("/dev/video2", "OBS Virtual Camera"), vcam.LoopbackDevice("/dev/video10", "GlassesData"))
    assert state.listed() == "/dev/video2 (OBS Virtual Camera), /dev/video10 (GlassesData)"
    unload(sysfs)
    assert vcam.loopback_state() == vcam.LoopbackState(False, ())


def test_module_installed_and_secure_boot_are_read_without_root(sysfs):
    modules = sysfs.tmp / "modules" / "6.8.0-88-generic"
    modules.mkdir(parents=True)
    assert vcam.loopback_module_installed("6.8.0-88-generic") is None  # no modules.dep: unknown
    (modules / "modules.dep").write_text("kernel/drivers/media/v4l2-core/videodev.ko.zst: kernel/drivers/media/mc/mc.ko.zst\n")
    assert vcam.loopback_module_installed("6.8.0-88-generic") is False
    (modules / "modules.dep").write_text("kernel/v4l2loopback/v4l2loopback.ko.zst: kernel/drivers/media/v4l2-core/videodev.ko.zst\n")
    assert vcam.loopback_module_installed("6.8.0-88-generic") is True
    (modules / "modules.dep").write_text("updates/dkms/v4l2loopback.ko.zst: kernel/drivers/media/v4l2-core/videodev.ko.zst\n")
    assert vcam.loopback_module_installed("6.8.0-88-generic") is True
    assert vcam.loopback_module_installed("missing-kernel") is None
    assert vcam.secure_boot_enabled() is None  # no EFI variable: a BIOS boot, or not readable
    variable = sysfs.tmp / "SecureBoot"
    variable.write_bytes(b"\x06\x00\x00\x00\x01")  # attributes, then the value
    assert vcam.secure_boot_enabled() is True
    variable.write_bytes(b"\x06\x00\x00\x00\x00")
    assert vcam.secure_boot_enabled() is False


def test_the_device_is_the_one_asked_for_else_the_glassesdata_one(sysfs, tmp_path):
    state = vcam.loopback_state()
    assert vcam.choose_loopback_device(None, state) == "/dev/video10"  # labelled GlassesData
    assert vcam.choose_loopback_device("/dev/video2", state) == "/dev/video2"
    # devices listed although /sys/module/v4l2loopback is missing (a kernel with it built in): use them
    assert vcam.choose_loopback_device(None, vcam.LoopbackState(False, state.devices)) == "/dev/video10"
    only_obs = vcam.LoopbackState(True, (vcam.LoopbackDevice("/dev/video2", "OBS Virtual Camera"),))
    assert vcam.choose_loopback_device(None, only_obs) is None  # pyvirtualcam takes the first it finds
    with pytest.raises(vcam.Unavailable) as missing:
        vcam.choose_loopback_device("/dev/video99", state)
    assert "/dev/video99 does not exist" in str(missing.value)
    assert "/dev/video2 (OBS Virtual Camera), /dev/video10 (GlassesData)" in missing.value.hint
    webcam = tmp_path / "video0"  # exists, but sysfs does not list it as a loopback device
    webcam.write_text("")
    with pytest.raises(vcam.Unavailable) as not_loopback:
        vcam.choose_loopback_device(str(webcam), state)
    assert "is not a v4l2loopback device" in str(not_loopback.value)
    assert vcam.choose_loopback_device(str(webcam), vcam.LoopbackState(True, ())) == str(webcam)  # pyvirtualcam decides


def test_no_module_means_the_steps_to_load_it(sysfs):
    unload(sysfs)
    state = vcam.loopback_state()
    with pytest.raises(vcam.Unavailable) as error:
        vcam.choose_loopback_device(None, state)
    assert str(error.value) == "no v4l2loopback device: the v4l2loopback module is not loaded"
    assert vcam.LINUX_MODPROBE in error.value.hint
    assert "video_nr=10" in vcam.LINUX_MODPROBE and "exclusive_caps=1" in vcam.LINUX_MODPROBE
    secure = vcam.loopback_setup_hint(state, installed=True, secure_boot=True)
    assert "Key was rejected by service" in secure and "mokutil --import /var/lib/shim-signed/mok/MOK.der" in secure
    absent = vcam.loopback_setup_hint(state, installed=False)
    assert "linux-main-modules-v4l2loopback-$(uname -r)" in absent and "v4l2loopback-dkms" in absent
    empty = vcam.loopback_setup_hint(vcam.LoopbackState(True, ()))
    assert "loaded but has no device" in empty and "modprobe -r v4l2loopback" in empty


@pytest.mark.parametrize(
    "error, words",
    [
        ("Could not access /dev/video10 due to missing permissions. Did you add your user to the video group?", "usermod -aG video"),
        ("Device /dev/video10 is already in use.", "devices=2"),
        ("Virtual camera output could not be configured, Device or resource busy", "OBS"),
        ("Device /dev/video0 is not a video output device.", "--virtual-camera-device"),
        ("something else entirely", "modprobe v4l2loopback"),
    ],
)
def test_pyvirtualcam_errors_become_steps(error, words):
    assert words in vcam.loopback_open_hint(error, "/dev/video10")


# ---------------------------------------------------------------- v4l2loopback: pyvirtualcam


class FakeLinuxCamera:
    """Stands in for pyvirtualcam.Camera with its v4l2loopback backend."""

    made: list = []
    fail: Exception | None = None

    def __init__(self, width, height, fps, fmt=None, device=None, **kwargs):
        if FakeLinuxCamera.fail is not None:
            raise FakeLinuxCamera.fail
        self.args = {"width": width, "height": height, "fps": fps, "fmt": fmt, "device": device, **kwargs}
        self.device = device or "/dev/video2"
        self.backend = "v4l2loopback"
        self.sent = 0
        self.closed = False
        FakeLinuxCamera.made.append(self)

    def send(self, picture):
        assert picture.shape == (self.args["height"], self.args["width"], 3) and picture.dtype == np.uint8
        self.sent += 1

    def close(self):
        self.closed = True


@pytest.fixture
def fake_pyvirtualcam(monkeypatch):
    module = types.ModuleType("pyvirtualcam")
    module.PixelFormat = enum.Enum("PixelFormat", "RGB BGR I420")
    module.Camera = FakeLinuxCamera
    FakeLinuxCamera.made = []
    FakeLinuxCamera.fail = None
    monkeypatch.setitem(sys.modules, "pyvirtualcam", module)
    monkeypatch.setattr("sys.platform", "linux")
    return module


def test_linux_opens_the_glassesdata_device_with_bgr_frames(sysfs, fake_pyvirtualcam):
    camera = vcam.open_pyvirtualcam(1920, 1080, 30)
    assert camera.args == {"width": 1920, "height": 1080, "fps": 30, "fmt": fake_pyvirtualcam.PixelFormat.BGR, "device": "/dev/video10"}
    camera = vcam.open_pyvirtualcam(1280, 720, 30, device="/dev/video2")
    assert camera.args["device"] == "/dev/video2"


def test_linux_without_the_module_never_calls_pyvirtualcam(sysfs, fake_pyvirtualcam):
    unload(sysfs)
    with pytest.raises(vcam.Unavailable) as error:
        vcam.open_pyvirtualcam(1920, 1080, 30)
    assert "module is not loaded" in str(error.value) and "modprobe" in error.value.hint
    assert FakeLinuxCamera.made == []


def test_linux_open_failure_keeps_pyvirtualcams_reason_and_adds_steps(sysfs, fake_pyvirtualcam):
    FakeLinuxCamera.fail = RuntimeError("Could not access /dev/video10 due to missing permissions. Did you add your user to the 'video' group?")
    with pytest.raises(vcam.Unavailable) as error:
        vcam.open_pyvirtualcam(1920, 1080, 30)
    assert "missing permissions" in str(error.value) and "'" not in str(error.value)
    assert "sudo usermod -aG video $USER" in error.value.hint


def test_macos_ignores_the_device_and_opens_obs(fake_pyvirtualcam, monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    camera = vcam.open_pyvirtualcam(1920, 1080, 30, device="/dev/video10")
    assert camera.args["device"] is None


def test_status_shows_the_requested_device_the_loopback_devices_and_the_check(sysfs, fake_pyvirtualcam, monkeypatch):
    monkeypatch.setattr(vcam, "SELF_CHECK_S", 0.2)
    out = vcam.VirtualCamera(FakeReceiver(frame()), size=(1280, 720), fps=50, device="/dev/video10")
    status = out.status()
    assert status["requested_device"] == "/dev/video10" and status["running"] is False and status["hint"] is None
    assert status["loopback"] == {
        "module_loaded": True,
        "devices": [{"path": "/dev/video2", "label": "OBS Virtual Camera"}, {"path": "/dev/video10", "label": "GlassesData"}],
    }
    ok, message = out.start()
    assert ok and message == "Virtual camera on: /dev/video10 (GlassesData)"
    wait_for(lambda: out.status()["check"] is not None)
    status = out.status()
    assert status["device"] == "/dev/video10" and status["device_label"] == "GlassesData" and status["backend"] == "v4l2loopback"
    assert status["check"].startswith("v4l2loopback took ") and "1280x720 (as I420)" in status["check"]
    assert status["last_error"] is None
    out.close()
    assert FakeLinuxCamera.made[0].closed and FakeLinuxCamera.made[0].sent > 0


def test_status_without_a_device_says_what_to_do_before_any_start(sysfs, fake_pyvirtualcam):
    unload(sysfs)
    out = vcam.VirtualCamera(FakeReceiver(frame()))
    status = out.status()
    assert status["loopback"] == {"module_loaded": False, "devices": []}
    assert vcam.LINUX_MODPROBE in status["hint"] and status["last_error"] is None
    ok, message = out.start()
    assert not ok and message == "Virtual camera unavailable: no v4l2loopback device: the v4l2loopback module is not loaded"
    assert out.status()["device"] is None and vcam.LINUX_MODPROBE in out.status()["hint"]


def test_a_fake_camera_or_macos_has_no_loopback_status(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    assert vcam.VirtualCamera(FakeReceiver(), open_camera=lambda w, h, f: None).status()["loopback"] is None
    monkeypatch.setattr("sys.platform", "darwin")
    assert vcam.VirtualCamera(FakeReceiver()).status()["loopback"] is None


# ---------------------------------------------------------------- the command line


def test_virtual_camera_device_flag(monkeypatch):
    args = cli.parse_args(["--virtual-camera", "--virtual-camera-device", "/dev/video10"])
    assert args.virtual_camera_device == "/dev/video10"
    assert cli.parse_args([]).virtual_camera_device is None


def test_the_browser_is_only_opened_in_a_graphical_session(monkeypatch, caplog):
    import webbrowser

    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url) or True)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(cli.os, "geteuid", lambda: 1000)
    log = logging.getLogger("test")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with caplog.at_level(logging.WARNING):
        assert cli.open_browser("http://127.0.0.1:8765/", log) is False
    assert opened == [] and "No graphical session" in caplog.text
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert cli.open_browser("http://127.0.0.1:8765/", log) is True and opened == ["http://127.0.0.1:8765/"]
    monkeypatch.setattr(cli.os, "geteuid", lambda: 0)
    assert cli.open_browser("http://127.0.0.1:8765/", log) is False and len(opened) == 1
    monkeypatch.setattr(cli.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(webbrowser, "open", lambda url: False)
    with caplog.at_level(logging.WARNING):
        assert cli.open_browser("http://127.0.0.1:8765/", log) is False
    assert "No browser could be opened" in caplog.text


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP on this platform")
def test_sighup_stops_the_receiver_unless_nohup_ignores_it():
    saved = {name: signal.getsignal(getattr(signal, name)) for name in ("SIGINT", "SIGTERM", "SIGHUP")}
    try:
        stop = threading.Event()
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
        cli.install_signal_handlers(stop)
        signal.getsignal(signal.SIGHUP)(signal.SIGHUP, None)
        assert stop.is_set()
        stop = threading.Event()
        signal.signal(signal.SIGHUP, signal.SIG_IGN)  # as nohup leaves it
        cli.install_signal_handlers(stop)
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert stop.is_set()
    finally:
        for name, handler in saved.items():
            signal.signal(getattr(signal, name), handler)
