"""Per-platform folders and other platform branches, with sys.platform and the environment monkeypatched."""

import socket
from pathlib import Path

import pytest

from glassesdata_receiver import link, paths, session


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for variable in ("XDG_STATE_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(variable, raising=False)
    return tmp_path


def test_macos_keeps_the_existing_folders(home, monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setenv("XDG_STATE_HOME", "/elsewhere")  # ignored on macOS
    assert paths.state_dir() == home / "Library" / "Application Support" / "GlassesData"
    assert paths.cache_dir() == home / "Library" / "Caches" / "GlassesData"
    assert paths.recordings_dir() == home / "Movies" / "GlassesData"


def test_linux_defaults_follow_xdg(home, monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    assert paths.state_dir() == home / ".local" / "state" / "glassesdata"
    assert paths.cache_dir() == home / ".cache" / "glassesdata"
    assert paths.recordings_dir() == home / "Videos" / "GlassesData"


def test_linux_honours_xdg_variables(home, monkeypatch, tmp_path):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert paths.state_dir() == tmp_path / "state" / "glassesdata"
    assert paths.cache_dir() == tmp_path / "cache" / "glassesdata"


def test_relative_xdg_values_are_ignored(home, monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("XDG_STATE_HOME", "relative/state")
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    assert paths.state_dir() == home / ".local" / "state" / "glassesdata"
    assert paths.cache_dir() == home / ".cache" / "glassesdata"


def test_computer_name_on_macos(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr(socket, "gethostname", lambda: "Studio-Mac.local")
    assert session.computer_name() == "Studio-Mac"
    monkeypatch.setattr(socket, "gethostname", lambda: "")
    assert session.computer_name() == "Mac"


def test_computer_name_on_linux(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(socket, "gethostname", lambda: "lab-laptop.example.edu")
    missing = str(tmp_path / "missing")
    assert session.computer_name(missing) == "lab-laptop"
    info = tmp_path / "machine-info"
    info.write_text('CHASSIS=laptop\nPRETTY_HOSTNAME="Zé\'s Lab Laptop"\n', encoding="utf-8")
    assert session.computer_name(str(info)) == "Zé's Lab Laptop"
    info.write_text("PRETTY_HOSTNAME=\n")
    assert session.computer_name(str(info)) == "lab-laptop"


def test_usbmuxd_hint_per_platform(monkeypatch):
    error = OSError("could not connect to usbmuxd at '/var/run/usbmuxd'")
    monkeypatch.setattr("sys.platform", "darwin")
    assert link.usbmuxd_unavailable(error).startswith("usbmuxd unavailable (")
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(link.shutil, "which", lambda name: None)
    monkeypatch.setattr(link.os.path, "exists", lambda path: False)
    assert "sudo apt install usbmuxd" in link.usbmuxd_unavailable(error)
    monkeypatch.setattr(link.shutil, "which", lambda name: "/usr/sbin/usbmuxd")
    assert "plug in the iPhone" in link.usbmuxd_unavailable(error)


def test_wifi_addresses_skip_container_bridges(monkeypatch):
    wifi = pytest.importorskip("glassesdata_receiver.wifi")

    class Adapter:
        def __init__(self, name, *ips):
            self.name = self.nice_name = name
            self.ips = [type("IP", (), {"ip": ip})() for ip in ips]

    adapters = [
        Adapter("lo", "127.0.0.1"), Adapter("docker0", "172.17.0.1"), Adapter("br-12ab", "172.18.0.1"),
        Adapter("wlp2s0", "192.168.1.20", ("fe80::1", 0, 2)), Adapter("tailscale0", "100.100.1.2"),
    ]
    monkeypatch.setattr(wifi.ifaddr, "get_adapters", lambda: adapters)
    assert wifi.local_ipv4_addresses() == ["192.168.1.20", "100.100.1.2"]


def test_default_paths_are_used_by_the_modules():
    wifi = pytest.importorskip("glassesdata_receiver.wifi")
    hands = pytest.importorskip("glassesdata_receiver.hands")
    assert wifi.STATE_DIR == paths.state_dir()
    assert hands.DEFAULT_MODEL == paths.cache_dir() / "hand_landmarker.task"
    assert isinstance(paths.recordings_dir(), Path)
