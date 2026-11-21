"""Per-platform folders and other platform branches, with sys.platform and the environment monkeypatched."""

import socket
from pathlib import Path

import pytest

from glassesdata_receiver import link, paths, session


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for variable in ("XDG_STATE_HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(variable, str(tmp_path / variable.lower()))
    return tmp_path


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_every_folder_is_inside_the_repository(home, monkeypatch, platform):
    """Nothing outside the project folder: HOME and the XDG variables play no part."""
    monkeypatch.setattr("sys.platform", platform)
    repo = Path(__file__).resolve().parents[2]
    assert paths.REPO_DIR == repo and (repo / "Receiver" / "glassesdata_receiver").is_dir()
    assert paths.local_dir() == repo / "local"
    assert paths.state_dir() == repo / "local" / "state"
    assert paths.cache_dir() == repo / "local" / "cache"
    assert paths.recordings_dir() == repo / "local" / "recordings"
    for folder in (paths.state_dir(), paths.cache_dir(), paths.recordings_dir()):
        assert home not in folder.parents


def test_local_is_ignored_by_git():
    lines = (paths.REPO_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "local/" in lines


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


def test_usbmuxd_hint_per_platform(monkeypatch, tmp_path):
    monkeypatch.setattr(link, "USB_DEVICES", str(tmp_path))  # no iPhone on this fake bus, whatever is plugged in
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
