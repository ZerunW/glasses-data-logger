"""Per-platform folders: the Wi-Fi identity (state), the hand model (cache) and the recordings.

macOS keeps the paths the receiver has always used (an identity may already exist
there). Every other platform (Linux) follows the XDG base directory specification.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP = "GlassesData"  # macOS folders and the visible recordings folder
XDG_APP = "glassesdata"  # hidden XDG folders use the lower-case name, as is usual there


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _xdg(variable: str, fallback: str) -> Path:
    """An XDG base directory. The specification says to ignore relative values."""
    value = os.environ.get(variable, "")
    if value and os.path.isabs(value):
        return Path(value)
    return Path.home() / fallback


def state_dir() -> Path:
    """Wi-Fi identity: certificate, key and pairing token."""
    if _is_macos():
        return Path.home() / "Library" / "Application Support" / APP
    return _xdg("XDG_STATE_HOME", ".local/state") / XDG_APP


def cache_dir() -> Path:
    """Downloaded files that can be fetched again (the hand model)."""
    if _is_macos():
        return Path.home() / "Library" / "Caches" / APP
    return _xdg("XDG_CACHE_HOME", ".cache") / XDG_APP


def _xdg_user_dir(name: str, fallback: str) -> Path:
    """A folder from ``user-dirs.dirs`` (xdg-user-dirs), such as XDG_VIDEOS_DIR.

    Desktops write that file in the session's language, so the Videos folder may be called
    ``Vidéos`` or ``视频``. The file holds shell-style lines, ``XDG_VIDEOS_DIR="$HOME/Videos"``,
    with paths either absolute or relative to ``$HOME``. A missing file or entry, or an entry
    that is the home folder itself (the specification's way to turn a folder off), gives
    ``~/<fallback>``.
    """
    home = Path.home()
    config = _xdg("XDG_CONFIG_HOME", ".config") / "user-dirs.dirs"
    try:
        lines = config.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        lines = []
    key = f"XDG_{name}_DIR="
    for line in lines:
        line = line.strip()
        if not line.startswith(key):
            continue
        value = line[len(key):].strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1]
        if value.startswith("$HOME"):
            folder = home / value[len("$HOME"):].lstrip("/")
        elif value.startswith("/"):
            folder = Path(value)
        else:
            continue
        if folder != home:
            return folder
    return home / fallback


def recordings_dir() -> Path:
    """Default folder for recordings: ~/Movies/GlassesData on macOS, the Videos folder on Linux."""
    if _is_macos():
        return Path.home() / "Movies" / APP
    return _xdg_user_dir("VIDEOS", "Videos") / APP


# Files in the state folder that every new recording gets a copy of (PIPELINE.md sections 3 and 4).


def calibration_dir(state: Path) -> Path:
    """Camera calibrations: one timestamped file per save, ``current.json`` and ``screen.json``."""
    return state / "calibration"


def current_calibration(state: Path) -> Path:
    """The calibration copied into new recordings as ``calibration.json``."""
    return calibration_dir(state) / "current.json"


def profile_path(state: Path) -> Path:
    """Body measurements, copied into new recordings as ``profile.json``."""
    return state / "profile.json"
