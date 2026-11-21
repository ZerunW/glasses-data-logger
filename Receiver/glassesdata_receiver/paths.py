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


def recordings_dir() -> Path:
    """Default folder for recordings."""
    if _is_macos():
        return Path.home() / "Movies" / APP
    return Path.home() / "Videos" / APP
