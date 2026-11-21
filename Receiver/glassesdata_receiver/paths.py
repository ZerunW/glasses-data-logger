"""Folders the receiver writes to, all inside the repository's gitignored ``local/`` folder.

The project keeps nothing outside its own folder, on every platform: the Wi-Fi identity,
camera calibration and body measurements in ``local/state``, the hand model in
``local/cache`` and the recordings in ``local/recordings``, next to ``local/venv``.
``--out`` and ``--state-dir`` move the recordings or the state elsewhere for one run.
"""

from __future__ import annotations

from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]  # glassesdata_receiver -> Receiver -> the repository


def local_dir() -> Path:
    """The repository's ``local/`` folder: everything this machine keeps for the project."""
    return REPO_DIR / "local"


def state_dir() -> Path:
    """Wi-Fi identity (certificate, key and pairing token), calibration and body measurements."""
    return local_dir() / "state"


def cache_dir() -> Path:
    """Downloaded files that can be fetched again (the hand model)."""
    return local_dir() / "cache"


def recordings_dir() -> Path:
    """Default folder for recordings."""
    return local_dir() / "recordings"


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
