"""Shared fixtures: a short synthetic recording generated once per test session."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

SYNTH_SECONDS = 2.0
SYNTH_SIZE = (180, 320)


@pytest.fixture(scope="session")
def synth_recording(tmp_path_factory) -> Path:
    """A 2 s, 180x320 synthetic recording (read-only: copy it before writing derived/)."""
    from glassesdata_pipeline.synth import generate

    out = tmp_path_factory.mktemp("synth") / "rec"
    generate(out, seconds=SYNTH_SECONDS, seed=0, size=SYNTH_SIZE)
    return out


@pytest.fixture
def recording_copy(synth_recording, tmp_path) -> Path:
    """A private copy of the synthetic recording (without derived/)."""
    dst = tmp_path / "rec"
    shutil.copytree(synth_recording, dst, ignore=shutil.ignore_patterns("derived"))
    return dst
