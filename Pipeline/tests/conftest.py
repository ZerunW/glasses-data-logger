"""Shared fixtures: a short synthetic recording generated once per test session."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

SYNTH_SECONDS = 2.0
SYNTH_SIZE = (180, 320)
BASETEMP = Path(__file__).resolve().parents[2] / "local" / "tmp" / "pytest-pipeline"


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Temporary test folders go in the repository's local/tmp, never the system temp folder
    (``--basetemp`` on the command line still wins; pytest empties that folder at every run)."""
    if not config.option.basetemp:
        BASETEMP.parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(BASETEMP)


@pytest.fixture(scope="session")
def synth_recording(tmp_path_factory) -> Path:
    """A 2 s, 180x320 synthetic recording with drawn hands (read-only: copy it before writing
    derived/). Its depth grid stays at the configured ``depth.downsample`` (45x80, fast; the
    tests are calibrated on it): ``synth.depth_width_px = 0`` keeps it out of pipeline.toml."""
    from glassesdata_pipeline.config import load_config
    from glassesdata_pipeline.synth import generate

    out = tmp_path_factory.mktemp("synth") / "rec"
    generate(out, seconds=SYNTH_SECONDS, seed=0, size=SYNTH_SIZE, config=load_config(sets=["synth.depth_width_px=0"]))
    return out


@pytest.fixture
def recording_copy(synth_recording, tmp_path) -> Path:
    """A private copy of the synthetic recording (without derived/)."""
    dst = tmp_path / "rec"
    shutil.copytree(synth_recording, dst, ignore=shutil.ignore_patterns("derived"))
    return dst
