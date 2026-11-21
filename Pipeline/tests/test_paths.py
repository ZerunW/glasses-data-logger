"""Default folders: everything stays inside the repository's local/ folder (PIPELINE.md section 0)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from glassesdata_pipeline import paths
from glassesdata_pipeline.backends import hamer
from glassesdata_pipeline.backends import mediapipe_hands as mph
from glassesdata_pipeline.backends import video_depth_anything as vda
from glassesdata_pipeline.errors import MissingDependency, PipelineError
from glassesdata_pipeline.steps import retarget

REPO = Path(__file__).resolve().parents[2]
LOCAL = REPO / "local"


def inside(path: Path, folder: Path = LOCAL) -> bool:
    return Path(path).resolve().is_relative_to(folder.resolve())


@pytest.fixture
def no_overrides(monkeypatch):
    for name in (paths.SRC_ENV, paths.MENAGERIE_ENV):
        monkeypatch.delenv(name, raising=False)


def test_default_folders_are_in_the_repository(no_overrides):
    assert paths.REPO_DIR == REPO and (REPO / "Pipeline" / "glassesdata_pipeline").is_dir()
    assert paths.local_dir() == LOCAL
    for folder in (paths.cache_dir(), paths.tmp_dir(), paths.recordings_dir(), paths.src_dir(), paths.menagerie_dir()):
        assert inside(folder), folder
    assert paths.src_dir() == LOCAL / "src"
    assert paths.menagerie_dir() == LOCAL / "src" / "mujoco_menagerie"
    assert mph.default_model_path() == LOCAL / "cache" / "hand_landmarker.task"  # shared with the receiver
    for name, value in paths.environment().items():
        assert inside(value), (name, value)


def test_environment_matches_the_shell_scripts():
    """scripts/local_dirs.sh exports the same variables, with the same folders, as paths.environment()."""
    text = (REPO / "Pipeline" / "scripts" / "local_dirs.sh").read_text()
    shell = {m[0]: m[1] for m in re.findall(r"^export (\w+)=\$\{\1:-\$LOCAL_DIR/([^}]*)\}$", text, re.M)}
    python = {name: path.relative_to(LOCAL).as_posix() for name, path in paths.environment().items()}
    assert shell == python


def test_use_local_dirs_keeps_variables_that_are_set():
    environ = {"HF_HOME": "/elsewhere/hf", "TORCH_HOME": ""}
    applied = paths.use_local_dirs(environ)
    assert environ["HF_HOME"] == "/elsewhere/hf" and "HF_HOME" not in applied
    assert environ["TORCH_HOME"] == str(LOCAL / "cache" / "torch") == applied["TORCH_HOME"]  # empty = not set
    assert set(environ) == set(paths.environment())
    assert all(inside(Path(value)) for name, value in environ.items() if name != "HF_HOME")


def test_importing_the_package_sets_the_caches():
    """``python -c "import glassesdata_pipeline"`` with none of the variables set points them all into local/."""
    env = {k: v for k, v in os.environ.items() if k not in paths.environment()}
    code = ("import json, os, glassesdata_pipeline as g\n"
            "print(json.dumps({k: os.environ.get(k) for k in g.paths.environment()}))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO / "Pipeline", env=env, capture_output=True, text=True,
                         check=True).stdout
    values = json.loads(out)
    assert values == {name: str(path) for name, path in paths.environment().items()}


def test_config_paths_are_relative_to_the_repository_root(no_overrides, tmp_path, monkeypatch):
    assert paths.resolve("local/src/x") == REPO / "local" / "src" / "x"
    assert paths.resolve("~/x") == Path.home() / "x"
    assert paths.resolve(tmp_path) == tmp_path
    monkeypatch.chdir(tmp_path)  # the working directory does not matter
    assert paths.resolve("local/src/x") == REPO / "local" / "src" / "x"


def test_src_and_menagerie_overrides(no_overrides, tmp_path, monkeypatch):
    monkeypatch.setenv(paths.SRC_ENV, str(tmp_path / "src"))
    assert paths.src_dir() == tmp_path / "src"
    assert paths.menagerie_dir() == tmp_path / "src" / "mujoco_menagerie"
    monkeypatch.setenv(paths.MENAGERIE_ENV, str(tmp_path / "menagerie"))
    assert paths.menagerie_dir() == tmp_path / "menagerie"  # "" = $MUJOCO_MENAGERIE first
    assert paths.menagerie_dir("does/not/exist") == tmp_path / "menagerie"  # a missing configured path falls back
    (tmp_path / "mine").mkdir()
    assert paths.menagerie_dir(tmp_path / "mine") == tmp_path / "mine"  # an existing configured path wins
    monkeypatch.delenv(paths.MENAGERIE_ENV)
    assert paths.menagerie_dir("does/not/exist") == REPO / "does" / "not" / "exist"


def test_retarget_reports_the_local_menagerie(no_overrides, tmp_path, monkeypatch):
    monkeypatch.setenv(paths.SRC_ENV, str(tmp_path))
    with pytest.raises(PipelineError, match=re.escape(str(tmp_path / "mujoco_menagerie"))):
        retarget.menagerie_path({"menagerie_path": ""})


def test_third_party_clones_default_to_local_src(no_overrides, tmp_path, monkeypatch):
    assert vda.repo_dir({}) == (LOCAL / "src" / "Video-Depth-Anything").resolve()
    assert vda.repo_dir({"vda_repo_dir": "local/src/vda"}) == (LOCAL / "src" / "vda").resolve()
    monkeypatch.setenv(paths.SRC_ENV, str(tmp_path))
    with pytest.raises(MissingDependency, match=re.escape(str((tmp_path / "WiLoR").resolve()))):
        hamer.repo_dir({"repo_dir": ""}, "wilor", "WiLoR", "WiLoR")
    (tmp_path / "hamer" / "hamer").mkdir(parents=True)
    try:
        assert hamer.repo_dir({}, "hamer", "HaMeR", "hamer") == (tmp_path / "hamer").resolve()
    finally:
        sys.path.remove(str((tmp_path / "hamer").resolve()))


def test_pytest_temp_folders_are_in_local_tmp(tmp_path, pytestconfig):
    basetemp = Path(pytestconfig.option.basetemp)
    if basetemp.resolve() != (LOCAL / "tmp" / "pytest-pipeline").resolve():
        pytest.skip(f"--basetemp {basetemp} given on the command line")
    assert inside(tmp_path, LOCAL / "tmp")


def test_make_tmp_dir(tmp_path):
    folder = paths.make_tmp_dir("test_paths_")
    try:
        assert folder.is_dir() and inside(folder, LOCAL / "tmp") and not any(folder.iterdir())
    finally:
        folder.rmdir()


def test_rerun_viewer_keeps_its_state_in_local(monkeypatch):
    from glassesdata_pipeline import viz

    for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.delenv(name, raising=False)
    env = viz.viewer_environment()
    assert inside(Path(env["XDG_CONFIG_HOME"])) and inside(Path(env["XDG_DATA_HOME"]))
    monkeypatch.setenv("XDG_CONFIG_HOME", "/elsewhere")
    assert viz.viewer_environment()["XDG_CONFIG_HOME"] == "/elsewhere"
