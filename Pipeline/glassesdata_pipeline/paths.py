"""Folders the pipeline uses by default, all inside the repository's gitignored ``local/`` folder.

The project keeps nothing outside its own folder, on every machine (PIPELINE.md section 0). The
layout is shared with ``Receiver/glassesdata_receiver/paths.py`` and is relative to this clone:

* ``local/venv`` - the Mac's Python environment; ``local/venv-pipeline`` - the Linux machine's
  pipeline environment (``GLASSESDATA_PIPELINE_VENV`` overrides; see ``scripts/setup_ubuntu.sh``),
* ``local/src/`` - third-party clones (``GLASSESDATA_SRC`` overrides): ``mujoco_menagerie``
  (``MUJOCO_MENAGERIE`` overrides), ``WiLoR``, ``hamer``, ``Video-Depth-Anything``,
* ``local/cache/`` - downloads: ``hand_landmarker.task`` (the receiver's copy) and the library
  caches of :func:`environment` (Hugging Face, torch hub, matplotlib, Warp kernels, ...),
* ``local/tmp/`` - temporary files (pytest: ``local/tmp/pytest-pipeline``),
* ``local/recordings/`` - recordings (the receiver's default), ``local/smoke/`` - smoke-test logs.

:func:`use_local_dirs` runs when the package is imported, before any heavy package: it sets the
variables of :func:`environment` with ``os.environ.setdefault``, so a variable that is already
set still wins. ``scripts/local_dirs.sh`` exports the same variables for the shell scripts.

Paths in the configuration (``retarget.menagerie_path``, ``hands2d.model_path``,
``hands2d.repo_dir``, ``depth.vda_repo_dir``, ...) go through :func:`resolve`: ``~`` is expanded
and a relative path is relative to the repository root, not to the working directory.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parents[2]  # glassesdata_pipeline -> Pipeline -> the repository
SRC_ENV = "GLASSESDATA_SRC"
MENAGERIE_ENV = "MUJOCO_MENAGERIE"


def local_dir() -> Path:
    """The repository's ``local/`` folder: everything this machine keeps for the project."""
    return REPO_DIR / "local"


def cache_dir() -> Path:
    """Downloaded files that can be fetched again (models, library caches)."""
    return local_dir() / "cache"


def tmp_dir() -> Path:
    """Temporary files."""
    return local_dir() / "tmp"


def recordings_dir() -> Path:
    """Default folder for recordings (the receiver writes them here)."""
    return local_dir() / "recordings"


def resolve(value: str | os.PathLike[str]) -> Path:
    """A path from the configuration: ``~`` expanded; a relative path is relative to the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_DIR / path


def src_dir() -> Path:
    """Third-party clones: ``$GLASSESDATA_SRC``, else ``local/src``."""
    value = os.environ.get(SRC_ENV, "")
    return Path(value).expanduser().absolute() if value else local_dir() / "src"


def menagerie_dir(configured: Any = "") -> Path:
    """The MuJoCo Menagerie clone (it may not exist; the caller checks).

    ``configured`` (``retarget.menagerie_path``, :func:`resolve`) when it is set and exists, or when
    ``$MUJOCO_MENAGERIE`` is not set; otherwise ``$MUJOCO_MENAGERIE``; with neither,
    ``src_dir() / "mujoco_menagerie"`` (where ``scripts/setup_ubuntu.sh`` clones it)."""
    env = os.environ.get(MENAGERIE_ENV, "")
    if configured:
        path = resolve(str(configured))
        if path.is_dir() or not env:
            return path
    if env:
        return Path(env).expanduser()
    return src_dir() / "mujoco_menagerie"


def make_tmp_dir(prefix: str) -> Path:
    """A new, empty folder in ``local/tmp`` (the caller removes it)."""
    base = tmp_dir()
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=base))


def environment() -> dict[str, Path]:
    """Environment variables that keep library caches and temporary files inside ``local/``.

    Each library would otherwise write to the home folder or the system temp folder."""
    cache = cache_dir()
    return {
        # Linux libraries that follow the XDG base directory spec (pip, Warp's default, the NVIDIA
        # and Mesa OpenGL shader caches of MuJoCo's EGL / OSMesa rendering, fontconfig, ...)
        "XDG_CACHE_HOME": cache,
        "HF_HOME": cache / "huggingface",  # transformers + huggingface_hub: Depth Anything V2, SAM 2 weights
        "TORCH_HOME": cache / "torch",  # torch.hub downloads (backbones of WiLoR / HaMeR)
        "MPLCONFIGDIR": cache / "matplotlib",  # matplotlib's font cache (macOS: ~/.matplotlib)
        "WARP_CACHE_PATH": cache / "warp",  # warp-lang's compiled kernels (MuJoCo Warp), see warp.config.kernel_cache_dir
        "CUDA_CACHE_PATH": cache / "nvidia" / "ComputeCache",  # the CUDA driver's JIT cache (~/.nv/ComputeCache)
        "TORCHINDUCTOR_CACHE_DIR": cache / "torchinductor",  # torch.compile (/tmp/torchinductor_<user>)
        "TRITON_CACHE_DIR": cache / "triton",  # Triton kernels of torch.compile (~/.triton/cache)
        "YOLO_CONFIG_DIR": cache / "ultralytics",  # ultralytics settings + fonts (WiLoR's hand detector)
        "FVCORE_CACHE": cache / "iopath",  # detectron2 / iopath downloads (HaMeR's ViTDet person boxes)
        "PIP_CACHE_DIR": cache / "pip",
        "TMPDIR": tmp_dir(),  # Python's tempfile and most tools (a value set by the system still wins)
    }


def use_local_dirs(environ: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Set the variables of :func:`environment` that are not set (or empty); returns the ones it set.

    Call before importing torch, transformers, matplotlib, warp, ... (the package does it when it
    is imported). Creates ``local/tmp`` when ``TMPDIR`` is set here (tempfile skips a missing one)."""
    environ = os.environ if environ is None else environ
    applied: dict[str, str] = {}
    for name, path in environment().items():
        if not environ.get(name):
            environ[name] = applied[name] = str(path)
    if "TMPDIR" in applied:
        try:
            tmp_dir().mkdir(parents=True, exist_ok=True)
        except OSError:  # read-only checkout: tempfile then falls back to the system folder
            pass
    return applied
