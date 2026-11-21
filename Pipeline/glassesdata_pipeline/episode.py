"""``Episode``: one recording folder and its ``derived/`` outputs (PIPELINE.md section 7).

>>> ep = Episode("/path/to/recording", {"depth": {"downsample": 2}})
>>> tl = ep.timeline()                 # dict of arrays from derived/timeline.npz
>>> cam = ep.camera()                  # geometry.Camera
>>> for index, image in ep.frames(range(0, 100, 10), downsample=4):
...     ...
>>> ep.save_npz("hands2d", ep.step_meta(backend="gt"), kp=kp, hand_score=score)
>>> ep.load_npz("hands2d")["kp"]

Names given to ``save_npz/load_npz/save_json/load_json/has/file`` are paths relative to
``derived/``; ``.npz`` / ``.json`` is appended when the name has no suffix.

Metadata: every ``.npz`` gets a string array ``meta_json`` and every ``.json`` dict output a
``"meta"`` key (section 5). Inside a step run by the runner, :meth:`Episode.step_meta` returns the
correct object (step, version, params, inputs, created, backend, calibration); pass it (or
nothing: ``meta=None`` uses ``step_meta()``) to the save functions. The runner's cache compares
these meta objects with the current configuration and inputs.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import geometry as geo
from . import skeleton
from .config import Config, load_config, normalize
from .errors import PipelineError
from .groundtruth import GroundTruth
from .recording import VideoInfo, probe_video, read_json

log = logging.getLogger(__name__)

DERIVED = "derived"
_UNSET = object()


class Episode:
    """A recording folder.

    Attributes:
        path: the recording folder (absolute).
        config: merged :class:`~glassesdata_pipeline.config.Config` (configs/*.toml <
            recording pipeline.toml < ``config_overrides``).
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        config_overrides: Mapping[str, Any] | None = None,
        *,
        config_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_dir():
            raise PipelineError(f"recording folder {self.path} does not exist")
        self.config: Config = load_config(recording=self.path, extra=config_overrides, directory=config_dir)
        self._context: dict[str, Any] | None = None  # set by the runner while a step runs
        self._gt: Any = _UNSET
        self._video: VideoInfo | None = None

    def __repr__(self) -> str:
        return f"Episode({str(self.path)!r})"

    # ---------------------------------------------------------------- folders and files

    @property
    def derived(self) -> Path:
        """``<recording>/derived`` (created on first access)."""
        path = self.path / DERIVED
        path.mkdir(exist_ok=True)
        return path

    def file(self, name: str, suffix: str = "") -> Path:
        """Path of ``derived/<name>`` for writing (``suffix`` appended when ``name`` has none); its
        parent folder is created, so ``np.lib.format.open_memmap(ep.file("depth/depth_f16.npy"), ...)``
        works directly."""
        path = self._path(name, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _path(self, name: str, suffix: str = "") -> Path:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise PipelineError(f"derived file names must be relative: {name!r}")
        if suffix and not rel.suffix:
            rel = rel.with_suffix(suffix)
        return self.path / DERIVED / rel

    def has(self, name: str) -> bool:
        """True when ``derived/<name>`` exists (also tries ``.npz`` and ``.json``)."""
        if self._path(name).exists():
            return True
        if not Path(name).suffix:
            return any(self._path(name, s).exists() for s in (".npz", ".json"))
        return False

    def info(self) -> dict[str, Any]:
        """``info.json`` (receiver metadata), ``{}`` when absent."""
        path = self.path / "info.json"
        return read_json(path) if path.is_file() else {}

    @property
    def is_synthetic(self) -> bool:
        return bool(self.info().get("synthetic")) or (self.path / "gt" / "gt.npz").is_file()

    @property
    def gt(self) -> GroundTruth | None:
        """The ground truth of a synthetic recording, or None."""
        if self._gt is _UNSET:
            self._gt = GroundTruth(self.path / "gt") if (self.path / "gt" / "gt.npz").is_file() else None
        return self._gt

    @property
    def video_path(self) -> Path:
        return self.path / "video.mp4"

    def video_info(self, count_frames: bool = False) -> VideoInfo:
        """Size (and, if asked, the demuxed frame count) of ``video.mp4``."""
        if self._video is None or (count_frames and self._video.frames < 0):
            self._video = probe_video(self.video_path, count_frames=count_frames)
        return self._video

    # ---------------------------------------------------------------- derived loaders

    def timeline(self) -> dict[str, Any]:
        """``derived/timeline.npz`` (step ``ingest``) as a dict of arrays."""
        if not self.has("timeline.npz"):
            raise PipelineError(f"{self.path}: no derived/timeline.npz - run the ingest step first")
        return self.load_npz("timeline")

    def camera(self) -> geo.Camera:
        """The camera in use: ``derived/camera.json`` if ingest ran, else resolved from the
        recording (``calibration.json`` or the ``[camera]`` default)."""
        if self.has("camera.json"):
            return geo.Camera.from_dict(self.load_json("camera.json"))
        return self.recording_camera()

    def profile(self) -> dict[str, Any]:
        """The body profile in use (all keys, ``bone_lengths_m``, ``defaulted``)."""
        if self.has("profile.json"):
            return self.load_json("profile.json")
        return self.recording_profile()

    # ---------------------------------------------------------------- resolution from raw files

    def recording_camera(self) -> geo.Camera:
        """Camera from ``calibration.json`` (K rescaled to the video size), or the default camera
        from ``[camera]`` (``source = "default"``) when the recording has none."""
        info = self.video_info()
        path = self.path / "calibration.json"
        if path.is_file():
            try:
                data = read_json(path)
                camera = geo.Camera.from_dict(data, info.width, info.height)
                if (camera.width, camera.height) != (int(data["width"]), int(data["height"])):
                    log.info("calibration %sx%s rescaled to the video size %dx%d",
                             data["width"], data["height"], info.width, info.height)
                return camera
            except (PipelineError, ValueError, KeyError, json.JSONDecodeError) as error:
                raise PipelineError(f"{path}: {error}") from error
        hfov = float(self.config.get("camera.hfov_deg", 65.0))
        log.warning("%s has no calibration.json; using a default pinhole camera (hfov %.1f deg)", self.path.name, hfov)
        return geo.Camera.fov_default(info.width, info.height, hfov, source="default")

    def recording_profile(self) -> dict[str, Any]:
        """``profile.json`` completed with defaults (section 4) plus ``bone_lengths_m``."""
        path = self.path / "profile.json"
        raw = read_json(path) if path.is_file() else None
        if raw is None:
            log.warning("%s has no profile.json; using default body measurements", self.path.name)
        profile = skeleton.complete_profile(raw)
        profile["bone_lengths_m"] = skeleton.bone_lengths(profile).tolist()
        return profile

    # ---------------------------------------------------------------- video frames

    def frames(
        self, indices: Iterable[int] | None = None, downsample: float = 1, gray: bool = False
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Decode ``video.mp4`` with PyAV: yields ``(frame_index, image)``, BGR uint8 ``(h, w, 3)``
        (or ``(h, w)`` with ``gray``), downsampled with ``cv2.INTER_AREA`` to the grid of
        :func:`geometry.downsample_size`. ``indices`` (any iterable) selects frames; decoding stops
        after the last one.

        With a timeline (``ingest`` ran) the frames are those of the timeline: frames the video has
        beyond it are not decoded, and a requested frame outside it, or a video that ends before
        the timeline (or a requested frame) does, raises :class:`PipelineError` (the timeline and
        the video disagree: run ingest again)."""
        import av  # lazy
        import cv2

        n_timeline = len(self.timeline()["t"]) if self.has("timeline.npz") else None
        wanted = None if indices is None else sorted({int(i) for i in indices})
        if wanted is not None and not wanted:
            return
        if wanted is not None and n_timeline is not None and (wanted[0] < 0 or wanted[-1] >= n_timeline):
            raise PipelineError(f"{self.path.name}: frame {wanted[-1] if wanted[-1] >= n_timeline else wanted[0]} "
                                f"requested, the timeline has frames 0..{n_timeline - 1}")
        wanted_set = None if wanted is None else set(wanted)
        last = (n_timeline - 1 if n_timeline is not None else None) if wanted is None else wanted[-1]
        decoded = 0
        with av.open(str(self.video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for index, frame in enumerate(container.decode(stream)):
                if last is not None and index > last:
                    break
                decoded = index + 1
                if wanted_set is not None and index not in wanted_set:
                    continue
                image = frame.to_ndarray(format="gray" if gray else "bgr24")
                if downsample != 1:
                    size = geo.downsample_size(image.shape[1], image.shape[0], downsample)
                    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                yield index, image
        if last is not None and decoded <= last:
            expected = f"the timeline has {n_timeline} frames" if n_timeline is not None else f"frame {last} was requested"
            raise PipelineError(f"{self.video_path.name} of {self.path.name} decodes to {decoded} frames but {expected}: "
                                "the video and derived/timeline.npz disagree (run ingest again with --force)")

    # ---------------------------------------------------------------- metadata

    def step_meta(self, backend: str | None = None, **extra: Any) -> dict[str, Any]:
        """The ``meta`` object for outputs of the running step (section 5)."""
        ctx = self._context or {}
        meta: dict[str, Any] = {
            "step": ctx.get("step"),
            "version": ctx.get("version"),
            "params": ctx.get("params", {}),
            "inputs": ctx.get("inputs", {}),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "backend": backend,
            "calibration": self._calibration_source(),
        }
        if "params_hash" in ctx:
            meta["params_hash"] = ctx["params_hash"]
        meta.update(extra)
        return normalize(meta)

    def _calibration_source(self) -> str:
        path = self._path("camera.json")
        try:
            if path.is_file():
                return str(read_json(path).get("source", "default"))
            calib = self.path / "calibration.json"
            if calib.is_file():
                return str(read_json(calib).get("source", "unknown"))
        except (OSError, json.JSONDecodeError):
            pass
        return "default"

    # ---------------------------------------------------------------- save / load

    def save_npz(self, name: str, meta: Mapping[str, Any] | None = None, /, **arrays: Any) -> Path:
        """Write ``derived/<name>.npz`` atomically with ``meta_json`` (``meta=None`` -> ``step_meta()``).

        Values: numpy arrays, numbers, bools or strings (stored as 0-d arrays)."""
        path = self.file(name, ".npz")
        if "meta_json" in arrays:
            raise PipelineError("'meta_json' is reserved")
        meta = self.step_meta() if meta is None else normalize(meta)
        payload: dict[str, np.ndarray] = {}
        for key, value in arrays.items():
            array = np.asarray(value)
            if array.dtype == object:
                raise PipelineError(f"{name}.{key}: object arrays are not allowed (use save_json for structured data)")
            payload[key] = array
        payload["meta_json"] = np.asarray(json.dumps(meta))
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, lambda handle: np.savez(handle, **payload))
        return path

    def load_npz(self, name: str) -> dict[str, Any]:
        """``derived/<name>.npz`` as a dict; ``meta_json`` is parsed into ``"meta"`` and 0-d arrays
        become Python scalars/strings."""
        path = self._path(name, ".npz")
        if not path.is_file():
            raise PipelineError(f"{path} does not exist")
        return load_npz_file(path)

    def save_json(self, name: str, obj: Any, meta: Mapping[str, Any] | None = None, /) -> Path:
        """Write ``derived/<name>.json``. A dict gets ``"meta"`` (``meta=None`` -> ``step_meta()``);
        other JSON values (e.g. the list in ``objects/index.json``) are written unchanged."""
        path = self.file(name, ".json")
        if isinstance(obj, Mapping):
            obj = {**dict(obj), "meta": self.step_meta() if meta is None else normalize(meta)}
        text = json.dumps(normalize(obj), indent=2, sort_keys=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, lambda handle: handle.write(text.encode()))
        return path

    def load_json(self, name: str) -> Any:
        """``derived/<name>.json`` as stored (dicts include ``"meta"``)."""
        path = self._path(name, ".json")
        if not path.is_file():
            raise PipelineError(f"{path} does not exist")
        return read_json(path)


def load_npz_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            value = data[key]
            if key == "meta_json":
                out["meta"] = json.loads(str(value))
            elif value.ndim == 0:
                out[key] = value.item()
            else:
                out[key] = value
    return out


def read_meta(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The meta object of a derived ``.npz`` (``meta_json``) or ``.json`` (``"meta"``), else None."""
    path = Path(path)
    try:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                if "meta_json" in data.files:
                    return json.loads(str(data["meta_json"]))
        elif path.suffix == ".json":
            obj = read_json(path)
            if isinstance(obj, dict) and isinstance(obj.get("meta"), dict):
                return obj["meta"]
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return None


def _atomic_write(path: Path, write: Any) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
