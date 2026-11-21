"""Configuration: merge TOML files and overrides, dotted access (PIPELINE.md section 6).

Precedence, lowest first:

1. every ``Pipeline/configs/*.toml`` (merged in file-name order; one file per section, so the
   loader never needs a fixed list - a new ``configs/slam.toml`` is picked up automatically),
2. ``<recording>/pipeline.toml``,
3. ``--config FILE``,
4. ``--set section.key=value`` (the value is parsed as TOML, falling back to a plain string).

Tables are merged recursively; any other value is replaced. Unknown keys are allowed.

>>> cfg = load_config(sets=["depth.downsample=2"])
>>> cfg["depth"]["downsample"], cfg.get("depth.downsample"), cfg.depth.downsample
(2, 2, 2)
>>> cfg.get("nope.missing", 5)
5

Attribute access (``cfg.depth``) is a convenience; use ``cfg["key"]`` for keys that clash with
dict methods (``items``, ``keys``, ``get``, ``copy``, ``update``, ``values``, ...).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .errors import ConfigError

log = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
"""``Pipeline/configs``; ``GLASSESDATA_PIPELINE_CONFIGS`` overrides it."""

RECORDING_CONFIG = "pipeline.toml"
_MISSING = object()


class Config(dict):
    """A nested dict with dotted-key access. Nested tables are :class:`Config` too."""

    def __init__(self, data: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        for key, value in dict(data or {}, **kwargs).items():
            self[key] = value

    # -- item access -------------------------------------------------------------------------

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, Mapping) and not isinstance(value, Config):
            value = Config(value)
        super().__setitem__(key, value)

    def __getitem__(self, key: str) -> Any:
        if dict.__contains__(self, key) or not (isinstance(key, str) and "." in key):
            return super().__getitem__(key)
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __contains__(self, key: object) -> bool:
        if dict.__contains__(self, key):
            return True
        return isinstance(key, str) and "." in key and self.get(key, _MISSING) is not _MISSING

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        """``cfg.get("depth.downsample", 4)``: dotted lookup with a default."""
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        node: Any = self
        for part in str(key).split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = dict.__getitem__(node, part) if isinstance(node, dict) else node[part]
        return node

    def section(self, name: str) -> "Config":
        """The table ``name`` (dotted allowed), or an empty Config when it does not exist."""
        value = self.get(name)
        return value if isinstance(value, Config) else Config()

    def set(self, key: str, value: Any) -> None:
        """``cfg.set("depth.downsample", 2)`` creates intermediate tables as needed."""
        parts = key.split(".")
        node: Config = self
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, Config):
                child = Config()
                node[part] = child
            node = child
        node[parts[-1]] = value

    # -- attribute access --------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return dict.__getitem__(self, name)
        except KeyError:
            raise AttributeError(f"config has no key {name!r}") from None

    # -- helpers -----------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """A plain, deep-copied dict (JSON-serialisable when the TOML values are)."""
        return {k: v.to_dict() if isinstance(v, Config) else copy.deepcopy(v) for k, v in self.items()}

    def copy(self) -> "Config":  # type: ignore[override]
        return Config(self.to_dict())

    def merged(self, other: Mapping[str, Any] | None) -> "Config":
        """A new Config with ``other`` merged on top (recursively)."""
        return Config(deep_merge(self.to_dict(), other or {}))


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``over`` into a copy of ``base``; tables merge, other values replace."""
    out: dict[str, Any] = {k: copy.deepcopy(dict(v) if isinstance(v, Config) else v) for k, v in base.items()}
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = _plain(value)
    return out


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    return copy.deepcopy(value)


def read_toml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse one TOML file; a syntax error becomes :class:`ConfigError` naming the file."""
    path = Path(path)
    try:
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: {error}") from error


def parse_value(text: str) -> Any:
    """Parse ``text`` as a TOML value (``2``, ``1.5``, ``true``, ``"x"``, ``[1, 2]``, ``{a = 1}``),
    falling back to the plain string (so ``--set hands2d.backend=gt`` works without quotes)."""
    try:
        return tomllib.loads(f"value = {text}")["value"]
    except tomllib.TOMLDecodeError:
        return text


def parse_set(item: str) -> dict[str, Any]:
    """``"depth.downsample=2"`` -> ``{"depth": {"downsample": 2}}``."""
    if "=" not in item:
        raise ConfigError(f"--set expects section.key=value, got {item!r}")
    key, _, text = item.partition("=")
    key = key.strip()
    if not key or any(not part for part in key.split(".")):
        raise ConfigError(f"--set has an invalid key: {item!r}")
    result: dict[str, Any] = {}
    node = result
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = parse_value(text.strip())
    return result


def config_dir() -> Path:
    return Path(os.environ.get("GLASSESDATA_PIPELINE_CONFIGS", CONFIG_DIR))


def load_defaults(directory: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Merge every ``*.toml`` in ``directory`` (default ``Pipeline/configs``) in name order."""
    directory = Path(directory) if directory is not None else config_dir()
    merged: dict[str, Any] = {}
    if not directory.is_dir():
        log.warning("config directory %s does not exist; using empty defaults", directory)
        return merged
    for path in sorted(directory.glob("*.toml")):
        data = read_toml(path)
        for section in data:
            if section in merged:
                log.debug("config section [%s] in %s is also defined in an earlier file; merging", section, path.name)
        merged = deep_merge(merged, data)
    return merged


def load_config(
    recording: str | os.PathLike[str] | None = None,
    config_file: str | os.PathLike[str] | None = None,
    sets: Iterable[str] = (),
    extra: Mapping[str, Any] | None = None,
    directory: str | os.PathLike[str] | None = None,
) -> Config:
    """The merged configuration.

    Args:
        recording: a recording folder; its ``pipeline.toml`` (if any) overrides the defaults.
        config_file: ``--config FILE``.
        sets: ``--set section.key=value`` strings, applied in order.
        extra: a dict merged last (highest precedence), e.g. ``Episode(config_overrides=...)``.
        directory: the defaults folder (default ``Pipeline/configs``).
    """
    merged = load_defaults(directory)
    if recording is not None:
        path = Path(recording) / RECORDING_CONFIG
        if path.is_file():
            merged = deep_merge(merged, read_toml(path))
    if config_file is not None:
        merged = deep_merge(merged, read_toml(config_file))
    for item in sets:
        merged = deep_merge(merged, parse_set(item))
    if extra:
        merged = deep_merge(merged, extra)
    return Config(merged)


def overrides_from_cli(config_file: str | os.PathLike[str] | None, sets: Iterable[str]) -> dict[str, Any]:
    """``--config`` and ``--set`` as one dict (``--set`` wins) for ``Episode(config_overrides=...)``."""
    merged: dict[str, Any] = read_toml(config_file) if config_file else {}
    for item in sets:
        merged = deep_merge(merged, parse_set(item))
    return merged


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)  # TOML dates and times


def normalize(value: Any) -> Any:
    """A JSON-compatible deep copy (TOML datetimes become strings); used for meta ``params``."""
    return json.loads(json.dumps(_jsonable(value), sort_keys=True))


def params_hash(params: Any) -> str:
    """A short stable hash of a JSON-compatible value."""
    blob = json.dumps(normalize(params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:16]
