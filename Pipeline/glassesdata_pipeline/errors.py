"""Exceptions shared by every pipeline module.

* :class:`PipelineError` - any expected, user-facing failure (bad input, missing step output).
* :class:`MissingDependency` - an optional heavy package is not installed. The message says
  what to install and which step needs it, e.g. ``pip install open3d (needed for step objects)``.
* :class:`ConfigError` - a configuration file or ``--set`` override cannot be used.
* :func:`require` - lazy import of an optional package that raises :class:`MissingDependency`.
"""

from __future__ import annotations

import importlib
from types import ModuleType


class PipelineError(Exception):
    """An expected failure with a message meant for the user."""


class ConfigError(PipelineError):
    """A configuration file or override is invalid."""


class MissingDependency(PipelineError, ImportError):
    """An optional (usually heavy) dependency is not installed.

    Subclasses ImportError so ``except ImportError`` also catches it.
    """


def require(module: str, pip_name: str | None = None, needed_for: str | None = None) -> ModuleType:
    """Import ``module`` lazily or raise :class:`MissingDependency` with an install hint.

    >>> np = require("numpy")
    >>> require("open3d", "open3d", "step objects")  # doctest: +SKIP
    MissingDependency: pip install open3d (needed for step objects)
    """
    try:
        return importlib.import_module(module)
    except ImportError as error:
        hint = f"pip install {pip_name or module.split('.')[0]}"
        if needed_for:
            hint += f" (needed for {needed_for})"
        raise MissingDependency(hint) from error
