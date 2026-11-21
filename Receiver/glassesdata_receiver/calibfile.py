"""Reading a stored ``calibration.json`` (PIPELINE.md section 3) safely.

One validator for every reader: the calibration page (``CalibrationStore``) and the
recorder, which copies the current calibration into each new recording. A file that
fails it is neither used nor copied, and the reason is shown instead. Standard library
only, so the recorder does not pull in OpenCV.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MODELS = ("pinhole", "fisheye")
DIST_SIZES = {"pinhole": (0, 4, 5, 8, 12, 14), "fisheye": (0, 4)}


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 100_000


def problem(data: Any) -> str | None:
    """Why ``data`` is not a usable calibration, or None when it is."""
    if not isinstance(data, dict):
        return "not a JSON object"
    model = data.get("model", "pinhole")
    if model not in MODELS:
        return f"unknown model {model!r} (pinhole or fisheye)"
    if not (_positive_int(data.get("width")) and _positive_int(data.get("height"))):
        return "width and height must be positive whole numbers"
    K = data.get("K")
    if not (isinstance(K, list) and len(K) == 3 and all(isinstance(row, list) and len(row) == 3 for row in K)):
        return "K must be a 3x3 matrix"
    if not all(_finite(value) for row in K for value in row):
        return "K must hold finite numbers"
    if not (float(K[0][0]) > 0 and float(K[1][1]) > 0):
        return "the focal lengths (fx, fy) must be positive"
    if [float(value) for value in K[2]] != [0.0, 0.0, 1.0] or float(K[1][0]) != 0.0:
        return "K must be [[fx, s, cx], [0, fy, cy], [0, 0, 1]]"
    dist = data.get("dist", [])
    if dist is None:
        dist = []
    if not (isinstance(dist, list) and all(_finite(value) for value in dist)):
        return "dist must be a list of finite numbers"
    if len(dist) not in DIST_SIZES[model]:
        sizes = ", ".join(str(size) for size in DIST_SIZES[model] if size)
        return f"a {model} dist has {sizes} values, not {len(dist)}"
    return None


def read(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """The calibration in ``path`` and None, or None and a note saying why it is not used
    (None and None when there is no file at all)."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, None
    except OSError as error:
        return None, f"{path.name} could not be read ({error.strerror or error}): not used"
    try:
        data = json.loads(text)
    except ValueError:
        return None, f"{path.name} is not valid JSON: not used"
    reason = problem(data)
    if reason is not None:
        return None, f"{path.name} is not a valid calibration ({reason}): not used"
    return data, None
