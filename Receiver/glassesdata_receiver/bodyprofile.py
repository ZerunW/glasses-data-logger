"""Body measurements: ``<state dir>/profile.json``, in the format of PIPELINE.md section 4.

The page edits the six values a person can measure with a tape; the two eye-to-shoulder
offsets always keep their defaults. Values are metres; ``measured`` says which ones the
person entered (anything else is the adult average below).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

VERSION = 1

#: Adult averages (PIPELINE.md section 4), used for every value not measured.
DEFAULTS: dict[str, float] = {
    "hand_length_m": 0.187,
    "palm_length_m": 0.107,
    "palm_width_m": 0.084,
    "forearm_length_m": 0.26,
    "upper_arm_length_m": 0.32,
    "shoulder_width_m": 0.39,
    "eye_to_shoulder_drop_m": 0.20,
    "eye_to_shoulder_back_m": 0.07,
}
KEYS = tuple(DEFAULTS)

#: What the page asks for: key -> (label, definition shown to the person, plausible adult range in metres).
#: The definitions are PIPELINE.md's, word for word apart from capitalisation.
MEASURABLE: dict[str, tuple[str, str, tuple[float, float]]] = {
    "hand_length_m": ("Hand length", "Wrist crease to middle fingertip", (0.13, 0.25)),
    "palm_length_m": ("Palm length", "Wrist crease to the base crease of the middle finger", (0.075, 0.14)),
    "palm_width_m": ("Palm width", "Across the knuckles of index to little finger", (0.06, 0.115)),
    "forearm_length_m": ("Forearm", "Wrist crease to the tip of the elbow", (0.18, 0.36)),
    "upper_arm_length_m": ("Upper arm", "Shoulder tip (acromion) to the tip of the elbow", (0.22, 0.42)),
    "shoulder_width_m": ("Shoulder width", "Between the two shoulder tips", (0.28, 0.50)),
}


def defaults() -> dict[str, Any]:
    """The profile of someone who measured nothing."""
    return {"version": VERSION, **DEFAULTS, "measured": {key: False for key in KEYS}}


def fields() -> list[dict[str, Any]]:
    """The form, for the page: one entry per measurable value, in order."""
    return [
        {"key": key, "label": label, "definition": definition, "default_m": DEFAULTS[key], "min_m": low, "max_m": high}
        for key, (label, definition, (low, high)) in MEASURABLE.items()
    ]


def load(path: Path) -> tuple[dict[str, Any], str | None]:
    """The saved profile completed with defaults, and a note if the file was unusable.

    A value that is missing, not a positive number or outside its plausible range counts as
    not measured (the pipeline does the same with ``complete_profile``).
    """
    profile = defaults()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return profile, None
    except (OSError, ValueError) as error:
        return profile, f"{path.name} could not be read ({error}); showing the defaults"
    if not isinstance(raw, dict):
        return profile, f"{path.name} is not a JSON object; showing the defaults"
    measured = raw.get("measured") if isinstance(raw.get("measured"), dict) else {}
    for key in MEASURABLE:
        value = raw.get(key)
        if _plausible(key, value) and measured.get(key) is True:
            profile[key] = round(float(value), 4)
            profile["measured"][key] = True
    return profile, None


def update(current: dict[str, Any], changes: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """Apply ``changes`` (``{key: metres | null}``) to ``current``.

    A number marks the value measured; null puts it back to its default. Keys not named
    keep their current value. Returns the new profile and ``{key: problem}`` (empty when
    valid); on any problem nothing should be saved.
    """
    if not isinstance(changes, dict) or not changes:
        return current, {"": 'send a JSON object such as {"hand_length_m": 0.19}'}
    errors: dict[str, str] = {}
    unknown = sorted(set(changes) - set(MEASURABLE))
    if unknown:
        errors[""] = "unknown value " + ", ".join(repr(name) for name in unknown) + " (allowed: " + ", ".join(MEASURABLE) + ")"
    profile = {**current, "measured": dict(current["measured"])}
    for key, value in changes.items():
        if key not in MEASURABLE:
            continue
        label, _, (low, high) = MEASURABLE[key]
        if value is None:
            profile[key] = DEFAULTS[key]
            profile["measured"][key] = False
        elif not _number(value):
            errors[key] = f"{label}: a number of metres, or null for the default"
        elif not low <= float(value) <= high:
            errors[key] = f"{label}: between {low * 100:g} and {high * 100:g} cm"
        else:
            profile[key] = round(float(value), 4)
            profile["measured"][key] = True
    if "hand_length_m" not in errors and "palm_length_m" not in errors:
        if profile["palm_length_m"] >= profile["hand_length_m"] - 0.02:
            errors["palm_length_m"] = "Palm length: must be at least 2 cm shorter than the hand length"
    return profile, errors


def save(path: Path, profile: dict[str, Any]) -> None:
    """Write the profile atomically (a reader never sees half a file)."""
    data = {"version": VERSION, **{key: profile[key] for key in KEYS}, "measured": {key: bool(profile["measured"].get(key)) for key in KEYS}}
    write_json_atomic(path, data)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _plausible(key: str, value: Any) -> bool:
    if not _number(value):
        return False
    low, high = MEASURABLE[key][2]
    return low <= float(value) <= high
