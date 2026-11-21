"""Reading the raw files of a recording folder (PIPELINE.md section 2).

The CSV headers are copies of ``Receiver/glassesdata_receiver/recorder.py`` (the pipeline does not
import the receiver); ``tests/test_recording_format.py`` checks that they still agree.
Empty cells become NaN (floats) or -1 (integer columns that must exist are validated).
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import PipelineError

log = logging.getLogger(__name__)

LANDMARKS = 21
FRAMES_HEADER = ["frame_index", "pts_us", "phone_time_ns", "mac_time_ns", "keyframe", "bytes"]
MOTION_HEADER = [
    "phone_time_ns", "glasses_time_ns", "ax", "ay", "az", "gx", "gy", "gz",
    "mx", "my", "mz", "qw", "qx", "qy", "qz", "source", "mac_time_ns",
]
INPUTS_HEADER = ["phone_time_ns", "glasses_time_ms", "source", "type", "detail", "mac_time_ns"]
DEVICE_HEADER = ["phone_time_ns", "battery", "charging", "don", "hinge", "thermal", "link", "mac_time_ns"]
CLOCK_HEADER = ["mac_send_ns", "mac_receive_ns", "phone_time_ns", "rtt_ns", "offset_ns"]
HANDS_HEADER = (
    ["pts_us", "phone_time_ns", "hand_index", "handedness", "score"]
    + [f"{axis}{i}" for i in range(LANDMARKS) for axis in ("x", "y", "z")]
    + [f"w{axis}{i}" for i in range(LANDMARKS) for axis in ("x", "y", "z")]
)


def read_csv(path: str | os.PathLike[str]) -> dict[str, list[str]]:
    """Columns of a CSV file as lists of strings (header row required)."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = [name.strip() for name in next(reader)]
        except StopIteration:
            raise PipelineError(f"{path} is empty (no header row)") from None
        columns: dict[str, list[str]] = {name: [] for name in header}
        for row in reader:
            if not row:
                continue
            row = row + [""] * (len(header) - len(row))
            for name, value in zip(header, row):
                columns[name].append(value.strip())
    return columns


def to_float(values: list[str]) -> np.ndarray:
    """Strings -> float64; empty or unparsable cells -> NaN."""
    out = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        if value:
            try:
                out[i] = float(value)
            except ValueError:
                pass
    return out


def to_int(values: list[str], missing: int = -1) -> np.ndarray:
    """Strings -> int64 (exact for nanosecond timestamps); empty cells -> ``missing``."""
    out = np.full(len(values), missing, dtype=np.int64)
    for i, value in enumerate(values):
        if value:
            try:
                out[i] = int(value)
            except ValueError:
                try:
                    out[i] = int(float(value))
                except ValueError:
                    pass
    return out


def to_bool(values: list[str]) -> np.ndarray:
    return np.array([v.strip().lower() in ("1", "true", "yes", "y", "t") for v in values], dtype=bool)


def read_frames(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """``frames.csv`` -> ``frame_index, pts_us, phone_ns, mac_ns, keyframe, bytes`` sorted by frame_index."""
    cols = read_csv(path)
    for name in ("frame_index", "pts_us", "phone_time_ns"):
        if name not in cols:
            raise PipelineError(f"{path} has no column {name}")
    out = {
        "frame_index": to_int(cols["frame_index"]),
        "pts_us": to_int(cols["pts_us"]),
        "phone_ns": to_int(cols["phone_time_ns"]),
        "mac_ns": to_int(cols.get("mac_time_ns", [""] * len(cols["frame_index"]))),
        "keyframe": to_bool(cols.get("keyframe", ["0"] * len(cols["frame_index"]))),
        "bytes": to_int(cols.get("bytes", [""] * len(cols["frame_index"]))),
    }
    if np.any(out["phone_ns"] < 0):
        raise PipelineError(f"{path}: some rows have no phone_time_ns")
    order = np.argsort(out["frame_index"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def read_motion(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """``motion.csv`` -> ``phone_ns[N]`` int64, ``glasses_ns[N]`` float (NaN when empty),
    ``acc/gyro/mag[N,3]``, ``quat[N,4]`` ([w,x,y,z]) float with NaN for empty cells,
    ``source[N]`` str, ``mac_ns[N]`` int64. Rows without phone_time_ns are dropped."""
    cols = read_csv(path)
    n = len(cols.get("phone_time_ns", []))

    def block(names: list[str]) -> np.ndarray:
        return np.stack([to_float(cols.get(name, [""] * n)) for name in names], axis=-1) if n else np.zeros((0, len(names)))

    phone = to_int(cols.get("phone_time_ns", []))
    out = {
        "phone_ns": phone,
        "glasses_ns": to_float(cols.get("glasses_time_ns", [""] * n)),
        "acc": block(["ax", "ay", "az"]),
        "gyro": block(["gx", "gy", "gz"]),
        "mag": block(["mx", "my", "mz"]),
        "quat": block(["qw", "qx", "qy", "qz"]),
        "source": np.array(cols.get("source", [""] * n), dtype=str) if n else np.zeros(0, dtype=str),
        "mac_ns": to_int(cols.get("mac_time_ns", [""] * n)),
    }
    keep = phone >= 0
    if not np.all(keep):
        log.warning("%s: dropping %d rows without phone_time_ns", path, int(np.sum(~keep)))
        out = {k: v[keep] for k, v in out.items()}
    return out


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frames: int  # -1 when not counted
    fps: float
    codec: str


def probe_video(path: str | os.PathLike[str], count_frames: bool = True) -> VideoInfo:
    """Size, codec and (optionally, by demuxing without decoding) the number of frames."""
    import av  # lazy: PyAV is a core dependency but slow to import

    path = Path(path)
    if not path.is_file():
        raise PipelineError(f"{path} does not exist")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        ctx = stream.codec_context
        fps = float(stream.average_rate) if stream.average_rate else float("nan")
        frames = -1
        if count_frames:
            frames = sum(1 for packet in container.demux(stream) if packet.size)
        return VideoInfo(int(ctx.width), int(ctx.height), frames, fps, str(ctx.name))
