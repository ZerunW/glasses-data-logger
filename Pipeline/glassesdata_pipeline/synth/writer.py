"""Writing a recording folder in the receiver's format (video.mp4 + CSVs + JSON)."""

from __future__ import annotations

import csv
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from ..recording import (
    CLOCK_HEADER, DEVICE_HEADER, FRAMES_HEADER, HANDS_HEADER, INPUTS_HEADER, MOTION_HEADER,
)

MICROSECONDS = Fraction(1, 1_000_000)


class VideoWriter:
    """HEVC (libx265) in a fragmented MP4 like the receiver's: ``hvc1`` tag, no B-frames, a
    keyframe every ``gop`` frames, timestamps in microseconds. ``write(image_bgr, pts_us)``
    returns nothing; ``close()`` returns one ``(keyframe, bytes)`` per frame in order."""

    def __init__(self, path: Path, width: int, height: int, fps: float, crf: float = 16, preset: str = "fast",
                 gop: int = 24) -> None:
        import av  # lazy

        self._av = av
        self.container = av.open(str(path), "w", format="mp4",
                                 options={"movflags": "frag_keyframe+empty_moov+default_base_moof"})
        stream = self.container.add_stream("libx265", rate=Fraction(fps).limit_denominator(1001))
        stream.width, stream.height, stream.pix_fmt = int(width), int(height), "yuv420p"
        stream.codec_tag = "hvc1"
        stream.time_base = MICROSECONDS
        stream.codec_context.time_base = MICROSECONDS
        stream.options = {
            "preset": str(preset),
            "crf": str(crf),
            "x265-params": f"bframes=0:keyint={gop}:min-keyint={gop}:scenecut=0:log-level=error",
        }
        self.stream = stream
        self.packets: list[tuple[int, bool, int]] = []

    def _mux(self, packets: Any) -> None:
        for packet in packets:
            self.packets.append((int(packet.pts), bool(packet.is_keyframe), int(packet.size)))
            self.container.mux(packet)

    def write(self, image_bgr: np.ndarray, pts_us: int) -> None:
        frame = self._av.VideoFrame.from_ndarray(np.ascontiguousarray(image_bgr), format="bgr24")
        frame.pts = int(pts_us)
        frame.time_base = MICROSECONDS
        self._mux(self.stream.encode(frame))

    def close(self) -> list[tuple[bool, int]]:
        self._mux(self.stream.encode())
        self.container.close()
        self.packets.sort(key=lambda p: p[0])
        return [(key, size) for _, key, size in self.packets]


def _fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return "" if not np.isfinite(value) else f"{float(value):.9g}"
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([_fmt(v) for v in row])


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def write_recording_csvs(
    out: Path,
    frames: dict[str, np.ndarray],
    motion: dict[str, np.ndarray],
    t0_ns: int,
    clock_offset_ns: int,
    seconds: float,
) -> None:
    """frames.csv, motion.csv, inputs.csv (header), device_state.csv, clock.csv, hands.csv (header)."""
    write_csv(out / "frames.csv", FRAMES_HEADER, [
        [i, frames["pts_us"][i], frames["phone_ns"][i], frames["mac_ns"][i], int(frames["keyframe"][i]), frames["bytes"][i]]
        for i in range(len(frames["phone_ns"]))
    ])
    rows = []
    for i in range(len(motion["phone_ns"])):
        a, g = motion["acc"][i], motion["gyro"][i]
        rows.append([motion["phone_ns"][i], motion["glasses_ns"][i], a[0], a[1], a[2], g[0], g[1], g[2],
                     None, None, None, None, None, None, None, "glasses", motion["mac_ns"][i]])
    write_csv(out / "motion.csv", MOTION_HEADER, rows)
    write_csv(out / "inputs.csv", INPUTS_HEADER, [])
    write_csv(out / "device_state.csv", DEVICE_HEADER, [
        [t0_ns - 500_000_000, 0.82, "false", "true", "open", "nominal", "synthetic", t0_ns - 500_000_000 - clock_offset_ns],
    ])
    clock_rows = []
    for k in range(int(seconds) + 2):
        phone = t0_ns + k * 1_000_000_000 - 200_000_000
        rtt = 2_000_000 + 137_000 * (k % 5)
        mac_send = phone - clock_offset_ns - rtt // 2
        clock_rows.append([mac_send, mac_send + rtt, phone, rtt, clock_offset_ns])
    write_csv(out / "clock.csv", CLOCK_HEADER, clock_rows)
    write_csv(out / "hands.csv", HANDS_HEADER, [])
