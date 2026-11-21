"""Hand tracking on a finished recording, every frame: ``python -m glassesdata_receiver.offline DIR``.

The live tracker only takes the newest decoded frame and skips the rest when it
cannot keep up. This tool decodes all of ``video.mp4`` and writes
``hands_offline.csv`` with the same columns as ``hands.csv``; ``pts_us`` and
``phone_time_ns`` come from ``frames.csv`` by frame index.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import av

from .hands import HandTracker, ensure_model, handedness_note
from .recorder import HANDS_HEADER, hand_rows

log = logging.getLogger(__name__)


def read_frame_times(path: Path) -> list[tuple[int, int]]:
    """(pts_us, phone_time_ns) per frame, in frame_index order."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["frame_index"]))
    return [(int(row["pts_us"]), int(row["phone_time_ns"])) for row in rows]


def process(recording: Path, max_hands: int = 2, model: Path | None = None, mirrored_input: bool = False) -> str:
    """Write hands_offline.csv for one recording folder and return a one-line summary."""
    times = read_frame_times(recording / "frames.csv")
    tracker = HandTracker(model or ensure_model(), max_hands, mirrored_input)
    frames = frames_with_hands = hand_count = 0
    tracking_s = 0.0
    try:
        with av.open(str(recording / "video.mp4")) as container, open(
            recording / "hands_offline.csv", "w", newline="", encoding="utf-8"
        ) as output:
            writer = csv.writer(output)
            writer.writerow(HANDS_HEADER)
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                if frames >= len(times):
                    log.warning("video.mp4 has more frames than frames.csv; stopping at %d", frames)
                    break
                pts_us, phone_time_ns = times[frames]
                started = time.perf_counter()
                hands = tracker.detect(frame.to_ndarray(format="bgr24"), pts_us // 1000)
                tracking_s += time.perf_counter() - started
                writer.writerows(hand_rows(pts_us, phone_time_ns, hands))
                frames += 1
                frames_with_hands += bool(hands)
                hand_count += len(hands)
    finally:
        tracker.close()
    _note_handedness(recording / "info.json", handedness_note(mirrored_input))
    if frames < len(times):
        log.warning("frames.csv lists %d frames but video.mp4 decoded only %d", len(times), frames)
    per_frame = f"{tracking_s * 1000 / frames:.1f} ms/frame" if frames else "no frames"
    return (
        f"{recording.name}: {frames} frames, {frames_with_hands} with hands ({hand_count} hands), "
        f"{per_frame} -> {recording / 'hands_offline.csv'}"
    )


def _note_handedness(path: Path, note: str) -> None:
    """Record the label convention of hands_offline.csv next to the live one."""
    if not path.is_file():
        return
    info = json.loads(path.read_text())
    info["handednessOffline"] = note
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(info, indent=2, sort_keys=True))
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m glassesdata_receiver.offline",
        description="Track hands on every frame of a recording and write hands_offline.csv.",
    )
    parser.add_argument("recording", type=Path, help="a recording folder (holds video.mp4 and frames.csv)")
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument(
        "--mirrored-input", action="store_true",
        help="the video is mirrored (selfie style): swap MediaPipe's hand labels",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("video.mp4", "frames.csv"):
        if not (args.recording / name).is_file():
            parser.error(f"{args.recording} has no {name}")
    print(process(args.recording, args.max_hands, mirrored_input=args.mirrored_input))
    return 0


if __name__ == "__main__":
    sys.exit(main())
