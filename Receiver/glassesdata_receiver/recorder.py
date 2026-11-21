"""One recording folder on the Mac: the original HEVC stream plus every sensor row.

video.mp4 is a fragmented MP4 written without re-encoding, so an interrupted
recording stays playable up to its last complete fragment. All CSV timestamps
keep their source clock domain:

* ``pts_us``         glasses video presentation time (from the SDK sample buffer)
* ``phone_time_ns``  iPhone monotonic clock when the phone received the item
* ``mac_time_ns``    Mac monotonic clock (``time.monotonic_ns``) when it arrived here

``clock.csv`` holds the PING/PONG samples used to map the phone clock onto the Mac clock.

When the receiver has a state folder, a new recording also gets a copy of the camera
calibration (``calibration.json``) and the body measurements (``profile.json``) in use
when it started (PIPELINE.md sections 3 and 4); ``info.json`` says what was copied, or why
nothing was.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import threading
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import av

from . import __version__, bodyprofile, calibfile, paths
from .hevc import hvcc_to_annexb, length_prefixed_to_annexb
from .protocol import VideoFormat, VideoPacket

MICROSECONDS = Fraction(1, 1_000_000)
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


def hand_rows(pts_us: int, phone_time_ns: int, hands: Iterable[Any]) -> list[list[Any]]:
    """hands.csv rows for one frame (also used by offline.py for hands_offline.csv)."""
    rows = []
    for index, hand in enumerate(hands):
        row: list[Any] = [pts_us, phone_time_ns, index, hand.handedness, round(hand.score, 4)]
        row += [round(v, 6) for point in hand.landmarks for v in point]
        row += [round(v, 6) for point in hand.world_landmarks for v in point]
        rows.append(row)
    return rows


def _cell(value: Any) -> Any:
    return "" if value is None else value


def _vector(body: dict[str, Any], key: str, size: int) -> list[Any]:
    value = body.get(key)
    if isinstance(value, list) and len(value) == size:
        return value
    return [""] * size


def attach_setup_files(state_dir: Path | None, directory: Path, fmt: VideoFormat) -> dict[str, Any]:
    """Copy the current calibration and body measurements into a new recording.

    Returns the ``info.json`` entries that describe them: ``calibration`` and ``profile``,
    each ``{"file": name or None, ...}`` with a ``note`` when something is missing, unusable or
    could not be written, or when the calibration was made at another video size (it is still
    copied; the pipeline rescales K for the same aspect ratio and refuses a different one).
    Each file is handled on its own: one failing never changes what is said about the other.
    """
    if state_dir is None:
        note = "this receiver runs without a state folder"
        return {"calibration": {"file": None, "note": note}, "profile": {"file": None, "note": note}}
    return {
        "calibration": _safely("camera calibration", lambda: _attach_calibration(state_dir, directory, fmt)),
        "profile": _safely("body measurements", lambda: _attach_profile(state_dir, directory)),
    }


def _safely(what: str, attach: Any) -> dict[str, Any]:
    try:
        return attach()
    except Exception as error:  # a copy failing must not stop the recording, nor touch the other file's entry
        return {"file": None, "note": f"could not copy the {what}: {error}"}


def _attach_calibration(state_dir: Path, directory: Path, fmt: VideoFormat) -> dict[str, Any]:
    data, note = calibfile.read(paths.current_calibration(state_dir))  # the same check as the calibration page
    if data is None:
        return {"file": None, "note": note or (
            "no camera calibration saved on this computer (calibrate on the web page): the pipeline uses its default camera"
        )}
    try:
        _write_json(directory / "calibration.json", data)  # exactly what was checked, even if the file changes meanwhile
    except OSError as error:
        return {"file": None, "note": f"could not copy the calibration: {error.strerror or error}"}
    entry = {"file": "calibration.json", **{key: data.get(key) for key in ("model", "width", "height", "rms_px", "created", "source")}}
    width, height = data["width"], data["height"]
    if (width, height) != (fmt.width, fmt.height):
        same_aspect = fmt.height > 0 and abs(width / height - fmt.width / fmt.height) <= 0.01 * fmt.width / fmt.height
        entry["note"] = (
            f"calibrated at {width}x{height}, this video is {fmt.width}x{fmt.height}: "
            + ("same aspect ratio, the pipeline rescales K" if same_aspect
               else "different aspect ratio, the pipeline will refuse it; calibrate again at this size")
        )
    return entry


def _attach_profile(state_dir: Path, directory: Path) -> dict[str, Any]:
    source = paths.profile_path(state_dir)
    if not source.exists():
        return {"file": None, "note": "no body measurements saved on this computer (enter them on the web page): the pipeline uses adult averages"}
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = None
    if not isinstance(raw, dict):
        return {"file": None, "note": f"{source} is not valid: not copied, the pipeline uses adult averages"}
    profile, note = bodyprofile.load(source)  # values that are not plausible are dropped, and the note says which
    try:
        bodyprofile.save(directory / "profile.json", profile)
    except OSError as error:
        return {"file": None, "note": f"could not copy the body measurements: {error.strerror or error}"}
    entry: dict[str, Any] = {"file": "profile.json", "measured": [key for key in bodyprofile.MEASURABLE if profile["measured"][key]]}
    if note:
        entry["note"] = note
    return entry


def _write_json(path: Path, data: Any, sort_keys: bool = False) -> None:
    """Write JSON atomically: a reader (or a crash) never sees half a file."""
    handle, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=sort_keys)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class Recording:
    """Thread-safe: rows may arrive from the network, decode and hand-tracking threads."""

    def __init__(self, root: Path, fmt: VideoFormat, metadata: dict[str, Any], setup_dir: Path | None = None) -> None:
        self._lock = threading.Lock()
        self.format = fmt
        self.directory = self._make_directory(root)
        self.started_at = datetime.now().astimezone()
        self.started_mac_ns = time.monotonic_ns()
        self.frames_written = 0
        self.hand_rows = 0
        self._closed = False
        self._metadata = dict(metadata)
        self._container: av.container.OutputContainer | None = None
        self._stream = None
        self._waiting_for_keyframe = True
        self._first_pts: int | None = None
        self._last_pts: int | None = None
        self._files: list[io.TextIOWrapper] = []
        self._frames = self._open_csv("frames.csv", FRAMES_HEADER)
        self._motion = self._open_csv("motion.csv", MOTION_HEADER)
        self._inputs = self._open_csv("inputs.csv", INPUTS_HEADER)
        self._device = self._open_csv("device_state.csv", DEVICE_HEADER)
        self._clock = self._open_csv("clock.csv", CLOCK_HEADER)
        self._hands = self._open_csv("hands.csv", HANDS_HEADER)
        self._metadata.update(attach_setup_files(setup_dir, self.directory, fmt))  # never raises: failures become notes
        self._write_info()

    # ------------------------------------------------------------ video

    def write_video(self, packet: VideoPacket, mac_time_ns: int) -> None:
        with self._lock:
            if self._closed:
                return
            if self._waiting_for_keyframe:
                if not packet.keyframe:
                    return
                self._open_video(packet)
                self._waiting_for_keyframe = False
            if self._last_pts is not None and packet.pts_us <= self._last_pts:
                return  # MP4 needs strictly increasing timestamps; the receiver rotates on a reset keyframe
            av_packet = av.Packet(length_prefixed_to_annexb(packet.data))
            av_packet.pts = av_packet.dts = packet.pts_us
            av_packet.time_base = MICROSECONDS
            av_packet.stream = self._stream
            av_packet.is_keyframe = packet.keyframe
            self._container.mux(av_packet)
            if self._first_pts is None:
                self._first_pts = packet.pts_us
            self._last_pts = packet.pts_us
            self._frames.writerow(
                [self.frames_written, packet.pts_us, packet.phone_time_ns, mac_time_ns, int(packet.keyframe), len(packet.data)]
            )
            self.frames_written += 1

    def _open_video(self, first_keyframe: VideoPacket) -> None:
        # PyAV can only copy codec parameters from an existing stream. Build one from the
        # parameter sets and the first keyframe so the muxer never opens an encoder.
        seed = av.open(
            io.BytesIO(hvcc_to_annexb(self.format.hvcc) + length_prefixed_to_annexb(first_keyframe.data)),
            format="hevc",
        )
        template = seed.streams.video[0]
        self._container = av.open(
            str(self.directory / "video.mp4"), "w", format="mp4",
            options={"movflags": "frag_keyframe+empty_moov+default_base_moof"},
        )
        self._stream = self._container.add_stream_from_template(template)
        self._stream.time_base = MICROSECONDS
        self._stream.codec_tag = "hvc1"  # QuickTime and AVFoundation need hvc1, not hev1
        seed.close()

    # ------------------------------------------------------------ sensors

    def write_motion(self, body: dict[str, Any], mac_time_ns: int) -> None:
        row = [_cell(body.get("t")), _cell(body.get("gt"))]
        row += _vector(body, "a", 3) + _vector(body, "g", 3) + _vector(body, "m", 3) + _vector(body, "q", 4)
        row += [_cell(body.get("src")), mac_time_ns]
        self._write(self._motion, row)

    def write_input(self, body: dict[str, Any], mac_time_ns: int) -> None:
        self._write(
            self._inputs,
            [_cell(body.get(k)) for k in ("t", "gt_ms", "src", "type", "detail")] + [mac_time_ns],
        )

    def write_device_state(self, body: dict[str, Any], mac_time_ns: int) -> None:
        self._write(
            self._device,
            [_cell(body.get(k)) for k in ("t", "battery", "charging", "don", "hinge", "thermal", "link")] + [mac_time_ns],
        )

    def write_clock(self, mac_send_ns: int, mac_receive_ns: int, phone_time_ns: int, offset_ns: int) -> None:
        self._write(self._clock, [mac_send_ns, mac_receive_ns, phone_time_ns, mac_receive_ns - mac_send_ns, offset_ns])

    def write_hands(self, pts_us: int, phone_time_ns: int, hands: Iterable[Any]) -> None:
        rows = hand_rows(pts_us, phone_time_ns, hands)
        with self._lock:
            if self._closed or self._first_pts is None or pts_us < self._first_pts:
                return  # that frame is not in video.mp4 (decoded before the recording's first keyframe)
            self._hands.writerows(rows)
            self.hand_rows += len(rows)

    def _write(self, writer: Any, row: list[Any]) -> None:
        with self._lock:
            if not self._closed:
                writer.writerow(row)

    # ------------------------------------------------------------ lifecycle

    def update_metadata(self, **values: Any) -> None:
        with self._lock:
            self._metadata.update(values)

    def close(self) -> Path:
        with self._lock:
            if self._closed:
                return self.directory
            self._closed = True
            # Close everything even if one step fails (a full disk, say); report the first error.
            steps = [self._container.close] if self._container is not None else []
            steps += [handle.close for handle in self._files]
            error: Exception | None = None
            for step in steps:
                try:
                    step()
                except Exception as failure:
                    error = error or failure
            try:  # a failed close is recorded as such, never as a complete recording
                self._write_info(ended=True, failure=error)
            except Exception as failure:
                error = error or failure
            if error is not None:
                raise error
        return self.directory

    @property
    def last_pts_us(self) -> int | None:
        """pts of the last frame written to video.mp4, None before the first keyframe."""
        with self._lock:
            return self._last_pts

    @property
    def duration_s(self) -> float:
        return (time.monotonic_ns() - self.started_mac_ns) / 1e9

    def _write_info(self, ended: bool = False, failure: Exception | None = None) -> None:
        info = {
            "receiver": f"glassesdata-receiver {__version__}",
            "startedAt": self.started_at.isoformat(),
            "startedMacTimeNs": self.started_mac_ns,
            "video": {"width": self.format.width, "height": self.format.height, "configuredFps": self.format.fps},
            "clockNote": (
                "pts_us is the glasses video timeline; phone_time_ns is the iPhone monotonic clock; "
                "mac_time_ns is this Mac's monotonic clock on arrival. clock.csv maps phone to Mac time: "
                "mac_time ≈ phone_time - offset_ns (use the rows with the smallest rtt_ns)."
            ),
            **self._metadata,
        }
        if ended and failure is not None:
            # Not the "complete" fields: the recordings list shows it as interrupted, with the reason.
            info.update(closeError=str(failure) or type(failure).__name__, closeFailedAt=datetime.now().astimezone().isoformat())
        elif ended:
            info.update(
                endedAt=datetime.now().astimezone().isoformat(),
                endedMacTimeNs=time.monotonic_ns(),
                framesWritten=self.frames_written,
                firstPtsUs=self._first_pts,
                lastPtsUs=self._last_pts,
                handRows=self.hand_rows,
            )
        _write_json(self.directory / "info.json", info, sort_keys=True)

    def _open_csv(self, name: str, header: list[str]) -> Any:
        handle = open(self.directory / name, "w", newline="", encoding="utf-8", buffering=1 << 16)
        self._files.append(handle)
        writer = csv.writer(handle)
        writer.writerow(header)
        return writer

    @staticmethod
    def _make_directory(root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        base = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        candidate = root / base
        suffix = 2
        while True:
            try:
                candidate.mkdir()
                return candidate
            except FileExistsError:
                candidate = root / f"{base}_{suffix}"
                suffix += 1
