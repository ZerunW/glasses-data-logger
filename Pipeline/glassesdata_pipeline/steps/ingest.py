"""Step ``ingest``: raw recording files -> ``derived/timeline.npz``, ``camera.json``, ``profile.json``.

* Video frames: ``frames.csv`` sorted by ``frame_index``; T = number of video frames (demuxed
  packets of ``video.mp4``; extra ``frames.csv`` rows are dropped with a warning). Episode time
  ``t = (phone_time_ns - t0_ns) * 1e-9`` with ``t0_ns`` = ``phone_time_ns`` of frame 0.
  ``phone_time_ns`` that is not strictly increasing (a stalled or reordered arrival) is repaired
  (:func:`repair_frame_times`: the offending frames get ``pts_us`` x 1000 + the median offset,
  or monotone interpolation without usable ``pts_us``); the meta records ``time_repair`` and
  ``repaired_frames``.
* IMU: ``motion.csv`` rows from ``ingest.imu_sources`` (default only ``"glasses"``, the head IMU;
  the app also writes ``"neuralBand"``, a wrist IMU, and ``"unknown"``) sorted by
  ``phone_time_ns``; ``imu_t`` in episode seconds; ``imu_source[N]`` (extra) the source of each
  row. Values are stored raw: the Meta SDK documents m/s^2, rad/s and µT; the ``imu`` step checks
  the units. Empty cells are NaN. A missing ``motion.csv`` gives N = 0 with a warning.
* Camera: ``calibration.json`` (K rescaled to the video size; aspect mismatch is an error) or the
  ``[camera]`` default (``source = "default"``).
* Profile: ``profile.json`` completed with defaults, plus ``bone_lengths_m[20]`` and ``defaulted``.
"""

from __future__ import annotations

import logging

import numpy as np

from ..episode import Episode
from ..errors import PipelineError
from ..recording import read_frames, read_motion

NAME = "ingest"
VERSION = 3  # 2: imu_source, default imu_sources = ["glasses"]; 3: frame time repair
REQUIRES: list[str] = []
INPUTS = ["frames.csv", "video.mp4", "motion.csv", "calibration.json", "profile.json"]
OUTPUTS = ["timeline.npz", "camera.json", "profile.json"]
CONFIG_SECTIONS = ["ingest", "camera"]

log = logging.getLogger(__name__)


def run(ep: Episode, cfg) -> dict:
    icfg = cfg.section("ingest")
    timeline, notes = build_timeline(ep, icfg)
    camera = ep.recording_camera()
    profile = ep.recording_profile()
    meta = ep.step_meta(calibration=camera.source)
    ep.save_npz("timeline", {**meta, **notes}, **timeline)
    ep.save_json("camera.json", camera.to_dict(), meta)
    ep.save_json("profile.json", profile, meta)
    n_glasses = int(np.sum(timeline["imu_is_glasses"]))
    return {
        "frames": int(len(timeline["t"])),
        "duration_s": round(float(timeline["t"][-1]), 3) if len(timeline["t"]) else 0.0,
        "imu": int(len(timeline["imu_t"])),
        "imu_glasses": n_glasses,
        "camera": camera.source,
        "profile_defaults": len(profile["defaulted"]),
    }


def build_timeline(ep: Episode, icfg) -> tuple[dict[str, np.ndarray], dict]:
    """The arrays of ``timeline.npz`` and notes for the meta."""
    frames_path = ep.path / "frames.csv"
    if not frames_path.is_file():
        raise PipelineError(f"{ep.path} has no frames.csv")
    frames = read_frames(frames_path)
    n_rows = len(frames["frame_index"])
    if n_rows == 0:
        raise PipelineError(f"{frames_path} has no rows")
    if np.any(np.diff(frames["frame_index"]) != 1) or frames["frame_index"][0] != 0:
        log.warning("frames.csv frame_index is not 0..%d without gaps; using row order", n_rows - 1)
    notes: dict = {}
    n = n_rows
    if bool(icfg.get("count_video_frames", True)) and ep.video_path.is_file():
        n_video = ep.video_info(count_frames=True).frames
        notes["video_frames"] = n_video
        if n_video != n_rows:
            log.warning("video.mp4 has %d frames but frames.csv lists %d; using %d", n_video, n_rows, min(n_video, n_rows))
        n = min(n_video, n_rows)
    elif not ep.video_path.is_file():
        log.warning("%s has no video.mp4", ep.path.name)
    frames = {k: v[:n] for k, v in frames.items()}
    phone, repair = repair_frame_times(frames["phone_ns"], frames["pts_us"])
    if repair is not None:
        log.warning("frames.csv phone_time_ns is not strictly increasing (%d steps <= 0): repaired %d frame times "
                    "(%s)", repair["non_increasing_steps"], repair["repaired_frames"], repair["time_repair"])
        notes.update(repair)
    t0 = int(phone[0])
    out: dict[str, np.ndarray] = {
        "frame_index": np.arange(n, dtype=np.int64),
        "pts_us": frames["pts_us"].astype(np.int64),
        "phone_ns": phone.astype(np.int64),
        "t": (phone - t0).astype(np.float64) * 1e-9,
        "keyframe": frames["keyframe"].astype(bool),
    }
    out.update(_imu_arrays(ep, icfg, t0))
    notes["t0_ns"] = t0
    return out, notes


def repair_frame_times(phone_ns: np.ndarray, pts_us: np.ndarray) -> tuple[np.ndarray, dict | None]:
    """Strictly increasing frame times: ``(phone_ns, None)`` when they already are, else the
    repaired times and notes for the meta.

    With strictly increasing ``pts_us`` (the glasses' own video timeline), each non-increasing
    pair ``(i, i+1)`` marks the frame of the two that is further from ``pts_us * 1000 + offset``
    (``offset`` = median of ``phone_ns - pts_us * 1000``), and marked frames get that prediction;
    if that is still not increasing, every frame does. Without usable ``pts_us``: frames that do
    not exceed every earlier time are dropped and re-interpolated over the frame index (the end
    extrapolated with the median step)."""
    phone = np.asarray(phone_ns, dtype=np.int64)
    n = len(phone)
    steps = np.diff(phone)
    if n < 2 or np.all(steps > 0):
        return phone.copy(), None
    info: dict = {"non_increasing_steps": int(np.sum(steps <= 0))}
    pts = np.asarray(pts_us, dtype=np.int64)
    if len(pts) == n and np.all(np.diff(pts) > 0):
        pred = pts * 1000 + int(np.median(phone - pts * 1000))
        resid = np.abs(phone - pred)
        bad = np.zeros(n, dtype=bool)
        for i in np.flatnonzero(steps <= 0):
            bad[i if resid[i] > resid[i + 1] else i + 1] = True
        fixed = np.where(bad, pred, phone)
        info["time_repair"] = "pts_us"
        if not np.all(np.diff(fixed) > 0):
            fixed, bad = pred, pred != phone
            info["time_repair"] = "pts_us (all frames)"
    else:
        running = np.maximum.accumulate(phone)
        keep = np.r_[True, phone[1:] > running[:-1]]
        idx = np.flatnonzero(keep)
        fixed = np.interp(np.arange(n), idx, phone[idx].astype(np.float64))
        good_steps = np.diff(phone[idx])
        step = float(np.median(good_steps)) if len(good_steps) else 1.0
        tail = np.arange(n) > idx[-1]
        fixed[tail] = phone[idx[-1]] + (np.arange(n)[tail] - idx[-1]) * step
        fixed = np.round(fixed).astype(np.int64)
        bad = ~keep
        info["time_repair"] = "interpolation"
    info["repaired_frames"] = int(np.sum(bad))
    info["repaired_frame_indices"] = np.flatnonzero(bad)[:50].tolist()
    return fixed.astype(np.int64), info


def _imu_arrays(ep: Episode, icfg, t0: int) -> dict[str, np.ndarray]:
    path = ep.path / "motion.csv"
    if not path.is_file():
        log.warning("%s has no motion.csv; the timeline has no IMU samples", ep.path.name)
        return _empty_imu()
    motion = read_motion(path)
    sources = [str(s) for s in icfg.get("imu_sources", ["glasses"])]
    if "neuralBand" in sources:
        log.info("keeping neuralBand rows (a wrist IMU): imu_source marks them; they are never the head IMU")
    keep = np.isin(motion["source"], sources) if len(motion["source"]) else np.zeros(0, dtype=bool)
    if len(keep) and not np.all(keep):
        log.info("dropping %d motion rows from sources other than %s", int(np.sum(~keep)), sources)
    motion = {k: v[keep] for k, v in motion.items()}
    order = np.argsort(motion["phone_ns"], kind="stable")
    motion = {k: v[order] for k, v in motion.items()}
    if bool(icfg.get("drop_duplicate_imu", True)) and len(motion["phone_ns"]):
        keep = np.ones(len(motion["phone_ns"]), dtype=bool)
        for source in np.unique(motion["source"]):
            idx = np.nonzero(motion["source"] == source)[0]
            dup = np.diff(motion["phone_ns"][idx]) == 0
            keep[idx[1:][dup]] = False
        if not np.all(keep):
            log.info("dropping %d duplicate IMU rows", int(np.sum(~keep)))
            motion = {k: v[keep] for k, v in motion.items()}
    is_glasses = motion["source"] == "glasses"
    t = (motion["phone_ns"] - t0).astype(np.float64) * 1e-9
    gap_warn = float(icfg.get("imu_gap_warn_s", 0.1))
    tg = t[is_glasses]
    if len(tg) > 1 and np.max(np.diff(tg)) > gap_warn:
        log.warning("glasses IMU has gaps up to %.3f s", float(np.max(np.diff(tg))))
    if not np.any(is_glasses) and len(t):
        log.warning("motion.csv has no glasses rows (only %s)", sorted(set(motion["source"].tolist())))
    return {
        "imu_t": t,
        "acc": motion["acc"].astype(np.float64),
        "gyro": motion["gyro"].astype(np.float64),
        "mag": motion["mag"].astype(np.float64),
        "quat": motion["quat"].astype(np.float64),
        "imu_is_glasses": is_glasses.astype(bool),
        "imu_source": motion["source"].astype(str),
    }


def _empty_imu() -> dict[str, np.ndarray]:
    return {
        "imu_t": np.zeros(0),
        "acc": np.zeros((0, 3)),
        "gyro": np.zeros((0, 3)),
        "mag": np.zeros((0, 3)),
        "quat": np.zeros((0, 4)),
        "imu_is_glasses": np.zeros(0, dtype=bool),
        "imu_source": np.zeros(0, dtype=str),
    }
