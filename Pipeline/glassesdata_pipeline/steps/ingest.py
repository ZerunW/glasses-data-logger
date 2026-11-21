"""Step ``ingest``: raw recording files -> ``derived/timeline.npz``, ``camera.json``, ``profile.json``.

* Video frames: ``frames.csv`` sorted by ``frame_index``; T = number of video frames (demuxed
  packets of ``video.mp4``; extra ``frames.csv`` rows are dropped with a warning). Episode time
  ``t = (phone_time_ns - t0_ns) * 1e-9`` with ``t0_ns`` = ``phone_time_ns`` of frame 0.
* IMU: ``motion.csv`` rows sorted by ``phone_time_ns``; ``imu_t`` in episode seconds; values are
  stored raw (units are decided by the ``imu`` step); empty cells are NaN. A missing
  ``motion.csv`` gives N = 0 with a warning.
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
VERSION = 1
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
    phone = frames["phone_ns"]
    if np.any(np.diff(phone) <= 0):
        log.warning("frames.csv phone_time_ns is not strictly increasing (%d steps <= 0)", int(np.sum(np.diff(phone) <= 0)))
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


def _imu_arrays(ep: Episode, icfg, t0: int) -> dict[str, np.ndarray]:
    path = ep.path / "motion.csv"
    if not path.is_file():
        log.warning("%s has no motion.csv; the timeline has no IMU samples", ep.path.name)
        return _empty_imu()
    motion = read_motion(path)
    sources = [str(s) for s in icfg.get("imu_sources", ["glasses", "phone"])]
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
    }


def _empty_imu() -> dict[str, np.ndarray]:
    return {
        "imu_t": np.zeros(0),
        "acc": np.zeros((0, 3)),
        "gyro": np.zeros((0, 3)),
        "mag": np.zeros((0, 3)),
        "quat": np.zeros((0, 4)),
        "imu_is_glasses": np.zeros(0, dtype=bool),
    }
