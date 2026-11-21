import json

import numpy as np
import pytest

from glassesdata_pipeline import geometry as geo
from glassesdata_pipeline.episode import Episode
from glassesdata_pipeline.errors import PipelineError
from glassesdata_pipeline.recording import read_frames
from glassesdata_pipeline.runner import Runner


def test_save_load_and_meta(tmp_path):
    (tmp_path / "rec").mkdir()
    ep = Episode(tmp_path / "rec")
    assert not ep.has("thing")
    ep.save_npz("sub/thing", {"step": "x", "params": {"a": 1}}, arr=np.arange(3), scalar=2.5, name="hello")
    assert ep.has("sub/thing") and ep.has("sub/thing.npz")
    d = ep.load_npz("sub/thing")
    assert d["meta"]["params"] == {"a": 1} and list(d["arr"]) == [0, 1, 2]
    assert d["scalar"] == 2.5 and d["name"] == "hello"
    ep.save_json("info", {"k": [1, 2]})
    assert ep.load_json("info")["meta"]["step"] is None  # outside the runner
    ep.save_json("list.json", [{"id": "obj0"}])
    assert ep.load_json("list.json") == [{"id": "obj0"}]
    with pytest.raises(PipelineError):
        ep.save_npz("bad", None, obj=np.array([{}], dtype=object))
    with pytest.raises(PipelineError):
        ep.load_npz("missing")
    with pytest.raises(PipelineError):
        ep.file("../escape")
    assert not list(ep.derived.glob(".*tmp"))  # atomic writes leave no temporary files


def test_episode_basics(synth_recording):
    ep = Episode(synth_recording, {"depth": {"downsample": 2}})
    assert ep.config.get("depth.downsample") == 2
    assert ep.config.get("hands2d.backend") == "gt"  # from the recording's pipeline.toml
    assert ep.is_synthetic and ep.gt is not None
    cam = ep.camera()  # no derived/camera.json: resolved from calibration.json
    assert cam.source == "synthetic" and (cam.width, cam.height) == (180, 320)
    profile = ep.profile()
    assert profile["hand_length_m"] == 0.195 and len(profile["bone_lengths_m"]) == 20
    frames = list(ep.frames(indices=[0, 5, 47], downsample=2))
    assert [i for i, _ in frames] == [0, 5, 47] and frames[0][1].shape == (160, 90, 3)
    gray = next(ep.frames(indices=[3], gray=True))[1]
    assert gray.shape == (320, 180) and gray.dtype == np.uint8
    assert len(list(ep.frames())) == 48
    assert not (synth_recording / "derived").exists() or not any((synth_recording / "derived").iterdir())


def test_ingest_on_synthetic_recording(recording_copy):
    ep = Episode(recording_copy)
    report = Runner(ep).run(steps=["ingest"])
    assert report.ok and report.by_name()["ingest"].status == "ran"
    tl = ep.timeline()
    frames = read_frames(recording_copy / "frames.csv")
    T = len(frames["phone_ns"])
    assert T == 48 and len(tl["t"]) == T and tl["t"][0] == 0
    assert np.allclose(np.diff(tl["t"]), 1 / 24, atol=1e-9)
    assert tl["keyframe"][0] and tl["keyframe"][24] and tl["keyframe"].sum() == 2
    assert np.array_equal(tl["phone_ns"], frames["phone_ns"]) and tl["phone_ns"].dtype == np.int64
    gt = ep.gt
    assert np.allclose(tl["imu_t"], gt["imu_t"], atol=1e-9)  # episode seconds, offset NOT applied
    assert tl["acc"].shape == (len(tl["imu_t"]), 3) and np.all(np.isfinite(tl["acc"]))
    assert np.all(np.isnan(tl["mag"])) and np.all(np.isnan(tl["quat"]))  # empty csv cells
    assert np.all(tl["imu_is_glasses"])
    assert 0.9 < np.median(np.linalg.norm(tl["acc"], axis=1)) < 1.1  # raw values in g
    cam = ep.camera()
    assert cam.source == "synthetic" and np.allclose(cam.K, gt["K"])
    assert tl["meta"]["calibration"] == "synthetic" and tl["meta"]["step"] == "ingest"
    profile = ep.load_json("profile.json")
    assert profile["defaulted"] == []  # the synthetic profile.json lists every value
    assert profile["measured"]["hand_length_m"] is True and profile["measured"]["palm_length_m"] is False
    assert len(profile["bone_lengths_m"]) == 20
    assert Runner(ep).run(steps=["ingest"]).by_name()["ingest"].status == "cached"


def test_ingest_without_optional_files(recording_copy, caplog):
    for name in ("motion.csv", "calibration.json", "profile.json"):
        (recording_copy / name).unlink()
    ep = Episode(recording_copy)
    report = Runner(ep).run(steps=["ingest"])
    assert report.ok
    tl = ep.timeline()
    assert tl["imu_t"].shape == (0,) and tl["acc"].shape == (0, 3)
    cam = ep.camera()
    assert cam.source == "default" and cam.dist.size == 0 and (cam.width, cam.height) == (180, 320)
    assert len(ep.load_json("profile.json")["defaulted"]) == 8
    text = caplog.text
    assert "no motion.csv" in text and "no calibration.json" in text and "no profile.json" in text


def test_ingest_scales_calibration_and_handles_empty_cells(recording_copy):
    calib = json.loads((recording_copy / "calibration.json").read_text())
    big = geo.Camera.from_dict(calib).resized(360, 640)
    data = big.to_dict()
    data["source"] = "charuco-web"
    (recording_copy / "calibration.json").write_text(json.dumps(data))
    lines = (recording_copy / "motion.csv").read_text().splitlines()
    cells = lines[5].split(",")
    cells[3] = ""  # an empty accelerometer cell
    lines[5] = ",".join(cells)
    lines.append(lines[-1])  # a duplicated row
    (recording_copy / "motion.csv").write_text("\n".join(lines) + "\n")
    ep = Episode(recording_copy)
    assert Runner(ep).run(steps=["ingest"]).ok
    cam = ep.camera()
    assert cam.source == "charuco-web" and (cam.width, cam.height) == (180, 320)
    assert np.allclose(cam.K, geo.Camera.from_dict(calib).K)
    tl = ep.timeline()
    assert np.isnan(tl["acc"][4, 1]) and len(tl["imu_t"]) == len(lines) - 2
    data["width"], data["height"] = 360, 360
    (recording_copy / "calibration.json").write_text(json.dumps(data))
    report = Runner(Episode(recording_copy)).run(steps=["ingest"])
    assert not report.ok and "aspect" in report.by_name()["ingest"].reason


def test_ingest_keeps_only_the_head_imu_by_default(recording_copy):
    """The app writes the SDK's MotionSource: "glasses" (head), "neuralBand" (wrist), "unknown".
    Only glasses rows are kept by default; other sources can be selected but stay marked."""
    lines = (recording_copy / "motion.csv").read_text().splitlines()
    header = lines[0].split(",")
    col = header.index("source")
    extra = []
    for line in lines[1:21]:
        cells = line.split(",")
        cells[col] = "neuralBand"
        extra.append(",".join(cells))
        cells[col] = "unknown"
        extra.append(",".join(cells))
    (recording_copy / "motion.csv").write_text("\n".join(lines + extra) + "\n")
    n_glasses = len(lines) - 1
    ep = Episode(recording_copy)
    assert Runner(ep).run(steps=["ingest"]).ok
    tl = ep.timeline()
    assert len(tl["imu_t"]) == n_glasses and np.all(tl["imu_is_glasses"])
    assert set(tl["imu_source"].tolist()) == {"glasses"}
    ep = Episode(recording_copy, {"ingest": {"imu_sources": ["glasses", "neuralBand"]}})
    assert Runner(ep).run(steps=["ingest"]).ok
    tl = ep.timeline()
    assert len(tl["imu_t"]) == n_glasses + 20
    assert np.array_equal(tl["imu_is_glasses"], tl["imu_source"] == "glasses")
    assert (tl["imu_source"] == "neuralBand").sum() == 20


def test_repair_frame_times():
    from glassesdata_pipeline.steps.ingest import repair_frame_times

    pts = 1_000_000 + np.arange(8) * 41_667
    phone = 5_000_000_000 + pts * 1000 + np.array([0, 3, -2, 1, 0, 2, -1, 0]) * 1_000_000  # jitter (ms)
    same, info = repair_frame_times(phone, pts)
    assert info is None and np.array_equal(same, phone)
    stall = phone.copy()
    stall[3] = stall[2]  # a repeated stamp
    fixed, info = repair_frame_times(stall, pts)
    assert np.all(np.diff(fixed) > 0) and info["time_repair"] == "pts_us" and info["repaired_frames"] == 1
    assert info["repaired_frame_indices"] == [3] and np.array_equal(np.delete(fixed, 3), np.delete(stall, 3))
    late = phone.copy()
    late[4] = phone[6] + 5_000_000  # one frame stamped far too late
    fixed, info = repair_frame_times(late, pts)
    assert np.all(np.diff(fixed) > 0) and info["repaired_frame_indices"] == [4]
    assert abs(fixed[4] - phone[4]) < 3_000_000  # back near its true time
    bad_pts = np.zeros(8, np.int64)  # no usable pts_us: monotone interpolation
    fixed, info = repair_frame_times(late, bad_pts)
    assert np.all(np.diff(fixed) > 0) and info["time_repair"] == "interpolation"
    tail = phone.copy()
    tail[-1] = tail[-2]
    fixed, _ = repair_frame_times(tail, bad_pts)
    assert np.all(np.diff(fixed) > 0)


def test_ingest_repairs_non_increasing_frame_times(recording_copy, caplog):
    lines = (recording_copy / "frames.csv").read_text().splitlines()
    header = lines[0].split(",")
    col = header.index("phone_time_ns")
    cells = [line.split(",") for line in lines[1:]]
    cells[10][col] = cells[9][col]  # frame 10 arrives with frame 9's stamp
    (recording_copy / "frames.csv").write_text("\n".join([lines[0]] + [",".join(c) for c in cells]) + "\n")
    ep = Episode(recording_copy)
    assert Runner(ep).run(steps=["ingest"]).ok
    tl = ep.timeline()
    assert np.all(np.diff(tl["t"]) > 0) and abs(tl["t"][10] - 10 / 24) < 1e-3
    assert tl["meta"]["time_repair"] == "pts_us" and tl["meta"]["repaired_frames"] == 1
    assert "repaired 1 frame" in caplog.text


def test_visual_rates_skip_non_positive_time_steps():
    from glassesdata_pipeline import geometry as geo
    from glassesdata_pipeline.core import imu_calib as ic

    t = np.array([0.0, 0.1, 0.1, 0.3])
    R = np.stack([geo.rot_z(0.1 * k) for k in range(4)])
    t0, t1, w, idx = ic.visual_rates(t, R, np.ones(4, bool))
    assert list(idx) == [0, 2] and np.all(np.isfinite(w)) and np.all(t1 > t0)
    assert w[0] == pytest.approx([0, 0, 1.0]) and w[1] == pytest.approx([0, 0, 0.5])


def test_frames_checks_the_timeline(recording_copy):
    ep = Episode(recording_copy)
    assert len(list(ep.frames(indices=[47]))) == 1  # no timeline yet: plain decoding
    with pytest.raises(PipelineError, match="decodes to 48 frames"):
        list(ep.frames(indices=[50]))
    assert Runner(ep).run(steps=["ingest"]).ok
    with pytest.raises(PipelineError, match="timeline has frames 0..47"):
        list(ep.frames(indices=[48]))
    tl = ep.timeline()
    meta = tl.pop("meta")
    ep.save_npz("timeline", meta, **{**tl, "t": tl["t"][:40]})  # a shorter timeline: frames beyond it are not decoded
    assert [i for i, _ in ep.frames()][-1] == 39
    ep.save_npz("timeline", meta, **{**tl, "t": np.r_[tl["t"], tl["t"][-1] + [0.04, 0.08]]})  # longer than the video
    with pytest.raises(PipelineError, match="decodes to 48 frames but the timeline has 50"):
        list(ep.frames())
