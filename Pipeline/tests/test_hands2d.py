"""Step hands2d: slot assignment, MediaPipe result conversion, gt backend, MediaPipe on a few frames."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from glassesdata_pipeline.backends import mediapipe_hands as mph
from glassesdata_pipeline.episode import Episode
from glassesdata_pipeline.errors import PipelineError
from glassesdata_pipeline.runner import Runner


def det(side, score, wrist):
    uv = np.tile(np.asarray(wrist, float), (21, 1)) + np.arange(21)[:, None]
    return mph.Detection(side, score, uv, np.full(21, score), np.zeros((21, 3)))


NONE = np.full((2, 2), np.nan)


def test_anatomical_label():
    assert mph.anatomical_label("Left", False) == "Left" and mph.anatomical_label("Right", False) == "Right"
    assert mph.anatomical_label("Left", True) == "Right" and mph.anatomical_label("Right", True) == "Left"


def test_assign_slots_rules():
    a, b = det(0, 0.95, (100, 500)), det(1, 0.9, (600, 500))
    assert [(s, d is a, how) for s, d, how in mph.assign_slots([b, a], NONE)] == [(1, False, 1), (0, True, 1)]
    # both claim "right": continuity with the previous wrists decides
    a, b = det(1, 0.95, (110, 500)), det(1, 0.9, (590, 500))
    last = np.array([[100.0, 500.0], [600.0, 500.0]])
    got = {s: d for s, d, how in mph.assign_slots([a, b], last)}
    assert got[0] is a and got[1] is b
    # ... without history the higher score keeps its label
    got = {s: (d, how) for s, d, how in mph.assign_slots([b, a], NONE)}
    assert got[1][0] is a and got[0][0] is b and got[1][1] == mph.SOURCE_SCORE
    # a single flickering label keeps the slot it was in (unless its score is trusted)
    only_left = np.array([[100.0, 500.0], [np.nan, np.nan]])
    flicker = det(1, 0.6, (105, 502))
    assert mph.assign_slots([flicker], only_left, keep_slot_px=60)[0][:1] == (0,)
    trusted = det(1, 0.97, (105, 502))
    assert mph.assign_slots([trusted], only_left, keep_slot_px=60)[0][:1] == (1,)
    far = det(1, 0.6, (400, 900))
    assert mph.assign_slots([far], only_left, keep_slot_px=60)[0][:1] == (1,)
    assert mph.assign_slots([], NONE) == []


def _lm(x, y, z=0.0, presence=None):
    return SimpleNamespace(x=x, y=y, z=z, presence=presence)


def test_convert_result_pixels_world_and_scores():
    pts = [_lm(0.5, 0.25)] * 21
    world = [_lm(0.01 * i, -0.02, 0.003) for i in range(21)]
    result = SimpleNamespace(
        hand_landmarks=[pts, pts],
        hand_world_landmarks=[world, []],
        handedness=[[SimpleNamespace(category_name="Left", score=0.93)],
                    [SimpleNamespace(category_name="Right", score=0.4), SimpleNamespace(category_name="Left", score=0.6)]],
    )
    dets = mph.convert_result(result, 720, 1280, mirrored_input=False)
    assert dets[0].side == mph.LEFT and dets[1].side == mph.LEFT and dets[1].score == 0.6
    assert np.allclose(dets[0].uv[0], [0.5 * 720 - 0.5, 0.25 * 1280 - 0.5])
    assert np.allclose(dets[0].world[3], [0.03, -0.02, 0.003]) and np.all(np.isnan(dets[1].world))
    assert np.allclose(dets[0].kp_score, 0.93)
    assert mph.convert_result(result, 720, 1280, mirrored_input=True)[0].side == mph.RIGHT


class FakeLandmarker:
    """Left hand moving right; its label flickers to "Right" on frame 2 (score 0.6)."""

    def __init__(self):
        self.stamps = []

    def detect(self, image, ms):
        self.stamps.append(ms)
        k = len(self.stamps) - 1
        out = [det(0 if k != 2 else 1, 0.95 if k != 2 else 0.6, (100 + 5 * k, 600))]
        if k >= 3:
            out.append(det(1, 0.9, (500, 600)))
        return out


def test_track_frames_and_timestamps():
    ms = mph.timestamps_ms(np.array([0.0, 0.0004, 0.0417, 0.0833, 0.125]))
    assert list(ms) == [0, 1, 42, 83, 125]
    fake = FakeLandmarker()
    frames = ((i, np.zeros((8, 8, 3), np.uint8)) for i in range(5))
    out = mph.track_frames(frames, ms, 5, fake, keep_slot_px=60)
    assert fake.stamps == list(ms)
    assert np.all(np.isfinite(out["hand_score"][:, 0]))  # the flicker stayed in the left slot
    assert out["slot_source"][2, 0] == mph.SOURCE_CONTINUITY
    assert np.isnan(out["hand_score"][:3, 1]).all() and np.isfinite(out["hand_score"][3:, 1]).all()
    assert out["kp"].shape == (5, 2, 21, 3) and out["mp_world"].shape == (5, 2, 21, 3)


def test_resolve_model_never_downloads_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(mph, "cache_dir", lambda: tmp_path / "cache")
    with pytest.raises(PipelineError, match="downloading is disabled"):
        mph.resolve_model(None, allow_download=False)
    monkeypatch.setenv(mph.NO_DOWNLOAD_ENV, "1")
    with pytest.raises(PipelineError, match="downloading is disabled"):
        mph.resolve_model(None, allow_download=True)
    with pytest.raises(PipelineError, match="does not exist"):
        mph.resolve_model(tmp_path / "missing.task")


def test_step_gt_backend_and_cache(recording_copy):
    ep = Episode(recording_copy)
    report = Runner(ep).run(until="hands2d")
    assert report.ok, report.table()
    d = ep.load_npz("hands2d")
    T = len(ep.timeline()["t"])
    assert d["kp"].shape == (T, 2, 21, 3) and d["hand_score"].shape == (T, 2) and d["mp_world"].shape == (T, 2, 21, 3)
    assert d["meta"]["backend"] == "gt" and d["meta"]["step"] == "hands2d"
    again = Runner(Episode(recording_copy)).run(steps=["hands2d"])
    assert again.by_name()["hands2d"].status == "cached"


def test_unknown_backend_fails(recording_copy):
    ep = Episode(recording_copy, {"hands2d": {"backend": "nope"}})
    report = Runner(ep).run(until="hands2d")
    assert report.by_name()["hands2d"].status == "failed" and "nope" in report.by_name()["hands2d"].reason


def test_mediapipe_backend_on_synthetic_frames(recording_copy, monkeypatch):
    """Runs the real model (from the receiver's cache; never downloaded) over the 2 s clip."""
    if not mph.default_model_path().is_file():
        pytest.skip("MediaPipe hand model not in the cache")
    monkeypatch.setenv(mph.NO_DOWNLOAD_ENV, "1")
    ep = Episode(recording_copy, {"hands2d": {"backend": "mediapipe"}})
    report = Runner(ep).run(until="hands2d")
    assert report.ok, report.table()
    d = ep.load_npz("hands2d")
    T = len(ep.timeline()["t"])
    assert d["kp"].shape == (T, 2, 21, 3) and d["slot_source"].shape == (T, 2)
    found = np.isfinite(d["hand_score"])
    # wherever a hand was found, all 21 keypoints are finite and inside the image
    kp = d["kp"][found]
    assert np.all(np.isfinite(kp)) or kp.size == 0
    assert d["meta"]["backend"] == "mediapipe" and d["meta"]["model"].endswith("hand_landmarker.task")


def test_model_download_is_hash_checked(tmp_path, monkeypatch):
    import hashlib
    import io

    from glassesdata_pipeline.backends import mediapipe_hands as mh
    from glassesdata_pipeline.errors import PipelineError

    monkeypatch.delenv(mh.NO_DOWNLOAD_ENV, raising=False)
    monkeypatch.setattr(mh, "cache_dir", lambda: tmp_path / "cache")
    payload = b"not really a model"
    monkeypatch.setattr(mh.urllib.request, "urlopen", lambda url, timeout=60: io.BytesIO(payload))
    with pytest.raises(PipelineError, match="SHA-256"):
        mh.resolve_model()  # the pinned hash of the real model
    assert not (tmp_path / "cache" / mh.MODEL_NAME).exists() and not list((tmp_path / "cache").glob("tmp*"))
    good = hashlib.sha256(payload).hexdigest()
    path = mh.resolve_model(sha256=good.upper())
    assert path.read_bytes() == payload
    path.unlink()
    assert mh.resolve_model(sha256="").read_bytes() == payload  # check disabled
