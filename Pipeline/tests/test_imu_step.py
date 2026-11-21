"""Step ``imu`` and core.imu_calib."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/fabricate.py (importlib mode)
from fabricate import fabricate_upstream  # noqa: E402
from glassesdata_pipeline import geometry as geo  # noqa: E402
from glassesdata_pipeline.core import imu_calib as ic  # noqa: E402
from glassesdata_pipeline.episode import Episode  # noqa: E402
from glassesdata_pipeline.eval import metrics as M  # noqa: E402
from glassesdata_pipeline.runner import Runner  # noqa: E402
from glassesdata_pipeline.steps import imu  # noqa: E402


@pytest.fixture(scope="module")
def imu_rec(synth_recording, tmp_path_factory):
    dst = tmp_path_factory.mktemp("imu") / "rec"
    shutil.copytree(synth_recording, dst, ignore=shutil.ignore_patterns("derived"))
    fabricate_upstream(dst)
    report = Runner(Episode(dst)).run(steps=["slam", "imu"], deps=False)
    assert report.ok, report.table()
    return dst


# ---------------------------------------------------------------- core.imu_calib


def test_interval_mean_exact_for_piecewise_linear():
    t = np.array([0.0, 1.0, 2.0, 4.0])
    x = np.array([[0.0], [2.0], [2.0], [6.0]])
    m = ic.interval_mean(t, x, np.array([0.0, 0.5, 1.5, -1.0, 3.0]), np.array([1.0, 1.5, 3.0, 0.5, 3.5]))
    assert m[0, 0] == pytest.approx(1.0)
    assert m[1, 0] == pytest.approx((0.5 * (1.0 + 2.0) / 2 + 0.5 * 2.0) / 1.0)
    assert m[2, 0] == pytest.approx((0.5 * 2.0 + 1.0 * (2.0 + 4.0) / 2) / 1.5)
    assert np.isnan(m[3, 0])
    assert m[4, 0] == pytest.approx(4.5)
    xn = x.copy()
    xn[2] = np.nan
    assert np.isnan(ic.interval_mean(t, xn, np.array([1.5]), np.array([3.0]))[0, 0])


def _analytic_motion(t):
    level = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # camera looking along +x
    R = geo.rot_z(0.6 * np.sin(1.3 * t)) @ geo.rot_x(0.4 * np.sin(0.9 * t)) @ geo.rot_y(0.3 * np.sin(1.7 * t)) @ level
    p = np.stack([0.2 * np.sin(t), 0.15 * np.sin(1.4 * t), 0.05 * np.sin(2 * t)], axis=-1)
    a = np.stack([-0.2 * np.sin(t), -0.15 * 1.96 * np.sin(1.4 * t), -0.05 * 4 * np.sin(2 * t)], axis=-1)
    return R, p, a


def test_gravity_preintegration_recovers_gravity_and_bias():
    fps, rate = 24.0, 200.0
    t = np.arange(0, 6.0, 1 / fps)
    t_imu = np.arange(-0.2, 6.2, 1 / rate)
    R_cam_imu = geo.so3_exp([0.3, -0.2, 1.0])
    R_wc, p, _ = _analytic_motion(t)
    R_wc_i, _, a_i = _analytic_motion(t_imu)
    b = np.array([0.15, -0.1, 0.2])
    g = np.array([0.0, 0.0, -ic.G0])
    R_wi = R_wc_i @ R_cam_imu
    f = np.einsum("nji,nj->ni", R_wi, a_i - g) + b
    # The trajectory frame is rotated arbitrarily (like a SLAM frame).
    S = geo.make_T(geo.so3_exp([0.4, 0.1, -0.7]), [0.3, 0.0, 0.1])
    T_slam_cam = S @ geo.make_T(R_wc, p)
    out = ic.gravity_preintegration(t, T_slam_cam, np.ones(len(t), bool), t_imu, f, R_cam_imu, 0.0,
                                    window_s=0.5, substeps=8, bias_sigma=None)
    g_slam_true = S[:3, :3] @ g
    assert ic.angle_deg(out["g"], g_slam_true) < 0.05
    assert np.linalg.norm(out["bias"] - b) < 0.02
    scale = ic.gravity_preintegration(t, T_slam_cam, np.ones(len(t), bool), t_imu, f, R_cam_imu, 0.0,
                                      window_s=0.5, fix_g_norm=False, estimate_scale=True, bias_sigma=None)
    assert scale["scale"] == pytest.approx(1.0, abs=0.01)
    R_ws = ic.gravity_alignment(out["g"], T_slam_cam[0, :3, :3])
    assert ic.angle_deg(R_ws @ out["g"], [0, 0, -1]) < 1e-6
    f0 = (R_ws @ T_slam_cam[0, :3, :3])[:, 2]
    assert abs(math.atan2(f0[1], f0[0])) < 1e-9


# ---------------------------------------------------------------- step


def test_imu_step_on_synthetic(imu_rec):
    ep = Episode(imu_rec)
    d = ep.load_npz("imu")
    gt = ep.gt
    T = gt.n_frames
    assert d["R_cam_imu"].shape == (3, 3) and d["gyro_bias"].shape == (3,) and d["gravity_cam"].shape == (T, 3)
    assert d["R_world_slam"].shape == (3, 3)
    e = M.imu_errors(d, gt, gt["T_world_cam"])
    assert abs(e["time_offset_error_s"]) < 0.005
    assert e["R_cam_imu_error_deg"] < 1.0
    assert e["gyro_bias_error_rad_s"] < 0.003
    assert d["accel_scale"] == pytest.approx(9.80665)
    # A 2 s clip barely constrains the accelerometer bias (it is only observable through head
    # rotation): ~0.5-1.5 deg here, ~0.1 deg on the default 8 s clip.
    assert e["gravity_error_deg"]["median"] < 3.0
    q = json.loads(d["quality"])
    assert q["gravity"] == "imu" and q["gyro_units"] == "rad/s" and q["source"] == "glasses"
    assert q["accel_units"] == "g"
    assert np.allclose(np.linalg.norm(d["gravity_cam"], axis=1), 1.0)
    # R_world_slam: the first camera looks along +x
    slam = ep.load_npz("slam")
    first = int(np.flatnonzero(slam["valid"])[0])
    f0 = d["R_world_slam"] @ slam["T_slam_cam"][first, :3, 2]
    assert abs(math.atan2(f0[1], f0[0])) < 1e-6


def _rewrite_timeline(ep: Episode, **changes):
    tl = ep.timeline()
    meta = tl.pop("meta")
    tl.update(changes)
    ep.save_npz("timeline", meta, **tl)


def test_gyro_units_auto_detects_deg_per_s(imu_rec, tmp_path):
    dst = tmp_path / "rec"
    shutil.copytree(imu_rec, dst)
    ep = Episode(dst)
    _rewrite_timeline(ep, gyro=ep.timeline()["gyro"] * (180.0 / math.pi))
    res = imu.calibrate(ep, imu.params(ep.config))
    assert res["gyro_scale"] == pytest.approx(math.pi / 180.0)
    assert res["quality"]["gyro_units"] == "deg/s"
    assert abs(res["time_offset_s"] - float(ep.gt["imu_time_offset_s"])) < 0.005
    assert np.linalg.norm(res["gyro_bias"] - ep.gt["gyro_bias"]) < 0.003


def test_no_glasses_imu(imu_rec, tmp_path, caplog):
    dst = tmp_path / "rec"
    shutil.copytree(imu_rec, dst)
    ep = Episode(dst)
    tl = ep.timeline()
    _rewrite_timeline(ep, imu_is_glasses=np.zeros(len(tl["imu_t"]), bool))
    res = imu.calibrate(ep, imu.params(ep.config))
    assert res["quality"]["gravity"] == "none"
    assert np.allclose(res["R_world_slam"], np.eye(3)) and np.all(np.isnan(res["gravity_cam"]))
    assert any("no glasses IMU" in r.message for r in caplog.records)
    # The phone IMU only when asked for (here the samples are the glasses' re-labelled).
    p = imu.params(ep.config)
    p["use_phone_imu"] = True
    _rewrite_timeline(ep, imu_source=np.full(len(tl["imu_t"]), "phone"))
    res = imu.calibrate(ep, p)
    assert res["quality"]["source"] == "phone" and res["quality"]["gravity"] == "phone"
    # A Neural Band (wrist IMU) is never taken for the head, not even with use_phone_imu.
    _rewrite_timeline(ep, imu_source=np.full(len(tl["imu_t"]), "neuralBand"))
    res = imu.calibrate(ep, p)
    assert res["quality"]["gravity"] == "none"


def test_accelerometer_in_m_per_s2_like_the_sdk(imu_rec, tmp_path, caplog):
    """The Meta SDK documents m/s^2 (the synthetic recordings write g): same result, no extra
    warnings."""
    import logging

    ep = Episode(imu_rec)
    with caplog.at_level(logging.WARNING):
        ref = imu.calibrate(ep, imu.params(ep.config))
    warnings_g = sorted(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    caplog.clear()
    dst = tmp_path / "rec"
    shutil.copytree(imu_rec, dst)
    ep = Episode(dst)
    _rewrite_timeline(ep, acc=ep.timeline()["acc"] * 9.80665)
    with caplog.at_level(logging.WARNING):
        res = imu.calibrate(ep, imu.params(ep.config))
    warnings_ms2 = sorted(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert res["quality"]["accel_units"] == "m/s2" and res["accel_scale"] == 1.0
    assert warnings_ms2 == warnings_g
    assert np.allclose(res["R_cam_imu"], ref["R_cam_imu"], atol=1e-9)
    assert np.allclose(res["gravity_cam"], ref["gravity_cam"], atol=1e-6)
    assert np.allclose(res["accel_bias"], ref["accel_bias"], atol=1e-6)


def test_fixed_extrinsic(imu_rec):
    ep = Episode(imu_rec)
    p = imu.params(ep.config)
    p.update(extrinsic="fixed", R_cam_imu=np.asarray(ep.gt["R_cam_imu"]).tolist())
    res = imu.calibrate(ep, p)
    assert np.allclose(res["R_cam_imu"], ep.gt["R_cam_imu"], atol=1e-9)
    assert abs(res["time_offset_s"] - float(ep.gt["imu_time_offset_s"])) < 0.005


def test_duplicate_imu_stamps_and_cancelling_gravity_estimates(monkeypatch):
    # 12 rows but only 6 distinct stamps: not enough samples after removing duplicates
    t = np.repeat(np.arange(6) * 0.01, 2)
    tl = {"imu_t": t, "imu_is_glasses": np.ones(12, bool), "acc": np.tile([0, 0, 1.0], (12, 1)),
          "gyro": np.zeros((12, 3))}
    assert imu._select_samples(tl, imu.params({}))[0] is None
    tl2 = {"imu_t": np.r_[np.arange(12) * 0.01, 0.12, 0.12], "imu_is_glasses": np.ones(14, bool),
           "acc": np.tile([0, 0, 1.0], (14, 1)), "gyro": np.zeros((14, 3))}
    samples, _ = imu._select_samples(tl2, imu.params({}))
    assert samples is not None and len(samples["t"]) == 13 and np.all(np.diff(samples["t"]) > 0)
    # forward and backward Mahony estimates pointing opposite ways: no NaN
    down = np.tile([0.0, 0.0, -1.0], (5, 1))
    calls = iter([down, -down[::-1]])
    monkeypatch.setattr(imu.imf, "gravity_in_sensor", lambda _: next(calls))
    monkeypatch.setattr(imu.imf, "mahony", lambda *a, **k: None)
    g = imu._mahony_gravity(np.arange(5) * 0.01, np.zeros((5, 3)), np.zeros((5, 3)), imu.params({}))
    assert np.all(np.isfinite(g)) and np.allclose(g, down)
