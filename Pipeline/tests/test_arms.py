"""Step ``arms`` and core.forearm_cue."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/fabricate.py (importlib mode)
from fabricate import fabricate_upstream  # noqa: E402
from glassesdata_pipeline import geometry as geo  # noqa: E402
from glassesdata_pipeline import skeleton as sk  # noqa: E402
from glassesdata_pipeline.core import forearm_cue as fc  # noqa: E402
from glassesdata_pipeline.episode import Episode  # noqa: E402
from glassesdata_pipeline.runner import Runner  # noqa: E402
from glassesdata_pipeline.steps import arms  # noqa: E402


@pytest.fixture(scope="module")
def arms_rec(synth_recording, tmp_path_factory):
    dst = tmp_path_factory.mktemp("arms") / "rec"
    shutil.copytree(synth_recording, dst, ignore=shutil.ignore_patterns("derived"))
    fabricate_upstream(dst)
    report = Runner(Episode(dst)).run(steps=["slam", "imu", "world", "arms"], deps=False)
    assert report.ok, report.table()
    return dst


# ---------------------------------------------------------------- core.forearm_cue


def _scene(forearm_bgr, angle_deg=20.0):
    """A 240x320 textured image with a skin hand (fingers up) and a forearm going down at
    ``angle_deg`` from vertical; keypoints and a depth map (forearm 0.5 m, rest 1.0 m)."""
    rng = np.random.default_rng(0)
    h, w = 320, 240
    noise = cv2.resize((rng.random((40, 30, 3)) * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    img = (0.5 * noise + np.array([40, 90, 60])).astype(np.uint8)  # greenish clutter, not skin
    wrist = np.array([120.0, 140.0])
    d = np.array([math.sin(math.radians(angle_deg)), math.cos(math.radians(angle_deg))])
    far = wrist + d * 170
    depth = np.full((h, w), 1.0, np.float32)
    fore = np.zeros((h, w), np.uint8)
    cv2.line(fore, tuple(int(v) for v in wrist), tuple(int(v) for v in far), 1, 24)
    shade = np.linspace(0.75, 1.0, h)[:, None, None]
    img = np.where(fore[..., None] > 0, (np.asarray(forearm_bgr, np.float64) * shade).astype(np.uint8), img)
    depth[fore > 0] = 0.5 + 0.0005 * np.arange(h)[:, None].repeat(w, 1)[fore > 0]
    kp = np.zeros((21, 3))
    kp[0, :2] = wrist
    for f, base in enumerate([1, 5, 9, 13, 17]):
        x = 100 + 10 * f
        for k in range(4):
            kp[base + k, :2] = (x, 110 - 14 * k)
    kp[:, 2] = 0.9
    hull = cv2.convexHull(np.round(kp[:, :2]).astype(np.int32))
    cv2.fillConvexPoly(img, hull, (120, 150, 205))
    return img, kp, depth, d


def _angle(cue, d):
    a = (cue.p1 - cue.p0) / np.linalg.norm(cue.p1 - cue.p0)
    return math.degrees(math.acos(np.clip(a @ d, -1, 1)))


def test_forearm_axis_bare_forearm_skin():
    img, kp, depth, d = _scene((120, 150, 205))
    cue = fc.forearm_axis(img, kp, "skin")
    assert cue.ok and cue.conf > 0.3 and _angle(cue, d) < 5.0
    assert np.allclose(cue.p0, kp[0, :2])
    assert cue.extent_px > 60  # wedge = 3 wrist-to-knuckle lengths without forearm_px


def test_forearm_axis_sleeve_region_and_depth():
    img, kp, depth, d = _scene((150, 95, 60))  # blue sleeve
    skin = fc.forearm_axis(img, kp, "skin")
    assert not skin.ok or skin.conf < 0.3        # skin colour cannot see a sleeve
    region = fc.forearm_axis(img, kp, "region")
    assert region.ok and region.conf > 0.3 and _angle(region, d) < 5.0
    both = fc.forearm_axis(img, kp, "region", depth)
    assert both.ok and _angle(both, d) < 5.0
    dep = fc.forearm_axis(img, kp, "depth", depth)
    assert dep.ok and _angle(dep, d) < 5.0
    assert not fc.forearm_axis(img, kp, "depth", None).ok
    with pytest.raises(ValueError):
        fc.forearm_axis(img, kp, "nope")


def test_cue_plane_contains_forearm():
    camera = geo.Camera.fov_default(360, 640, 65.0)
    T_wc = geo.look_at([0.0, 0.0, 1.5], [0.6, 0.1, 0.8])
    W = np.array([0.55, 0.05, 0.85])
    E = np.array([0.35, -0.2, 0.95])
    Tcw = geo.inv_T(T_wc)
    p0, p1 = camera.project(geo.transform(Tcw, np.stack([W, W + 0.4 * (E - W)])))
    plane, side = fc.cue_plane(camera, T_wc, p0, p1, W)
    assert abs(plane[:3] @ E + plane[3]) < 1e-6 and abs(plane[:3] @ W + plane[3]) < 1e-6
    assert side @ (E - W) > 0


# ---------------------------------------------------------------- shoulders / prior


def test_smooth_yaw_zero_phase_and_wrap():
    t = np.arange(0, 10, 1 / 24)
    yaw = 0.3 * np.sin(2 * np.pi * 0.2 * t)
    valid = np.ones(len(t), bool)
    out = arms.smooth_yaw(t, yaw, valid, 0.5)
    lags = np.arange(-10, 11)
    corr = [np.dot(np.roll(out, k)[20:-20], yaw[20:-20]) for k in lags]
    assert lags[int(np.argmax(corr))] == 0                    # no phase lag
    assert np.std(out) < np.std(yaw)
    assert np.allclose(arms.smooth_yaw(t, yaw, valid, 0.0), yaw)
    wrapped = np.angle(np.exp(1j * (np.pi - 0.05 + 0.1 * t / 10)))  # crosses +-pi
    sm = arms.smooth_yaw(t, wrapped, valid, 1.0)
    assert np.all(np.abs(np.diff(sm)) < 0.01)
    valid[50:60] = False
    assert np.all(np.isfinite(arms.smooth_yaw(t, yaw, valid, 0.5)))


def test_prior_direction_is_the_requested_swivel():
    S = np.array([[0.0, 0.2, 1.2]])
    W = np.array([[0.4, 0.1, 0.8]])
    for side in ("left", "right"):
        prior = arms.prior_directions(S, W, 0.31, 0.27, 25.0, side)
        E = sk.elbow_from_swivel(S, W, 0.31, 0.27, math.radians(25.0), side)
        centre = 0.5 * (E + sk.elbow_from_swivel(S, W, 0.31, 0.27, math.radians(25.0) + math.pi, side))
        assert np.allclose(np.cross(prior, E - centre), 0.0, atol=1e-12) and (prior @ (E - centre).T) > 0
        assert sk.swivel_of_elbow(S, W, E, side) == pytest.approx(math.radians(25.0))


# ---------------------------------------------------------------- step


def test_arms_step_outputs(arms_rec):
    ep = Episode(arms_rec)
    d = ep.load_npz("arms")
    T = ep.gt.n_frames
    for key, shape in (("shoulder", (T, 2, 3)), ("elbow", (T, 2, 3)), ("wrist", (T, 2, 3)), ("elbow_conf", (T, 2)),
                       ("method", (T, 2)), ("torso_yaw", (T,)), ("cue_plane", (T, 2, 4))):
        assert d[key].shape == shape, key
    assert d["method"].dtype == np.int8 and set(np.unique(d["method"])) <= {0, 1, 2}
    assert np.all(np.isnan(d["elbow"][d["method"] == 0]))
    profile = ep.profile()
    ok = d["method"] > 0
    upper = np.linalg.norm(d["elbow"] - d["shoulder"], axis=-1)[ok]
    fore = np.linalg.norm(d["elbow"] - d["wrist"], axis=-1)[ok]
    assert np.allclose(upper, profile["upper_arm_length_m"], atol=1e-6)
    assert np.allclose(fore, profile["forearm_length_m"], atol=1e-6)
    q = json.loads(d["quality"])
    assert q["cue"] == "region" and "cue_stats" in q
    hw = ep.load_npz("hands_world")
    assert np.allclose(d["wrist"], hw["joints"][:, :, 0], equal_nan=True)


def test_arms_cues_and_torso_yaw(arms_rec):
    ep = Episode(arms_rec)
    gt = ep.gt
    Eg, Sg = gt["elbow_world"], gt["shoulder_world"]
    errs = {}
    for cue in ("none", "gt"):
        p = arms.params(ep.config)
        p.update(cue=cue, torso_yaw_tau_s=0.0)
        r = arms.solve(ep, p)
        errs[cue] = float(np.nanmean(np.linalg.norm(r["elbow"] - Eg, axis=-1)))
        # the ground truth uses the instantaneous head yaw
        assert np.nanmean(np.linalg.norm(r["shoulder"] - Sg, axis=-1)) < 0.01
        if cue == "gt":
            assert np.sum(r["method"] == 1) > 10
    assert errs["gt"] < 0.05 and errs["none"] < 0.08
    p = arms.params(ep.config)
    p.update(cue="none", torso_yaw_tau_s=1.0)
    r = arms.solve(ep, p)
    assert np.nanmean(np.linalg.norm(r["shoulder"] - Sg, axis=-1)) < 0.06
