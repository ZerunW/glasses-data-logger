"""Physics check: numpy scoring (Mac) and the MuJoCo run (skipped without mujoco + Menagerie)."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/fabricate_sim.py (importlib mode)
from fabricate_sim import fabricate_sim  # noqa: E402

from glassesdata_pipeline import geometry as geo  # noqa: E402
from glassesdata_pipeline.sim.physics_check import column_map, lerp_pose, object_stats, plausibility  # noqa: E402

CFG = {"lift_threshold_m": 0.03, "lift_success_ratio": 0.5, "pos_scale_m": 0.05, "rot_scale_deg": 30.0,
       "penetration_tol_m": 0.005, "penetration_scale_m": 0.01}


def _lift_track(T=60, height=0.15):
    z = np.concatenate([np.zeros(20), np.linspace(0, height, 10), np.full(20, height), np.linspace(height, 0, 10)])
    poses = geo.make_T(geo.rot_z(np.linspace(0, 0.3, T)), np.column_stack([np.full(T, 0.5), np.zeros(T), -0.65 + z]))
    grasped = np.zeros((T, 2), bool)
    grasped[18:52, 1] = True
    return poses, grasped


def test_identical_trajectories_score_one():
    rec, grasped = _lift_track()
    valid = np.ones(len(rec), bool)
    s = object_stats(rec, rec, valid, grasped, CFG)
    assert s["pos_dev_m"]["max"] == 0 and s["rot_dev_deg"]["max"] < 1e-6
    assert s["demo_lifts"] and s["lift_success"] and s["demo_lift_m"] == pytest.approx(0.15)
    score, comp = plausibility({"a": s}, 0.0, CFG)
    assert score == pytest.approx(1.0) and comp["lift_rate"] == 1.0


def test_dropped_object_fails_the_lift_and_lowers_the_score():
    rec, grasped = _lift_track()
    sim = np.repeat(rec[:1], len(rec), axis=0)  # the object never leaves the table
    s = object_stats(sim, rec, np.ones(len(rec), bool), grasped, CFG)
    assert s["demo_lifts"] and not s["lift_success"] and s["sim_lift_m"] == pytest.approx(0.0)
    assert s["pos_dev_m"]["max"] == pytest.approx(0.15)
    score, comp = plausibility({"a": s}, 0.02, CFG)
    assert comp["lift"] == 0.5 and comp["penetration"] == pytest.approx(np.exp(-1.5)) and 0 < score < 0.3
    still = object_stats(sim, sim, np.ones(len(rec), bool), None, CFG)  # no lift in the demo
    assert not still["demo_lifts"] and plausibility({"b": still}, 0.0, CFG)[1]["lift_rate"] is None


def test_nothing_compared_is_inconclusive():
    rec, _ = _lift_track()
    none = object_stats(rec, rec, np.zeros(len(rec), bool), None, CFG)  # no valid reconstructed pose
    score, comp = plausibility({"a": none}, 0.0, CFG)
    assert score is None and comp["tracking"] is None and comp["status"].startswith("inconclusive")
    score, comp = plausibility({}, 0.0, CFG)
    assert score is None and comp["compared_objects"] == 0
    good = object_stats(rec, rec, np.ones(len(rec), bool), None, CFG)
    assert plausibility({"a": none, "b": good}, 0.0, CFG)[1]["status"] == "ok"


def test_invalid_frames_are_ignored_and_helpers():
    rec, _ = _lift_track()
    sim = rec.copy()
    sim[30:, :3, 3] += 1.0
    valid = np.zeros(len(rec), bool)
    valid[:30] = True
    s = object_stats(sim, rec, valid, None, CFG)
    assert s["frames"] == 30 and s["pos_dev_m"]["max"] == 0
    assert object_stats(sim, rec, np.zeros(len(rec), bool), None, CFG)["status"]
    p, q = lerp_pose([0, 0, 0], geo.mat_to_quat(np.eye(3)), [1, 0, 0], geo.mat_to_quat(geo.rot_z(1.0)), 0.5)
    assert np.allclose(p, [0.5, 0, 0]) and geo.rotation_angle(geo.quat_to_mat(q), geo.rot_z(0.5)) < 1e-9
    m = column_map(["a", "b", "c"], [0, 7, 8], [7, 1, 1], ["obj", "b", "a"], [0, 7, 8])
    assert m == [(0, 8, 7), (7, 7, 1)]


@pytest.mark.skipif(not os.environ.get("MUJOCO_MENAGERIE"), reason="set MUJOCO_MENAGERIE to a Menagerie clone")
def test_physics_step_runs(recording_copy):
    pytest.importorskip("mujoco")
    pytest.importorskip("mink")
    from glassesdata_pipeline.episode import Episode
    from glassesdata_pipeline.runner import Runner

    fabricate_sim(recording_copy)
    ep = Episode(recording_copy, {"retarget": {"menagerie_path": os.environ["MUJOCO_MENAGERIE"]},
                                  "scene": {"replay": "none"}})
    report = Runner(ep).run(steps=["scene", "retarget", "physics"], deps=False)
    assert report.ok, report.table()
    result = ep.load_json("physics.json")
    assert 0.0 <= result["plausibility"] <= 1.0 and set(result["objects"]) == set(ep.gt.object_ids)
