import json

import cv2
import numpy as np
import pytest

from glassesdata_pipeline import geometry as geo
from glassesdata_pipeline import skeleton as sk
from glassesdata_pipeline.episode import Episode
from glassesdata_pipeline.eval.report import evaluate, run_eval

G = 9.80665


def write_perfect_outputs(ep: Episode) -> None:
    """Derived files equal to the ground truth (section-5 formats)."""
    gt = ep.gt
    T = gt.n_frames
    meta = {"step": "test", "version": 1}
    cam = ep.camera()
    uv = cam.project(gt["joints_cam"])
    vis = gt["visible"]
    kp = np.concatenate([uv, np.full(uv.shape[:-1] + (1,), 0.9)], -1)
    kp[~vis] = np.nan
    ep.save_npz("hands2d", meta, kp=kp, hand_score=np.where(vis.any(-1), 0.9, np.nan), mp_world=np.full(kp.shape, np.nan))
    full = gt.depth()
    w, h = geo.downsample_size(full.shape[2], full.shape[1], 4)
    depth = np.lib.format.open_memmap(ep.file("depth/depth_f16.npy"), mode="w+", dtype=np.float16, shape=(T, h, w))
    for i in range(T):
        depth[i] = cv2.resize(np.asarray(full[i], np.float32), (w, h), interpolation=cv2.INTER_AREA)
    depth.flush()
    ep.save_npz("depth/index", meta, frames=np.arange(T), scale=h / full.shape[1])
    ep.save_npz("hand3d", meta, joints_cam=gt["joints_cam"], valid=np.ones((T, 2), bool), reproj_px=np.zeros((T, 2)),
                depth_scale=np.ones(T), depth_scale_smooth=np.ones(T))
    T_wc = gt["T_world_cam"]
    ep.save_npz("slam", meta, T_slam_cam=geo.inv_T(T_wc[0]) @ T_wc, valid=np.ones(T, bool), inliers=np.full(T, 100),
                is_keyframe=np.zeros(T, bool), source="test")
    ep.save_npz("camera_traj", meta, T_world_cam=T_wc, valid=np.ones(T, bool))
    J = gt["joints_world"]
    bones = np.linalg.norm(J[0][:, 1:] - J[0][:, sk.PARENTS[1:]], axis=-1)
    ep.save_npz("hands_world", meta, joints=J, raw=J, observed=np.ones((T, 2), bool), filled=np.zeros((T, 2), bool),
                bone_lengths=bones)
    ep.save_npz("arms", meta, shoulder=gt["shoulder_world"], elbow=gt["elbow_world"], wrist=J[:, :, 0],
                elbow_conf=np.ones((T, 2)), method=np.ones((T, 2), np.int8))
    ids = gt.object_ids
    ep.save_json("objects/index.json", [{"id": f"obj{k}", "prompt": {}, "first_frame": 0, "last_frame": T - 1}
                                        for k in range(len(ids))])
    for k in range(len(ids)):
        # an arbitrary (but fixed) object-frame rotation: only rotation changes are compared
        R_fix = geo.so3_exp([0.3, -0.2, 1.0])
        T_obj = gt["T_world_obj"][:, k] @ geo.make_T(R_fix)
        ep.save_npz(f"objects/obj{k}/poses", meta, T_world_obj=T_obj, valid=np.ones(T, bool), fitness=np.ones(T),
                    grasped=gt["grasped"][:, k])
    ep.save_json("objects/table.json", {"normal": [0, 0, 1], "offset": 0.0, "height_m": float(gt["table_height_m"]),
                                        "polygon": gt["table_polygon"].tolist()}, meta)
    gravity = np.einsum("tji,j->ti", T_wc[:, :3, :3], [0.0, 0.0, -1.0])
    ep.save_npz("imu", meta, R_cam_imu=gt["R_cam_imu"], time_offset_s=gt["imu_time_offset_s"], gyro_bias=gt["gyro_bias"],
                accel_scale=G, gravity_cam=gravity, R_world_slam=np.eye(3), quality="{}")


def test_eval_with_perfect_outputs_is_zero(recording_copy):
    ep = Episode(recording_copy)
    write_perfect_outputs(ep)
    result = run_eval(ep)
    m = result["metrics"]
    assert all(v.get("status", "ok") in ("ok", "pending") for v in m.values()), m
    assert m["hands2d"]["px_mean"] < 1e-9 and m["hands2d"]["detection_rate"] == 1.0
    assert m["depth"]["pipeline_scaled"]["abs_rel"] < 1e-3 and m["depth"]["pipeline_scaled"]["delta1"] == 1.0
    assert m["arms"]["elbow"]["all"]["max"] < 1e-12 and m["arms"]["shoulder"]["all"]["max"] < 1e-12
    for obj in m["objects"]["objects"].values():
        assert obj["translation_m"]["max"] < 1e-12 and obj["rotation_change_deg"]["max"] < 1e-4
    assert {o["matched_gt"] for o in m["objects"]["objects"].values()} == {"box_a", "box_b"}
    assert m["objects"]["table_height_error_m"] == 0
    imu = m["imu"]
    assert imu["time_offset_error_s"] == 0 and imu["R_cam_imu_error_deg"] < 1e-6 and imu["gyro_bias_error_rad_s"] == 0
    assert imu["gravity_error_deg"]["max"] < 1e-4
    for name in ("eval.json", "eval.md", "eval/trajectory_topview.png", "eval/errors_per_frame.png"):
        assert (ep.derived / name).is_file(), name
    saved = json.loads((ep.derived / "eval.json").read_text())
    assert saved["meta"]["step"] == "eval" and "derived/hand3d.npz" in saved["meta"]["inputs"]
    assert "## depth" in (ep.derived / "eval.md").read_text()


def test_eval_trajectory_and_joint_metrics(recording_copy):
    pytest.importorskip("glassesdata_pipeline.core.align")
    ep = Episode(recording_copy)
    write_perfect_outputs(ep)
    m = evaluate(ep)
    for group in ("camera", "slam"):
        assert m[group]["status"] == "ok"
        assert m[group]["ate_sim3"]["rmse"] < 1e-9 and m[group]["rpe_1"]["rot_mean_deg"] < 1e-4
    assert m["camera"]["ate_none"]["rmse"] < 1e-12
    assert m["hand3d"]["mpjpe"] < 1e-12 and m["hand3d"]["pa_mpjpe"] < 1e-9
    assert m["hands_world"]["mpjpe"] < 1e-12 and m["hands_world"]["bone_length_error_mm"] < 1e-9
    # a scaled, shifted SLAM trajectory is perfect after Sim(3) alignment but not SE(3)
    T = ep.gt["T_world_cam"].copy()
    T[:, :3, 3] = 0.5 * T[:, :3, 3] + [1, 2, 3]
    ep.save_npz("slam", {"step": "test"}, T_slam_cam=T, valid=np.ones(len(T), bool))
    m = evaluate(ep)
    assert m["slam"]["ate_sim3"]["rmse"] < 1e-9 and np.isclose(m["slam"]["ate_sim3"]["scale"], 2.0)
    assert m["slam"]["ate_se3"]["rmse"] > 1e-3


def test_eval_with_noisy_gt_backends(recording_copy):
    """hands2d and depth from the gt backends: errors match the configured noise."""
    from glassesdata_pipeline.backends import gt as gtb

    ep = Episode(recording_copy)
    h = gtb.hands2d(ep, ep.config)
    ep.save_npz("hands2d", {"step": "hands2d"}, **h)
    d = gtb.depth(ep, ep.config)
    mm = np.lib.format.open_memmap(ep.file("depth/depth_f16.npy"), mode="w+", dtype=np.float16, shape=d["depth"].shape)
    mm[:] = d["depth"]
    mm.flush()
    ep.save_npz("depth/index", {"step": "depth"}, frames=d["frames"], scale=d["scale"])
    m = evaluate(ep)
    assert 1.8 < m["hands2d"]["px_mean"] < 3.2  # mean of a 2D Gaussian norm with sigma 2: 2.5
    assert 0.06 < m["depth"]["raw"]["abs_rel"] < 0.16  # 10 % scale error, drift and 3 % noise


def test_eval_skips_missing_outputs(recording_copy):
    ep = Episode(recording_copy)
    m = evaluate(ep)
    assert {v["status"] for k, v in m.items() if not k.startswith("_")} == {"skipped"}


def test_eval_marks_align_metrics_pending_without_core_align(recording_copy, monkeypatch):
    from glassesdata_pipeline.eval import metrics

    monkeypatch.setattr(metrics, "align_module", lambda: None)
    ep = Episode(recording_copy)
    write_perfect_outputs(ep)
    m = evaluate(ep)
    for group in ("camera", "slam", "hand3d", "hands_world"):
        assert m[group]["status"] == "pending", group
    assert m["arms"]["status"] == "ok" and m["depth"]["status"] == "ok"
