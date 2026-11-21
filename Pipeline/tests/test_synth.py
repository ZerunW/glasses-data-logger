import csv
import json
import tomllib

import av
import cv2
import numpy as np
import pytest

from glassesdata_pipeline import geometry as geo
from glassesdata_pipeline import skeleton as sk
from glassesdata_pipeline.groundtruth import GroundTruth, unpack_mask
from glassesdata_pipeline.recording import FRAMES_HEADER, MOTION_HEADER, read_frames, read_motion
from glassesdata_pipeline.synth.imu import G, integrate_gyro, synthesize
from glassesdata_pipeline.synth.scene import HAND_LABEL, Box, Capsule, Scene, Texture, build_scene
from glassesdata_pipeline.synth.generate import depth_downsample_for, hand_capsules, synthetic_camera
from glassesdata_pipeline.synth.motion import Script, box_signed_distance


@pytest.fixture(scope="module")
def gt(synth_recording):
    return GroundTruth(synth_recording / "gt")


@pytest.fixture(scope="module")
def camera(synth_recording):
    return geo.Camera.from_dict(json.loads((synth_recording / "calibration.json").read_text()))


def test_recording_files(synth_recording, gt):
    rec = synth_recording
    for name in ("video.mp4", "frames.csv", "motion.csv", "inputs.csv", "device_state.csv", "clock.csv",
                 "hands.csv", "info.json", "calibration.json", "profile.json", "pipeline.toml"):
        assert (rec / name).is_file(), name
    with open(rec / "frames.csv") as f:
        assert next(csv.reader(f)) == FRAMES_HEADER
    with open(rec / "motion.csv") as f:
        assert next(csv.reader(f)) == MOTION_HEADER
    info = json.loads((rec / "info.json").read_text())
    assert info["synthetic"] is True and info["framesWritten"] == 48 and info["video"]["width"] == 180
    calib = json.loads((rec / "calibration.json").read_text())
    assert calib["source"] == "synthetic" and calib["model"] == "pinhole" and len(calib["dist"]) == 5
    toml = tomllib.loads((rec / "pipeline.toml").read_text())
    assert toml["hands2d"]["backend"] == "gt" and toml["depth"]["backend"] == "gt"
    assert "downsample" not in toml["depth"]  # the test fixture sets synth.depth_width_px = 0
    assert toml["objects"]["mask_backend"] == "gt"
    motion = read_motion(rec / "motion.csv")
    assert set(motion["source"]) == {"glasses"} and np.all(np.isnan(motion["quat"])) and np.all(np.isnan(motion["mag"]))
    assert gt.n_frames == 48 and gt["joints_world"].shape == (48, 2, 21, 3) and gt["T_world_obj"].shape == (48, 2, 4, 4)
    for key in ("T_world_cam", "K", "joints_world", "joints_cam", "visible", "shoulder_world", "elbow_world",
                "T_world_obj", "R_cam_imu", "imu_time_offset_s", "gyro_bias", "accel_bias", "table_height_m",
                "object_sizes"):
        assert key in gt, key
    assert gt.depth().shape == (48, 320, 180) and gt.depth().dtype == np.float16


def test_video_frames_match_renders(synth_recording, gt, camera):
    frames = read_frames(synth_recording / "frames.csv")
    with av.open(str(synth_recording / "video.mp4")) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "hevc" and stream.codec_tag == "hvc1"
        packets = [p for p in container.demux(stream) if p.size]
        assert all(p.pts == p.dts for p in packets)  # no B-frames
        assert [i for i, p in enumerate(packets) if p.is_keyframe] == [0, 24]
    with av.open(str(synth_recording / "video.mp4")) as container:
        decoded = [f.to_ndarray(format="bgr24") for f in container.decode(video=0)]
    assert len(decoded) == len(frames["phone_ns"]) == gt.n_frames
    assert list(frames["keyframe"]) == [i % 24 == 0 for i in range(48)]
    # Re-render two frames from the ground truth and compare with the decoded video (hands and
    # arms are drawn by default: synth.draw_hands = true).
    assert json.loads((synth_recording / "gt" / "meta.json").read_text())["draw_hands"] is True
    for i in (7, 40):
        image = _rerender(gt, camera, i)["image"]
        assert cv2.PSNR(image, decoded[i]) > 30
        without = _rerender(gt, camera, i, hands=False)["image"]
        assert cv2.PSNR(without, decoded[i]) < cv2.PSNR(image, decoded[i]) - 3  # the hands are in the video


def _rerender(gt, camera, i, hands=True):
    """Render frame ``i`` again from the ground truth (seed 0 scene), with or without the hand and
    arm capsules."""
    scene, _ = build_scene(np.random.default_rng(0), float(gt["table_height_m"]))
    T_room_world = geo.inv_T(gt["T_world_room"])
    to_room = lambda p: geo.transform(T_room_world, p)  # noqa: E731
    for k, box in enumerate(scene.objects()):
        box.set_pose(T_room_world @ gt["T_world_obj"][i, k])
    capsules = []
    if hands:
        for h in range(2):
            capsules += hand_capsules(to_room(gt["joints_world"][i, h]), to_room(gt["shoulder_world"][i, h]),
                                      to_room(gt["elbow_world"][i, h]))
    return scene.render(camera, T_room_world @ gt["T_world_cam"][i], capsules or None)


def _inside_convex(xy, poly, margin):
    """Points at least ``margin`` inside a convex polygon (either orientation)."""
    e0, e1 = poly[1] - poly[0], poly[2] - poly[1]
    sign = np.sign(e0[0] * e1[1] - e0[1] * e1[0])
    inside = np.ones(xy.shape[:-1], dtype=bool)
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        edge = (b - a) / np.linalg.norm(b - a)
        rel = xy - a
        inside &= sign * (edge[0] * rel[..., 1] - edge[1] * rel[..., 0]) > margin
    return inside


def test_depth_matches_analytic_planes(gt, camera):
    """Table-top pixels (not covered by an object) have exactly the ray/plane depth; nothing is
    below the floor."""
    rays = camera.pixel_rays()
    depth = gt.depth()
    poly = gt["table_polygon"]
    masks = gt.masks()
    checked = 0
    for i in (0, 20, 47):
        T = gt["T_world_cam"][i]
        d_world = rays @ T[:3, :3].T
        o = T[:3, 3]
        for z_plane, kind in ((float(gt["table_top_z"]), "table"), (float(gt["floor_z"]), "floor")):
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (z_plane - o[2]) / d_world[..., 2]
            p = o + s[..., None] * d_world
            ok = s > 0
            if kind == "table":
                inside = _inside_convex(p[..., :2], poly, margin=0.01)
                free = ~np.any([unpack_mask(masks["packed"][i, k], masks["shape"]) for k in range(2)], axis=0)
                free &= _rerender(gt, camera, i)["label"] != HAND_LABEL  # hands/arms are drawn
                free = cv2.erode(free.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                sel = ok & inside & free
                assert sel.sum() > 1000
                err = np.abs(depth[i][sel].astype(np.float64) - s[sel]) / s[sel]
                assert err.max() < 2e-3  # float16 storage
                checked += int(sel.sum())
            else:
                assert np.all(depth[i][ok].astype(np.float64) <= s[ok] * 1.002 + 1e-3)
                on_floor = np.abs(depth[i][ok].astype(np.float64) - s[ok]) < 2e-3 * s[ok]
                assert on_floor.mean() > 0.05
    assert checked > 3000


def test_joint_projections_and_visibility(gt, camera):
    uv = camera.project(gt["joints_cam"])
    vis = gt["visible"]
    assert np.all(camera.in_image(uv[vis])) and np.all(gt["joints_cam"][..., 2][vis] > 0)
    assert np.allclose(camera.K, gt["K"])
    assert vis[:, 1, 0].mean() > 0.8  # the right wrist is in view most of the time
    T = gt["T_world_cam"]
    J = geo.transform(geo.inv_T(T)[:, None, None], gt["joints_world"])
    assert np.allclose(J, gt["joints_cam"], atol=1e-9)
    # The scene depth (without the drawn hands) at visible joints is not much closer than the joint
    # itself; the recorded depth there is the hand's own surface: in front of the joint centre, by
    # less than the thickest capsule.
    depth = gt.depth()
    for i in (0, 24, 47):
        h_idx, j_idx = np.nonzero(vis[i])
        u = np.clip(np.rint(uv[i, h_idx, j_idx, 0]).astype(int), 0, camera.width - 1)
        v = np.clip(np.rint(uv[i, h_idx, j_idx, 1]).astype(int), 0, camera.height - 1)
        z = gt["joints_cam"][i, h_idx, j_idx, 2]
        scene_depth = _rerender(gt, camera, i, hands=False)["scene_depth"].astype(np.float64)
        assert len(z) and np.all(z <= scene_depth[v, u] * 1.002 + 0.0105)
        surface = depth[i, v, u].astype(np.float64)
        assert np.all(surface <= z * 1.002 + 1e-3) and np.median(z - surface) < 0.03


def test_world_frame_convention(gt):
    T0 = gt["T_world_cam"][0]
    assert np.allclose(T0[:3, 3], 0, atol=1e-12)
    assert abs(geo.yaw_of(T0[:3, :3])) < 1e-9
    assert np.isclose(gt["table_top_z"] - gt["floor_z"], gt["table_height_m"])
    pitch = np.degrees(np.arcsin(gt["T_world_cam"][:, 2, 2]))  # optical axis elevation
    assert -60 < pitch.min() and pitch.max() < -35
    eye_height = gt["T_world_cam"][:, 2, 3] - gt["floor_z"]
    assert 1.35 < eye_height.min() and eye_height.max() < 1.6


def test_hand_skeleton_and_arms(synth_recording, gt):
    profile = json.loads((synth_recording / "profile.json").read_text())
    bones = sk.bone_lengths(profile)
    J = gt["joints_world"]
    lengths = np.linalg.norm(J[:, :, 1:] - J[:, :, sk.PARENTS[1:]], axis=-1)
    assert np.allclose(lengths, bones, atol=1e-9)
    assert np.allclose(sk.wrist_frame(J[:, 1], "right"), gt["T_world_wrist"][:, 1], atol=1e-9)
    S, E, W = gt["shoulder_world"], gt["elbow_world"], J[:, :, 0]
    assert np.allclose(np.linalg.norm(E - S, axis=-1), profile["upper_arm_length_m"])
    assert np.allclose(np.linalg.norm(E - W, axis=-1), profile["forearm_length_m"])
    assert np.allclose(S, sk.shoulders_from_head(gt["T_world_cam"], profile))
    for slot, side in enumerate(("left", "right")):
        assert np.allclose(sk.swivel_of_elbow(S[:, slot], W[:, slot], E[:, slot], side), gt["elbow_swivel"][:, slot])
    assert gt["reach_ratio"].max() < 0.95
    assert np.all(J[..., 2] > gt["table_top_z"])  # no joint inside the table


def test_grasped_box_follows_the_hand(gt):
    held = gt["grasped"][:, 0, 1]
    assert held.any() and not held[0] and not held[-1]
    T_wo = gt["T_world_obj"][:, 0]
    rel = geo.inv_T(gt["T_world_wrist"][:, 1]) @ T_wo
    assert np.allclose(rel[held], rel[held][0], atol=1e-9)  # rigid while grasped
    moved = np.linalg.norm(T_wo[-1, :3, 3] - T_wo[0, :3, 3])
    assert 0.25 < moved < 0.35
    lift = T_wo[:, 2, 3].max() - T_wo[0, 2, 3]
    assert 0.13 < lift < 0.17
    half_h = gt["object_sizes"][0, 2] / 2
    assert np.isclose(T_wo[-1, 2, 3], gt["table_top_z"] + half_h)  # placed back on the table
    assert np.allclose(T_wo[-1, :3, 2], [0, 0, 1], atol=1e-9)  # upright
    assert np.allclose(gt["T_world_obj"][:, 1], gt["T_world_obj"][0, 1])  # box_b static


def test_hands_do_not_penetrate_objects(gt):
    J = gt["joints_world"]
    for k in range(gt["T_world_obj"].shape[1]):
        local = geo.transform(geo.inv_T(gt["T_world_obj"][:, k])[:, None, None], J)
        sd = box_signed_distance(local, gt["object_sizes"][k])
        assert sd.min() > 0.005, (k, sd.min())  # joints are bone centre lines: keep a finger radius away


def test_imu_gyro_integrates_to_gt_rotation(synth_recording, gt):
    motion = read_motion(synth_recording / "motion.csv")
    t0 = read_frames(synth_recording / "frames.csv")["phone_ns"][0]
    t_imu = (motion["phone_ns"] - t0) * 1e-9
    R_ci, bias, offset = gt["R_cam_imu"], gt["gyro_bias"], gt["imu_time_offset_s"]
    omega_cam = (motion["gyro"] - bias) @ R_ci.T
    R = gt["T_world_cam"][:, :3, :3]
    t = gt["t"]
    for a, b in ((0, 24), (10, 40), (24, 47)):
        est = integrate_gyro(t_imu - offset, omega_cam, t[a], t[b])
        true = R[a].T @ R[b]
        assert np.degrees(geo.rotation_angle(est, true)) < 0.5
    # without the time offset the error grows: the offset is really there
    bad = integrate_gyro(t_imu, omega_cam, t[10], t[40])
    good = integrate_gyro(t_imu - offset, omega_cam, t[10], t[40])
    true = R[10].T @ R[40]
    assert geo.rotation_angle(bad, true) > geo.rotation_angle(good, true)


def test_imu_accelerometer_is_specific_force_in_g(synth_recording, gt):
    motion = read_motion(synth_recording / "motion.csv")
    t0 = read_frames(synth_recording / "frames.csv")["phone_ns"][0]
    t_imu = (motion["phone_ns"] - t0) * 1e-9
    assert np.allclose(t_imu, gt["imu_t"], atol=1e-9)
    R_ci, bias = gt["R_cam_imu"], gt["accel_bias"]
    assert 0.97 < np.median(np.linalg.norm(motion["acc"], axis=1)) < 1.03  # units of g
    acc_cam = (motion["acc"] - bias) @ R_ci.T * G  # m/s^2 in the camera frame
    residual = acc_cam - gt["imu_specific_force_cam"]
    assert abs(residual.mean()) < 0.03 and 0.05 < residual.std() < 0.11  # 0.008 g white noise
    # the stored specific force is the physics of the trajectory: compare with second differences
    # of the frame-rate poses (a coarse check; the 2 s clip moves fast)
    T, t = gt["T_world_cam"], gt["t"]
    dt = t[1] - t[0]
    p = T[:, :3, 3]
    a_world = (p[2:] - 2 * p[1:-1] + p[:-2]) / dt**2
    expected = np.einsum("nji,nj->ni", T[1:-1, :3, :3], a_world - [0, 0, -G])
    offset = gt["imu_time_offset_s"]
    stored = np.stack([np.interp(t[1:-1] + offset, t_imu, gt["imu_specific_force_cam"][:, k]) for k in range(3)], -1)
    err = np.linalg.norm(stored - expected, axis=1)
    assert np.median(err) < 1.0  # vs |f| ~ 9.8 m/s^2; the exact physics is tested analytically below
    wrong = np.einsum("nij,nj->ni", T[1:-1, :3, :3], a_world - [0, 0, -G])  # R instead of R^T
    assert np.median(err) < 0.2 * np.median(np.linalg.norm(stored - wrong, axis=1))
    up_cam = np.einsum("nji,j->ni", T[1:-1, :3, :3], [0, 0, 1])
    cos = np.sum(stored * up_cam, axis=1) / np.linalg.norm(stored, axis=1)
    assert np.median(np.degrees(np.arccos(np.clip(cos, -1, 1)))) < 15  # mostly gravity, up (the 2 s clip moves fast)


def test_imu_synthesis_on_an_analytic_trajectory():
    w, A, W = 0.7, 0.05, 2 * np.pi * 0.8

    def pose(t):
        return geo.make_T(geo.rot_z(w * t) @ geo.rot_x(0.3), np.stack([A * np.sin(W * t), 0 * t, 0 * t], -1))

    R_ci = geo.so3_exp([0.1, -0.2, 0.3]) @ np.array([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]])
    t = np.linspace(0, 2, 121)
    out = synthesize(pose, t, R_ci, 0.02, np.array([0.01, 0, 0]), np.array([0, 0.02, 0]), 0.0, 0.0,
                     np.random.default_rng(0))
    tp = t - 0.02
    R = pose(tp)[:, :3, :3]
    omega_cam = np.einsum("nji,j->ni", R, [0, 0, w])  # world-z rotation seen in the camera frame
    assert np.allclose(out["gyro"], omega_cam @ R_ci + [0.01, 0, 0], atol=1e-6)
    f_world = np.stack([-A * W**2 * np.sin(W * tp), 0 * tp, G + 0 * tp], -1)
    f_cam = np.einsum("nji,nj->ni", R, f_world)
    assert np.allclose(out["acc"], f_cam @ R_ci / G + [0, 0.02, 0], atol=1e-5)


def test_masks_cover_the_objects(gt, camera):
    masks = gt.masks()
    assert masks["packed"].shape[:2] == (48, 2) and tuple(masks["shape"]) == (320, 180)
    assert list(masks["object_ids"]) == ["box_a", "box_b"]
    for i in (0, 30):
        for k in range(2):
            m = gt.mask(i, k)
            assert m.sum() > 20
            centre = camera.project(geo.transform(geo.inv_T(gt["T_world_cam"][i]), gt["T_world_obj"][i, k, :3, 3]))
            u, v = np.rint(centre).astype(int)
            assert m[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4].any()


def test_reach_holds_for_other_seeds_and_lengths():
    _, layout = build_scene(np.random.default_rng(0))
    for seed in (1, 2, 3):
        for seconds in (1.0, 2.0, 8.0, 20.0):
            script = Script(seconds, layout, {"upper_arm_length_m": 0.30, "forearm_length_m": 0.25}, seed)
            assert script.reach(np.linspace(0, seconds, 200)).max() < 0.95


def test_capsule_rendering_occludes():
    cam = synthetic_camera(64, 96, 65.0, [0, 0, 0, 0, 0])
    wall = Box("wall", np.array([1.0, 1.0, 0.01]), np.array([0.0, 0.0, 2.0]))
    wall.textures = [Texture(np.full((8, 8, 3), 100, np.uint8), 2.0, 2.0)] * 6
    scene = Scene([wall])
    capsule = Capsule(np.array([-0.05, 0.0, 1.0]), np.array([0.05, 0.0, 1.0]), 0.03, np.array([0, 0, 255.0], np.float32))
    out = scene.render(cam, np.eye(4), [capsule])
    cy, cx = 47, 31  # pixel centre near the optical axis: (31.5, 47.5)
    assert np.isclose(out["scene_depth"][cy, cx], 2.0 - 0.01, atol=1e-6)
    ray = cam.pixel_rays()[cy, cx]
    # depth of the cylinder surface along this ray: solve |(ray * z) - axis point| = r in the y-z plane
    y = ray[1]
    z_expected = (1.0 - np.sqrt((1 + y**2) * 0.03**2 - y**2)) / (1 + y**2)
    assert np.isclose(out["depth"][cy, cx], z_expected, atol=1e-5)
    assert out["label"][cy, cx] == -2 and out["image"][cy, cx, 2] > out["image"][cy, cx, 0]
    assert out["depth"][0, 0] == out["scene_depth"][0, 0]


def test_depth_grid_matches_the_glasses():
    """pipeline.toml picks depth.downsample so the depth grid is ~synth.depth_width_px wide (the
    glasses' 720 px at downsample 4 = 180 px)."""
    assert [depth_downsample_for(w, 180) for w in (90, 180, 270, 360, 720)] == [1, 1, 2, 2, 4]
    assert depth_downsample_for(360, 0) == 0
