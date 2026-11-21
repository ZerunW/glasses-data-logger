"""Write a synthetic recording folder with ground truth (PIPELINE.md section 8).

>>> from glassesdata_pipeline.synth import generate
>>> summary = generate("local/tmp/synth_demo", seconds=8, seed=0, size=(360, 640))   # from the repository root

The folder has the receiver's files (video.mp4, frames.csv, motion.csv, inputs.csv,
device_state.csv, clock.csv, hands.csv, info.json), ``calibration.json`` (source "synthetic"),
``profile.json``, ``pipeline.toml`` (gt backends) and ``gt/`` (see :mod:`..groundtruth`).
"""

from __future__ import annotations

import logging
import math
import shutil
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from .. import geometry as geo
from .. import skeleton as sk
from ..config import Config, load_config
from ..errors import PipelineError
from ..groundtruth import pack_mask
from . import imu as imu_mod
from .motion import Script
from .scene import SKIN_BGR, SLEEVE_BGR, Capsule, build_scene
from .writer import VideoWriter, write_json, write_recording_csvs

log = logging.getLogger(__name__)

T0_NS = 5_000_000_000_000  # phone_time_ns of video frame 0 (an arbitrary monotonic clock value)
PTS0_US = 1_000_000  # glasses video timeline of frame 0
GLASSES_CLOCK_OFFSET_NS = 1_234_567_890  # glasses_time_ns = phone_time_ns - this
MAC_CLOCK_OFFSET_NS = 4_321_000_000_000  # mac_time ~ phone_time - offset
STARTED = "2026-11-21T00:00:00+00:00"
PIPELINE_TOML = """# Written by the synthetic generator: use the ground-truth backends (gt/ plus configurable noise,
# [synth.gt_noise]) where a learned model would be needed. Remove these lines to test real models.
[hands2d]
backend = "gt"

[depth]
backend = "gt"
{depth_downsample}
[objects]
mask_backend = "gt"
"""


def depth_downsample_for(width: int, depth_width_px: float) -> int:
    """``depth.downsample`` that gives a depth grid about ``depth_width_px`` wide (0 = none: keep
    the configured default). The real glasses' 720 px frames at ``downsample = 4`` give 180 px; a
    360 px synthetic clip needs 2 for the same angular resolution of the depth, which hand3d's depth
    scale, slam's landmarks and the objects' geometry depend on."""
    if not depth_width_px or depth_width_px <= 0:
        return 0
    return max(1, int(round(width / float(depth_width_px))))


def parse_size(size: str | tuple[int, int] | list[int]) -> tuple[int, int]:
    """``"360x640"`` or ``(360, 640)`` -> ``(width, height)``."""
    if isinstance(size, str):
        try:
            w, h = (int(v) for v in size.lower().split("x"))
        except ValueError:
            raise PipelineError(f"size must be WIDTHxHEIGHT, got {size!r}") from None
    else:
        w, h = (int(v) for v in size)
    if w < 16 or h < 16:
        raise PipelineError(f"size {w}x{h} is too small")
    return w, h


def synthetic_camera(width: int, height: int, hfov_deg: float, dist: list[float]) -> geo.Camera:
    base = geo.Camera.fov_default(width, height, hfov_deg)
    return geo.Camera(base.K, dist, width, height, "pinhole", "synthetic")


def hand_capsules(joints: np.ndarray, shoulder: np.ndarray, elbow: np.ndarray) -> list[Capsule]:
    """Skin-coloured capsules along the bones of one hand plus sleeve-coloured arm segments."""
    caps: list[Capsule] = []
    for i in range(1, 21):
        radius = 0.010 if i <= 4 else (0.013 if i in (5, 9, 13, 17) else 0.0085)
        if i in (8, 12, 16, 20):
            radius = 0.0075
        caps.append(Capsule(joints[sk.PARENTS[i]], joints[i], radius, SKIN_BGR))
    caps.append(Capsule(joints[5], joints[17], 0.013, SKIN_BGR))
    caps.append(Capsule(joints[0], (joints[5] + joints[17]) / 2, 0.022, SKIN_BGR))
    caps.append(Capsule(elbow, joints[0], 0.030, SLEEVE_BGR))
    caps.append(Capsule(shoulder, elbow, 0.045, SLEEVE_BGR))
    return caps


def generate(
    out_dir: str | Path,
    seconds: float | None = None,
    seed: int | None = None,
    size: str | tuple[int, int] | None = None,
    draw_hands: bool | None = None,
    config: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render and write a synthetic recording into ``out_dir``; returns a summary dict.

    Arguments left as None come from ``config["synth"]`` (default: ``configs/synth.toml``).
    """
    started = time.perf_counter()
    cfg = Config(config) if config is not None else load_config()
    s = cfg.section("synth")
    seconds = float(s.get("seconds", 8.0) if seconds is None else seconds)
    seed = int(s.get("seed", 0) if seed is None else seed)
    width, height = parse_size(s.get("size", "360x640") if size is None else size)
    draw_hands = bool(s.get("draw_hands", True) if draw_hands is None else draw_hands)
    fps = float(s.get("fps", 24))
    n_frames = int(round(seconds * fps))
    if n_frames < 2:
        raise PipelineError("the clip needs at least two frames")

    out = Path(out_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise PipelineError(f"{out} is not empty (use overwrite=True / --overwrite)")
        shutil.rmtree(out)
    (out / "gt").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    table_height = float(s.get("table_height_m", 0.75))
    scene, layout = build_scene(rng, table_height)
    profile = sk.complete_profile(dict(s.get("profile", {})))
    script = Script(seconds, layout, profile, seed)
    camera = synthetic_camera(width, height, float(s.get("hfov_deg", 65.0)), list(s.get("dist", [-0.05, 0.01, 0, 0, 0])))
    camera.pixel_rays()  # build the ray cache once

    # ------------------------------------------------------------ ground-truth trajectories
    t = np.arange(n_frames) / fps
    T_room_cam = script.camera_pose(t)
    yaw0 = float(geo.yaw_of(T_room_cam[0, :3, :3]))
    R_wr = geo.rot_z(-yaw0)
    T_world_room = geo.make_T(R_wr, -R_wr @ T_room_cam[0, :3, 3])
    to_world = lambda T: T_world_room @ T  # noqa: E731
    T_world_cam = to_world(T_room_cam)
    sides = ("left", "right")
    joints_room = np.stack([script.joints(t, side) for side in sides], axis=1)  # [T,2,21,3]
    wrist_room = np.stack([script.wrist_pose(t, side) for side in sides], axis=1)  # [T,2,4,4]
    fingers = [script.fingers(t, side) for side in sides]
    flexion = np.stack([f[0] for f in fingers], axis=1)
    abduction = np.stack([f[1] for f in fingers], axis=1)
    shoulders_room = script.shoulders(t)
    elbows_room = script.elbows(t)
    swivel = np.stack([script.swivel(t, side) for side in sides], axis=1)
    reach = script.reach(t)
    if reach.max() > 0.95:
        raise PipelineError(f"a wrist is out of reach (ratio {reach.max():.3f}); adjust the script")
    K = len(layout["object_ids"])
    T_room_obj = np.stack([script.object_pose(t, k) for k in range(K)], axis=1)  # [T,K,4,4]
    grasped = script.grasped(t)

    joints_world = geo.transform(T_world_room, joints_room)
    T_cam_world = geo.inv_T(T_world_cam)
    joints_cam = geo.transform(T_cam_world[:, None, None], joints_world)

    # ------------------------------------------------------------ render + encode
    pts_us = PTS0_US + np.round(np.arange(n_frames) * 1e6 / fps).astype(np.int64)
    phone_ns = T0_NS + np.round(t * 1e9).astype(np.int64)
    depth_mm = np.lib.format.open_memmap(out / "gt" / "depth_f16.npy", mode="w+", dtype=np.float16,
                                         shape=(n_frames, height, width))
    n_bytes = (height * width + 7) // 8
    packed = np.zeros((n_frames, K, n_bytes), np.uint8)
    visible = np.zeros((n_frames, 2, 21), dtype=bool)
    tol = float(s.get("visibility_tolerance_m", 0.01))
    writer = VideoWriter(out / "video.mp4", width, height, fps, s.get("crf", 16), s.get("preset", "fast"), int(s.get("gop", 24)))
    verify_every = max(1, n_frames // 12)
    kept: dict[int, np.ndarray] = {}
    objects = scene.objects()
    render_s = 0.0
    try:
        for i in range(n_frames):
            tick = time.perf_counter()
            for k, box in enumerate(objects):
                box.set_pose(T_room_obj[i, k])
            capsules = None
            if draw_hands:
                capsules = []
                for slot in range(2):
                    capsules += hand_capsules(joints_room[i, slot], shoulders_room[i, slot], elbows_room[i, slot])
            frame = scene.render(camera, T_room_cam[i], capsules)
            render_s += time.perf_counter() - tick
            writer.write(frame["image"], int(pts_us[i]))
            if i % verify_every == 0 or i == n_frames - 1:
                kept[i] = frame["image"]
            depth_mm[i] = frame["depth"].astype(np.float16)
            label = frame["label"]
            for k, box in enumerate(objects):
                packed[i, k] = pack_mask(label == scene.boxes.index(box))
            visible[i] = _visibility(camera, joints_cam[i], frame["scene_depth"], tol)
    finally:
        packets = writer.close()
    depth_mm.flush()
    del depth_mm
    if len(packets) != n_frames:
        raise PipelineError(f"the encoder produced {len(packets)} packets for {n_frames} frames")

    # ------------------------------------------------------------ IMU
    icfg = s.section("imu")
    rate = float(icfg.get("rate_hz", 60.0))
    pad = float(icfg.get("padding_s", 0.3))
    t_imu = np.arange(-pad, (n_frames - 1) / fps + pad + 1e-9, 1.0 / rate)
    R_cam_imu = imu_mod.r_cam_imu(icfg.get("permutation", np.eye(3).tolist()), icfg.get("misalignment_deg", [0, 0, 0]))
    offset = float(icfg.get("time_offset_s", 0.02))
    gyro_bias = np.asarray(icfg.get("gyro_bias", [0.0, 0.0, 0.0]), dtype=np.float64)
    accel_bias = np.asarray(icfg.get("accel_bias", [0.0, 0.0, 0.0]), dtype=np.float64)
    imu = imu_mod.synthesize(
        lambda tt: T_world_room @ script.camera_pose(tt), t_imu, R_cam_imu, offset, gyro_bias, accel_bias,
        float(icfg.get("gyro_noise", 0.004)), float(icfg.get("accel_noise", 0.008)), np.random.default_rng(seed + 7),
    )

    # ------------------------------------------------------------ CSV / JSON files
    frames = {
        "pts_us": pts_us,
        "phone_ns": phone_ns,
        "mac_ns": phone_ns - MAC_CLOCK_OFFSET_NS + 3_000_000,
        "keyframe": np.array([p[0] for p in packets]),
        "bytes": np.array([p[1] for p in packets]),
    }
    imu_phone = T0_NS + np.round(t_imu * 1e9).astype(np.int64)
    motion = {
        "phone_ns": imu_phone,
        "glasses_ns": imu_phone - GLASSES_CLOCK_OFFSET_NS,
        "mac_ns": imu_phone - MAC_CLOCK_OFFSET_NS + 2_000_000,
        "acc": imu["acc"],
        "gyro": imu["gyro"],
    }
    write_recording_csvs(out, frames, motion, T0_NS, MAC_CLOCK_OFFSET_NS, seconds)
    calibration = camera.to_dict()
    calibration.update({"rms_px": 0.0, "created": STARTED, "device": "synthetic", "source": "synthetic"})
    write_json(out / "calibration.json", calibration)
    measured_keys = set(dict(s.get("profile", {})))
    profile_json = {"version": 1, **{k: profile[k] for k in sk.PROFILE_KEYS},
                    "measured": {k: k in measured_keys for k in sk.PROFILE_KEYS}}
    write_json(out / "profile.json", profile_json)
    ds = depth_downsample_for(width, float(s.get("depth_width_px", 180)))
    ds_line = (f"# depth grid ~{s.get('depth_width_px', 180)} px wide like the glasses' (synth.depth_width_px)\n"
               f"downsample = {ds}\n") if ds else ""
    (out / "pipeline.toml").write_text(PIPELINE_TOML.format(depth_downsample=ds_line))
    write_json(out / "info.json", {
        "receiver": f"glassesdata-pipeline synth {__version__}",
        "startedAt": STARTED,
        "startedMacTimeNs": int(T0_NS - MAC_CLOCK_OFFSET_NS),
        "video": {"width": width, "height": height, "configuredFps": fps},
        "clockNote": (
            "pts_us is the glasses video timeline; phone_time_ns is the iPhone monotonic clock; "
            "mac_time_ns is this Mac's monotonic clock on arrival. clock.csv maps phone to Mac time: "
            "mac_time ≈ phone_time - offset_ns (use the rows with the smallest rtt_ns)."
        ),
        "handedness": "anatomical: Left is the person's own left hand (synthetic ground truth)",
        "link": "synthetic",
        "device": "synthetic",
        "app": f"glassesdata-pipeline synth {__version__}",
        "endedAt": STARTED,
        "endedMacTimeNs": int(T0_NS - MAC_CLOCK_OFFSET_NS + phone_ns[-1] - phone_ns[0]),
        "framesWritten": n_frames,
        "firstPtsUs": int(pts_us[0]),
        "lastPtsUs": int(pts_us[-1]),
        "handRows": 0,
        "synthetic": True,
        "synth": {"seed": seed, "seconds": seconds, "size": [width, height], "draw_hands": draw_hands},
    })

    # ------------------------------------------------------------ gt/
    top_world = float((T_world_room @ np.array([0, 0, table_height, 1.0]))[2])
    floor_world = float(T_world_room[2, 3])
    tx, ty = layout["table"]["x"], layout["table"]["y"]
    polygon_room = np.array([[tx[0], ty[0], table_height], [tx[1], ty[0], table_height],
                             [tx[1], ty[1], table_height], [tx[0], ty[1], table_height]])
    polygon_world = geo.transform(T_world_room, polygon_room)[:, :2]
    np.savez(
        out / "gt" / "gt.npz",
        T_world_cam=T_world_cam,
        K=camera.K,
        joints_world=joints_world,
        joints_cam=joints_cam,
        visible=visible,
        shoulder_world=geo.transform(T_world_room, shoulders_room),
        elbow_world=geo.transform(T_world_room, elbows_room),
        T_world_obj=to_world(T_room_obj),
        R_cam_imu=R_cam_imu,
        imu_time_offset_s=np.float64(offset),
        gyro_bias=gyro_bias,
        accel_bias=accel_bias,
        table_height_m=np.float64(table_height),
        object_sizes=np.asarray(layout["object_sizes"], dtype=np.float64),
        # extras
        t=t,
        dist=camera.dist,
        image_size=np.array([width, height]),
        T_world_room=T_world_room,
        floor_z=np.float64(floor_world),
        table_top_z=np.float64(top_world),
        table_polygon=polygon_world,
        T_world_wrist=to_world(wrist_room),
        flexion=flexion,
        abduction=abduction,
        elbow_swivel=swivel,
        reach_ratio=reach,
        grasped=grasped,
        object_ids=np.array(layout["object_ids"]),
        imu_t=t_imu,
        imu_omega_cam=imu["omega_cam"],
        imu_specific_force_cam=imu["specific_force_cam"],
        gravity_world=np.array([0.0, 0.0, -1.0]),
    )
    np.savez_compressed(out / "gt" / "masks.npz", frames=np.arange(n_frames), packed=packed,
                        shape=np.array([height, width]), object_ids=np.array(layout["object_ids"]))
    verify = _verify(out, n_frames, kept) if bool(s.get("verify", True)) else {}
    elapsed = time.perf_counter() - started
    vis_right = float(np.mean(visible[:, 1, 0]))
    vis_left = float(np.mean(visible[:, 0, 0]))
    meta = {
        "generator": f"glassesdata_pipeline.synth {__version__}",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed, "seconds": seconds, "fps": fps, "frames": n_frames, "size": [width, height],
        "draw_hands": draw_hands,
        "object_ids": layout["object_ids"],
        "units": {"length": "m", "angle": "rad", "time": "s", "acc": "g (9.80665 m/s^2)", "gyro": "rad/s"},
        "frames_note": "frame i is rendered at t = i / fps; phone_time_ns = T0 + t * 1e9 exactly",
        "world": "z up, origin at the camera centre of frame 0, frame 0 optical axis along +x (PIPELINE.md section 1)",
        "room": "the scene is built in a room frame (floor z = 0); T_world_room maps it to the world",
        "hand_frame": "T_world_wrist: canonical hand frame of skeleton.py (x radial for the right hand, y distal, z palmar; left = mirror)",
        "visibility": f"in front of the camera, inside the image and not behind scene geometry by more than {tol} m (hands themselves never occlude)",
        "shoulders": "skeleton.shoulders_from_head(T_world_cam, profile)",
        "elbows": "skeleton.elbow_from_swivel(shoulder, wrist, upper_arm, forearm, elbow_swivel, side): swivel 0 = lowest, positive = outward",
        "imu": {
            "model": "see glassesdata_pipeline/synth/imu.py",
            "rate_hz": rate, "time_offset_s": offset, "R_cam_imu": R_cam_imu.tolist(),
            "gyro_bias_rad_s": gyro_bias.tolist(), "accel_bias_g": accel_bias.tolist(),
            "gyro_noise": float(icfg.get("gyro_noise", 0.004)), "accel_noise": float(icfg.get("accel_noise", 0.008)),
            "lever_arm_m": [0.0, 0.0, 0.0],
            "convention": "v_cam = R_cam_imu @ v_imu; an IMU sample stamped t measures the motion at video time t - time_offset_s",
        },
        "objects": "T_world_obj: box frame, origin at the box centroid, axes along its edges; object_sizes = full extents",
        "grasp": {"object": layout["object_ids"][0], "hand": "right", "t_grasp": script.t_grasp, "t_release": script.t_release},
        "profile": profile_json,
        "stats": {
            "right_wrist_visible": round(vis_right, 3), "left_wrist_visible": round(vis_left, 3),
            "reach_ratio_max": round(float(reach.max()), 4),
            "eye_height_m": [round(float(T_room_cam[:, 2, 3].min()), 3), round(float(T_room_cam[:, 2, 3].max()), 3)],
            "camera_path_m": round(float(np.sum(np.linalg.norm(np.diff(T_world_cam[:, :3, 3], axis=0), axis=1))), 3),
            "render_s_per_frame": round(render_s / n_frames, 4),
            "total_s": round(elapsed, 2),
        },
        "verify": verify,
    }
    write_json(out / "gt" / "meta.json", meta)
    log.info("synthetic recording %s: %d frames %dx%d in %.1f s (render %.3f s/frame)",
             out, n_frames, width, height, elapsed, render_s / n_frames)
    return {"path": str(out), "frames": n_frames, "size": [width, height], "seconds": round(elapsed, 2),
            "render_s_per_frame": round(render_s / n_frames, 4), **({"verify": verify} if verify else {})}


def _visibility(camera: geo.Camera, joints_cam: np.ndarray, scene_depth: np.ndarray, tol: float) -> np.ndarray:
    """``(2, 21)`` visibility of joints in one frame."""
    uv = camera.project(joints_cam)
    z = joints_cam[..., 2]
    inside = camera.in_image(uv) & (z > 0.02)
    vis = np.zeros(z.shape, dtype=bool)
    if np.any(inside):
        u = np.clip(np.rint(uv[..., 0][inside]).astype(int), 0, camera.width - 1)
        v = np.clip(np.rint(uv[..., 1][inside]).astype(int), 0, camera.height - 1)
        vis[inside] = z[inside] <= scene_depth[v, u] + tol
    return vis


def _verify(out: Path, n_frames: int, kept: dict[int, np.ndarray]) -> dict[str, Any]:
    """Decode video.mp4: frame count and PSNR of the kept renders."""
    import av
    import cv2

    psnr: dict[int, float] = {}
    count = 0
    with av.open(str(out / "video.mp4")) as container:
        for index, frame in enumerate(container.decode(video=0)):
            count += 1
            if index in kept:
                psnr[index] = float(cv2.PSNR(kept[index], frame.to_ndarray(format="bgr24")))
    result = {"decoded_frames": count, "psnr_db_min": round(min(psnr.values()), 2) if psnr else math.nan,
              "psnr_frames": sorted(psnr)}
    if count != n_frames:
        raise PipelineError(f"video.mp4 decodes to {count} frames, expected {n_frames}")
    if psnr and min(psnr.values()) < 30.0:
        log.warning("video PSNR %.1f dB is below 30 dB", min(psnr.values()))
    return result
