"""Rerun recording of an episode: ``derived/viz/episode.rrd`` (needs ``rerun-sdk``; written against
**rerun-sdk 0.43.2**, the version pinned in ``requirements.txt``).

APIs used (checked in the 0.43.2 sources): ``rr.RecordingStream(application_id, recording_id=)``,
``.save(path)``, ``.log(path, archetype, static=)``, ``.set_time(timeline, duration=|sequence=)``,
``rr.send_columns(path, indexes=[rr.TimeColumn(name, duration=)], columns=..., recording=)``,
``rr.AssetVideo(path=)`` + ``.read_frame_timestamps_nanos()`` + ``rr.VideoFrameReference.columns_nanos``,
``rr.Pinhole(image_from_camera=, resolution=, camera_xyz=rr.ViewCoordinates.RDF)``,
``rr.Transform3D(translation=, mat3x3=)``, ``rr.Mesh3D(vertex_positions=, triangle_indices=,
albedo_factor=)``, ``rr.LineStrips3D/2D(strips, colors=, radii=)``, ``rr.Points2D(positions, colors=, radii=)``,
``rr.Scalars.columns(scalars=)``, ``rr.EncodedImage(contents=, media_type=)``,
``rr.ViewCoordinates.RIGHT_HAND_Z_UP``.

Entity tree::

    world                         (z up)
    world/camera                  Transform3D per frame (T_world_cam)
    world/camera/image            Pinhole (OpenCV axes, RDF) + video / JPEG frames + 2D keypoints
    world/camera_path             static line
    world/hands/<side>            LineStrips3D per frame (5 finger chains + knuckle line)
    world/arms/<side>             LineStrips3D shoulder-elbow-wrist
    world/objects/<id>            Transform3D per frame; .../mesh static Mesh3D (object frame)
    world/table                   static Mesh3D + outline
    world/gt/...                  ground truth (synthetic recordings)
    imu/acc/{x,y,z}, imu/gyro/... scalars; plots/... wrist and object heights, retarget error

The video: ``rerun_video = "asset"`` logs ``video.mp4`` once as an ``AssetVideo`` and references
its frames by timestamp (cheap). Rerun's native viewer decodes H.265 through a separately
installed FFmpeg >= 5.1; use ``rerun_video = "jpeg"`` (downsampled JPEG frames) when that is not
available, or ``"none"``. The pinhole and the 2D keypoints are in the pixels of the logged images
(the JPEG size for ``"jpeg"``, :func:`jpeg_size`, :func:`to_image_px`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..episode import Episode
from ..errors import require
from ..sim import mjcf
from ..sim.data import load_episode_data
from ..sim.replay import render_meshes

log = logging.getLogger(__name__)

RERUN_PIN = "rerun-sdk==0.43.2"
CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20], [5, 9, 13, 17]]


def _rgb(code: str, alpha: int = 255) -> list[int]:
    code = code.lstrip("#")
    return [int(code[0:2], 16), int(code[2:4], 16), int(code[4:6], 16), alpha]


def _strips(J: np.ndarray) -> list[np.ndarray]:
    return [J[c] for c in CHAINS if np.all(np.isfinite(J[c]))]


def jpeg_size(width: int, height: int, max_height: int) -> tuple[int, int]:
    """Size of a JPEG frame of the ``rerun_video = "jpeg"`` fallback: downscaled to ``max_height``
    when taller (aspect kept)."""
    if height <= max_height:
        return int(width), int(height)
    return int(round(width * max_height / height)), int(max_height)


def to_image_px(uv: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> np.ndarray:
    """Pixels of a ``src_wh`` image -> pixels of the same image resized to ``dst_wh`` (pixel
    centres at integers, the convention of :meth:`geometry.Camera.resized`)."""
    uv = np.asarray(uv, dtype=np.float64)
    if tuple(src_wh) == tuple(dst_wh):
        return uv
    s = np.array([dst_wh[0] / src_wh[0], dst_wh[1] / src_wh[1]])
    return (uv + 0.5) * s - 0.5


def export_rrd(ep: Episode, out_path: str | Path, vcfg: Any) -> dict[str, Any]:
    """Write the ``.rrd``; returns ``{"path", "frames", "video"}``."""
    rr = require("rerun", RERUN_PIN, "the Rerun export (step viz)")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_episode_data(ep, with_gt=bool(vcfg.get("rerun_ground_truth", True)))
    rec = rr.RecordingStream(application_id="glassesdata", recording_id=ep.path.name)
    rec.save(str(out_path))
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    t = data.t
    cam = data.camera

    # ---------------------------------------------------------------- static content
    # (the pinhole is logged after the video: its resolution is that of the logged images)
    if data.T_world_cam is not None:
        ok = np.all(np.isfinite(data.T_world_cam), axis=(-1, -2))
        rec.log("world/camera_path", rr.LineStrips3D([data.T_world_cam[ok, :3, 3]], colors=[_rgb("#3b3a37")], radii=[0.002]),
                static=True)
    if data.table_polygon is not None and data.table_top_z is not None:
        P = np.column_stack([data.table_polygon, np.full(len(data.table_polygon), data.table_top_z)])
        tris = np.array([[0, i, i + 1] for i in range(1, len(P) - 1)], dtype=np.uint32)
        rec.log("world/table", rr.Mesh3D(vertex_positions=P, triangle_indices=tris, albedo_factor=_rgb("#ebe5dc", 200)),
                static=True)
        rec.log("world/table/outline", rr.LineStrips3D([np.vstack([P, P[:1]])], colors=[_rgb("#b9ae9f")], radii=[0.002]),
                static=True)
    for k, obj in enumerate(data.objects):
        V, F = render_meshes(obj, max_triangles=20000)
        color = _rgb(mjcf.OBJECT_HEX[k % len(mjcf.OBJECT_HEX)], 220)
        rec.log(f"world/objects/{obj.id}/mesh", rr.Mesh3D(vertex_positions=V, triangle_indices=F.astype(np.uint32),
                                                         albedo_factor=color), static=True)
    if data.gt is not None:
        rec.log("world/gt/camera_path", rr.LineStrips3D([data.gt.T_world_cam[:, :3, 3]], colors=[_rgb("#8a8985")],
                                                        radii=[0.0015]), static=True)

    # ---------------------------------------------------------------- video
    video_mode = str(vcfg.get("rerun_video", "asset"))
    video_info = "none"
    image_wh = (cam.width, cam.height) if cam is not None else None  # pixels of the logged images
    if video_mode == "asset" and ep.video_path.is_file():
        try:
            asset = rr.AssetVideo(path=str(ep.video_path))
            rec.log("world/camera/image/video", asset, static=True)
            stamps = asset.read_frame_timestamps_nanos()
            n = min(len(stamps), data.T)
            rr.send_columns("world/camera/image/video", indexes=[rr.TimeColumn("time", duration=t[:n])],
                            columns=rr.VideoFrameReference.columns_nanos(stamps[:n]), recording=rec)
            video_info = f"asset ({n} frame references)"
        except Exception as error:  # noqa: BLE001 - e.g. a container the SDK cannot parse
            log.warning("AssetVideo failed (%s); logging JPEG frames instead", error)
            video_mode = "jpeg"
    if video_mode == "jpeg" and ep.video_path.is_file():
        import cv2

        stride = max(1, int(vcfg.get("rerun_jpeg_stride", 1)))
        quality = int(vcfg.get("rerun_jpeg_quality", 80))
        height = int(vcfg.get("rerun_jpeg_height", 640))
        count = 0
        for f, image in ep.frames(range(0, data.T, stride)):
            size = jpeg_size(image.shape[1], image.shape[0], height)
            if size != (image.shape[1], image.shape[0]):
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                continue
            rec.set_time("time", duration=float(t[f]))
            rec.log("world/camera/image/video", rr.EncodedImage(contents=buf.tobytes(), media_type="image/jpeg"))
            count += 1
        video_info = f"jpeg ({count} frames, stride {stride})"
        if cam is not None:
            image_wh = jpeg_size(cam.width, cam.height, height)
    if cam is not None:
        pin = cam if image_wh == (cam.width, cam.height) else cam.resized(*image_wh)
        rec.log("world/camera/image", rr.Pinhole(image_from_camera=pin.K, resolution=list(image_wh),
                                                 camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.15), static=True)

    # ---------------------------------------------------------------- per frame
    kp2d = None
    if ep.has("hands2d.npz"):
        kp2d = np.asarray(ep.load_npz("hands2d.npz")["kp"], dtype=np.float64)
    held = [obj.held_poses() for obj in data.objects]
    sides = ("left", "right")
    for f in range(data.T):
        rec.set_time("time", duration=float(t[f]))
        rec.set_time("frame", sequence=int(f))
        if data.T_world_cam is not None and np.all(np.isfinite(data.T_world_cam[f])):
            T = data.T_world_cam[f]
            rec.log("world/camera", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        for slot, side in enumerate(sides):
            color = _rgb(mjcf.HAND_HEX[side])
            if data.joints is not None:
                rec.log(f"world/hands/{side}", rr.LineStrips3D(_strips(data.joints[f, slot]), colors=[color], radii=[0.003]))
            if data.shoulder is not None and data.elbow is not None:
                wrist = data.wrist[f, slot] if data.wrist is not None else (
                    data.joints[f, slot, 0] if data.joints is not None else np.full(3, np.nan))
                arm = np.stack([data.shoulder[f, slot], data.elbow[f, slot], wrist])
                strips = [arm] if np.all(np.isfinite(arm)) else []
                rec.log(f"world/arms/{side}", rr.LineStrips3D(strips, colors=[_rgb("#9a9994")], radii=[0.012]))
            if data.gt is not None:
                rec.log(f"world/gt/hands/{side}", rr.LineStrips3D(_strips(data.gt.joints[f, slot]),
                                                                  colors=[_rgb("#8a8985", 160)], radii=[0.0015]))
            if kp2d is not None and f < len(kp2d):
                uv = kp2d[f, slot, :, :2]
                if cam is not None:  # full-resolution pixels -> pixels of the logged images
                    uv = to_image_px(uv, (cam.width, cam.height), image_wh)
                ok = np.all(np.isfinite(uv), axis=-1)
                rec.log(f"world/camera/image/hands/{side}", rr.Points2D(uv[ok], colors=[color], radii=[2.0]))
                strips2d = [uv[c] for c in CHAINS if np.all(ok[c])]
                rec.log(f"world/camera/image/hands/{side}/bones", rr.LineStrips2D(strips2d, colors=[color], radii=[1.0]))
        for k, obj in enumerate(data.objects):
            T = held[k][f]
            if np.all(np.isfinite(T)):
                rec.log(f"world/objects/{obj.id}", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))

    # ---------------------------------------------------------------- scalar plots (columnar)
    def scalars(path: str, times: np.ndarray, values: np.ndarray) -> None:
        ok = np.isfinite(values)
        if np.any(ok):
            rr.send_columns(path, indexes=[rr.TimeColumn("time", duration=times[ok])],
                            columns=rr.Scalars.columns(scalars=values[ok]), recording=rec)

    if data.imu is not None:
        imu_t = data.imu["imu_t"]
        for name in ("acc", "gyro"):
            if name in data.imu:
                for i, axis in enumerate("xyz"):
                    scalars(f"imu/{name}/{axis}", imu_t, np.asarray(data.imu[name][:, i], dtype=np.float64))
    if data.joints is not None:
        for slot, side in enumerate(sides):
            scalars(f"plots/wrist_z/{side}", t, data.joints[:, slot, 0, 2])
    for obj in data.objects:
        scalars(f"plots/object_z/{obj.id}", t, np.where(obj.valid, obj.T_world_obj[:, 2, 3], np.nan))
    if data.elbow_conf is not None:
        for slot, side in enumerate(sides):
            scalars(f"plots/elbow_conf/{side}", t, data.elbow_conf[:, slot])
    if data.retarget is not None and "err_m" in data.retarget:
        err = np.asarray(data.retarget["err_m"], dtype=np.float64)
        scalars("plots/retarget_err_m", t[: len(err)], err)
    rec.flush()
    rec.disconnect()
    return {"path": str(out_path), "frames": data.T, "video": video_info}
