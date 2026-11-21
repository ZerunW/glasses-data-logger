"""eval-charuco on a synthetic board video: known poses, depth with a known scale error."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from glassesdata_pipeline import geometry as geo
from glassesdata_pipeline.episode import Episode
from glassesdata_pipeline.eval import charuco

SPEC = {"dictionary": "DICT_5X5_100", "squares_x": 9, "squares_y": 6, "square_m": 0.03, "marker_m": 0.022,
        "css_px_per_mm": 3.78}
W, H = 540, 960
DS = 4
MODEL_ERROR = 1.25  # metric / model


def _poses(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    size = np.array([SPEC["squares_x"], SPEC["squares_y"], 0.0]) * SPEC["square_m"]
    out = []
    for k in range(n):
        R = geo.so3_exp([0.25 * np.sin(k), 0.3 * np.cos(0.7 * k), 0.1 * k / n])
        centre = np.array([0.01 * k - 0.03, 0.02 * np.sin(k), 0.36 + 0.02 * k])
        t = centre - R @ (size / 2)
        out.append((R, t))
    return out


def _render(camera: geo.Camera, R: np.ndarray, t: np.ndarray, board_img: np.ndarray, px_per_sq: int) -> np.ndarray:
    rays = camera.pixel_rays()
    n = R[:, 2]
    s = float(n @ t) / (rays @ n)
    local = (rays * s[..., None] - t) @ R
    pad = px_per_sq  # one square of white screen around the board
    map_x = (local[..., 0] / SPEC["square_m"] * px_per_sq - 0.5 + pad).astype(np.float32)
    map_y = (local[..., 1] / SPEC["square_m"] * px_per_sq - 0.5 + pad).astype(np.float32)
    screen = cv2.copyMakeBorder(board_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    gray = cv2.remap(screen, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=90)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


@pytest.fixture(scope="module")
def board_recording(tmp_path_factory):
    from glassesdata_pipeline.synth.writer import VideoWriter

    rec = tmp_path_factory.mktemp("charuco") / "rec"
    rec.mkdir()
    base = geo.Camera.fov_default(W, H, 65.0)
    camera = geo.Camera(base.K, [-0.05, 0.01, 0.0, 0.0, 0.0], W, H, "pinhole", "charuco-web")
    calib = camera.to_dict()
    calib["board"] = SPEC
    (rec / "calibration.json").write_text(json.dumps(calib))
    px_per_sq = 60
    board = charuco.make_board(SPEC)
    board_img = board.generateImage((SPEC["squares_x"] * px_per_sq, SPEC["squares_y"] * px_per_sq), marginSize=0, borderBits=1)
    poses = _poses(8)
    writer = VideoWriter(rec / "video.mp4", W, H, 24.0, 16, "fast", 24)
    try:
        for k, (R, t) in enumerate(poses):
            writer.write(_render(camera, R, t, board_img, px_per_sq), 1_000_000 + k * 41_667)
    finally:
        writer.close()
    ep = Episode(rec)
    grid = camera.downsampled(DS)
    depth = np.lib.format.open_memmap(ep.file("depth/depth_f16.npy"), mode="w+", dtype=np.float16,
                                      shape=(len(poses), grid.height, grid.width))
    size = (SPEC["squares_x"] * SPEC["square_m"], SPEC["squares_y"] * SPEC["square_m"])
    for k, (R, t) in enumerate(poses):
        true = charuco.board_depth(grid, R, t, size)
        depth[k] = np.where(np.isfinite(true), true, 0.8) / MODEL_ERROR
    depth.flush()
    del depth
    ep.save_npz("depth/index", {"step": "depth", "backend": "test"}, frames=np.arange(len(poses)), scale=grid.height / H)
    return rec, poses


def test_board_pose_and_depth(board_recording):
    rec, poses = board_recording
    ep = Episode(rec)
    camera = ep.camera()
    spec = charuco.board_spec(ep)
    board = charuco.make_board(spec)
    detector = charuco.make_detector(board)
    for k, gray in ep.frames(range(len(poses)), gray=True):
        pose = charuco.board_pose(detector, board, gray, camera)
        assert pose is not None, k
        R, t = poses[k]
        assert np.degrees(geo.rotation_angle(pose["R"], R)) < 1.0
        assert np.linalg.norm(pose["t"] - t) < 0.004
        assert pose["reproj_px"] < 1.0


def test_eval_charuco_scale_and_errors(board_recording, tmp_path):
    import shutil

    rec, poses = board_recording
    work = tmp_path / "rec"
    shutil.copytree(rec, work)
    ep = Episode(work)
    m = charuco.run(ep)
    assert m["frames_used"] == len(poses)
    assert abs(m["scale_board_median"] / MODEL_ERROR - 1) < 0.01
    assert abs(m["raw"]["abs_rel"] - (1 - 1 / MODEL_ERROR)) < 0.01
    assert "scaled" not in m
    # with hand3d's scale: the scaled depth matches the board
    T = len(poses)
    scale = np.full(T, MODEL_ERROR)
    ep.save_npz("hand3d", {"step": "hand3d"}, joints_cam=np.full((T, 2, 21, 3), np.nan), valid=np.zeros((T, 2), bool),
                reproj_px=np.full((T, 2), np.nan), depth_scale=np.full(T, np.nan), depth_scale_smooth=scale)
    assert charuco.main([str(work), "--max-frames", "4"]) == 0
    saved = json.loads((ep.derived / "eval_charuco.json").read_text())
    m = saved["metrics"]
    assert m["frames_used"] == 4 and saved["meta"]["step"] == "eval-charuco"
    assert m["scaled"]["abs_rel"] < 0.01 and abs(m["pipeline_scale_rel_error"]["median"]) < 0.01
    assert (ep.derived / "eval_charuco.png").is_file() and (ep.derived / "eval_charuco.md").is_file()


def test_board_spec_from_cli_and_errors(board_recording, tmp_path):
    rec, _ = board_recording
    ep = Episode(rec)
    spec = charuco.board_spec(ep, {"squares_x": 7})
    assert spec["squares_x"] == 7 and spec["dictionary"] == "DICT_5X5_100"
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "video.mp4").write_bytes((rec / "video.mp4").read_bytes())
    with pytest.raises(Exception, match="board definition"):
        charuco.board_spec(Episode(empty))
