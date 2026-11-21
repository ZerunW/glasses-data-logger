"""Synthetic camera views of a ChArUco board, for the calibration tests.

Every output pixel is traced back through a known camera (OpenCV pinhole or fisheye
distortion) to the board plane and sampled from the board image (``cv2.remap``),
supersampled 2x and averaged, then blurred and given noise like a compressed video frame.
The board is drawn on a white "screen" with a dark surround.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from glassesdata_receiver.calibration import BoardSpec

TEXTURE_PX = 96  # texture pixels per board square
SCREEN_MARGIN = 0.5  # white around the board, in squares


class SyntheticCamera:
    def __init__(self, K, dist, size: tuple[int, int], model: str = "pinhole", supersample: int = 2) -> None:
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64)
        self.size = size
        self.model = model
        self.ss = supersample
        width, height = size[0] * supersample, size[1] * supersample
        us, vs = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
        points = np.stack([(us + 0.5) / supersample - 0.5, (vs + 0.5) / supersample - 0.5], -1).reshape(-1, 1, 2)
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-10)
        if model == "fisheye":
            normalised = cv2.fisheye.undistortPoints(points, self.K, self.dist, criteria=criteria)
        else:
            normalised = cv2.undistortPoints(points, self.K, self.dist, None, None, None, criteria)
        self.rays = np.concatenate([normalised.reshape(-1, 2), np.ones((len(points), 1))], axis=1)
        self.shape = (height, width)

    def project(self, object_points: np.ndarray, rvec, tvec) -> np.ndarray:
        obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
        if self.model == "fisheye":
            image, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, self.K, self.dist)
        else:
            image, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)
        return image.reshape(-1, 2)

    def render(self, spec: BoardSpec, rvec, tvec, rng=None, blur: float = 0.6, noise: float = 2.0) -> np.ndarray:
        """A grey camera image (uint8) of the board at pose (rvec, tvec): board -> camera."""
        texture, px_per_m, offset = board_texture(spec)
        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        t = np.asarray(tvec, dtype=np.float64).reshape(3)
        in_board = self.rays @ R  # = (R^T d) for every ray d, as rows
        t_board = R.T @ t
        with np.errstate(divide="ignore", invalid="ignore"):
            s = t_board[2] / in_board[:, 2]
        X = s[:, None] * in_board - t_board
        behind = ~(s > 0)
        map_x = (X[:, 0] * px_per_m + offset - 0.5).astype(np.float32).reshape(self.shape)
        map_y = (X[:, 1] * px_per_m + offset - 0.5).astype(np.float32).reshape(self.shape)
        map_x[behind.reshape(self.shape)] = -1e6
        big = cv2.remap(texture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=55)
        image = cv2.resize(big, self.size, interpolation=cv2.INTER_AREA).astype(np.float32)
        if blur:
            image = cv2.GaussianBlur(image, (0, 0), blur)
        if noise and rng is not None:
            image = image + rng.normal(0.0, noise, image.shape)
        return np.clip(image, 0, 255).astype(np.uint8)


_textures: dict[BoardSpec, tuple[np.ndarray, float, float]] = {}


def board_texture(spec: BoardSpec) -> tuple[np.ndarray, float, float]:
    """The board drawn at TEXTURE_PX per square on a white screen; pixels per metre; offset of the board origin."""
    if spec not in _textures:
        board = spec.board()
        image = board.generateImage((spec.squares_x * TEXTURE_PX, spec.squares_y * TEXTURE_PX), marginSize=0, borderBits=1)
        margin = int(SCREEN_MARGIN * TEXTURE_PX)
        image = cv2.copyMakeBorder(image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)
        _textures[spec] = (image, TEXTURE_PX / spec.square_m, float(margin))
    return _textures[spec]


def look_at(camera_position: np.ndarray, target: np.ndarray, roll_rad: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """rvec, tvec (board -> camera) of a camera at ``camera_position`` looking at ``target`` (board coordinates).
    The camera's y axis points roughly along the board's y (down the screen)."""
    z = target - camera_position
    z = z / np.linalg.norm(z)
    x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    roll = cv2.Rodrigues(np.array([0.0, 0.0, roll_rad]))[0]
    R = roll @ R
    t = -R @ camera_position
    return cv2.Rodrigues(R)[0].ravel(), t


def aim(rvec, tvec, K, pixel) -> tuple[np.ndarray, np.ndarray]:
    """Turn the camera about its centre so the ray that pointed along the optical axis now
    passes through ``pixel`` (a distortion-free approximation): the board moves there."""
    direction = np.array([(pixel[0] - K[0, 2]) / K[0, 0], (pixel[1] - K[1, 2]) / K[1, 1], 1.0])
    direction /= np.linalg.norm(direction)
    axis = np.cross(np.array([0.0, 0.0, 1.0]), direction)
    if np.linalg.norm(axis) < 1e-12:
        return np.asarray(rvec, dtype=np.float64), np.asarray(tvec, dtype=np.float64)
    angle = math.acos(float(np.clip(direction[2], -1, 1)))
    turn = cv2.Rodrigues(axis / np.linalg.norm(axis) * angle)[0]  # maps the optical axis onto `direction`
    R = turn @ cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    return cv2.Rodrigues(R)[0].ravel(), turn @ np.asarray(tvec, dtype=np.float64)


def poses(
    spec: BoardSpec, camera: SyntheticCamera, count: int, rng, distance=(0.25, 0.55), tilt_deg=(0.0, 40.0), visible=0.6
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``count`` poses spread over the image (target positions taken from a 3 x 3 grid in turn)
    at random distances, tilts and rolls. Like a person looking at a screen, a view may cut
    off part of the board, but at least ``visible`` of its corners are in the image."""
    width, height = camera.size
    board_w, board_h = spec.squares_x * spec.square_m, spec.squares_y * spec.square_m
    centre = np.array([board_w / 2, board_h / 2, 0.0])
    outline = np.array([[0, 0, 0], [board_w, 0, 0], [board_w, board_h, 0], [0, board_h, 0]], dtype=np.float64)
    inner = np.asarray(spec.board().getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    attempts = 0
    while len(out) < count:
        attempts += 1
        assert attempts < count * 400, "could not place the board"
        cell = len(out) % 9
        target = np.array([(cell % 3 + rng.uniform(0.0, 1.0)) / 3 * width, (cell // 3 + rng.uniform(0.0, 1.0)) / 3 * height])
        d = rng.uniform(*distance)
        tilt = math.radians(rng.uniform(*tilt_deg))
        azimuth = rng.uniform(0, 2 * math.pi)
        position = centre + d * np.array([math.sin(tilt) * math.cos(azimuth), math.sin(tilt) * math.sin(azimuth), -math.cos(tilt)])
        rvec, tvec = look_at(position, centre, rng.uniform(-0.35, 0.35))
        rvec, tvec = aim(rvec, tvec, camera.K, target)
        R = cv2.Rodrigues(rvec)[0]
        if np.any((outline @ R.T + tvec)[:, 2] < 0.05):
            continue
        if camera.model == "fisheye":  # keep the board well inside the fisheye's valid field
            rays = inner @ R.T + tvec
            if np.any(np.arccos(rays[:, 2] / np.linalg.norm(rays, axis=1)) > math.radians(80)):
                continue
        points = camera.project(inner, rvec, tvec)
        margin = 0.02 * width
        inside = (points[:, 0] > margin) & (points[:, 1] > margin) & (points[:, 0] < width - margin) & (points[:, 1] < height - margin)
        if inside.mean() < visible:
            continue
        out.append((rvec, tvec))
    return out


def write_hevc(path, frames, size: tuple[int, int], fps: int = 30) -> int:
    """Encode grey or BGR frames as HEVC in MP4 without B-frames (what tools/fake_sender.py
    plays, like the glasses stream). Returns the number of frames written."""
    import av

    count = 0
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("libx265", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.options = {
            "preset": "ultrafast", "crf": "12",
            "x265-params": f"bframes=0:keyint={fps}:min-keyint={fps}:scenecut=0:log-level=error",
        }
        for image in frames:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = count
            for packet in stream.encode(frame):
                container.mux(packet)
            count += 1
        for packet in stream.encode():
            container.mux(packet)
    return count


def calibration_video(path, spec: BoardSpec, camera: SyntheticCamera, views: int, rng, hold_frames: int = 16, fps: int = 30) -> list:
    """A video that holds each of ``views`` poses for ``hold_frames`` frames, like a person
    pausing at each position. Returns the poses."""
    chosen = poses(spec, camera, views, rng)
    images = (camera.render(spec, rvec, tvec, rng) for rvec, tvec in chosen)

    def frames():
        for image in images:
            for _ in range(hold_frames):
                yield image

    write_hevc(path, frames(), camera.size, fps)
    return chosen
