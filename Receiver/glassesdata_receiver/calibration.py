"""Camera calibration with a ChArUco board shown on this computer's screen.

The person wears the glasses and looks at the screen. The page first measures the screen
(a bank card held against an on-screen outline gives CSS pixels per millimetre), then shows
a board that :func:`design_board` sized to the screen, drawn by OpenCV at one image pixel
per device pixel so it is sharp and of known physical size.

While the board is shown, a :class:`CaptureSession` thread looks at the receiver's newest
decoded frame at most ``1 / DETECT_INTERVAL_S`` times a second (it only ever waits on the
decoder's condition variable, so decoding and recording never wait for it), detects the
board and keeps the views that add something (:class:`SampleSelector`): parts of the image
not covered yet, or a distance or tilt not seen yet, judged with a provisional camera
(65° horizontal field of view, or the current calibration).

:func:`solve` fits both OpenCV models, the pinhole one (``calibrateCamera``: k1 k2 p1 p2
k3) and the fisheye one (``cv2.fisheye``), drops views whose error is more than three times
the median and fits again, and keeps the model with the lower error on views left out of
the fit (k-fold). The result is written as ``calibration.json`` (PIPELINE.md section 3) by
:class:`CalibrationStore`: a timestamped file plus ``current.json``, which every new
recording copies.

:class:`CalibrationService` ties it together for the web server's ``/api/calibration``
endpoints.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import cv2
import numpy as np

from . import paths
from .bodyprofile import write_json_atomic

if TYPE_CHECKING:
    from .receiver import Receiver

log = logging.getLogger(__name__)

DICTIONARY = "DICT_5X5_100"
MARKER_MODULES = 7  # a 5x5 marker plus its one-module black border, in modules per side
MARKER_RATIO = 0.75  # marker side / square side
MIN_SQUARE_MM = 25.0
MAX_SQUARE_MM = 45.0
EDGE_MARGIN_MM = 6.0  # white around the board inside the area the page gives it
CARD_MM = (85.60, 53.98)  # ISO/IEC 7810 ID-1: bank cards, many ID cards

GUESS_HFOV_DEG = 65.0
MIN_CORNERS = 12
TARGET_SAMPLES = 25
MAX_SAMPLES = 40
MIN_SOLVE_SAMPLES = 6
DETECT_INTERVAL_S = 0.2  # at most 5 detections a second
NO_VIDEO_S = 1.5
STEADY_FRACTION = 0.012  # of the image width: how far the corners may move between two looks
DUPLICATE_FRACTION = 0.04  # of the image width: a view this close to a kept one adds nothing
MIN_ACCEPT_GAP_S = 0.5
BIN_CAP = 2  # views per pose bin that still count as new
GRID_SHORT_SIDE = 6  # coverage cells along the image's short side
DISTANCE_EDGES = (1.35, 2.0)  # distance / (distance at which the board spans the image width)
TILT_EDGES_DEG = (15.0, 30.0)
PREVIEW_HEIGHT = 360
UNDISTORTED_HEIGHT = 560
OUTLIER_FACTOR = 3.0
FISHEYE_MARGIN = 0.9  # fisheye wins only with a held-out error below 90 % of the pinhole one
FOLDS = 5
DEVICE = "Ray-Ban Meta"
SOURCE = "charuco-web"
_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- the board


@dataclass(frozen=True)
class BoardSpec:
    """A ChArUco board drawn at ``square_px`` device pixels per square.

    ``square_m`` and ``marker_m`` follow from the pixel sizes and the measured screen scale,
    so the drawing is exact: the page shows the image at one image pixel per device pixel.
    """

    squares_x: int
    squares_y: int
    square_m: float
    marker_m: float
    css_px_per_mm: float
    device_pixel_ratio: float
    square_px: int
    marker_px: int
    dictionary: str = DICTIONARY

    @property
    def width_px(self) -> int:
        return self.squares_x * self.square_px

    @property
    def height_px(self) -> int:
        return self.squares_y * self.square_px

    def board(self) -> Any:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        return cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_m, self.marker_m, dictionary)

    def image(self) -> np.ndarray:
        """The board, black on white, exactly ``width_px`` x ``height_px`` (no margin)."""
        return self.board().generateImage((self.width_px, self.height_px), marginSize=0, borderBits=1)

    def png(self) -> bytes:
        ok, data = cv2.imencode(".png", self.image())
        if not ok:  # pragma: no cover - a plain 8-bit image always encodes
            raise RuntimeError("could not encode the board")
        return data.tobytes()

    def to_json(self) -> dict[str, Any]:
        """The ``board`` entry of calibration.json (PIPELINE.md section 3)."""
        return {
            "dictionary": self.dictionary,
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_m": round(self.square_m, 7),
            "marker_m": round(self.marker_m, 7),
            "css_px_per_mm": round(self.css_px_per_mm, 4),
        }

    def describe(self) -> dict[str, Any]:
        """For the page: the board and how to show it."""
        return {
            **self.to_json(),
            "device_pixel_ratio": self.device_pixel_ratio,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "css_width": self.width_px / self.device_pixel_ratio,
            "css_height": self.height_px / self.device_pixel_ratio,
            "corners": (self.squares_x - 1) * (self.squares_y - 1),
        }


def _marker_px(square_px: int) -> int:
    """A marker size in whole modules, near MARKER_RATIO of the square, centred on whole pixels."""
    best = None
    for modules in range(1, square_px // MARKER_MODULES + 1):
        marker = modules * MARKER_MODULES
        ratio = marker / square_px
        if not 0.6 <= ratio <= 0.82:
            continue
        score = abs(ratio - MARKER_RATIO) + (0.05 if (square_px - marker) % 2 else 0.0)
        if best is None or score < best[0]:
            best = (score, marker)
    return best[1] if best else max(1, round(square_px * MARKER_RATIO))


def design_board(css_px_per_mm: float, device_pixel_ratio: float, area_css: tuple[float, float]) -> BoardSpec:
    """The board for an area of the screen (CSS pixels) at a measured scale.

    Squares of at least ``MIN_SQUARE_MM`` (smaller only if the area is too small for a 4 x 3
    board) and at most ``MAX_SQUARE_MM``, about six along the short side, and as many as fill
    the area. Raises ValueError for impossible inputs.
    """
    for name, value, low, high in (
        ("css_px_per_mm", css_px_per_mm, 1.0, 40.0),
        ("device_pixel_ratio", device_pixel_ratio, 0.5, 8.0),
        ("area width", area_css[0], 100.0, 20000.0),
        ("area height", area_css[1], 100.0, 20000.0),
    ):
        if not (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and low <= value <= high):
            raise ValueError(f"{name} must be a number from {low:g} to {high:g}")
    device_per_mm = css_px_per_mm * device_pixel_ratio
    width_mm = area_css[0] / css_px_per_mm - 2 * EDGE_MARGIN_MM
    height_mm = area_css[1] / css_px_per_mm - 2 * EDGE_MARGIN_MM
    target = min(max(min(width_mm, height_mm) / 6.0, MIN_SQUARE_MM), MAX_SQUARE_MM)
    squares_x = max(4, int(width_mm // target))
    squares_y = max(3, int(height_mm // target))
    while squares_x * squares_y // 2 > 100:  # DICT_5X5_100 has 100 markers
        if squares_x >= squares_y:
            squares_x -= 1
        else:
            squares_y -= 1
    square_mm = min(width_mm / squares_x, height_mm / squares_y, MAX_SQUARE_MM)
    # The other direction takes as many squares of that size as fit (a narrow screen gets more rows).
    squares_x = max(squares_x, min(int(width_mm // square_mm), 200 // squares_y))
    squares_y = max(squares_y, min(int(height_mm // square_mm), 200 // squares_x))
    square_px = int(math.floor(square_mm * device_per_mm))
    if square_px < 4 * MARKER_MODULES:
        raise ValueError("the screen area is too small for a board")
    marker_px = _marker_px(square_px)
    return BoardSpec(
        squares_x=squares_x,
        squares_y=squares_y,
        square_m=square_px / device_per_mm / 1000.0,
        marker_m=marker_px / device_per_mm / 1000.0,
        css_px_per_mm=float(css_px_per_mm),
        device_pixel_ratio=float(device_pixel_ratio),
        square_px=square_px,
        marker_px=marker_px,
    )


def detect(detector: Any, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """ChArUco corners (N x 2, float32), their ids (N, int32) and the marker outlines."""
    corners, ids, markers, _ = detector.detectBoard(gray)
    if ids is None or corners is None or len(np.ravel(ids)) == 0:
        return np.zeros((0, 2), np.float32), np.zeros(0, np.int32), list(markers or [])
    return np.reshape(corners, (-1, 2)).astype(np.float32), np.ravel(ids).astype(np.int32), list(markers or [])


def make_detector(spec: BoardSpec) -> Any:
    parameters = cv2.aruco.CharucoParameters()
    parameters.tryRefineMarkers = True
    return cv2.aruco.CharucoDetector(spec.board(), parameters)


# ---------------------------------------------------------------- choosing views


def guess_camera(width: int, height: int, hfov_deg: float = GUESS_HFOV_DEG) -> np.ndarray:
    """A pinhole camera with the given horizontal field of view (the pipeline's default camera)."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0.0, (width - 1) / 2.0], [0.0, fx, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


def provisional_camera(width: int, height: int, current: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    """K and distortion to judge distance and tilt with: the current calibration rescaled to this
    size when it has the same aspect, else the 65° guess."""
    try:
        if current and abs(width / height - current["width"] / current["height"]) < 0.01 * width / height:
            scale = width / current["width"]
            K = np.array(current["K"], dtype=np.float64)
            K[0, :] *= scale
            K[1, :] *= scale
            K[2] = [0.0, 0.0, 1.0]
            dist = np.array(current.get("dist") or [], dtype=np.float64) if current.get("model") == "pinhole" else np.zeros(0)
            return K, dist
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        pass
    return guess_camera(width, height), np.zeros(0)


@dataclass(frozen=True)
class Pose:
    distance_m: float
    reach: float  # distance / the distance at which the board would span the image width
    tilt_deg: float
    side: str  # where the camera is relative to the board's normal: "", "left", "right", "above", "below"

    @property
    def distance_bin(self) -> str:
        return "near" if self.reach < DISTANCE_EDGES[0] else "mid" if self.reach < DISTANCE_EDGES[1] else "far"

    @property
    def tilt_bin(self) -> int:
        return 0 if self.tilt_deg < TILT_EDGES_DEG[0] else 1 if self.tilt_deg < TILT_EDGES_DEG[1] else 2

    @property
    def key(self) -> tuple[str, int, str]:
        return self.distance_bin, self.tilt_bin, self.side if self.tilt_bin else ""


def estimate_pose(
    object_points: np.ndarray, image_points: np.ndarray, K: np.ndarray, dist: np.ndarray, board_size_m: tuple[float, float], width: int
) -> Pose | None:
    """Where the camera is relative to the board, with a provisional camera."""
    if len(object_points) < 4:
        return None
    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points.astype(np.float64), image_points.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_IPPE
        )
    except cv2.error:
        return None
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    camera = -R.T @ tvec.reshape(3)
    view = camera - np.array([board_size_m[0] / 2.0, board_size_m[1] / 2.0, 0.0])
    distance = float(np.linalg.norm(view))
    if distance <= 0 or view[2] >= 0:  # the board's front faces -z; anything else is a bad solution
        return None
    tilt = math.degrees(math.acos(min(1.0, -view[2] / distance)))
    side = ""
    if tilt >= TILT_EDGES_DEG[0]:
        if abs(view[0]) >= abs(view[1]):
            side = "left" if view[0] < 0 else "right"
        else:
            side = "above" if view[1] < 0 else "below"
    span = K[0, 0] * board_size_m[0] / width
    return Pose(distance, distance / span, tilt, side)


@dataclass
class Sample:
    corners: np.ndarray  # N x 2 float32, pixels
    ids: np.ndarray  # N int32, ChArUco corner ids
    pose: Pose | None
    time: float
    jpeg: bytes | None = None  # the frame, for the undistorted preview


@dataclass
class Look:
    """What one look at a frame found."""

    corners: int = 0
    steady: bool = False
    accepted: bool = False
    reason: str = ""
    pose: Pose | None = None
    points: list[list[float]] = field(default_factory=list)  # corners normalised to the image, for the page
    markers: int = 0
    near_edge: bool = False


REGION_NAMES = (("top left", "top", "top right"), ("left", "middle", "right"), ("bottom left", "bottom", "bottom right"))


class SampleSelector:
    """Keeps the views that add something: new parts of the image, or a new distance or tilt.

    Not thread-safe; :class:`CaptureSession` calls it with its lock held.
    """

    def __init__(self, spec: BoardSpec, width: int, height: int, K: np.ndarray, dist: np.ndarray) -> None:
        self.spec = spec
        self.width, self.height = width, height
        self.K, self.dist = K, dist
        self.object_points = np.asarray(spec.board().getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
        self.board_size_m = (spec.squares_x * spec.square_m, spec.squares_y * spec.square_m)
        if width <= height:
            self.cols, self.rows = GRID_SHORT_SIDE, max(GRID_SHORT_SIDE, round(GRID_SHORT_SIDE * height / width))
        else:
            self.rows, self.cols = GRID_SHORT_SIDE, max(GRID_SHORT_SIDE, round(GRID_SHORT_SIDE * width / height))
        self.cells = np.zeros((self.rows, self.cols), dtype=np.int32)  # kept views with a corner in each cell
        self.bins: Counter[tuple[str, int, str]] = Counter()
        self.samples: list[Sample] = []
        self._previous: tuple[np.ndarray, np.ndarray] | None = None
        self._last_accept = -math.inf
        self._board = spec.board()

    # -------------------------------------------------------- one look

    def consider(
        self, corners: np.ndarray, ids: np.ndarray, now: float, markers: int = 0, encode: Callable[[], bytes] | None = None
    ) -> Look:
        look = Look(corners=len(ids), markers=markers)
        look.points = [[round(float(x) / self.width, 4), round(float(y) / self.height, 4)] for x, y in corners]
        previous, self._previous = self._previous, (ids, corners)
        if len(ids):
            margin = 0.02 * self.width
            look.near_edge = bool(
                corners[:, 0].min() < margin or corners[:, 1].min() < margin
                or corners[:, 0].max() > self.width - 1 - margin or corners[:, 1].max() > self.height - 1 - margin
            )
        if len(ids) >= 4:
            look.pose = estimate_pose(self.object_points[ids], corners, self.K, self.dist, self.board_size_m, self.width)
        look.steady = previous is not None and self._moved(previous, ids, corners) < STEADY_FRACTION * self.width
        if len(self.samples) >= MAX_SAMPLES:
            look.reason = "enough views"
        elif len(ids) < MIN_CORNERS:
            look.reason = "too few corners" if len(ids) else "no board"
        elif self._board.checkCharucoCornersCollinear(ids.reshape(-1, 1)):
            look.reason = "corners in a line"
        elif not look.steady:
            look.reason = "moving"
        elif now - self._last_accept < MIN_ACCEPT_GAP_S:
            look.reason = "just kept one"
        elif any(self._duplicate(sample, ids, corners) for sample in self.samples):
            look.reason = "already have this view"
        else:
            new_cells = self._new_cells(corners)
            key = look.pose.key if look.pose is not None else None
            if new_cells == 0 and (key is None or self.bins[key] >= BIN_CAP):
                look.reason = "nothing new"
            else:
                self._keep(Sample(corners.copy(), ids.copy(), look.pose, now, encode() if encode else None))
                look.accepted = True
                look.reason = f"kept: {new_cells} new cells" if new_cells else "kept: new distance or tilt"
        return look

    def _keep(self, sample: Sample) -> None:
        self.samples.append(sample)
        for col, row in self._cells_of(sample.corners):
            self.cells[row, col] += 1
        if sample.pose is not None:
            self.bins[sample.pose.key] += 1
        self._last_accept = sample.time

    @staticmethod
    def _moved(previous: tuple[np.ndarray, np.ndarray], ids: np.ndarray, corners: np.ndarray) -> float:
        """Mean distance the shared corners moved (infinite when too few are shared)."""
        before_ids, before = previous
        shared, here, there = np.intersect1d(ids, before_ids, return_indices=True)
        if len(shared) < max(4, len(ids) // 2):
            return math.inf
        return float(np.linalg.norm(corners[here] - before[there], axis=1).mean())

    def _duplicate(self, sample: Sample, ids: np.ndarray, corners: np.ndarray) -> bool:
        shared, here, there = np.intersect1d(ids, sample.ids, return_indices=True)
        if len(shared) < 0.6 * min(len(ids), len(sample.ids)):
            return False
        return float(np.linalg.norm(corners[here] - sample.corners[there], axis=1).mean()) < DUPLICATE_FRACTION * self.width

    def _cells_of(self, corners: np.ndarray) -> set[tuple[int, int]]:
        cols = np.clip((corners[:, 0] / self.width * self.cols).astype(int), 0, self.cols - 1)
        rows = np.clip((corners[:, 1] / self.height * self.rows).astype(int), 0, self.rows - 1)
        return set(zip(cols.tolist(), rows.tolist()))

    def _new_cells(self, corners: np.ndarray) -> int:
        return sum(1 for col, row in self._cells_of(corners) if self.cells[row, col] == 0)

    # -------------------------------------------------------- progress

    @property
    def coverage(self) -> float:
        return float((self.cells > 0).mean())

    def region_coverage(self) -> list[list[float]]:
        """Covered fraction of each third of the image (3 x 3, rows first)."""
        covered = self.cells > 0
        out = []
        for r in range(3):
            row = []
            for c in range(3):
                block = covered[r * self.rows // 3 : (r + 1) * self.rows // 3, c * self.cols // 3 : (c + 1) * self.cols // 3]
                row.append(float(block.mean()) if block.size else 1.0)
            out.append(row)
        return out

    def distance_counts(self) -> dict[str, int]:
        counts = {"near": 0, "mid": 0, "far": 0}
        for (distance, _, _), count in self.bins.items():
            counts[distance] += count
        return counts

    def side_counts(self) -> dict[str, int]:
        counts = {"left": 0, "right": 0, "above": 0, "below": 0}
        for (_, tilt, side), count in self.bins.items():
            if tilt and side:
                counts[side] += count
        return counts

    def next_step(self) -> str:
        """What would add the most now, in words the person can act on."""
        if len(self.samples) >= TARGET_SAMPLES:
            return "Enough views: press Solve"
        regions = self.region_coverage()
        worst = min(((regions[r][c], r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)))
        if worst[0] < 0.5:
            return f"Next: board in the {REGION_NAMES[worst[1]][worst[2]]} of your view"
        distances = self.distance_counts()
        if distances["near"] == 0:
            return "Next: move closer"
        if distances["far"] == 0:
            return "Next: step back"
        sides = self.side_counts()
        for side in ("left", "right", "above", "below"):
            if sides[side] == 0:
                return f"Next: look at the board from {'the ' + side if side in ('left', 'right') else side}"
        return "Keep going: vary distance and angle"


def guidance(look: Look | None, selector: SampleSelector | None, video: bool) -> str:
    """One short line for the person wearing the glasses."""
    if not video:
        return "No video from the glasses"
    if look is None or selector is None:
        return "Look at the board"
    if look.corners == 0:
        return "Look at the board" if look.markers == 0 else "Show more of the board"
    if look.corners < MIN_CORNERS:
        return "Show the whole board" if look.near_edge else "Move closer"
    if len(selector.samples) >= MAX_SAMPLES:
        return "Enough views: press Solve"
    if not look.steady:
        return "Hold still"
    return selector.next_step()


# ---------------------------------------------------------------- the capture thread


class CaptureSession:
    """Detects the board in the receiver's newest frame while capturing (its own thread)."""

    def __init__(self, receiver: "Receiver", spec: BoardSpec, current: dict[str, Any] | None = None) -> None:
        self.receiver = receiver
        self.spec = spec
        self.created = utc_now()
        self._current = current
        self._detector = make_detector(spec)
        self._lock = threading.Lock()
        self._selector: SampleSelector | None = None
        self._look: Look | None = None
        self._look_at = 0.0
        self._video = False
        self._preview: bytes | None = None
        self._sequence = 0
        self._accepted = 0
        self._note = ""
        self._failures = 0
        self._capturing = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gd-calibration", daemon=True)

    def start(self) -> None:
        self._capturing.set()
        self._thread.start()

    def set_capturing(self, on: bool) -> None:
        if on:
            self._capturing.set()
        else:
            self._capturing.clear()

    @property
    def capturing(self) -> bool:
        return self._capturing.is_set()

    def close(self) -> None:
        self._stop.set()
        self._capturing.set()  # wake the loop
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(2.0)

    # -------------------------------------------------------- loop

    def _run(self) -> None:
        last_index = -1
        next_at = 0.0
        last_frame = time.monotonic()
        while not self._stop.is_set():
            if not self._capturing.wait(0.25):
                continue
            if self._stop.is_set():
                break
            delay = next_at - time.monotonic()
            if delay > 0 and self._stop.wait(delay):
                break
            frame = self.receiver.wait_frame(last_index, timeout=0.5)  # the newest frame; never holds up the decoder
            if frame is None or not self.receiver.busy:
                if time.monotonic() - last_frame > NO_VIDEO_S:
                    with self._lock:
                        self._video = False
                continue
            last_frame = time.monotonic()
            last_index = frame.index
            next_at = last_frame + DETECT_INTERVAL_S
            try:
                self.process(frame.image, last_frame)
            except Exception:
                self._failures += 1
                if self._failures == 1:
                    log.exception("Calibration: could not process a frame")

    def process(self, image: np.ndarray, now: float) -> Look:
        """Detect the board in one BGR frame and keep it if it adds something."""
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, markers = detect(self._detector, gray)
        with self._lock:
            selector = self._selector
            if selector is None or (selector.width, selector.height) != (width, height):
                if selector is not None and selector.samples:
                    self._note = f"The video size changed to {width}x{height}: the views so far were dropped"
                K, dist = provisional_camera(width, height, self._current)
                selector = self._selector = SampleSelector(self.spec, width, height, K, dist)
            look = selector.consider(corners, ids, now, len(markers), lambda: _jpeg(image, 92))
        preview = _preview(image, corners, markers, look.accepted)
        with self._lock:  # the look and its picture are published together
            self._look = look
            self._look_at = now
            self._preview = preview
            self._video = True
            self._sequence += 1
            if look.accepted:
                self._accepted += 1
                self._note = ""
        return look

    # -------------------------------------------------------- for the page and the solver

    def preview_jpeg(self) -> bytes | None:
        with self._lock:
            return self._preview

    def snapshot(self) -> tuple[tuple[int, int] | None, list[Sample]]:
        with self._lock:
            selector = self._selector
            if selector is None:
                return None, []
            return (selector.width, selector.height), list(selector.samples)

    def status(self) -> dict[str, Any]:
        with self._lock:
            selector, look = self._selector, self._look
            video = self._video
            out: dict[str, Any] = {
                "capturing": self.capturing,
                "created": self.created,
                "board": self.spec.describe(),
                "video": {"ok": video, "width": selector.width if selector else None, "height": selector.height if selector else None},
                "samples": len(selector.samples) if selector else 0,
                "target": TARGET_SAMPLES,
                "max": MAX_SAMPLES,
                "min_solve": MIN_SOLVE_SAMPLES,
                "min_corners": MIN_CORNERS,
                "sequence": self._sequence,
                "accepted_total": self._accepted,
                "coverage": None,
                "distances": selector.distance_counts() if selector else None,
                "sides": selector.side_counts() if selector else None,
                "last": None,
                "guidance": guidance(look, selector, video) if self.capturing else "Paused",
                "note": self._note or None,
            }
            if selector is not None:
                out["coverage"] = {
                    "cols": selector.cols, "rows": selector.rows, "cells": selector.cells.tolist(),
                    "fraction": round(selector.coverage, 3),
                }
            if look is not None:
                out["last"] = {
                    "corners": look.corners,
                    "steady": look.steady,
                    "accepted": look.accepted,
                    "reason": look.reason,
                    "points": look.points,
                    "age_s": round(time.monotonic() - self._look_at, 2),
                    "distance_m": round(look.pose.distance_m, 3) if look.pose else None,
                    "tilt_deg": round(look.pose.tilt_deg, 1) if look.pose else None,
                }
        return out


def _jpeg(image: np.ndarray, quality: int) -> bytes:
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return data.tobytes() if ok else b""


def _preview(image: np.ndarray, corners: np.ndarray, markers: list[np.ndarray], accepted: bool) -> bytes:
    """The frame, PREVIEW_HEIGHT tall, with the detected markers and corners drawn."""
    height, width = image.shape[:2]
    scale = PREVIEW_HEIGHT / height
    small = cv2.resize(image, (max(1, round(width * scale)), PREVIEW_HEIGHT), interpolation=cv2.INTER_AREA)
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    for marker in markers:
        outline = (np.reshape(marker, (-1, 2)) * scale).round().astype(np.int32)
        cv2.polylines(small, [outline], True, (240, 180, 90), 1, cv2.LINE_AA)
    color = (120, 220, 60) if accepted else (90, 210, 255)
    for x, y in corners * scale:
        cv2.circle(small, (int(round(x)), int(round(y))), 2, color, -1, cv2.LINE_AA)
    return _jpeg(small, 80)


# ---------------------------------------------------------------- solving


@dataclass
class Fit:
    model: str  # "pinhole" or "fisheye"
    K: np.ndarray
    dist: np.ndarray
    rms_px: float
    per_view_px: list[float]  # for the views used, in order


def _points(board: Any, samples: list[Sample]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    objects, images = [], []
    for sample in samples:
        obj, img = board.matchImagePoints(sample.corners.reshape(-1, 1, 2), sample.ids.reshape(-1, 1))
        objects.append(np.asarray(obj, dtype=np.float64).reshape(-1, 1, 3))
        images.append(np.asarray(img, dtype=np.float64).reshape(-1, 1, 2))
    return objects, images


def _rms(errors: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(np.sum(errors.reshape(-1, 2) ** 2, axis=1))))) if errors.size else math.nan


def fit_pinhole(objects: list[np.ndarray], images: list[np.ndarray], size: tuple[int, int]) -> Fit:
    _, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [o.astype(np.float32) for o in objects], [i.astype(np.float32) for i in images], size, None, None, flags=0, criteria=_CRITERIA
    )
    dist = np.ravel(dist)[:5]
    per_view, all_errors = [], []
    for obj, img, rvec, tvec in zip(objects, images, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        error = projected.reshape(-1, 2) - img.reshape(-1, 2)
        per_view.append(_rms(error))
        all_errors.append(error)
    return Fit("pinhole", K, dist, _rms(np.concatenate(all_errors)), per_view)


def fit_fisheye(objects: list[np.ndarray], images: list[np.ndarray], size: tuple[int, int], guess: np.ndarray | None = None) -> Fit:
    """``guess``: a starting K (the pinhole fit's). OpenCV's own start, a focal length of the
    image's long side / pi, is so short for a normal lens that points near the image corners
    fall outside the model and the first pose estimate fails."""
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    K0 = np.zeros((3, 3))
    if guess is not None:
        flags |= cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
        K0 = np.array(guess, dtype=np.float64)
    _, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objects, images, size, K0, np.zeros((4, 1)), flags=flags, criteria=_CRITERIA)
    D = np.ravel(D)[:4]
    per_view, all_errors = [], []
    for obj, img, rvec, tvec in zip(objects, images, rvecs, tvecs):
        projected, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, D)
        error = projected.reshape(-1, 2) - img.reshape(-1, 2)
        per_view.append(_rms(error))
        all_errors.append(error)
    if not np.all(np.isfinite(K)) or K[0, 0] <= 0:
        raise ValueError("the fisheye fit diverged")
    return Fit("fisheye", K, D, _rms(np.concatenate(all_errors)), per_view)


def held_out_error(fit: Fit, obj: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Reprojection errors (N x 2) of a view not used in ``fit``: its pose is found with the fitted
    intrinsics, then the board is projected back."""
    if fit.model == "pinhole":
        ok, rvec, tvec = cv2.solvePnP(obj, img, fit.K, fit.dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            raise ValueError("no pose")
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, fit.K, fit.dist, rvec, tvec)
        projected, _ = cv2.projectPoints(obj, rvec, tvec, fit.K, fit.dist)
    else:
        ok, rvec, tvec = cv2.fisheye.solvePnP(obj, img, fit.K, fit.dist)
        if not ok:
            raise ValueError("no pose")
        projected, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, fit.K, fit.dist)
    return projected.reshape(-1, 2) - img.reshape(-1, 2)


def cross_validate(fitter: Callable[..., Fit], objects: list[np.ndarray], images: list[np.ndarray], size: tuple[int, int]) -> float:
    """RMS error on views left out of the fit, over ``FOLDS`` folds."""
    count = len(objects)
    folds = min(FOLDS, count)
    errors = []
    for fold in range(folds):
        test = list(range(fold, count, folds))
        train = [index for index in range(count) if index % folds != fold]
        if len(train) < 3:
            continue
        fit = fitter([objects[i] for i in train], [images[i] for i in train], size)
        for index in test:
            errors.append(held_out_error(fit, objects[index], images[index]))
    if not errors:
        return math.nan
    return _rms(np.concatenate(errors))


def _fit_without_outliers(fitter: Callable[..., Fit], objects: list[np.ndarray], images: list[np.ndarray], size: tuple[int, int]) -> tuple[Fit, list[int]]:
    """Fit, drop the views whose error is above OUTLIER_FACTOR x the median, fit again.
    Returns the fit and the indices of the views it used."""
    used = list(range(len(objects)))
    fit = fitter(objects, images, size)
    median = float(np.median(fit.per_view_px))
    keep = [index for index, error in zip(used, fit.per_view_px) if error <= OUTLIER_FACTOR * median]
    if len(keep) < len(used) and len(keep) >= MIN_SOLVE_SAMPLES:
        used = keep
        fit = fitter([objects[i] for i in used], [images[i] for i in used], size)
    return fit, used


def field_of_view(model: str, K: np.ndarray, dist: np.ndarray, width: int, height: int) -> dict[str, float]:
    """Angles between the rays through opposite edges of the image (degrees), distortion included."""
    cx, cy = K[0, 2], K[1, 2]
    left, right, top, bottom = -0.5, width - 0.5, -0.5, height - 0.5
    points = np.array([[left, cy], [right, cy], [cx, top], [cx, bottom], [left, top], [right, bottom]], dtype=np.float64).reshape(-1, 1, 2)
    if model == "fisheye":
        normalised = cv2.fisheye.undistortPoints(points, K, dist, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12))
    else:
        normalised = cv2.undistortPoints(points, K, dist, None, None, None, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12))
    rays = np.concatenate([normalised.reshape(-1, 2), np.ones((len(points), 1))], axis=1)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)

    def angle(a: int, b: int) -> float:
        return round(math.degrees(math.acos(float(np.clip(rays[a] @ rays[b], -1.0, 1.0)))), 1)

    return {"horizontal": angle(0, 1), "vertical": angle(2, 3), "diagonal": angle(4, 5)}


@dataclass
class Solution:
    width: int
    height: int
    chosen: Fit
    fits: dict[str, Fit | None]
    held_out_px: dict[str, float | None]
    used: dict[str, list[int]]  # sample indices each model used
    views: int
    why: str
    board: dict[str, Any]
    preview: bytes | None = None
    created: str = field(default_factory=utc_now)

    @property
    def fov_deg(self) -> dict[str, float]:
        return field_of_view(self.chosen.model, self.chosen.K, self.chosen.dist, self.width, self.height)

    def calibration(self) -> dict[str, Any]:
        """calibration.json, exactly as PIPELINE.md section 3 describes it."""
        return {
            "version": 1,
            "model": self.chosen.model,
            "width": self.width,
            "height": self.height,
            "K": [[float(value) for value in row] for row in self.chosen.K],
            "dist": [float(value) for value in self.chosen.dist],
            "rms_px": round(self.chosen.rms_px, 4),
            "created": self.created,
            "device": DEVICE,
            "source": SOURCE,
            "board": dict(self.board),
        }

    def describe(self) -> dict[str, Any]:
        """For the page."""
        used = set(self.used[self.chosen.model])
        per_view: list[float | None] = [None] * self.views
        for index, error in zip(self.used[self.chosen.model], self.chosen.per_view_px):
            per_view[index] = round(error, 3)
        return {
            **_summary(self.calibration()),
            "views": self.views,
            "views_used": len(used),
            "dropped": [index for index in range(self.views) if index not in used],
            "per_view_px": per_view,
            "rms_by_model": {name: (round(fit.rms_px, 4) if fit else None) for name, fit in self.fits.items()},
            "held_out_px": {name: (round(value, 4) if value is not None and math.isfinite(value) else None) for name, value in self.held_out_px.items()},
            "why": self.why,
        }


def solve(spec: BoardSpec | dict[str, Any], samples: list[Sample], size: tuple[int, int]) -> Solution:
    """Fit both models and choose one (see the module docstring). Raises ValueError with a reason
    the person can read when it cannot."""
    if len(samples) < MIN_SOLVE_SAMPLES:
        raise ValueError(f"need at least {MIN_SOLVE_SAMPLES} views, have {len(samples)}")
    if isinstance(spec, BoardSpec):
        board, board_json = spec.board(), spec.to_json()
    else:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
        board = cv2.aruco.CharucoBoard((spec["squares_x"], spec["squares_y"]), spec["square_m"], spec["marker_m"], dictionary)
        board_json = dict(spec)
    objects, images = _points(board, samples)
    fits: dict[str, Fit | None] = {}
    used: dict[str, list[int]] = {}
    held: dict[str, float | None] = {}
    def fisheye_fitter(objs: list[np.ndarray], imgs: list[np.ndarray], image_size: tuple[int, int]) -> Fit:
        pinhole_fit = fits.get("pinhole")
        if pinhole_fit is not None:
            try:
                return fit_fisheye(objs, imgs, image_size, pinhole_fit.K)
            except (cv2.error, ValueError, np.linalg.LinAlgError):
                pass
        return fit_fisheye(objs, imgs, image_size)

    for name, fitter in (("pinhole", fit_pinhole), ("fisheye", fisheye_fitter)):
        try:
            fit, indices = _fit_without_outliers(fitter, objects, images, size)
            fits[name], used[name] = fit, indices
            held[name] = cross_validate(fitter, [objects[i] for i in indices], [images[i] for i in indices], size)
        except (cv2.error, ValueError, np.linalg.LinAlgError) as error:
            log.info("Calibration: the %s fit failed: %s", name, error)
            fits[name], used[name], held[name] = None, [], None
    pinhole, fisheye = fits.get("pinhole"), fits.get("fisheye")
    if pinhole is None and fisheye is None:
        raise ValueError("neither camera model could be fitted: capture more varied views")

    def usable(value: float | None) -> bool:
        return value is not None and math.isfinite(value)

    if pinhole is not None and (fisheye is None or not usable(held["fisheye"])):
        chosen, why = pinhole, "Pinhole: the fisheye model could not be fitted"
    elif pinhole is None or not usable(held["pinhole"]):
        chosen, why = fisheye, "Fisheye: the pinhole model could not be fitted"
    elif held["fisheye"] < FISHEYE_MARGIN * held["pinhole"]:
        chosen = fisheye
        why = f"Fisheye: its error on views left out of the fit ({held['fisheye']:.2f} px) is clearly lower than pinhole's ({held['pinhole']:.2f} px)"
    else:
        chosen = pinhole
        why = f"Pinhole: its error on views left out of the fit is {held['pinhole']:.2f} px (fisheye {held['fisheye']:.2f} px)"
    solution = Solution(size[0], size[1], chosen, fits, held, used, len(samples), why, board_json)
    try:
        solution.preview = undistorted_preview(solution, samples)
    except (cv2.error, ValueError):
        log.debug("Calibration: no undistorted preview", exc_info=True)
    return solution


def undistorted_preview(solution: Solution, samples: list[Sample]) -> bytes | None:
    """A kept view, as captured and undistorted, side by side: of the views that show most of the
    board, the one that reaches furthest into a corner of the image (where distortion shows)."""
    candidates = [(i, s) for i, s in enumerate(samples) if s.jpeg and i in solution.used[solution.chosen.model]]
    if not candidates:
        return None
    most = max(len(s.ids) for _, s in candidates)
    candidates = [(i, s) for i, s in candidates if len(s.ids) >= 0.8 * most] or candidates
    centre = np.array([solution.width / 2.0, solution.height / 2.0])
    index, sample = max(candidates, key=lambda item: float(np.linalg.norm(item[1].corners - centre, axis=1).max()))
    image = cv2.imdecode(np.frombuffer(sample.jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    fit, size = solution.chosen, (solution.width, solution.height)
    if fit.model == "fisheye":
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(fit.K, fit.dist, size, np.eye(3), balance=0.3)
        undistorted = cv2.fisheye.undistortImage(image, fit.K, fit.dist, Knew=new_K, new_size=size)
        points = cv2.fisheye.undistortPoints(sample.corners.reshape(-1, 1, 2).astype(np.float64), fit.K, fit.dist, P=new_K)
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(fit.K, fit.dist, size, 0.3)
        undistorted = cv2.undistort(image, fit.K, fit.dist, None, new_K)
        points = cv2.undistortPoints(sample.corners.reshape(-1, 1, 2).astype(np.float64), fit.K, fit.dist, None, None, new_K)
    scale = UNDISTORTED_HEIGHT / solution.height
    size_small = (round(solution.width * scale), UNDISTORTED_HEIGHT)
    small = [cv2.resize(picture, size_small, interpolation=cv2.INTER_AREA) for picture in (image, undistorted)]
    per_row = solution.board["squares_x"] - 1
    _draw_rows(small[0], sample.corners.reshape(-1, 2) * scale, sample.ids, per_row)  # curved where the lens bends them
    _draw_rows(small[1], points.reshape(-1, 2) * scale, sample.ids, per_row)  # straight when the calibration is right
    gap = np.full((UNDISTORTED_HEIGHT, 12, 3), 255, np.uint8)
    return _jpeg(np.hstack([small[0], gap, small[1]]), 85)


def _draw_rows(image: np.ndarray, points: np.ndarray, ids: np.ndarray, per_row: int) -> None:
    """Joins the corners of each board row and column."""
    grid: dict[tuple[int, int], np.ndarray] = {(int(i) // per_row, int(i) % per_row): p for i, p in zip(ids.tolist(), points)}
    lines = []
    for row in {r for r, _ in grid}:
        lines.append([grid[(row, col)] for col in range(per_row) if (row, col) in grid])
    for col in {c for _, c in grid}:
        lines.append([grid[(row, col)] for row in range(max(r for r, _ in grid) + 1) if (row, col) in grid])
    for line in lines:
        if len(line) >= 2:
            cv2.polylines(image, [np.array(line).round().astype(np.int32)], False, (255, 120, 110), 1, cv2.LINE_AA)


def _summary(calibration: dict[str, Any]) -> dict[str, Any]:
    """The facts the page shows about a calibration."""
    K = np.array(calibration["K"], dtype=np.float64)
    dist = np.array(calibration.get("dist") or [], dtype=np.float64)
    out = {key: calibration.get(key) for key in ("model", "width", "height", "K", "dist", "rms_px", "created", "device", "source", "board")}
    out["fx"], out["fy"], out["cx"], out["cy"] = (round(float(v), 2) for v in (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    try:
        out["fov_deg"] = field_of_view(calibration.get("model", "pinhole"), K, dist, int(calibration["width"]), int(calibration["height"]))
    except (cv2.error, ValueError, KeyError):
        out["fov_deg"] = None
    return out


# ---------------------------------------------------------------- files


class CalibrationStore:
    """``<state>/calibration/``: ``calibration_<UTC time>.json`` per save, ``current.json``
    (what new recordings copy) and ``screen.json`` (the measured screen scale)."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.folder = paths.calibration_dir(state_dir)
        self.current_path = paths.current_calibration(state_dir)
        self.screen_path = self.folder / "screen.json"

    def current(self) -> dict[str, Any] | None:
        data = _read_json(self.current_path)
        if not isinstance(data, dict) or not {"K", "width", "height"} <= set(data):
            return None
        return data

    def save(self, calibration: dict[str, Any]) -> Path:
        stamp = calibration["created"].replace("-", "").replace(":", "")
        path = self.folder / f"calibration_{stamp}.json"
        write_json_atomic(path, calibration)
        write_json_atomic(self.current_path, calibration)
        return path

    def screen(self) -> dict[str, Any] | None:
        data = _read_json(self.screen_path)
        return data if isinstance(data, dict) else None

    def save_screen(self, screen: dict[str, Any]) -> None:
        write_json_atomic(self.screen_path, screen)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- the service behind /api/calibration


def _number(body: dict[str, Any], key: str, low: float, high: float) -> float:
    value = body.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{key} must be a number") from None
    if not (math.isfinite(number) and low <= number <= high):
        raise ValueError(f"{key} must be from {low:g} to {high:g}")
    return number


def _pair(body: dict[str, Any], key: str) -> tuple[float, float]:
    value = body.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{key} must be [width, height] in CSS pixels")
    return _number({"w": value[0]}, "w", 100, 20000), _number({"h": value[1]}, "h", 100, 20000)


def _only(body: Any, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError("send a JSON object")
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise ValueError("unknown field " + ", ".join(repr(name) for name in unknown))
    return body


class CalibrationService:
    """State behind ``/api/calibration/...``. Methods return ``(HTTP status, JSON payload)``."""

    def __init__(self, receiver: "Receiver", state_dir: Path | None) -> None:
        self.receiver = receiver
        self.store = CalibrationStore(state_dir) if state_dir is not None else None
        self._lock = threading.Lock()
        self._session: CaptureSession | None = None
        self._solution: Solution | None = None
        self._solving = False
        self._closed = False

    # -------------------------------------------------------- reads

    def state(self) -> dict[str, Any]:
        current = self.store.current() if self.store else None
        with self._lock:
            session, solution, solving = self._session, self._solution, self._solving
        return {
            "card_mm": list(CARD_MM),
            "storage": str(self.store.folder) if self.store else None,
            "screen": self.store.screen() if self.store else None,
            "current": _summary(current) if current else None,
            "session": None if session is None else {
                **session.status(), "solving": solving, "solution": solution.describe() if solution else None,
            },
        }

    def board_png(self) -> bytes | None:
        with self._lock:
            session = self._session
        return session.spec.png() if session else None

    def preview_jpeg(self) -> bytes | None:
        with self._lock:
            session = self._session
        return session.preview_jpeg() if session else None

    def undistorted_jpeg(self) -> bytes | None:
        with self._lock:
            return self._solution.preview if self._solution else None

    # -------------------------------------------------------- actions

    def save_screen(self, body: Any) -> tuple[int, dict[str, Any]]:
        body = _only(body, {"css_px_per_mm", "device_pixel_ratio", "screen_css"})
        screen = {
            "css_px_per_mm": round(_number(body, "css_px_per_mm", 1.0, 40.0), 4),
            "device_pixel_ratio": _number(body, "device_pixel_ratio", 0.5, 8.0),
            "screen_css": list(_pair(body, "screen_css")) if "screen_css" in body else None,
            "card_mm": list(CARD_MM),
            "measured": utc_now(),
        }
        if self.store is None:
            return 409, {"ok": False, "message": "This receiver has no state folder to save to"}
        self.store.save_screen(screen)
        return 200, {"ok": True, "message": f"Screen scale saved: {screen['css_px_per_mm']:.2f} px per mm", "screen": screen}

    def start(self, body: Any) -> tuple[int, dict[str, Any]]:
        """Design the board for the area the page offers and capture. The same board as the
        running session resumes it (its views are kept); a different one starts over."""
        body = _only(body, {"css_px_per_mm", "device_pixel_ratio", "area_css"})
        spec = design_board(
            _number(body, "css_px_per_mm", 1.0, 40.0), _number(body, "device_pixel_ratio", 0.5, 8.0), _pair(body, "area_css")
        )
        with self._lock:
            if self._closed:
                return 409, {"ok": False, "message": "The receiver is shutting down"}
            old = self._session
            if old is not None and old.spec == spec:
                old.set_capturing(True)
                return 200, {"ok": True, "message": "Capturing", "kept": True, "board": spec.describe()}
            self._session = CaptureSession(self.receiver, spec, self.store.current() if self.store else None)
            self._solution = None
            self._session.start()
        if old is not None:
            old.close()
        squares = f"{spec.squares_x}x{spec.squares_y} board, {spec.square_m * 1000:.1f} mm squares"
        return 200, {"ok": True, "message": f"Capturing with a {squares}", "kept": False, "board": spec.describe()}

    def capture(self, body: Any) -> tuple[int, dict[str, Any]]:
        body = _only(body, {"active"})
        if not isinstance(body.get("active"), bool):
            raise ValueError("active must be true or false")
        with self._lock:
            session = self._session
        if session is None:
            return 409, {"ok": False, "message": "No calibration is running: start one first"}
        session.set_capturing(body["active"])
        return 200, {"ok": True, "message": "Capturing" if body["active"] else "Paused"}

    def solve(self) -> tuple[int, dict[str, Any]]:
        with self._lock:
            session = self._session
            if session is None:
                return 409, {"ok": False, "message": "No calibration is running: start one first"}
            if self._solving:
                return 409, {"ok": False, "message": "Already solving"}
            self._solving = True
        try:
            size, samples = session.snapshot()
            if size is None:
                return 409, {"ok": False, "message": "No video has been seen yet"}
            started = time.monotonic()
            try:
                solution = solve(session.spec, samples, size)
            except ValueError as error:
                return 409, {"ok": False, "message": f"Cannot solve: {error}"}
            log.info(
                "Calibration solved in %.1f s: %s, %d views, rms %.3f px", time.monotonic() - started,
                solution.chosen.model, solution.views, solution.chosen.rms_px,
            )
            with self._lock:
                if self._session is session:
                    self._solution = solution
            return 200, {"ok": True, "message": f"Solved: {solution.chosen.rms_px:.2f} px RMS", "solution": solution.describe()}
        finally:
            with self._lock:
                self._solving = False

    def save(self) -> tuple[int, dict[str, Any]]:
        with self._lock:
            solution = self._solution
        if solution is None:
            return 409, {"ok": False, "message": "Nothing to save: solve first"}
        if self.store is None:
            return 409, {"ok": False, "message": "This receiver has no state folder to save to"}
        calibration = solution.calibration()
        path = self.store.save(calibration)
        log.info("Calibration saved to %s", path)
        return 200, {"ok": True, "message": "Calibration saved: new recordings use it", "current": _summary(calibration), "file": str(path)}

    def reset(self) -> tuple[int, dict[str, Any]]:
        with self._lock:
            session, self._session, self._solution = self._session, None, None
        if session is not None:
            session.close()
        return 200, {"ok": True, "message": "Calibration views discarded"}

    def close(self) -> None:
        with self._lock:
            self._closed = True
            session, self._session = self._session, None
        if session is not None:
            session.close()
