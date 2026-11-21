"""Output to a virtual camera, so Zoom and other apps can use the glasses video.

The glasses video is portrait (720x1280), but the OBS Virtual Camera on macOS only takes
landscape frames (1920x1080 or 1280x720): any other size is dropped without an error,
the native layer only prints "Pixel buffer size mismatch" on standard error. So the
picture is always composed onto a landscape canvas of exactly the configured size:

* ``fit`` (default): the whole frame, scaled to the canvas height and centred; the sides
  show a blurred, darkened, enlarged copy of the same frame.
* ``fill``: a landscape band of the frame scaled to cover the canvas; ``position`` moves
  the band from the top (0) to the bottom (1) of the frame.

One thread (``gd-vcam``) takes the newest decoded frame, composes it and sends it at a
steady rate (``fps``), repeating the last picture when no new frame has arrived. It
never waits for the decoder: like the web page's ``JpegFeed`` it only ever reads the
latest frame. Without video it sends a calm placeholder card instead of a frozen frame.

Requires ``pip install pyvirtualcam`` (imported only when the camera starts). On macOS
it also needs OBS (https://obsproject.com): macOS only lets a signed camera extension
provide a system camera, and OBS ships one. On Linux pyvirtualcam writes to a
v4l2loopback device instead (package ``v4l2loopback-dkms``).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from .viewer import hands_for

log = logging.getLogger(__name__)

CANVAS_SIZES = ((1920, 1080), (1280, 720))  # the sizes the OBS Virtual Camera accepts
DEFAULT_SIZE = CANVAS_SIZES[0]
DEFAULT_FPS = 30
MODES = ("fit", "fill")
STALE_AFTER_S = 3.0  # no new frame for this long: the video stopped, show the card
BACKDROP_DIVISOR = 12  # fit mode blurs a copy this many times smaller than the canvas
BACKDROP_DIM = 0.5  # brightness of the blurred sides
SELF_CHECK_S = 2.0  # how long the start-up check watches for OBS rejecting the frames
OBS_REJECTION = b"Pixel buffer size mismatch"
PREVIEW_WIDTH = 480  # the web page's thumbnail
PREVIEW_QUALITY = 75

_CARD_BG = (11, 9, 9)  # BGR of the page's dark background, #09090b
_CARD_SURFACE = (20, 17, 17)
_CARD_LINE = (40, 36, 36)
_CARD_TEXT = (241, 236, 236)
_CARD_MUTED = (174, 163, 163)
_CARD_FAINT = (149, 138, 138)
_CARD_ACCENT = (255, 137, 129)  # #8189ff


class Unavailable(Exception):
    """The virtual camera cannot be opened. ``hint`` says what to do, in words."""

    def __init__(self, message: str, hint: str) -> None:
        super().__init__(message)
        self.hint = hint


def parse_size(text: str) -> tuple[int, int]:
    """ "1920x1080" -> (1920, 1080); only the sizes in ``CANVAS_SIZES``."""
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise ValueError(f"not a size: {text!r} (use WIDTHxHEIGHT)") from None
    if (width, height) not in CANVAS_SIZES:
        raise ValueError(f"unsupported size {width}x{height}: use " + " or ".join(f"{w}x{h}" for w, h in CANVAS_SIZES))
    return width, height


def setup_hint(platform: str | None = None) -> str:
    platform = platform or sys.platform
    if platform == "darwin":
        return (
            "Install OBS Studio (obsproject.com). Open it, click Start Virtual Camera once and allow the camera "
            "extension when macOS asks (System Settings > General > Login Items & Extensions > Camera Extensions), "
            "then click Stop Virtual Camera and quit OBS. Then start the virtual camera here again."
        )
    if platform.startswith("linux"):
        return (
            "Load v4l2loopback: sudo apt install v4l2loopback-dkms, then "
            "sudo modprobe v4l2loopback devices=1 exclusive_caps=1 card_label=GlassesData. "
            "Then start the virtual camera here again."
        )
    return "Install OBS Studio (obsproject.com) and start its virtual camera once. Then start the virtual camera here again."


INSTALL_HINT = (
    "Install it into the receiver's environment: ~/.venvs/glassesdata-receiver/bin/pip install pyvirtualcam, "
    "then restart the receiver."
)


def open_pyvirtualcam(width: int, height: int, fps: int) -> Any:
    """The real camera: pyvirtualcam, imported here and only here. Raises ``Unavailable``."""
    try:
        import pyvirtualcam
    except ImportError:
        raise Unavailable("pyvirtualcam is not installed", INSTALL_HINT) from None
    try:
        return pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
    except Exception as error:
        # pyvirtualcam lists one line per backend it tried; keep them, without the quotes.
        detail = " ".join(str(error).split()).replace("'", "")
        raise Unavailable(f"the virtual camera device could not be opened ({detail})", setup_hint()) from error


# ---------------------------------------------------------------- composition


@dataclass(frozen=True)
class Layout:
    """Where the picture goes. All rectangles are (x, y, width, height) in pixels.

    ``crop``: the part of the source frame that is shown; ``video``: where it lands on the
    canvas, scaled to that size. ``backdrop``: the canvas has room around the video (fit
    mode, filled with the blurred copy).
    """

    canvas: tuple[int, int]
    video: tuple[int, int, int, int]
    crop: tuple[int, int, int, int]
    backdrop: bool


def layout(source: tuple[int, int], canvas: tuple[int, int], mode: str = "fit", position: float = 0.5) -> Layout:
    """``source`` and ``canvas`` are (width, height)."""
    source_w, source_h = source
    canvas_w, canvas_h = canvas
    if mode == "fill":
        crop = _cover_crop(source_w, source_h, canvas_w / canvas_h, position)
        return Layout(canvas, (0, 0, canvas_w, canvas_h), crop, False)
    scale = min(canvas_w / source_w, canvas_h / source_h)
    video_w = min(canvas_w, max(1, round(source_w * scale)))
    video_h = min(canvas_h, max(1, round(source_h * scale)))
    video = ((canvas_w - video_w) // 2, (canvas_h - video_h) // 2, video_w, video_h)
    return Layout(canvas, video, (0, 0, source_w, source_h), (video_w, video_h) != (canvas_w, canvas_h))


def _cover_crop(source_w: int, source_h: int, aspect: float, position: float) -> tuple[int, int, int, int]:
    """The largest band of the source with this aspect (width / height). A taller source
    gives a horizontal band placed by ``position`` (0 top, 1 bottom); a wider one is centred."""
    if source_w / source_h < aspect:
        crop_h = min(source_h, max(1, round(source_w / aspect)))
        top = round(min(max(position, 0.0), 1.0) * (source_h - crop_h))
        return 0, top, source_w, crop_h
    crop_w = min(source_w, max(1, round(source_h * aspect)))
    return (source_w - crop_w) // 2, 0, crop_w, source_h


def _backdrop(image: np.ndarray, canvas_w: int, canvas_h: int) -> np.ndarray:
    """The blurred, darkened, enlarged copy behind a fitted picture. Blurred small, then scaled up."""
    height, width = image.shape[:2]
    x, y, w, h = _cover_crop(width, height, canvas_w / canvas_h, 0.5)
    small_size = (max(4, canvas_w // BACKDROP_DIVISOR), max(4, canvas_h // BACKDROP_DIVISOR))
    # Two cheap steps instead of one slow INTER_AREA: linear to twice the size, then an exact 2:1 average.
    double = cv2.resize(image[y : y + h, x : x + w], (small_size[0] * 2, small_size[1] * 2), interpolation=cv2.INTER_LINEAR)
    small = cv2.resize(double, small_size, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=max(1.0, small_size[0] / 48))
    small = cv2.convertScaleAbs(small, alpha=BACKDROP_DIM)
    return cv2.resize(small, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)


@dataclass(frozen=True)
class _MappedHand:
    handedness: str
    score: float
    landmarks: list[tuple[float, float, float]]


def map_hands(hands: list[Any], source: tuple[int, int], crop: tuple[int, int, int, int]) -> list[_MappedHand]:
    """Landmarks normalised to the source frame -> normalised to the cropped picture.

    A hand entirely outside the crop is left out (its "L"/"R" tag would otherwise be
    pinned to the picture's edge); one partly inside is kept and clipped by the drawing."""
    source_w, source_h = source
    x0, y0, crop_w, crop_h = crop
    mapped = []
    for hand in hands:
        points = [((x * source_w - x0) / crop_w, (y * source_h - y0) / crop_h, z) for x, y, z, *_ in hand.landmarks]
        if any(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y, _ in points):
            mapped.append(_MappedHand(hand.handedness, hand.score, points))
    return mapped


def compose(image: np.ndarray, canvas: tuple[int, int], mode: str = "fit", position: float = 0.5, hands: list[Any] | None = None) -> np.ndarray:
    """A new BGR canvas of exactly ``canvas`` (width, height) with ``image`` placed by ``mode``.

    ``hands`` (from the hand tracker, normalised to ``image``) are drawn on the video part
    only, never on the blurred sides.
    """
    canvas_w, canvas_h = canvas
    height, width = image.shape[:2]
    plan = layout((width, height), canvas, mode, position)
    cx, cy, cw, ch = plan.crop
    vx, vy, vw, vh = plan.video
    # INTER_AREA only for strong reductions: at other ratios it costs ~8 ms a frame for no visible gain.
    strong = vw < cw * 0.5
    region = cv2.resize(image[cy : cy + ch, cx : cx + cw], (vw, vh), interpolation=cv2.INTER_AREA if strong else cv2.INTER_LINEAR)
    if hands:
        from .hands import draw_hands  # imports MediaPipe; only needed when there are hands

        draw_hands(region, map_hands(hands, (width, height), plan.crop))
    if not plan.backdrop:
        return region if (vw, vh) == (canvas_w, canvas_h) else _place(np.zeros((canvas_h, canvas_w, 3), np.uint8), region, vx, vy)
    return _place(_backdrop(image, canvas_w, canvas_h), region, vx, vy)


def _place(canvas: np.ndarray, region: np.ndarray, x: int, y: int) -> np.ndarray:
    canvas[y : y + region.shape[0], x : x + region.shape[1]] = region
    return canvas


def placeholder_card(canvas: tuple[int, int], detail: str = "") -> np.ndarray:
    """The picture sent without video: the product name, "No video" and why, on the page's dark background."""
    width, height = canvas
    unit = height / 1080
    image = np.full((height, width, 3), _CARD_BG, np.uint8)
    card_w, card_h = round(560 * unit), round(300 * unit)
    left, top = (width - card_w) // 2, (height - card_h) // 2
    radius = round(28 * unit)
    _rounded_rect(image, (left, top, card_w, card_h), radius, _CARD_SURFACE, _CARD_LINE, max(1, round(1.5 * unit)))
    # the brand mark: the page's accent square with a pair of glasses
    mark = round(64 * unit)
    mark_x, mark_y = (width - mark) // 2, top + round(44 * unit)
    _rounded_rect(image, (mark_x, mark_y, mark, mark), round(16 * unit), _CARD_ACCENT)
    lens = max(2, round(9 * unit))
    centre_y = mark_y + mark // 2 + round(4 * unit)
    for dx in (-1, 1):
        cv2.circle(image, (width // 2 + dx * round(13 * unit), centre_y), lens, _CARD_BG, max(2, round(3 * unit)), cv2.LINE_AA)
    cv2.line(image, (width // 2 - round(4 * unit), centre_y - round(2 * unit)), (width // 2 + round(4 * unit), centre_y - round(2 * unit)),
             _CARD_BG, max(2, round(3 * unit)), cv2.LINE_AA)
    baseline = mark_y + mark + round(66 * unit)
    _centred_text(image, "GlassesData", baseline, 1.45 * unit, _CARD_TEXT, max(1, round(2 * unit)))
    _centred_text(image, "No video", baseline + round(52 * unit), 1.0 * unit, _CARD_MUTED, max(1, round(1.6 * unit)))
    if detail:
        _centred_text(image, detail, baseline + round(92 * unit), 0.72 * unit, _CARD_FAINT, max(1, round(1.2 * unit)))
    return image


def _centred_text(image: np.ndarray, text: str, baseline: int, scale: float, color: tuple[int, int, int], weight: int) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    (text_w, _), _ = cv2.getTextSize(text, font, scale, weight)
    cv2.putText(image, text, ((image.shape[1] - text_w) // 2, baseline), font, scale, color, weight, cv2.LINE_AA)


def _rounded_rect(image: np.ndarray, rect: tuple[int, int, int, int], radius: int, fill: tuple[int, int, int],
                  border: tuple[int, int, int] | None = None, border_width: int = 1) -> None:
    x, y, w, h = rect
    radius = max(1, min(radius, w // 2, h // 2))

    def shape(color: tuple[int, int, int], inset: int) -> None:
        r = max(1, radius - inset)
        x0, y0, x1, y1 = x + inset, y + inset, x + w - 1 - inset, y + h - 1 - inset
        cv2.rectangle(image, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(image, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(image, (cx, cy), r, color, -1, cv2.LINE_AA)

    if border is not None:
        shape(border, 0)
        shape(fill, border_width)
    else:
        shape(fill, 0)


# ---------------------------------------------------------------- the start-up check


class StderrWatch:
    """Copies this process's standard error (fd 2) through a pipe for a while and notes
    whether ``needle`` went by. Everything is passed on unchanged.

    OBS's camera extension drops a frame of the wrong size without an error; its native
    code only prints "Pixel buffer size mismatch" to fd 2. Watching fd 2 for the first
    seconds is the only way the sender can notice. (A pipe, not a file: other threads
    keep logging meanwhile, and their output must reach the terminal in order.)
    """

    def __init__(self, needle: bytes) -> None:
        self.needle = needle
        self.found = False
        self._saved: int | None = None
        self._read_fd: int | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        try:
            sys.stderr.flush()
            saved = os.dup(2)
        except (OSError, ValueError):
            return False
        try:
            read_fd, write_fd = os.pipe()
        except OSError:
            os.close(saved)
            return False
        try:
            os.dup2(write_fd, 2)
        except OSError:
            for fd in (read_fd, write_fd, saved):
                os.close(fd)
            return False
        os.close(write_fd)  # fd 2 is now the only writer: restoring it ends the pipe
        self._saved, self._read_fd = saved, read_fd
        self._thread = threading.Thread(target=self._pump, name="gd-vcam-check", daemon=True)
        try:
            self._thread.start()
        except RuntimeError:  # nobody would drain the pipe: give fd 2 back at once
            self._thread = None
            self.stop()
            return False
        return True

    def _pump(self) -> None:
        tail = b""
        keep = max(0, len(self.needle) - 1)
        while True:
            try:
                data = os.read(self._read_fd, 65536)
            except OSError:
                data = b""
            if not data:
                return
            view = memoryview(data)
            while view:
                try:
                    view = view[os.write(self._saved, view):]
                except OSError:
                    break  # the terminal went away: keep draining so no writer blocks
            if self.needle in tail + data:
                self.found = True
            tail = (tail + data)[-keep:] if keep else b""

    def stop(self) -> None:
        if self._saved is None:
            return
        try:
            sys.stderr.flush()
        except (OSError, ValueError):
            pass
        os.dup2(self._saved, 2)
        if self._thread is not None:
            self._thread.join(2.0)
            if self._thread.is_alive():
                # The pump is stuck writing to a stalled terminal: leave its two fds open
                # (closing them could hand their numbers to another file it then writes to).
                self._saved = self._read_fd = None
                return
        for fd in (self._read_fd, self._saved):
            try:
                os.close(fd)
            except OSError:
                pass
        self._saved = self._read_fd = None


# ---------------------------------------------------------------- the sender


class VirtualCamera:
    """The virtual camera output, started and stopped while the receiver runs.

    ``receiver`` needs ``busy``, ``latest_frame()``, ``latest_hands()`` and ``hands_active``.
    ``open_camera(width, height, fps)`` returns an object with ``send(bgr)``, ``close()``,
    ``device`` and ``backend`` (default: pyvirtualcam) or raises ``Unavailable``.

    Thread safety: ``start``/``stop`` are serialised; settings and status are guarded by
    ``_lock``; only the ``gd-vcam`` thread touches the camera object while it runs.
    """

    def __init__(
        self,
        receiver: Any,
        size: tuple[int, int] = DEFAULT_SIZE,
        fps: int = DEFAULT_FPS,
        mode: str = "fit",
        position: float = 0.5,
        overlay: bool = False,
        open_camera: Callable[[int, int, int], Any] | None = None,
        watch_stderr: bool | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if tuple(size) not in CANVAS_SIZES:
            raise ValueError(f"unsupported canvas size {size}")
        if not 1 <= fps <= 60:
            raise ValueError("fps must be between 1 and 60")
        self._receiver = receiver
        self.size: tuple[int, int] = (int(size[0]), int(size[1]))
        self.fps = int(fps)
        self._open_camera = open_camera or open_pyvirtualcam
        self._uses_pyvirtualcam = open_camera is None
        self._watch_stderr = watch_stderr
        self._clock = clock
        self._lock = threading.Lock()
        self._control = threading.Lock()
        self._mode = "fit"
        self._position = 0.5
        self._overlay = False
        self._settings_version = 0
        self.set(mode=mode, position=position, overlay=overlay)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._closed = False  # set by close(): the receiver is shutting down, no more starts
        self._device: str | None = None
        self._backend: str | None = None
        self._frames_sent = 0
        self._last_error: str | None = None
        self._hint: str | None = None
        self._check: str | None = None
        self._showing: str | None = None  # "video" or "card" while running
        self._picture: np.ndarray | None = None  # the canvas being sent (never modified once published)
        self._picture_version = 0
        self._preview: tuple[int, bytes] | None = None
        self._preview_lock = threading.Lock()
        self._cards: dict[tuple[tuple[int, int], str], np.ndarray] = {}
        self._available = self._check_available()

    # ------------------------------------------------------------ control (any thread)

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> tuple[bool, str]:
        """Open the device and start sending. Returns (ok, message for people)."""
        with self._control:
            if self._closed:
                return False, "The receiver is shutting down"
            if self.running:
                return True, f"The virtual camera is already on ({self._device})"
            self._available = self._check_available()
            width, height = self.size
            try:
                camera = self._open_camera(width, height, self.fps)
            except Unavailable as error:
                return self._failed(str(error), error.hint)
            except Exception as error:  # a fake or a future backend: still a reason in words
                return self._failed(f"the virtual camera could not be opened ({error})", setup_hint())
            device = str(getattr(camera, "device", "") or "virtual camera")
            backend = str(getattr(camera, "backend", "") or "")
            with self._lock:
                self._device, self._backend = device, backend
                self._frames_sent = 0
                self._last_error = self._hint = None
                self._check = None
                self._available = True
            self._stop = threading.Event()
            watch = None
            try:
                watch = self._make_watch(backend)
                self._thread = threading.Thread(target=self._run, args=(camera, self._stop, watch), name="gd-vcam", daemon=True)
                self._thread.start()
            except Exception as error:  # the device is open: never leave it so
                self._thread = None
                if watch is not None:
                    watch.stop()
                try:
                    camera.close()
                except Exception:
                    log.debug("closing the virtual camera failed", exc_info=True)
                return self._failed(f"the virtual camera could not start ({error})", setup_hint())
            mode, position = self._mode, self._position
            detail = f"{mode}" + (f", band at {position:.2f}" if mode == "fill" else "")
            log.info(
                "Virtual camera self-check: opened %r (backend %s); sending %dx%d at %d fps, %s. "
                "The sender cannot confirm that an app receives the frames: open the camera in an app to check.",
                device, backend or "unknown", width, height, self.fps, detail,
            )
            return True, f"Virtual camera on: {device}"

    def stop(self) -> tuple[bool, str]:
        with self._control:
            thread = self._thread
            if thread is None or not thread.is_alive():
                return True, "The virtual camera is already off"
            self._stop.set()
            thread.join(5.0)
            if thread.is_alive():  # pragma: no cover - a backend stuck in send()
                log.warning("The virtual camera thread did not stop within 5 s")
                return False, "The virtual camera did not stop within 5 s: its device is not answering"
            log.info("Virtual camera off")
            return True, "Virtual camera off"

    def close(self) -> None:
        """At shutdown: stop, and refuse any later start (a page may still ask for one)."""
        with self._control:
            self._closed = True
        self.stop()

    def set(self, mode: str | None = None, position: float | None = None, overlay: bool | None = None) -> None:
        """Change settings; they apply to the next frame sent. Raises ValueError for bad values."""
        if mode is not None and mode not in MODES:
            raise ValueError("mode must be " + " or ".join(MODES))
        if position is not None:
            if isinstance(position, bool) or not isinstance(position, (int, float)) or not 0.0 <= float(position) <= 1.0:
                raise ValueError("position must be a number from 0 (top) to 1 (bottom)")
        if overlay is not None and not isinstance(overlay, bool):
            raise ValueError("overlay must be true or false")
        with self._lock:
            if mode is not None:
                self._mode = mode
            if position is not None:
                self._position = float(position)
            if overlay is not None:
                self._overlay = overlay
            self._settings_version += 1

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self.running
            return {
                "available": self._available,
                "running": running,
                "device": self._device,
                "backend": self._backend,
                "width": self.size[0],
                "height": self.size[1],
                "fps": self.fps,
                "mode": self._mode,
                "position": round(self._position, 3),
                "overlay": self._overlay,
                "frames_sent": self._frames_sent,
                "showing": self._showing if running else None,
                "check": self._check,
                "last_error": self._last_error,
                "hint": self._hint if not running else None,
            }

    def preview_jpeg(self) -> bytes | None:
        """The picture being sent, scaled to ``PREVIEW_WIDTH``, as JPEG; None while off.

        Encoded in the caller's thread, at most once per new picture, so viewers of the
        page never slow the sender."""
        with self._lock:
            picture, version = (self._picture, self._picture_version) if self.running else (None, 0)
        if picture is None:
            return None
        with self._preview_lock:
            if self._preview is not None and self._preview[0] == version:
                return self._preview[1]
            height, width = picture.shape[:2]
            small = cv2.resize(picture, (PREVIEW_WIDTH, round(height * PREVIEW_WIDTH / width)), interpolation=cv2.INTER_AREA)
            ok, buffer = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_QUALITY])
            if not ok:
                return None
            self._preview = (version, buffer.tobytes())
            return self._preview[1]

    # ------------------------------------------------------------ the gd-vcam thread

    def _run(self, camera: Any, stop: threading.Event, watch: StderrWatch | None) -> None:
        interval = 1.0 / self.fps
        next_at = self._clock()
        check_until = next_at + SELF_CHECK_S
        state = _PictureState()
        try:
            while not stop.is_set():
                picture, showing = self._picture_for(state)
                camera.send(picture)
                with self._lock:
                    self._frames_sent += 1
                    self._showing = showing
                    if picture is not self._picture:
                        self._picture = picture
                        self._picture_version += 1
                if watch is not None and self._clock() >= check_until:
                    self._finish_check(watch)
                    watch = None
                next_at += interval
                delay = next_at - self._clock()
                if delay < -interval:  # fell behind (a slow send): carry on from now, never burst
                    next_at = self._clock()
                    delay = 0.0
                if delay > 0 and stop.wait(delay):
                    break
        except Exception as error:
            log.exception("The virtual camera stopped")
            with self._lock:
                self._last_error = f"sending failed: {error}"
                self._hint = setup_hint() if self._uses_pyvirtualcam else None
        finally:
            if watch is not None:
                self._finish_check(watch)
            try:
                camera.close()
            except Exception:
                log.debug("closing the virtual camera failed", exc_info=True)
            with self._lock:
                self._showing = None
                self._picture = None

    def _picture_for(self, state: "_PictureState") -> tuple[np.ndarray, str]:
        """The canvas to send now, recomposed only when the frame, its hands or a setting changed."""
        receiver = self._receiver
        now = self._clock()
        frame = receiver.latest_frame() if receiver.busy else None
        if frame is not None and frame.index != state.frame_index:
            state.frame_index, state.frame_seen = frame.index, now
        with self._lock:
            mode, position, overlay, version = self._mode, self._position, self._overlay, self._settings_version
        if frame is None or now - state.frame_seen > STALE_AFTER_S:
            if not receiver.busy:
                detail = "No iPhone connected"
            elif frame is None:
                detail = "Waiting for the glasses video"
            else:
                detail = "The video stopped"
            key: tuple[Any, ...] = ("card", detail)
            if key != state.key:
                state.key, state.picture = key, self._card(detail)
            return state.picture, "card"
        hands: list[Any] = []
        hands_key = None
        if overlay and receiver.hands_active:
            result = receiver.latest_hands()
            hands = hands_for(frame.index, result)
            hands_key = result.frame_index if hands else None
        key = ("video", frame.index, hands_key, version)
        if key != state.key:
            state.key, state.picture = key, compose(frame.image, self.size, mode, position, hands)
        return state.picture, "video"

    def _card(self, detail: str) -> np.ndarray:
        cache_key = (self.size, detail)
        if cache_key not in self._cards:
            self._cards[cache_key] = placeholder_card(self.size, detail)
        return self._cards[cache_key]

    # ------------------------------------------------------------ helpers

    def _check_available(self) -> bool:
        if not self._uses_pyvirtualcam:
            return True
        try:
            return importlib.util.find_spec("pyvirtualcam") is not None
        except (ImportError, ValueError):
            return False

    def _failed(self, message: str, hint: str) -> tuple[bool, str]:
        with self._lock:
            self._last_error = message
            self._hint = hint
            self._device = self._backend = None  # not the device of an earlier run
            self._frames_sent = 0
        log.error("Virtual camera unavailable: %s. %s", message, hint)
        return False, f"Virtual camera unavailable: {message}"

    def _make_watch(self, backend: str) -> StderrWatch | None:
        wanted = self._watch_stderr if self._watch_stderr is not None else (sys.platform == "darwin" and backend == "obs")
        if not wanted:
            return None
        watch = StderrWatch(OBS_REJECTION)
        return watch if watch.start() else None

    def _finish_check(self, watch: StderrWatch) -> None:
        watch.stop()
        width, height = self.size
        if watch.found:
            text = (f"OBS rejected the {width}x{height} frames (it printed \"Pixel buffer size mismatch\"), "
                    "so apps receive nothing. Try the other size (--virtual-camera-size).")
            log.error("Virtual camera self-check: %s", text)
            with self._lock:
                self._check = text
                self._last_error = text
        else:
            text = f"OBS took {width}x{height} frames without complaint for {SELF_CHECK_S:g} s"
            log.info("Virtual camera self-check: %s (delivery to an app cannot be confirmed from here)", text)
            with self._lock:
                self._check = text


class _PictureState:
    """What the sender thread last composed (owned by that thread)."""

    def __init__(self) -> None:
        self.key: tuple[Any, ...] | None = None
        self.picture: np.ndarray | None = None
        self.frame_index: int | None = None
        self.frame_seen = 0.0
