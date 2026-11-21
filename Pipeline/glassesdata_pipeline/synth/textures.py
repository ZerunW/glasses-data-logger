"""Procedural textures with many corners and edges (for ORB/SIFT), sampled with mipmaps."""

from __future__ import annotations

import string

import cv2
import numpy as np

_FONTS = (
    cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_PLAIN,
)


def _value_noise(rng: np.random.Generator, h: int, w: int, cells: int) -> np.ndarray:
    small = rng.normal(size=(max(2, h // cells), max(2, w // cells))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _color(rng: np.random.Generator, base: np.ndarray, spread: float) -> tuple[int, int, int]:
    c = np.clip(base + rng.normal(scale=spread, size=3), 0, 255)
    return tuple(int(v) for v in c)


def make_texture(
    rng: np.random.Generator,
    height: int,
    width: int,
    base_bgr: tuple[int, int, int],
    detail: float = 1.0,
    text: bool = True,
    style: str = "muted",
) -> np.ndarray:
    """A ``(height, width, 3)`` uint8 BGR texture around ``base_bgr``: multi-scale noise plus
    rectangles, checker patches, circles, line segments and random glyphs (many corners for
    ORB/SIFT). ``detail`` scales the number of shapes (per 256x256 texels). ``style``:
    ``"muted"`` (scenery: shapes close to the base colour), ``"grain"`` (wood-like streaks),
    ``"busy"`` (objects: saturated, high contrast)."""
    base = np.array(base_bgr, dtype=np.float32)
    busy = style == "busy"
    img = np.empty((height, width, 3), np.float32)
    img[:] = base
    lum = np.zeros((height, width), np.float32)
    for cells, amp in ((max(height, width) // 6, 22.0), (24, 12.0), (6, 7.0), (2, 4.0)):
        lum += amp * _value_noise(rng, height, width, max(1, cells))
    if style == "grain":
        streaks = rng.normal(size=(height, max(2, width // 48))).astype(np.float32)
        lum += 14.0 * cv2.resize(streaks, (width, height), interpolation=cv2.INTER_CUBIC)
    img += lum[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    area = height * width / (256.0 * 256.0)
    count = lambda k: max(1, int(round(k * area * detail)))  # noqa: E731
    spread = 75.0 if busy else 35.0
    max_frac = 3 if busy else 6

    for _ in range(count(8 if busy else 5)):  # filled rectangles, some outlined
        w = int(rng.integers(4, max(5, width // max_frac)))
        h = int(rng.integers(4, max(5, height // max_frac)))
        x0, y0 = int(rng.integers(0, max(1, width - w))), int(rng.integers(0, max(1, height - h)))
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), _color(rng, base, spread), -1)
        alpha = float(rng.uniform(0.5, 0.95) if busy else rng.uniform(0.25, 0.6))
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        if rng.random() < 0.6:
            cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), _color(rng, base * 0.35, 25), int(rng.integers(1, 3)))
    for _ in range(count(4 if busy else 2)):  # checker patches
        cell = int(rng.integers(3, 9))
        nx, ny = int(rng.integers(3, 7)), int(rng.integers(3, 7))
        x0 = int(rng.integers(0, max(1, width - nx * cell)))
        y0 = int(rng.integers(0, max(1, height - ny * cell)))
        dark = _color(rng, base * 0.3, 15)
        light = _color(rng, np.full(3, 225.0) if busy else np.minimum(base * 1.5 + 30, 240), 15)
        for i in range(ny):
            for j in range(nx):
                cv2.rectangle(img, (x0 + j * cell, y0 + i * cell), (x0 + (j + 1) * cell - 1, y0 + (i + 1) * cell - 1),
                              dark if (i + j) % 2 else light, -1)
    for _ in range(count(3 if busy else 2)):  # circles and rings
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        radius = int(rng.integers(3, max(4, min(width, height) // (8 if busy else 14))))
        cv2.circle(img, center, radius, _color(rng, base, spread), int(rng.choice([-1, 1, 2])), cv2.LINE_AA)
    for _ in range(count(5)):  # line segments
        p0 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        p1 = (int(np.clip(p0[0] + rng.normal(scale=width / 8), 0, width - 1)),
              int(np.clip(p0[1] + rng.normal(scale=height / 8), 0, height - 1)))
        cv2.line(img, p0, p1, _color(rng, base * 0.45, 25), int(rng.integers(1, 3)), cv2.LINE_AA)
    if text:
        letters = string.ascii_uppercase + string.digits
        for _ in range(count(8 if busy else 6)):
            word = "".join(rng.choice(list(letters), size=int(rng.integers(2, 6))))
            scale = float(rng.uniform(0.3, 0.9))
            org = (int(rng.integers(0, max(1, width - 40))), int(rng.integers(12, max(13, height))))
            cv2.putText(img, word, org, int(rng.choice(_FONTS)), scale, _color(rng, base * 0.3, 30),
                        int(rng.integers(1, 3)), cv2.LINE_AA)
    return img


class Texture:
    """A texture covering one face: ``texel_u`` x ``texel_v`` metres per texel at level 0;
    bilinear sampling with a mip level chosen from the pixel footprint (clamped at the borders)."""

    def __init__(self, image: np.ndarray, extent_u: float, extent_v: float) -> None:
        self.levels: list[np.ndarray] = [image.astype(np.float32)]
        while min(self.levels[-1].shape[:2]) > 4:
            self.levels.append(cv2.pyrDown(self.levels[-1]))
        h, w = image.shape[:2]
        self.texel_u = extent_u / w
        self.texel_v = extent_v / h
        self.texel = min(self.texel_u, self.texel_v)

    def sample(self, s: np.ndarray, t: np.ndarray, footprint: np.ndarray) -> np.ndarray:
        """Colours ``(N, 3)`` float32 at face coordinates ``s`` (along u) and ``t`` (along v), metres."""
        out = np.empty((len(s), 3), np.float32)
        level = np.clip(np.rint(np.log2(np.maximum(footprint / self.texel, 1e-6))), 0, len(self.levels) - 1).astype(int)
        for L in np.unique(level):
            idx = np.nonzero(level == L)[0]
            out[idx] = _bilinear(self.levels[L], s[idx] / self.texel_u / 2.0**L - 0.5, t[idx] / self.texel_v / 2.0**L - 0.5)
        return out


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.clip(x, 0.0, w - 1.0)
    y = np.clip(y, 0.0, h - 1.0)
    x0 = np.minimum(np.floor(x).astype(np.int64), w - 2) if w > 1 else np.zeros(len(x), np.int64)
    y0 = np.minimum(np.floor(y).astype(np.int64), h - 2) if h > 1 else np.zeros(len(y), np.int64)
    fx = (x - x0).astype(np.float32)[:, None]
    fy = (y - y0).astype(np.float32)[:, None]
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bottom * fy
