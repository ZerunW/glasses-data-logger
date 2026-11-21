"""Image cue for the forearm direction (NumPy + OpenCV; no file I/O).

From one hand's 2D keypoints, find the visible forearm next to the wrist and fit its 2D axis.
The axis (wrist pixel -> a pixel further along the forearm) defines, with the camera centre, a
plane that contains the forearm and hence the elbow (``core.elbow.plane_from_image_line``).

Search region: a wedge with its apex at the wrist, opening ``wedge_deg`` around the direction
*away from the fingers* (knuckles -> wrist), ``reach`` x the wrist-knuckle distance long; the
hand itself (dilated convex hull of its keypoints) is excluded. Inside it, a region connected to
seeds just beyond the wrist is grown by one of three modes:

* ``"skin"``: skin colour (YCrCb box, ``cr``/``cb`` ranges) - bare forearms only;
* ``"region"``: colour-agnostic region growing from the seeds (flood fill in Lab with a
  per-step tolerance, Canny edges as barriers), intersected with depth continuity when a depth
  map is given - works on sleeves too;
* ``"depth"``: depth continuity only (flood fill in log-depth from the seeds).

The axis is the principal direction of the region pixels relative to the wrist (a line through
the wrist), its far end the ``far_quantile`` of their projections. Confidence combines the
elongation of the region, its extent relative to the expected visible length, and penalises
regions that fill the whole wedge (a leak into the background) or are truncated by the image
border (wrist at the border, region running along it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .. import geometry as geo

MODES = ("skin", "region", "depth")


@dataclass
class CueParams:
    wedge_deg: float = 55.0          # half opening of the search wedge around "away from the fingers"
    reach: float = 3.0               # wedge length in wrist-to-knuckle distances (without forearm_px)
    reach_forearm: float = 1.05      # wedge length in expected forearm lengths (with forearm_px)
    hand_dilate: float = 0.12        # hand hull dilation, in wrist-to-knuckle distances
    seed_from: float = 0.25          # seeds along the wedge axis, from/to (wrist-to-knuckle units)
    seed_to: float = 0.6
    min_extent: float = 0.6          # shortest accepted axis, wrist-to-knuckle units
    full_length: float = 0.5         # extent (in wedge lengths) that gets full length confidence
    far_quantile: float = 0.95
    cr: tuple[float, float] = (133.0, 175.0)
    cb: tuple[float, float] = (77.0, 130.0)
    skin_wrist_radius: float = 0.35  # skin components must come this close to the wrist (knuckle units)
    lab_step: float = 5.0            # region: flood-fill tolerance per neighbouring pixel (Lab units)
    canny: tuple[float, float] = (40.0, 110.0)
    depth_step_rel: float = 0.03     # depth: tolerance per neighbouring pixel (relative)
    max_fill: float = 0.85           # region covering more of the wedge than this = leak
    edge_fraction: float = 0.06      # region pixels on the image border above this lower the confidence (0 at 2x)
    wrist_border: float = 0.5        # wrist closer to the image border than this (knuckle units) lowers it too
    min_pixels: int = 25

    @classmethod
    def from_mapping(cls, m: dict[str, Any] | None) -> "CueParams":
        p = cls()
        for k, v in dict(m or {}).items():
            if hasattr(p, k):
                setattr(p, k, tuple(v) if isinstance(getattr(p, k), tuple) else type(getattr(p, k))(v))
        return p


@dataclass
class ForearmCue:
    ok: bool
    p0: NDArray[np.float64] = field(default_factory=lambda: np.full(2, np.nan))  # wrist pixel
    p1: NDArray[np.float64] = field(default_factory=lambda: np.full(2, np.nan))  # far axis pixel
    conf: float = 0.0
    pixels: int = 0
    extent_px: float = 0.0
    reason: str = ""


def skin_mask(bgr: NDArray[np.uint8], cr=(133.0, 175.0), cb=(77.0, 130.0)) -> NDArray[np.bool_]:
    """Skin-coloured pixels (YCrCb box test)."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Cr = ycc[..., 1].astype(np.float32)
    Cb = ycc[..., 2].astype(np.float32)
    Y = ycc[..., 0]
    return (Cr >= cr[0]) & (Cr <= cr[1]) & (Cb >= cb[0]) & (Cb <= cb[1]) & (Y > 30)


def _wedge_mask(shape, apex, direction, length, half_angle_deg) -> NDArray[np.bool_]:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    dx = xx - apex[0]
    dy = yy - apex[1]
    along = dx * direction[0] + dy * direction[1]
    r = np.hypot(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = along / np.where(r > 0, r, 1.0)
    return (along >= 0) & (r <= length) & (cosang >= math.cos(math.radians(half_angle_deg)))


def _hand_hull(shape, kp_xy, dilate_px) -> NDArray[np.bool_]:
    m = np.zeros(shape, np.uint8)
    hull = cv2.convexHull(np.round(kp_xy).astype(np.int32))
    cv2.fillConvexPoly(m, hull, 1)
    r = int(round(dilate_px))
    if r > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return m > 0


def _flood(image: NDArray[Any], blocked: NDArray[np.bool_], seeds: list[tuple[int, int]], tol: Any) -> NDArray[np.bool_]:
    """Union of gradient-mode flood fills (4-connected) from ``seeds``, not entering ``blocked``."""
    h, w = blocked.shape
    out = np.zeros((h, w), bool)
    for sx, sy in seeds:
        if not (0 <= sx < w and 0 <= sy < h) or blocked[sy, sx] or out[sy, sx]:
            continue
        mask = np.zeros((h + 2, w + 2), np.uint8)
        mask[1:-1, 1:-1] = blocked.astype(np.uint8)
        img = image.copy()
        cv2.floodFill(img, mask, (int(sx), int(sy)), 0, tol, tol, 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        out |= mask[1:-1, 1:-1] == 255
    return out


def forearm_axis(
    bgr: NDArray[np.uint8],
    kp: NDArray[np.float64],
    mode: str = "region",
    depth: NDArray[np.float32] | None = None,
    params: CueParams | None = None,
    forearm_px: float | None = None,
) -> ForearmCue:
    """Forearm axis of one hand in ``bgr`` (any grid; ``kp`` (21, 2+) pixels in that grid).

    ``depth`` (optional) is a depth map of the same grid (NaN/<=0 invalid). ``forearm_px``
    (optional) is the expected image length of a forearm parallel to the image plane at the
    wrist's depth (``f * forearm_length / z``); it sets the wedge length (``reach_forearm``),
    otherwise ``reach`` wrist-to-knuckle distances are used. Returns a :class:`ForearmCue` with
    ``p0`` = the wrist keypoint and ``p1`` = the far end of the axis.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    p = params or CueParams()
    kp = np.asarray(kp, dtype=np.float64)[:, :2]
    ok = np.all(np.isfinite(kp), axis=1)
    if not ok[0] or ok.sum() < 4:
        return ForearmCue(False, reason="keypoints")
    h, w = bgr.shape[:2]
    wrist = kp[0]
    base = [5, 9, 13, 17]
    knuck = kp[base][ok[base]].mean(axis=0) if ok[base].any() else kp[1:][ok[1:]].mean(axis=0)
    axis = wrist - knuck
    unit = float(np.linalg.norm(axis))
    if unit < 3.0:
        return ForearmCue(False, reason="hand too small")
    d = axis / unit
    length = p.reach * unit
    if forearm_px is not None and np.isfinite(forearm_px) and forearm_px > 0:
        length = max(p.reach_forearm * float(forearm_px), unit)
    if not (0 <= wrist[0] <= w - 1 and 0 <= wrist[1] <= h - 1):
        return ForearmCue(False, reason="wrist outside the image")
    # Work on the bounding box of the wedge only.
    x0, y0 = int(max(0, math.floor(wrist[0] - length - 2))), int(max(0, math.floor(wrist[1] - length - 2)))
    x1, y1 = int(min(w, math.ceil(wrist[0] + length + 3))), int(min(h, math.ceil(wrist[1] + length + 3)))
    origin = np.array([x0, y0], dtype=np.float64)
    full_w, full_h = w, h
    wrist_border = float(min(wrist[0], wrist[1], w - 1 - wrist[0], h - 1 - wrist[1]))
    bgr = bgr[y0:y1, x0:x1]
    if depth is not None:
        depth = np.asarray(depth)[y0:y1, x0:x1]
    kp = kp - origin
    wrist = wrist - origin
    h, w = bgr.shape[:2]
    wedge = _wedge_mask((h, w), wrist, d, length, p.wedge_deg)
    hand = _hand_hull((h, w), kp[ok], p.hand_dilate * unit)
    allowed = wedge & ~hand
    if allowed.sum() < p.min_pixels:
        return ForearmCue(False, reason="no room")
    seeds = []
    for s in np.linspace(p.seed_from, p.seed_to, 4):
        for ang in (-12.0, 0.0, 12.0):
            c, sn = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            dd = np.array([c * d[0] - sn * d[1], sn * d[0] + c * d[1]])
            q = np.round(wrist + dd * s * unit).astype(int)
            if 0 <= q[0] < w and 0 <= q[1] < h and allowed[q[1], q[0]]:
                seeds.append((int(q[0]), int(q[1])))
    if not seeds:
        return ForearmCue(False, reason="no seed outside the hand")

    depth_ok = None
    if depth is not None:
        dv = np.asarray(depth, dtype=np.float32)
        good = np.isfinite(dv) & (dv > 0)
        if good.any():
            logd = np.where(good, np.log(np.where(good, dv, 1.0)), 0.0).astype(np.float32)
            tol = float(math.log1p(p.depth_step_rel))
            depth_ok = _flood(logd, ~(allowed & good), seeds, (tol,))

    if mode == "skin":
        region = skin_mask(bgr, p.cr, p.cb) & allowed
        n, labels = cv2.connectedComponents(region.astype(np.uint8), connectivity=8)
        # Only skin that starts at the wrist: components containing a seed or touching a small
        # disc around the wrist (a sleeve over the wrist leaves no such component: no cue).
        keep = {int(labels[sy, sx]) for sx, sy in seeds if labels[sy, sx] > 0}
        disc = (np.hypot(*np.mgrid[0:h, 0:w][::-1] - wrist[:, None, None]) <= p.skin_wrist_radius * unit) & region
        keep |= {int(v) for v in np.unique(labels[disc]) if v > 0}
        region = np.isin(labels, list(keep)) if keep else np.zeros_like(region)
        if depth_ok is not None:
            region &= cv2.dilate(depth_ok.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    elif mode == "region":
        lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (3, 3), 0), cv2.COLOR_BGR2Lab)
        edges = cv2.Canny(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), p.canny[0], p.canny[1]) > 0
        blocked = ~allowed | edges
        region = _flood(lab, blocked, seeds, (p.lab_step,) * 3)
        if depth_ok is not None:
            region &= cv2.dilate(depth_ok.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    else:  # depth
        if depth_ok is None:
            return ForearmCue(False, reason="no depth")
        region = depth_ok

    ys, xs = np.nonzero(region)
    if xs.size < p.min_pixels:
        return ForearmCue(False, pixels=int(xs.size), reason="region too small")
    fill = xs.size / max(int(allowed.sum()), 1)
    rel = np.stack([xs - wrist[0], ys - wrist[1]], axis=1).astype(np.float64)
    M = rel.T @ rel / len(rel)
    evals, evecs = np.linalg.eigh(M)
    e = evecs[:, -1]
    if e @ d < 0:
        e = -e
    proj = rel @ e
    extent = float(np.quantile(proj, p.far_quantile))
    if extent < p.min_extent * unit:
        return ForearmCue(False, pixels=int(xs.size), extent_px=extent, reason="axis too short")
    centred = rel - rel.mean(axis=0)
    cev = np.linalg.eigvalsh(centred.T @ centred / len(rel))
    elong = math.sqrt(max(cev[-1], 1e-9) / max(cev[0], 1e-9))
    s_elong = float(np.clip((elong - 1.5) / 2.5, 0.0, 1.0))
    s_len = float(np.clip(extent / (p.full_length * length), 0.0, 1.0))
    s_fill = 1.0 if fill <= p.max_fill else float(np.clip((1.0 - fill) / (1.0 - p.max_fill), 0.0, 1.0))
    s_dir = float(np.clip((e @ d - math.cos(math.radians(p.wedge_deg))) / (1 - math.cos(math.radians(p.wedge_deg))), 0.0, 1.0))
    # Truncation by the image border: a wrist at the border, or a region running along the
    # border (its visible part follows the image edge rather than the forearm).
    gx, gy = xs + origin[0], ys + origin[1]
    on_edge = (gx <= 1) | (gy <= 1) | (gx >= full_w - 2) | (gy >= full_h - 2)
    s_edge = float(np.clip(1.0 - (on_edge.mean() - p.edge_fraction) / p.edge_fraction, 0.0, 1.0))
    s_wrist = float(np.clip(wrist_border / max(p.wrist_border * unit, 1e-6), 0.0, 1.0))
    conf = s_elong * s_len * s_fill * (0.5 + 0.5 * s_dir) * s_edge * s_wrist
    return ForearmCue(True, wrist + origin, wrist + origin + e * extent, float(conf), int(xs.size), extent)


def cue_plane(
    camera: geo.Camera,
    T_world_cam: NDArray[np.float64],
    p0: NDArray[np.float64],
    p1: NDArray[np.float64],
    wrist_world: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Plane ``[nx, ny, nz, d]`` (world) through the camera centre containing the viewing rays of
    pixels ``p0`` (wrist) and ``p1`` (forearm), and the direction from the wrist along the
    forearm (``cue_side`` of ``solve_elbow``): the point on ray ``p1`` at the wrist's camera depth
    minus the wrist (or ray ``p1`` minus ray ``p0`` at unit depth without ``wrist_world``).
    NaN when a pixel does not undistort."""
    from .elbow import plane_from_image_line

    xy = camera.undistort_points(np.stack([p0, p1]))
    if not np.all(np.isfinite(xy)):
        return np.full(4, np.nan), np.full(3, np.nan)
    rays_cam = np.concatenate([xy, np.ones((2, 1))], axis=1)
    R = T_world_cam[:3, :3]
    c = T_world_cam[:3, 3]
    ra, rb = rays_cam @ R.T
    n, dd = plane_from_image_line(c, ra, rb)
    if wrist_world is not None and np.all(np.isfinite(wrist_world)):
        z = float((geo.inv_T(T_world_cam) @ np.append(wrist_world, 1.0))[2])
        side = c + rb * z - wrist_world
    else:
        side = rb - ra
    return np.append(n, dd), side


__all__ = ["CueParams", "ForearmCue", "MODES", "skin_mask", "forearm_axis", "cue_plane"]
