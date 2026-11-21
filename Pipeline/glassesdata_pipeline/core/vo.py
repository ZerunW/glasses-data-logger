"""Building blocks of the metric visual odometry used by the ``slam`` step (NumPy + OpenCV).

No file I/O. Images are grey ``uint8`` in a (possibly downsampled) grid; points are ``(N, 2)``
float pixel coordinates in that grid (pixel centres at integers). PnP works on undistorted
normalized coordinates (``K = I``), so its thresholds are in normalized units (pixels / f).

* :func:`hand_mask` - mask of the hands (dilated convex hull of the 2D keypoints, optionally
  extended along the forearm) in any grid.
* :func:`border_mask`, :func:`detect` - GFTT corners in empty grid cells, outside masks.
* :func:`track` - pyramidal KLT with a forward-backward check.
* :func:`prepare_depth`, :func:`sample_depth` - bilinear depth lookup that rejects depth edges
  (found on a median-filtered copy, so per-pixel noise is not taken for an edge).
* :func:`pnp` - RANSAC PnP + Levenberg-Marquardt refinement on the inliers.
* :func:`orb_features`, :func:`match_descriptors` - ORB features of a keyframe and ratio-test
  matching (for pose-graph edges to older keyframes and relocalisation).
* :func:`grid_to_grid` - map pixel coordinates between grids of different downsampling.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .. import geometry as geo


# =============================================================================================
# Grids and masks
# =============================================================================================


def grid_to_grid(uv: NDArray[np.float64], src_size: tuple[int, int], dst_size: tuple[int, int]) -> NDArray[np.float64]:
    """Pixel coordinates in a ``src_size`` (w, h) image -> the same points in a ``dst_size``
    image of the same field of view (pixel-centre convention ``u' = (u + 0.5) s - 0.5``)."""
    uv = np.asarray(uv, dtype=np.float64)
    sx = dst_size[0] / src_size[0]
    sy = dst_size[1] / src_size[1]
    out = np.empty_like(uv)
    out[..., 0] = (uv[..., 0] + 0.5) * sx - 0.5
    out[..., 1] = (uv[..., 1] + 0.5) * sy - 0.5
    return out


def hand_mask(
    kp: NDArray[np.float64] | None,
    full_size: tuple[int, int],
    grid_size: tuple[int, int],
    margin_px: float = 12.0,
    forearm: float = 1.5,
    min_score: float = 0.0,
) -> NDArray[np.uint8]:
    """``uint8`` mask (255 = hand) in the ``grid_size`` (w, h) image.

    ``kp`` is ``(H, 21, 3)`` keypoints ``(u, v, score)`` in the ``full_size`` image (NaN when
    missing). Each hand's convex hull is dilated by ``margin_px`` (grid pixels). With ``forearm``
    > 0 a quadrilateral from the wrist away from the knuckles, ``forearm`` x the wrist-to-middle
    knuckle distance long and one palm width wide, covers the visible forearm/sleeve too.
    """
    w, h = grid_size
    mask = np.zeros((h, w), np.uint8)
    if kp is None:
        return mask
    kp = np.asarray(kp, dtype=np.float64).reshape(-1, 21, 3)
    for hand in kp:
        ok = np.all(np.isfinite(hand[:, :2]), axis=1)
        if np.isfinite(min_score) and min_score > 0:
            ok &= ~(hand[:, 2] < min_score)
        if ok.sum() < 3:
            continue
        uv = grid_to_grid(hand[ok, :2], full_size, grid_size)
        hull = cv2.convexHull(np.round(uv).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        if forearm > 0 and ok[0] and ok[9]:
            wrist = grid_to_grid(hand[0, :2], full_size, grid_size)
            knuck = grid_to_grid(hand[9, :2], full_size, grid_size)
            axis = wrist - knuck
            length = float(np.linalg.norm(axis))
            if length > 1.0:
                d = axis / length
                n = np.array([-d[1], d[0]])
                width = 0.5 * length
                if ok[5] and ok[17]:
                    width = max(width, 0.6 * float(np.linalg.norm(
                        grid_to_grid(hand[5, :2], full_size, grid_size) - grid_to_grid(hand[17, :2], full_size, grid_size))))
                far = wrist + d * forearm * length
                quad = np.array([wrist + n * width, wrist - n * width, far - n * width * 1.2, far + n * width * 1.2])
                cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    if margin_px > 0 and mask.any():
        r = int(math.ceil(margin_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        mask = cv2.dilate(mask, kernel)
    return mask


def border_mask(size: tuple[int, int], border_px: int) -> NDArray[np.uint8]:
    """``uint8`` mask (255 = usable) excluding ``border_px`` pixels at every edge."""
    w, h = size
    m = np.zeros((h, w), np.uint8)
    b = int(border_px)
    m[b:h - b, b:w - b] = 255
    return m


# =============================================================================================
# Features and tracking
# =============================================================================================


def detect(
    image: NDArray[np.uint8],
    usable: NDArray[np.uint8],
    existing: NDArray[np.float64] | None = None,
    max_corners: int = 400,
    quality: float = 0.01,
    min_distance: float = 8.0,
    cell_px: int = 40,
    per_cell: int = 4,
) -> NDArray[np.float64]:
    """New GFTT corners ``(M, 2)`` where ``usable`` is nonzero, at least ``min_distance`` from
    ``existing`` points, only in grid cells (``cell_px``) holding fewer than ``per_cell`` points.
    At most ``max_corners`` minus the existing count are returned (strongest first)."""
    h, w = image.shape[:2]
    mask = usable.copy()
    existing = np.zeros((0, 2)) if existing is None else np.asarray(existing, dtype=np.float64).reshape(-1, 2)
    budget = int(max_corners) - len(existing)
    if budget <= 0:
        return np.zeros((0, 2))
    cw = max(1, int(math.ceil(w / cell_px)))
    ch = max(1, int(math.ceil(h / cell_px)))
    counts = np.zeros((ch, cw), np.int64)
    if len(existing):
        cx = np.clip((existing[:, 0] // cell_px).astype(int), 0, cw - 1)
        cy = np.clip((existing[:, 1] // cell_px).astype(int), 0, ch - 1)
        np.add.at(counts, (cy, cx), 1)
        for (u, v) in np.round(existing).astype(int):
            cv2.circle(mask, (int(u), int(v)), int(min_distance), 0, -1)
    for cy_, cx_ in zip(*np.nonzero(counts >= per_cell)):
        mask[cy_ * cell_px:(cy_ + 1) * cell_px, cx_ * cell_px:(cx_ + 1) * cell_px] = 0
    if not mask.any():
        return np.zeros((0, 2))
    pts = cv2.goodFeaturesToTrack(image, maxCorners=int(budget) * 3, qualityLevel=float(quality),
                                  minDistance=float(min_distance), mask=mask, blockSize=5)
    if pts is None:
        return np.zeros((0, 2))
    pts = pts.reshape(-1, 2).astype(np.float64)
    # Keep the cell quota: walk the (strength-ordered) corners and fill each cell up to per_cell.
    keep = []
    cxs = np.clip((pts[:, 0] // cell_px).astype(int), 0, cw - 1)
    cys = np.clip((pts[:, 1] // cell_px).astype(int), 0, ch - 1)
    for k in range(len(pts)):
        if counts[cys[k], cxs[k]] < per_cell:
            counts[cys[k], cxs[k]] += 1
            keep.append(k)
            if len(keep) >= budget:
                break
    pts = pts[keep]
    if len(pts):
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01)
        refined = cv2.cornerSubPix(image, pts.astype(np.float32).reshape(-1, 1, 2), (3, 3), (-1, -1), crit)
        pts = refined.reshape(-1, 2).astype(np.float64)
    return pts


def track(
    prev: NDArray[np.uint8],
    cur: NDArray[np.uint8],
    pts: NDArray[np.float64],
    win: int = 21,
    levels: int = 3,
    fb_thresh: float = 1.0,
    guess: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Pyramidal Lucas-Kanade from ``prev`` to ``cur`` with a forward-backward check: a point is
    kept when tracking it back lands within ``fb_thresh`` pixels of where it started and it stays
    inside the image. Returns ``(new_pts (N, 2), ok (N,))``."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0, bool)
    p0 = pts.astype(np.float32).reshape(-1, 1, 2)
    params = dict(winSize=(int(win), int(win)), maxLevel=int(levels),
                  criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    flags = 0
    p1_init = None
    if guess is not None:
        p1_init = np.asarray(guess, dtype=np.float32).reshape(-1, 1, 2).copy()
        flags = cv2.OPTFLOW_USE_INITIAL_FLOW
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, p1_init, flags=flags, **params)
    p0b, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, p0.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **params)
    p1 = p1.reshape(-1, 2).astype(np.float64)
    fb = np.linalg.norm(p0b.reshape(-1, 2) - pts, axis=1)
    h, w = cur.shape[:2]
    ok = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < fb_thresh)
    ok &= (p1[:, 0] >= 0) & (p1[:, 0] <= w - 1) & (p1[:, 1] >= 0) & (p1[:, 1] <= h - 1)
    return p1, ok


# =============================================================================================
# Depth
# =============================================================================================


class DepthMap:
    """A depth map prepared for lookups: ``values`` (float32, NaN where invalid, optionally
    median filtered) and ``ok`` (bool, valid and away from depth edges)."""

    def __init__(self, values: NDArray[np.float32], ok: NDArray[np.bool_]) -> None:
        self.values = values
        self.ok = ok

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    def scaled(self, factor: float) -> "DepthMap":
        return DepthMap(self.values * np.float32(factor), self.ok)


def prepare_depth(depth: NDArray[Any], median: int = 3, edge_rel: float = 0.08, edge_median: int = 5,
                  edge_window: int = 3) -> DepthMap:
    """Depth lookup structure.

    ``values``: float32 copy with non-positive / non-finite pixels NaN, median filtered with
    ``median`` (3 or 5; 0/1 = off) to suppress per-pixel noise.
    ``ok``: the pixel is valid and not at a depth edge. Edges are found on a more strongly
    filtered copy (median ``edge_median``) so that pixel noise is not mistaken for an edge: a pixel
    is an edge when ``max / min`` over its ``edge_window`` x ``edge_window`` neighbourhood exceeds
    ``1 + edge_rel``.
    """
    d = np.asarray(depth, dtype=np.float32).copy()
    bad = ~np.isfinite(d) | (d <= 0)
    filled = np.where(bad, 0.0, d).astype(np.float32)
    values = cv2.medianBlur(filled, int(median) if median in (3, 5) else 3) if median and median > 1 else filled.copy()
    for_edges = cv2.medianBlur(filled, int(edge_median) if edge_median in (3, 5) else 5) if edge_median and edge_median > 1 else values
    k = max(1, int(edge_window))
    kernel = np.ones((k, k), np.uint8)
    hi = cv2.dilate(for_edges, kernel)
    lo_src = np.where(for_edges > 0, for_edges, np.float32(np.inf)).astype(np.float32)
    lo = cv2.erode(lo_src, kernel)
    bad_grown = cv2.dilate(bad.astype(np.uint8), kernel) > 0
    ok = ~bad_grown & (values > 0) & (hi <= lo * np.float32(1.0 + edge_rel))
    values = values.astype(np.float32)
    values[(values <= 0) | bad] = np.nan
    return DepthMap(values, ok)


def sample_depth(
    depth: DepthMap,
    uv: NDArray[np.float64],
    min_depth: float = 0.1,
    max_depth: float = 10.0,
) -> NDArray[np.float64]:
    """Bilinear depth at ``uv`` (pixels of the depth grid). NaN outside the image, outside
    ``[min_depth, max_depth]``, or when any of the four bilinear neighbours is invalid or at a
    depth edge (``depth.ok`` False)."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    h, w = depth.shape
    out = np.full(len(uv), np.nan)
    if len(uv) == 0:
        return out
    u, v = uv[:, 0], uv[:, 1]
    inside = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    if not np.any(inside):
        return out
    u0 = np.clip(np.floor(u[inside]).astype(int), 0, w - 2)
    v0 = np.clip(np.floor(v[inside]).astype(int), 0, h - 2)
    fu = u[inside] - u0
    fv = v[inside] - v0
    D, OK = depth.values, depth.ok
    val = ((1 - fv) * ((1 - fu) * D[v0, u0] + fu * D[v0, u0 + 1])
           + fv * ((1 - fu) * D[v0 + 1, u0] + fu * D[v0 + 1, u0 + 1]))
    ok = OK[v0, u0] & OK[v0, u0 + 1] & OK[v0 + 1, u0] & OK[v0 + 1, u0 + 1]
    ok &= np.isfinite(val) & (val >= min_depth) & (val <= max_depth)
    out[inside] = np.where(ok, val, np.nan)
    return out


# =============================================================================================
# PnP
# =============================================================================================


def pnp(
    X: NDArray[np.float64],
    xy: NDArray[np.float64],
    thresh: float,
    iterations: int = 200,
    confidence: float = 0.999,
    min_inliers: int = 12,
    guess: NDArray[np.float64] | None = None,
) -> tuple[bool, NDArray[np.float64], NDArray[np.bool_]]:
    """Pose of a camera from 3D points ``X`` (N, 3) in a reference frame and their undistorted
    normalized image coordinates ``xy`` (N, 2).

    RANSAC (EPnP minimal solver, threshold ``thresh`` in normalized units) followed by two rounds
    of Levenberg-Marquardt refinement on the inliers (re-selecting inliers in between).

    Returns ``(ok, T_cam_ref (4, 4), inliers (N,))``: ``x_cam = T_cam_ref @ X``.
    """
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    n = len(X)
    inl = np.zeros(n, bool)
    if n < max(int(min_inliers), 6):
        return False, np.eye(4), inl
    K = np.eye(3)
    use_guess = guess is not None
    rvec0 = tvec0 = None
    if use_guess:
        rvec0 = cv2.Rodrigues(np.asarray(guess, dtype=np.float64)[:3, :3])[0]
        tvec0 = np.asarray(guess, dtype=np.float64)[:3, 3].reshape(3, 1).copy()
    try:
        ok, rvec, tvec, idx = cv2.solvePnPRansac(
            X, xy, K, None, rvec0, tvec0, useExtrinsicGuess=use_guess, iterationsCount=int(iterations),
            reprojectionError=float(thresh), confidence=float(confidence), flags=cv2.SOLVEPNP_EPNP)
    except cv2.error:
        return False, np.eye(4), inl
    if not ok or idx is None or len(idx) < min_inliers:
        return False, np.eye(4), inl
    inl[idx.reshape(-1)] = True
    for _ in range(2):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(X[inl], xy[inl], K, None, rvec, tvec)
        except cv2.error:
            break
        err = reprojection_error(X, xy, rvec, tvec)
        new = err < thresh
        if new.sum() < min_inliers:
            break
        inl = new
    R = cv2.Rodrigues(rvec)[0]
    T = geo.make_T(R, tvec.reshape(3))
    good = inl.sum() >= min_inliers and np.all(np.isfinite(T))
    return bool(good), T, inl


def reprojection_error(X, xy, rvec, tvec) -> NDArray[np.float64]:
    """Normalized-plane distance between ``xy`` and the projection of ``X`` (NaN behind)."""
    R = cv2.Rodrigues(rvec)[0]
    Xc = X @ R.T + np.asarray(tvec).reshape(1, 3)
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = Xc[:, :2] / z[:, None]
    err = np.linalg.norm(p - xy, axis=1)
    return np.where(z > 1e-6, err, np.inf)


# =============================================================================================
# Descriptors
# =============================================================================================


def orb_features(image: NDArray[np.uint8], usable: NDArray[np.uint8], n_features: int = 800,
                 edge: int = 15) -> tuple[NDArray[np.float64], NDArray[np.uint8] | None]:
    """ORB keypoints ``(M, 2)`` and descriptors ``(M, 32)`` inside ``usable``."""
    orb = cv2.ORB_create(nfeatures=int(n_features), scaleFactor=1.2, nlevels=6, edgeThreshold=int(edge),
                         patchSize=int(edge), fastThreshold=10)
    kps, des = orb.detectAndCompute(image, usable)
    if des is None or not kps:
        return np.zeros((0, 2)), None
    return np.array([k.pt for k in kps], dtype=np.float64), des


def match_descriptors(des_a: NDArray[np.uint8] | None, des_b: NDArray[np.uint8] | None,
                      ratio: float = 0.8) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Mutual nearest neighbours (Hamming) that also pass Lowe's ratio test: ``(ia, ib)``."""
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64))
    if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
        return empty
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    ab = bf.knnMatch(des_a, des_b, k=2)
    ba = bf.match(des_b, des_a)
    back = {m.queryIdx: m.trainIdx for m in ba}
    ia, ib = [], []
    for pair in ab:
        if len(pair) < 2:
            continue
        m, m2 = pair
        if m.distance < ratio * m2.distance and back.get(m.trainIdx) == m.queryIdx:
            ia.append(m.queryIdx)
            ib.append(m.trainIdx)
    return np.asarray(ia, np.int64), np.asarray(ib, np.int64)


__all__ = [
    "grid_to_grid", "hand_mask", "border_mask", "detect", "track", "DepthMap", "prepare_depth", "sample_depth",
    "pnp", "reprojection_error", "orb_features", "match_descriptors",
]
