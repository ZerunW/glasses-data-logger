"""Planes and 2-D polygons (numpy + OpenCV): least-squares and RANSAC plane fits, horizontal
support planes, convex polygons (hull, area, intersection over union), occupancy components.

Plane convention: ``normal . p + offset = 0`` with a unit ``normal`` (the Hessian normal form used by
Open3D's ``segment_plane``); the signed distance of ``p`` is ``normal . p + offset``. Normals are
oriented towards ``up`` (default world +z) when one is given.
"""

from __future__ import annotations

from typing import Any

import numpy as np

UP = np.array([0.0, 0.0, 1.0])


def fit_plane(points: Any, weights: Any = None, up: Any = UP) -> tuple[np.ndarray, float]:
    """Weighted total-least-squares plane ``(normal, offset)`` (smallest principal axis)."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 3:
        raise ValueError("need at least 3 points for a plane")
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=np.float64)
    c = (w[:, None] * p).sum(axis=0) / w.sum()
    d = p - c
    cov = (w[:, None] * d).T @ d
    _, vecs = np.linalg.eigh(cov)
    n = vecs[:, 0]
    if up is not None and n @ np.asarray(up, dtype=np.float64) < 0:
        n = -n
    return n, float(-n @ c)


def plane_distance(points: Any, normal: Any, offset: float) -> np.ndarray:
    """Signed distances ``normal . p + offset``."""
    return np.asarray(points, dtype=np.float64) @ np.asarray(normal, dtype=np.float64) + float(offset)


def plane_height(normal: Any, offset: float, xy: Any) -> np.ndarray:
    """z of the plane above the points ``xy`` ``(..., 2)`` (needs a non-vertical normal)."""
    n = np.asarray(normal, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    return -(offset + n[0] * xy[..., 0] + n[1] * xy[..., 1]) / n[2]


def tilt_deg(normal: Any, up: Any = UP) -> float:
    n = np.asarray(normal, dtype=np.float64)
    u = np.asarray(up, dtype=np.float64)
    return float(np.degrees(np.arccos(np.clip(abs(n @ u) / (np.linalg.norm(n) * np.linalg.norm(u)), 0.0, 1.0))))


def ransac_plane(
    points: Any,
    threshold: Any,
    iterations: int = 256,
    seed: int = 0,
    up: Any = UP,
    max_tilt_deg: float | None = None,
    refine: int = 3,
    batch: int = 64,
) -> dict[str, Any] | None:
    """RANSAC plane with a scalar or per-point inlier ``threshold``.

    Hypotheses from 3 random points; with ``max_tilt_deg`` only planes whose normal is within that
    angle of ``up`` are kept. The best hypothesis (most inliers) is refined ``refine`` times by a
    least-squares fit of its inliers. Returns ``{"normal", "offset", "inliers" (bool N),
    "n_inliers", "rms"}`` or None when no hypothesis survives.
    """
    p = np.asarray(points, dtype=np.float64)
    n_pts = len(p)
    if n_pts < 3:
        return None
    thr = np.broadcast_to(np.asarray(threshold, dtype=np.float64), (n_pts,))
    rng = np.random.default_rng(seed)
    up_v = np.asarray(up, dtype=np.float64) if up is not None else None
    cos_max = np.cos(np.radians(max_tilt_deg)) if max_tilt_deg is not None else -1.0
    best_count, best = -1, None
    centre = p.mean(axis=0)
    q = p - centre  # numerically nicer
    for start in range(0, iterations, batch):
        b = min(batch, iterations - start)
        idx = rng.integers(0, n_pts, size=(b, 3))
        a0, a1, a2 = q[idx[:, 0]], q[idx[:, 1]], q[idx[:, 2]]
        nrm = np.cross(a1 - a0, a2 - a0)
        length = np.linalg.norm(nrm, axis=1)
        ok = length > 1e-12
        nrm = nrm / np.where(ok, length, 1.0)[:, None]
        if up_v is not None:
            sign = np.where(nrm @ up_v < 0, -1.0, 1.0)
            nrm *= sign[:, None]
            ok &= nrm @ up_v >= cos_max
        if not np.any(ok):
            continue
        nrm, a0 = nrm[ok], a0[ok]
        off = -np.einsum("ij,ij->i", nrm, a0)
        counts = (np.abs(q @ nrm.T + off) <= thr[:, None]).sum(axis=0)
        k = int(np.argmax(counts))
        if counts[k] > best_count:
            best_count, best = int(counts[k]), (nrm[k], float(off[k]))
    if best is None:
        return None
    normal, offset = best
    inliers = np.abs(q @ normal + offset) <= thr
    for _ in range(max(refine, 0)):
        if inliers.sum() < 3:
            break
        normal_r, offset_r = fit_plane(q[inliers], up=up_v)
        if up_v is not None and max_tilt_deg is not None and tilt_deg(normal_r, up_v) > max_tilt_deg:
            break
        normal, offset = normal_r, offset_r
        inliers = np.abs(q @ normal + offset) <= thr
    offset_world = float(offset - normal @ centre)
    dist = p[inliers] @ normal + offset_world
    return {
        "normal": normal,
        "offset": offset_world,
        "inliers": inliers,
        "n_inliers": int(inliers.sum()),
        "rms": float(np.sqrt(np.mean(dist**2))) if len(dist) else float("nan"),
    }


def horizontal_planes(
    points: Any,
    threshold: Any,
    max_planes: int = 4,
    min_inliers: int = 500,
    max_tilt_deg: float = 10.0,
    iterations: int = 256,
    seed: int = 0,
    up: Any = UP,
) -> list[dict[str, Any]]:
    """Sequential RANSAC: up to ``max_planes`` planes within ``max_tilt_deg`` of horizontal, each
    with at least ``min_inliers`` points (inliers of earlier planes are removed first). Each dict
    has ``normal``, ``offset``, ``inliers`` (indices into ``points``), ``n_inliers``, ``rms`` and
    ``z`` (median height of the inliers). Sorted by height, lowest first."""
    p = np.asarray(points, dtype=np.float64)
    thr_all = np.broadcast_to(np.asarray(threshold, dtype=np.float64), (len(p),))
    remaining = np.arange(len(p))
    planes: list[dict[str, Any]] = []
    for k in range(max_planes):
        if len(remaining) < max(min_inliers, 3):
            break
        res = ransac_plane(p[remaining], thr_all[remaining], iterations, seed + k, up, max_tilt_deg)
        if res is None or res["n_inliers"] < min_inliers:
            break
        idx = remaining[res["inliers"]]
        planes.append({
            "normal": res["normal"], "offset": res["offset"], "inliers": idx, "n_inliers": len(idx),
            "rms": res["rms"], "z": float(np.median(p[idx, 2])),
        })
        remaining = remaining[~res["inliers"]]
    planes.sort(key=lambda pl: pl["z"])
    return planes


# =============================================================================================
# 2-D polygons
# =============================================================================================


def convex_polygon(xy: Any) -> np.ndarray:
    """Convex hull ``(K, 2)`` of 2-D points, counter-clockwise (OpenCV ``convexHull``)."""
    import cv2

    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 3:
        return pts.copy()
    hull = cv2.convexHull(pts.astype(np.float32), clockwise=False).reshape(-1, 2).astype(np.float64)
    if polygon_area(hull) < 0:  # OpenCV's orientation flag refers to image (y-down) axes
        hull = hull[::-1]
    return hull


def polygon_area(poly: Any) -> float:
    """Signed area (shoelace; positive when counter-clockwise)."""
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def convex_polygon_iou(a: Any, b: Any) -> dict[str, float]:
    """Intersection over union of two convex polygons, and the intersection as a fraction of each
    (``frac_a`` = |a & b| / |a|, ``frac_b`` = |a & b| / |b|)."""
    import cv2

    pa = convex_polygon(a)
    pb = convex_polygon(b)
    area_a, area_b = abs(polygon_area(pa)), abs(polygon_area(pb))
    if area_a <= 0 or area_b <= 0:
        return {"iou": 0.0, "frac_a": 0.0, "frac_b": 0.0, "area_a": area_a, "area_b": area_b}
    inter, _ = cv2.intersectConvexConvex(pa.astype(np.float32), pb.astype(np.float32))
    inter = float(max(inter, 0.0))
    return {"iou": inter / (area_a + area_b - inter), "frac_a": inter / area_a, "frac_b": inter / area_b,
            "area_a": area_a, "area_b": area_b, "intersection": inter}


def component_mask(xy: Any, cell: float, seeds: Any = None, close_cells: int = 1, min_count: int = 1) -> np.ndarray:
    """Points of one connected region of a 2-D occupancy grid.

    The points are binned into ``cell``-sized squares (cells with fewer than ``min_count`` points
    are empty), gaps are closed with a ``close_cells`` morphological closing, and the 8-connected
    component that contains a seed (or, without seeds inside any component, the one nearest to the
    seeds; without seeds, the one with most points) is returned as a bool mask over ``xy``.
    """
    import cv2

    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    lo = pts.min(axis=0) - (close_cells + 1) * cell
    ij = np.floor((pts - lo) / cell).astype(np.int64)
    shape = ij.max(axis=0) + close_cells + 2
    counts = np.zeros((int(shape[1]), int(shape[0])), dtype=np.int32)
    np.add.at(counts, (ij[:, 1], ij[:, 0]), 1)
    occ = (counts >= min_count).astype(np.uint8)
    if close_cells > 0:
        kernel = np.ones((2 * close_cells + 1, 2 * close_cells + 1), np.uint8)
        occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, kernel)
    n_lab, labels = cv2.connectedComponents(occ, connectivity=8)
    lab_pts = labels[ij[:, 1], ij[:, 0]]
    if n_lab <= 1:
        return np.ones(len(pts), dtype=bool)
    chosen = None
    if seeds is not None and len(np.asarray(seeds).reshape(-1, 2)):
        s = np.asarray(seeds, dtype=np.float64).reshape(-1, 2)
        sij = np.floor((s - lo) / cell).astype(np.int64)
        inside = (sij[:, 0] >= 0) & (sij[:, 1] >= 0) & (sij[:, 0] < labels.shape[1]) & (sij[:, 1] < labels.shape[0])
        labs = labels[sij[inside, 1], sij[inside, 0]] if np.any(inside) else np.zeros(0, int)
        labs = labs[labs > 0]
        if len(labs):
            chosen = int(np.bincount(labs).argmax())
        else:
            # nearest occupied point to any seed
            d = np.min(np.linalg.norm(pts[:, None, :] - s[None], axis=-1), axis=1)
            chosen = int(lab_pts[int(np.argmin(d))])
    if chosen is None or chosen == 0:
        counts_lab = np.bincount(lab_pts, minlength=n_lab)
        counts_lab[0] = 0
        chosen = int(counts_lab.argmax())
    return lab_pts == chosen
