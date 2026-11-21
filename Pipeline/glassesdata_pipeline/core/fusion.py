"""Depth fusion for one object (numpy + OpenCV): back-projection, point-cloud filters, a small
TSDF volume, shape completion against a support plane and the object frame.

Point clouds are ``(N, 3)`` float64 arrays. Depth maps are metric z-depth on the distorted image
grid of a :class:`~glassesdata_pipeline.geometry.Camera` (the depth grid's camera, see
``Camera.downsampled``); ``rays`` are that camera's ``pixel_rays()`` (``X_cam = ray * depth``).

TSDF (Curless & Levoy 1996; KinectFusion): every voxel keeps the weighted mean of the truncated,
normalised projective signed distance ``clip((d_obs - z_voxel) / trunc, -1, 1)``, updated from
pixels inside the object's mask; with ``carve`` also from pixels outside the mask whose observed
depth lies clearly behind the voxel (free space, +1). Averaging along the viewing rays is what
removes per-pixel depth noise, which is mostly along the ray. The surface is extracted as the
zero crossings between neighbouring observed voxels (linear interpolation), with normals from the
TSDF gradient (pointing out of the object).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import geometry as geo
from .knn import KNN, self_knn
from .plane import plane_distance


# =============================================================================================
# Masks and back-projection
# =============================================================================================


def erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Binary erosion by ``pixels`` (3x3 cross steps); 0 returns the mask unchanged."""
    import cv2

    m = np.asarray(mask, dtype=bool)
    if pixels <= 0:
        return m
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    return cv2.erode(m.astype(np.uint8), kernel, iterations=int(pixels)).astype(bool)


def dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    import cv2

    m = np.asarray(mask, dtype=bool)
    if pixels <= 0:
        return m
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    return cv2.dilate(m.astype(np.uint8), kernel).astype(bool)


def backproject(depth: np.ndarray, mask: np.ndarray | None, rays: np.ndarray, min_depth: float = 0.05,
                max_depth: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame points of the masked pixels with a finite depth in range, and their pixel
    indices ``(N, 2)`` as ``(row, col)``."""
    d = np.asarray(depth, dtype=np.float64)
    ok = np.isfinite(d) & (d >= min_depth) & (d <= max_depth) & np.all(np.isfinite(rays), axis=-1)
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    rows, cols = np.nonzero(ok)
    return rays[rows, cols] * d[rows, cols, None], np.stack([rows, cols], axis=1)


# =============================================================================================
# Point-cloud filters
# =============================================================================================


def voxel_downsample(points: Any, voxel: float, normals: Any = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Mean of the points (and renormalised mean of the normals) in each occupied voxel."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) == 0:
        return p.reshape(0, 3), None if normals is None else np.zeros((0, 3))
    key = np.floor((p - p.min(axis=0)) / voxel).astype(np.int64)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    out = np.zeros((len(counts), 3))
    np.add.at(out, inverse, p)
    out /= counts[:, None]
    if normals is None:
        return out, None
    nsum = np.zeros((len(counts), 3))
    np.add.at(nsum, inverse, np.asarray(normals, dtype=np.float64))
    nsum /= np.maximum(np.linalg.norm(nsum, axis=1, keepdims=True), 1e-12)
    return out, nsum


def statistical_outlier_mask(points: Any, k: int = 16, std_ratio: float = 2.0) -> np.ndarray:
    """Open3D's statistical outlier removal: keep points whose mean distance to their ``k``
    neighbours is below mean + ``std_ratio`` x std of that statistic."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) <= k:
        return np.ones(len(p), dtype=bool)
    d, _ = self_knn(p, k)
    mean_d = d.mean(axis=1)
    return mean_d <= mean_d.mean() + std_ratio * mean_d.std()


def estimate_normals(points: Any, k: int = 16, toward: Any = None) -> np.ndarray:
    """Unit normals by local PCA of the ``k`` nearest neighbours. Oriented towards the point
    ``toward`` (a viewpoint) when given, else away from the cloud's centroid."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 3:
        return np.tile([0.0, 0.0, 1.0], (len(p), 1))
    _, idx = KNN(p).query(p, min(k, len(p)))
    nb = p[idx]
    c = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c)
    _, vecs = np.linalg.eigh(cov)
    n = vecs[:, :, 0]
    ref = (np.asarray(toward, dtype=np.float64) - p) if toward is not None else (p - p.mean(axis=0))
    flip = np.einsum("ij,ij->i", n, ref) < 0
    n[flip] *= -1
    return n


def mls_smooth(points: Any, k: int = 24, iterations: int = 1, radius: float | None = None) -> np.ndarray:
    """Moving-least-squares style denoising: every point is projected onto the plane fitted (PCA)
    to its ``k`` nearest neighbours (only those within ``radius`` when given). Removes noise
    across the surface; sharp edges are rounded over about the neighbourhood size."""
    p = np.asarray(points, dtype=np.float64).copy()
    if len(p) < 4:
        return p
    k = min(int(k), len(p))
    for _ in range(max(int(iterations), 0)):
        d, idx = KNN(p).query(p, k)
        nb = p[idx]
        if radius is not None:
            w = (d <= radius).astype(np.float64)
            w[:, 0] = 1.0
        else:
            w = np.ones(d.shape)
        c = (nb * w[..., None]).sum(axis=1) / w.sum(axis=1, keepdims=True)
        q = (nb - c[:, None, :]) * np.sqrt(w)[..., None]
        cov = np.einsum("nki,nkj->nij", q, q)
        _, vecs = np.linalg.eigh(cov)
        n = vecs[:, :, 0]
        p = p - np.einsum("ij,ij->i", p - c, n)[:, None] * n
    return p


# =============================================================================================
# TSDF volume
# =============================================================================================


class TSDFVolume:
    """A dense TSDF over an axis-aligned world box.

    Args:
        lo, hi: world bounds (3,). The voxel size grows when the grid would exceed ``max_voxels``.
        voxel: voxel edge (m).
        trunc: truncation distance (m).
    """

    def __init__(self, lo: Any, hi: Any, voxel: float, trunc: float, max_voxels: int = 3_000_000) -> None:
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        size = np.maximum(hi - lo, voxel)
        voxel = float(voxel)
        while np.prod(np.ceil(size / voxel)) > max_voxels:
            voxel *= 1.25
        self.voxel = voxel
        self.trunc = float(trunc)
        self.dims = np.ceil(size / voxel).astype(np.int64)
        self.lo = lo
        nx, ny, nz = (int(v) for v in self.dims)
        grid = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), axis=-1)
        self.centres = (lo + (grid.reshape(-1, 3) + 0.5) * voxel)
        self.tsdf = np.zeros(len(self.centres))
        self.weight = np.zeros(len(self.centres))
        self.n_obs = np.zeros(len(self.centres), dtype=np.int32)  # surface observations (inside the mask)

    def integrate(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        camera: geo.Camera,
        T_world_cam: np.ndarray,
        carve_mask: np.ndarray | None = None,
        carve_margin: float | None = None,
        free_weight: float = 1.0,
    ) -> int:
        """Fuse one metric depth map. ``mask``: pixels of the object (surface updates);
        ``carve_mask``: pixels that are known *not* to be the object (free-space updates where the
        observed depth is more than ``carve_margin`` (default ``trunc``) behind the voxel). Returns
        the number of voxels updated."""
        X = geo.transform(geo.inv_T(T_world_cam), self.centres)
        z = X[:, 2]
        uv = camera.project(X)
        u = np.rint(uv[:, 0])
        v = np.rint(uv[:, 1])
        h, w = depth.shape
        inside = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (v >= 0) & (u < w) & (v < h) & (z > 0.05)
        idx = np.nonzero(inside)[0]
        uu, vv = u[idx].astype(np.int64), v[idx].astype(np.int64)
        d = np.asarray(depth, dtype=np.float64)[vv, uu]
        valid_d = np.isfinite(d) & (d > 0)
        sdf = d - z[idx]
        in_mask = np.asarray(mask, dtype=bool)[vv, uu] & valid_d
        surf = in_mask & (sdf >= -self.trunc)
        value = np.clip(sdf / self.trunc, -1.0, 1.0)
        upd = idx[surf]
        self._update(upd, value[surf], 1.0)
        self.n_obs[upd[np.abs(sdf[surf]) < self.trunc]] += 1
        count = len(upd)
        if carve_mask is not None:
            margin = self.trunc if carve_margin is None else float(carve_margin)
            free = np.asarray(carve_mask, dtype=bool)[vv, uu] & valid_d & (sdf > margin) & ~in_mask
            self._update(idx[free], np.ones(int(free.sum())), free_weight)
            count += int(free.sum())
        return count

    def _update(self, idx: np.ndarray, value: np.ndarray, weight: float) -> None:
        if len(idx) == 0:
            return
        w_old = self.weight[idx]
        self.tsdf[idx] = (self.tsdf[idx] * w_old + value * weight) / (w_old + weight)
        self.weight[idx] = w_old + weight

    def grid(self, values: np.ndarray) -> np.ndarray:
        return values.reshape(tuple(int(v) for v in self.dims))

    def extract_surface(self, min_weight: float = 3.0, min_obs: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Zero-crossing points ``(N, 3)`` between face-adjacent voxels that both have at least
        ``min_weight`` total weight (one of them at least ``min_obs`` surface observations), and
        unit normals from the TSDF gradient (pointing to free space)."""
        tsdf = self.grid(self.tsdf)
        weight = self.grid(self.weight)
        n_obs = self.grid(self.n_obs)
        observed = weight >= min_weight
        grads = np.gradient(np.where(observed, tsdf, 0.0), self.voxel)
        pts, nrm = [], []
        for axis in range(3):
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[axis] = slice(0, -1)
            b[axis] = slice(1, None)
            ta, tb = tsdf[tuple(a)], tsdf[tuple(b)]
            ok = observed[tuple(a)] & observed[tuple(b)] & (np.sign(ta) != np.sign(tb)) & (ta != tb)
            ok &= (n_obs[tuple(a)] >= min_obs) | (n_obs[tuple(b)] >= min_obs)
            ii = np.argwhere(ok)
            if len(ii) == 0:
                continue
            fa = ta[ok]
            fb = tb[ok]
            frac = fa / (fa - fb)
            p = self.lo + (ii + 0.5) * self.voxel
            p[:, axis] += frac * self.voxel
            ib = ii.copy()
            ib[:, axis] += 1
            g = np.zeros((len(ii), 3))
            for c in range(3):
                g[:, c] = (1 - frac) * grads[c][tuple(ii.T)] + frac * grads[c][tuple(ib.T)]
            pts.append(p)
            nrm.append(g)
        if not pts:
            return np.zeros((0, 3)), np.zeros((0, 3))
        P = np.concatenate(pts)
        G = np.concatenate(nrm)
        norm = np.linalg.norm(G, axis=1)
        keep = norm > 1e-9
        return P[keep], G[keep] / norm[keep, None]


# =============================================================================================
# Completion and the object frame
# =============================================================================================


def extrude_to_plane(points: Any, normal: Any, offset: float, layers: int = 0, max_gap: float = np.inf) -> np.ndarray:
    """The points plus their projections onto the support plane (and ``layers`` evenly spaced
    copies in between), i.e. the object is assumed to stand on the plane with vertical walls
    under everything that was seen. Points further than ``max_gap`` above the plane are not
    extruded. Returns only the added points."""
    p = np.asarray(points, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    h = plane_distance(p, n, offset)
    sel = (h > 0) & (h <= max_gap)
    base = p[sel] - h[sel, None] * n
    added = [base]
    for k in range(1, layers + 1):
        f = k / (layers + 1)
        added.append(p[sel] - (f * h[sel])[:, None] * n)
    return np.concatenate(added) if added else np.zeros((0, 3))


def mirror_points(points: Any, centre: Any, axis: Any) -> np.ndarray:
    """Reflection of the points about the plane through ``centre`` with normal ``axis``."""
    p = np.asarray(points, dtype=np.float64)
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    d = (p - centre) @ a
    return p - 2 * d[:, None] * a


def footprint_axes(points: Any, up: Any = (0.0, 0.0, 1.0), method: str = "min_area_rect") -> np.ndarray:
    """Rotation ``(3, 3)`` whose columns are the object axes x, y, z: z = ``up``; x, y in the
    horizontal plane from the footprint (points projected along ``up``), x along the longer side.

    ``method``: ``"min_area_rect"`` (OpenCV's minimum-area bounding rectangle of the footprint:
    exact for boxes and well defined for square footprints, where PCA is not) or ``"pca"``."""
    import cv2

    p = np.asarray(points, dtype=np.float64)
    z = np.asarray(up, dtype=np.float64)
    z = z / np.linalg.norm(z)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(z, np.cross(helper, z))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(z, e1)
    xy = np.stack([(p - p.mean(axis=0)) @ e1, (p - p.mean(axis=0)) @ e2], axis=1)
    if method == "pca" or len(xy) < 3:
        cov = xy.T @ xy
        _, vecs = np.linalg.eigh(cov)
        d2 = vecs[:, 1]
    elif method == "min_area_rect":
        (_, _), (w, h), angle = cv2.minAreaRect(xy.astype(np.float32))
        th = np.radians(angle)
        d2 = np.array([np.cos(th), np.sin(th)])
        if h > w:  # OpenCV's width is along `angle`; take the longer side as x
            d2 = np.array([-np.sin(th), np.cos(th)])
    else:
        raise ValueError(f"unknown footprint axes method {method!r}")
    x = d2[0] * e1 + d2[1] * e2
    # Deterministic sign: x points away from the origin's side as little as possible -> towards +e1
    if x @ e1 < 0 or (abs(x @ e1) < 1e-9 and x @ e2 < 0):
        x = -x
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)
