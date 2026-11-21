"""Nearest neighbours of 3-D points (numpy + OpenCV's FLANN; no scipy).

:class:`KNN` indexes a reference set once and answers k-nearest-neighbour queries:

* ``method="flann"``: OpenCV's randomized kd-trees (``cv2.flann_Index``, approximate; with the
  default 4 trees and 128 checks the recall of the exact neighbours is > 99.9 % on 3-D points),
* ``method="brute"``: exact, chunked distance matrices (fine up to ~1e8 pairs),
* ``method="auto"`` (default): brute force when ``len(ref) * len(query)`` is small, else FLANN.

Distances are Euclidean (not squared). Queries return ``(dist, idx)`` of shape ``(M, k)`` sorted by
distance. ``k`` larger than the reference set is clipped.
"""

from __future__ import annotations

from typing import Any

import numpy as np

BRUTE_MAX_PAIRS = 2_000_000  # "auto" uses exact brute force below this many reference x query pairs


class KNN:
    """k-nearest-neighbour index over ``ref`` ``(N, 3)`` (any dimension D works)."""

    def __init__(self, ref: Any, method: str = "auto", trees: int = 4, checks: int = 128) -> None:
        self.ref = np.ascontiguousarray(np.asarray(ref, dtype=np.float64).reshape(len(ref), -1))
        if method not in ("auto", "flann", "brute"):
            raise ValueError(f"unknown kNN method {method!r}")
        self.method = method
        self.trees = int(trees)
        self.checks = int(checks)
        self._flann = None
        self._ref32: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.ref)

    def _use_flann(self, n_query: int) -> bool:
        if self.method == "flann":
            return True
        if self.method == "brute":
            return False
        return len(self.ref) * max(n_query, 1) > BRUTE_MAX_PAIRS and len(self.ref) > 64

    def query(self, query: Any, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """``(dist (M, k), idx (M, k))`` of the ``k`` nearest reference points of each query."""
        q = np.asarray(query, dtype=np.float64).reshape(-1, self.ref.shape[1])
        n = len(self.ref)
        if n == 0:
            raise ValueError("kNN reference set is empty")
        k = max(1, min(int(k), n))
        if len(q) == 0:
            return np.zeros((0, k)), np.zeros((0, k), dtype=np.int64)
        if self._use_flann(len(q)):
            return self._query_flann(q, k)
        return self._query_brute(q, k)

    def _query_flann(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        if self._flann is None:
            # FLANN works in float32; centre the data so large world offsets keep their precision.
            self._centre = self.ref.mean(axis=0)
            self._ref32 = np.ascontiguousarray((self.ref - self._centre).astype(np.float32))
            self._flann = cv2.flann_Index(self._ref32, dict(algorithm=1, trees=self.trees))
        q32 = np.ascontiguousarray((q - self._centre).astype(np.float32))
        idx, _ = self._flann.knnSearch(q32, k, params=dict(checks=self.checks))
        idx = np.asarray(idx, dtype=np.int64).reshape(len(q), k)
        idx = np.clip(idx, 0, len(self.ref) - 1)  # FLANN pads with -1 when it finds fewer than k
        # Exact distances in float64 from the returned indices, re-sorted.
        dist = np.linalg.norm(self.ref[idx] - q[:, None, :], axis=-1)
        order = np.argsort(dist, axis=1, kind="stable")
        return np.take_along_axis(dist, order, 1), np.take_along_axis(idx, order, 1)

    def _query_brute(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.ref)
        chunk = max(1, int(BRUTE_MAX_PAIRS // max(n, 1)))
        dist_out = np.empty((len(q), k))
        idx_out = np.empty((len(q), k), dtype=np.int64)
        ref_sq = np.sum(self.ref**2, axis=1)
        for start in range(0, len(q), chunk):
            qq = q[start:start + chunk]
            d2 = np.sum(qq**2, axis=1)[:, None] + ref_sq[None, :] - 2.0 * qq @ self.ref.T
            np.maximum(d2, 0.0, out=d2)
            if k < n:
                part = np.argpartition(d2, k - 1, axis=1)[:, :k]
            else:
                part = np.broadcast_to(np.arange(n), (len(qq), n)).copy()
            dpart = np.take_along_axis(d2, part, 1)
            order = np.argsort(dpart, axis=1, kind="stable")
            idx = np.take_along_axis(part, order, 1)
            # exact distances (the expanded form above loses precision for close points)
            dist = np.linalg.norm(self.ref[idx] - qq[:, None, :], axis=-1)
            dist_out[start:start + chunk] = dist
            idx_out[start:start + chunk] = idx
        return dist_out, idx_out

    def nearest(self, query: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(dist (M,), idx (M,))`` of the single nearest reference point."""
        d, i = self.query(query, 1)
        return d[:, 0], i[:, 0]


def knn(ref: Any, query: Any, k: int = 1, method: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """One-shot :meth:`KNN.query`."""
    return KNN(ref, method=method).query(query, k)


def self_knn(points: Any, k: int, method: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` nearest *other* points of every point: ``(dist (N, k), idx (N, k))`` (the point
    itself, at distance 0, is removed; duplicates of it may remain)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        raise ValueError("need at least two points")
    k = min(int(k), len(pts) - 1)
    d, i = KNN(pts, method=method).query(pts, k + 1)
    own = i == np.arange(len(pts))[:, None]
    # Drop the self match (usually column 0); when it is missing (duplicates), drop the last column.
    drop = np.where(own.any(axis=1), np.argmax(own, axis=1), k)
    keep = np.ones_like(own)
    keep[np.arange(len(pts)), drop] = False
    return d[keep].reshape(len(pts), k), i[keep].reshape(len(pts), k)
