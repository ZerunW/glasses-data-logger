"""CoACD approximate convex decomposition for the ``objects`` step
(``objects.collision_backend = "coacd"``).

API used (``pip install coacd``; https://github.com/SarahWeiii/CoACD, ``python/package/__init__.py``):
``coacd.Mesh(vertices (N,3) double, indices (M,3) int32)``, ``coacd.run_coacd(mesh, threshold=0.05,
max_convex_hull=-1, preprocess_mode="auto", preprocess_resolution=50, resolution=2000,
mcts_nodes=20, mcts_iterations=150, mcts_max_depth=3, pca=False, merge=True, ..., seed=0)`` returns
a list of ``[vertices, indices]`` pairs; ``coacd.set_log_level("error")``. Every part is
re-hulled here so that each written ``part_*.obj`` is exactly convex and outward oriented.
Written, not run on the development Mac (no coacd there).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..core.hull import hull_mesh
from ..errors import require

DEFAULTS = {"threshold": 0.05, "max_convex_hull": -1, "preprocess_mode": "auto", "seed": 0}


def decompose(vertices: np.ndarray, faces: np.ndarray, ccfg: Mapping[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convex parts ``[(V, F), ...]`` of a closed triangle mesh (object frame)."""
    coacd = require("coacd", "coacd", "objects.collision_backend = coacd")
    p = {**DEFAULTS, **dict(ccfg or {})}
    if hasattr(coacd, "set_log_level"):
        coacd.set_log_level("error")
    mesh = coacd.Mesh(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32))
    parts = coacd.run_coacd(mesh, threshold=float(p["threshold"]), max_convex_hull=int(p["max_convex_hull"]),
                            preprocess_mode=str(p["preprocess_mode"]), seed=int(p["seed"]))
    out = []
    for v, _ in parts:
        v = np.asarray(v, dtype=np.float64)
        if len(v) >= 4:
            try:
                out.append(hull_mesh(v))
            except ValueError:
                continue  # flat sliver
    if not out:
        raise RuntimeError("CoACD returned no usable part")
    return out
