"""Open3D surface reconstruction for the ``objects`` step (``objects.mesh_backend = "open3d"``).

API used (Open3D >= 0.17, ``open3d.geometry.TriangleMesh``):
``create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False, n_threads=-1)
-> (mesh, densities)`` (the point cloud needs normals), ``create_from_point_cloud_alpha_shape(pcd,
alpha)``, ``remove_vertices_by_mask``, ``remove_degenerate_triangles``,
``remove_unreferenced_vertices``, ``orient_triangles``, ``is_watertight``.

The step keeps the result only when it is a closed, oriented manifold (checked with
``core.hull.is_closed_manifold``) and otherwise falls back to the convex hull. Written, not run on
the development Mac (no open3d there).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..errors import require

DEFAULTS = {"method": "poisson", "poisson_depth": 8, "density_quantile": 0.05, "alpha": 0.02}


def reconstruct(points: np.ndarray, normals: np.ndarray, ocfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """``(V, F)`` from oriented points (world coordinates)."""
    o3d = require("open3d", "open3d", "objects.mesh_backend = open3d")
    p = {**DEFAULTS, **dict(ocfg or {})}
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    if p["method"] == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(p["poisson_depth"]))
        dens = np.asarray(densities)
        if len(dens) and float(p["density_quantile"]) > 0:
            mesh.remove_vertices_by_mask(dens < np.quantile(dens, float(p["density_quantile"])))
    elif p["method"] == "alpha":
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, float(p["alpha"]))
    else:
        raise ValueError(f"unknown objects.open3d.method {p['method']!r} (poisson or alpha)")
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.orient_triangles()
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.triangles, dtype=np.int64)
    if len(F) and _signed_volume(V, F) < 0:
        F = F[:, ::-1].copy()
    return V, F


def _signed_volume(V: np.ndarray, F: np.ndarray) -> float:
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
