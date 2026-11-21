"""Triangle meshes in numpy: 3-D convex hull, volume / centroid / inertia, sampling, OBJ files.

* :func:`convex_hull_3d` - Quickhull (deterministic, numpy only). Returns a closed, outward-oriented
  triangle mesh made of input points; coplanar facets are triangulated. (Drafted by ChatGPT through
  the owner's relay from a written specification, reviewed and tested here: see
  ``tests/test_core_geometry3d.py``.)
* :func:`mesh_volume_centroid`, :func:`mesh_inertia` - mass properties of a closed mesh (signed
  tetrahedra; inertia about the centroid for unit density).
* :func:`sample_surface` - area-uniform points and face normals on a mesh.
* :func:`write_obj` / :func:`read_obj` - Wavefront OBJ (``v`` / ``f`` lines, 1-based indices).
* :func:`hull_mesh` - hull of a point cloud with a tolerance suited to noisy data.
"""

from __future__ import annotations

import heapq
import math
import os
from pathlib import Path

import numpy as np


class _Face:
    """One outward-facing triangle and its neighbors across directed edges."""

    __slots__ = ("tri", "adj", "nx", "ny", "nz", "offset", "outside")

    def __init__(self, tri: tuple[int, int, int], xyz: np.ndarray) -> None:
        self.tri = tri
        self.adj = [-1, -1, -1]
        a, b, c = tri
        ax, ay, az = xyz[a]
        bx, by, bz = xyz[b]
        cx, cy, cz = xyz[c]
        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz - az
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        size = math.sqrt(nx * nx + ny * ny + nz * nz)
        if not size > 0.0:
            raise ValueError("A zero-area hull triangle was encountered")
        self.nx, self.ny, self.nz = nx / size, ny / size, nz / size
        self.offset = self.nx * ax + self.ny * ay + self.nz * az
        self.outside = np.empty(0, dtype=np.int64)

    def distance(self, xyz: np.ndarray) -> float:
        """Signed distance to this face for a single centered point."""
        return (
            self.nx * xyz[0]
            + self.ny * xyz[1]
            + self.nz * xyz[2]
            - self.offset
        )


def _initial_tetrahedron(xyz: np.ndarray, eps: float) -> tuple[int, int, int, int]:
    """Construct a wide, deterministic nondegenerate tetrahedron."""
    extrema = np.unique(
        np.concatenate((np.argmin(xyz, axis=0), np.argmax(xyz, axis=0)))
    )
    p = xyz[extrema]
    delta = p[:, None, :] - p[None, :, :]
    distance2 = np.einsum("ijk,ijk->ij", delta, delta)
    i, j = np.unravel_index(int(np.argmax(distance2)), distance2.shape)
    a, b = int(extrema[i]), int(extrema[j])
    length2 = float(distance2[i, j])
    if length2 <= eps * eps:
        raise ValueError("Points are collinear within eps")

    axis = xyz[b] - xyz[a]
    relative = xyz - xyz[a]
    cross = np.cross(relative, axis)
    heights2 = np.einsum("ij,ij->i", cross, cross) / length2
    c = int(np.argmax(heights2))
    if heights2[c] <= eps * eps:
        raise ValueError("Points are collinear within eps")

    normal = np.cross(axis, xyz[c] - xyz[a])
    normal /= np.linalg.norm(normal)
    distances = np.abs(relative @ normal)
    d = int(np.argmax(distances))
    if distances[d] <= eps:
        raise ValueError("Points are coplanar within eps")
    return a, b, c, d


def _assign(
    candidates: np.ndarray,
    new_ids: list[int],
    faces: list[_Face | None],
    xyz: np.ndarray,
    eps: float,
    heap: list[tuple[float, int]],
) -> None:
    """Assign each candidate to at most one new face, maximizing distance."""
    if len(candidates) == 0:
        return
    new_faces = [faces[fid] for fid in new_ids]
    # All entries are non-None: faces were just created.
    normal = np.array([(f.nx, f.ny, f.nz) for f in new_faces], dtype=np.float64)
    offsets = np.array([f.offset for f in new_faces], dtype=np.float64)
    distances = xyz[candidates] @ normal.T - offsets
    owners = np.argmax(distances, axis=1)
    best = distances[np.arange(len(candidates)), owners]
    eligible = best > eps
    if not np.any(eligible):
        return
    candidates = candidates[eligible]
    owners = owners[eligible]
    best = best[eligible]

    for local_id, fid in enumerate(new_ids):
        keep = owners == local_id
        if not np.any(keep):
            continue
        face = faces[fid]
        face.outside = candidates[keep]
        farthest = float(np.max(best[keep]))
        heapq.heappush(heap, (-farthest, fid))


def convex_hull_3d(
    points: np.ndarray, eps: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the triangular convex hull of an ``(N, 3)`` point cloud.

    Return ``(vertices, faces)`` of dtypes float64 and int64. Vertices are
    unique input points (unaltered coordinates). All triangles face outward;
    every mesh edge belongs to two triangles with opposite orientations.
    Points on a hull facet or edge, to within ``eps``, are not added merely
    to triangulate it.

    ``eps`` is a *distance* in input units. By default it is the larger of
    ``1e-9 * bounding_box_diagonal`` and ``1e-12``. Nonfinite rows are ignored.
    Raise ValueError if fewer than four finite distinct points remain, or if
    the cloud is collinear or coplanar to within the given tolerance.
    """
    raw = np.asarray(points, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    valid = raw[np.isfinite(raw).all(axis=1)]
    if len(valid) < 4:
        raise ValueError("Need at least four finite points")
    original = np.unique(valid, axis=0)
    if len(original) < 4:
        raise ValueError("Need at least four distinct finite points")

    low = np.min(original, axis=0)
    high = np.max(original, axis=0)
    span = high - low
    diagonal = float(np.linalg.norm(span))
    if eps is None:
        tolerance = max(1e-9 * diagonal, 1e-12)
    else:
        tolerance = float(eps)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("eps must be a nonnegative finite distance")

    # Centering the working coordinates protects small clouds with a large
    # world-coordinate offset. Return original point values at the end.
    center = low + 0.5 * span
    xyz = original - center
    tetra = _initial_tetrahedron(xyz, tolerance)
    inside = np.mean(xyz[list(tetra)], axis=0)

    # Each initial face is oriented using a point strictly inside the tetra.
    face_triples = (
        (tetra[0], tetra[1], tetra[2]),
        (tetra[0], tetra[3], tetra[1]),
        (tetra[0], tetra[2], tetra[3]),
        (tetra[1], tetra[3], tetra[2]),
    )
    faces: list[_Face | None] = []
    for tri in face_triples:
        face = _Face(tri, xyz)
        if face.distance(inside) > 0.0:
            face = _Face((tri[0], tri[2], tri[1]), xyz)
        faces.append(face)

    # Directed edges must be paired with their reversed counterparts.
    edges: dict[tuple[int, int], tuple[int, int]] = {}
    for fid, face in enumerate(faces):
        t = face.tri
        for edge_index in range(3):
            u, v = t[edge_index], t[(edge_index + 1) % 3]
            reversed_edge = (v, u)
            partner = edges.pop(reversed_edge, None)
            if partner is None:
                edges[(u, v)] = (fid, edge_index)
            else:
                other_id, other_index = partner
                face.adj[edge_index] = other_id
                faces[other_id].adj[other_index] = fid
    if edges:
        raise ValueError("Could not orient the initial tetrahedron")

    heap: list[tuple[float, int]] = []
    _assign(np.arange(len(xyz), dtype=np.int64), list(range(4)), faces, xyz, tolerance, heap)

    while heap:
        _, seed = heapq.heappop(heap)
        seed_face = faces[seed]
        if seed_face is None or not len(seed_face.outside):
            continue
        # Each outside set was assigned with its signed distances, so the
        # farthest point is again chosen from that (possibly updated) face.
        ids = seed_face.outside
        fs = seed_face
        pxyz = xyz[ids]
        distances = pxyz @ np.array((fs.nx, fs.ny, fs.nz)) - fs.offset
        farthest_index = int(np.argmax(distances))
        if distances[farthest_index] <= tolerance:
            # Guard against a last-ulp change at the tolerance boundary.
            seed_face.outside = np.empty(0, dtype=np.int64)
            continue
        apex = int(ids[farthest_index])
        apex_point = xyz[apex]

        # Visit only faces reachable through the visible surface patch.
        visible = {seed}
        examined = {seed}
        todo = [seed]
        while todo:
            current = faces[todo.pop()]
            for nb in current.adj:
                if nb in examined:
                    continue
                examined.add(nb)
                other = faces[nb]
                if other.distance(apex_point) > tolerance:
                    visible.add(nb)
                    todo.append(nb)

        # Horizon edges inherit the orientation of the visible face. This is
        # opposite to the orientation in the still-existing adjacent face.
        horizon: list[tuple[int, int, int, int]] = []
        displaced: list[np.ndarray] = []
        for old_id in sorted(visible):
            old = faces[old_id]
            if len(old.outside):
                displaced.append(old.outside)
            t = old.tri
            for edge_index, nb in enumerate(old.adj):
                if nb not in visible:
                    a, b = t[edge_index], t[(edge_index + 1) % 3]
                    neighbor_edge_index = faces[nb].adj.index(old_id)
                    horizon.append((a, b, nb, neighbor_edge_index))

        # Boundary of a connected visible patch is one directed cycle.
        successor = {edge[0]: edge for edge in horizon}
        if len(successor) != len(horizon):
            raise ValueError("Non-manifold horizon (numerically unstable input)")
        ordered: list[tuple[int, int, int, int]] = []
        first = min(successor)
        current_vertex = first
        for _ in range(len(horizon)):
            edge = successor.pop(current_vertex, None)
            if edge is None:
                raise ValueError("Broken horizon (numerically unstable input)")
            ordered.append(edge)
            current_vertex = edge[1]
        if current_vertex != first or successor:
            raise ValueError("Disconnected horizon (numerically unstable input)")

        # Make new triangles before invalidating the old ones. For each new
        # triangle (u, v, apex), neighbor 0 is across (u, v), neighbor 1
        # across (v, apex), and neighbor 2 across (apex, u).
        new_ids: list[int] = []
        for u, v, nb, nb_edge in ordered:
            new_id = len(faces)
            new_ids.append(new_id)
            new_face = _Face((u, v, apex), xyz)
            new_face.adj[0] = nb
            faces.append(new_face)
            faces[nb].adj[nb_edge] = new_id
        count = len(new_ids)
        for k, fid in enumerate(new_ids):
            faces[fid].adj[1] = new_ids[(k + 1) % count]
            faces[fid].adj[2] = new_ids[(k - 1) % count]

        for old_id in visible:
            faces[old_id] = None

        if displaced:
            candidates = np.concatenate(displaced)
            candidates = candidates[candidates != apex]
            _assign(candidates, new_ids, faces, xyz, tolerance, heap)

    output_faces = np.asarray(
        [f.tri for f in faces if f is not None], dtype=np.int64
    ).reshape(-1, 3)
    used = np.unique(output_faces)
    output_vertices = original[used].copy()
    output_faces = np.searchsorted(used, output_faces).astype(np.int64)
    return output_vertices, output_faces


def mesh_volume_centroid(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[float, np.ndarray]:
    """Volume and centroid of a closed, outward-oriented triangle mesh.

    Signed tetrahedra use the mean vertex position as origin. This avoids
    catastrophic cancellation when the mesh has a large coordinate offset.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("Expected vertices (M, 3) and faces (F, 3)")
    if len(v) == 0 or len(f) == 0 or not np.isfinite(v).all():
        raise ValueError("Mesh must be nonempty and finite")
    if np.any(f < 0) or np.any(f >= len(v)):
        raise ValueError("Triangle index out of bounds")
    reference = np.mean(v, axis=0)
    local = v - reference
    a, b, c = local[f[:, 0]], local[f[:, 1]], local[f[:, 2]]
    six_volumes = np.einsum("ij,ij->i", a, np.cross(b, c))
    six_total = float(np.sum(six_volumes))
    if not six_total > 0.0:
        raise ValueError("Mesh must have positive outward-oriented volume")
    centroid = reference + np.sum(
        (a + b + c) * six_volumes[:, None], axis=0
    ) / (4.0 * six_total)
    return six_total / 6.0, centroid

def mesh_inertia(vertices: np.ndarray, faces: np.ndarray, density: float = 1.0) -> tuple[float, np.ndarray, np.ndarray]:
    """Mass, centre of mass (3,) and inertia tensor (3, 3) about the centre of mass of a closed,
    outward-oriented triangle mesh of uniform ``density`` (signed tetrahedra; Tonon 2004)."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    volume, com = mesh_volume_centroid(v, f)
    a, b, c = v[f[:, 0]] - com, v[f[:, 1]] - com, v[f[:, 2]] - com
    det = np.einsum("ij,ij->i", a, np.cross(b, c))  # 6 x signed volume of tetra (com, a, b, c)
    # Second moments  int x_i x_j dV over each tetrahedron with one vertex at the origin:
    # det / 120 * (sum_k p_k,i p_k,j + (sum_k p_k,i)(sum_k p_k,j)), p in {0, a, b, c}.
    s = a + b + c
    outer = (np.einsum("ni,nj->nij", a, a) + np.einsum("ni,nj->nij", b, b) + np.einsum("ni,nj->nij", c, c)
             + np.einsum("ni,nj->nij", s, s))
    C = np.einsum("n,nij->ij", det, outer) / 120.0 * density  # covariance-like integral
    inertia = np.trace(C) * np.eye(3) - C
    return float(volume * density), com, inertia


def mesh_extents(vertices: np.ndarray) -> np.ndarray:
    """Axis-aligned extents (3,) of the vertices (max - min)."""
    v = np.asarray(vertices, dtype=np.float64)
    return v.max(axis=0) - v.min(axis=0)


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit normals (F, 3) and areas (F,) of the triangles."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    norm = np.linalg.norm(n, axis=1)
    return n / np.maximum(norm, 1e-300)[:, None], 0.5 * norm


def sample_surface(vertices: np.ndarray, faces: np.ndarray, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``n`` points uniformly distributed over the mesh area and their face normals."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    normals, areas = face_normals(v, f)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(f), size=int(n), p=areas / areas.sum())
    r1, r2 = rng.random(int(n)), rng.random(int(n))
    s = np.sqrt(r1)
    w0, w1, w2 = 1 - s, s * (1 - r2), s * r2
    tri = v[f[pick]]
    points = w0[:, None] * tri[:, 0] + w1[:, None] * tri[:, 1] + w2[:, None] * tri[:, 2]
    return points, normals[pick]


def convex_signed_distance(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Signed distance of points to a *convex* closed mesh: the largest face-plane distance (exact
    inside, where it is negative; a lower bound of the true distance outside near edges/corners)."""
    v = np.asarray(vertices, dtype=np.float64)
    normals, _ = face_normals(v, faces)
    offsets = np.einsum("ij,ij->i", normals, v[np.asarray(faces)[:, 0]])
    return np.max(np.asarray(points, dtype=np.float64) @ normals.T - offsets, axis=1)


def hull_mesh(points: np.ndarray, rel_eps: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """Convex hull of a (noisy) point cloud with a tolerance of ``rel_eps`` x the bounding-box
    diagonal (merges nearly coplanar facets, avoids sliver triangles)."""
    p = np.asarray(points, dtype=np.float64)
    p = p[np.all(np.isfinite(p), axis=1)]
    if len(p) < 4:
        raise ValueError("need at least 4 points for a 3-D hull")
    diag = float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))
    return convex_hull_3d(p, eps=max(rel_eps * diag, 1e-12))


def write_obj(path: str | os.PathLike[str], vertices: np.ndarray, faces: np.ndarray, comment: str = "") -> Path:
    """Write a Wavefront OBJ (vertices in metres, 1-based triangle indices)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64) + 1
    lines = [f"# {line}" for line in comment.splitlines()] if comment else []
    lines += [f"v {x:.7f} {y:.7f} {z:.7f}" for x, y, z in v]
    lines += [f"f {a} {b} {c}" for a, b, c in f]
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)
    return path


def read_obj(path: str | os.PathLike[str]) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (V, 3) and triangles (F, 3, 0-based) of an OBJ file (polygons are fanned; texture /
    normal indices are ignored)."""
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append([float(x) for x in parts[1:4]])
        elif parts[0] == "f":
            idx = [int(p.split("/")[0]) for p in parts[1:]]
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
            for k in range(1, len(idx) - 1):
                faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3), np.asarray(faces, dtype=np.int64).reshape(-1, 3)


def is_closed_manifold(faces: np.ndarray) -> bool:
    """True when every directed edge appears once and its reverse once (closed, oriented 2-manifold)."""
    f = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    directed = {tuple(e) for e in edges.tolist()}
    if len(directed) != len(edges):
        return False
    return all((b, a) in directed for a, b in directed)


def fibonacci_directions(n: int) -> np.ndarray:
    """``n`` nearly uniform unit vectors on the sphere."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    phi = np.pi * (1 + 5**0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def halfspace_intersection(directions: np.ndarray, supports: np.ndarray, inside: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convex polytope ``{x : d_j . x <= h_j}`` as a closed mesh ``(V, F)``, by point-plane duality
    about a strictly interior point ``inside`` (hull of ``d_j / (h_j - d_j . inside)``)."""
    D = np.asarray(directions, dtype=np.float64)
    h = np.asarray(supports, dtype=np.float64)
    c = np.asarray(inside, dtype=np.float64)
    b = h - D @ c
    if np.any(b <= 0):
        raise ValueError("the interior point is not strictly inside every half-space")
    dual = D / b[:, None]
    Vd, Fd = convex_hull_3d(dual, eps=1e-9 * float(np.linalg.norm(dual.max(0) - dual.min(0))))
    n, _ = face_normals(Vd, Fd)
    o = np.einsum("ij,ij->i", n, Vd[Fd[:, 0]])
    verts = c + n / o[:, None]
    # coplanar dual triangles give the same primal vertex
    key = np.round(verts / max(1e-9, 1e-7 * float(np.abs(verts - c).max())))
    _, first = np.unique(key, axis=0, return_index=True)
    return convex_hull_3d(verts[np.sort(first)])


def support_polytope(
    points: np.ndarray,
    n_dirs: int = 256,
    extra_dirs: np.ndarray | None = None,
    trim: int = 10,
    exact_dirs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """A convex hull that ignores the most extreme points (robust to noisy surface points).

    For every direction ``d`` (``n_dirs`` Fibonacci directions plus ``extra_dirs``) the support
    ``h(d)`` is the ``trim``-th largest projection ``d . p`` instead of the largest one, and the
    shape is the intersection of the half-spaces ``d . x <= h(d)``. The plain hull of noisy points
    is inflated by the noise extremes (about +2.5 sigma on every side); trimming a few points per
    direction removes most of that while costing little at true corners (several points lie near
    a corner). Directions in ``exact_dirs`` (e.g. the support plane's downward normal, where the
    points are exact) use the untrimmed maximum. Returns ``(V, F, info)``.
    """
    P = np.asarray(points, dtype=np.float64)
    D = fibonacci_directions(n_dirs)
    extras = [np.asarray(e, dtype=np.float64).reshape(-1, 3) for e in (extra_dirs, exact_dirs) if e is not None]
    n_exact = 0 if exact_dirs is None else len(np.asarray(exact_dirs).reshape(-1, 3))
    if extras:
        E = np.concatenate(extras)
        D = np.concatenate([D, E / np.linalg.norm(E, axis=1, keepdims=True)])
    proj = P @ D.T  # (n_points, n_dirs)
    k = int(min(max(trim, 0), len(P) - 1))
    h = -np.partition(-proj, k, axis=0)[k] if k > 0 else proj.max(axis=0)
    if n_exact:
        h[-n_exact:] = proj[:, -n_exact:].max(axis=0)
    inside = np.median(P, axis=0)
    margin = h - D @ inside
    if np.any(margin <= 1e-6):
        # the median point is outside some half-space (very thin shape): use the hull instead
        V, F = hull_mesh(P)
        return V, F, {"method": "hull (fallback)", "trim": k}
    V, F = halfspace_intersection(D, h, inside)
    return V, F, {"method": "trimmed support polytope", "trim": k, "dirs": len(D)}
