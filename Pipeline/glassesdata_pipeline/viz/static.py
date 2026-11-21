"""Static matplotlib figures of an episode (``derived/viz/*.png``); numpy + matplotlib only.

Works with whatever derived files exist (every figure skips the layers it has no data for) and
overlays the ground truth (dashed, same colour as the entity) when ``gt/`` exists:

* ``topview.png`` - world xy: table, camera path with heading ticks and time labels, both wrists,
  object centroids with their first/last footprints.
* ``snapshots.png`` - 3D views at 4 frames (start, grasp, highest lift, end when the objects report
  grasps; else evenly spaced): camera frustum, hand skeletons, arms, objects, table.
* ``timelines.png`` - wrist height above the table, hand tracking state (observed / filled),
  grasp flags, object height above the table, elbow confidence, camera height; retargeting error
  and simulated-object deviation when ``retarget``/``physics`` ran.
* ``overlay.png`` - the world reconstruction projected back into 6 video frames with the camera
  trajectory (a direct check of camera poses, hands, objects and table against the video).

Colours: left hand blue, right hand orange, objects aqua/violet/magenta/... in a fixed order
(``sim.mjcf.HAND_HEX`` / ``OBJECT_HEX``), the camera in neutral ink.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .. import geometry as geo
from .. import skeleton as sk
from ..episode import Episode
from ..sim import mjcf
from ..sim.data import EpisodeData, load_episode_data
from ..sim.replay import render_meshes

log = logging.getLogger(__name__)

FIGURES = ("topview", "snapshots", "timelines", "overlay")
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
TABLE_FILL = "#ebe5dc"
TABLE_EDGE = "#b9ae9f"
CAMERA = "#3b3a37"
SIDES = ("left", "right")
STYLE = {
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "axes.edgecolor": "#cfcdc8",
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.facecolor": SURFACE, "figure.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": 7.5, "lines.linewidth": 1.6, "axes.titlelocation": "left",
}


def object_hex(k: int) -> str:
    return mjcf.OBJECT_HEX[k % len(mjcf.OBJECT_HEX)]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _segments_where(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[(start, stop)]`` runs of True (stop exclusive)."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _table_z(data: EpisodeData) -> float | None:
    if data.table_top_z is not None:
        return data.table_top_z
    if data.gt is not None and data.gt.table_top_z is not None:
        return data.gt.table_top_z
    return None


def _object_footprint(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Convex hull (xy) of an object's vertices at pose T."""
    import cv2

    P = geo.transform(T, V)[:, :2].astype(np.float32)
    hull = cv2.convexHull(P).reshape(-1, 2)
    return np.vstack([hull, hull[:1]])


def _event_frames(data: EpisodeData, n: int = 4) -> list[tuple[int, str]]:
    """Frames for the snapshots: start / grasp / highest lift / end when grasps are known."""
    T = data.T
    ok = np.ones(T, bool)
    if data.T_world_cam is not None:
        ok = np.all(np.isfinite(data.T_world_cam), axis=(-1, -2))
    valid = np.flatnonzero(ok)
    if not len(valid):
        valid = np.arange(T)
    picks: list[tuple[int, str]] = [(int(valid[0]), "start")]
    grasp = None
    for obj in data.objects:
        g = np.flatnonzero(obj.grasped.any(axis=1))
        if len(g):
            grasp = (obj, g)
            break
    if grasp is not None:
        obj, g = grasp
        z = obj.T_world_obj[:, 2, 3]
        zg = np.where(np.isin(np.arange(T), g), z, -np.inf)
        picks += [(int(g[0]), f"grasp {obj.id}"), (int(np.argmax(zg)), f"highest {obj.id}"), (int(valid[-1]), "end")]
    else:
        idx = valid[np.linspace(0, len(valid) - 1, n).round().astype(int)]
        picks = [(int(f), "") for f in idx]
    seen: set[int] = set()
    out = []
    for f, label in picks:
        if f not in seen:
            seen.add(f)
            out.append((f, label))
    return out[:n]


# =============================================================================================
# Top view
# =============================================================================================


def fig_topview(data: EpisodeData, out: Path) -> Path | None:
    plt = _plt()
    import matplotlib

    with matplotlib.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 6.4))
        drew = False
        handles = []
        if data.table_polygon is not None:
            poly = np.vstack([data.table_polygon, data.table_polygon[:1]])
            ax.fill(poly[:, 0], poly[:, 1], color=TABLE_FILL, zorder=0)
            ax.plot(poly[:, 0], poly[:, 1], color=TABLE_EDGE, lw=1.0, zorder=1)
            handles.append(matplotlib.patches.Patch(color=TABLE_FILL, label="table (objects/table.json)"))
            drew = True
        if data.gt is not None and data.gt.table_polygon is not None:
            poly = np.vstack([data.gt.table_polygon, data.gt.table_polygon[:1]])
            ax.plot(poly[:, 0], poly[:, 1], color=TABLE_EDGE, lw=0.8, ls="--", zorder=1)
        # camera
        if data.gt is not None:
            g = data.gt.T_world_cam[:, :3, 3]
            ax.plot(g[:, 0], g[:, 1], color=MUTED, lw=1.0, ls="--", zorder=2)
        if data.T_world_cam is not None:
            ok = np.all(np.isfinite(data.T_world_cam), axis=(-1, -2))
            if data.cam_valid is not None:
                ok &= data.cam_valid
            P = data.T_world_cam[:, :3, 3]
            for a, b in _segments_where(ok):
                ax.plot(P[a:b, 0], P[a:b, 1], color=CAMERA, lw=1.8, zorder=3, solid_capstyle="round")
            step = max(1, int(round(data.fps)))
            idx = np.flatnonzero(ok)[::step]
            if len(idx):
                fwd = data.T_world_cam[idx, :3, 2]
                fwd = fwd[:, :2] / np.maximum(np.linalg.norm(fwd[:, :2], axis=1, keepdims=True), 1e-9)
                ax.quiver(P[idx, 0], P[idx, 1], fwd[:, 0], fwd[:, 1], color=CAMERA, angles="xy", scale_units="xy",
                          scale=1 / 0.08, width=0.003, headwidth=3.5, zorder=4, alpha=0.7)
                first, last = np.flatnonzero(ok)[[0, -1]]
                ax.plot(*P[first, :2], marker="o", ms=6, color=CAMERA, mec="white", mew=1.0, zorder=7)
                ax.plot(*P[last, :2], marker="s", ms=6, color=CAMERA, mec="white", mew=1.0, zorder=7)
                ax.annotate(f"start {data.t[first]:.1f}s", P[first, :2], textcoords="offset points", xytext=(-8, -12),
                            fontsize=7, color=INK2, ha="right")
                ax.annotate(f"end {data.t[last]:.1f}s", P[last, :2], textcoords="offset points", xytext=(-8, 8),
                            fontsize=7, color=INK2, ha="right")
            handles.append(matplotlib.lines.Line2D([], [], color=CAMERA, lw=1.8, marker="o", ms=4,
                                                   label="camera path (o start, \u25a0 end, ticks: heading every 1 s)"))
            drew = True
        # wrists
        for slot, side in enumerate(SIDES):
            color = mjcf.HAND_HEX[side]
            if data.gt is not None:
                g = data.gt.joints[:, slot, 0]
                ax.plot(g[:, 0], g[:, 1], color=color, lw=0.9, ls="--", alpha=0.7, zorder=2)
            if data.joints is not None:
                W = data.joints[:, slot, 0]
                ok = np.all(np.isfinite(W), axis=-1)
                if np.any(ok):
                    for a, b in _segments_where(ok):
                        ax.plot(W[a:b, 0], W[a:b, 1], color=color, lw=1.6, zorder=5)
                    handles.append(matplotlib.lines.Line2D([], [], color=color, lw=1.6, label=f"{side} wrist"))
                    drew = True
        # objects
        for k, obj in enumerate(data.objects):
            color = object_hex(k)
            if data.gt is not None and k < data.gt.T_world_obj.shape[1]:
                gk = _match_gt_object(data, obj)
                if gk is not None:
                    g = data.gt.T_world_obj[:, gk, :3, 3]
                    ax.plot(g[:, 0], g[:, 1], color=color, lw=0.9, ls="--", zorder=2)
            P = obj.T_world_obj[:, :3, 3]
            ok = obj.valid
            if not np.any(ok):
                continue
            V, _ = render_meshes(obj)
            first, last = np.flatnonzero(ok)[[0, -1]]
            fp = _object_footprint(V, obj.T_world_obj[first])
            ax.fill(fp[:, 0], fp[:, 1], color=color, alpha=0.25, zorder=3, lw=0)
            lp = _object_footprint(V, obj.T_world_obj[last])
            ax.plot(lp[:, 0], lp[:, 1], color=color, lw=1.2, zorder=6)
            for a, b in _segments_where(ok):
                ax.plot(P[a:b, 0], P[a:b, 1], color=color, lw=1.4, zorder=6)
            ax.annotate(obj.id, P[last, :2], textcoords="offset points", xytext=(6, -10), fontsize=7.5, color=INK)
            handles.append(matplotlib.lines.Line2D([], [], color=color, lw=1.4, label=f"{obj.id} centre (fill: first pose, outline: last)"))
            drew = True
        if not drew:
            plt.close(fig)
            return None
        if data.gt is not None:
            handles.append(matplotlib.lines.Line2D([], [], color=MUTED, lw=0.9, ls="--", label="ground truth (dashed)"))
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("world x (m)  - first camera heading")
        ax.set_ylabel("world y (m)")
        dur = data.t[-1] - data.t[0] if data.T > 1 else 0.0
        ax.set_title(f"{data.name} · top view (world xy, z up) · {dur:.1f} s, {data.T} frames")
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        fig.tight_layout()
        fig.savefig(out, dpi=140, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
    return out


def _match_gt_object(data: EpisodeData, obj: Any) -> int | None:
    """GT object index: same id, else the nearest at the object's first valid frame."""
    gt = data.gt
    if gt is None:
        return None
    if obj.id in gt.object_ids:
        return gt.object_ids.index(obj.id)
    f = obj.first_valid
    if f is None:
        return None
    d = np.linalg.norm(gt.T_world_obj[f, :, :3, 3] - obj.T_world_obj[f, :3, 3], axis=-1)
    return int(np.nanargmin(d)) if np.any(np.isfinite(d)) else None


# =============================================================================================
# 3D snapshots
# =============================================================================================


def _clip_polygon(poly: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clipping of a 2D polygon to the box ``[lo, hi]``."""
    out = [np.asarray(p, dtype=np.float64) for p in poly]
    for axis in (0, 1):
        for bound, keep in ((lo[axis], lambda p, a=axis, b=lo[axis]: p[a] >= b), (hi[axis], lambda p, a=axis, b=hi[axis]: p[a] <= b)):
            src, out = out, []
            for i in range(len(src)):
                cur, prev = src[i], src[i - 1]
                if keep(cur):
                    if not keep(prev):
                        s = (bound - prev[axis]) / (cur[axis] - prev[axis])
                        out.append(prev + s * (cur - prev))
                    out.append(cur)
                elif keep(prev):
                    s = (bound - prev[axis]) / (cur[axis] - prev[axis])
                    out.append(prev + s * (cur - prev))
            if not out:
                return np.zeros((0, 2))
    return np.array(out)


def _frustum(camera: geo.Camera, T: np.ndarray, depth: float = 0.12) -> list[np.ndarray]:
    w, h = camera.width, camera.height
    corners = np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]])
    rays = np.column_stack([(corners[:, 0] - camera.cx) / camera.fx, (corners[:, 1] - camera.cy) / camera.fy, np.ones(4)])
    pts = geo.transform(T, rays * depth)
    c = T[:3, 3]
    lines = [np.vstack([c, p]) for p in pts]
    lines.append(np.vstack([pts, pts[:1]]))
    up = geo.transform(T, np.array([[0.0, -0.5 * h / camera.fy * depth * 1.25, depth]]))[0]  # image-up marker
    lines.append(np.vstack([(pts[0] + pts[1]) / 2, up]))
    return lines


def fig_snapshots(data: EpisodeData, out: Path) -> Path | None:
    if data.T_world_cam is None and data.joints is None and not data.objects:
        return None
    plt = _plt()
    import matplotlib
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    frames = _event_frames(data)
    with matplotlib.rc_context({**STYLE, "axes.grid": False}):
        fig = plt.figure(figsize=(3.4 * len(frames), 4.0))
        table_z = _table_z(data)
        for i, (f, label) in enumerate(frames):
            ax = fig.add_subplot(1, len(frames), i + 1, projection="3d")
            ax.computed_zorder = False  # fixed layering: table < objects < arms < hands < camera
            pts: list[np.ndarray] = []
            for k, obj in enumerate(data.objects):
                T = obj.held_poses()[f]
                if not np.all(np.isfinite(T)):
                    continue
                V, F = render_meshes(obj, max_triangles=400)
                P = geo.transform(T, V)
                ax.add_collection3d(Poly3DCollection(P[F], facecolor=object_hex(k), edgecolor=object_hex(k), alpha=0.35,
                                                     lw=0.3, zorder=1))
                pts.append(P)
                if data.gt is not None:
                    gk = _match_gt_object(data, obj)
                    if gk is not None and gk < len(data.gt.object_sizes):
                        Vg, Fg = mjcf.box_mesh(data.gt.object_sizes[gk])
                        Pg = geo.transform(data.gt.T_world_obj[f, gk], Vg)
                        ax.add_collection3d(Poly3DCollection(Pg[Fg], facecolor="none", edgecolor=MUTED, lw=0.4,
                                                             linestyle="--", zorder=1))
            for slot, side in enumerate(SIDES):
                color = mjcf.HAND_HEX[side]
                if data.gt is not None:
                    G = data.gt.joints[f, slot]
                    for a, b in sk.BONES:
                        ax.plot(*np.stack([G[a], G[b]]).T, color=MUTED, lw=0.6, ls="--", zorder=3)
                if data.joints is not None:
                    J = data.joints[f, slot]
                    if np.all(np.isfinite(J)):
                        for a, b in sk.BONES:
                            ax.plot(*np.stack([J[a], J[b]]).T, color=color, lw=1.4, zorder=4)
                        ax.scatter(*J.T, color=color, s=5, depthshade=False, zorder=4)
                        pts.append(J)
                if data.shoulder is not None and data.elbow is not None:
                    wrist = data.wrist[f, slot] if data.wrist is not None else (
                        data.joints[f, slot, 0] if data.joints is not None else np.full(3, np.nan))
                    arm = np.stack([data.shoulder[f, slot], data.elbow[f, slot], wrist])
                    if np.all(np.isfinite(arm)):
                        ax.plot(*arm.T, color="#9a9994", lw=3.0, solid_capstyle="round", zorder=2)
                        ax.scatter(*arm.T, color="#7d7c78", s=10, depthshade=False, zorder=2)
                        pts.append(arm)
            if data.T_world_cam is not None and np.all(np.isfinite(data.T_world_cam[f])) and data.camera is not None:
                for line in _frustum(data.camera, data.T_world_cam[f]):
                    ax.plot(*line.T, color=CAMERA, lw=0.9, zorder=5)
                pts.append(data.T_world_cam[f, :3, 3][None])
            if not pts and data.table_polygon is None:
                ax.set_axis_off()
                continue
            allp = np.concatenate(pts) if pts else np.column_stack([data.table_polygon, np.zeros(len(data.table_polygon))])
            allp = allp[np.all(np.isfinite(allp), axis=1)]
            lo, hi = allp.min(axis=0), allp.max(axis=0)
            if table_z is not None:
                lo[2], hi[2] = min(lo[2], table_z), max(hi[2], table_z)
            centre, half = (lo + hi) / 2, max(float(np.max(hi - lo)) / 2, 0.15) * 1.08
            if data.table_polygon is not None and table_z is not None:
                clipped = _clip_polygon(data.table_polygon, centre[:2] - half, centre[:2] + half)
                if len(clipped) >= 3:
                    poly = np.column_stack([clipped, np.full(len(clipped), table_z)])
                    ax.add_collection3d(Poly3DCollection([poly], facecolor=TABLE_FILL, edgecolor=TABLE_EDGE, alpha=0.7,
                                                         lw=0.6, zorder=0))
            ax.set_xlim(centre[0] - half, centre[0] + half)
            ax.set_ylim(centre[1] - half, centre[1] + half)
            ax.set_zlim(centre[2] - half, centre[2] + half)
            ax.set_box_aspect((1, 1, 1))
            yaw = 0.0
            if data.T_world_cam is not None and np.all(np.isfinite(data.T_world_cam[f])):
                yaw = float(np.degrees(geo.yaw_of(data.T_world_cam[f, :3, :3])))
            ax.view_init(elev=24, azim=yaw + 215)
            ax.set_title(f"t = {data.t[f]:.2f} s  #{f}" + (f"\n{label}" if label else ""), fontsize=8, loc="center")
            ax.tick_params(labelsize=6, pad=0)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.set_facecolor((0.985, 0.985, 0.98, 1.0))
                axis.pane.set_edgecolor(GRID)
        handles = [matplotlib.lines.Line2D([], [], color=mjcf.HAND_HEX[s], lw=1.3, label=f"{s} hand") for s in SIDES]
        handles += [matplotlib.lines.Line2D([], [], color="#9a9994", lw=3, label="arm (shoulder-elbow-wrist)"),
                    matplotlib.lines.Line2D([], [], color=CAMERA, lw=0.9, label="camera frustum (12 cm)")]
        handles += [matplotlib.patches.Patch(color=object_hex(k), alpha=0.5, label=o.id) for k, o in enumerate(data.objects)]
        if data.gt is not None:
            handles.append(matplotlib.lines.Line2D([], [], color=MUTED, lw=0.6, ls="--", label="ground truth"))
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7))
        fig.suptitle(f"{data.name} · 3D snapshots (world, z up; table clipped to the view)", x=0.01, y=0.99, ha="left",
                     fontsize=9, color=INK)
        fig.subplots_adjust(left=0.0, right=1.0, top=0.84, bottom=0.12, wspace=0.0)
        fig.savefig(out, dpi=140)
        plt.close(fig)
    return out


# =============================================================================================
# Timelines
# =============================================================================================


def fig_timelines(data: EpisodeData, out: Path) -> Path | None:
    plt = _plt()
    import matplotlib

    t = data.t
    table_z = _table_z(data)
    ref = table_z if table_z is not None else 0.0
    ref_label = "above table" if table_z is not None else "world z"
    panels: list[tuple[str, float, Any]] = []

    def p_wrist(ax):
        for slot, side in enumerate(SIDES):
            color = mjcf.HAND_HEX[side]
            if data.gt is not None:
                ax.plot(t, data.gt.joints[:, slot, 0, 2] - ref, color=color, lw=0.9, ls="--", alpha=0.75)
            if data.joints is not None:
                ax.plot(t, data.joints[:, slot, 0, 2] - ref, color=color, lw=1.6, label=f"{side}")
        _panel(ax, f"Wrist height {ref_label}", "m")
        ax.legend(loc="upper right", ncol=2)

    def p_state(ax):
        rows = []
        for slot, side in enumerate(SIDES):
            y = 1 - slot
            color = mjcf.HAND_HEX[side]
            valid = np.all(np.isfinite(data.joints[:, slot]), axis=(-1, -2)) if data.joints is not None else np.zeros(data.T, bool)
            observed = data.observed[:, slot] if data.observed is not None else valid
            filled = (data.filled[:, slot] if data.filled is not None else np.zeros(data.T, bool)) & valid
            for a, b in _segments_where(observed & valid):
                ax.broken_barh([(t[a], t[min(b, data.T - 1)] - t[a] + 1 / data.fps)], (y - 0.32, 0.64), color=color)
            for a, b in _segments_where(filled & ~observed):
                ax.broken_barh([(t[a], t[min(b, data.T - 1)] - t[a] + 1 / data.fps)], (y - 0.32, 0.64), facecolor="none",
                               edgecolor=color, hatch="////", lw=0.6)
            rows.append((y, side))
        ax.set_yticks([y for y, _ in rows], [s for _, s in rows])
        ax.set_ylim(-0.6, 1.6)
        ax.grid(False)
        _panel(ax, "Hand tracking: solid = observed, hatched = filled by the world step", "")

    def p_grasp(ax):
        labels = []
        y = 0
        for k, obj in enumerate(data.objects):
            for slot, side in enumerate(SIDES):
                g = obj.grasped[:, slot]
                gk = _match_gt_object(data, obj) if data.gt is not None else None
                gt_g = data.gt.grasped[:, gk, slot] if gk is not None else None
                if not np.any(g) and (gt_g is None or not np.any(gt_g)):
                    continue
                for a, b in _segments_where(g):
                    ax.broken_barh([(t[a], t[min(b, data.T - 1)] - t[a] + 1 / data.fps)], (y - 0.3, 0.6), color=object_hex(k))
                if gt_g is not None:
                    for a, b in _segments_where(gt_g):
                        ax.broken_barh([(t[a], t[min(b, data.T - 1)] - t[a] + 1 / data.fps)], (y - 0.42, 0.84), facecolor="none",
                                       edgecolor=MUTED, ls="--", lw=0.8)
                labels.append((y, f"{obj.id} · {side}"))
                y += 1
        if not labels:
            ax.text(0.5, 0.5, "no grasp flags in objects/*/poses.npz", transform=ax.transAxes, ha="center", va="center",
                    color=MUTED)
            ax.set_yticks([])
        else:
            ax.set_yticks([y for y, _ in labels], [s for _, s in labels])
            ax.set_ylim(-0.7, len(labels) - 0.3)
        ax.grid(False)
        _panel(ax, "Grasp flags (objects/*/poses.npz)" + ("; dashed box = ground truth" if data.gt is not None else ""), "")

    def p_objects(ax):
        for k, obj in enumerate(data.objects):
            color = object_hex(k)
            gk = _match_gt_object(data, obj) if data.gt is not None else None
            if gk is not None:
                ax.plot(t, data.gt.T_world_obj[:, gk, 2, 3] - ref, color=color, lw=0.9, ls="--")
            z = np.where(obj.valid, obj.T_world_obj[:, 2, 3], np.nan)
            ax.plot(t, z - ref, color=color, lw=1.6, label=obj.id)
            if data.physics_traj is not None and "T_world_obj" in data.physics_traj:
                ids = [str(v) for v in np.atleast_1d(data.physics_traj.get("object_ids", []))]
                if obj.id in ids:
                    zs = np.asarray(data.physics_traj["T_world_obj"])[:, ids.index(obj.id), 2, 3]
                    ax.plot(t[: len(zs)], zs - ref, color=color, lw=1.0, ls=":", label=f"{obj.id} (physics)")
        _panel(ax, f"Object centre height {ref_label}" + (" (dotted = physics)" if data.physics_traj is not None else ""), "m")
        ax.legend(loc="upper right", ncol=3)

    def p_elbow(ax):
        for slot, side in enumerate(SIDES):
            ax.plot(t, data.elbow_conf[:, slot], color=mjcf.HAND_HEX[side], lw=1.4, label=side)
        if data.arm_method is not None:
            prior = (data.arm_method == 2).any(axis=1)
            for a, b in _segments_where(prior):
                ax.axvspan(t[a], t[min(b, data.T - 1)], color="#f1efe9", lw=0, zorder=0)
        ax.set_ylim(-0.05, 1.05)
        _panel(ax, "Elbow confidence (shaded: prior only)", "")

    def p_elbow_error(ax):
        for slot, side in enumerate(SIDES):
            e = np.linalg.norm(data.elbow[:, slot] - data.gt.elbow[:, slot], axis=-1) * 100
            ax.plot(t, e, color=mjcf.HAND_HEX[side], lw=1.2, label=side)
        _panel(ax, "Elbow error vs ground truth", "cm")

    def p_camera(ax):
        if data.gt is not None:
            ax.plot(t, data.gt.T_world_cam[:, 2, 3] - ref, color=MUTED, lw=0.9, ls="--", label="ground truth")
        if data.T_world_cam is not None:
            ax.plot(t, data.T_world_cam[:, 2, 3] - ref, color=CAMERA, lw=1.5, label="camera_traj")
        _panel(ax, f"Camera height {ref_label}", "m")
        ax.legend(loc="upper right", ncol=2)

    def p_retarget(ax):
        r = data.retarget
        err = np.asarray(r.get("err_m_hand", np.asarray(r["err_m"])[:, None]), dtype=np.float64)
        hands = [str(h) for h in np.atleast_1d(r.get("hands", ["all"]))]
        for i in range(err.shape[1]):
            color = mjcf.HAND_HEX.get(hands[i] if i < len(hands) else "", CAMERA)
            ax.plot(t[: len(err)], err[:, i] * 1000, color=color, lw=1.2, label=hands[i] if i < len(hands) else "")
        _panel(ax, f"Retargeting keypoint error ({r.get('robot', '')})", "mm")
        ax.legend(loc="upper right", ncol=2)

    def p_physics(ax):
        objs = data.physics.get("objects", {})
        for k, obj in enumerate(data.objects):
            dev = objs.get(obj.id, {}).get("per_frame", {}).get("pos_dev_m")
            if dev is not None:
                ax.plot(t[: len(dev)], np.asarray(dev, dtype=np.float64) * 100, color=object_hex(k), lw=1.2, label=obj.id)
        score = data.physics.get("plausibility")
        note = f"plausibility {score:.2f}" if score is not None else str(data.physics.get("status") or "")
        _panel(ax, "Simulated vs reconstructed object position" + (f"  ·  {note}" if note else ""), "cm")

    if data.joints is not None or data.gt is not None:
        panels.append(("wrist", 1.2, p_wrist))
    if data.joints is not None:
        panels.append(("state", 0.55, p_state))
    if data.objects:
        panels.append(("grasp", 0.35 + 0.2 * max(1, sum(int(o.grasped.any()) for o in data.objects) * 2), p_grasp))
        panels.append(("objects", 1.1, p_objects))
    if data.elbow_conf is not None:
        panels.append(("elbow", 0.8, p_elbow))
    if data.elbow is not None and data.gt is not None:
        panels.append(("elbow_err", 0.8, p_elbow_error))
    if data.T_world_cam is not None or data.gt is not None:
        panels.append(("camera", 0.8, p_camera))
    if data.retarget is not None and "err_m" in data.retarget:
        panels.append(("retarget", 0.8, p_retarget))
    if data.physics is not None and data.physics.get("objects"):
        panels.append(("physics", 0.8, p_physics))
    if not panels:
        return None
    with matplotlib.rc_context(STYLE):
        ratios = [p[1] for p in panels]
        fig, axes = plt.subplots(len(panels), 1, figsize=(8.5, 1.05 * sum(ratios) + 0.3 * len(panels) + 0.9), sharex=True,
                                 gridspec_kw={"height_ratios": ratios, "hspace": 0.55})
        axes = np.atleast_1d(axes)
        for ax, (_, _, fn) in zip(axes, panels):
            fn(ax)
        axes[-1].set_xlabel("episode time (s)")
        axes[0].set_xlim(t[0], t[-1] if data.T > 1 else t[0] + 1)
        title = f"{data.name} · timelines"
        if data.gt is not None:
            title += "  (dashed = ground truth)"
        fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, color=INK)
        fig.align_ylabels(axes)
        fig.subplots_adjust(left=0.13, right=0.98, top=1 - 0.55 / fig.get_figheight(), bottom=0.5 / fig.get_figheight())
        fig.savefig(out, dpi=140)
        plt.close(fig)
    return out


def _panel(ax: Any, title: str, unit: str) -> None:
    ax.text(0.0, 1.03, title, transform=ax.transAxes, fontsize=8, color=INK, ha="left", va="bottom")
    ax.set_ylabel(unit)


# =============================================================================================
# Reprojection overlay
# =============================================================================================


def fig_overlay(ep: Episode, data: EpisodeData, out: Path, n: int = 6) -> Path | None:
    if data.T_world_cam is None or data.camera is None or not ep.video_path.is_file():
        return None
    import cv2

    plt = _plt()
    import matplotlib

    cam = data.camera
    ok = np.all(np.isfinite(data.T_world_cam), axis=(-1, -2))
    idx = np.flatnonzero(ok)
    if not len(idx):
        return None
    picks = np.unique(idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).round().astype(int)])
    images = dict(ep.frames(picks.tolist()))
    table_z = data.table_top_z
    meshes = [render_meshes(o, max_triangles=400) for o in data.objects]
    held = [o.held_poses() for o in data.objects]
    edges = [_mesh_edges(V, F) for V, F in meshes]

    def bgr(code: str) -> tuple[int, int, int]:
        return int(code[5:7], 16), int(code[3:5], 16), int(code[1:3], 16)

    def draw_poly(img, P_world, T_cam_world, color, closed, thickness):
        uv = cam.project(geo.transform(T_cam_world, P_world))
        if np.any(~np.isfinite(uv)):
            return
        cv2.polylines(img, [np.round(uv * 4).astype(np.int32)], closed, color, thickness, cv2.LINE_AA, shift=2)

    with matplotlib.rc_context({**STYLE, "axes.grid": False}):
        fig, axes = plt.subplots(1, len(picks), figsize=(2.2 * len(picks), 2.2 * cam.height / cam.width + 0.7), squeeze=False)
        for c, f in enumerate(picks):
            img = images.get(int(f))
            if img is None:
                continue
            img = img.copy()
            Tcw = geo.inv_T(data.T_world_cam[f])
            if data.table_polygon is not None and table_z is not None:
                poly = np.column_stack([data.table_polygon, np.full(len(data.table_polygon), table_z)])
                draw_poly(img, poly, Tcw, bgr("#b9ae9f"), True, 2)
            for k, (V, F) in enumerate(meshes):
                T = held[k][f]
                if not np.all(np.isfinite(T)):
                    continue
                P = geo.transform(T, V)
                for a, b in edges[k]:
                    draw_poly(img, P[[a, b]], Tcw, bgr(object_hex(k)), False, 2)
            for slot, side in enumerate(SIDES):
                if data.joints is None:
                    continue
                J = data.joints[f, slot]
                if not np.all(np.isfinite(J)):
                    continue
                for a, b in sk.BONES:
                    draw_poly(img, J[[a, b]], Tcw, bgr(mjcf.HAND_HEX[side]), False, 2)
                if data.shoulder is not None and data.elbow is not None:
                    arm = np.stack([data.shoulder[f, slot], data.elbow[f, slot], J[0]])
                    if np.all(np.isfinite(arm)):
                        draw_poly(img, arm, Tcw, (150, 150, 150), False, 3)
            axes[0, c].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0, c].set_title(f"t = {data.t[f]:.2f} s  #{f}", fontsize=8, loc="center")
            axes[0, c].set_xticks([])
            axes[0, c].set_yticks([])
        fig.suptitle(f"{data.name} · world reconstruction projected into the video (camera_traj + calibration)",
                     x=0.01, ha="left", fontsize=9, color=INK)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
    return out


def _mesh_edges(V: np.ndarray, F: np.ndarray) -> list[tuple[int, int]]:
    from ..sim.replay import feature_edges

    E = feature_edges(V, F)
    out = set()
    for i, tri in enumerate(F):
        for k in range(3):
            if E[i, k]:
                a, b = int(tri[k]), int(tri[(k + 1) % 3])
                out.add((min(a, b), max(a, b)))
    return sorted(out)


# =============================================================================================
# Entry point
# =============================================================================================


def make_figures(ep: Episode, out_dir: str | Path | None = None, figures: Sequence[str] = FIGURES,
                 with_gt: bool = True, data: EpisodeData | None = None) -> dict[str, Path]:
    """Write the figures that the available data allows; returns ``{figure: path}``."""
    out_dir = Path(out_dir) if out_dir is not None else ep.derived / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = data if data is not None else load_episode_data(ep, with_gt=with_gt)
    written: dict[str, Path] = {}
    makers = {
        "topview": lambda: fig_topview(data, out_dir / "topview.png"),
        "snapshots": lambda: fig_snapshots(data, out_dir / "snapshots.png"),
        "timelines": lambda: fig_timelines(data, out_dir / "timelines.png"),
        "overlay": lambda: fig_overlay(ep, data, out_dir / "overlay.png"),
    }
    for name in figures:
        if name not in makers:
            raise ValueError(f"unknown figure {name!r}; figures: {', '.join(makers)}")
        try:
            path = makers[name]()
        except Exception:  # noqa: BLE001 - one broken figure must not hide the others
            log.exception("figure %s failed", name)
            continue
        if path is not None:
            written[name] = path
    return written
