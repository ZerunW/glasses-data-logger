"""``eval-charuco REC``: check the depth maps (and the hand-derived scale) against an on-screen ChArUco board.

For a recording that films the receiver's calibration board (PIPELINE.md section 3: the board is
``calibration.json["board"]``, or given with ``--dictionary/--squares-x/--squares-y/--square-m/
--marker-m``):

1. every depth frame (``derived/depth/index.npz``; ``--stride``/``--max-frames`` thin them) is decoded
   and the board detected (``cv2.aruco.CharucoDetector``, same board layout and parameters as the
   receiver's calibration page);
2. the board pose comes from PnP on the undistorted corners with the calibrated camera (IPPE for
   the plane, then Levenberg-Marquardt); frames with fewer than ``--min-corners`` corners or a
   reprojection error above ``--max-reproj-px`` are skipped;
3. the board plane gives a reference z-depth for every pixel of the depth grid inside the board
   polygon (shrunk by ``--margin`` squares);
4. the depth map is compared there, raw and multiplied by ``hand3d.depth_scale_smooth`` when
   ``derived/hand3d.npz`` exists: AbsRel, RMSE, delta < 1.25, and the median ratio
   ``board / model`` (the true ``metric / model`` scale, the quantity ``hand3d`` estimates).

Writes ``derived/eval_charuco.json`` (with ``meta``), ``eval_charuco.md`` and ``eval_charuco.png``
(per frame: board scale vs ``depth_scale_smooth``; AbsRel raw vs scaled).
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from .. import geometry as geo
from ..config import normalize
from ..episode import Episode
from ..errors import PipelineError

log = logging.getLogger(__name__)

VERSION = 1
BOARD_KEYS = ("dictionary", "squares_x", "squares_y", "square_m", "marker_m")


# =============================================================================================
# Board and pose
# =============================================================================================


def board_spec(ep: Episode, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The board: ``calibration.json["board"]`` completed / overridden by ``overrides``."""
    spec: dict[str, Any] = {}
    calib = ep.path / "calibration.json"
    if calib.is_file():
        import json

        spec.update(json.loads(calib.read_text()).get("board") or {})
    spec.update({k: v for k, v in (overrides or {}).items() if v is not None})
    missing = [k for k in BOARD_KEYS if k not in spec]
    if missing:
        raise PipelineError(f"board definition lacks {', '.join(missing)}: put a 'board' entry in calibration.json "
                            "or pass --dictionary --squares-x --squares-y --square-m --marker-m")
    return {
        "dictionary": str(spec["dictionary"]), "squares_x": int(spec["squares_x"]), "squares_y": int(spec["squares_y"]),
        "square_m": float(spec["square_m"]), "marker_m": float(spec["marker_m"]),
    }


def make_board(spec: Mapping[str, Any]) -> Any:
    name = str(spec["dictionary"])
    if not hasattr(cv2.aruco, name):
        raise PipelineError(f"unknown ArUco dictionary {name!r}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    return cv2.aruco.CharucoBoard((int(spec["squares_x"]), int(spec["squares_y"])), float(spec["square_m"]),
                                  float(spec["marker_m"]), dictionary)


def make_detector(board: Any) -> Any:
    params = cv2.aruco.CharucoParameters()
    params.tryRefineMarkers = True
    return cv2.aruco.CharucoDetector(board, params)


def board_pose(detector: Any, board: Any, gray: np.ndarray, camera: geo.Camera,
               min_corners: int = 8) -> dict[str, Any] | None:
    """``{"R", "t", "corners", "reproj_px"}`` (board frame -> camera) or None."""
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or corners is None or len(np.ravel(ids)) < min_corners:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    obj = np.asarray(obj, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(img, dtype=np.float64).reshape(-1, 2)
    xy = camera.undistort_points(img)
    ok = np.all(np.isfinite(xy), axis=1)
    if ok.sum() < min_corners:
        return None
    obj, img, xy = obj[ok], img[ok], xy[ok]
    try:
        found, rvec, tvec = cv2.solvePnP(obj, xy, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
        if not found:
            return None
        found, rvec, tvec = cv2.solvePnP(obj, xy, np.eye(3), None, rvec, tvec, useExtrinsicGuess=True,
                                         flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        return None
    if not found:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    proj = camera.project(obj @ R.T + t)
    err = np.linalg.norm(proj - img, axis=1)
    return {"R": R, "t": t, "corners": int(len(obj)), "reproj_px": float(np.sqrt(np.mean(err**2)))}


def board_depth(camera_grid: geo.Camera, R: np.ndarray, t: np.ndarray, size_m: tuple[float, float],
                margin_m: float = 0.0) -> np.ndarray:
    """z-depth of the board plane at every pixel of ``camera_grid``'s image inside the board
    rectangle ``[margin, size - margin]`` (board frame), NaN elsewhere."""
    rays = camera_grid.pixel_rays()
    n = R[:, 2]
    denom = rays @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        s = float(n @ t) / denom
    P = rays * s[..., None]
    local = (P - t) @ R  # board coordinates
    inside = (
        (s > 0) & np.isfinite(s)
        & (local[..., 0] >= margin_m) & (local[..., 0] <= size_m[0] - margin_m)
        & (local[..., 1] >= margin_m) & (local[..., 1] <= size_m[1] - margin_m)
    )
    return np.where(inside, s, np.nan)


# =============================================================================================
# Evaluation
# =============================================================================================


def _errors(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    m = np.isfinite(pred) & np.isfinite(ref) & (pred > 0) & (ref > 0)
    if not np.any(m):
        return {"n": 0}
    p, g = pred[m], ref[m]
    ratio = np.maximum(p / g, g / p)
    return {"abs_rel": float(np.mean(np.abs(p - g) / g)), "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
            "delta1": float(np.mean(ratio < 1.25)), "n": int(m.sum())}


class _Acc:
    def __init__(self) -> None:
        self.abs = self.sq = self.d1 = 0.0
        self.n = 0

    def add(self, p: np.ndarray, g: np.ndarray) -> None:
        m = np.isfinite(p) & np.isfinite(g) & (p > 0) & (g > 0)
        if not np.any(m):
            return
        pp, gg = p[m], g[m]
        self.abs += float(np.sum(np.abs(pp - gg) / gg))
        self.sq += float(np.sum((pp - gg) ** 2))
        self.d1 += float(np.sum(np.maximum(pp / gg, gg / pp) < 1.25))
        self.n += int(m.sum())

    def result(self) -> dict[str, Any]:
        if self.n == 0:
            return {"n": 0}
        return {"abs_rel": self.abs / self.n, "rmse": math.sqrt(self.sq / self.n), "delta1": self.d1 / self.n, "n": self.n}


def evaluate(ep: Episode, spec: Mapping[str, Any], stride: int = 1, max_frames: int = 0, min_corners: int = 8,
             max_reproj_px: float = 3.0, margin_squares: float = 0.25, min_pixels: int = 20) -> dict[str, Any]:
    """Metrics and per-frame arrays (see the module docstring)."""
    if not (ep.has("depth/index.npz") and ep.has("depth/depth_f16.npy")):
        raise PipelineError(f"{ep.path}: no derived/depth (run the depth step first)")
    index = ep.load_npz("depth/index")
    frames = np.asarray(index["frames"], dtype=np.int64).reshape(-1)
    maps = np.load(ep.derived / "depth" / "depth_f16.npy", mmap_mode="r")
    h, w = maps.shape[1:]
    camera = ep.camera()
    grid = camera.resized(w, h)
    scale = None
    if ep.has("hand3d.npz"):
        h3 = ep.load_npz("hand3d")
        if "depth_scale_smooth" in h3:
            scale = np.asarray(h3["depth_scale_smooth"], dtype=np.float64)
    rows = np.arange(0, len(frames), max(1, int(stride)))
    if max_frames and len(rows) > max_frames:
        rows = rows[np.unique(np.linspace(0, len(rows) - 1, int(max_frames)).round().astype(int))]
    board = make_board(spec)
    detector = make_detector(board)
    size = (spec["squares_x"] * spec["square_m"], spec["squares_y"] * spec["square_m"])
    margin = float(margin_squares) * spec["square_m"]
    wanted = {int(frames[r]): int(r) for r in rows}
    per: dict[str, list] = {k: [] for k in ("frame", "corners", "reproj_px", "distance_m", "pixels", "scale_board",
                                            "abs_rel_raw", "abs_rel_scaled", "scale_pipeline")}
    raw_acc, scaled_acc = _Acc(), _Acc()
    skipped = {"no_board": 0, "reproj": 0, "few_pixels": 0}
    for frame, gray in ep.frames(sorted(wanted), gray=True):
        pose = board_pose(detector, board, gray, camera, min_corners)
        if pose is None:
            skipped["no_board"] += 1
            continue
        if pose["reproj_px"] > max_reproj_px:
            skipped["reproj"] += 1
            continue
        ref = board_depth(grid, pose["R"], pose["t"], size, margin)
        model = np.asarray(maps[wanted[frame]], dtype=np.float64)
        m = np.isfinite(ref) & np.isfinite(model) & (model > 0)
        if m.sum() < min_pixels:
            skipped["few_pixels"] += 1
            continue
        raw_acc.add(model[m], ref[m])
        s = float(scale[frame]) if scale is not None and frame < len(scale) and np.isfinite(scale[frame]) else math.nan
        if math.isfinite(s):
            scaled_acc.add(model[m] * s, ref[m])
        per["frame"].append(frame)
        per["corners"].append(pose["corners"])
        per["reproj_px"].append(pose["reproj_px"])
        per["distance_m"].append(float(np.linalg.norm(pose["t"])))
        per["pixels"].append(int(m.sum()))
        per["scale_board"].append(float(np.median(ref[m] / model[m])))
        per["abs_rel_raw"].append(_errors(model[m], ref[m])["abs_rel"])
        per["abs_rel_scaled"].append(_errors(model[m] * s, ref[m])["abs_rel"] if math.isfinite(s) else math.nan)
        per["scale_pipeline"].append(s)
    arrays = {k: np.asarray(v, dtype=np.float64 if k not in ("frame", "corners", "pixels") else np.int64)
              for k, v in per.items()}
    out: dict[str, Any] = {
        "board": dict(spec),
        "frames_tried": int(len(rows)),
        "frames_used": int(len(per["frame"])),
        "skipped": skipped,
        "raw": raw_acc.result(),
    }
    if len(per["frame"]):
        sb = arrays["scale_board"]
        out["scale_board_median"] = float(np.median(sb))
        out["scale_board_spread"] = float(np.median(np.abs(sb - np.median(sb))) * 1.4826)
        out["board_distance_m"] = [float(arrays["distance_m"].min()), float(arrays["distance_m"].max())]
        out["reproj_px_median"] = float(np.median(arrays["reproj_px"]))
    if scale is not None:
        out["scaled"] = scaled_acc.result()
        sp = arrays["scale_pipeline"]
        ok = np.isfinite(sp)
        if np.any(ok):
            rel = sp[ok] / arrays["scale_board"][ok] - 1.0
            out["pipeline_scale_rel_error"] = {"median": float(np.median(rel)), "mean_abs": float(np.mean(np.abs(rel))),
                                               "max_abs": float(np.max(np.abs(rel)))}
        out["note"] = "scaled = depth x hand3d.depth_scale_smooth; scale_board = median(board depth / model depth)"
    else:
        out["note"] = "no derived/hand3d.npz: only the raw depth is evaluated"
    return {"metrics": out, "per_frame": arrays}


# =============================================================================================
# Output
# =============================================================================================


def _markdown(ep: Episode, m: Mapping[str, Any]) -> str:
    lines = [f"# ChArUco depth check of `{ep.path.name}`", "",
             f"Created {datetime.now(timezone.utc).isoformat(timespec='seconds')}. {m.get('note', '')}", "",
             "| metric | value |", "|---|---|"]

    def flat(prefix: str, value: Any):
        if isinstance(value, Mapping):
            for k, v in value.items():
                yield from flat(f"{prefix}{k}.", v)
        else:
            yield prefix.rstrip("."), value

    for key, value in flat("", m):
        if key == "note":
            continue
        text = f"{value:.4g}" if isinstance(value, float) and math.isfinite(value) else str(value)
        lines.append(f"| {key} | {text} |")
    return "\n".join(lines) + "\n"


def _plot(path, per: Mapping[str, np.ndarray], fps: float = 24.0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    f = per["frame"]
    ax = axes[0]
    ax.plot(f, per["scale_board"], "o-", ms=3, color="k", label="board: median(board z / model z)")
    if np.any(np.isfinite(per["scale_pipeline"])):
        ax.plot(f, per["scale_pipeline"], "-", color="tab:red", label="hand3d depth_scale_smooth")
    ax.set_ylabel("metric / model")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(f, per["abs_rel_raw"], "o-", ms=3, color="tab:gray", label="AbsRel raw")
    if np.any(np.isfinite(per["abs_rel_scaled"])):
        ax.plot(f, per["abs_rel_scaled"], "o-", ms=3, color="tab:blue", label="AbsRel scaled by hand3d")
    ax.set_xlabel("video frame")
    ax.set_ylabel("AbsRel")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("Depth vs on-screen ChArUco board")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(ep: Episode, overrides: Mapping[str, Any] | None = None, plot: bool = True, **options: Any) -> dict[str, Any]:
    """Evaluate and write ``derived/eval_charuco.json/.md/.png``; returns the metrics."""
    from ..runner import input_stats

    spec = board_spec(ep, overrides)
    result = evaluate(ep, spec, **options)
    m = result["metrics"]
    meta = {
        "step": "eval-charuco", "version": VERSION, "params": normalize({"board": spec, **options}),
        "inputs": input_stats(ep, ["video.mp4", "calibration.json", "derived/depth/index.npz",
                                   "derived/depth/depth_f16.npy", "derived/hand3d.npz"]),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": None, "calibration": ep.step_meta()["calibration"],
    }
    per = {k: v.tolist() for k, v in result["per_frame"].items()}
    ep.save_json("eval_charuco.json", {"recording": str(ep.path), "metrics": m, "per_frame": per}, meta)
    ep.file("eval_charuco.md").write_text(_markdown(ep, m))
    if plot and m["frames_used"]:
        _plot(ep.file("eval_charuco.png"), result["per_frame"])
    return m


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m glassesdata_pipeline eval-charuco",
                                     description="Depth maps (and hand3d's scale) vs an on-screen ChArUco board.")
    parser.add_argument("recording")
    parser.add_argument("--dictionary", help="e.g. DICT_5X5_100 (default: calibration.json board)")
    parser.add_argument("--squares-x", type=int)
    parser.add_argument("--squares-y", type=int)
    parser.add_argument("--square-m", type=float, help="square side on the screen, metres")
    parser.add_argument("--marker-m", type=float, help="marker side on the screen, metres")
    parser.add_argument("--stride", type=int, default=1, help="use every k-th depth frame")
    parser.add_argument("--max-frames", type=int, default=0, help="at most this many frames (evenly spaced); 0 = all")
    parser.add_argument("--min-corners", type=int, default=8)
    parser.add_argument("--max-reproj-px", type=float, default=3.0)
    parser.add_argument("--margin", type=float, default=0.25, help="ignore this many squares at the board edge")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:
        ep = Episode(args.recording)
        overrides = {"dictionary": args.dictionary, "squares_x": args.squares_x, "squares_y": args.squares_y,
                     "square_m": args.square_m, "marker_m": args.marker_m}
        m = run(ep, overrides, plot=not args.no_plot, stride=args.stride, max_frames=args.max_frames,
                min_corners=args.min_corners, max_reproj_px=args.max_reproj_px, margin_squares=args.margin)
    except PipelineError as error:
        log.error("%s", error)
        return 1
    print(f"frames used {m['frames_used']} of {m['frames_tried']}; raw AbsRel {m['raw'].get('abs_rel', float('nan')):.4f}; "
          f"board scale (metric/model) {m.get('scale_board_median', float('nan')):.4f}")
    if "scaled" in m:
        print(f"scaled by hand3d: AbsRel {m['scaled'].get('abs_rel', float('nan')):.4f}; "
              f"hand3d scale vs board {m.get('pipeline_scale_rel_error', {}).get('median', float('nan')):+.4f}")
    print(f"wrote {ep.derived / 'eval_charuco.json'}")
    return 0
