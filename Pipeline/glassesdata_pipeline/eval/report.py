"""Evaluate ``derived/`` against ``gt/`` and write ``derived/eval.json``, ``eval.md``, ``eval/*.png``.

Every metric group is computed when its inputs exist and skipped with a note otherwise:

==============  =========================================================================
group           inputs (derived/)
==============  =========================================================================
hands2d         hands2d.npz
depth           depth/depth_f16.npy + depth/index.npz (+ hand3d.npz depth_scale_smooth)
slam            slam.npz (T_slam_cam, valid)
camera          camera_traj.npz (T_world_cam, valid)
hand3d          hand3d.npz (joints_cam, valid)
hands_world     hands_world.npz (joints)
arms            arms.npz (shoulder, elbow, wrist)
objects         objects/index.json + objects/<id>/poses.npz, objects/table.json
imu             imu.npz
==============  =========================================================================
"""

from __future__ import annotations

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
from ..skeleton import PARENTS
from . import metrics as M

log = logging.getLogger(__name__)

VERSION = 1


def _skip(reason: str) -> dict[str, Any]:
    return {"status": "skipped", "reason": reason}


def _load(ep: Episode, name: str) -> dict[str, Any] | None:
    return ep.load_npz(name) if ep.has(name) else None


def evaluate(ep: Episode, cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """All metrics as a dict ``{group: {...}}``; per-frame curves go under ``"_curves"``."""
    cfg = ep.config if cfg is None else cfg
    ecfg = cfg.get("eval", {}) or {}
    gt = ep.gt
    if gt is None:
        raise PipelineError(f"{ep.path} has no gt/ (use eval-charuco for real recordings)")
    results: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    T_gt = np.asarray(gt["T_world_cam"], dtype=np.float64)
    T = T_gt.shape[0]
    groups = (
        ("hands2d", _eval_hands2d), ("depth", _eval_depth), ("slam", _eval_slam), ("camera", _eval_camera),
        ("hand3d", _eval_hand3d), ("hands_world", _eval_hands_world), ("arms", _eval_arms),
        ("objects", _eval_objects), ("imu", _eval_imu),
    )
    for name, fn in groups:
        try:
            results[name] = fn(ep, gt, ecfg, curves, T)
        except Exception as error:  # noqa: BLE001 - one broken output must not hide the others
            log.exception("eval %s failed", name)
            results[name] = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
    results["_curves"] = curves
    return results


# ---------------------------------------------------------------- groups


def _eval_hands2d(ep, gt, ecfg, curves, T):
    d = _load(ep, "hands2d.npz")
    if d is None:
        return _skip("no derived/hands2d.npz")
    uv = ep.camera().project(np.asarray(gt["joints_cam"]))
    out = {"status": "ok", **M.keypoint2d_errors(d["kp"], uv, gt["visible"])}
    return out


def _eval_depth(ep, gt, ecfg, curves, T):
    if not (ep.has("depth/index.npz") and ep.has("depth/depth_f16.npy")):
        return _skip("no derived/depth/depth_f16.npy + index.npz")
    index = ep.load_npz("depth/index.npz")
    pred = np.load(ep.derived / "depth" / "depth_f16.npy", mmap_mode="r")
    frames = np.asarray(index["frames"], dtype=np.int64)
    full = gt.depth()
    H, W = full.shape[1:]
    h, w = pred.shape[1:]
    scale = None
    hand3d = _load(ep, "hand3d.npz")
    if hand3d is not None and "depth_scale_smooth" in hand3d:
        scale = np.asarray(hand3d["depth_scale_smooth"], dtype=np.float64)
    rows = np.arange(len(frames))
    limit = int(ecfg.get("depth_max_frames", 0) or 0)
    if limit and len(rows) > limit:
        rows = np.unique(np.linspace(0, len(rows) - 1, limit).round().astype(int))
    lo, hi = float(ecfg.get("depth_min_m", 0.05)), float(ecfg.get("depth_max_m", 10.0))
    acc_raw = {"abs": 0.0, "sq": 0.0, "d1": 0.0, "n": 0}
    acc_scaled = {"abs": 0.0, "sq": 0.0, "d1": 0.0, "n": 0}
    per_frame = np.full(T, np.nan)
    for r in rows:
        f = int(frames[r])
        g = np.asarray(full[f], dtype=np.float32)
        if (w, h) != (W, H):
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        p = np.asarray(pred[r], dtype=np.float64)
        _accumulate(acc_raw, p, g, lo, hi)
        if scale is not None and np.isfinite(scale[f]):
            e = _accumulate(acc_scaled, p * scale[f], g, lo, hi)
        else:
            e = M.depth_errors(p, g, lo, hi).get("abs_rel", np.nan)
        per_frame[f] = e
    curves["depth_abs_rel"] = per_frame
    out = {"status": "ok", "frames": int(len(rows)), "raw": _finish(acc_raw)}
    if scale is not None:
        out["pipeline_scaled"] = _finish(acc_scaled)
        out["note"] = "pipeline_scaled multiplies each frame by hand3d.depth_scale_smooth (no GT alignment)"
    else:
        out["note"] = "no hand3d.npz depth_scale_smooth: raw depth evaluated as metric"
    return out


def _accumulate(acc, p, g, lo, hi) -> float:
    m = np.isfinite(p) & np.isfinite(g) & (p > 0) & (g >= lo) & (g <= hi)
    if not np.any(m):
        return float("nan")
    pp, gg = p[m], g[m].astype(np.float64)
    rel = np.abs(pp - gg) / gg
    acc["abs"] += float(rel.sum())
    acc["sq"] += float(((pp - gg) ** 2).sum())
    acc["d1"] += float((np.maximum(pp / gg, gg / pp) < 1.25).sum())
    acc["n"] += int(m.sum())
    return float(rel.mean())


def _finish(acc) -> dict[str, Any]:
    n = acc["n"]
    if n == 0:
        return {"n": 0}
    return {"abs_rel": acc["abs"] / n, "rmse": math.sqrt(acc["sq"] / n), "delta1": acc["d1"] / n, "n": n}


def _eval_slam(ep, gt, ecfg, curves, T):
    d = _load(ep, "slam.npz")
    if d is None:
        return _skip("no derived/slam.npz")
    est = np.asarray(d["T_slam_cam"], dtype=np.float64)
    valid = np.asarray(d.get("valid", np.ones(len(est), bool)), dtype=bool)
    out = M.trajectory_errors(est, gt["T_world_cam"], valid, ecfg.get("rpe_deltas", [1, 24]), aligns=("se3", "sim3"))
    curves["slam_position_error"] = M.per_frame_position_error(est, gt["T_world_cam"], valid, "sim3")
    curves["slam_positions_aligned"] = _aligned_positions(est, gt["T_world_cam"], valid)
    return out


def _eval_camera(ep, gt, ecfg, curves, T):
    d = _load(ep, "camera_traj.npz")
    if d is None:
        return _skip("no derived/camera_traj.npz")
    est = np.asarray(d["T_world_cam"], dtype=np.float64)
    valid = np.asarray(d.get("valid", np.ones(len(est), bool)), dtype=bool)
    out = M.trajectory_errors(est, gt["T_world_cam"], valid, ecfg.get("rpe_deltas", [1, 24]))
    curves["camera_position_error"] = M.per_frame_position_error(est, gt["T_world_cam"], valid, "none")
    curves["camera_positions"] = np.where(valid[:, None], est[:, :3, 3], np.nan)
    if out.get("status") == "ok":
        R_err = np.degrees(geo.rotation_angle(est[valid, :3, :3], np.asarray(gt["T_world_cam"])[valid, :3, :3]))
        out["rotation_error_deg_none"] = M._stats(R_err)
    return out


def _aligned_positions(est, gt_T, valid):
    align = M.align_module()
    if align is None:
        return np.where(valid[:, None], est[:, :3, 3], np.nan)
    try:
        aligned = align.ate(est, gt_T, align="sim3", valid=valid)["aligned"]
    except ValueError:
        aligned = est
    return np.where(valid[:, None], aligned[:, :3, 3], np.nan)


def _eval_hand3d(ep, gt, ecfg, curves, T):
    d = _load(ep, "hand3d.npz")
    if d is None:
        return _skip("no derived/hand3d.npz")
    valid = np.asarray(d.get("valid", np.ones((T, 2), bool)), dtype=bool)
    mask = valid[..., None] & (np.asarray(gt["visible"], bool) if ecfg.get("hands_visible_only", False) else True)
    out = M.joint_errors(d["joints_cam"], gt["joints_cam"], mask)
    out.update(M.hand_camera_errors(d["joints_cam"], gt["joints_cam"], mask))
    curves["hand3d_error"] = M.per_frame_joint_error(d["joints_cam"], gt["joints_cam"], valid)
    if "depth_scale_smooth" in d:
        out["depth_scale_smooth_mean"] = float(np.nanmean(d["depth_scale_smooth"])) if np.any(np.isfinite(d["depth_scale_smooth"])) else None
        truth = _depth_scale_truth(ep)
        if truth is not None and len(truth) == len(d["depth_scale_smooth"]):
            out["depth_scale"] = M.depth_scale_errors(d["depth_scale_smooth"], d.get("depth_scale"), truth)
            curves["depth_scale_rel_error"] = np.asarray(d["depth_scale_smooth"], dtype=np.float64) / truth - 1.0
    return out


def _depth_scale_truth(ep) -> np.ndarray | None:
    """True ``metric / model`` ratio per frame when ``derived/depth`` came from the gt backend (its
    noise parameters are read from the depth output's meta), else None."""
    if not ep.has("depth/index.npz"):
        return None
    meta = ep.load_npz("depth/index").get("meta", {}) or {}
    if meta.get("backend") != "gt":
        return None
    from ..backends import gt as gtb

    noise = ((meta.get("params") or {}).get("synth") or {}).get("gt_noise")
    cfg = {"synth": {"gt_noise": noise}} if noise is not None else ep.config
    return 1.0 / np.asarray(gtb.depth_scale_truth(ep, cfg), dtype=np.float64)


def _eval_hands_world(ep, gt, ecfg, curves, T):
    d = _load(ep, "hands_world.npz")
    if d is None:
        return _skip("no derived/hands_world.npz")
    J = np.asarray(d["joints"], dtype=np.float64)
    valid = np.all(np.isfinite(J), axis=-1)
    out = M.joint_errors(J, gt["joints_world"], valid)
    if "bone_lengths" in d:
        J0 = np.asarray(gt["joints_world"])[0]  # bone lengths are constant over time
        true = np.stack([np.linalg.norm(J0[:, i] - J0[:, PARENTS[i]], axis=-1) for i in range(1, 21)], axis=-1)
        out["bone_length_error_mm"] = float(np.nanmean(np.abs(np.asarray(d["bone_lengths"]) - true)) * 1000)
        # lengths of the output skeleton itself (bone_lengths above are the refinement targets)
        got = np.stack([np.linalg.norm(J[:, :, i] - J[:, :, PARENTS[i]], axis=-1) for i in range(1, 21)], axis=-1)
        out["joints_bone_length_error_mm"] = float(np.nanmean(np.abs(got - true[None])) * 1000) if np.isfinite(got).any() else None
    if "raw" in d:  # before smoothing (hand3d mapped to the world)
        raw = np.asarray(d["raw"], dtype=np.float64)
        out["raw_mpjpe"] = M.joint_errors(raw, gt["joints_world"], np.all(np.isfinite(raw), axis=-1), procrustes=False).get("mpjpe")
    curves["hands_world_error"] = M.per_frame_joint_error(J, gt["joints_world"], np.any(valid, axis=-1))
    return out


def _eval_arms(ep, gt, ecfg, curves, T):
    d = _load(ep, "arms.npz")
    if d is None:
        return _skip("no derived/arms.npz")
    out: dict[str, Any] = {"status": "ok"}
    for key, gt_key in (("shoulder", "shoulder_world"), ("elbow", "elbow_world")):
        if key in d:
            pred = np.asarray(d[key], dtype=np.float64)
            out[key] = M.point_errors(pred, gt[gt_key], np.all(np.isfinite(pred), axis=-1))
    if "wrist" in d:
        pred = np.asarray(d["wrist"], dtype=np.float64)
        out["wrist"] = M.point_errors(pred, np.asarray(gt["joints_world"])[:, :, 0], np.all(np.isfinite(pred), axis=-1))
    if "elbow" in d:
        curves["elbow_error"] = np.linalg.norm(np.asarray(d["elbow"], dtype=np.float64) - gt["elbow_world"], axis=-1)
        if "method" in d:  # 1 = forearm cue, 2 = prior only
            method = np.asarray(d["method"])
            for code, name in ((1, "elbow_cue_frames"), (2, "elbow_prior_frames")):
                out[name] = M._stats(np.where(method == code, curves["elbow_error"], np.nan))
    return out


def _eval_objects(ep, gt, ecfg, curves, T):
    if not ep.has("objects/index.json"):
        return _skip("no derived/objects/index.json")
    index = ep.load_json("objects/index.json")
    items = index.get("items", []) if isinstance(index, dict) else index
    gt_T = np.asarray(gt["T_world_obj"], dtype=np.float64)
    ids = gt.object_ids
    out: dict[str, Any] = {"status": "ok", "objects": {}}
    used: set[int] = set()
    for item in items:
        oid = str(item.get("id"))
        path = f"objects/{oid}/poses.npz"
        if not ep.has(path):
            out["objects"][oid] = _skip(f"no derived/{path}")
            continue
        d = ep.load_npz(path)
        est = np.asarray(d["T_world_obj"], dtype=np.float64)
        valid = np.asarray(d.get("valid", np.ones(len(est), bool)), dtype=bool) & np.all(np.isfinite(est), axis=(-2, -1))
        if not np.any(valid):
            out["objects"][oid] = {"status": "skipped", "reason": "no valid pose"}
            continue
        f = int(np.flatnonzero(valid)[0])
        dist = [np.linalg.norm(est[f, :3, 3] - gt_T[f, k, :3, 3]) if k not in used else np.inf for k in range(len(ids))]
        k = int(np.argmin(dist))
        used.add(k)
        res = M.object_pose_errors(est, gt_T[:, k], valid)
        res["matched_gt"] = ids[k]
        res.update(_object_extras(ep, gt, item, oid, k, d, est, valid))
        out["objects"][oid] = res
        curves[f"object_{oid}_translation_error"] = np.where(valid, np.linalg.norm(est[:, :3, 3] - gt_T[:, k, :3, 3], axis=-1), np.nan)
    if ep.has("objects/table.json"):
        table = ep.load_json("objects/table.json")
        if "height_m" in table:
            out["table_height_error_m"] = float(table["height_m"]) - float(gt["table_height_m"])
        out["table"] = _table_extras(table, gt)
    return out


def _object_extras(ep, gt, item, oid, k, d, est, valid) -> dict[str, Any]:
    """Size, per-phase pose errors (static / grasped, from gt grasped), grasp labels and how many
    prompts fall inside the matched object's ground-truth mask."""
    out: dict[str, Any] = {}
    gt_T = np.asarray(gt["T_world_obj"], dtype=np.float64)[:, k]
    held = np.asarray(gt["grasped"], dtype=bool)[:, k].any(axis=-1) if "grasped" in gt else np.zeros(len(est), bool)
    out["phases"] = {name: M.object_pose_errors(est, gt_T, valid & m)
                     for name, m in (("static", ~held), ("grasped", held)) if np.any(valid & m)}
    if ep.has(f"objects/{oid}/props.json"):
        props = ep.load_json(f"objects/{oid}/props.json")
        if "extent_m" in props:
            out["size"] = M.object_size_errors(props["extent_m"], np.asarray(gt["object_sizes"])[k], props.get("volume_m3"))
    if "grasped" in d and "grasped" in gt:
        out["grasp"] = M.grasp_label_errors(d["grasped"], np.asarray(gt["grasped"])[:, k])
    hits = {"auto": [], "manual": []}
    for prompt in item.get("prompts", [item.get("prompt", {})]):
        try:
            m = gt.mask(int(prompt["frame"]), k)
            hit = bool(m[int(round(prompt["y"])), int(round(prompt["x"]))])
        except (KeyError, IndexError, TypeError):
            continue
        hits["manual" if prompt.get("source") == "manual" else "auto"].append(hit)
    out["prompt_hits"] = {f"{s}_hit_rate": (float(np.mean(v)) if v else None) for s, v in hits.items()}
    out["prompt_hits"]["n_auto"] = len(hits["auto"])
    return out


def _table_extras(table, gt) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if table.get("top_z") is not None and "table_top_z" in gt:
        out["top_z_error_m"] = float(table["top_z"]) - float(gt["table_top_z"])
    poly = np.asarray(table.get("polygon", []), dtype=np.float64).reshape(-1, 2)
    if len(poly) >= 3 and "table_polygon" in gt:
        from ..core.plane import convex_polygon_iou

        r = convex_polygon_iou(poly, np.asarray(gt["table_polygon"], dtype=np.float64))
        out.update(polygon_iou=r["iou"], polygon_precision=r["frac_a"], polygon_coverage=r["frac_b"])
    if "normal" in table:
        n = np.asarray(table["normal"], dtype=np.float64)
        out["normal_tilt_deg"] = float(np.degrees(np.arccos(np.clip(abs(n[2]) / np.linalg.norm(n), -1, 1))))
    return out


def _eval_imu(ep, gt, ecfg, curves, T):
    d = _load(ep, "imu.npz")
    if d is None:
        return _skip("no derived/imu.npz")
    return {"status": "ok", **M.imu_errors(d, gt, gt["T_world_cam"])}


# ---------------------------------------------------------------- writing


def _markdown(results: Mapping[str, Any], ep: Episode) -> str:
    lines = [f"# Evaluation of `{ep.path.name}`", "", f"Created {datetime.now(timezone.utc).isoformat(timespec='seconds')}.", ""]
    for group, value in results.items():
        if group.startswith("_"):
            continue
        lines.append(f"## {group}")
        lines.append("")
        status = value.get("status", "ok") if isinstance(value, Mapping) else "ok"
        if status != "ok":
            lines.append(f"*{status}*: {value.get('reason', '')}")
            lines.append("")
            continue
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for key, v in _flatten(value):
            if key == "status":
                continue
            lines.append(f"| {key} | {_fmt(v)} |")
        lines.append("")
    return "\n".join(lines)


def _flatten(value: Any, prefix: str = ""):
    if isinstance(value, Mapping):
        for k, v in value.items():
            yield from _flatten(v, f"{prefix}{k}.")
    else:
        yield prefix.rstrip("."), value


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.4g}"
    return str(v)


def _plots(ep: Episode, gt, curves: Mapping[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = ep.derived / "eval"
    out_dir.mkdir(exist_ok=True)
    written = []
    T_gt = np.asarray(gt["T_world_cam"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(T_gt[:, 0, 3], T_gt[:, 1, 3], "k-", lw=2, label="ground truth")
    if "camera_positions" in curves:
        p = curves["camera_positions"]
        ax.plot(p[:, 0], p[:, 1], "-", color="tab:blue", label="world step (no alignment)")
    if "slam_positions_aligned" in curves:
        p = curves["slam_positions_aligned"]
        ax.plot(p[:, 0], p[:, 1], "--", color="tab:orange", label="slam (Sim(3) aligned)")
    ax.scatter(T_gt[0, 0, 3], T_gt[0, 1, 3], c="k", marker="o", zorder=3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Camera trajectory, top view (world)")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = out_dir / "trajectory_topview.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    series = [(k, v) for k, v in curves.items() if k not in ("camera_positions", "slam_positions_aligned")]
    if series:
        t = np.asarray(gt.get("t", np.arange(T_gt.shape[0]) / 24.0))
        fig, axes = plt.subplots(len(series), 1, figsize=(7, 1.8 * len(series) + 0.6), sharex=True, squeeze=False)
        for ax, (name, values) in zip(axes[:, 0], series):
            v = np.asarray(values, dtype=np.float64)
            if v.ndim == 2:
                for slot, side in enumerate(("left", "right")):
                    ax.plot(t[: len(v)], v[:, slot], label=side)
                ax.legend(fontsize=7, loc="upper right")
            else:
                ax.plot(t[: len(v)], v)
            ax.set_ylabel(name.replace("_", " "), fontsize=8)
            ax.grid(alpha=0.3)
        axes[-1, 0].set_xlabel("time (s)")
        fig.tight_layout()
        path = out_dir / "errors_per_frame.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(str(path))
    return written


def run_eval(ep: Episode, cfg: Mapping[str, Any] | None = None, plots: bool | None = None) -> dict[str, Any]:
    """Evaluate and write ``derived/eval.json``, ``derived/eval.md`` and ``derived/eval/*.png``."""
    from ..runner import input_stats

    cfg = ep.config if cfg is None else cfg
    ecfg = cfg.get("eval", {}) or {}
    results = evaluate(ep, cfg)
    curves = results.pop("_curves")
    files = [p.relative_to(ep.path).as_posix() for p in sorted(ep.derived.rglob("*")) if p.is_file()
             and p.suffix in (".npz", ".npy", ".json") and not p.name.startswith("eval") and "_runs" not in p.parts]
    meta = {
        "step": "eval", "version": VERSION, "params": normalize({"eval": ecfg}),
        "inputs": input_stats(ep, files + ["gt/gt.npz"]),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": None, "calibration": ep.step_meta()["calibration"],
    }
    pngs = _plots(ep, ep.gt, curves) if (ecfg.get("plots", True) if plots is None else plots) else []
    ep.save_json("eval.json", {"recording": str(ep.path), "metrics": results, "plots": pngs}, meta)
    md = ep.file("eval.md")
    md.write_text(_markdown(results, ep))
    summary = {k: v.get("status", "ok") for k, v in results.items()}
    log.info("eval: %s", ", ".join(f"{k}={v}" for k, v in summary.items()))
    return {"metrics": results, "plots": pngs, "json": str(ep.file("eval.json")), "markdown": str(md)}


__all__ = ["evaluate", "run_eval"]
