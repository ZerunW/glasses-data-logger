"""``train REC --play``: roll out a checkpoint on the CPU MuJoCo backend (one world, no domain
randomisation, episode 1 starts at the demo start without noise, later episodes with the reset
noise), write ``<run>/play/episode_<k>.mp4`` with the reference object trajectory (green), the
simulated object path (orange) and the current reference target overlaid, and print / save the
success metrics (``<run>/play/metrics.json``):

* object position error (mean / max / final, m) and orientation error (mean / max, degrees),
* completed fraction of the demo and the termination reason,
* tracking success: the demo was completed and the mean position error < ``play.success_pos_m``,
* lift success (when the demo lifts the object by >= ``play.lift_height_m``): the simulated object
  rises by at least ``play.lift_fraction`` of the demo's lift.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import PipelineError

log = logging.getLogger(__name__)


# =============================================================================================
# metrics (pure numpy)
# =============================================================================================


def lift_success(ref_z: Any, sim_z: Any, lift_height_m: float, fraction: float) -> dict[str, Any]:
    """Lift relative to the episode start: ``success`` is None when the demo does not lift."""
    ref_z = np.asarray(ref_z, dtype=np.float64)
    sim_z = np.asarray(sim_z, dtype=np.float64)
    ref_lift = float(np.max(ref_z - ref_z[0])) if ref_z.size else 0.0
    sim_lift = float(np.max(sim_z - sim_z[0])) if sim_z.size else 0.0
    demo_lifts = ref_lift >= lift_height_m
    return {"demo_lifts": bool(demo_lifts), "ref_lift_m": ref_lift, "sim_lift_m": sim_lift,
            "success": (bool(sim_lift >= fraction * ref_lift) if demo_lifts else None)}


def episode_metrics(pos_err: Any, rot_err: Any, ref_z: Any, sim_z: Any, steps_total: int, reason: str,
                    play: Mapping[str, Any]) -> dict[str, Any]:
    """Metrics of one episode from per-step errors (after each action) and object heights
    (including the initial state)."""
    pos_err = np.asarray(pos_err, dtype=np.float64)
    rot_err = np.asarray(rot_err, dtype=np.float64)
    n = int(pos_err.size)
    completed = reason == "time_out"
    mean_pos = float(np.mean(pos_err)) if n else math.nan
    return {
        "steps": n,
        "completed_fraction": n / max(steps_total, 1),
        "termination": reason,
        "obj_pos_err_mean_m": mean_pos,
        "obj_pos_err_max_m": float(np.max(pos_err)) if n else math.nan,
        "obj_pos_err_final_m": float(pos_err[-1]) if n else math.nan,
        "obj_rot_err_mean_deg": float(np.degrees(np.mean(rot_err))) if n else math.nan,
        "obj_rot_err_max_deg": float(np.degrees(np.max(rot_err))) if n else math.nan,
        "tracking_success": bool(completed and mean_pos < float(play.get("success_pos_m", 0.05))),
        "lift": lift_success(ref_z, sim_z, float(play.get("lift_height_m", 0.03)), float(play.get("lift_fraction", 0.5))),
    }


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {}
    lifts = [e["lift"]["success"] for e in episodes if e["lift"]["success"] is not None]
    return {
        "episodes": len(episodes),
        "obj_pos_err_mean_m": float(np.mean([e["obj_pos_err_mean_m"] for e in episodes])),
        "obj_rot_err_mean_deg": float(np.mean([e["obj_rot_err_mean_deg"] for e in episodes])),
        "completed_fraction": float(np.mean([e["completed_fraction"] for e in episodes])),
        "tracking_success_rate": float(np.mean([e["tracking_success"] for e in episodes])),
        "lift_success_rate": (float(np.mean(lifts)) if lifts else None),
    }


def termination_reason(reasons: Mapping[str, Any], terminated: bool, truncated: bool) -> str:
    if truncated:
        return "time_out"
    if not terminated:
        return "running"
    hit = [k for k, v in reasons.items() if bool(v)]
    return ",".join(hit) or "terminated"


# =============================================================================================
# rollout and video (Linux machine)
# =============================================================================================


def _add_sphere(mujoco: Any, scene: Any, pos: Any, radius: float, rgba: tuple[float, float, float, float]) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([radius, 0.0, 0.0]), pos=np.asarray(pos, dtype=np.float64),
                        mat=np.eye(3).flatten(), rgba=np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


class _Video:
    def __init__(self, mujoco: Any, model: Any, path: Path, fps: float, play: Mapping[str, Any], ref_path: np.ndarray) -> None:
        import cv2

        self.cv2 = cv2
        self.mujoco = mujoco
        width = min(int(play.get("width", 640)), int(model.vis.global_.offwidth))
        height = min(int(play.get("height", 480)), int(model.vis.global_.offheight))
        self.renderer = mujoco.Renderer(model, height, width)
        name = str(play.get("camera", "") or "")
        if name:
            self.camera: Any = name
        else:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = ref_path.mean(axis=0)
            cam.distance = float(play.get("distance", 0.7))
            cam.azimuth = float(play.get("azimuth", 135.0))
            cam.elevation = float(play.get("elevation", -25.0))
            self.camera = cam
        self.ref_path = ref_path[:: max(1, len(ref_path) // 60)]
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self.writer.isOpened():
            raise PipelineError(f"cannot open {path} for writing (OpenCV mp4v)")

    def frame(self, data: Any, sim_path: list[np.ndarray], target: np.ndarray, text: str) -> None:
        mj = self.mujoco
        self.renderer.update_scene(data, camera=self.camera)
        scn = self.renderer.scene
        for p in self.ref_path:
            _add_sphere(mj, scn, p, 0.004, (0.1, 0.75, 0.3, 0.8))
        for p in sim_path[:: max(1, len(sim_path) // 60)]:
            _add_sphere(mj, scn, p, 0.004, (0.95, 0.45, 0.1, 0.9))
        _add_sphere(mj, scn, target, 0.012, (0.1, 0.9, 0.3, 0.35))
        rgb = self.renderer.render()
        bgr = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
        self.cv2.putText(bgr, text, (10, 24), self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, self.cv2.LINE_AA)
        self.writer.write(bgr)

    def close(self) -> None:
        self.writer.release()
        self.renderer.close()


def run_play(ep: Any, robot: str, side: str, train: Mapping[str, Any], run_dir: Path, checkpoint: str | None = None,
             episodes: int | None = None, video: bool = True, seed: int = 0) -> dict[str, Any]:
    if "DISPLAY" not in os.environ:
        os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering; must be set before mujoco is imported
    import torch

    from .train import _import_runner, checkpoints, make_environment, runner_config
    from .vecenv import wrap

    play = dict(train.get("play") or {})
    episodes = int(episodes or play.get("episodes", 1))
    video = video and bool(play.get("video", True))
    out_dir = Path(run_dir) / "play"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(checkpoint) if checkpoint else (checkpoints(Path(run_dir))[-1][1] if checkpoints(Path(run_dir)) else None)
    if ckpt is None or not ckpt.is_file():
        raise PipelineError(f"no checkpoint to play in {run_dir}")
    cfg_file = Path(run_dir) / "config.json"
    if cfg_file.is_file():
        saved = json.loads(cfg_file.read_text())
        train = {**dict(saved.get("train") or {}), "play": play}  # the run's task settings, today's play settings
        runner_cfg = saved.get("runner") or runner_config(train.get("ppo") or {})
    else:
        runner_cfg = runner_config(train.get("ppo") or {})
    task, scene, sim = make_environment(ep, robot, side, train, "mujoco", 1, "cpu", seed, out_dir, play=True)
    task.auto_reset = False
    env = wrap(task, {"play": True})
    OnPolicyRunner = _import_runner()
    runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device="cpu")
    runner.load(str(ckpt), map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")
    mujoco = sim.mujoco
    ref = scene.reference
    results = []
    ids = torch.zeros(1, dtype=torch.long)
    for k in range(episodes):
        task.reset(ids, start=torch.zeros(1, dtype=torch.long), noise=k > 0)
        task._read()
        writer = _Video(mujoco, sim.models[0], out_dir / f"episode_{k}.mp4", 1.0 / task.cfg.control_dt, play,
                        ref.obj_pos) if video else None
        pos_err, rot_err = [], []
        sim_z = [float(task.state["obj_pos"][0, 2])]
        ref_z = [float(ref.obj_pos[0, 2])]
        sim_path = [task.state["obj_pos"][0].numpy().copy()]
        reason = "running"
        with torch.inference_mode():
            obs = env.get_observations()
            for _ in range(task.T - 1):
                if writer is not None:
                    i = int(task.idx[0])
                    text = f"t {i * task.cfg.control_dt:5.2f} s  pos err {1000 * (pos_err[-1] if pos_err else 0):5.1f} mm"
                    writer.frame(sim.datas[0], sim_path, ref.obj_pos[min(i + 1, task.T - 1)], text)
                actions = policy(obs)
                policy_obs, priv, _, terminated, truncated, info = task.step(actions)
                obs = env.pack(policy_obs, priv)
                pos_err.append(float(info["errors"]["obj_pos"][0]))
                rot_err.append(float(info["errors"]["obj_rot"][0]))
                sim_z.append(float(task.state["obj_pos"][0, 2]))
                ref_z.append(float(ref.obj_pos[int(task.idx[0]), 2]))
                sim_path.append(task.state["obj_pos"][0].numpy().copy())
                reason = termination_reason({key: v[0] for key, v in info["reasons"].items()},
                                            bool(terminated[0]), bool(truncated[0]))
                if reason != "running":
                    break
        if writer is not None:
            writer.frame(sim.datas[0], sim_path, ref.obj_pos[int(task.idx[0])], f"end: {reason}")
            writer.close()
        metrics = episode_metrics(pos_err, rot_err, ref_z, sim_z, task.T - 1, reason, play)
        metrics["video"] = str(out_dir / f"episode_{k}.mp4") if video else None
        results.append(metrics)
        print(f"episode {k}: {reason}, {metrics['steps']} steps ({100 * metrics['completed_fraction']:.0f}% of the demo), "
              f"object error {1000 * metrics['obj_pos_err_mean_m']:.1f} mm / {metrics['obj_rot_err_mean_deg']:.1f} deg, "
              f"lift {metrics['lift']}")
    summary = {"checkpoint": str(ckpt), **summarize(results)}
    (out_dir / "metrics.json").write_text(json.dumps({"summary": summary, "episodes": results}, indent=2))
    return {"summary": summary, "episodes": results}
