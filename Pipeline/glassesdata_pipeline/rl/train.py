"""``python -m glassesdata_pipeline train REC [options]`` - PPO (rsl-rl-lib 6.0.0) on one episode.

::

    train REC --robot shadow_right --backend warp --num-envs 4096 --iterations 2000
    train REC --backend mujoco --num-envs 32 --iterations 5       # CPU smoke test
    train REC --resume [--run NAME]                                # continue the latest (or named) run
    train REC --play [--run NAME | --checkpoint PATH]              # CPU rollout, video, metrics

Logs go to ``REC/derived/rl/<run>/``: TensorBoard event files, checkpoints ``model_<it>.pt``,
``config.json`` (the ``[train]`` table, the rsl_rl runner config, versions, the scene names) and
``scene_<robot>_<side>.xml`` + ``scene_names.json`` (the composed model). ``--play`` writes
``play/episode_<k>.mp4`` and ``play/metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..config import overrides_from_cli
from ..errors import ConfigError, MissingDependency, PipelineError

log = logging.getLogger(__name__)

PINNED = {"rsl-rl-lib": "6.0.0", "mujoco-warp": "4.0.1", "warp-lang": "1.19.0", "mujoco": "4.0.1"}
PPO_KEYS = {
    "num_steps_per_env", "save_interval", "logger", "actor_hidden_dims", "critic_hidden_dims", "activation",
    "obs_normalization", "init_noise_std", "noise_std_type", "learning_rate", "schedule", "desired_kl",
    "num_learning_epochs", "num_mini_batches", "clip_param", "gamma", "lam", "value_loss_coef", "entropy_coef",
    "max_grad_norm", "use_clipped_value_loss",
}
ALGORITHM_KEYS = ("learning_rate", "schedule", "desired_kl", "num_learning_epochs", "num_mini_batches", "clip_param",
                  "gamma", "lam", "value_loss_coef", "entropy_coef", "max_grad_norm", "use_clipped_value_loss")
CHECKPOINT_RE = re.compile(r"^model_(\d+)\.pt$")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m glassesdata_pipeline train",
                                description="Train a robot hand to reproduce the episode's object motion (rsl_rl PPO).")
    p.add_argument("recording")
    p.add_argument("--robot", help="<robot>_<side>, e.g. shadow_right (default train.robot)")
    p.add_argument("--backend", choices=("warp", "mujoco"), help="default train.backend")
    p.add_argument("--num-envs", type=int, help="default train.num_envs")
    p.add_argument("--iterations", type=int, help="default train.iterations")
    p.add_argument("--device", help="torch/warp device, default train.device (CPU backend: cpu)")
    p.add_argument("--seed", type=int)
    p.add_argument("--run", help="run folder under derived/rl/ (new run name, or the run to resume / play)")
    p.add_argument("--resume", action="store_true", help="continue from the latest checkpoint of the run")
    p.add_argument("--play", action="store_true", help="roll out a checkpoint on the CPU backend; video + metrics")
    p.add_argument("--checkpoint", help="checkpoint file for --play / --resume (default: the latest)")
    p.add_argument("--episodes", type=int, help="--play episodes (default train.play.episodes)")
    p.add_argument("--no-video", action="store_true", help="--play without the video")
    p.add_argument("--allow-unaligned", action="store_true",
                   help="train / play although the world is not gravity-aligned (camera_traj.npz meta, step world)")
    p.add_argument("--config", metavar="FILE", help="extra TOML merged over configs/ and pipeline.toml")
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="SECTION.KEY=VALUE")
    return p


# =============================================================================================
# pure helpers (tested on the Mac)
# =============================================================================================


def runner_config(ppo: Mapping[str, Any], run_name: str = "") -> dict[str, Any]:
    """The ``train_cfg`` dict for ``rsl_rl.runners.OnPolicyRunner`` (rsl-rl-lib 6.0.0 schema:
    ``algorithm`` / ``actor`` / ``critic`` with ``class_name``, ``obs_groups`` with the sets
    ``actor`` and ``critic``)."""
    ppo = dict(ppo or {})
    unknown = set(ppo) - PPO_KEYS
    if unknown:
        raise ConfigError(f"[train.ppo] has unknown keys: {', '.join(sorted(unknown))}")
    get = ppo.get
    if get("schedule", "adaptive") not in ("adaptive", "fixed"):
        raise ConfigError("train.ppo.schedule must be 'adaptive' or 'fixed'")
    if get("noise_std_type", "scalar") not in ("scalar", "log"):
        raise ConfigError("train.ppo.noise_std_type must be 'scalar' or 'log'")
    defaults = {"learning_rate": 3e-4, "schedule": "adaptive", "desired_kl": 0.01, "num_learning_epochs": 5,
                "num_mini_batches": 4, "clip_param": 0.2, "gamma": 0.99, "lam": 0.95, "value_loss_coef": 1.0,
                "entropy_coef": 0.0, "max_grad_norm": 1.0, "use_clipped_value_loss": True}
    algorithm = {"class_name": "PPO", **{k: get(k, defaults[k]) for k in ALGORITHM_KEYS}}
    norm = bool(get("obs_normalization", True))
    activation = str(get("activation", "elu"))
    cfg: dict[str, Any] = {
        "num_steps_per_env": int(get("num_steps_per_env", 24)),
        "save_interval": int(get("save_interval", 50)),
        "logger": str(get("logger", "tensorboard")),
        "obs_groups": {"actor": ["policy"], "critic": ["policy", "privileged"]},
        "algorithm": algorithm,
        "actor": {"class_name": "MLPModel", "hidden_dims": [int(v) for v in get("actor_hidden_dims", [512, 256, 128])],
                  "activation": activation, "obs_normalization": norm,
                  "distribution_cfg": {"class_name": "GaussianDistribution",
                                       "init_std": float(get("init_noise_std", 0.5)),
                                       "std_type": str(get("noise_std_type", "scalar"))}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [int(v) for v in get("critic_hidden_dims", [512, 256, 128])],
                   "activation": activation, "obs_normalization": norm},
    }
    if run_name:
        cfg["run_name"] = run_name
    return cfg


def checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """``[(iteration, path)]`` of ``model_<it>.pt`` in ``run_dir``, sorted by iteration."""
    out = []
    if run_dir.is_dir():
        for path in run_dir.iterdir():
            match = CHECKPOINT_RE.match(path.name)
            if match:
                out.append((int(match.group(1)), path))
    return sorted(out)


def latest_run(rl_root: Path) -> Path | None:
    """The run folder whose newest checkpoint is the most recent (None when there is none)."""
    best: tuple[float, Path] | None = None
    if not rl_root.is_dir():
        return None
    for run in rl_root.iterdir():
        cps = checkpoints(run) if run.is_dir() else []
        if cps:
            mtime = max(p.stat().st_mtime for _, p in cps)
            if best is None or mtime > best[0]:
                best = (mtime, run)
    return None if best is None else best[1]


def resolve_run(rl_root: Path, run: str | None, must_exist: bool) -> Path:
    if run:
        path = rl_root / run
        if must_exist and not checkpoints(path):
            raise PipelineError(f"{path} has no checkpoints (model_<it>.pt)")
        return path
    if must_exist:
        latest = latest_run(rl_root)
        if latest is None:
            raise PipelineError(f"no run with checkpoints under {rl_root}")
        return latest
    raise ValueError("run name needed")


def new_run_name(robot: str, side: str, name: str = "") -> str:
    return name or f"{time.strftime('%Y%m%d-%H%M%S')}_{robot}_{side}"


def installed_versions() -> dict[str, str | None]:
    from importlib import metadata

    out: dict[str, str | None] = {}
    for dist in list(PINNED) + ["torch", "tensordict"]:
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def check_versions(versions: Mapping[str, str | None], backend: str) -> list[str]:
    """Warnings for installed versions that differ from the ones this code was written against."""
    needed = ["rsl-rl-lib", "mujoco"] + (["mujoco-warp", "warp-lang"] if backend == "warp" else [])
    notes = []
    for dist in needed:
        have = versions.get(dist)
        if have is not None and have != PINNED[dist]:
            notes.append(f"{dist} {have} is installed; this code was written against {PINNED[dist]}")
    return notes


# =============================================================================================
# heavy parts (Linux machine)
# =============================================================================================


def _import_runner() -> Any:
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")  # rsl_rl's logger imports GitPython
    try:
        from rsl_rl.runners.on_policy_runner import OnPolicyRunner
    except ImportError as error:
        raise MissingDependency("pip install rsl-rl-lib==6.0.0 (needed for train)") from error
    return OnPolicyRunner


def make_environment(ep: Any, robot: str, side: str, train: Mapping[str, Any], backend: str, num_envs: int,
                     device: str, seed: int, out_dir: Path, play: bool = False) -> tuple[Any, Any, Any]:
    """``(task, loaded_scene, backend)`` with torch tensors on ``device``."""
    try:
        import torch
    except ImportError as error:
        raise MissingDependency("pip install torch (needed for train)") from error
    from .scene_adapter import load_scene
    from .task import TrackingTask, parse_task_config

    cfg = parse_task_config(train)
    if play:
        cfg.domain.mass_scale = (1.0, 1.0)
        cfg.domain.friction_scale = (1.0, 1.0)
    scene = load_scene(ep, robot, side, train, out_dir, for_warp=backend == "warp")
    for note in scene.notes:
        log.info("model: %s", note)
    torch.manual_seed(seed)
    if backend == "warp":
        from .env_warp import WarpBackend

        w = dict(train.get("warp") or {})
        sim = WarpBackend(scene.mj_model, scene.task_model, num_envs, device=device,
                          nconmax=int(w.get("nconmax", 96)), njmax=int(w.get("njmax", 512)),
                          graph_capture=bool(w.get("graph_capture", True)), sync=bool(w.get("sync", True)),
                          overflow_check_interval=int(w.get("overflow_check_interval", 100)))
    else:
        from .env_mujoco import MujocoBackend

        sim = MujocoBackend(scene.mj_model, scene.task_model, num_envs, xp=torch, device=device)
    task = TrackingTask(torch, sim, scene.task_model, scene.reference, cfg, seed=seed, device=torch.device(device))
    log.info("task: %d envs, %d actions, %d policy obs, %d privileged obs, reference %d steps (%.2f s), object %s",
             task.num_envs, task.num_actions, task.num_policy_obs, task.num_privileged_obs, task.T,
             task.T * cfg.control_dt, scene.reference.object_id)
    return task, scene, sim


def check_gravity(ep: Any, allow: bool) -> None:
    """Refuse a world that is not gravity-aligned (no IMU gravity in step ``world``): MuJoCo's
    gravity (-z) would not point down, so the learned behaviour would be wrong."""
    from ..sim.data import gravity_aligned

    if gravity_aligned(ep) is False:
        if not allow:
            raise PipelineError(f"{ep.path.name}: the world is not gravity-aligned (derived/camera_traj.npz: no IMU "
                                "gravity), so MuJoCo's gravity would not point down; fix the imu step, or pass "
                                "--allow-unaligned to train anyway")
        log.warning("the world is not gravity-aligned; training anyway (--allow-unaligned)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from ..episode import Episode
    from .reference import parse_robot

    try:
        ep = Episode(args.recording, overrides_from_cli(args.config, args.sets))
        train = ep.config.section("train").to_dict()
        robot, side = parse_robot(args.robot or str(train.get("robot", "shadow_right")))
        backend = args.backend or str(train.get("backend", "warp"))
        seed = int(args.seed if args.seed is not None else train.get("seed", 0))
        from .task import parse_task_config

        parse_task_config(train)  # fail fast on configuration errors
        runner_config(train.get("ppo") or {})
        check_gravity(ep, args.allow_unaligned)
        rl_root = ep.derived / "rl"
        if args.play:
            from .play import run_play

            run_dir = Path(args.checkpoint).parent if args.checkpoint else resolve_run(rl_root, args.run, True)
            result = run_play(ep, robot, side, train, run_dir, checkpoint=args.checkpoint,
                              episodes=args.episodes, video=not args.no_video, seed=seed)
            print(json.dumps(result["summary"], indent=2))
            return 0
        OnPolicyRunner = _import_runner()  # dependency check before anything is written
        if args.resume:
            run_dir = resolve_run(rl_root, args.run, True)
        else:
            run_dir = rl_root / new_run_name(robot, side, args.run or str(train.get("run_name", "")))
        num_envs = int(args.num_envs or train.get("num_envs", 4096))
        iterations = int(args.iterations or train.get("iterations", 2000))
        device = args.device or (str(train.get("device", "cuda:0")) if backend == "warp" else "cpu")
        runner_cfg = runner_config(train.get("ppo") or {}, run_dir.name)
        versions = installed_versions()
        for note in check_versions(versions, backend):
            log.warning("%s", note)
        run_dir.mkdir(parents=True, exist_ok=True)
        task, scene, sim = make_environment(ep, robot, side, train, backend, num_envs, device, seed, run_dir)
        from .vecenv import wrap

        env = wrap(task, {"train": train, "robot": robot, "side": side, "backend": backend})
        (run_dir / ("config_resume.json" if args.resume else "config.json")).write_text(json.dumps({
            "argv": list(argv or sys.argv[1:]), "recording": str(ep.path), "robot": robot, "side": side,
            "backend": backend, "num_envs": num_envs, "iterations": iterations, "device": device, "seed": seed,
            "train": train, "runner": runner_cfg, "versions": versions, "object": scene.reference.object_id,
            "obs": {"policy": task.num_policy_obs, "privileged": task.num_privileged_obs},
            "actions": task.num_actions, "reference_steps": task.T,
        }, indent=2, default=str))
        runner = OnPolicyRunner(env, runner_cfg, log_dir=str(run_dir), device=device)
        if args.resume:
            ckpt = Path(args.checkpoint) if args.checkpoint else checkpoints(run_dir)[-1][1]
            log.info("resuming from %s", ckpt)
            runner.load(str(ckpt), map_location=device)
        runner.learn(num_learning_iterations=iterations,
                     init_at_random_ep_len=bool(train.get("init_at_random_ep_len", False)))
        last = checkpoints(run_dir)
        overflow = getattr(sim, "overflow_worlds", 0)
        print(f"run {run_dir}")
        print(f"last checkpoint {last[-1][1] if last else 'none'}")
        if overflow:
            print(f"warning: {overflow} world-steps with MuJoCo Warp overflow (raise train.warp.nconmax/njmax)")
        print(f"tensorboard --logdir {rl_root}")
        print(f"play: python -m glassesdata_pipeline train {ep.path} --robot {robot}_{side} --play --run {run_dir.name}")
        return 0
    except PipelineError as error:
        log.error("%s", error)
        print(f"error: {error}", file=sys.stderr)
        return 1
