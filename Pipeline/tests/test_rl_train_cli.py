"""Training CLI helpers (rl/train.py), the rsl_rl runner configuration, run / checkpoint
discovery and the --play metrics (rl/play.py). Nothing here needs torch, MuJoCo or rsl_rl."""

from __future__ import annotations

import importlib.util
import os
import time

import numpy as np
import pytest

from glassesdata_pipeline.config import load_config
from glassesdata_pipeline.errors import ConfigError, PipelineError
from glassesdata_pipeline.rl import play as P
from glassesdata_pipeline.rl import train as Tr


def test_runner_config_matches_rsl_rl_6_schema():
    ppo = load_config().section("train").to_dict()["ppo"]
    cfg = Tr.runner_config(ppo, "run1")
    assert set(cfg) == {"num_steps_per_env", "save_interval", "logger", "obs_groups", "algorithm", "actor", "critic",
                        "run_name"}
    assert cfg["obs_groups"] == {"actor": ["policy"], "critic": ["policy", "privileged"]}
    alg = cfg["algorithm"]
    assert alg["class_name"] == "PPO"
    # every algorithm key is a keyword of rsl_rl 6.0.0 PPO.__init__ (device / multi_gpu_cfg come from the runner)
    ppo_kwargs = {"num_learning_epochs", "num_mini_batches", "clip_param", "gamma", "lam", "value_loss_coef",
                  "entropy_coef", "learning_rate", "max_grad_norm", "optimizer", "use_clipped_value_loss", "schedule",
                  "desired_kl", "normalize_advantage_per_mini_batch", "use_mixed_precision", "rnd_cfg", "symmetry_cfg"}
    assert set(alg) - {"class_name"} <= ppo_kwargs
    mlp_kwargs = {"hidden_dims", "activation", "obs_normalization", "network_cfg", "distribution_cfg"}
    for name in ("actor", "critic"):
        assert cfg[name]["class_name"] == "MLPModel"
        assert set(cfg[name]) - {"class_name"} <= mlp_kwargs
    dist = cfg["actor"]["distribution_cfg"]
    assert dist["class_name"] == "GaussianDistribution"
    assert set(dist) - {"class_name"} <= {"init_std", "std_range", "std_type", "learn_std"}
    assert "distribution_cfg" not in cfg["critic"]
    assert cfg["algorithm"]["learning_rate"] == pytest.approx(3e-4)
    assert cfg["actor"]["hidden_dims"] == [512, 256, 128]


@pytest.mark.parametrize("bad", [{"lr": 1e-3}, {"schedule": "cosine"}, {"noise_std_type": "exp"}])
def test_runner_config_errors(bad):
    with pytest.raises(ConfigError):
        Tr.runner_config(bad)


def test_checkpoint_and_run_discovery(tmp_path):
    root = tmp_path / "rl"
    assert Tr.latest_run(root) is None
    a, b, c = root / "a", root / "b", root / "c"
    for d in (a, b, c):
        d.mkdir(parents=True)
    for it in (0, 50, 100):
        (a / f"model_{it}.pt").write_bytes(b"x")
    (a / "model_best.pt").write_bytes(b"x")
    (b / "model_7.pt").write_bytes(b"x")
    old = time.time() - 100
    for p in a.iterdir():
        os.utime(p, (old, old))
    assert [it for it, _ in Tr.checkpoints(a)] == [0, 50, 100]
    assert Tr.latest_run(root) == b
    assert Tr.resolve_run(root, None, True) == b
    assert Tr.resolve_run(root, "a", True) == a
    with pytest.raises(PipelineError):
        Tr.resolve_run(root, "c", True)
    assert Tr.new_run_name("shadow", "right", "mine") == "mine"
    assert Tr.new_run_name("shadow", "right").endswith("_shadow_right")


def test_version_check():
    notes = Tr.check_versions({"rsl-rl-lib": "5.5.2", "mujoco": "4.0.1", "mujoco-warp": None}, "warp")
    assert len(notes) == 1 and "5.5.2" in notes[0]
    assert Tr.check_versions({"rsl-rl-lib": "6.0.0", "mujoco": "4.0.1"}, "mujoco") == []


def test_parser():
    args = Tr.build_parser().parse_args(["REC", "--robot", "shadow_left", "--backend", "mujoco", "--num-envs", "16",
                                         "--iterations", "3", "--resume", "--set", "train.seed=3"])
    assert args.robot == "shadow_left" and args.backend == "mujoco" and args.num_envs == 16
    assert args.iterations == 3 and args.resume and not args.play and args.sets == ["train.seed=3"]


@pytest.mark.skipif(importlib.util.find_spec("rsl_rl") is not None, reason="rsl_rl installed")
def test_main_reports_missing_dependency(tmp_path, capsys):
    rec = tmp_path / "rec"
    rec.mkdir()
    from glassesdata_pipeline.__main__ import main as cli

    assert cli(["train", str(rec), "--backend", "mujoco", "--num-envs", "4"]) == 1
    assert "rsl-rl-lib" in capsys.readouterr().err
    assert not (rec / "derived" / "rl").exists() or not any((rec / "derived" / "rl").iterdir())
    assert cli(["train", str(rec), "--set", "train.reward.bogus=1"]) == 1
    assert "bogus" in capsys.readouterr().err
    assert cli(["train", str(rec), "--play"]) == 1


# =============================================================================================
# play metrics
# =============================================================================================


def test_lift_success():
    ref = np.r_[np.zeros(5), np.linspace(0, 0.1, 10)] + 0.8
    ok = P.lift_success(ref, np.r_[np.zeros(5), np.linspace(0, 0.06, 10)] + 0.8, 0.03, 0.5)
    assert ok["demo_lifts"] and ok["success"] and ok["ref_lift_m"] == pytest.approx(0.1)
    bad = P.lift_success(ref, np.full(15, 0.8), 0.03, 0.5)
    assert bad["success"] is False
    flat = P.lift_success(np.full(10, 0.8), np.full(10, 0.8), 0.03, 0.5)
    assert flat["demo_lifts"] is False and flat["success"] is None


def test_episode_metrics_and_summary():
    play = {"success_pos_m": 0.05, "lift_height_m": 0.03, "lift_fraction": 0.5}
    pos = np.full(20, 0.02)
    rot = np.full(20, np.radians(10))
    z_ref = np.r_[np.zeros(5), np.linspace(0, 0.1, 16)]
    m = P.episode_metrics(pos, rot, z_ref, z_ref * 0.9, 20, "time_out", play)
    assert m["completed_fraction"] == 1.0 and m["tracking_success"] and m["lift"]["success"]
    assert m["obj_rot_err_mean_deg"] == pytest.approx(10.0)
    m2 = P.episode_metrics(pos[:8], rot[:8], z_ref[:9], z_ref[:9], 20, "obj_pos", play)
    assert m2["completed_fraction"] == pytest.approx(0.4) and not m2["tracking_success"]
    s = P.summarize([m, m2])
    assert s["episodes"] == 2 and s["tracking_success_rate"] == 0.5 and s["lift_success_rate"] == 1.0
    assert P.termination_reason({"obj_pos": True, "fell": False}, True, False) == "obj_pos"
    assert P.termination_reason({}, False, True) == "time_out"
    assert P.termination_reason({}, False, False) == "running"
