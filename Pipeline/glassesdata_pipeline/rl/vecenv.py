"""Adapter from :class:`~glassesdata_pipeline.rl.task.TrackingTask` (torch) to the ``VecEnv``
interface of ``rsl-rl-lib==6.0.0`` (``rsl_rl/env/vec_env.py`` at tag ``v6.0.0``):

* attributes ``num_envs``, ``num_actions``, ``max_episode_length``, ``episode_length_buf``,
  ``device``, ``cfg``;
* ``get_observations() -> TensorDict`` with the groups ``"policy"`` (actor and critic) and
  ``"privileged"`` (critic only; ``obs_groups = {"actor": ["policy"], "critic": ["policy",
  "privileged"]}``), each ``(num_envs, dim)``;
* ``step(actions) -> (TensorDict, rewards (N,), dones (N,) long, extras)`` with
  ``extras["time_outs"]`` (bool, truncation for value bootstrapping) and ``extras["log"]`` (scalars;
  keys containing ``/`` are written to TensorBoard as they are).

The runner calls ``step`` inside ``torch.inference_mode()``; every persistent buffer of the task is
created at construction and only updated in place, and observations are fresh tensors each step
(rsl_rl keeps a reference to them until the transition is stored).
"""

from __future__ import annotations

from typing import Any

_CLASS: Any = None


def _build_class() -> Any:
    from rsl_rl.env.vec_env import VecEnv
    from tensordict import TensorDict

    class TrackingVecEnv(VecEnv):
        def __init__(self, task: Any, cfg: dict[str, Any]) -> None:
            self.task = task
            self.num_envs = int(task.num_envs)
            self.num_actions = int(task.num_actions)
            self.max_episode_length = int(task.max_episode_length)
            self.device = task.device
            self.cfg = cfg

        @property
        def episode_length_buf(self) -> Any:
            return self.task.ep_steps

        @episode_length_buf.setter
        def episode_length_buf(self, value: Any) -> None:  # the runner assigns it for random initial lengths
            self.task.ep_steps[:] = value.to(self.task.ep_steps.device, self.task.ep_steps.dtype)

        def pack(self, policy: Any, privileged: Any) -> Any:
            return TensorDict({"policy": policy, "privileged": privileged}, batch_size=[self.num_envs], device=self.device)

        def get_observations(self) -> Any:
            return self.pack(*self.task.observations())

        def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
            policy, privileged, reward, terminated, truncated, info = self.task.step(actions.to(self.device))
            dones = (terminated | truncated).long()
            extras = {"time_outs": truncated, "log": log_scalars(info, terminated, truncated)}
            return self.pack(policy, privileged), reward, dones, extras

    return TrackingVecEnv


def log_scalars(info: dict[str, Any], terminated: Any, truncated: Any) -> dict[str, Any]:
    """TensorBoard scalars for one step (rsl_rl averages them over the rollout)."""
    out: dict[str, Any] = {}
    for key, value in info["terms"].items():
        out[f"Reward/{key}"] = value.float().mean()
    for key, value in info["errors"].items():
        out[f"Error/{key}"] = value.float().mean()
    for key, value in info["reasons"].items():
        out[f"Termination/{key}"] = value.float().mean()
    out["Termination/time_out"] = truncated.float().mean()
    out["Termination/any_failure"] = terminated.float().mean()
    return out


def wrap(task: Any, cfg: dict[str, Any] | None = None) -> Any:
    """A ``rsl_rl`` ``VecEnv`` around ``task``."""
    global _CLASS
    if _CLASS is None:
        _CLASS = _build_class()
    return _CLASS(task, dict(cfg or {}))
