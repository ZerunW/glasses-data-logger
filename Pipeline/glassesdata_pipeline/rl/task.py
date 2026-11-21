"""The object-trajectory tracking task: configuration, reference-state initialisation, action
scaling, termination, and :class:`TrackingTask` - the backend-independent environment core.

Everything here is written against an array module ``xp`` (numpy or torch). The physics backend
(:mod:`.env_mujoco` on the CPU, :mod:`.env_warp` on the GPU) only moves state in and out of the
simulator; this module decides what the state, the targets, the rewards and the resets are.
The unit tests drive :class:`TrackingTask` with numpy and a fake backend.

Gaps in the demonstration (``reference.observed`` / ``long_gap``): episodes start only at observed
steps (:func:`start_candidates`), an episode ends as a time-out when it reaches a step inside a
gap longer than ``train.max_gap_s``, and the tracking reward terms are zero on unobserved steps
(the reference there is interpolated, not measured).

Time convention: the environment is at demo control step ``idx`` (``0 .. T-1`` of the resampled
reference). An action is applied as a residual on the reference at ``idx + 1``; after
``decimation`` physics steps the index advances to ``idx + 1`` and rewards / termination compare
the simulated state with the reference at that index. Observations then show the reference at
``idx + 1`` and ``idx + lookahead`` (clamped to the demo end).

Backend protocol (see :class:`SimBackend`): ``read() -> dict`` with ``qpos (N, nq)``,
``qvel (N, nv)``, ``qfrc_actuator (N, nv)``, ``body_pos (N, 2, 3)`` and ``body_quat (N, 2, 4)``
for the bodies ``[hand root, object]``, ``site_pos (N, K, 3)`` for the fingertips;
``apply(ctrl (N, nu), mocap_pos (N, 3) | None, mocap_quat (N, 4) | None)`` writes the targets and
advances ``decimation`` physics steps; ``reset(env_ids, qpos, qvel, ctrl, mocap_pos, mocap_quat)``
writes full states for some worlds and recomputes their kinematics; ``set_domain(env_ids,
mass_scale, friction_scale)`` applies the per-world randomisation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Protocol

import numpy as np

from ..errors import ConfigError
from . import obs as obs_mod
from . import rewards as rew_mod
from . import xmath as xm

# =============================================================================================
# configuration ([train] in configs/train.toml)
# =============================================================================================

REWARD_TERMS = ("obj_pos", "obj_rot", "hand_qpos", "base_pos", "base_rot", "tips", "contact")
PENALTY_TERMS = ("action_rate", "energy", "termination")
TOP_LEVEL_KEYS = {
    "robot", "backend", "num_envs", "iterations", "seed", "device", "object", "hand", "segment_s",
    "control_dt", "lookahead_steps", "max_episode_steps", "min_episode_steps", "p_start_at_zero",
    "smooth_sigma_s", "max_gap_s", "run_name", "init_at_random_ep_len",
    "scene", "noise", "action", "reward", "termination", "domain", "obs", "ppo", "warp", "play",
}


@dataclass
class NoiseConfig:
    finger_qpos: float = 0.05
    base_pos: float = 0.005
    base_rot: float = 0.02
    obj_pos_xy: float = 0.005
    obj_pos_z: float = 0.002
    obj_rot: float = 0.02


@dataclass
class ActionConfig:
    clip: float = 1.0
    base_pos: float = 0.02
    base_rot: float = 0.10
    finger: float = 0.30


@dataclass
class RewardConfig:
    weights: dict[str, float] = field(default_factory=lambda: {
        "obj_pos": 5.0, "obj_rot": 2.0, "hand_qpos": 1.0, "base_pos": 0.5, "base_rot": 0.5,
        "tips": 1.0, "contact": 1.0, "action_rate": 0.01, "energy": 0.0005, "termination": 10.0})
    sigmas: dict[str, float] = field(default_factory=lambda: {
        "obj_pos": 0.05, "obj_rot": 0.5, "hand_qpos": 0.4, "base_pos": 0.05, "base_rot": 0.3,
        "tips": 0.03, "contact": 0.02})
    contact_near_m: float = 0.03


@dataclass
class TerminationConfig:
    obj_pos_m: float = 0.10
    obj_rot_rad: float = 0.0  # 0 = off
    fall_below_table_m: float = 0.05
    base_pos_m: float = 0.25  # 0 = off
    grace_steps: int = 0


@dataclass
class DomainConfig:
    mass_scale: tuple[float, float] = (0.7, 1.3)
    friction_scale: tuple[float, float] = (0.7, 1.3)
    resample_on_reset: bool = True


@dataclass
class ObsConfig:
    vel_scale: float = 0.1
    clip: float = 100.0


@dataclass
class TaskConfig:
    control_dt: float = 0.02
    lookahead_steps: int = 5
    max_episode_steps: int = 0  # 0 = until the demo end
    min_episode_steps: int = 10
    p_start_at_zero: float = 0.1
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)


def _sub(cls: type, data: Mapping[str, Any] | None, where: str) -> Any:
    data = dict(data or {})
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"[{where}] has unknown keys: {', '.join(sorted(unknown))}")
    out = cls()
    for key, value in data.items():
        default = getattr(out, key)
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ConfigError(f"{where}.{key} must be true or false, got {value!r}")
        elif isinstance(default, tuple):
            if not (isinstance(value, (list, tuple)) and len(value) == 2):
                raise ConfigError(f"{where}.{key} must be [low, high], got {value!r}")
            value = (float(value[0]), float(value[1]))
            if value[0] > value[1] or value[0] <= 0:
                raise ConfigError(f"{where}.{key} must satisfy 0 < low <= high, got {value!r}")
        elif isinstance(default, int):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"{where}.{key} must be an integer, got {value!r}")
        elif isinstance(default, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{where}.{key} must be a number, got {value!r}")
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ConfigError(f"{where}.{key} must be a finite number >= 0, got {value!r}")
        setattr(out, key, value)
    return out


def parse_task_config(train: Mapping[str, Any] | None) -> TaskConfig:
    """:class:`TaskConfig` from the ``[train]`` table (missing keys take the defaults above;
    unknown keys and invalid values raise :class:`~glassesdata_pipeline.errors.ConfigError`)."""
    train = dict(train or {})
    unknown = set(train) - TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"[train] has unknown keys: {', '.join(sorted(unknown))}")
    cfg = TaskConfig()
    if train.get("hand", "auto") not in ("auto", "left", "right"):
        raise ConfigError(f"train.hand must be 'auto', 'left' or 'right', got {train['hand']!r}")
    if train.get("backend", "warp") not in ("warp", "mujoco"):
        raise ConfigError(f"train.backend must be 'warp' or 'mujoco', got {train['backend']!r}")
    for key in ("control_dt", "p_start_at_zero"):
        if key in train:
            value = train[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ConfigError(f"train.{key} must be a number, got {value!r}")
            setattr(cfg, key, float(value))
    for key in ("lookahead_steps", "max_episode_steps", "min_episode_steps"):
        if key in train:
            value = train[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigError(f"train.{key} must be an integer >= 0, got {value!r}")
            setattr(cfg, key, value)
    if cfg.control_dt <= 0:
        raise ConfigError("train.control_dt must be > 0")
    if not 0.0 <= cfg.p_start_at_zero <= 1.0:
        raise ConfigError("train.p_start_at_zero must be in [0, 1]")
    if cfg.lookahead_steps < 1:
        raise ConfigError("train.lookahead_steps must be >= 1")
    cfg.noise = _sub(NoiseConfig, train.get("noise"), "train.noise")
    cfg.action = _sub(ActionConfig, train.get("action"), "train.action")
    if cfg.action.clip <= 0:
        raise ConfigError("train.action.clip must be > 0")
    cfg.termination = _sub(TerminationConfig, train.get("termination"), "train.termination")
    cfg.domain = _sub(DomainConfig, train.get("domain"), "train.domain")
    cfg.obs = _sub(ObsConfig, train.get("obs"), "train.obs")
    reward = dict(train.get("reward") or {})
    rc = RewardConfig()
    for key, value in reward.items():
        if key == "sigma":
            for name, sigma in dict(value).items():
                if name not in rc.sigmas:
                    raise ConfigError(f"train.reward.sigma.{name}: unknown term (known: {', '.join(rc.sigmas)})")
                if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not sigma > 0:
                    raise ConfigError(f"train.reward.sigma.{name} must be > 0, got {sigma!r}")
                rc.sigmas[name] = float(sigma)
        elif key == "contact_near_m":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                raise ConfigError(f"train.reward.contact_near_m must be > 0, got {value!r}")
            rc.contact_near_m = float(value)
        elif key in REWARD_TERMS or key in PENALTY_TERMS:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ConfigError(f"train.reward.{key} must be a weight >= 0, got {value!r}")
            rc.weights[key] = float(value)
        else:
            raise ConfigError(f"train.reward.{key}: unknown reward term "
                              f"(known: {', '.join(REWARD_TERMS + PENALTY_TERMS)}, sigma, contact_near_m)")
    cfg.reward = rc
    return cfg


# =============================================================================================
# pure functions: phase / time sampling, initial state, actions, termination
# =============================================================================================


def max_start_step(num_steps: int, min_episode_steps: int) -> int:
    """Largest start index that still leaves ``min_episode_steps`` steps before the demo end
    (``0`` when the demo is shorter than that)."""
    return max(0, num_steps - 1 - min_episode_steps)


def sample_start_steps(xp: Any, u: Any, u_zero: Any, num_steps: int, min_episode_steps: int,
                       p_start_at_zero: float) -> Any:
    """Reference state initialisation: start indices ``(n,)`` drawn uniformly from
    ``[0, max_start_step]`` using uniforms ``u`` in ``[0, 1)``; with probability
    ``p_start_at_zero`` (decided by ``u_zero``) the episode starts at the demo start."""
    hi = max_start_step(num_steps, min_episode_steps)
    idx = xm.to_int(xp, xp.floor(u * (hi + 1)))
    idx = xm.clip(xp, idx, 0, hi)
    return xp.where(u_zero < p_start_at_zero, idx * 0, idx)


def start_candidates(observed: Any, long_gap: Any, num_steps: int, min_episode_steps: int) -> np.ndarray:
    """Allowed start steps (numpy, increasing): observed steps outside long gaps, preferably with
    at least ``min_episode_steps`` steps before the next long gap or the demo end; falls back to
    any observed step at or before :func:`max_start_step`, then any observed step, then ``[0]``."""
    T = int(num_steps)
    obs = np.ones(T, bool) if observed is None else np.asarray(observed, dtype=bool).reshape(T)
    gap = np.zeros(T, bool) if long_gap is None else np.asarray(long_gap, dtype=bool).reshape(T)
    allowed = obs & ~gap
    stop = np.full(T, T - 1)  # the next step that ends an episode (a long-gap step or the demo end)
    nxt = T - 1
    for k in range(T - 1, -1, -1):
        if gap[k]:
            nxt = k
        stop[k] = nxt
    for cand in (allowed & (stop - np.arange(T) >= min_episode_steps),
                 allowed & (np.arange(T) <= max_start_step(T, min_episode_steps)), allowed):
        if cand.any():
            return np.nonzero(cand)[0].astype(np.int64)
    return np.zeros(1, dtype=np.int64)


def sample_from_candidates(xp: Any, u: Any, u_zero: Any, candidates: Any, p_start_at_zero: float) -> Any:
    """Start steps ``(n,)``: uniform over ``candidates`` (an index array on the device) using
    uniforms ``u``; with probability ``p_start_at_zero`` (``u_zero``) the first candidate."""
    n = int(candidates.shape[0])
    k = xm.clip(xp, xm.to_int(xp, xp.floor(u * n)), 0, n - 1)
    k = xp.where(u_zero < p_start_at_zero, k * 0, k)
    return candidates[k]


def offset_indices(xp: Any, idx: Any, offset: int, num_steps: int) -> Any:
    """``clamp(idx + offset, 0, T - 1)``."""
    return xm.clip(xp, idx + offset, 0, num_steps - 1)


def phase(xp: Any, idx: Any, num_steps: int, like: Any) -> Any:
    """Demo phase in ``[0, 1]`` as a float array like ``like``."""
    return xm.to_float(xp, idx, like) / float(max(num_steps - 1, 1))


def noisy_hand_qpos(xp: Any, q: Any, normals: Any, std: Any, quat_blocks: list[int], lo: Any, hi: Any) -> Any:
    """Reference hand qpos plus Gaussian noise (per-column ``std``), quaternion blocks (start
    columns of free/ball joint quaternions) renormalised, limited joints clipped to ``[lo, hi]``
    (``-inf/+inf`` for unlimited columns)."""
    out = q + normals * std
    if quat_blocks:
        parts = []
        last = 0
        for start in quat_blocks:
            parts += [out[..., last:start], xm.quat_normalize(xp, out[..., start:start + 4])]
            last = start + 4
        parts.append(out[..., last:])
        out = xm.cat(xp, parts, -1)
    return xp.minimum(xp.maximum(out, lo), hi)


def noisy_object_pose(xp: Any, pos: Any, quat: Any, n_pos: Any, n_rot: Any, noise: NoiseConfig) -> tuple[Any, Any]:
    """Object pose noise: horizontal ``obj_pos_xy``, vertical ``|N| * obj_pos_z`` upwards only (never
    into the table), rotation by a random rotation vector with std ``obj_rot``."""
    dz = xp.abs(n_pos[..., 2:3]) * noise.obj_pos_z
    dpos = xm.cat(xp, [n_pos[..., 0:2] * noise.obj_pos_xy, dz], -1)
    dq = xm.quat_from_rotvec(xp, n_rot * noise.obj_rot)
    return pos + dpos, xm.quat_normalize(xp, xm.quat_mul(xp, dq, quat))


def clip_actions(xp: Any, actions: Any, limit: float) -> Any:
    return xm.clip(xp, xm.nan_to_zero(xp, actions), -limit, limit)


def ctrl_targets(xp: Any, a: Any, ref_ctrl: Any, scale: Any, lo: Any, hi: Any) -> Any:
    """Actuator targets: reference ``ref_ctrl`` (N, nu) plus residual ``a * scale``, clipped to the
    control range ``[lo, hi]`` (arrays (nu,))."""
    return xp.minimum(xp.maximum(ref_ctrl + a * scale, lo), hi)


def base_targets(xp: Any, a: Any, ref_pos: Any, ref_quat: Any, pos_scale: float, rot_scale: float) -> tuple[Any, Any]:
    """Floating-base target pose: ``pos = ref + a[:3] * pos_scale``, orientation = (rotation vector
    ``a[3:6] * rot_scale`` in the world frame) * reference."""
    pos = ref_pos + a[..., 0:3] * pos_scale
    quat = xm.quat_mul(xp, xm.quat_from_rotvec(xp, a[..., 3:6] * rot_scale), ref_quat)
    return pos, xm.quat_normalize(xp, quat)


def mocap_from_root(xp: Any, root_pos: Any, root_quat: Any, off_pos: Any, off_quat: Any) -> tuple[Any, Any]:
    """Mocap pose that puts the welded hand root at ``root`` when the weld holds the root at
    ``T_mocap_root = (off_pos, off_quat)``: ``q_m = q_r * conj(q_off)``, ``p_m = p_r - R(q_m) p_off``."""
    q_m = xm.quat_mul(xp, root_quat, xm.quat_conj(xp, off_quat))
    p_m = root_pos - xm.quat_rotate(xp, q_m, off_pos)
    return p_m, q_m


def termination(xp: Any, obj_pos_err: Any, obj_rot_err: Any, obj_z: Any, table_z: float, base_pos_err: Any,
                finite: Any, ep_steps: Any, cfg: TerminationConfig) -> tuple[Any, dict[str, Any]]:
    """Early termination (failure): object too far from the reference, rotated too far (when
    ``obj_rot_rad > 0``), fallen below the table, the hand base far from its reference (when
    ``base_pos_m > 0``), or a non-finite simulator state. Error checks start after
    ``grace_steps`` steps of the episode; non-finite states always terminate."""
    armed = ep_steps >= cfg.grace_steps
    false = armed & ~armed
    reasons = {
        "obj_pos": armed & (obj_pos_err > cfg.obj_pos_m),
        "obj_rot": (armed & (obj_rot_err > cfg.obj_rot_rad)) if cfg.obj_rot_rad > 0 else false,
        "fell": (armed & (obj_z < table_z - cfg.fall_below_table_m)) if math.isfinite(table_z) else false,
        "base_pos": (armed & (base_pos_err > cfg.base_pos_m)) if cfg.base_pos_m > 0 else false,
        "nonfinite": ~finite,
    }
    done = reasons["nonfinite"]
    for key in ("obj_pos", "obj_rot", "fell", "base_pos"):
        done = done | reasons[key]
    return done, reasons


def time_outs(xp: Any, idx: Any, ep_steps: Any, num_steps: int, max_episode_steps: int, stop: Any = None) -> Any:
    """Truncation: the demo end is reached, the episode is ``max_episode_steps`` long (> 0), or
    (``stop``, a ``(T,)`` boolean array) the step is inside a long gap of the demonstration."""
    out = idx >= num_steps - 1
    if max_episode_steps > 0:
        out = out | (ep_steps >= max_episode_steps)
    if stop is not None:
        out = out | stop[idx]
    return out


def mask_tracking_terms(terms: dict[str, Any], observed: Any) -> dict[str, Any]:
    """Tracking reward terms times ``observed`` (0/1 per world): no reward for matching an
    interpolated reference; penalties are unchanged."""
    return {key: (value * observed if key in REWARD_TERMS else value) for key, value in terms.items()}


# =============================================================================================
# the environment core
# =============================================================================================


class SimBackend(Protocol):
    num_envs: int

    def read(self) -> dict[str, Any]: ...

    def apply(self, ctrl: Any, mocap_pos: Any | None, mocap_quat: Any | None) -> None: ...

    def reset(self, env_ids: Any, qpos: Any, qvel: Any, ctrl: Any, mocap_pos: Any | None, mocap_quat: Any | None) -> None: ...

    def set_domain(self, env_ids: Any, mass_scale: Any, friction_scale: Any) -> None: ...


class TrackingTask:
    """Backend-independent vectorised tracking environment.

    Args:
        xp: numpy or torch.
        sim: a :class:`SimBackend` with ``num_envs`` worlds.
        model: :class:`~glassesdata_pipeline.rl.scene_adapter.TaskModel` (numpy arrays).
        ref: :class:`~glassesdata_pipeline.rl.reference.DemoReference` (numpy, control rate),
            with ``base_pos``, ``base_quat``, ``tips_pos`` and ``ctrl`` filled in.
        cfg: :class:`TaskConfig`.
    """

    def __init__(self, xp: Any, sim: SimBackend, model: Any, ref: Any, cfg: TaskConfig, seed: int = 0,
                 device: Any = None, dtype: Any = None) -> None:
        self.xp = xp
        self.sim = sim
        self.model = model
        self.cfg = cfg
        self.device = device
        self.num_envs = int(sim.num_envs)
        self.T = int(ref.num_steps)
        if self.T < 2:
            raise ConfigError("the reference has fewer than 2 control steps")
        ref.check_complete()
        torch_ = xm.is_torch(xp)
        if dtype is None:
            dtype = xp.float32 if torch_ else np.float64
        self.dtype = dtype
        self._f = (lambda a: xp.as_tensor(np.asarray(a), dtype=dtype, device=device)) if torch_ else \
            (lambda a: np.asarray(a, dtype=dtype))
        self._i = lambda a: xm.index_array(xp, a, device)
        f, i = self._f, self._i
        m = model
        # reference on the device
        self.ref = {
            "obj_pos": f(ref.obj_pos), "obj_quat": f(ref.obj_quat), "obj_linvel": f(ref.obj_linvel),
            "obj_angvel": f(ref.obj_angvel), "hand_q": f(ref.hand_qpos), "hand_qd": f(ref.hand_qvel),
            "base_pos": f(ref.base_pos), "base_quat": f(ref.base_quat), "tips": f(ref.tips_pos),
            "ctrl": f(ref.ctrl), "grasp": f(ref.grasp),
        }
        self.ref["finger_q"] = self.ref["hand_q"][:, i(m.finger_q_cols)]
        self.origin = f(ref.obj_pos[0])
        long_gap = np.zeros(self.T, bool) if getattr(ref, "long_gap", None) is None else np.asarray(ref.long_gap, bool)
        observed = np.ones(self.T, bool) if getattr(ref, "observed", None) is None else np.asarray(ref.observed, bool)
        self.observed_w = f(observed.astype(np.float64))  # (T,) 1 = measured reference, 0 = interpolated
        self.stop = xp.as_tensor(long_gap, device=device) if torch_ else long_gap  # (T,) bool
        self.start_cands = i(start_candidates(observed, long_gap, self.T, cfg.min_episode_steps))
        # model arrays
        self.qpos0 = f(m.qpos0)
        self.hand_qadr, self.hand_vadr = i(m.hand_qadr), i(m.hand_vadr)
        self.obj_qadr, self.obj_vadr = i(m.obj_qadr), i(m.obj_vadr)
        self.finger_q_sim = i(np.asarray(m.hand_qadr)[np.asarray(m.finger_q_cols, dtype=np.int64)])
        self.finger_v_sim = i(np.asarray(m.hand_vadr)[np.asarray(m.finger_v_cols, dtype=np.int64)])
        self.base_v_sim = i(np.asarray(m.hand_vadr)[np.asarray(m.base_v_cols, dtype=np.int64)])
        self.act_ids = i(m.act_ids)
        self.ctrl_lo, self.ctrl_hi = f(m.ctrl_lo), f(m.ctrl_hi)
        self.ctrl_scale = f(m.ctrl_scale(cfg.action))
        self.q_std = f(m.qpos_noise_std(cfg.noise))
        self.q_lo, self.q_hi = f(m.hand_q_lo), f(m.hand_q_hi)
        self.mocap = m.base_mode == "mocap"
        self.off_pos, self.off_quat = f(m.mocap_offset_pos), f(m.mocap_offset_quat)
        self.num_base_actions = 6 if self.mocap else 0
        self.num_actions = self.num_base_actions + len(m.act_ids)
        self.n_tips = int(ref.tips_pos.shape[1])
        steps = self.T - 1
        self.max_episode_length = min(steps, cfg.max_episode_steps) if cfg.max_episode_steps > 0 else steps
        # buffers (updated in place)
        n = self.num_envs
        self.idx = self._i(np.zeros(n))
        self.ep_steps = self._i(np.zeros(n))
        self.prev_action = f(np.zeros((n, self.num_actions)))
        self.mass_scale = f(np.ones(n))
        self.friction_scale = f(np.ones(n))
        self.rng = xm.Rng(xp, seed, device=device, dtype=dtype)
        self.state: dict[str, Any] = {}
        self.auto_reset = True  # False (play): finished worlds keep their final state

        self.all_ids = self._i(np.arange(n))
        self.sample_domain(self.all_ids)
        self.reset(self.all_ids)
        self.num_policy_obs, self.num_privileged_obs = (int(o.shape[-1]) for o in self.observations())

    # ------------------------------------------------------------------ resets

    def sample_domain(self, env_ids: Any) -> None:
        d = self.cfg.domain
        k = int(env_ids.shape[0])
        mass = self.rng.uniform((k,), *d.mass_scale)
        fric = self.rng.uniform((k,), *d.friction_scale)
        self.mass_scale[env_ids] = mass
        self.friction_scale[env_ids] = fric
        self.sim.set_domain(env_ids, mass, fric)

    def reset(self, env_ids: Any, start: Any | None = None, noise: bool = True) -> None:
        """Reference state initialisation of ``env_ids`` (at random demo steps unless ``start``)."""
        xp, cfg, k = self.xp, self.cfg, int(env_ids.shape[0])
        if k == 0:
            return
        if start is None:
            start = sample_from_candidates(xp, self.rng.uniform((k,)), self.rng.uniform((k,)), self.start_cands,
                                           cfg.p_start_at_zero)
        r = self.ref
        hand_q = r["hand_q"][start]
        obj_pos, obj_quat = r["obj_pos"][start], r["obj_quat"][start]
        if noise:
            hand_q = noisy_hand_qpos(xp, hand_q, self.rng.normal(tuple(hand_q.shape)), self.q_std,
                                     list(self.model.quat_blocks), self.q_lo, self.q_hi)
            obj_pos, obj_quat = noisy_object_pose(xp, obj_pos, obj_quat, self.rng.normal((k, 3)),
                                                  self.rng.normal((k, 3)), cfg.noise)
        qpos = self.qpos0[None, :] + xm.zeros(xp, (k, int(self.qpos0.shape[0])), self.qpos0)
        qpos[:, self.hand_qadr] = hand_q
        qpos[:, self.obj_qadr] = xm.cat(xp, [obj_pos, obj_quat], -1)
        nv = int(self.model.nv)
        qvel = xm.zeros(xp, (k, nv), self.qpos0)
        qvel[:, self.hand_vadr] = r["hand_qd"][start]
        # free joint velocity: linear in the world frame, angular in the body frame
        ang_local = xm.quat_rotate_inv(xp, obj_quat, r["obj_angvel"][start])
        qvel[:, self.obj_vadr] = xm.cat(xp, [r["obj_linvel"][start], ang_local], -1)
        ctrl = xm.zeros(xp, (k, int(self.model.nu)), self.qpos0)
        ctrl[:, self.act_ids] = xp.minimum(xp.maximum(r["ctrl"][start], self.ctrl_lo), self.ctrl_hi)
        mocap_pos = mocap_quat = None
        if self.mocap:
            mocap_pos, mocap_quat = mocap_from_root(xp, r["base_pos"][start], r["base_quat"][start],
                                                    self.off_pos, self.off_quat)
        self.sim.reset(env_ids, qpos, qvel, ctrl, mocap_pos, mocap_quat)
        self.idx[env_ids] = start
        self.ep_steps[env_ids] = 0
        self.prev_action[env_ids] = 0.0
        self.state = {}  # the simulator state changed: the next observations() reads it again

    # ------------------------------------------------------------------ state, observations

    def _read(self) -> dict[str, Any]:
        xp = self.xp
        raw = self.sim.read()
        qpos, qvel = raw["qpos"], raw["qvel"]
        obj_quat = xm.quat_normalize(xp, raw["body_quat"][:, 1])
        s = {
            "finite": xm.isfinite_rows(xp, qpos) & xm.isfinite_rows(xp, qvel),
            "finger_q": qpos[:, self.finger_q_sim],
            "finger_qd": qvel[:, self.finger_v_sim],
            "base_pos": raw["body_pos"][:, 0],
            "base_quat": xm.quat_normalize(xp, raw["body_quat"][:, 0]),
            "base_vel": qvel[:, self.base_v_sim],
            "obj_pos": raw["body_pos"][:, 1],
            "obj_quat": obj_quat,
            "obj_linvel": qvel[:, self.obj_vadr[:3]],
            "obj_angvel": xm.quat_rotate(xp, obj_quat, qvel[:, self.obj_vadr[3:]]),
            "tips": raw["site_pos"],
            "hand_qfrc": raw["qfrc_actuator"][:, self.hand_vadr],
            "hand_qvel": qvel[:, self.hand_vadr],
        }
        self.state = s
        return s

    def _ref_at(self, offset: int) -> dict[str, Any]:
        j = offset_indices(self.xp, self.idx, offset, self.T)
        return {k: v[j] for k, v in self.ref.items()}

    def observations(self) -> tuple[Any, Any]:
        """``(policy (N, n_obs), privileged (N, n_priv))`` for the current state."""
        xp = self.xp
        s = self.state or self._read()
        r_now = self._ref_at(0)
        r1 = self._ref_at(1)
        rk = self._ref_at(self.cfg.lookahead_steps)
        ph = phase(xp, self.idx, self.T, s["obj_pos"][:, 0])
        policy = obs_mod.policy_obs(xp, s, r1, rk, ph, self.prev_action, self.origin, self.cfg.obs)
        priv = obs_mod.privileged_obs(xp, s, r_now, self.mass_scale, self.friction_scale, self.cfg.obs)
        return policy, priv

    # ------------------------------------------------------------------ step

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
        """Apply ``actions (N, num_actions)``; returns ``(policy_obs, privileged_obs, reward,
        terminated, truncated, info)``. Finished worlds are reset before the observations are
        built (their observations are the first of the next episode)."""
        xp, cfg = self.xp, self.cfg
        a = clip_actions(xp, actions, cfg.action.clip)
        r_next = self._ref_at(1)
        ctrl = xm.zeros(xp, (self.num_envs, int(self.model.nu)), self.qpos0)
        a_ctrl = a[:, self.num_base_actions:]
        ctrl[:, self.act_ids] = ctrl_targets(xp, a_ctrl, r_next["ctrl"], self.ctrl_scale, self.ctrl_lo, self.ctrl_hi)
        mocap_pos = mocap_quat = None
        if self.mocap:
            root_pos, root_quat = base_targets(xp, a[:, :6], r_next["base_pos"], r_next["base_quat"],
                                               cfg.action.base_pos, cfg.action.base_rot)
            mocap_pos, mocap_quat = mocap_from_root(xp, root_pos, root_quat, self.off_pos, self.off_quat)
        self.sim.apply(ctrl, mocap_pos, mocap_quat)
        self.idx[:] = xm.clip(xp, self.idx + 1, 0, self.T - 1)
        self.ep_steps[:] = self.ep_steps + 1

        s = self._read()
        r = self._ref_at(0)
        errors = rew_mod.tracking_errors(xp, s, r)
        terms = rew_mod.reward_terms(xp, s, r, errors, a, self.prev_action, cfg.reward)
        terms = mask_tracking_terms(terms, self.observed_w[self.idx])
        terminated, reasons = termination(xp, errors["obj_pos"], errors["obj_rot"], s["obj_pos"][:, 2],
                                          float(self.model.table_z), errors["base_pos"], s["finite"],
                                          self.ep_steps, cfg.termination)
        truncated = time_outs(xp, self.idx, self.ep_steps, self.T, cfg.max_episode_steps, self.stop) & ~terminated
        reward = rew_mod.total_reward(xp, terms, terminated, cfg.reward.weights)
        reward = xm.nan_to_zero(xp, reward)

        info = {"terms": terms, "errors": errors, "reasons": reasons}
        done = terminated | truncated
        self.prev_action[:] = a
        done_ids = xm.nonzero(xp, done)
        if self.auto_reset and int(done_ids.shape[0]) > 0:
            if cfg.domain.resample_on_reset:
                self.sample_domain(done_ids)
            self.reset(done_ids)
            self._read()
        policy, priv = self.observations()
        return policy, priv, reward, terminated, truncated, info
