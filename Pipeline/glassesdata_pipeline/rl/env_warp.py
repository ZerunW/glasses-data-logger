"""GPU backend: MuJoCo Warp (``mujoco_warp`` 4.0.1, ``warp-lang`` 1.19.0) with ``nworld = num_envs``.

Written against the v4.0.1 sources (never run on the development Mac):

* ``mjw.put_model(mjm, batch_sizes={...})`` - per-world copies of ``body_mass``,
  ``body_inertia`` and ``geom_friction`` (fields whose array spec starts with ``"*"``; kernels
  index them ``worldid % shape[0]``) for domain randomisation. ``set_const`` is not called after
  editing them, so ``body_subtreemass`` / ``*_invweight0`` keep their nominal values: subtree COM
  bookkeeping and constraint regularisation use the nominal mass, the dynamics use the randomised one.
* ``mjw.make_data(mjm, nworld, nconmax, njmax)`` - ``nconmax`` contacts and ``njmax`` constraint
  rows per world (``train.warp``); overflow is a per-world bitmask ``d.overflow`` that is read every
  ``overflow_check_interval`` steps (a device-to-host copy).
* ``wp.to_torch`` - zero-copy torch views (``(nworld, nbody)`` vec3 arrays become
  ``(nworld, nbody, 3)``); the views are created once and only written in place.
* ``mjw.step`` x ``decimation`` + ``mjw.kinematics`` captured once in a CUDA graph (``wp.ScopedCapture`` /
  ``wp.capture_launch``); ``ctrl`` and ``mocap_*`` are written in place before each launch.
  Torch and Warp may run on different streams, so the device is synchronised around every launch
  and reset (``train.warp.sync``).
* Resets: ``mjw.reset_data(m, d, reset=mask)`` with a persistent ``wp.bool`` mask of shape
  ``(nworld,)``, then the new ``qpos``/``qvel``/``ctrl``/mocap rows, then ``mjw.kinematics`` (reset
  does not recompute ``xpos``/``site_xpos``). Quaternions are ``[w, x, y, z]`` like MuJoCo.

Model features: MJWarp supports the features used by the composed Shadow Hand model (fixed
tendons, weld equality to a mocap body, position actuators, mesh collision through convex
GJK/EPA, elliptic cones, implicitfast). :func:`scene_adapter.warp_compat` replaces or rejects
what MJWarp lacks (PGS solver, plugins, DISTANCE equalities, a few option flags).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..errors import MissingDependency, PipelineError
from .scene_adapter import TaskModel

log = logging.getLogger(__name__)


def require_warp() -> tuple[Any, Any, Any]:
    try:
        import torch
        import warp as wp
        import mujoco_warp as mjw
    except ImportError as error:
        raise MissingDependency("pip install mujoco-warp==4.0.1 warp-lang==1.19.0 torch (needed for --backend warp)") from error
    if not torch.cuda.is_available():
        raise PipelineError("--backend warp needs a CUDA GPU visible to torch (torch.cuda.is_available() is False)")
    return torch, wp, mjw


class WarpBackend:
    def __init__(self, mj_model: Any, tm: TaskModel, num_envs: int, device: str = "cuda:0", nconmax: int = 96,
                 njmax: int = 512, graph_capture: bool = True, sync: bool = True, overflow_check_interval: int = 100,
                 randomize: bool = True) -> None:
        torch, wp, mjw = require_warp()
        self.torch, self.wp, self.mjw = torch, wp, mjw
        self.xp = torch
        self.device = device
        self.num_envs = n = int(num_envs)
        self.tm = tm
        self.decimation = int(tm.decimation)
        self.sync_streams = bool(sync)
        self.overflow_check_interval = int(overflow_check_interval)
        self.steps = 0
        self.overflow_worlds = 0
        wp.init()
        batch = {"body_mass": n, "body_inertia": n, "geom_friction": n} if randomize else None
        with wp.ScopedDevice(device):
            try:
                self.m = mjw.put_model(mj_model, batch_sizes=batch)
            except NotImplementedError as error:
                raise PipelineError(f"MuJoCo Warp cannot load this model: {error}") from error
            self.d = mjw.make_data(mj_model, nworld=n, nconmax=int(nconmax), njmax=int(njmax))
        to_t = wp.to_torch
        d = self.d
        self.qpos, self.qvel, self.ctrl = to_t(d.qpos), to_t(d.qvel), to_t(d.ctrl)
        self.qfrc_actuator = to_t(d.qfrc_actuator)
        self.xpos, self.xquat, self.site_xpos = to_t(d.xpos), to_t(d.xquat), to_t(d.site_xpos)
        self.overflow = to_t(d.overflow)
        self.mocap_id = int(tm.mocap_id)
        if self.mocap_id >= 0:
            self.mocap_pos, self.mocap_quat = to_t(d.mocap_pos), to_t(d.mocap_quat)
        self.randomize = randomize
        tdev = torch.device(device)
        self.bodies = torch.as_tensor(np.array([tm.root_body_id, tm.object_body_id]), dtype=torch.long, device=tdev)
        self.sites = torch.as_tensor(np.asarray(tm.tip_site_ids), dtype=torch.long, device=tdev)
        self.fric_geoms = torch.as_tensor(np.concatenate([tm.object_geom_ids, tm.hand_geom_ids]).astype(np.int64),
                                          dtype=torch.long, device=tdev)
        if randomize:
            self.body_mass = to_t(self.m.body_mass)  # (n, nbody)
            self.body_inertia = to_t(self.m.body_inertia)  # (n, nbody, 3)
            self.geom_friction = to_t(self.m.geom_friction)  # (n, ngeom, 3)
            for name, view, value in (("body_mass", self.body_mass, mj_model.body_mass),
                                      ("body_inertia", self.body_inertia, mj_model.body_inertia),
                                      ("geom_friction", self.geom_friction, mj_model.geom_friction)):
                if view.shape[0] != n:
                    raise PipelineError(f"put_model(batch_sizes=...) gave {name} shape {tuple(view.shape)}, expected ({n}, ...)")
                view.copy_(torch.as_tensor(np.asarray(value), dtype=view.dtype, device=tdev).expand_as(view))
            ob = int(tm.object_body_id)
            self.base_mass = float(mj_model.body_mass[ob])
            self.base_inertia = torch.as_tensor(np.asarray(mj_model.body_inertia[ob]), dtype=torch.float32, device=tdev)
            self.base_friction = torch.as_tensor(np.asarray(mj_model.geom_friction)[np.asarray(self.fric_geoms.cpu()), 0],
                                                 dtype=torch.float32, device=tdev)
        with wp.ScopedDevice(device):
            self._mask_wp = wp.zeros(n, dtype=wp.bool)
        self._mask = to_t(self._mask_wp)
        self.graph = None
        self._sync()
        with wp.ScopedDevice(device):
            mjw.step(self.m, self.d)  # warm-up: compiles the kernels before capture
            wp.synchronize_device(device)
            if graph_capture:
                with wp.ScopedCapture() as capture:
                    self._substeps()
                self.graph = capture.graph
        self._sync()

    def _substeps(self) -> None:
        """``decimation`` physics steps, then kinematics so ``xpos``/``xquat``/``site_xpos`` match the
        new ``qpos`` (``step`` computes them before integrating)."""
        for _ in range(self.decimation):
            self.mjw.step(self.m, self.d)
        self.mjw.kinematics(self.m, self.d)

    def _sync(self) -> None:
        if self.sync_streams:
            self.wp.synchronize_device(self.device)

    # ------------------------------------------------------------------ protocol

    def read(self) -> dict[str, Any]:
        return {
            "qpos": self.qpos, "qvel": self.qvel, "qfrc_actuator": self.qfrc_actuator,
            "body_pos": self.xpos[:, self.bodies], "body_quat": self.xquat[:, self.bodies],
            "site_pos": self.site_xpos[:, self.sites],
        }

    def apply(self, ctrl: Any, mocap_pos: Any | None, mocap_quat: Any | None) -> None:
        self.ctrl.copy_(ctrl)
        if mocap_pos is not None and self.mocap_id >= 0:
            self.mocap_pos[:, self.mocap_id] = mocap_pos
            self.mocap_quat[:, self.mocap_id] = mocap_quat
        self._sync()
        with self.wp.ScopedDevice(self.device):
            if self.graph is not None:
                self.wp.capture_launch(self.graph)
            else:
                self._substeps()
        self._sync()
        self.steps += 1
        if self.overflow_check_interval > 0 and self.steps % self.overflow_check_interval == 0:
            bad = int((self.overflow != 0).sum().item())
            if bad:
                self.overflow_worlds += bad
                log.warning("MuJoCo Warp overflow in %d worlds (bitmask %s): raise train.warp.nconmax / njmax",
                            bad, int(self.overflow.max().item()))

    def reset(self, env_ids: Any, qpos: Any, qvel: Any, ctrl: Any, mocap_pos: Any | None, mocap_quat: Any | None) -> None:
        self._mask.zero_()
        self._mask[env_ids] = True
        self._sync()
        with self.wp.ScopedDevice(self.device):
            self.mjw.reset_data(self.m, self.d, reset=self._mask_wp)
        self._sync()
        self.qpos[env_ids] = qpos
        self.qvel[env_ids] = qvel
        self.ctrl[env_ids] = ctrl
        if mocap_pos is not None and self.mocap_id >= 0:
            self.mocap_pos[env_ids, self.mocap_id] = mocap_pos
            self.mocap_quat[env_ids, self.mocap_id] = mocap_quat
        self._sync()
        with self.wp.ScopedDevice(self.device):
            self.mjw.kinematics(self.m, self.d)
        self._sync()

    def set_domain(self, env_ids: Any, mass_scale: Any, friction_scale: Any) -> None:
        if not self.randomize:
            return
        ob = int(self.tm.object_body_id)
        self.body_mass[env_ids, ob] = self.base_mass * mass_scale
        self.body_inertia[env_ids, ob] = self.base_inertia[None, :] * mass_scale[:, None]
        rows = env_ids[:, None]
        self.geom_friction[rows, self.fric_geoms[None, :], 0] = self.base_friction[None, :] * friction_scale[:, None]
