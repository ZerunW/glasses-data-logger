"""CPU MuJoCo backend: ``num_envs`` independent ``MjData`` stepped in a Python loop.

For smoke tests (16-64 worlds), ``--play`` and debugging; implements the backend protocol of
:mod:`.task`. Every world gets its own ``MjModel`` copy so the per-world domain randomisation
(object mass and inertia, hand/object sliding friction) can edit model fields; after editing,
``mj_setConst`` recomputes the derived constants. ``mujoco.rollout`` is not used: it runs open-loop
control sequences, while the policy needs the state after every control step.

Outputs are ``xp`` arrays (torch tensors on ``device`` for training; numpy works too).
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

from .scene_adapter import TaskModel, require_mujoco

log = logging.getLogger(__name__)


class MujocoBackend:
    def __init__(self, mj_model: Any, tm: TaskModel, num_envs: int, xp: Any = None, device: Any = "cpu",
                 dtype: Any = None) -> None:
        mujoco = require_mujoco()
        if xp is None:
            import torch as xp  # noqa: N813
        self.mujoco = mujoco
        self.xp = xp
        self.device = device
        self.torch = getattr(xp, "__name__", "") == "torch"
        self.dtype = dtype or (xp.float32 if self.torch else np.float64)
        self.num_envs = int(num_envs)
        self.tm = tm
        self.decimation = int(tm.decimation)
        self.models = [mj_model] + [self._copy_model(mj_model) for _ in range(self.num_envs - 1)]
        self.datas = [mujoco.MjData(m) for m in self.models]
        self.bodies = np.array([tm.root_body_id, tm.object_body_id], dtype=np.int64)
        self.sites = np.asarray(tm.tip_site_ids, dtype=np.int64)
        self.fric_geoms = np.concatenate([tm.object_geom_ids, tm.hand_geom_ids]).astype(np.int64)
        self.base_mass = float(mj_model.body_mass[tm.object_body_id])
        self.base_inertia = np.array(mj_model.body_inertia[tm.object_body_id])
        self.base_friction = np.array(mj_model.geom_friction[self.fric_geoms, 0])
        self.mocap_id = int(tm.mocap_id)
        for m, d in zip(self.models, self.datas):
            mujoco.mj_forward(m, d)

    @staticmethod
    def _copy_model(m: Any) -> Any:
        return copy.deepcopy(m)

    # ------------------------------------------------------------------ conversion

    def _out(self, a: np.ndarray) -> Any:
        if self.torch:
            return self.xp.as_tensor(a, dtype=self.dtype, device=self.device)
        return np.asarray(a, dtype=self.dtype)

    def _in(self, a: Any) -> np.ndarray | None:
        if a is None:
            return None
        if self.torch:
            return a.detach().to("cpu").double().numpy()
        return np.asarray(a, dtype=np.float64)

    # ------------------------------------------------------------------ protocol

    def read(self) -> dict[str, Any]:
        ds = self.datas
        return {
            "qpos": self._out(np.stack([d.qpos for d in ds])),
            "qvel": self._out(np.stack([d.qvel for d in ds])),
            "qfrc_actuator": self._out(np.stack([d.qfrc_actuator for d in ds])),
            "body_pos": self._out(np.stack([d.xpos[self.bodies] for d in ds])),
            "body_quat": self._out(np.stack([d.xquat[self.bodies] for d in ds])),
            "site_pos": self._out(np.stack([d.site_xpos[self.sites] for d in ds])),
        }

    def apply(self, ctrl: Any, mocap_pos: Any | None, mocap_quat: Any | None) -> None:
        mj = self.mujoco
        ctrl = self._in(ctrl)
        mp, mq = self._in(mocap_pos), self._in(mocap_quat)
        for i, (m, d) in enumerate(zip(self.models, self.datas)):
            d.ctrl[:] = ctrl[i]
            if mp is not None and self.mocap_id >= 0:
                d.mocap_pos[self.mocap_id] = mp[i]
                d.mocap_quat[self.mocap_id] = mq[i]
            mj.mj_step(m, d, nstep=self.decimation)
            mj.mj_kinematics(m, d)  # mj_step leaves xpos/site_xpos at the pre-integration state

    def reset(self, env_ids: Any, qpos: Any, qvel: Any, ctrl: Any, mocap_pos: Any | None, mocap_quat: Any | None) -> None:
        mj = self.mujoco
        ids = self._ids(env_ids)
        qpos, qvel, ctrl = self._in(qpos), self._in(qvel), self._in(ctrl)
        mp, mq = self._in(mocap_pos), self._in(mocap_quat)
        for k, i in enumerate(ids):
            m, d = self.models[i], self.datas[i]
            mj.mj_resetData(m, d)
            d.qpos[:] = qpos[k]
            d.qvel[:] = qvel[k]
            d.ctrl[:] = ctrl[k]
            if mp is not None and self.mocap_id >= 0:
                d.mocap_pos[self.mocap_id] = mp[k]
                d.mocap_quat[self.mocap_id] = mq[k]
            mj.mj_forward(m, d)

    def set_domain(self, env_ids: Any, mass_scale: Any, friction_scale: Any) -> None:
        mj = self.mujoco
        ids = self._ids(env_ids)
        ms, fs = self._in(mass_scale), self._in(friction_scale)
        ob = self.tm.object_body_id
        for k, i in enumerate(ids):
            m = self.models[i]
            m.body_mass[ob] = self.base_mass * ms[k]
            m.body_inertia[ob] = self.base_inertia * ms[k]
            m.geom_friction[self.fric_geoms, 0] = self.base_friction * fs[k]
            if hasattr(mj, "mj_setConst"):
                mj.mj_setConst(m, self.datas[i])

    def _ids(self, env_ids: Any) -> list[int]:
        if self.torch:
            return [int(v) for v in env_ids.detach().to("cpu").tolist()]
        return [int(v) for v in np.asarray(env_ids).reshape(-1)]
