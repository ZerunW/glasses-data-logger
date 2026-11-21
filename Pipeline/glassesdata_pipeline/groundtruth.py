"""Ground truth of a synthetic recording (``<recording>/gt/``, PIPELINE.md section 8).

Files:

* ``gt/gt.npz`` - per-frame arrays, all in the pipeline's world frame (z up, origin at the
  first camera centre, first optical axis along +x): ``T_world_cam[T,4,4]``, ``K[3,3]``,
  ``joints_world[T,2,21,3]``, ``joints_cam[T,2,21,3]``, ``visible[T,2,21]``,
  ``shoulder_world[T,2,3]``, ``elbow_world[T,2,3]``, ``T_world_obj[T,K,4,4]``, ``R_cam_imu[3,3]``,
  ``imu_time_offset_s``, ``gyro_bias[3]``, ``accel_bias[3]``, ``table_height_m``,
  ``object_sizes[K,3]`` plus extras documented in ``gt/meta.json`` (``t``, ``T_world_wrist``,
  ``elbow_swivel``, ``flexion``, ``abduction``, ``grasped``, ``floor_z``, ``table_top_z``, ...).
* ``gt/depth_f16.npy`` - ``[T,h,w]`` float16 z-depth in metres at full resolution.
* ``gt/masks.npz`` - ``frames[T]``, ``packed[T,K,ceil(h*w/8)]`` (``np.packbits`` of each flattened
  bool mask), ``shape`` = (h, w), ``object_ids[K]``.
* ``gt/meta.json`` - generator settings, units and conventions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def unpack_mask(packed: np.ndarray, shape: tuple[int, int] | np.ndarray) -> np.ndarray:
    """Inverse of ``np.packbits(mask.reshape(-1))``."""
    h, w = (int(v) for v in shape)
    return np.unpackbits(np.asarray(packed, dtype=np.uint8), count=h * w).reshape(h, w).astype(bool)


def pack_mask(mask: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(mask, dtype=bool).reshape(-1))


class GroundTruth:
    """Lazy access to ``gt/``: ``gt["T_world_cam"]`` or ``gt.T_world_cam``, ``gt.depth()``,
    ``gt.masks()``, ``gt.mask(frame, k)``, ``gt.meta`` (dict), ``gt.object_ids``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._arrays: dict[str, Any] | None = None
        self._masks: dict[str, Any] | None = None
        meta_path = self.path / "meta.json"
        self.meta: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.is_file() else {}

    # -- gt.npz ------------------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if self._arrays is None:
            with np.load(self.path / "gt.npz", allow_pickle=False) as data:
                self._arrays = {k: (data[k].item() if data[k].ndim == 0 else data[k]) for k in data.files}
        return self._arrays

    def __getitem__(self, key: str) -> Any:
        return self._load()[key]

    def __contains__(self, key: object) -> bool:
        return key in self._load()

    def keys(self) -> list[str]:
        return list(self._load())

    def get(self, key: str, default: Any = None) -> Any:
        return self._load().get(key, default)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._load()[name]
        except KeyError:
            raise AttributeError(f"gt has no array {name!r}") from None

    @property
    def n_frames(self) -> int:
        return int(self["T_world_cam"].shape[0])

    @property
    def object_ids(self) -> list[str]:
        ids = self.get("object_ids")
        if ids is not None:
            return [str(v) for v in np.atleast_1d(ids)]
        return [f"obj{k}" for k in range(self["T_world_obj"].shape[1])]

    # -- depth and masks ---------------------------------------------------------------------

    def depth(self) -> np.ndarray:
        """Memory-mapped ``[T,h,w]`` float16 depth (metres, z-depth, full resolution)."""
        return np.load(self.path / "depth_f16.npy", mmap_mode="r")

    def masks(self) -> dict[str, Any]:
        """``{"frames", "packed", "shape", "object_ids"}`` from ``gt/masks.npz``."""
        if self._masks is None:
            with np.load(self.path / "masks.npz", allow_pickle=False) as data:
                self._masks = {k: data[k] for k in data.files}
        return self._masks

    def mask(self, frame: int, k: int | str) -> np.ndarray:
        """Bool ``(h, w)`` mask of object ``k`` (index or id) in video frame ``frame``."""
        masks = self.masks()
        if isinstance(k, str):
            k = [str(v) for v in masks["object_ids"]].index(k)
        row = int(np.searchsorted(masks["frames"], frame))
        if row >= len(masks["frames"]) or masks["frames"][row] != frame:
            raise KeyError(f"no gt mask for frame {frame}")
        return unpack_mask(masks["packed"][row, k], masks["shape"])
