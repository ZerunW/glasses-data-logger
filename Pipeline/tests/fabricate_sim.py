"""Fabricate the inputs of the simulation steps (``scene``, ``retarget``, ``physics``, ``viz``) of a
synthetic recording from its ground truth, in the PIPELINE.md section-5 formats, so those steps
can be developed and tested before ``world``/``arms``/``objects`` exist.

>>> import sys; sys.path.insert(0, "Pipeline/tests")       # pytest puts tests/ on sys.path itself
>>> from fabricate_sim import fabricate_sim
>>> info = fabricate_sim("local/tmp/synth_demo")              # runs ingest first when needed
>>> # then: python -m glassesdata_pipeline run local/tmp/synth_demo --steps scene,viz --no-deps

Written into ``derived/`` (meta ``backend = "fabricated-gt"``):

* ``camera_traj.npz`` - ``T_world_cam`` = GT, ``valid`` all true.
* ``hands_world.npz`` - ``joints`` = ``raw`` = GT ``joints_world`` (+ optional noise),
  ``observed`` = the wrist and at least 6 joints are visible, ``filled`` = not observed,
  ``bone_lengths`` from the profile.
* ``arms.npz`` - GT shoulder/elbow, ``wrist`` = GT joint 0, ``elbow_conf`` 0.9, ``method`` 1.
* ``objects/index.json`` + per object ``mesh.obj`` (a box of ``object_sizes``, object frame at the
  centroid), ``collision/part_*.obj`` (the box, or ``split_parts`` halves along x),
  ``poses.npz`` (GT ``T_world_obj``, ``valid``, ``fitness`` 1, ``grasped`` = GT), ``props.json``
  (``density`` kg/m^3, ``friction``).
* ``objects/table.json`` - ``normal [0, 0, 1]``, ``offset = -top_z`` (plane ``n . x + offset = 0``),
  ``height_m = top_z`` (world z of the table top; the contract leaves the meaning open, the
  ``scene`` step does not rely on it), ``polygon`` = GT table corners (world xy).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from glassesdata_pipeline.episode import Episode
from glassesdata_pipeline.runner import Runner
from glassesdata_pipeline.sim import mjcf

BACKEND = "fabricated-gt"


def _meta(ep: Episode, step: str) -> dict[str, Any]:
    meta = ep.step_meta(backend=BACKEND)
    meta["step"] = step
    return meta


def fabricate_sim(
    recording: str | Path | Episode,
    *,
    density: float = 400.0,
    friction: float = 0.9,
    joint_noise_m: float = 0.0,
    split_parts: tuple[str, ...] = ("box_b",),
    seed: int = 0,
    write: tuple[str, ...] = ("camera_traj", "hands_world", "arms", "objects"),
) -> dict[str, Any]:
    """Write the fabricated files; returns a summary."""
    ep = recording if isinstance(recording, Episode) else Episode(recording)
    gt = ep.gt
    if gt is None:
        raise ValueError(f"{ep.path} is not a synthetic recording (no gt/)")
    if not ep.has("timeline.npz"):
        report = Runner(ep).run(steps=["ingest"], deps=False)
        if not report.ok:
            raise RuntimeError(report.table())
    rng = np.random.default_rng(seed)
    T_world_cam = np.asarray(gt["T_world_cam"], dtype=np.float64)
    T = T_world_cam.shape[0]
    out: dict[str, Any] = {"frames": T}

    if "camera_traj" in write:
        ep.save_npz("camera_traj", _meta(ep, "world"), T_world_cam=T_world_cam, valid=np.ones(T, bool))
    joints = np.asarray(gt["joints_world"], dtype=np.float64)
    if joint_noise_m > 0:
        joints = joints + rng.normal(scale=joint_noise_m, size=joints.shape)
    visible = np.asarray(gt["visible"], dtype=bool)
    observed = visible[..., 0] & (visible.sum(axis=-1) >= 6)
    if "hands_world" in write:
        bones = np.asarray(ep.profile()["bone_lengths_m"], dtype=np.float64)
        ep.save_npz("hands_world", _meta(ep, "world"), joints=joints, raw=joints, observed=observed, filled=~observed,
                    bone_lengths=np.stack([bones, bones]))
        out["observed_fraction"] = [round(float(v), 3) for v in observed.mean(axis=0)]
    if "arms" in write:
        ep.save_npz("arms", _meta(ep, "arms"), shoulder=np.asarray(gt["shoulder_world"]), elbow=np.asarray(gt["elbow_world"]),
                    wrist=joints[:, :, 0], elbow_conf=np.full((T, 2), 0.9), method=np.ones((T, 2), np.int8))
    if "objects" in write:
        sizes = np.asarray(gt["object_sizes"], dtype=np.float64)
        T_obj = np.asarray(gt["T_world_obj"], dtype=np.float64)
        grasped = np.asarray(gt["grasped"], dtype=bool)  # (T, K, 2)
        index = []
        for k, oid in enumerate(gt.object_ids):
            folder = ep.derived / "objects" / oid
            (folder / "collision").mkdir(parents=True, exist_ok=True)
            for old in (folder / "collision").glob("part_*.obj"):
                old.unlink()
            V, F = mjcf.box_mesh(sizes[k])
            mjcf.write_obj(folder / "mesh.obj", V, F, f"{oid}: box {sizes[k].tolist()} (fabricated from gt)")
            if oid in split_parts:
                half = sizes[k] * np.array([0.5, 1.0, 1.0])
                for j, cx in enumerate((-sizes[k][0] / 4, sizes[k][0] / 4)):
                    Vp, Fp = mjcf.box_mesh(half, (cx, 0.0, 0.0))
                    mjcf.write_obj(folder / "collision" / f"part_{j}.obj", Vp, Fp)
            else:
                mjcf.write_obj(folder / "collision" / "part_0.obj", V, F)
            ep.save_npz(f"objects/{oid}/poses", _meta(ep, "objects"), T_world_obj=T_obj[:, k], valid=np.ones(T, bool),
                        fitness=np.ones(T), grasped=grasped[:, k])
            volume = float(np.prod(sizes[k]))
            ep.save_json(f"objects/{oid}/props.json", {"volume_m3": volume, "mass_kg": volume * density,
                                                        "com": [0.0, 0.0, 0.0], "friction": friction},
                         _meta(ep, "objects"))
            index.append({"id": oid, "prompt": {"source": BACKEND}, "first_frame": 0, "last_frame": T - 1})
        (ep.derived / "objects" / "index.json").write_text(json.dumps(index, indent=2))
        top = float(gt["table_top_z"])
        polygon = np.asarray(gt["table_polygon"], dtype=np.float64)
        ep.save_json("objects/table.json", {"normal": [0.0, 0.0, 1.0], "offset": -top, "height_m": top,
                                            "polygon": polygon.tolist()}, _meta(ep, "objects"))
        out["objects"] = list(gt.object_ids)
    return out


def hand_slots_observed(ep: Episode) -> np.ndarray:
    """``(2,)`` fraction of frames each hand is observed in the fabricated ``hands_world``."""
    return np.asarray(ep.load_npz("hands_world")["observed"], dtype=float).mean(axis=0)

