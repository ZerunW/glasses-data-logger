"""Step ``objects`` on the session's synthetic recording with fabricated upstream files
(``tests/fabricate_objects.py``): formats of every section-5 output, the automatic prompt, the
reconstructed size, the poses, the grasp labels, the table, manual prompts and merging, and the
evaluation group."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fabricate_objects import fabricate_upstream  # noqa: E402

from glassesdata_pipeline import geometry as geo  # noqa: E402
from glassesdata_pipeline.core import hull as hl  # noqa: E402
from glassesdata_pipeline.episode import Episode  # noqa: E402
from glassesdata_pipeline.runner import Runner  # noqa: E402

# The 2 s test clip runs the 8 s choreography four times faster: the hand is faster at onset.
OVERRIDES = {"depth": {"downsample": 2}, "objects": {"mask_backend": "gt", "grasp": {"slow_mps": 0.8}}}


@pytest.fixture(scope="module")
def objects_run(tmp_path_factory, synth_recording):
    import shutil

    rec = tmp_path_factory.mktemp("objects") / "rec"
    shutil.copytree(synth_recording, rec, ignore=shutil.ignore_patterns("derived"))
    ep = Episode(rec, OVERRIDES)
    fabricate_upstream(ep)
    report = Runner(ep).run(["objects"], deps=False)
    assert report.ok, report.table()
    return ep, report


def test_outputs_follow_the_contract(objects_run):
    ep, report = objects_run
    assert report.by_name()["objects"].status == "ran"
    index = ep.load_json("objects/index.json")
    assert isinstance(index, list) and len(index) == 1
    item = index[0]
    assert set(item) >= {"id", "prompt", "first_frame", "last_frame"} and item["id"] == "obj0"
    assert item["prompt"]["source"] == "grasp" and {"frame", "x", "y", "box"} <= set(item["prompt"])
    T = ep.gt.n_frames
    folder = ep.derived / "objects" / "obj0"
    masks = ep.load_npz("objects/obj0/masks")
    h, w = (int(v) for v in masks["shape"])
    depth = np.load(ep.derived / "depth" / "depth_f16.npy", mmap_mode="r")
    assert (h, w) == depth.shape[1:]
    assert masks["packed"].shape == (len(masks["frames"]), (h * w + 7) // 8) and masks["packed"].dtype == np.uint8
    assert masks["frames"][0] == item["first_frame"] and masks["frames"][-1] == item["last_frame"]
    V, F = hl.read_obj(folder / "mesh.obj")
    assert hl.is_closed_manifold(F)
    assert np.linalg.norm(hl.mesh_volume_centroid(V, F)[1]) < 1e-6  # origin at the centroid
    parts = sorted((folder / "collision").glob("part_*.obj"))
    assert parts and all(hl.is_closed_manifold(hl.read_obj(p)[1]) for p in parts)
    poses = ep.load_npz("objects/obj0/poses")
    assert poses["T_world_obj"].shape == (T, 4, 4) and poses["valid"].shape == (T,) and poses["valid"].dtype == bool
    assert poses["fitness"].shape == (T,) and poses["grasped"].shape == (T, 2) and poses["grasped"].dtype == bool
    assert np.allclose(poses["T_world_obj"][:, :3, :3] @ np.swapaxes(poses["T_world_obj"][:, :3, :3], 1, 2), np.eye(3), atol=1e-9)
    props = ep.load_json("objects/obj0/props.json")
    assert {"volume_m3", "mass_kg", "com", "friction", "meta"} <= set(props)
    assert props["mass_kg"] == pytest.approx(500.0 * props["volume_m3"]) and np.linalg.norm(props["com"]) < 1e-9
    table = ep.load_json("objects/table.json")
    assert {"normal", "offset", "height_m", "polygon", "meta"} <= set(table) and table["found"]
    assert table["meta"]["step"] == "objects"
    # the runner caches the step
    again = Runner(ep).run(["objects"], deps=False)
    assert again.by_name()["objects"].status == "cached"


def test_prompt_shape_poses_grasp_and_table(objects_run):
    ep, _ = objects_run
    gt = ep.gt
    item = ep.load_json("objects/index.json")[0]
    p = item["prompt"]
    assert gt.mask(p["frame"], 0)[int(round(p["y"])), int(round(p["x"]))]  # inside box_a
    props = ep.load_json("objects/obj0/props.json")
    ext = np.array(props["extent_m"])
    size = gt["object_sizes"][0]
    assert abs(ext[2] - size[2]) < 0.03
    assert np.all(np.abs(np.sort(ext[:2]) - np.sort(size[:2])) < 0.03)
    poses = ep.load_npz("objects/obj0/poses")
    est = poses["T_world_obj"]
    true = gt["T_world_obj"][:, 0]
    err = np.linalg.norm(est[:, :3, 3] - true[:, :3, 3], axis=1)
    assert np.median(err) < 0.03 and err.max() < 0.06
    d_est = est[:, :3, :3] @ est[0, :3, :3].T
    d_true = true[:, :3, :3] @ true[0, :3, :3].T
    assert np.degrees(geo.rotation_angle(d_est, d_true)).max() < 15.0
    g_true = gt["grasped"][:, 0]
    assert np.mean(poses["grasped"] == g_true) > 0.85 and poses["grasped"][:, 1].any() and not poses["grasped"][:, 0].any()
    table = ep.load_json("objects/table.json")
    assert abs(table["top_z"] - float(gt["table_top_z"])) < 0.015
    assert abs(table["height_m"] - float(gt["table_height_m"])) < 0.03
    grasps = ep.load_json("objects/grasps.json")
    assert len(grasps["episodes"]) == 1 and grasps["episodes"][0]["hand"] == 1


def test_eval_reports_object_metrics(objects_run):
    from glassesdata_pipeline.eval.report import evaluate

    ep, _ = objects_run
    m = evaluate(ep)["objects"]
    assert m["status"] == "ok"
    obj = m["objects"]["obj0"]
    assert obj["matched_gt"] == "box_a" and obj["translation_m"]["median"] < 0.03
    assert abs(obj["size"]["error_m"][2]) < 0.03 and obj["size"]["true_m"][2] == pytest.approx(0.11)
    assert obj["prompt_hits"]["auto_hit_rate"] == 1.0 and obj["prompt_hits"]["n_auto"] == 1
    assert obj["grasp"]["frame_accuracy"] > 0.85 and obj["grasp"]["recall"] > 0.8
    assert {"static", "grasped"} <= set(obj["phases"])
    assert m["table"]["polygon_iou"] > 0.5 and abs(m["table"]["top_z_error_m"]) < 0.015


def test_manual_prompts_merge_and_add_objects(tmp_path, synth_recording):
    """A manual click on the grasped box merges with its grasp prompt; a click on the other box
    adds a second (static) object."""
    import shutil

    rec = tmp_path / "rec"
    shutil.copytree(synth_recording, rec, ignore=shutil.ignore_patterns("derived"))
    ep = Episode(rec, OVERRIDES)
    gt = ep.gt
    cam = ep.camera()
    clicks = []
    for k in (0, 1):
        f = 2
        uv = cam.project(geo.transform(geo.inv_T(gt["T_world_cam"][f]), gt["T_world_obj"][f, k, :3, 3]))
        clicks.append({"frame": f, "x": float(uv[0]), "y": float(uv[1])})
    ep = Episode(rec, {**OVERRIDES, "objects": {**OVERRIDES["objects"], "prompts": clicks}})
    fabricate_upstream(ep)
    report = Runner(ep).run(["objects"], deps=False)
    assert report.ok, report.table()
    index = ep.load_json("objects/index.json")
    assert len(index) == 2
    by_sources = sorted(sorted(p["source"] for p in item["prompts"]) for item in index)
    assert by_sources == [["grasp", "manual"], ["manual"]]
    static = next(item for item in index if len(item["prompts"]) == 1)
    poses = ep.load_npz(f"objects/{static['id']}/poses")
    true = gt["T_world_obj"][:, 1]
    err = np.linalg.norm(poses["T_world_obj"][:, :3, 3] - true[:, :3, 3], axis=1)
    assert np.median(err) < 0.03 and not poses["grasped"].any()
    ext = np.array(ep.load_json(f"objects/{static['id']}/props.json")["extent_m"])
    assert np.all(np.abs(np.sort(ext[:2]) - np.sort(gt["object_sizes"][1][:2])) < 0.04)


def test_heavy_backends_are_lazy_and_report_missing_dependencies(tmp_path):
    from glassesdata_pipeline.backends import coacd as coacd_b
    from glassesdata_pipeline.backends import image_to_3d, mesh_open3d
    from glassesdata_pipeline.backends import sam2 as sam2_b
    from glassesdata_pipeline.errors import MissingDependency

    for name, call in (
        ("open3d", lambda: mesh_open3d.reconstruct(np.zeros((10, 3)), np.zeros((10, 3)), {})),
        ("coacd", lambda: coacd_b.decompose(np.zeros((4, 3)), np.zeros((1, 3), int), {})),
        ("torch", lambda: sam2_b.load_predictor({})),
    ):
        try:
            __import__(name)
        except ImportError:
            with pytest.raises(MissingDependency):
                call()
    assert callable(image_to_3d.generate_aligned)
