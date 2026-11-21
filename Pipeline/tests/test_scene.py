import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/fabricate_sim.py (importlib mode)
from fabricate_sim import fabricate_sim  # noqa: E402
from glassesdata_pipeline import geometry as geo  # noqa: E402
from glassesdata_pipeline.episode import Episode
from glassesdata_pipeline.runner import Runner
from glassesdata_pipeline.sim import frames, mjcf, replay
from glassesdata_pipeline.sim.data import ObjectData, load_episode_data, resolve_table_top


# ------------------------------------------------------------------ camera convention


def _random_poses(rng, n):
    out = []
    for _ in range(n):
        eye = rng.uniform(-1, 1, 3)
        target = eye + rng.normal(size=3)
        T = geo.look_at(eye, target)
        T[:3, :3] = T[:3, :3] @ geo.rot_z(rng.uniform(-math.pi, math.pi))  # arbitrary roll
        out.append(T)
    return out


def test_mujoco_camera_projects_like_opencv():
    rng = np.random.default_rng(0)
    cam = geo.Camera.fov_default(360, 640, 65.0)
    fovy = frames.fovy_deg(cam)
    for T in _random_poses(rng, 5):
        local = np.column_stack([rng.uniform(-0.5, 0.5, 200), rng.uniform(-0.8, 0.8, 200), rng.uniform(0.2, 3.0, 200)])
        P = geo.transform(T, local)
        uv_cv = cam.project(local)
        pos, quat = frames.mujoco_camera_pose(T)
        uv_mj = frames.project_mujoco(P, pos, quat, fovy, cam.width, cam.height)
        assert np.allclose(uv_cv, uv_mj, atol=1e-6)
        # the scene's head body (OpenCV frame) + camera quat 0 1 0 0 gives the same camera pose
        q_head = geo.mat_to_quat(T[:3, :3])
        q_cam = frames.quat_mul(q_head, frames.QUAT_CV_MJ)
        assert min(np.abs(q_cam - quat).max(), np.abs(q_cam + quat).max()) < 1e-9
        R = geo.quat_to_mat(quat)
        assert np.allclose(-R[:, 2], T[:3, 2])  # MuJoCo looks along -z = OpenCV +z
        assert np.allclose(R[:, 1], -T[:3, 1])  # MuJoCo +y up = OpenCV -y
    behind = frames.project_mujoco(np.array([[0.0, 0.0, -1.0]]), *frames.mujoco_camera_pose(np.eye(4)), fovy, 360, 640)
    assert np.all(np.isnan(behind))


def test_fovy_and_matching_camera():
    cam = geo.Camera([[500.0, 0, 300.0], [0, 480.0, 700.0], [0, 0, 1]], [-0.1, 0.02, 0, 0, 0], 720, 1280)
    fovy = frames.fovy_deg(cam)
    assert fovy == pytest.approx(math.degrees(2 * math.atan(640 / 480.0)))
    K = frames.mujoco_K(fovy, 720, 1280)
    assert K[1, 1] == pytest.approx(480.0) and K[0, 2] == pytest.approx(359.5) and K[1, 2] == pytest.approx(639.5)
    match = frames.mujoco_camera_matching(cam)
    assert match.dist.size == 0 and match.fy == pytest.approx(480.0)


def test_quaternion_helpers():
    q0 = geo.mat_to_quat(geo.rot_z(0.2))
    q1 = geo.mat_to_quat(geo.rot_z(1.0))
    mid = frames.slerp(q0, -q1, 0.5)  # sign of q1 must not matter
    assert geo.rotation_angle(geo.quat_to_mat(mid), geo.rot_z(0.6)) < 1e-9
    seq = np.array([q0, -q0, q0, -q1])
    aligned = frames.align_hemisphere(seq)
    assert np.all(np.sum(aligned[1:] * aligned[:-1], axis=1) > 0)
    x = np.sin(np.linspace(0, 3, 50))[:, None]
    x[10] = np.nan
    s = frames.smooth_series(x, 2.0)
    assert np.all(np.isfinite(s)) and abs(s[25, 0] - x[25, 0]) < 0.02
    assert frames.fmt([1.0, -0.0, 1e-7]) == "1 0 1e-07"


# ------------------------------------------------------------------ meshes, mass properties, table


def test_obj_roundtrip_and_polygons(tmp_path):
    V, F = mjcf.box_mesh([0.1, 0.2, 0.3])
    path = mjcf.write_obj(tmp_path / "box.obj", V, F)
    V2, F2 = mjcf.read_obj(path)
    assert np.allclose(V, V2, atol=1e-6) and np.array_equal(F, F2)
    quad = tmp_path / "quad.obj"
    quad.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nf 1/1 2/1 3/1 4/1\nf -4 -2 -1\n")
    Vq, Fq = mjcf.read_obj(quad)
    assert Fq.tolist() == [[0, 1, 2], [0, 2, 3], [0, 2, 3]]


def test_mass_properties_of_a_box():
    size = np.array([0.1, 0.2, 0.3])
    V, F = mjcf.box_mesh(size, center=(0.5, -0.2, 0.1))
    vol, c, I = mjcf.mesh_mass_properties(V, F)
    assert vol == pytest.approx(np.prod(size))
    assert np.allclose(c, [0.5, -0.2, 0.1])
    assert np.allclose(I * 1.0 / vol, mjcf.box_inertia(1.0, size), atol=1e-12)
    vol2, c2, I2 = mjcf.mesh_mass_properties(V, F[:, ::-1])  # inward-oriented
    assert vol2 == pytest.approx(vol) and np.allclose(I2, I)
    # two halves = the whole box (parallel-axis theorem in the scene step)
    from glassesdata_pipeline.steps.scene import object_spec

    halves = [mjcf.box_mesh([0.05, 0.2, 0.3], (cx, 0, 0)) for cx in (-0.025, 0.025)]
    obj = ObjectData("b", np.tile(np.eye(4), (3, 1, 1)), np.ones(3, bool), np.zeros((3, 2), bool),
                     mesh=mjcf.box_mesh(size), collision=halves, props={"mass_kg": 2.0})
    spec = object_spec(obj, 0, 500.0, 0.8)
    assert spec.mass == 2.0 and np.allclose(spec.com, 0) and np.allclose(spec.inertia, mjcf.box_inertia(2.0, size))
    assert spec.friction == (0.8, 0.005, 0.0001)


def test_table_from_polygon():
    R = geo.rot_z(0.4)[:2, :2]
    local = np.array([[-0.6, -0.4], [0.6, -0.4], [0.6, 0.4], [-0.6, 0.4]])
    poly = local @ R.T + [0.7, 0.1]
    table = mjcf.table_from_polygon(poly, -0.7, 0.75)
    assert np.allclose(table.center_xy, [0.7, 0.1], atol=1e-5)
    assert table.floor_z == pytest.approx(-1.45)
    corners = mjcf.table_corners(table)
    d = np.linalg.norm(corners[:, None, :2] - poly[None], axis=-1)
    assert d.min(axis=1).max() < 1e-4  # same rectangle (any corner order / yaw by 90 deg)


def test_table_top_conventions():
    V, F = mjcf.box_mesh([0.1, 0.1, 0.1])
    obj = ObjectData("a", np.tile(geo.make_T(None, [0.5, 0, -0.65]), (2, 1, 1)), np.ones(2, bool), np.zeros((2, 2), bool),
                     mesh=(V, F))
    poly = np.array([[0, -1], [1, -1], [1, 1], [0, 1.0]])
    assert resolve_table_top({"normal": [0, 0, 1], "offset": 0.7}, poly, [obj]) == pytest.approx(-0.7)
    assert resolve_table_top({"normal": [0, 0, 1], "offset": -0.7}, poly, [obj]) == pytest.approx(-0.7)  # n.x = d
    assert resolve_table_top({"normal": [0, 0, 1], "offset": 0.7, "height_m": 0.75}, poly, []) == pytest.approx(-0.7)
    assert resolve_table_top({"height_m": -0.7}, poly, [obj]) == pytest.approx(-0.7)


# ------------------------------------------------------------------ the scene step


def _fabricated(recording_copy):
    fabricate_sim(recording_copy)
    return Episode(recording_copy, {"scene": {"replay": "wireframe", "replay_max_frames": 10, "replay_height": 320,
                                              "replay_sheet_frames": 3}})


def test_scene_step_on_synthetic(recording_copy):
    ep = _fabricated(recording_copy)
    report = Runner(ep).run(steps=["scene"], deps=False)
    assert report.ok, report.table()
    xml = ep.derived / "sim" / "scene.xml"
    root = ET.parse(xml).getroot()
    assert mjcf.check_references(root, xml.parent) == []
    for mesh in root.iter("mesh"):
        assert (xml.parent / "assets" / mesh.get("file")).is_file()
    gt = ep.gt
    K = len(gt.object_ids)
    stats = mjcf.scene_stats(root)
    assert stats["mocap_bodies"] == 2 * 21 + 2 * 3 + 1
    assert stats["bodies"] == stats["mocap_bodies"] + K + 1  # + objects + table
    assert stats["free_joints"] == K and stats["cameras"] == 2
    assert root.find("option").get("timestep") == "0.002" and root.find("option").get("gravity") == "0 0 -9.81"
    names = {b.get("name"): b for b in root.iter("body")}
    for k, oid in enumerate(gt.object_ids):
        body = names[mjcf.object_body(oid)]
        pos = np.array(body.get("pos").split(), float)
        assert np.allclose(pos, gt["T_world_obj"][0, k, :3, 3], atol=1e-5)
        quat = np.array(body.get("quat").split(), float)
        assert geo.rotation_angle(geo.quat_to_mat(quat), gt["T_world_obj"][0, k, :3, :3]) < 1e-4
        assert body.find("freejoint").get("name") == mjcf.object_joint(oid)
        assert float(body.find("inertial").get("mass")) > 0
    assert len(names[mjcf.object_body("box_b")].findall("geom")) == 3  # visual + 2 collision parts
    head = names[mjcf.HEAD_BODY]
    cam = head.find("camera")
    assert cam.get("quat") == "0 1 0 0"
    assert float(cam.get("fovy")) == pytest.approx(frames.fovy_deg(ep.camera()), abs=1e-3)
    table = names[mjcf.TABLE_BODY].find("geom")
    half_h = float(table.get("size").split()[2])
    top = float(names[mjcf.TABLE_BODY].get("pos").split()[2]) + half_h
    assert top == pytest.approx(float(gt["table_top_z"]), abs=1e-5)
    assert 2 * half_h == pytest.approx(0.75)
    floor = next(g for g in root.iter("geom") if g.get("name") == mjcf.FLOOR_GEOM)
    assert float(floor.get("pos").split()[2]) == pytest.approx(float(gt["floor_z"]), abs=1e-4)
    for geom in root.iter("geom"):
        if geom.get("class") == "gd_human":
            assert geom.get("class") and root.find("default/default[@class='gd_human']/geom").get("contype") == "0"
    info = json.loads((ep.derived / "sim" / "scene.json").read_text())
    assert info["replay"]["renderer"] == "wireframe" and info["replay"]["frames"] == 10
    assert info["meta"]["step"] == "scene"
    import av

    with av.open(str(ep.derived / "sim" / "replay.mp4")) as container:
        stream = container.streams.video[0]
        decoded = sum(1 for _ in container.decode(stream))
        assert decoded == 10 and stream.codec_context.height == 320
    assert (ep.derived / "sim" / "replay_sheet.png").stat().st_size > 10_000
    assert Runner(ep).run(steps=["scene"], deps=False).by_name()["scene"].status == "cached"


def test_scene_without_objects_or_arms(recording_copy):
    fabricate_sim(recording_copy, write=("camera_traj", "hands_world"))
    ep = Episode(recording_copy, {"scene": {"replay": "none"}})
    from glassesdata_pipeline.steps import scene

    summary = scene.run(ep, ep.config)
    root = ET.parse(ep.derived / "sim" / "scene.xml").getroot()
    assert summary["objects"] == 0 and mjcf.check_references(root, ep.derived / "sim") == []
    assert mjcf.scene_stats(root)["mocap_bodies"] == 2 * 21 + 1
    floor = next(g for g in root.iter("geom") if g.get("name") == mjcf.FLOOR_GEOM)
    assert float(floor.get("pos").split()[2]) == pytest.approx(-1.45, abs=1e-6)  # first camera at the origin


def test_wireframe_renderer_draws_where_the_camera_looks(recording_copy):
    fabricate_sim(recording_copy)
    data = load_episode_data(Episode(recording_copy), with_gt=False)
    table = mjcf.table_from_polygon(data.table_polygon, data.table_top_z, 0.75)
    cam = data.camera
    fovy = frames.fovy_deg(cam)
    renderer = replay.WireframeRenderer(replay.scene_geometry(data, table, table.floor_z), fovy, cam.width, cam.height)
    state = replay.EpisodeStates(data)(0)
    img = renderer.render(state)
    assert img.shape == (cam.height, cam.width, 3)
    pos, quat = frames.mujoco_camera_pose(state.T_world_cam)
    for k, obj in enumerate(data.objects):
        uv = frames.project_mujoco(obj.T_world_obj[0, :3, 3], pos, quat, fovy, cam.width, cam.height)
        if not (0 <= uv[0] < cam.width and 0 <= uv[1] < cam.height):
            continue
        r, g, b, _ = mjcf.object_color(k)
        pixel = img[int(round(uv[1])), int(round(uv[0]))].astype(float)
        hue = np.array([b, g, r]) / max(b, g, r)
        assert np.allclose(pixel / pixel.max(), hue, atol=0.25)  # the object's colour (shaded) at its centre


def test_scene_without_table_gets_an_inferred_support(recording_copy, caplog):
    """No objects/table.json: a support box under the objects (top = their lowest resting bottom)
    instead of a floor 1.45 m below the camera; a world that is not gravity-aligned is recorded."""
    from glassesdata_pipeline.sim.data import gravity_aligned
    from glassesdata_pipeline.steps import scene

    fabricate_sim(recording_copy)
    ep = Episode(recording_copy, {"scene": {"replay": "none"}})
    (ep.derived / "objects" / "table.json").unlink()
    traj = ep.load_npz("camera_traj")
    assert gravity_aligned(ep) is None  # the fabricated file has no flag
    ep.save_npz("camera_traj", {**traj.pop("meta"), "gravity_aligned": False}, **traj)
    assert gravity_aligned(ep) is False
    scene.run(ep, ep.config)
    info = json.loads((ep.derived / "sim" / "scene.json").read_text())
    assert info["table"]["support"] == "inferred" and info["gravity_aligned"] is False
    assert "not gravity-aligned" in caplog.text
    gt = ep.gt
    bottoms = []
    for k in range(len(gt.object_ids)):
        V = mjcf.box_mesh(gt["object_sizes"][k])[0]
        bottoms.append(geo.transform(gt["T_world_obj"][0, k], V)[:, 2].min())
    data = load_episode_data(ep, with_gt=False)
    est = min(geo.transform(o.T_world_obj[o.first_valid], mjcf.box_mesh(o.extents())[0])[:, 2].min() for o in data.objects)
    assert info["table"]["top_z"] == pytest.approx(est, abs=1e-6)
    assert info["table"]["top_z"] == pytest.approx(min(bottoms), abs=0.01)
    assert info["floor_z"] == pytest.approx(-1.45, abs=1e-6) and info["table"]["floor_z"] == info["floor_z"]
    root = ET.parse(ep.derived / "sim" / "scene.xml").getroot()
    names = {b.get("name"): b for b in root.iter("body")}
    assert mjcf.TABLE_BODY in names
