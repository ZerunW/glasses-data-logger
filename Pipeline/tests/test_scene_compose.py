"""Composition of scenes with Menagerie robot hands (pure XML; MuJoCo only in the skipped test).

The fake Menagerie below reproduces the structure of ``shadow_hand/right_hand.xml`` /
``left_hand.xml`` (default classes, materials, meshes named by file, the forearm -> wrist -> palm
chain, five fingers with the same body names and offsets, the fixed J1+J2 tendons, wrist and finger
actuators, contact excludes, the unprefixed ``grasp_site``) with tiny OBJ files.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from glassesdata_pipeline import geometry as geo
from glassesdata_pipeline.sim import mjcf, robots


def _finger(p, short, knuckle_pos, metacarpal=False):
    inner = (f'<body name="{p}{short}knuckle" pos="{"0 0 0.06579" if metacarpal else knuckle_pos}">'
             f'<joint name="{p}{short.upper()}J4" class="knuckle"/>'
             f'<geom class="plastic_visual" mesh="f_knuckle" material="metallic"/>'
             f'<body name="{p}{short}proximal"><joint name="{p}{short.upper()}J3" class="proximal"/>'
             f'<geom class="plastic_visual" mesh="f_proximal"/><geom size="0.009 0.02" pos="0 0 0.025" type="capsule" class="plastic_collision"/>'
             f'<body name="{p}{short}middle" pos="0 0 0.045"><joint name="{p}{short.upper()}J2" class="middle_distal"/>'
             f'<body name="{p}{short}distal" pos="0 0 0.025"><joint name="{p}{short.upper()}J1" class="middle_distal"/>'
             f'<geom class="plastic_collision" type="mesh" mesh="f_distal_pst"/></body></body></body></body>')
    if metacarpal:
        return (f'<body name="{p}lfmetacarpal" pos="{knuckle_pos}"><joint name="{p}LFJ5" class="metacarpal"/>'
                f'<geom class="plastic_visual" mesh="lf_metacarpal"/>{inner}</body>')
    return inner


def fake_shadow_xml(side: str) -> str:
    p = "rh_" if side == "right" else "lh_"
    sx = 1 if side == "right" else -1
    cls = f"{side}_hand"
    thumb_quat = "0.92388 0 0.382683 0" if side == "right" else "0 -0.382683 0 0.92388"
    meshes = "".join(f'<mesh class="{cls}" file="{m}.obj"/>' for m in
                     ("forearm_0", "wrist", "palm", "f_knuckle", "f_proximal", "f_distal_pst", "lf_metacarpal", "th_distal_pst"))
    fingers = (_finger(p, "ff", f"{0.033 * sx} 0 0.095") + _finger(p, "mf", f"{0.011 * sx} 0 0.099")
               + _finger(p, "rf", f"{-0.011 * sx} 0 0.095") + _finger(p, "lf", f"{-0.033 * sx} 0 0.02071", metacarpal=True))
    thumb = (f'<body name="{p}thbase" pos="{0.034 * sx} -0.00858 0.029" quat="{thumb_quat}"><joint name="{p}THJ5" class="thbase"/>'
             f'<body name="{p}thproximal"><joint name="{p}THJ4" class="thproximal"/>'
             f'<body name="{p}thhub" pos="0 0 0.038"><joint name="{p}THJ3" class="thhub"/>'
             f'<body name="{p}thmiddle"><joint name="{p}THJ2" class="thmiddle"/>'
             f'<body name="{p}thdistal" pos="0 0 0.032" quat="1 0 0 -1"><joint name="{p}THJ1" class="thdistal"/>'
             f'<geom class="plastic_collision" type="mesh" mesh="th_distal_pst"/></body></body></body></body></body>')
    site = '<site name="grasp_site" pos="0 -.035 0.09" group="4"/>' if side == "right" else ""
    tendons = "".join(f'<fixed name="{p}{f}J0"><joint joint="{p}{f}J2" coef="1"/><joint joint="{p}{f}J1" coef="1"/></fixed>'
                      for f in ("FF", "MF", "RF", "LF"))
    acts = [("WRJ2", "joint", "wrist_y"), ("WRJ1", "joint", "wrist_x"), ("THJ5", "joint", "thbase"),
            ("THJ4", "joint", "thproximal"), ("THJ3", "joint", "thhub"), ("THJ2", "joint", "thmiddle"),
            ("THJ1", "joint", "thdistal")]
    for f in ("FF", "MF", "RF", "LF"):
        if f == "LF":
            acts.append(("LFJ5", "joint", "metacarpal"))
        acts += [(f"{f}J4", "joint", "knuckle"), (f"{f}J3", "joint", "proximal"), (f"{f}J0", "tendon", "middle_distal")]
    actuators = "".join(f'<position name="{p}A_{n}" {kind}="{p}{n}" class="{c}"/>' for n, kind, c in acts)
    return f"""<mujoco model="{side}_shadow_hand">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>
  <option cone="elliptic" impratio="10"/>
  <default><default class="{cls}">
    <mesh scale="0.001 0.001 0.001"/><joint axis="1 0 0" damping="0.05"/><position forcerange="-1 1"/>
    <default class="wrist"><joint damping="0.5"/>
      <default class="wrist_y"><joint axis="0 1 0" range="-0.52 0.17"/><position kp="10" ctrlrange="-0.52 0.17"/></default>
      <default class="wrist_x"><joint range="-0.69 0.48"/><position kp="8" ctrlrange="-0.69 0.48"/></default></default>
    <default class="thumb">
      <default class="thbase"><joint axis="0 0 -1" range="-1.04 1.04"/><position ctrlrange="-1.04 1.04"/></default>
      <default class="thproximal"><joint range="0 1.22"/><position ctrlrange="0 1.22"/></default>
      <default class="thhub"><joint range="-0.2 0.2"/><position ctrlrange="-0.2 0.2"/></default>
      <default class="thmiddle"><joint axis="0 -1 0" range="-0.69 0.69"/><position ctrlrange="-0.69 0.69"/></default>
      <default class="thdistal"><joint range="-0.26 1.57"/><position ctrlrange="-0.26 1.57"/></default></default>
    <default class="metacarpal"><joint axis="0.573576 0 0.819152" range="0 0.78"/><position ctrlrange="0 0.78"/></default>
    <default class="knuckle"><joint axis="0 -1 0" range="-0.34 0.34"/><position ctrlrange="-0.34 0.34"/></default>
    <default class="proximal"><joint range="-0.26 1.57"/><position ctrlrange="-0.26 1.57"/></default>
    <default class="middle_distal"><joint range="0 1.57"/><position kp="0.5" ctrlrange="0 3.1415"/></default>
    <default class="plastic"><geom solref="0.005 1"/>
      <default class="plastic_visual"><geom type="mesh" material="black" contype="0" conaffinity="0" group="2"/></default>
      <default class="plastic_collision"><geom group="3"/></default></default>
  </default></default>
  <asset><material name="black" rgba="0.16 0.16 0.16 1"/><material name="gray" rgba="0.8 0.8 0.8 1"/>
    <material name="metallic" rgba="0.9 0.9 0.9 1"/>{meshes}</asset>
  <worldbody>
    <body name="{p}forearm" childclass="{cls}" quat="0 1 0 1">
      <inertial mass="3" pos="0 0 0.09" diaginertia="0.0138 0.0138 0.00744"/>
      <geom class="plastic_visual" mesh="forearm_0" material="gray"/>
      <body name="{p}wrist" pos="0.01 0 0.21301" quat="1 0 0 1"><joint class="wrist_y" name="{p}WRJ2"/>
        <geom class="plastic_visual" mesh="wrist" material="metallic"/>
        <body name="{p}palm" pos="0 0 0.034"><inertial mass="0.3" pos="0 0 0.035" diaginertia="0.0005 0.0004 0.0002"/>
          <joint class="wrist_x" name="{p}WRJ1"/>{site}<geom class="plastic_visual" mesh="palm"/>
          {fingers}{thumb}
        </body></body></body>
  </worldbody>
  <contact><exclude body1="{p}wrist" body2="{p}forearm"/><exclude body1="{p}thproximal" body2="{p}thmiddle"/></contact>
  <tendon>{tendons}</tendon>
  <actuator>{actuators}</actuator>
</mujoco>"""


@pytest.fixture
def menagerie(tmp_path):
    root = tmp_path / "menagerie"
    assets = root / "shadow_hand" / "assets"
    assets.mkdir(parents=True)
    V, F = mjcf.box_mesh([0.02, 0.02, 0.02])
    for name in ("forearm_0", "wrist", "palm", "f_knuckle", "f_proximal", "f_distal_pst", "lf_metacarpal", "th_distal_pst"):
        mjcf.write_obj(assets / f"{name}.obj", V, F)
    for side in ("right", "left"):
        (root / "shadow_hand" / f"{side}_hand.xml").write_text(fake_shadow_xml(side))
    return root


def _defined(root, tag):
    return [el.get("name") for el in root.iter(tag) if el.get("name")]


def test_fake_menagerie_matches_the_rest_keypoint_table(menagerie):
    for side in ("right", "left"):
        kp = robots.rest_keypoints_from_xml(robots.SHADOW, side, menagerie / robots.SHADOW.files[side])
        assert np.allclose(kp, robots.SHADOW.rest_keypoints(side), atol=1e-6)


def test_compose_two_floating_hands(menagerie, tmp_path):
    out = tmp_path / "robot.xml"
    model = mjcf.compose(out, robots.SHADOW, ["right", "left"], menagerie, base="free")
    root = ET.parse(out).getroot()
    assert mjcf.check_references(root, out.parent) == []
    # every robot name prefixed, no clashes (grasp_site, class names, meshes named after files)
    for tag in ("body", "joint", "site", "mesh", "material", "position"):
        for name in _defined(root, tag):
            assert name.startswith(("right/", "left/")), (tag, name)
    assert "right/grasp_site" in _defined(root, "site")
    for mesh in root.iter("mesh"):
        if mesh.get("file"):
            assert Path(mesh.get("file")).is_absolute() and Path(mesh.get("file")).is_file()
    # floating palm: free joint, forearm/wrist and the wrist joints/actuators/excludes dropped
    free = [j.get("name") for j in root.iter("freejoint")]
    assert free == ["right/base_free", "left/base_free"]
    bodies = _defined(root, "body")
    assert "right/rh_palm" in bodies and "right/rh_forearm" not in bodies and "left/lh_wrist" not in bodies
    joints = _defined(root, "joint")
    assert not any("WRJ" in j for j in joints) and len(joints) == 2 * 22
    acts = [a.get("name") for a in root.find("actuator")]
    assert len(acts) == 2 * 18 and not any("WRJ" in a for a in acts)
    assert len(root.find("tendon")) == 8
    excludes = [(e.get("body1"), e.get("body2")) for e in root.find("contact")]
    assert ("right/rh_thproximal", "right/rh_thmiddle") in excludes and len(excludes) == 2
    palm = next(b for b in root.iter("body") if b.get("name") == "right/rh_palm")
    assert palm.get("childclass") == "right/right_hand"
    assert root.find(".//default[@class='right/robot_root']/default[@class='right/right_hand']") is not None
    # fingertip sites + keypoint frames
    sites = _defined(root, "site")
    for tip in ("thtip", "fftip", "mftip", "rftip", "lftip"):
        assert f"right/rh_{tip}" in sites and f"left/lh_{tip}" in sites
    hand = model.hands["right"]
    assert hand.base_body == hand.palm_body == "right/rh_palm" and hand.mocap_body is None
    assert len(hand.keypoints) == 21 and hand.removed_joints == ["right/rh_WRJ1"]
    defined = set(bodies) | set(sites)
    assert all(name in defined for _, _, name, _ in hand.keypoints)
    assert root.find("option").get("cone") == "elliptic"
    # the composed file keeps the rest geometry: keypoints in the palm frame equal the table
    frames = robots.rest_frames_from_xml(out)
    palm_T = frames["right/rh_palm"]
    tip = next(s for s in root.iter("site") if s.get("name") == "right/rh_fftip")
    distal = frames["right/rh_ffdistal"]
    p = geo.transform(geo.inv_T(palm_T), geo.transform(distal, np.array(tip.get("pos").split(), float)))
    assert np.allclose(p, robots.SHADOW.rest_keypoints("right")[8], atol=1e-6)


def test_compose_forearm_base_keeps_the_wrist(menagerie, tmp_path):
    out = tmp_path / "robot.xml"
    T0 = geo.make_T(geo.rot_z(0.3), [0.1, 0.2, 0.3])
    model = mjcf.compose(out, robots.SHADOW, ["right"], menagerie, base_body="forearm", initial_base={"right": T0})
    root = ET.parse(out).getroot()
    assert mjcf.check_references(root, out.parent) == []
    forearm = next(b for b in root.iter("body") if b.get("name") == "right/rh_forearm")
    assert forearm.find("freejoint") is not None and forearm.get("childclass") == "right/right_hand"
    assert np.allclose(np.array(forearm.get("pos").split(), float), [0.1, 0.2, 0.3])
    assert "right/rh_WRJ2" in _defined(root, "joint")
    assert len(root.find("actuator")) == 20
    assert model.hands["right"].base_body == "right/rh_forearm" and model.hands["right"].palm_body == "right/rh_palm"


def test_compose_scene_with_mocap_hands(menagerie, tmp_path):
    sim = tmp_path / "rec" / "sim"
    spec = mjcf.SceneSpec(objects=[mjcf.ObjectSpec("cup", geo.make_T(None, [0.5, 0, -0.6]), size=np.array([0.05, 0.05, 0.1]))],
                          hands={"right": None}, head_pose=np.eye(4))
    V, F = mjcf.box_mesh([0.05, 0.05, 0.1])
    part = mjcf.write_obj(tmp_path / "part.obj", V, F)
    spec.objects[0].collision_meshes = [part]
    info = mjcf.write_scene(spec, sim)
    out = tmp_path / "elsewhere" / "physics.xml"
    model = mjcf.compose(out, robots.SHADOW, ["right"], menagerie, scene_xml=info.xml, base="mocap",
                         weld_solref=(0.03, 1.0))
    root = ET.parse(out).getroot()
    assert mjcf.check_references(root, out.parent) == []
    assert root.find("compiler").get("meshdir") is None  # scene meshes made absolute
    cup_mesh = next(m for m in root.iter("mesh") if m.get("name") == "obj_cup_col0")
    assert Path(cup_mesh.get("file")).is_file()
    assert "obj_cup" in _defined(root, "body") and "head" in _defined(root, "body")
    weld = root.find("equality/weld")
    assert weld.get("body1") == "right/base_target" and weld.get("body2") == "right/rh_palm"
    assert weld.get("relpose") == "0 0 0 1 0 0 0" and weld.get("solref") == "0.03 1"
    target = next(b for b in root.iter("body") if b.get("name") == "right/base_target")
    assert target.get("mocap") == "true"
    assert model.hands["right"].mocap_body == "right/base_target" and model.hands["right"].weld == "right/base_weld"
    stats = mjcf.scene_stats(root)
    assert stats["free_joints"] == 2 and stats["equalities"] == 1


def test_check_references_finds_problems(tmp_path):
    root = ET.fromstring('<mujoco><compiler meshdir="assets"/><asset><mesh file="missing.obj"/></asset>'
                         '<worldbody><body name="a"><geom mesh="nope" material="m"/></body><body name="a"/></worldbody>'
                         '<actuator><position joint="j"/></actuator></mujoco>')
    problems = mjcf.check_references(root, tmp_path)
    text = " | ".join(problems)
    assert "duplicate body name 'a'" in text and "mesh='nope'" in text and "material='m'" in text
    assert "joint='j'" in text and "missing.obj" in text


@pytest.mark.skipif(not os.environ.get("MUJOCO_MENAGERIE"), reason="set MUJOCO_MENAGERIE to a Menagerie clone")
def test_composed_models_compile_in_mujoco(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    menagerie = Path(os.environ["MUJOCO_MENAGERIE"])
    out = tmp_path / "robot.xml"
    mjcf.compose(out, robots.SHADOW, ["right", "left"], menagerie, base="mocap")
    model = mujoco.MjModel.from_xml_path(str(out))
    assert model.nmocap == 2 and model.neq == 2 and model.nu == 36 and model.njnt == 2 + 44
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right/rh_palm")
    T_palm = geo.make_T(np.array(data.xmat[palm]).reshape(3, 3), np.array(data.xpos[palm]))
    for joint, kind, name, _ in robots.SHADOW.keypoint_frames("right"):
        obj = mujoco.mjtObj.mjOBJ_BODY if kind == "body" else mujoco.mjtObj.mjOBJ_SITE
        ident = mujoco.mj_name2id(model, obj, "right/" + name)
        p = data.xpos[ident] if kind == "body" else data.site_xpos[ident]
        assert np.allclose(geo.transform(geo.inv_T(T_palm), p), robots.SHADOW.rest_keypoints("right")[joint], atol=1e-6)
