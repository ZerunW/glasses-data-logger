"""Simulation side of the pipeline: MuJoCo scenes, kinematic replay, retargeting and the physics check.

Modules (numpy-only unless noted; MuJoCo / mink are imported lazily inside functions):

* :mod:`.frames` - camera and pose conventions (OpenCV camera <-> MuJoCo camera, quaternions,
  slerp, the numpy model of a MuJoCo perspective camera).
* :mod:`.mjcf` - MJCF generation with ``xml.etree``: the episode scene (``sim/scene.xml``) and
  :func:`.mjcf.compose` that adds floating robot hands (Menagerie) to a scene or an empty world.
* :mod:`.robots` - robot tables (Shadow Hand names, palm frame, human -> robot keypoint map) and
  the keypoint scaling math.
* :mod:`.replay` - kinematic replay video (MuJoCo renderer on Linux, a numpy wireframe renderer
  anywhere) and the matplotlib replay sheet.
* :mod:`.retarget_mink` - per-frame differential IK with mink (needs ``mujoco`` + ``mink``).
* :mod:`.physics_check` - drive the robot hands through the reconstructed episode in MuJoCo and
  compare the simulated objects with the reconstructed ones (needs ``mujoco``).
"""
