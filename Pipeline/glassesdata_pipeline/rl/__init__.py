"""RL: learn, in MuJoCo, a policy for a floating dexterous robot hand that reproduces the object
motion of one reconstructed demonstration (``python -m glassesdata_pipeline train REC``).

Pinned versions this code was written against (never run on the development Mac; see
``Pipeline/requirements.txt``): ``rsl-rl-lib==6.0.0`` (``OnPolicyRunner``, ``VecEnv``, ``PPO``,
``MLPModel``, ``GaussianDistribution``), ``mujoco==4.0.1``, ``mujoco-warp==4.0.1``,
``warp-lang==1.19.0``.

Modules
=======

* :mod:`.xmath` - array-module helpers and ``[w, x, y, z]`` quaternion math for ``xp`` = numpy or torch.
* :mod:`.reference` - the demo as a reference trajectory at the control rate (numpy).
* :mod:`.task` - configuration (``[train]``), reference state initialisation, action scaling,
  termination, and :class:`~.task.TrackingTask`, the backend-independent environment core.
* :mod:`.obs`, :mod:`.rewards` - observation blocks and reward terms (pure ``xp`` functions).
* :mod:`.scene_adapter` - the composed MuJoCo model (``sim.mjcf.compose``) -> :class:`~.scene_adapter.TaskModel`.
* :mod:`.env_mujoco` (CPU, N ``MjData``), :mod:`.env_warp` (GPU, MuJoCo Warp ``nworld = N``) - physics backends.
* :mod:`.vecenv` - the rsl_rl ``VecEnv`` adapter; :mod:`.train` - CLI and PPO; :mod:`.play` - rollout video and metrics.

Design choices
==============

Reference. One demo episode is the reference: the robot hand qpos from ``retarget_<robot>.npz``
(floating base free joint + finger joints, columns matched by name to the RL model) and the pose
of the manipulated object from ``objects/<id>/poses.npz``. The manipulated object is the one with
the most ``grasped`` frames for the robot's hand (``train.object`` overrides). Both are cut to the
frames where the hand and the object are known, gaps are interpolated (slerp for rotations), the
video timestamps are resampled to the control rate (``train.control_dt`` = 0.02 s, i.e. 10
physics substeps of 2 ms), lightly smoothed, and differentiated for velocities. The reference
base pose, fingertip positions and actuator targets come from MuJoCo forward kinematics of the
reference qpos (the reference qpos is first clipped to the joint limits).

Initial states. Reference state initialisation: every reset starts a world at a random demo step
(at least ``min_episode_steps`` before the end; with ``p_start_at_zero`` at step 0) in the
reference hand state and object pose and velocities, plus Gaussian noise (fingers, base, object
xy / yaw-free rotation, upward-only height noise).

Episode end. At the demo end (a time-out: rsl_rl bootstraps the value), or early (failure, with a
termination penalty) when the object is more than ``obj_pos_m`` from the reference, rotated more
than ``obj_rot_rad`` (off by default), below the table top by ``fall_below_table_m`` (fell), the
hand base is more than ``base_pos_m`` off, or the simulator state is not finite.

Observation (policy, 2-D ``(N, dim)``, see :mod:`.obs`): finger joint positions/velocities, base
pose (position + 6D rotation) and velocity, object pose (position + 6D) and linear/angular
velocity, the reference object pose at t+1 and t+k, the rotation from the object to the reference
at t+1, the reference finger qpos and base pose at t+1, fingertip - object offsets, phase, and
the previous action. The critic additionally sees the domain parameters, the demo's grasp flag
and fingertip offsets, and the current object errors.

Action. Residuals on the reference at t+1, clipped to ``[-1, 1]`` and scaled: base position
(``action.base_pos`` m) and orientation (rotation vector, ``action.base_rot`` rad) move the mocap
target of the welded hand base (or the base position actuators of a 6-joint base); finger
actuator targets get ``action.finger`` rad per coupled joint, clipped to the control range.

Reward (:mod:`.rewards`, weights in ``configs/train.toml``): exp-kernel object position and
orientation tracking, finger/base/fingertip imitation, a contact bonus while the demo grasps
(fingertips at least as close to the object as in the demo), action-rate and energy penalties,
and the termination penalty.

Domain randomisation (per world, resampled at reset): object mass and inertia scale, sliding
friction scale of the object and hand geoms (MuJoCo Warp: per-world model fields via
``put_model(batch_sizes=...)``; CPU: per-world ``MjModel`` copies), and the initial-state noise.

Backends. ``--backend warp``: MuJoCo Warp with one world per environment, physics substeps in a
CUDA graph, state exchanged zero-copy with torch, observations and rewards computed in torch.
``--backend mujoco``: N CPU ``MjData`` in a Python loop for smoke tests and ``--play``. Both
implement the same small protocol (:class:`~.task.SimBackend`), so the task code is shared and
unit-tested with numpy.
"""
