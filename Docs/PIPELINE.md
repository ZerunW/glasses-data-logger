# Offline pipeline: data format and step contracts

This document is the contract between the offline pipeline's modules. Every step reads and writes
only the files described here. Change this document first, then the code.

Package: `Pipeline/glassesdata_pipeline/` (Python 3.11+). Command line: `python -m glassesdata_pipeline`.

## 0. Environments and dependencies

- **Core** (must import and be tested on the development Mac): stdlib, `numpy`, `cv2` (OpenCV 5 with
  `cv2.aruco`), `av` (PyAV, has `libx265`), `mediapipe`, `PIL`, `matplotlib`. There is **no scipy and no
  yaml** on the Mac. Configuration is TOML read with `tomllib`.
- **Heavy** (run on the Ubuntu 24.04 + CUDA machine only): `torch`, `transformers`, `open3d`,
  `trimesh`, `coacd`, `mujoco`, `mujoco_warp`, `warp`, `mink`, `rerun`, `rsl_rl`, model weights.
  Import these lazily, inside the function or backend that needs them, never at module import.
  When one is missing, raise `MissingDependency("pip install X (needed for step Y)")`.
- Tests that need a heavy package start with `pytest.importorskip("name")`.
- Mac test command: `cd Pipeline && ~/.venvs/glassesdata-receiver/bin/python -m pytest tests -q -p no:cacheprovider`.
- Nothing may be installed on the Mac. `Pipeline/requirements.txt` (full) and
  `Pipeline/requirements-core.txt` describe the Linux install.

## 1. Conventions

- Units: metres, radians, seconds. Raw timestamps are integer nanoseconds.
- **Master clock**: the phone's monotonic clock (`phone_time_ns`). Video frames, IMU samples and
  inputs all carry it. Episode time `t = (phone_time_ns - t0_ns) * 1e-9`, where `t0_ns` is the
  `phone_time_ns` of video frame 0.
- **Camera frame**: OpenCV (x right, y down, z forward). Pixel (u, v): pixel centres at integer
  coordinates, origin at the top-left pixel. Intrinsics refer to the decoded video frame as stored
  (720×1280 portrait for the glasses at `.high`); frames are never rotated.
- **World frame**: z up (gravity = −z). Origin at the camera centre of the first valid trajectory
  frame. Yaw: the first camera's optical axis, projected onto the horizontal plane, points along +x.
- Poses: `T_a_b` is a 4×4 matrix mapping coordinates in frame b to frame a (`p_a = T_a_b @ p_b`).
  Quaternions are stored `[w, x, y, z]`.
- **Hand joints**: MediaPipe's 21-landmark order: 0 wrist; thumb 1 CMC, 2 MCP, 3 IP, 4 tip; index 5
  MCP, 6 PIP, 7 DIP, 8 tip; middle 9–12; ring 13–16; pinky 17–20. Parent of each joint:
  `PARENTS = [-1, 0,1,2,3, 0,5,6,7, 0,9,10,11, 0,13,14,15, 0,17,18,19]`. Bone i (i = 1..20) joins
  joint i to `PARENTS[i]`.
- **Hand slots**: arrays use slot 0 = left hand, slot 1 = right hand (anatomical, after the
  receiver's `anatomical_label`). Missing values are NaN.
- Per-frame arrays have first dimension T = number of decoded video frames.

## 2. A recording folder (input)

Written by the receiver (`Receiver/glassesdata_receiver/recorder.py`):

| File | Columns / content |
|---|---|
| `video.mp4` | HEVC (`hvc1`), no B-frames, fragmented MP4, frames in order |
| `frames.csv` | `frame_index, pts_us, phone_time_ns, mac_time_ns, keyframe, bytes` |
| `motion.csv` | `phone_time_ns, glasses_time_ns, ax, ay, az, gx, gy, gz, mx, my, mz, qw, qx, qy, qz, source, mac_time_ns` (empty cells when the SDK gave no value; `source` is `glasses` or `phone`) |
| `inputs.csv` | `phone_time_ns, glasses_time_ms, source, type, detail, mac_time_ns` |
| `device_state.csv` | `phone_time_ns, battery, charging, don, hinge, thermal, link, mac_time_ns` |
| `clock.csv` | `mac_send_ns, mac_receive_ns, phone_time_ns, rtt_ns, offset_ns` |
| `hands.csv` | live (frame-skipping) MediaPipe results; not used by the pipeline |
| `info.json` | receiver metadata |
| `calibration.json` | camera calibration in use when recording started (optional, see §3) |
| `profile.json` | body measurements in use when recording started (optional, see §4) |
| `pipeline.toml` | optional per-recording configuration overrides (§6) |
| `gt/` | ground truth, synthetic recordings only (§8) |

IMU units are not documented by the SDK. The `imu` step detects accelerometer units from the
median norm at rest (≈1 → g, ≈9.8 → m/s²) and takes gyroscope units from config
(`imu.gyro_units = "rad/s" | "deg/s" | "auto"`; auto compares with the visual rotation rate).

## 3. `calibration.json` (shared with the receiver's calibration page)

```json
{"version": 1, "model": "pinhole", "width": 720, "height": 1280,
 "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
 "dist": [k1, k2, p1, p2, k3],
 "rms_px": 0.41, "created": "2026-11-21T03:00:00Z", "device": "Ray-Ban Meta",
 "source": "charuco-web"}
```

- `model` is `"pinhole"` (OpenCV `calibrateCamera`, `dist` has 4, 5, 8, 12 or 14 values) or `"fisheye"`
  (OpenCV `cv2.fisheye`, `dist` = `[k1, k2, k3, k4]`).
- `source`: `"charuco-web"`, `"synthetic"` or `"default"`.
- Optional `"board"`: the ChArUco board used, so `eval-charuco` can reuse it:
  `{"dictionary": "DICT_5X5_100", "squares_x": 9, "squares_y": 6, "square_m": 0.03,
  "marker_m": 0.022, "css_px_per_mm": 3.78}`.
- If the recording has none, the pipeline uses `[camera]` in the config (a horizontal field of view
  guess, no distortion) and marks every result `calibration: "default"`.
- If the recording's resolution differs from the calibration's, K is scaled; a different aspect
  ratio is an error.

## 4. `profile.json` (shared with the receiver's body-measurement page)

```json
{"version": 1,
 "hand_length_m": 0.187, "palm_length_m": 0.107, "palm_width_m": 0.084,
 "forearm_length_m": 0.26, "upper_arm_length_m": 0.32, "shoulder_width_m": 0.39,
 "eye_to_shoulder_drop_m": 0.20, "eye_to_shoulder_back_m": 0.07,
 "measured": {"hand_length_m": true, "palm_length_m": false}}
```

Definitions shown to the user: hand length = wrist crease to middle fingertip; palm length = wrist
crease to the base crease of the middle finger; palm width = across the knuckles of index to
little finger; forearm = wrist crease to the tip of the elbow; upper arm = shoulder tip (acromion)
to the tip of the elbow; shoulder width = between the two shoulder tips. The two eye-to-shoulder
offsets are not measured by the user (defaults). The values above are the defaults (adult averages)
used for anything not measured; `measured` says which values the user entered.

Hand bone lengths are derived from `hand_length_m` with the Buchholz et al. (1992) segment ratios
(fraction of hand length; verify against the paper before relying on them):
thumb MC 0.251, PP 0.196, DP 0.158; index MC 0.374, PP 0.265, MP 0.143, DP 0.097; middle MC 0.373,
PP 0.277, MP 0.170, DP 0.108; ring MC 0.336, PP 0.259, MP 0.165, DP 0.107; little MC 0.295,
PP 0.206, MP 0.117, DP 0.093. The thumb CMC joint is placed from the wrist by a fixed template
offset. Palm width rescales the lateral spacing of the MCP joints. `skeleton.py` implements this and
returns a canonical 21×3 rest pose (wrist at origin) plus the 20 bone lengths.

## 5. Derived files (`<recording>/derived/`)

Every `.npz` contains an extra string array `meta_json` with
`{"step", "version", "params", "inputs", "created", "backend", "calibration"}`; every `.json`
output has the same object under the key `"meta"`. `inputs` maps each input file to
`[size, mtime_ns]`; the runner skips a step whose params and inputs are unchanged unless `--force`.

| Step | Output | Arrays / content |
|---|---|---|
| `ingest` | `timeline.npz` | `frame_index[T]` int64, `pts_us[T]`, `phone_ns[T]` int64, `t[T]` float64, `keyframe[T]` bool, `imu_t[N]` (episode seconds), `acc[N,3]`, `gyro[N,3]`, `mag[N,3]`, `quat[N,4]`, `imu_is_glasses[N]` bool |
| | `camera.json` | the calibration in use (§3) plus `"source"` |
| | `profile.json` | the profile in use (§4) plus `bone_lengths_m[20]` |
| `hands2d` | `hands2d.npz` | `kp[T,2,21,3]` (u, v, score), `hand_score[T,2]`, `mp_world[T,2,21,3]` (MediaPipe hand-centred metric landmarks, NaN if the backend has none) |
| `depth` | `depth/depth_f16.npy` | memmap float16 `[F,h,w]` metres, in the distorted image grid downsampled by `depth.downsample` |
| | `depth/index.npz` | `frames[F]` (video frame indices that have depth), `scale` (= h / H) |
| `hand3d` | `hand3d.npz` | `joints_cam[T,2,21,3]`, `valid[T,2]`, `reproj_px[T,2]`, `depth_scale[T]` (raw ratio hand-metric / depth-model, NaN when unknown), `depth_scale_smooth[T]` |
| `slam` | `slam.npz` | `T_slam_cam[T,4,4]` (first valid frame = identity, metric), `valid[T]`, `inliers[T]`, `is_keyframe[T]`, `source` str |
| `imu` | `imu.npz` | `R_cam_imu[3,3]`, `time_offset_s` (IMU time = video time + offset), `gyro_bias[3]`, `accel_scale` (to m/s²), `gravity_cam[T,3]` (unit, pointing down, camera frame), `R_world_slam[3,3]`, `quality` json |
| `world` | `camera_traj.npz` | `T_world_cam[T,4,4]`, `valid[T]` |
| | `hands_world.npz` | `joints[T,2,21,3]` (smoothed), `raw[T,2,21,3]`, `observed[T,2]`, `filled[T,2]`, `bone_lengths[2,20]` |
| `arms` | `arms.npz` | `shoulder[T,2,3]`, `elbow[T,2,3]`, `wrist[T,2,3]`, `elbow_conf[T,2]`, `method[T,2]` int8 (0 none, 1 forearm cue, 2 prior only) |
| `objects` | `objects/index.json` | `[{"id": "obj0", "prompt": {...}, "first_frame": i, "last_frame": j}]` |
| | `objects/<id>/masks.npz` | `frames[F]`, `packed[F, ...]` (`np.packbits` of bool masks at depth resolution), `shape` |
| | `objects/<id>/mesh.obj` | visual mesh in the object frame (origin at the centroid) |
| | `objects/<id>/collision/part_*.obj` | convex parts (convex hull when decomposition is unavailable) |
| | `objects/<id>/poses.npz` | `T_world_obj[T,4,4]`, `valid[T]`, `fitness[T]`, `grasped[T,2]` |
| | `objects/<id>/props.json` | `volume_m3, mass_kg, com[3], friction` |
| | `objects/table.json` | `normal[3], offset, height_m, polygon[[x,y],...]` (world) |
| `scene` | `sim/scene.xml` (+ `sim/assets/`) | MuJoCo MJCF: world, table, objects (free joints), mocap human hand skeleton |
| | `sim/replay.mp4` | side-by-side source video / kinematic replay |
| `retarget` | `retarget_<robot>.npz` | `qpos[T,nq]`, `joint_names`, `err_m[T]`, `robot` |
| `physics` | `physics.json` | per-object deviation of simulated vs reconstructed trajectory, plausibility score |
| `viz` | `viz/episode.rrd`, `viz/*.png` | Rerun recording; static matplotlib figures |
| `eval` | `eval.json`, `eval.md`, `eval/*.png` | metrics against `gt/` or against the ChArUco board |

Depth maps are large: at `downsample = 4` a 720×1280 frame is 180×320 float16 (115 kB).

## 6. Configuration

Every tunable parameter lives in `Pipeline/configs/<section>.toml`, one file per section, each
parameter with a comment (`camera.toml` holds `[camera]`, `slam.toml` holds `[slam]` and its
sub-tables, and so on for `ingest`, `hands2d`, `depth`, `hand3d`, `imu`, `world`, `arms`, `objects`,
`scene`, `retarget`, `physics`, `viz`, `train`, `eval`, `synth`). The loader merges all files in
`configs/` (one file per section keeps parallel work conflict-free). Precedence, lowest first:
`configs/*.toml`, `<recording>/pipeline.toml`, `--config FILE`, `--set section.key=value` (value
parsed as TOML, falling back to a plain string).

Backends are chosen per step, for example `hands2d.backend = "mediapipe" | "wilor" | "hamer" | "gt"`,
`depth.backend = "depth_anything_v2" | "video_depth_anything" | "gt"`,
`slam.backend = "builtin" | "import"` (with `slam.import_path` to a TUM-format trajectory from
MegaSaM / DROID-SLAM / MASt3R-SLAM), `objects.mask_backend = "sam2" | "gt"`. The `gt` backends exist
only for synthetic recordings: they read `gt/` and add configurable noise.

## 7. Code layout and APIs

```
Pipeline/
  configs/default.toml
  requirements.txt, requirements-core.txt
  scripts/setup_ubuntu.sh, scripts/omen_smoke.sh
  glassesdata_pipeline/
    __main__.py   CLI: run, synth, eval, eval-charuco, view, train, list-steps
    config.py     load/merge TOML, dotted access
    episode.py    Episode (below)
    runner.py     step registry, dependency order, caching, logging
    geometry.py   Camera (project/unproject/undistort), SE(3)/SO(3) helpers
    skeleton.py   hand topology, PARENTS, profile → rest pose and bone lengths
    errors.py     MissingDependency, PipelineError
    steps/        one module per step: ingest, hands2d, depth, hand3d, slam, imu, world, arms,
                  objects, scene, retarget, physics, viz
    core/         pure algorithms, no file I/O (smoothing, elbow, imu_filter, align, scale, vo,
                  pose_graph, icp, plane, fusion, ...)
    backends/     model wrappers (mediapipe_hands, wilor, hamer, depth_anything, sam2, gt, ...)
    synth/        synthetic recording generator
    eval/         metrics, report, charuco evaluation
    sim/          MJCF generation, kinematic replay, retargeting, physics check
    rl/           tracking task, MuJoCo and MuJoCo Warp environments, PPO training
  tests/
```

`Episode(path, config_overrides=None)`:
- `.path`, `.derived` (creates `derived/`), `.config` (merged), `.gt` (None or a loaded ground truth)
- `.timeline()`, `.camera()` → `geometry.Camera`, `.profile()` → dict
- `.frames(indices=None, downsample=1, gray=False)` → iterator of `(frame_index, image)` decoded with
  PyAV, BGR uint8
- `.save_npz(name, meta, **arrays)`, `.load_npz(name)`, `.save_json(name, obj, meta)`, `.load_json(name)`
- `.has(name)`

A step module exposes `NAME`, `REQUIRES` (step names), `OUTPUTS` (relative paths),
`run(ep: Episode, cfg: dict) -> dict` (returns a short summary for the log).

Step order: ingest → hands2d → depth → hand3d → slam → imu → world → arms → objects → scene →
retarget → physics → viz. `slam` runs visual-only (with metric depth); `imu` then estimates the
camera–IMU rotation and time offset by aligning gyro rates with the visual rotation rates, and the
gravity direction; `world` applies the gravity alignment and produces the final trajectory.

## 8. Synthetic recordings (`synth`)

`python -m glassesdata_pipeline synth OUT_DIR [--seconds 8] [--seed 0] [--size 360x640]` writes a
complete recording folder in the receiver's format (§2), rendered with numpy/OpenCV only (analytic
ray casting against textured planes and boxes), plus `calibration.json` (`source: "synthetic"`),
`profile.json`, a `pipeline.toml` that selects the `gt` backends where a learned model is needed,
and `gt/`:

- `gt/gt.npz`: `T_world_cam[T,4,4]`, `K[3,3]`, `joints_world[T,2,21,3]`, `joints_cam[T,2,21,3]`,
  `visible[T,2,21]`, `shoulder_world[T,2,3]`, `elbow_world[T,2,3]`, `T_world_obj[T,K,4,4]`,
  `R_cam_imu[3,3]`, `imu_time_offset_s`, `gyro_bias[3]`, `accel_bias[3]`, `table_height_m`,
  `object_sizes[K,3]`
- `gt/depth_f16.npy`: `[T,h,w]` float16 metres at full synthetic resolution
- `gt/masks.npz`: per-object packed masks at full resolution, `frames`, `shape`

Content: a textured room, a table, one or two textured boxes, a head-like camera path looking at
the table, one right hand (21 joints from the profile's bone lengths, not rendered in the image)
that reaches, grasps, lifts, moves and places a box, the matching shoulder/elbow, and IMU at 60 Hz
with noise, bias, a known camera–IMU rotation and a known time offset.

## 9. Evaluation

`eval` computes whatever both `gt/` (or a ChArUco recording) and `derived/` provide:
depth AbsRel / RMSE / δ<1.25 (after the pipeline's own scale, not GT-aligned), camera ATE (Sim(3)
and SE(3) aligned) and RPE, hand MPJPE in camera and world frames (and PA-MPJPE), elbow error,
object translation / rotation error, IMU time offset and rotation error. `eval-charuco` uses a
recording of the on-screen ChArUco board: the board pose from PnP is the reference for depth and
scale.
