# GlassesData offline pipeline

Turns a recording folder from the GlassesData receiver (Ray-Ban Meta: one RGB camera + IMU) into
hand and arm skeletons, a head trajectory and objects in a gravity-aligned world, then a MuJoCo
scene, robot-hand retargeting, a physics check and RL training data. **The contract (file formats,
conventions, step APIs) is [`Docs/PIPELINE.md`](../Docs/PIPELINE.md).** Change that document
first, then the code.

## Run on the Mac (development)

```sh
cd Pipeline
PY=~/.venvs/glassesdata-receiver/bin/python          # core packages only; nothing heavy is installed here

$PY -m glassesdata_pipeline synth /tmp/synth_demo --seconds 8      # synthetic recording + gt/ (hands drawn)
$PY -m glassesdata_pipeline run /tmp/synth_demo                    # every step; exit 0
$PY -m glassesdata_pipeline eval /tmp/synth_demo                   # derived/eval.json, eval.md, eval/*.png
$PY -m glassesdata_pipeline list-steps /tmp/synth_demo             # the steps and their cache state
$PY -m glassesdata_pipeline run /tmp/synth_demo --steps objects --force
$PY -m glassesdata_pipeline run REC --from slam --until world --set slam.backend=import
$PY -m pytest tests -q -p no:cacheprovider                         # tests (~40 s)
```

On the Mac `retarget` needs MuJoCo + mink: the run reports it as `skipped (needs mujoco: pip
install mujoco ...)`, `physics` (which needs its output) as `blocked`, and still exits 0; the scene
replay uses the numpy wireframe renderer. A step that fails for any other reason exits 1, and so
does a step you name with `--steps` that cannot run. Step statuses: `ran`, `cached` (outputs up to
date for the same code version, parameters and inputs), `skipped` (an optional package is
missing), `blocked` (a required step did not produce its outputs), `failed`.

Keep recordings and generated data out of this repository (it lives in OneDrive).

## Run on the Linux GPU laptop (Ubuntu 24.04, NVIDIA)

```sh
Pipeline/scripts/setup_ubuntu.sh            # packages, venv ~/.venvs/glassesdata-pipeline, torch (CUDA), requirements,
                                            # SAM 2, ~/src/mujoco_menagerie; --models downloads the weights; asks before sudo
Pipeline/scripts/omen_smoke.sh [REC]        # first run: tests, synthetic end-to-end with gt and real models,
                                            # retarget/physics/Rerun, RL (MuJoCo CPU + Warp GPU), REC if given
```

`omen_smoke.sh` writes every log into `~/glassesdata-smoke/<date>_<time>/` and ends with a summary
table (`summary.txt`); it keeps going after a failure. Set `MUJOCO_GL=egl` in shells that run the
pipeline (MuJoCo's off-screen rendering on the NVIDIA GPU). RTX 50-series GPUs need a CUDA 12.8 or
newer torch build; the setup script picks the wheel index from the driver (`--cuda cu130|cu128`).
Optional hand models: `--with-wilor` / `--with-hamer` (both need `MANO_RIGHT.pkl` from
https://mano.is.tue.mpg.de, which you download yourself after registering).

## Steps

| Step | Output (`derived/`) | Default backend / method |
|---|---|---|
| `ingest` | `timeline.npz`, `camera.json`, `profile.json` | frames + glasses IMU (`ingest.imu_sources = ["glasses"]`) |
| `hands2d` | `hands2d.npz` | MediaPipe (`wilor`, `hamer` on CUDA; `gt` for synthetic) |
| `depth` | `depth/depth_f16.npy`, `depth/index.npz` | Depth Anything V2 metric indoor (`video_depth_anything`; `gt`) |
| `hand3d` | `hand3d.npz` | PnP with the user's hand size; depth scale from the hands |
| `slam` | `slam.npz` | metric visual odometry + pose graph (or an imported TUM trajectory) |
| `imu` | `imu.npz` | camera-IMU rotation, time offset, biases, gravity |
| `world` | `camera_traj.npz`, `hands_world.npz` | gravity-aligned trajectory, smoothed hands |
| `arms` | `arms.npz` | shoulders from the head, elbows with an image forearm cue |
| `objects` | `objects/` | grasp-triggered prompts, SAM 2 masks (`gt`), fused hull meshes, poses, table |
| `scene` | `sim/scene.xml`, `sim/replay.mp4` | MuJoCo MJCF + kinematic replay (wireframe without MuJoCo) |
| `retarget` | `retarget_shadow.npz` | Shadow Hand, mink IK (needs MuJoCo + Menagerie) |
| `physics` | `physics.json` | replay with free objects, plausibility score |
| `viz` | `viz/*.png`, `viz/episode.rrd` | matplotlib figures; Rerun when `rerun-sdk` is installed |

`train REC` (package `rl/`) trains a floating robot hand with PPO (rsl_rl) to reproduce the
object's motion: `--backend mujoco` (CPU) or `warp` (MuJoCo Warp, GPU), `--play` rolls out a
checkpoint. `view REC` opens the Rerun file; `eval-charuco REC` evaluates against the on-screen
ChArUco board. Methods and parameters: each step module's docstring and `configs/<section>.toml`.

IMU: the app writes the Meta SDK's motion sources (`glasses`, `neuralBand` = a wrist IMU,
`unknown`); only the glasses rows are the head IMU. The SDK documents m/s^2, rad/s and µT; the
`imu` step checks the units anyway (`imu.accel_units` / `gyro_units = "auto"`).

## Synthetic recordings

`synth` renders a room, a table, two boxes and the person's hands and arms (capsules; a real
depth model sees the hands, which `hand3d` needs for the depth scale; `--no-draw-hands` leaves
them out of the image), with IMU, `calibration.json`, `profile.json`, `gt/` and a `pipeline.toml`
that selects the `gt` backends (with the noise of `[synth.gt_noise]`) and a `depth.downsample` that
gives the depth grid the glasses' angular resolution (`synth.depth_width_px = 180`, i.e. 2 for the
default 360x640). Accuracy of the whole chain on 8 s clips (gt hands2d / depth / masks with noise,
every other step real; seeds 3 and 7, November 2026):

| | seed 3 | seed 7 |
|---|---|---|
| hand3d MPJPE (camera) | 9.7 mm | 9.2 mm |
| depth scale error (smoothed) | 0.7 % | 0.6 % |
| slam ATE (SE(3)) | 0.6 mm | 0.6 mm |
| gravity error | 0.35° | 0.33° |
| hands MPJPE (world) | 8.8 mm | 7.9 mm |
| elbow error | 1.1 cm | 1.2 cm |
| object position (static / held) | 0.9 / 1.7 cm | 0.8 / 1.5 cm |
| table height error | -1.8 mm | 0.7 mm |

## Layout

| Path | What |
|---|---|
| `configs/<section>.toml` | every tunable parameter, one file per section (merged automatically) |
| `glassesdata_pipeline/config.py` | load/merge TOML, `--set`, dotted access |
| `glassesdata_pipeline/episode.py` | `Episode`: a recording folder, `derived/` I/O, frames, camera, profile, `gt` |
| `glassesdata_pipeline/runner.py` | step registry (fixed order), dependencies, caching via `meta_json`, skipped/blocked steps |
| `glassesdata_pipeline/steps/` | one module per step (the 13 above); see `steps/__init__.py` for the contract |
| `glassesdata_pipeline/geometry.py` | `Camera` (pinhole/fisheye, OpenCV-compatible), SE(3)/SO(3) helpers |
| `glassesdata_pipeline/skeleton.py` | hand topology, profile, rest pose + forward kinematics, arm model |
| `glassesdata_pipeline/backends/` | model wrappers (mediapipe_hands, wilor, hamer, depth_anything, video_depth_anything, sam2, open3d, coacd, ...) and `gt.py` (ground-truth backends with noise) |
| `glassesdata_pipeline/core/` | pure algorithms (smoothing, elbow, imu_filter, align, vo, pose_graph, icp, plane, fusion, hull, grasp, ...) |
| `glassesdata_pipeline/synth/` | synthetic recordings with ground truth; `synth/preview.py` writes frame / GT-overlay / depth PNGs |
| `glassesdata_pipeline/groundtruth.py`, `recording.py` | `gt/` access (`ep.gt`); raw recording readers (CSV headers copied from the receiver) |
| `glassesdata_pipeline/eval/` | metrics against `gt/`, the report, the ChArUco evaluation |
| `glassesdata_pipeline/sim/` | MJCF scene + robot composition (`mjcf.compose`), replay, Shadow Hand tables, mink retargeting, physics check |
| `glassesdata_pipeline/rl/` | tracking task, MuJoCo and MuJoCo Warp environments, PPO training, play |
| `glassesdata_pipeline/viz/` | static figures `derived/viz/*.png`, Rerun export, `view REC` |
| `scripts/` | `setup_ubuntu.sh`, `omen_smoke.sh` (Linux GPU laptop) |
| `requirements-core.txt`, `requirements.txt` | the Mac's packages; the full Linux install (pins also in `pyproject.toml` `[heavy]`) |

## Writing a step

Create `glassesdata_pipeline/steps/<name>.py` with `NAME`, `VERSION`, `REQUIRES`, `INPUTS`
(relative to the recording), `OUTPUTS` (relative to `derived/`), `CONFIG_SECTIONS` and
`run(ep, cfg) -> dict`; save outputs with `ep.save_npz(name, ep.step_meta(backend=...), ...)`.
Import heavy packages inside `run` and raise `errors.MissingDependency` when they are missing (the
runner then reports the step as skipped). Bump `VERSION` when the output changes for the same
inputs. Test against a synthetic recording (`synth.generate`) and its `gt/`.

To develop a step without the steps before it, `tests/fabricate.py` (hands2d / depth / hand3d) and
`tests/fabricate_objects.py` (everything `objects` reads) write upstream files from a synthetic
recording's `gt/` with realistic noise:

```sh
PYTHONPATH=.:tests $PY -c "import sys; from fabricate import fabricate_upstream; fabricate_upstream(sys.argv[1])" /tmp/synth_demo
$PY -m glassesdata_pipeline run /tmp/synth_demo --steps slam,imu,world,arms --no-deps && $PY -m glassesdata_pipeline eval /tmp/synth_demo
```

Fabricated noise is white; the real chain's hand errors are partly slow (hand3d smooths its depth),
which a smoother cannot remove: expect the chained numbers above, not the fabricated ones.
