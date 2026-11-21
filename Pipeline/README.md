# GlassesData offline pipeline

Turns a recording folder from the GlassesData receiver (Ray-Ban Meta: one RGB camera + IMU) into
hand and arm skeletons, a head trajectory and objects in a gravity-aligned world, then a MuJoCo
scene and RL training data. **The contract (file formats, conventions, step APIs) is
[`Docs/PIPELINE.md`](../Docs/PIPELINE.md).** Change that document first, then the code.

## Run

```sh
cd Pipeline
PY=~/.venvs/glassesdata-receiver/bin/python          # on the Mac; any Python 3.11+ elsewhere

$PY -m glassesdata_pipeline synth /tmp/synth_demo --seconds 8      # synthetic recording + gt/
$PY -m glassesdata_pipeline list-steps /tmp/synth_demo             # which steps exist, cache state
$PY -m glassesdata_pipeline run /tmp/synth_demo                    # all implemented steps
$PY -m glassesdata_pipeline run /tmp/synth_demo --steps ingest --force
$PY -m glassesdata_pipeline run REC --from slam --until world --set slam.backend=import
$PY -m glassesdata_pipeline eval /tmp/synth_demo                   # derived/eval.json, eval.md, eval/*.png
$PY -m pytest tests -q -p no:cacheprovider                         # tests (< 2 min)
```

Keep recordings and generated data out of this repository (it lives in OneDrive).

## Layout

| Path | What |
|---|---|
| `configs/<section>.toml` | every tunable parameter, one file per section (merged automatically) |
| `glassesdata_pipeline/config.py` | load/merge TOML, `--set`, dotted access |
| `glassesdata_pipeline/episode.py` | `Episode`: a recording folder, `derived/` I/O, frames, camera, profile, `gt` |
| `glassesdata_pipeline/runner.py` | step registry (fixed order), dependencies, caching via `meta_json`, logging |
| `glassesdata_pipeline/steps/` | one module per step (`ingest` so far); see `steps/__init__.py` for the contract |
| `glassesdata_pipeline/geometry.py` | `Camera` (pinhole/fisheye, OpenCV-compatible), SE(3)/SO(3) helpers |
| `glassesdata_pipeline/skeleton.py` | hand topology, profile, rest pose + forward kinematics, arm model |
| `glassesdata_pipeline/synth/` | synthetic recordings with ground truth (ray-cast room, table, boxes, hands, IMU); `synth/preview.py` writes frame / GT-overlay / depth PNGs |
| `glassesdata_pipeline/backends/gt.py` | ground-truth backends (`hands2d`, `depth`, `masks`) with noise from `[synth.gt_noise]` |
| `glassesdata_pipeline/groundtruth.py` | `GroundTruth`: lazy access to `gt/` (`ep.gt`) |
| `glassesdata_pipeline/recording.py` | raw recording readers (CSV headers copied from the receiver) |
| `glassesdata_pipeline/eval/` | metrics against `gt/` and the report |
| `glassesdata_pipeline/core/` | pure algorithms (smoothing, elbow, imu_filter, align, ...) |
| `glassesdata_pipeline/sim/` | MJCF scene + robot composition (`mjcf.compose`), replay (MuJoCo or numpy wireframe), Shadow Hand tables, mink retargeting, physics check |
| `glassesdata_pipeline/viz/` | static figures `derived/viz/*.png`, Rerun export, `view REC` (opens `derived/viz/episode.rrd`) |

## Writing a step

Create `glassesdata_pipeline/steps/<name>.py` with `NAME`, `VERSION`, `REQUIRES`, `INPUTS`
(relative to the recording), `OUTPUTS` (relative to `derived/`), `CONFIG_SECTIONS` and
`run(ep, cfg) -> dict`; save outputs with `ep.save_npz(name, ep.step_meta(backend=...), ...)`.
Import heavy packages inside `run` and raise `errors.MissingDependency` when they are missing.
Test against a synthetic recording (`synth.generate`) and its `gt/`.

## Trajectory, IMU, world, arms

`slam` (metric visual odometry: KLT + depth-lifted keyframes + SE(3) pose graph; or `backend =
"import"` for a MegaSaM / DROID-SLAM / MASt3R-SLAM TUM file), `imu` (camera-IMU rotation, time
offset, gyro/accelerometer bias, gravity), `world` (gravity-aligned camera trajectory, smoothed
hands) and `arms` (shoulders, elbows with an image forearm cue). Methods are in each module's
docstring, parameters in `configs/{slam,imu,world,arms}.toml`. To develop them without the learned
models, `tests/fabricate.py` writes `hands2d`/`depth`/`hand3d` from a synthetic recording's `gt/`:

```sh
PYTHONPATH=.:tests $PY -c "import sys; from fabricate import fabricate_upstream; fabricate_upstream(sys.argv[1])" /tmp/synth_demo
$PY -m glassesdata_pipeline run /tmp/synth_demo --steps slam,imu,world,arms --no-deps && $PY -m glassesdata_pipeline eval /tmp/synth_demo
```
