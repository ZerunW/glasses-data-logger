# Validation record

Last updated 2026-11-21. Everything below ran on the development Mac (macOS, Python 3.11 in
`~/.venvs/glassesdata-receiver`, Xcode 27.2). **Nothing has run with the real glasses, the real
iPhone, Ubuntu or a CUDA GPU yet.** The synthetic numbers come from idealised data (no motion
blur, a global shutter, an exact calibration, ground-truth detectors with added noise); real video
will be worse.

## 1. Automated tests

| Suite | Command | Result on 2026-11-21 |
|---|---|---|
| Receiver | `cd Receiver && ~/.venvs/glassesdata-receiver/bin/python -m pytest tests -q -p no:cacheprovider` | **339 passed** (about 1.5 min) |
| Pipeline | `cd Pipeline && ~/.venvs/glassesdata-receiver/bin/python -m pytest tests -q -p no:cacheprovider` | **308 passed, 10 skipped** (about 40 s) |
| iOS (unit + UI, simulator) | see below | 78 unit test methods in `CameraAccessTests`, 20 UI test methods in `CameraAccessUITests` (including `ScreenshotTourTests`). Signed simulator run on iPhone 18 Pro: **78 unit + 20 UI passed** (three consecutive full runs on 2026-11-21, the last ~10 min) |

iOS tests on a simulator, signed with the team from `Config/Signing.local.xcconfig` (an unsigned
build with `CODE_SIGNING_ALLOWED=NO` crashes at launch in `Wearables.configure()`). This is the
command the iOS work was accepted with:

```sh
xcodebuild test -project GlassesData.xcodeproj -scheme GlassesData \
  -destination 'platform=iOS Simulator,name=iPhone 18 Pro' \
  -derivedDataPath /tmp/GlassesDataSimDerived \
  -test-timeouts-enabled YES -parallel-testing-enabled NO
```

Run it on an otherwise idle Mac: under heavy load the mock stream delivers no frames within the
tests' 10 s wait and the integration tests fail spuriously. List simulators with
`xcrun simctl list devices available`. The UI tests move the app's USB listener off port 7900
(`GLASSESDATA_LINK_PORT`) and drive it with `FakeReceiver`, so a receiver running on the same Mac
does not interfere; the scheme runs tests serially because some tests share the MockDeviceKit
singleton. `ScreenshotTourTests` attaches a screenshot of each main screen state
(`xcrun xcresulttool export attachments --path <xcresult> --output-path <dir>`).

**Pipeline skips** (heavy packages are not installed on the Mac): `test_retarget.py`,
`test_scene_compose.py`, `test_scene_physics.py` (need a MuJoCo Menagerie clone in
`MUJOCO_MENAGERIE`), `test_rl_heavy.py` (4 need `torch`, 1 needs `mujoco`), `test_rl_math.py` (one
torch case), `test_viz.py` (one needs `rerun`).

**What the suites cover** (tests per file from `--collect-only`):

- Receiver: GDL1 framing against byte fixtures shared with the Swift tests (`test_protocol`, 11;
  `LinkProtocolTests.swift` on the iOS side), sessions, handshake, tokens, clock and remote record
  control (`test_session`, 29), Wi-Fi TLS and pairing (`test_wifi`, 12), the web server's access
  rules (key, `Host`, CSRF headers) and endpoints (`test_web`, 44), recording, folder rotation,
  validated calibration/profile copies, atomic `info.json` and failed closes (`test_recorder`, 24;
  `test_hevc`, 9), calibration (model choice, sanity checks, stale solutions, board size limits)
  and body measurements (`test_calibration`, 35; `test_setup_web`, 17), command-line checks such
  as port ranges (`test_cli_args`, 15), usbmuxd client (`test_usbmux`, 11), virtual camera
  composition and backends (`test_vcam`, 86), Linux code paths with a simulated `sys.platform`,
  sysfs, usbmuxd and pyvirtualcam (`test_linux`, 27; `test_paths`, 9), offline tracking and the
  viewer (10).
- Pipeline: every step against a synthetic recording and its `gt/` (often with upstream files
  fabricated from ground truth: `tests/fabricate.py`, `tests/fabricate_objects.py`,
  `tests/fabricate_sim.py`), the core math modules, configuration, runner, caching and the
  skipped/blocked statuses, the evaluation, MJCF composition and the RL task, observations,
  rewards and reference in numpy.

## 2. End-to-end checks without hardware

- **Receiver calibration, end to end:**
  `tests/test_calibration.py::test_end_to_end_calibration_over_http_and_a_recording_that_carries_both_files`
  renders ChArUco views through a known camera (`tests/charuco_synth.py`), encodes them as HEVC,
  streams them with `tools/fake_sender.py` into `python -m glassesdata_receiver --tcp`, drives the
  calibration through the HTTP API and checks the result: intrinsics within 0.03 % of the true
  camera, RMS 0.12 px. The test's own thresholds are 1 % and 0.5 px.
- **Receiver without a phone:** `tools/fake_sender.py` plays an HEVC file as the app would (USB
  style, or Wi-Fi with TLS and the pinned fingerprint), with simulated IMU, device state, stats, a
  skewed clock and, with `--record-after`/`--record-for`, the phone's Record button
  (`Receiver/README.md`, Testing without the iPhone).
- **The app's own link code against the receiver:** `Receiver/tools/swift_sender/run.sh` compiles
  the app's `LinkProtocol`, `LinkPairing`, `LinkTLS` and `ComputerLink` sources for macOS and
  streams an HEVC file to the receiver over TCP (USB style) or pinned TLS (Wi-Fi style).
  Run on 2026-11-20: video and sensors arrived and recorded over both paths, and a wrong
  certificate fingerprint (refused by the app's TLS pin) and a wrong token (refused by the
  receiver) were both rejected. The record-control path (CONTROL /
  RECORDER_STATE) was also exercised from the app's UI tests against the real receiver over TCP.
- **Synthetic pipeline on the Mac:** section 3.
- **Ubuntu scripts:** `Receiver/scripts/setup_ubuntu.sh` and `check_linux.sh` were only checked
  with `bash -n` and read through.

## 3. Synthetic evaluation

### The whole chain

Each run: an 8 s synthetic recording (360×640, 24 fps, hands drawn), every step run, then `eval`.
The learned models are replaced by the `gt` backends with the noise of `[synth.gt_noise]`: 2D hand
keypoints with σ = 2 px per axis (about 2.5 px mean error), depth with a 10 % scale error (±5 %
slow drift) and 3 % per-pixel noise, ground-truth object masks. Every other step is the real code.

```sh
cd Pipeline
PY=~/.venvs/glassesdata-receiver/bin/python
$PY -m glassesdata_pipeline synth /tmp/synth_s3 --seconds 8 --seed 3
$PY -m glassesdata_pipeline run /tmp/synth_s3      # retarget skipped, physics blocked on the Mac; exit 0
$PY -m glassesdata_pipeline eval /tmp/synth_s3     # derived/eval.md
```

| Metric | seed 3 | seed 7 | seed 11 |
|---|---|---|---|
| `hand3d` MPJPE, camera frame (median about 8 mm) | 9.7 mm | 9.2 mm | 9.3 mm |
| depth-scale error (smoothed) | 0.7 % | 0.6 % | 0.8 % |
| `slam` ATE, SE(3) aligned | 0.62 mm | 0.63 mm | 0.57 mm |
| gravity direction error | 0.35° | 0.33° | 0.27° |
| IMU time offset | exact | exact | exact |
| hands MPJPE, world frame | 8.8 mm | 7.9 mm | 7.5 mm |
| elbow error (RMS) | 1.14 cm | 1.17 cm | 1.48 cm |
| object position (static about 0.8 cm, held about 1.6 cm) | 1.15 cm | 1.01 cm | 1.13 cm |
| table height error | −1.8 mm | 0.7 mm | −1.1 mm |

Known weaknesses seen in these runs:

- the table polygon extends towards the person (the forearms are not masked out of the table
  points);
- `hand3d` is off by 4-5 cm on a hand that is leaving the image;
- gravity is only good to about 0.3° on short clips;
- object volume comes out 9-12 % too small.

`synth --no-draw-hands` leaves the hands out of the image. The depth map at a hand's pixels then
shows the table behind it and the hand-derived depth scale is biased (about 15 % low in an earlier
run), which spoils everything metric after `hand3d`; keep the default for evaluations.

### Single steps with ground-truth inputs

Measured while the steps were written, with upstream files from the `gt` backends or fabricated
from ground truth with white noise (`tests/fabricate.py`, `tests/fabricate_objects.py`). The
real chain's errors are partly slow (`hand3d` smooths its depth), which later smoothing cannot
remove, so expect the chained numbers above rather than these.

| Step | Result |
|---|---|
| `hand3d` | depth-scale error 0.8 % mean, 3.2 % max; wrist depth error median 4.8 mm; camera-frame MPJPE median 7.9 mm |
| `slam` | ATE 0.8 mm (SE(3) aligned) over a 0.75 m head path |
| `imu` | time offset error 0.05 ms; camera–IMU rotation error 0.08°; gravity direction error 0.1° median |
| `world` | hand MPJPE 4.6 mm after smoothing (10.2 mm raw) |
| `arms` | elbow error 1.2 cm with the image forearm cue, 2.1 cm with the prior only |
| `objects` | grasp found automatically; size error 3-7 mm; position about 1 cm; rotation 0.1° while static, about 4° while held; table height 3.6 mm |
| receiver calibration (section 2) | intrinsics within 0.03 %, RMS 0.12 px |

### Not measured

- `retarget`, `physics`, the Rerun export: they need MuJoCo, mink and rerun-sdk, which are not
  installed on the Mac.
- RL (`train`): written and never run. The task math, observations, rewards and reference are
  unit-tested with numpy; `Pipeline/scripts/omen_smoke.sh` runs the first CPU and GPU training on
  the laptop.

## 4. Not verified anywhere yet

- **The real iPhone and glasses:** installation with a free Personal Team profile, Meta Developer
  Mode with `MetaAppID = 0`, Motion and Inputs authorization, the HEVC stream at `.high`
  (720×1280, 24 fps), the IMU's units and axes, the USB link through usbmuxd, Wi-Fi pairing, TLS
  pinning and Bonjour discovery on a real network, record-priority buffering, the Record button
  controlling the computer, saving on the iPhone.
- **Files-picker storage** in iCloud Drive and on USB drives (staging, moving, a drive removed
  during a recording, free space).
- **Ubuntu 24.04:** the receiver, `Receiver/scripts/setup_ubuntu.sh`, `check_linux.sh`, Linux
  usbmuxd pairing, the v4l2loopback virtual camera, Python 3.12 (the tests have only run on 3.11 on
  macOS); `Pipeline/scripts/setup_ubuntu.sh` and `omen_smoke.sh` (checked with `bash -n` only).
- **Virtual camera in a meeting app.** Not checked yet: the receiver opened "OBS Virtual Camera"
  and sent 1920×1080 at 30 fps without a rejection, but reading the camera back from an automated
  session was blocked by macOS camera permission. Check by hand: QuickTime Player > New Movie
  Recording > OBS Virtual Camera.
- **CUDA and every heavy backend:** WiLoR, HaMeR, Depth Anything V2, Video Depth Anything, SAM 2,
  SAM 3D Objects (`image_to_3d`), Open3D meshing, CoACD, the MuJoCo renderer, `retarget` (mink and
  the Menagerie Shadow Hand), `physics`, the Rerun export and `view`, MuJoCo Warp, rsl_rl training
  and `train --play`. These modules are written against pinned versions (`Pipeline/requirements.txt`)
  but have never been imported.
- **MediaPipe on first-person footage** (hands seen from above, partly out of view, with motion
  blur), live in the receiver or offline in `hands2d`.
- **Camera calibration and `eval-charuco` on a real recording** of the on-screen board.
- **`slam.backend = "import"`** with real MegaSaM, DROID-SLAM or MASt3R-SLAM output.

## 5. Hardware test plan

Do the phases in order; each one needs the previous. Keep the terminal output, the receiver's page
screenshots and the recording folders.

**A. iPhone app and glasses (Mac, USB).**

1. Build and install with the Personal Team (root `README.md`). Enable Developer Mode for the
   glasses in the Meta AI app, **Connect**, **Start session**, **Preview**. Pass: video at 24 fps
   in the app, sensor cards update, no Motion/Inputs authorization error.
2. Start `python -m glassesdata_receiver --usb --open`, set the app's Computer link to USB. Pass:
   signal chain shows glasses ≈ iPhone ≈ computer ≈ decode ≈ 24 fps for 5 minutes, preview lag
   under 0.5 s, no drop episodes when nothing else runs.
3. Note the facts the code assumes: video size 720×1280 at `.high`; accelerometer norm at rest
   (≈ 9.8 means m/s², ≈ 1 means g); gyroscope units (turn the head 90° and compare); the motion
   `source` value; the IMU rate.
4. Record 30 s from the page, then from the phone's Record button, then stop from the other side.
   Pass: one folder each, `video.mp4` plays, `frames.csv` rows = video frames, `clock.csv` round
   trips of a few ms, `calibration.json`/`profile.json` copied when they exist.
5. Unplug the cable during a recording, plug it back in. Pass: the recording is closed cleanly
   (`info.json` has `endedAt`), the phone shows the link state honestly.

**B. Calibration and scale (Mac).**

6. Camera calibration on the page: card scale, 25 views, Solve, Save. Record the chosen model, RMS
   and field of view. Pass: RMS under 0.5 px, straight lines in the undistorted check picture.
7. Enter the body measurements.
8. Record 60 s looking at the on-screen board from 0.3 to 1 m with one hand visible, and a short
   tabletop demo (reach, grasp, lift, move and place a box).
9. On the Mac: `run REC --steps ingest,hands2d` (MediaPipe). Pass: hands found in most frames of
   the demo, left and right correct.

**C. Wi-Fi and phone storage.**

10. `--wifi --pair`, scan the code, record over Wi-Fi. Restart the receiver: Bonjour reconnects.
    Turn Bonjour off (`--no-bonjour`): the pairing's addresses work. Walk away to provoke stalls
    while recording: `phone_record_priority` true, the queue grows and drains, no gap in
    `frames.csv` unless the 64 MiB budget is exceeded.
11. `--reset-pairing`: the phone reports that it needs pairing. **Forget this computer** works.
12. Save on this iPhone: the app's folder, a folder in iCloud Drive, a USB drive (also unplugged
    during a recording). Pass: recordings arrive or fall back to the app's folder with a message.

**D. Ubuntu on the Omen.**

13. `Receiver/scripts/setup_ubuntu.sh`, then `check_linux.sh`; keep both outputs. Make the iPhone
    trust the laptop (tap Trust, or `idevicepair pair`), then steps 2, 4 and 10 on Ubuntu. Virtual
    camera in Zoom or Meet (`ffplay /dev/video10` first).

**E. Pipeline on the Omen.**

14. `Pipeline/scripts/setup_ubuntu.sh`, `export MUJOCO_GL=egl`, then
    `Pipeline/scripts/omen_smoke.sh`: environment report, the full test suite with the heavy tests,
    the synthetic chain with gt backends and then with Depth Anything V2 and SAM 2, `retarget`,
    `physics` and the Rerun export, and RL on MuJoCo CPU and MuJoCo Warp. Pass: every stage in
    `summary.txt` passes, and the synthetic numbers are close to section 3.
15. The real recordings of step 8: `run REC` with `depth_anything_v2`, then `eval-charuco` on the
    board recording. Pass gate before building further: hand-derived scale within a few percent
    of the board's, and about 1-3 cm absolute hand accuracy.
16. `objects` with SAM 2 on the tabletop demo, `scene` (MuJoCo replay), `retarget`, `physics`.
17. `train REC --backend mujoco --num-envs 32 --iterations 5` (CPU smoke test), then
    `train REC` (MuJoCo Warp, 4096 worlds; the laptop GPU has 8 GB). Record GPU memory, iteration
    time and `--play` metrics.

**F. Security checks** (see `Docs/SECURITY_REVIEW.md`).

18. From a second user account on the computer, connect to the phone through usbmuxd while the
    app listens (expected to work: confirms the documented exposure).
19. Plug the phone into a computer it has never trusted, with the app listening, and try to
    connect. The outcome is not known yet; record it in `Docs/SECURITY_REVIEW.md`.
20. With `--web-host 0.0.0.0`, another device gets 403 without the key and works with it.
