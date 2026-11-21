# GlassesData

Record first-person demonstrations with Ray-Ban Meta glasses (one RGB camera and an IMU), turn
them into metric hand and arm skeletons, a head trajectory and object models in a gravity-aligned
world, rebuild the scene in MuJoCo and train a robot-hand policy there with RL, in the style of
EgoEngine. Three parts: an iPhone app that forwards the glasses' data, a Python receiver that
records it on a computer, and an offline pipeline.

**Status:** everything is written; the app is tested on the iOS simulator with a mock device, the
receiver and the pipeline's core on synthetic data on a Mac. **Nothing has run with the real
glasses, the real iPhone, Ubuntu or a GPU yet.** See
[Status](#status) and [First real test](#first-real-test).

## How it fits together

```text
Ray-Ban Meta glasses
  │  Meta Wearables DAT SDK 1.0.1: HEVC 720×1280 at 24 fps (.high), IMU, inputs, device state
  ▼
iPhone app (CameraAccess/, SwiftUI)
  │  optional copy on the iPhone ("Save on this iPhone", folder chosen in Files)
  │  GDL1 protocol, one of:
  │    USB   the app listens on 127.0.0.1:7900; the computer connects through usbmuxd
  │          (no pairing, no encryption)
  │    Wi-Fi the app connects to the computer: TLS 1.3, pinned certificate, token
  │          (QR pairing gdpair://v1?..., Bonjour _glassesdata._tcp)
  ▼
Receiver (Receiver/, Python, macOS or Ubuntu 24.04)
  │  web page on 127.0.0.1:8765: live video + hands, signal chain, Record, calibration,
  │  body measurements; virtual camera (OBS on macOS, v4l2loopback on Linux)
  │  recording folder: video.mp4 (original HEVC) + CSVs + calibration.json + profile.json + info.json
  ▼
Offline pipeline (Pipeline/; heavy steps on the Ubuntu + CUDA laptop)
  │  ingest → hands2d → depth → hand3d → slam → imu → world → arms → objects
  │         → scene → retarget → physics → viz            (all outputs in <recording>/derived/)
  ▼
train: PPO (rsl_rl) in MuJoCo / MuJoCo Warp, a Shadow Hand reproducing the demo's object motion
```

## Repository layout

| Path | What |
|---|---|
| `CameraAccess/` | iPhone app (SwiftUI, Meta Wearables DAT SDK 1.0.1). `Streaming/`: GDL1 link, USB listener, Wi-Fi TLS client, pairing. `Recording/`: saving on the iPhone, folder choice |
| `CameraAccessTests/`, `CameraAccessUITests/` | XCTest unit and UI tests (MockDeviceKit, `FakeReceiver`) |
| `Config/` | `Signing.xcconfig` (shared defaults) and your gitignored `Signing.local.xcconfig` |
| `GlassesData.xcodeproj` | Xcode project, scheme `GlassesData` |
| `Receiver/` | Python receiver: `glassesdata_receiver/`, `tests/`, `tools/` (`fake_sender.py`, `swift_sender/`), `scripts/` (Ubuntu setup and check) |
| `Pipeline/` | Offline pipeline: `glassesdata_pipeline/` (steps, core math, backends, sim, viz, rl, synth, eval), `configs/`, `tests/` |
| `Docs/` | `PIPELINE.md` (the pipeline's data contract), `SECURITY_REVIEW.md`, `VALIDATION.md`, `CameraAccess-original-README.md` |

Component documentation: [Receiver/README.md](Receiver/README.md) (long and complete: web page,
endpoints, recording format, clocks, calibration, virtual camera, Ubuntu) and
[Pipeline/README.md](Pipeline/README.md) with the contract in [Docs/PIPELINE.md](Docs/PIPELINE.md).
The iPhone app is documented in this file.

## Status

| Part | Written | Verified how | Not verified |
|---|---|---|---|
| iPhone app: link (USB, Wi-Fi), split screen with cards, Record controls the computer, Save on this iPhone, record-priority buffering | yes | 78 unit + 20 UI tests pass on a signed simulator with a mock device and a fake receiver (three consecutive full runs) | real iPhone and glasses; Files-picker folders in iCloud Drive and on USB drives |
| GDL1 protocol | yes | the same byte fixtures in the Swift and Python tests; the app's own link code (`tools/swift_sender`) against the receiver over TCP and pinned TLS | over a real cable or network |
| Receiver: recording, web page, live hands, Wi-Fi pairing | yes | 339 tests pass on macOS; `tools/fake_sender.py` | a real phone; MediaPipe on first-person footage |
| Receiver: camera calibration, body measurements | yes | end-to-end test with synthetic ChArUco views over HTTP: intrinsics within 0.03 %, RMS 0.12 px; calibration files validated before use and before copying into recordings | the glasses' real camera |
| Receiver: virtual camera | yes | unit tests | not yet seen in QuickTime/Zoom (automated read-back blocked by macOS camera permission) |
| Receiver on Ubuntu 24.04, `setup_ubuntu.sh`, `check_linux.sh` | yes | Linux paths unit-tested with a simulated platform; scripts `bash -n` only | never run on Ubuntu |
| Pipeline `ingest` … `arms`, `scene`, static `viz`, `eval`, `eval-charuco` | yes | 308 tests pass, 10 skipped (heavy packages); whole chain on synthetic recordings, three seeds ([VALIDATION.md](Docs/VALIDATION.md)) | real recordings |
| `hands2d` MediaPipe backend | yes | tests | first-person footage |
| Learned backends: WiLoR, HaMeR, Depth Anything V2, Video Depth Anything, SAM 2, SAM 3D Objects | written | never run | everything (needs CUDA) |
| `objects` (table, unknown objects, poses, grasps) | yes | tests with fabricated inputs; whole synthetic chain with gt masks: object position about 1 cm | SAM 2 masks, real scenes |
| `retarget` (mink, Shadow Hand), `physics`, Rerun export | written | numpy parts tested; on the Mac the run reports them as skipped | MuJoCo, mink, Rerun never imported |
| RL `train` (MuJoCo / MuJoCo Warp, rsl_rl PPO) | written | task math, observations, rewards, reference tested with numpy | never run: torch, MuJoCo, MuJoCo Warp, rsl_rl |
| `Pipeline/scripts/setup_ubuntu.sh`, `omen_smoke.sh` | yes | `bash -n` only | never run on the laptop |

## Quick starts

### 1. iPhone app: build, sign, install

Needs a Mac with Xcode 26.4 or later (last built with Xcode 27.2), an iPhone on iOS 17.2 or later,
the glasses paired in the Meta AI app, and an Apple Account (a free one is enough).

1. Open `GlassesData.xcodeproj` and let Swift Package Manager resolve `meta-wearables-dat-ios` 1.0.1.
2. Signing, kept out of git:

   ```sh
   cp Config/Signing.local.xcconfig.example Config/Signing.local.xcconfig
   ```

   Set `DEVELOPMENT_TEAM` to your team ID (Xcode Settings > Accounts) and the three
   `GD_*_BUNDLE_ID` values to identifiers of your own, such as `com.yourname.glassesdata`. Do not
   set the team or bundle ID in Xcode's Signing & Capabilities pane: that writes them into
   `project.pbxproj`. Before committing, check `git diff GlassesData.xcodeproj/project.pbxproj` for
   a `DEVELOPMENT_TEAM`, a `DevelopmentTeam` or a literal `PRODUCT_BUNDLE_IDENTIFIER`.
3. Connect the iPhone by cable, trust the Mac, enable Developer Mode (Settings > Privacy &
   Security), select it as the run destination and press ⌘R. If iOS asks, trust the developer in
   Settings > General > VPN & Device Management.
4. A free Personal Team profile **expires after 7 days**: run from Xcode again. Uninstalling the
   app deletes recordings kept in its own folder.
5. Glasses: enable Developer Mode for them in the Meta AI app. The app runs in Meta's Developer
   Mode (`MWDAT.MetaAppID = "0"` in `Info.plist`, no App ID or client token). Keep the `glassesdata`
   URL scheme and `MWDAT.AppLinkURLScheme` in sync.
6. In the app: **Connect my glasses** (confirm in Meta AI), **Start session**, **Preview**. The
   screen is split: the video beside a column of live numbers, and swipeable cards under it
   (Motion, Inputs, Glasses, Computer, Storage). **Computer** sets the link (Off, USB (default) or
   Wi-Fi) and holds **Pair over Wi-Fi**. **Storage** holds **Save on this iPhone** (off by default)
   and its folder. **Record** starts and stops the computer's recording while the link is connected
   (and follows recordings started on the computer), and records on the iPhone too when Save on
   this iPhone is on.

Recordings on the iPhone go to `Documents/Recordings` (Files > On My iPhone > GlassesData) or to a
folder picked in Files (written to a temporary folder first and moved there at the end). Each
folder `YYYY-MM-DD_HHMMSS` holds `video.mov` (the glasses' HEVC, no re-encoding, plus the glasses
microphone over Bluetooth HFP when the mic is on), `motion.csv`, `inputs.csv`,
`device_state.csv` (the receiver's columns without `mac_time_ns`) and `info.json`.

While the computer records, the phone buffers video through short Wi-Fi stalls (up to 64 MiB)
instead of dropping it; otherwise it keeps latency low (2 MiB) and drops to the next keyframe.

### 2. Receiver on macOS

```sh
cd Receiver
python3.11 -m venv ~/.venvs/glassesdata-receiver        # outside OneDrive / iCloud Drive
~/.venvs/glassesdata-receiver/bin/pip install -r requirements-dev.txt
~/.venvs/glassesdata-receiver/bin/python -m glassesdata_receiver --usb --open
```

Plug in the iPhone, trust the computer, set the app's Computer link to USB. The page shows the
video, the signal chain Glasses → iPhone → Computer → Decode → Hands, and the Record button.

- **Wi-Fi:** `--wifi --pair` once, scan the code in the app (Computer > Pair over Wi-Fi); later
  just `--wifi`. The code is a credential.
- **Virtual camera:** `pip install pyvirtualcam`, install OBS and start its virtual camera once,
  then `--virtual-camera` (or `V` on the page). Apps see "OBS Virtual Camera".
- **Without a phone:** `python tools/fake_sender.py ../CameraAccess/TestResources/plant.mp4 --listen 7901 --loop`
  and `python -m glassesdata_receiver --tcp 127.0.0.1:7901`.

### 3. Receiver on Ubuntu 24.04 (the Omen laptop)

Not run on Ubuntu yet; keep the scripts' output from the first run.

```sh
# the repository on the laptop, outside any synced folder (git clone or copy)
Receiver/scripts/setup_ubuntu.sh      # apt packages, venv (Python 3.12), v4l2loopback, ufw; asks before each sudo step
Receiver/scripts/check_linux.sh       # read-only diagnosis
cd Receiver && ~/.venvs/glassesdata-receiver/bin/python -m glassesdata_receiver --usb --open
```

Unlock the iPhone and tap Trust once (or `idevicepair pair`). The virtual camera is `/dev/video10`
labelled GlassesData. Details and troubleshooting: [Receiver/README.md](Receiver/README.md#ubuntu-2404).

### 4. Calibration and body measurements

Once per glasses and video resolution, with the glasses on and streaming, on the computer the
receiver runs on:

1. **Camera calibration > Calibrate.** Hold a bank card against the screen and match the outline
   (this sets the board's real size); **Show the board**; look at it from different distances,
   angles and parts of your view until 25 views are kept; **Solve**; check the undistorted
   picture; **Save**. Save is refused when the solution fails a sanity check (fx / fy between
   0.8 and 1.25, principal point inside the image, RMS at most 1.5 px) or views were kept after solving: solve again.
2. **Body measurements > Edit measurements:** hand length, palm length, palm width, forearm, upper
   arm, shoulder width in cm. Empty fields keep adult averages.

Every new recording gets both (`calibration.json`, `profile.json`), checked first: a calibration
file that fails the check is not copied, an implausible stored measurement falls back to its
average, and `info.json` says why. Without a calibration the pipeline assumes a 65° horizontal
field of view and no distortion.

### 5. Record

Press Record on the page (`R`), the Record button in the app, or start the receiver with
`--record`. A recording is a folder `~/Movies/GlassesData/YYYY-MM-DD_HHMMSS/` (macOS) or
`~/Videos/GlassesData/...` (Ubuntu) with `video.mp4` (the original HEVC), `frames.csv`,
`motion.csv`, `inputs.csv`, `device_state.csv`, `clock.csv`, `hands.csv`, `calibration.json`,
`profile.json` and `info.json` (format: [Receiver/README.md](Receiver/README.md#recordings)).
If closing a recording fails (a full disk, say), `info.json` gets `closeError` and the page lists
the recording as interrupted with the reason. Keep recordings out of this repository, which lives
in OneDrive.

### 6. Offline pipeline on the Omen

Not run on the laptop yet: both scripts were written on the Mac and checked with `bash -n` only;
keep their output from the first run.

```sh
# the repository on the laptop, outside any synced folder
Pipeline/scripts/setup_ubuntu.sh            # Ubuntu packages, NVIDIA driver check, venv ~/.venvs/glassesdata-pipeline,
                                            # torch (cu130 for driver >= 580, cu128 for 570-579; --cuda overrides),
                                            # requirements, SAM 2, ~/src/mujoco_menagerie; --models downloads the
                                            # weights; idempotent; asks before every sudo step
export MUJOCO_GL=egl                        # MuJoCo renders off-screen on the GPU (add to ~/.bashrc)
Pipeline/scripts/omen_smoke.sh [REC]        # first run: tests, synthetic end to end (gt, then Depth Anything V2
                                            # and SAM 2), retarget/physics/Rerun, RL on MuJoCo CPU and Warp GPU,
                                            # REC if given; logs + summary.txt in ~/glassesdata-smoke/<date>_<time>/
```

Then, for a recording `REC` (copied from the receiver's folder):

```sh
cd Pipeline
PY=~/.venvs/glassesdata-pipeline/bin/python
$PY -m glassesdata_pipeline list-steps REC          # steps and cache state
$PY -m glassesdata_pipeline run REC                 # ingest … viz into REC/derived/ (cached; --force to redo)
$PY -m glassesdata_pipeline eval-charuco REC_BOARD  # depth and hand-derived scale against the on-screen board
$PY -m glassesdata_pipeline view REC                # Rerun viewer
$PY -m glassesdata_pipeline train REC --backend mujoco --num-envs 32 --iterations 5   # CPU smoke test
$PY -m glassesdata_pipeline train REC               # MuJoCo Warp, 4096 worlds, 2000 PPO iterations
```

`--set section.key=value` overrides any parameter in `Pipeline/configs/`. The Menagerie is
expected at `~/src/mujoco_menagerie` (`retarget.menagerie_path`; `MUJOCO_MENAGERIE` is used when
that path does not exist). WiLoR and
HaMeR are optional hand backends (`setup_ubuntu.sh --with-wilor` / `--with-hamer`; both need
`MANO_RIGHT.pkl`, which you download yourself after registering). More in
[Pipeline/README.md](Pipeline/README.md).

### 7. Synthetic demo on any machine

No hardware and no GPU. Uses the core packages only (`Pipeline/requirements-core.txt`; on the
development Mac use the receiver's environment, which has them).

```sh
cd Pipeline
PY=~/.venvs/glassesdata-receiver/bin/python
$PY -m glassesdata_pipeline synth /tmp/synth_demo --seconds 8   # recording + gt/, hands drawn
$PY -m glassesdata_pipeline run /tmp/synth_demo                 # every step; exit 0
$PY -m glassesdata_pipeline eval /tmp/synth_demo                # derived/eval.md
```

About a minute on the Mac. Without MuJoCo and mink, `run` reports `retarget` as skipped and
`physics` as blocked, renders the scene replay as a wireframe and still exits 0 (it exits 1 if a
step fails, or if a step you named with `--steps` cannot run). `synth --no-draw-hands` leaves the
hands out of the image; the hand-derived depth scale is then biased (the depth map at the hands
shows the table), so keep the default for anything metric.

## First real test

The shortest path from hardware to a first processed recording. The full plan with pass criteria
is in [Docs/VALIDATION.md](Docs/VALIDATION.md#5-hardware-test-plan).

- [ ] `Config/Signing.local.xcconfig` filled in; the app installs and launches on the iPhone.
- [ ] Glasses in Developer Mode; **Connect my glasses**, **Start session**, **Preview**: video in
      the app, Motion and Inputs cards update.
- [ ] Receiver `--usb --open`, app link on USB: every hop of the signal chain near 24 fps, preview
      under 0.5 s behind.
- [ ] Write down what the code assumes: video size (720×1280?), accelerometer norm at rest
      (9.8 → m/s²), gyroscope units, the motion `source` value, the IMU rate.
- [ ] Camera calibration saved (note the RMS and field of view); body measurements entered.
- [ ] 30 s recordings started from the page and from the app; the folder has every file and
      `video.mp4` plays.
- [ ] A recording of the on-screen board from 0.3 to 1 m with a hand in view, and a tabletop
      demo: reach, grasp, lift, move and place a box.
- [ ] On the Mac: `run REC --steps ingest,hands2d` finds the hands with the right sides.
- [ ] Wi-Fi: pair, record, reconnect after a receiver restart.
- [ ] Save on this iPhone: the app's folder, an iCloud Drive folder, a USB drive.
- [ ] On the Omen: full `run` on both recordings, then `eval-charuco`: the hand-derived scale
      should agree with the board's within a few percent before building further.

## Privacy and security

Recordings show everything the wearer looks at, including other people. The USB link has no
authentication: any process on the computer can reach the app's port through usbmuxd while the
app listens, so set the link to Off when not in use. Wi-Fi uses TLS 1.3 with a pinned
certificate and a token. The receiver's page listens on 127.0.0.1 only unless told otherwise.
Apart from Meta's SDK and the Meta AI app on the phone, no component sends data to a cloud or LLM
service at runtime; the computer side only downloads model weights. Details and open issues:
[Docs/SECURITY_REVIEW.md](Docs/SECURITY_REVIEW.md).

## Source and license

The iPhone app is based on Meta's CameraAccess sample at commit `bbae597` (Release 1.0.1) and uses
`facebook/meta-wearables-dat-ios` 1.0.1 at revision `0bd369298eef6a980ad0bdb6c8a5502f2c02a08c`.
See the [original CameraAccess README](Docs/CameraAccess-original-README.md) and the
[Meta license](LICENSE).
