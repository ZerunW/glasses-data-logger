# GlassesData

GlassesData extends Meta's CameraAccess sample with motion and input capture, live device status, and recording archives. It uses Meta Wearables DAT SDK **1.0.1**. The original camera connection, preview, photo capture, HFP microphone recording, playback, and sharing flows are available.

## Set up Xcode and sign with a free Apple Account

1. Install **Xcode 26.4 or later** with Swift 6.3 or later. In Xcode Settings → Locations, select that Xcode installation under Command Line Tools. Building for iOS requires the full Xcode installation.
2. Open `GlassesData.xcodeproj` and wait for Swift Package Manager to resolve Meta SDK **1.0.1**. Select the **GlassesData** scheme.
3. Sign in under Xcode Settings → Accounts. For the **GlassesData** app target, open **Signing & Capabilities**, enable **Automatically manage signing**, and select your **Personal Team**.
4. Change the bundle identifier from `com.example.GlassesData` to a unique value such as `com.yourname.GlassesData`. To run tests on a device, also select a team and use distinct bundle identifiers for `CameraAccessTests` and `CameraAccessUITests`.
5. Connect your iPhone over USB, unlock it, and approve the trust prompt. Enable **Developer Mode** under Settings → Privacy & Security and follow the restart prompts. Select the iPhone as the Xcode run destination and press **⌘R**. If iOS asks you to trust the developer, go to Settings → General → VPN & Device Management.
6. Apple limits the lifetime of free provisioning profiles and the number of devices and apps you can provision. Run the app from Xcode again when the profile expires. Copy any recordings you want to keep before uninstalling the app, which deletes its Documents directory.

### Personal Team capabilities

`CODE_SIGN_ENTITLEMENTS` points to `CameraAccess/PersonalTeam.entitlements` by default. This file includes Keychain Sharing. `Info.plist` declares the Bluetooth, External Accessory, and Audio background modes.

Apple's free developer accounts support Background Modes and Keychain Sharing. The Personal Team configuration omits Hotspot and Access WiFi Information because these capabilities require a paid account. The original `CameraAccess/CameraAccess.entitlements` is included for reference. [Apple Account and Personal Team overview](https://developer.apple.com/help/account/basics/about-your-developer-account) · [Supported iOS capabilities](https://developer.apple.com/help/account/reference/supported-capabilities-ios)

Test the data link with the Adventurer and its installed firmware. If the glasses require the app to establish a Wi-Fi link using either restricted capability, free provisioning may prevent that link from working.

## Privacy and security

Recordings include video, optional glasses microphone audio, motion samples, input events, and device status. They are stored in the app's Documents directory, visible through iPhone Files and Finder File Sharing, and may be included in the device's backups. Delete recordings you no longer need. Photos and videos shared with another app are subject to that app's handling.

The app sets `MWDAT.CrashReporting.OptOut` to `true`, as supported by Meta SDK 1.0.1. This only opts out of the SDK crash-reporting feature; registration, camera permission flows, device connection, and other Meta SDK behavior may still exchange data. Refer to [Meta's developer documentation](https://wearables.developer.meta.com/docs/develop/) for its current practices.

Camera streaming is disabled until started. USB mode binds the sender to iPhone loopback and requires approval for each connection. Wi-Fi must be enabled separately and resets on stop. Wi-Fi traffic is unencrypted and IUCM has no cryptographic peer authentication, so use only on trusted networks. The connection prompt does not prove the peer's identity. TetherCam's third-party camera extension has a separate local sink policy and TetherCam's own update-check settings; consult its [privacy policy](https://tethercam.app/privacy).

## Configure Meta Developer Mode

- Enable Developer Mode for the glasses in the Meta AI app. Pair them with the iPhone, power them on, unfold them, and put them on.
- `MWDAT.MetaAppID` in `Info.plist` is set to the string `0` for Developer Mode. This setup uses no Wearables Developer Center App ID or Client Token.
- Keep the `glassesdata` URL scheme and the `glassesdata://` value of `MWDAT.AppLinkURLScheme` in sync. You can change the bundle identifier independently.
- Xcode expands `MWDAT.TeamID` from `$(DEVELOPMENT_TEAM)`.
- Motion and Inputs are experimental and require authorization. Access in Developer Mode still needs testing on the glasses. The panel displays an experimental feature authorization message for `InputsError.permissionDenied`. SDK 1.0.1 defines three `MotionError` cases: `sensorUnavailable`, `capabilityClosed`, and `deviceDisconnected`.

## First run

1. Tap **Connect**, complete registration in Meta AI, and return to GlassesData. Grant Bluetooth and local network access if prompted.
2. Tap **Start session**. Motion and Inputs attach once the session reaches `.started`. The sensors are available independently of the camera preview.
3. Expand the **Sensor Panel**. Choose 5, 10, 15, 24, 30, or 60 Hz. The default is **30 Hz**. Changing the rate reattaches Motion. The rate picker is disabled during recording.
4. Tap **Preview**. For first use, follow the camera permission prompts in Meta AI and return to the app. Use the existing controls to take and share photos.
5. Use the microphone button to enable the glasses' Bluetooth HFP microphone. The app records microphone audio only if permission is granted and the glasses' HFP input is available.
6. Tap **Record** to capture video and sensor data together. Tap again to stop. After saving, you can play or share the video. Dismissing playback leaves the archive in place.
7. If the app goes into the background, the session ends, or the glasses disconnect, the app stops the active recording, flushes the CSV buffers, and closes the files. Start a new session as needed. Wait for saving to finish before recording another clip.
8. Find recordings in **Files → Browse → On My iPhone → GlassesData → Recordings**. You can also copy a recording folder with iPhone File Sharing in Finder.

The panel shows the latest **10 seconds** of sensor history and **20 input events**. The chart and latest-value display update at up to 10 Hz to reduce UI overhead; the samples-per-second indicator counts all received samples. CSV logging includes every sample and event the app receives while recording. Missing components appear as unavailable in the panel and as empty CSV fields. Transport interruptions or full SDK buffers can cause samples to be lost before the app receives them.

## Mac camera output with TetherCam

GlassesData can send the existing HEVC camera stream to the signed TetherCam for Mac app. TetherCam provides the system camera used by Zoom and other macOS apps. The iPhone sender does not need a paid Apple Developer account.

1. Download the signed Mac app from [TetherCam releases](https://github.com/Kanevry/tethercam/releases) and follow its instructions to install it in Applications and approve its camera extension.
2. On the iPhone, start the glasses session and preview. Tap the **Mac camera** icon and start broadcasting. The sender defaults to a loopback-only TCP listener on port 7878 for USB. Each connection requires approval on the iPhone before video is sent. TetherCam waits five seconds for its handshake, so approve promptly; it will retry if the request times out. The sender stops when the preview or session ends.
3. For USB, connect the iPhone with a data cable, trust the Mac when prompted, and open TetherCam. TetherCam connects through usbmuxd. It does not require an iPhone Personal Hotspot.
4. For Wi-Fi, connect both devices to the same trusted network, stop broadcasting and enable **Allow unencrypted Wi-Fi connections**, then start broadcasting again. Quit TetherCam if it is running, then launch its Mac binary from Terminal using the IP address displayed by GlassesData:

   ```bash
   /Applications/TetherCam.app/Contents/MacOS/TetherCam --debug-tcp IPHONE_IP:7878
   ```

   This is TetherCam's debug TCP mode rather than its normal USB connection. If several addresses are shown, choose the one reachable from your Mac. USB and Wi-Fi cannot receive simultaneously.
5. Select **TetherCam** as the camera in Zoom or another application. The source is the glasses stream, configured at 24 fps. TetherCam's camera may present a 30 fps output by repeating frames; that does not make the glasses capture 30 fps.

GlassesData asks for approval on the iPhone for each incoming connection and defaults to a loopback-only listener for USB. The IUCM protocol has no cryptographic peer authentication or transport encryption. An approved Wi-Fi stream can be observed or modified by someone with network access. Only approve connections you initiated, use Wi-Fi on a trusted network, and stop broadcasting when finished. The approved endpoint display is informational and does not prove device identity. The signed TetherCam extension has its own local sink authorization behavior and is not modified by this project. The TetherCam camera extension carries video only. Local recording and optional HFP audio recording remain separate.

The TetherCam compatibility layer includes MIT-licensed protocol code. See `CameraAccess/Streaming/TetherCam-LICENSE.txt`. The iOS app and test targets compile without signing. A local test passed actual HEVC from the bundled sample through the GlassesData sender to the unmodified TetherCam receiver and decoder. Real-glasses USB, Wi-Fi, format changes, latency, and Zoom use still need device testing.

## Recording files and timestamps

```text
Documents/Recordings/2026-11-11_001512/
  video.mov
  motion.csv
  inputs.csv
  device_state.csv
  info.json
```

Folders use the iPhone's local time. A numeric suffix prevents collisions when recordings start in the same second. If recording stops before the first video frame is written, or video writing fails, the folder may contain only the logs and metadata.

All `phone_time_ns` values use the same iPhone monotonic clock, `mach_continuous_time` converted to nanoseconds. The `glasses_time_ns` and `glasses_time_ms` columns preserve the timestamps supplied by the SDK. Compute time differences within the same clock domain.

Use the phone timestamp of the first successfully written video frame in `info.json` to align the video and CSV files approximately. Transport and scheduling delay affect this timestamp. The video uses the sample's HEVC timeline handling. Pauses and source timestamp resets, including mock feed loops, can affect alignment over longer recordings.

## CSV fields

Each UTF-8 CSV starts with a header row. Fields are comma separated and quoted as needed. Missing numeric values are empty.

### motion.csv

| Column | Meaning |
|---|---|
| `phone_time_ns` | iPhone monotonic timestamp when the app receives the sample, in nanoseconds |
| `glasses_time_ns` | Original `MotionSample.timestampNs` from the device, in nanoseconds |
| `ax`, `ay`, `az` | Acceleration along x, y, and z, in m/s² |
| `gx`, `gy`, `gz` | Angular velocity about x, y, and z, in rad/s |
| `mx`, `my`, `mz` | Magnetic field along x, y, and z, in µT |
| `qw` | Scalar component of the orientation quaternion |
| `qx`, `qy`, `qz` | x, y, and z components of the orientation quaternion |
| `source` | `glasses`, `neuralBand`, or `unknown` |

The file preserves SDK values, including acceleration with gravity, the original axes, and the original quaternion. The panel normalizes the quaternion and computes ZYX Euler angles in degrees: roll about x, pitch about y, and yaw about z. Verify the coordinate conventions and behavior near singular orientations on the hardware before using these values for orientation analysis.

### inputs.csv

| Column | Meaning |
|---|---|
| `phone_time_ns` | iPhone monotonic timestamp when the app receives the event, in nanoseconds |
| `glasses_time_ms` | Original `InputEvent.timestampMs` from the device, in milliseconds |
| `source` | `captouch`, `neuralBand`, `captureButton`, `actionButton`, `neuralBandDrag`, or `unknown` |
| `type` | `nav`, `select`, `back`, `button`, `capture`, or `drag` |
| `detail` | Direction for navigation, button type for button events, `shortPress` / `hold` / `doublePress` for capture events, or action and x/y/dx/dy for drag events. Empty for select and back events. |

The app observes input events for display and logging. With `consumeBack=false`, the system handles Back. Physical interactions can still cause the system to pause or end a session.

### device_state.csv

The app logs the latest device state at the start of recording when available, then logs state changes.

| Column | Meaning |
|---|---|
| `phone_time_ns` | iPhone monotonic timestamp of the state callback or initial snapshot, in nanoseconds |
| `battery` | Battery percentage, empty when unavailable |
| `charging` | SDK charging state |
| `don` | SDK wearing state |
| `hinge` | SDK hinge state |
| `thermal` | SDK thermal severity level |
| `link` | SDK connection state |

`unknown` is an explicit SDK state. An empty field means the value was unavailable.

### info.json

- `startedAt` and `endedAt` record the start and stop triggers in ISO 8601. `startedPhoneTimeNs` and `endedPhoneTimeNs` record the corresponding iPhone monotonic timestamps. Video and sensor capture begin together once the archive folder is ready.
- `samplingRateHz` is the requested Motion sampling rate. The panel's samples-per-second indicator counts received samples.
- `configuredFrameRate` is the requested video frame rate. `videoNominalFrameRate` is the MOV track's nominal frame rate when available. `videoWidth` and `videoHeight` come from the first written frame.
- `wroteHFPAudio` indicates whether HFP microphone samples were written.
- `firstVideoFramePhoneTimeNs`, `firstVideoFramePTSSeconds`, and `firstVideoFrameSourcePTSSeconds` record the first written frame's iPhone timestamp, MOV presentation time, and source presentation time.

Optional fields are omitted when unavailable.

## Test with MockDeviceKit

In a Debug build, open the debug menu using the bug icon and create a mock device. Power it on, unfold it, and mark it as worn. Choose the bundled video or the iPhone camera as the preview source. Use the motion test feed or inject navigation, select, back, button, capture, and drag events. Start the session and wait for Inputs to become active before injecting events. The SDK drops events injected before activation.

Use the mock to check the panel and recording files. Verify authorization, IMU output, HFP audio, and the data link with physical glasses. A mock device has no glasses HFP microphone.

## Build and run tests

With full Xcode installed, run this command from the project directory:

```bash
xcodebuild -project GlassesData.xcodeproj -scheme GlassesData \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath /tmp/GlassesDataDerivedData CODE_SIGNING_ALLOWED=NO build
```

List installed simulators with `xcodebuild -showdestinations -project GlassesData.xcodeproj -scheme GlassesData`. To run the tests, use `-destination 'platform=iOS Simulator,id=SIMULATOR_ID' test` with a simulator ID from that list. The scheme runs tests serially because some CameraAccess tests share a MockDeviceKit singleton.

The delivery environment had only Command Line Tools. The iOS build, simulator run, and signing remain unverified. See the [validation record](Docs/VALIDATION.md) for completed checks.

## Verify on physical hardware

- Installation with a Personal Team profile and the Adventurer data link, including any Wi-Fi capability the glasses require.
- Motion and Inputs authorization and firmware support with `MetaAppID=0`.
- Delivered sample rate at each setting, missing fields, sensor axes, quaternion orientation, event sources, and capture gestures.
- Stability, power use, temperature, and dropped samples while the camera, Motion, and Inputs run concurrently.
- HFP routing, denied microphone permission, interruption recovery, and agreement between the audio track and `info.json`.
- Playable MOV files and complete CSVs after backgrounding, locking, disconnecting, folding the glasses, ending the session, and making consecutive short recordings.
- First frame alignment, drift over long recordings, changes to the system clock, and pause/resume behavior.
- Archive visibility in Files, storage exhaustion errors, and the files available after a write failure.

The app finalizes files during normal lifecycle transitions. Force quitting, abrupt system termination, or power loss can leave buffered rows and an unfinished MOV on disk.

## Source and license

Based on Meta's CameraAccess sample at commit `bbae597` (Release 1.0.1). The implementation was checked against the Swift interfaces distributed with the 1.0.1 XCFrameworks at commit `0bd369298eef6a980ad0bdb6c8a5502f2c02a08c`, the 1.0.0 changelog, and Meta's Motion, Inputs, session lifecycle, and DAT conventions guides.

See the [original CameraAccess README](Docs/CameraAccess-original-README.md) and the [Meta license](LICENSE).
