# GlassesData

GlassesData extends Meta's CameraAccess sample with motion and input capture, live device status, and recording archives. It uses Meta Wearables DAT SDK **1.0.1**. The original camera connection, preview, photo capture, HFP microphone recording, playback, and sharing flows are available.

## Set up Xcode and sign with a free Apple Account

1. Install **Xcode 26.4 or later** with Swift 6.3 or later. In Xcode Settings → Locations, select that Xcode installation under Command Line Tools. Building for iOS requires the full Xcode installation.
2. Open `GlassesData.xcodeproj` and wait for Swift Package Manager to resolve Meta SDK **1.0.1**. Select the **GlassesData** scheme.
3. Sign in under Xcode Settings → Accounts. Copy `Config/Signing.local.xcconfig.example` to `Config/Signing.local.xcconfig`, then set `DEVELOPMENT_TEAM` to your Personal Team ID and replace the three bundle identifiers with unique values such as `com.yourname.glassesdata`. Git ignores this file, so your team ID and bundle identifiers stay out of commits. All three targets use automatic signing and read these values, so tests can also run on a device.
4. Change the team and bundle identifiers only in `Config/Signing.local.xcconfig`. Leave the **Team** and **Bundle Identifier** fields in Xcode's General and **Signing & Capabilities** panes unchanged: editing them writes literal values into `project.pbxproj`, which is committed. Before committing, check `git diff GlassesData.xcodeproj/project.pbxproj` for a `DEVELOPMENT_TEAM` value, a `DevelopmentTeam` entry under `TargetAttributes`, or a literal `PRODUCT_BUNDLE_IDENTIFIER`, and remove any you find. Without a local file, the project uses the `com.example` defaults in `Config/Signing.xcconfig` with no team; unsigned and simulator builds work, but running on a device requires the local file.
5. Connect your iPhone over USB, unlock it, and approve the trust prompt. Enable **Developer Mode** under Settings → Privacy & Security and follow the restart prompts. Select the iPhone as the Xcode run destination and press **⌘R**. If iOS asks you to trust the developer, go to Settings → General → VPN & Device Management.
6. Apple limits the lifetime of free provisioning profiles and the number of devices and apps you can provision. Run the app from Xcode again when the profile expires. Copy any recordings you want to keep before uninstalling the app, which deletes its Documents directory.

### Personal Team capabilities

`CODE_SIGN_ENTITLEMENTS` points to `CameraAccess/PersonalTeam.entitlements` by default. This file includes Keychain Sharing. `Info.plist` declares the Bluetooth, External Accessory, and Audio background modes.

Apple's free developer accounts support Background Modes and Keychain Sharing. The Personal Team configuration omits Hotspot and Access WiFi Information because these capabilities require a paid account. The original `CameraAccess/CameraAccess.entitlements` is included for reference. [Apple Account and Personal Team overview](https://developer.apple.com/help/account/basics/about-your-developer-account) · [Supported iOS capabilities](https://developer.apple.com/help/account/reference/supported-capabilities-ios)

Test the data link with the Adventurer and its installed firmware. If the glasses require the app to establish a Wi-Fi link using either restricted capability, free provisioning may prevent that link from working.

## Privacy and security

Recordings include video, optional glasses microphone audio, motion samples, input events, and device status. They are stored in the app's Documents directory, visible through iPhone Files and Finder File Sharing, and may be included in the device's backups. Delete recordings you no longer need. Photos and videos shared with another app are subject to that app's handling.

The app sets `MWDAT.CrashReporting.OptOut` to `true`, as supported by Meta SDK 1.0.1. This only opts out of the SDK crash-reporting feature; registration, camera permission flows, device connection, and other Meta SDK behavior may still exchange data. Refer to [Meta's developer documentation](https://wearables.developer.meta.com/docs/develop/) for its current practices.

Camera streaming is disabled until started. USB mode binds the sender to iPhone loopback and requires approval for each connection. Wi-Fi must be enabled separately and resets on stop. Wi-Fi uses TLS and requires approval on the iPhone. Paired devices keep a device-local secret for reconnection; when no paired device exists, the app uses a temporary key displayed on the iPhone. TetherCam's third-party camera extension has a separate local sink policy and TetherCam's own update-check settings; consult its [privacy policy](https://tethercam.app/privacy).

## Configure Meta Developer Mode

- Enable Developer Mode for the glasses in the Meta AI app. Pair them with the iPhone, power them on, unfold them, and put them on.
- `MWDAT.MetaAppID` in `Info.plist` is set to the string `0` for Developer Mode. This setup uses no Wearables Developer Center App ID or Client Token.
- Keep the `glassesdata` URL scheme and the `glassesdata://` value of `MWDAT.AppLinkURLScheme` in sync. You can change the bundle identifier independently.
- Xcode expands `MWDAT.TeamID` from `$(DEVELOPMENT_TEAM)`.
- Motion and Inputs are experimental and require authorization. Access in Developer Mode still needs testing on the glasses. The panel displays an experimental feature authorization message for `InputsError.permissionDenied`. SDK 1.0.1 defines three `MotionError` cases: `sensorUnavailable`, `capabilityClosed`, and `deviceDisconnected`.

## First run

1. Tap **Connect**, complete registration in Meta AI, and return to GlassesData. Grant Bluetooth and local network access if prompted.
2. Tap **Start session**. Motion and Inputs attach once the session reaches `.started`. The sensors are available independently of the camera preview.
3. The **Sensor Panel** stays visible above the capture controls. Scroll inside it to view the charts, recent inputs, and device state. Choose 5, 10, 15, 24, 30, or 60 Hz. The default is **30 Hz**. Changing the rate reattaches Motion. The rate picker is disabled during recording.
4. Tap **Preview**. For first use, follow the camera permission prompts in Meta AI and return to the app. Use the existing controls to take and share photos.
5. Use the microphone button to enable the glasses' Bluetooth HFP microphone. The app records microphone audio only if permission is granted and the glasses' HFP input is available.
6. Tap **Record** to capture video and sensor data together. Tap again to stop. After saving, you can play or share the video. Dismissing playback leaves the archive in place.
7. If the app goes into the background, the session ends, or the glasses disconnect, the app stops the active recording, flushes the CSV buffers, and closes the files. Start a new session as needed. Wait for saving to finish before recording another clip.
8. Find recordings in **Files → Browse → On My iPhone → GlassesData → Recordings**. You can also copy a recording folder with iPhone File Sharing in Finder.

The panel shows the latest **10 seconds** of sensor history and **20 input events**. The chart and latest-value display update at up to 10 Hz to reduce UI overhead; the samples-per-second indicator counts all received samples. CSV logging includes every sample and event the app receives while recording. Missing components appear as unavailable in the panel and as empty CSV fields. Transport interruptions or full SDK buffers can cause samples to be lost before the app receives them.

## Mac camera output with TetherCam

GlassesData Mac includes the original signed TetherCam app as a bundled receiver. Its camera extension provides the system camera used by Zoom and other macOS apps. The iPhone only needs GlassesData; the TetherCam iPhone app is not used. The iPhone sender does not need a paid Apple Developer account.

1. Move GlassesData Mac into Applications and open it. When macOS requests it, approve the TetherCam camera extension in System Settings → General → Login Items & Extensions → Camera Extensions. The TetherCam receiver is included inside GlassesData Mac and is started automatically. Activation from the bundled app on a Mac that has never installed the extension still requires validation.
2. On the iPhone, start the glasses session and preview. Tap the **Mac camera** icon and start broadcasting. The sender defaults to a loopback-only TCP listener on port 7878 for USB. Each connection requires approval on the iPhone before video is sent. The sender completes the receiver handshake and sends codec format metadata while waiting up to 30 seconds for approval; no video frames are sent before you allow the connection. The sender stops when the preview or session ends.
3. For USB, select **USB** in GlassesData Mac, connect the iPhone with a data cable, and trust the Mac when prompted. Start USB broadcasting on the iPhone and approve the connection. The bundled receiver connects through usbmuxd. It does not require an iPhone Personal Hotspot.
4. For Wi-Fi, open **GlassesData Mac** and the iPhone camera preview. On the iPhone, open **Mac camera**, stop broadcasting if needed, and select **Pair a Mac**. On the Mac, select the nearby iPhone and choose **Pair with iPhone**. Scan the QR code shown on the Mac with the iPhone, or enter its six-digit code manually. Confirming the code saves a device-local pairing secret in each device's Keychain. The scanner needs camera access; manual entry is available if scanning fails.

   Select **Encrypted Wi-Fi** on the Mac, then enable **Use encrypted Wi-Fi** on the iPhone and start broadcasting. The Mac app discovers the iPhone and attempts to connect automatically; approve each video connection on the iPhone. Stop broadcasting when finished. To revoke trust, choose **Forget paired Mac** on the iPhone and **Forget device** on the Mac. Pairing a new device replaces the stored device. GlassesData starts its bundled TetherCam receiver in headless mode when the Mac app opens and stops the receiver when the app quits. macOS manages the installed camera extension separately.

   To build the Mac app from source with Xcode installed, obtain the original signed TetherCam Mac app from [TetherCam releases](https://github.com/Kanevry/tethercam/releases) and place it at `/Applications/TetherCam.app`, or point `TETHERCAM_APP` at it. Run `./MacRelay/build.sh` from this project folder. The build verifies the publisher's signature and embeds the original app unchanged. Open `MacRelay/build/GlassesData Mac.app`; you can then move it to Applications. This local GlassesData build is ad-hoc signed and is not notarized. The Mac app follows the system light or dark appearance, as do the iPhone's native panels.

5. Select **TetherCam** as the camera in Zoom or another application. The source is the glasses stream, configured at 24 fps. TetherCam's camera may present a 30 fps output by repeating frames; that does not make the glasses capture 30 fps.

The iPhone and Mac relay authenticate each other with a 256-bit random pre-shared key and use TLS 1.2 with AES-GCM. A paired device key persists in the local Keychain until forgotten; the unpaired fallback key is not persisted. This PSK cipher suite does not provide forward secrecy: disclosure of the persistent paired key could expose captured traffic from sessions that used that key. Enter the key privately and stop broadcasting when finished. TetherCam communicates with the relay over Mac loopback, not encrypted; another local process could connect to that port. The iPhone-to-Mac Wi-Fi segment remains protected by TLS. TetherCam's signed camera extension also has its own local sink authorization policy, which this project cannot change. The camera extension carries video only; local recording and optional HFP audio remain separate.

The TetherCam compatibility layer includes MIT-licensed protocol code. See `CameraAccess/Streaming/TetherCam-LICENSE.txt`. The iOS app and test targets compile without signing. A local test passed actual HEVC from the bundled sample through the GlassesData sender to the unmodified TetherCam receiver and decoder. Real-glasses USB, encrypted Wi-Fi, format changes, latency, and Zoom use still need device testing.

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

MockDeviceKit is not exposed in the app UI. It is enabled only for UI tests launched with the `--ui-testing` argument; the test harness can then control mock-device setup through the test server. The mock API can simulate device state, camera content, motion samples, and input events.

Mock data is useful for checking the sensor panel and recording pipeline. Verify authorization, real IMU output, HFP audio, and the data link with physical glasses. A mock device has no glasses HFP microphone.

## Build and run tests

With full Xcode installed, run this command from the project directory:

```bash
xcodebuild -project GlassesData.xcodeproj -scheme GlassesData \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath /tmp/GlassesDataDerivedData CODE_SIGNING_ALLOWED=NO build
```

List installed simulators with `xcodebuild -showdestinations -project GlassesData.xcodeproj -scheme GlassesData`. To run the tests, use `-destination 'platform=iOS Simulator,id=SIMULATOR_ID' test` with a simulator ID from that list. The scheme runs tests serially because some CameraAccess tests share a MockDeviceKit singleton.

An iOS device build and development signing have been verified. No iOS simulator runtime was available for the latest checks; see the [validation record](Docs/VALIDATION.md) for completed checks and remaining limitations.

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

## Local security checks

Run `python3 SecurityTests/tls_integration.py` and `python3 SecurityTests/broadcast_approval.py` on macOS with Xcode command-line tools and an OpenSSL installation that supports TLS 1.2 PSK suites. These tests use public test fixtures and loopback traffic. They check successful TLS, incorrect-key and plaintext rejection, encrypted-record tampering, absence of a known test payload in captured traffic, and the broadcast server's approval behavior. They do not establish physical-device reliability or audit the entire app.

Run `python3 SecurityTests/wireless_interop.py /path/to/tethercam` with a local TetherCam v0.4.0 source checkout to exercise the GlassesData sender and Mac relay over encrypted loopback with the TetherCam receiver and decoder. It delays approval for six seconds, checks that no decoded video arrives before approval, then requires at least ten decoded sample frames. The check does not exercise physical Wi-Fi, the camera extension, or real glasses.

Run `python3 SecurityTests/wireless_wirecheck.py` to inspect the actual protocol messages through the synthetic encrypted relay and check that video frames are gated until approval.

Short-code pairing uses ephemeral X25519 key agreement, a commitment exchange, and confirmation messages. The six-digit code must be transferred in person. Pairing accepts one attempt at a time and must be started explicitly on the iPhone. `python3 SecurityTests/pairing_protocol.py` tests matching codes, wrong codes, and altered commitments with local synthetic peers. This new pairing protocol and the camera scanner have not received independent security review or physical-device validation.
