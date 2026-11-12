# Validation record

- At the initial review, the upstream Meta repository working tree was clean and project changes were made in the `GlassesData/` copy.
- All Swift files passed `swiftc -frontend -parse`. The project file, Info.plist, and signing configuration passed `plutil` validation. The scheme references the expected targets and products.
- On macOS, 100,000 readings from `MonotonicClock` were nondecreasing. An elapsed-time check confirmed nanosecond units.
- An archive test on macOS used stand-ins for the SDK data types. It verified that all 650 motion samples were saved, periodic flushing worked, CSV headers and fields matched the expected format, missing values were empty, same-second recordings had distinct folders, the video was moved, and JSON timestamps were saved. Orientation conversion and input classification checks also passed with stand-in data.
- At the initial review, only Command Line Tools were available and an iOS build was not run. Subsequent Xcode build results are recorded below.

See the [hardware verification checklist](../README.md#verify-on-physical-hardware) for the remaining checks.

## TetherCam review (2026-11-11)

- Xcode 27.2 built the iOS app and test targets for a generic iOS device with code signing disabled. There were no installed simulator devices, so XCTest and UI tests were compiled but not run on iOS.
- TetherCam Mac core source passed 46 tests with no failures or skips after building its HEVC test source. Mac app code and protocol paths matched the v0.4.0 tag. These tests do not verify an installed camera extension.
- The official TetherCam v0.4.0 DMG and embedded extension passed signature checks. The DMG had a valid stapled notarization ticket and Gatekeeper accepted the app. App signature and notarization identify the publisher; they do not establish the absence of vulnerabilities or prove source/binary equivalence.
- A local integration test sent the bundled HEVC sample through the GlassesData sender and the original TetherCam receiver and decoder, which delivered at least ten frames. The sender test automatically approved the connection; it did not exercise the iOS confirmation dialog.
- Separate local socket tests confirmed that the default USB listener rejects a non-loopback connection, sends no data before approval, closes a rejected or timed-out connection without sending data, and sends HEVC configuration and a keyframe after approval. USB forwarding on an actual iPhone has not been verified.
- Wi-Fi IUCM remains unencrypted and lacks cryptographic peer authentication. A user-approved connection can still be observed or modified by a network attacker. The signed third-party camera extension accepts local sink clients and cannot be changed here without replacing its signed binary.
