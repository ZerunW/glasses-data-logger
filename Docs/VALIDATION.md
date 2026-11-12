# Validation record

## Wireless relay simulation (2026-11-12)

- A renewed development-signed iPhone archive built successfully. The IPA was extracted and its signature, developer-mode MetaAppID, registered-device coverage, and archive-binary match were checked. The iPhone was unavailable, so installation and real-device operation were not checked.
- An isolated lifecycle harness exercised the actual Mac model and application termination delegate with an empty test trust store, discovery disabled, and extension activation disabled. It observed the bundled Wi-Fi receiver stop, the USB receiver start, and the USB child stop on normal app termination. Physical USB communication, forced termination, and full GUI interaction were not checked.
- `SecurityTests/wireless_wirecheck.py` inspected real IUCM messages across the synthetic TLS relay. It observed CONFIG before approval, no VIDEO until the delayed approval, and VIDEO afterward.
- The original Developer ID-signed TetherCam app was copied unchanged into the GlassesData Mac app's `Contents/Helpers` directory. The nested app passed its publisher requirement and the outer app passed deep signature verification. A local synthetic-video run with the bundled receiver and the already enabled camera extension reported a streaming link and pushed six frames. No app consumed the virtual camera, and first-time extension activation from the nested app on a clean Mac was not tested.
- The Mac app includes separate USB and encrypted Wi-Fi receiver modes. The USB mode and switching modes have been compiled but not exercised with a physical iPhone.
- Pairing checks passed for matching and incorrect codes, tampering, pending confirmation, and host/client agreement.
- TLS checks passed for the correct key, incorrect key, altered encrypted records, plaintext rejection, and confidentiality of the test payload on the captured loopback connection.
- The broadcast control check passed for HELLO/START with approval, rejection, and an incorrect TLS key. This check does not inject camera frames.
- `SecurityTests/wireless_interop.py` sent the bundled HEVC sample through the actual GlassesData TLS sender and Mac relay to the TetherCam v0.4.0 receiver and decoder. With approval delayed by six seconds, it observed no decoded frames before approval and at least ten afterward. This used loopback and synthetic approval; it did not exercise an iPhone, actual Wi-Fi, the glasses, or an app consuming the macOS camera extension.
- The local Mac reported the TetherCam camera extension as activated and enabled. No iOS Simulator device was available for this run.
- The installed signed TetherCam app also reported a streaming link, a ready camera extension, and six sample frames pushed to the extension through the encrypted relay. No application consumed the virtual camera during that check, so it does not verify a meeting app's captured picture.
- Starting and quitting the packaged app and its helper still require a full interactive check.

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
