# Security and privacy review

Review date: 2026-11-11. This is a focused source and configuration review, not an independent penetration test or a guarantee of security.

## Data handling

- The user starts the camera preview and recording. Recordings contain video, optional HFP microphone audio, motion samples, input events, and device state under `Documents/Recordings`. File Sharing and Files access are enabled, and recordings may be included in device backups.
- The Mac camera sender is started separately and stops with the camera stream. USB mode binds to `127.0.0.1:7878`. Wi-Fi requires an explicit toggle and resets when broadcasting stops. Each connection requires iPhone-side approval; an unapproved request expires after 4.5 seconds. Endpoint information in the prompt does not authenticate the peer.
- The IUCM stream uses neither encryption nor cryptographic authentication. An attacker on the same network can observe or alter approved Wi-Fi traffic. Use USB for sensitive video, avoid untrusted networks, and stop broadcasting when finished. USB forwarding and network isolation still require physical-device testing.
- The app sets the Meta SDK `MWDAT.CrashReporting.OptOut` option. This covers the SDK's crash reporting only. Registration, permission requests, device connections, and other SDK behavior may exchange data. The Meta SDK is a precompiled dependency and was not independently audited.
- The reviewed TetherCam v0.4.0 Mac source includes an optional GitHub update check and permits local clients to feed its camera sink. The existing signed TetherCam binary is maintained by its author and is not modified by this project. Camera-source access is also governed by macOS privacy controls and the receiving application.

## Dependencies and distribution

- The project pins `facebook/meta-wearables-dat-ios` at version 1.0.1 and revision `0bd369298eef6a980ad0bdb6c8a5502f2c02a08c`. Retain Meta's terms when redistributing its sample-derived files.
- The two IUCM protocol files match the reviewed TetherCam source. Their MIT license is included in `CameraAccess/Streaming/TetherCam-LICENSE.txt`.
- TetherCam's official v0.4.0 DMG was checked with `codesign`, `stapler`, and Gatekeeper. Signature and notarization validate the publisher and package status; they do not prove source-to-binary equivalence or that the app is vulnerability-free.
- The publication snapshot was checked for common credential patterns, private keys, personal filesystem paths, user-specific Xcode state, recording folders, local tool directories, and logs. Included binary assets were compared byte-for-byte against Meta's public sample. This scan can miss secrets in unexpected formats.

## Verification limits

Xcode built the iOS app and test targets with signing disabled. The available Mac had no iOS simulator runtime, so iOS tests and the connection approval UI were not executed. TetherCam's Mac core tests and a local HEVC sender-to-receiver test passed. Real glasses, an iPhone, USB forwarding, Wi-Fi, the installed camera extension, and third-party calling apps still require end-to-end testing. See `VALIDATION.md` for details.
