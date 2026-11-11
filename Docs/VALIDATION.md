# Validation record

- The upstream Meta repository working tree was clean. Project changes were made in the `GlassesData/` copy.
- All Swift files passed `swiftc -frontend -parse`. The project file, Info.plist, and signing configuration passed `plutil` validation. The scheme references the expected targets and products.
- On macOS, 100,000 readings from `MonotonicClock` were nondecreasing. An elapsed-time check confirmed nanosecond units.
- An archive test on macOS used stand-ins for the SDK data types. It verified that all 650 motion samples were saved, periodic flushing worked, CSV headers and fields matched the expected format, missing values were empty, same-second recordings had distinct folders, the video was moved, and JSON timestamps were saved. Orientation conversion and input classification checks also passed with stand-in data.
- The project includes `RecordingStoreTests`, `SensorModelsTests`, and the original CameraAccess tests. Running them requires full Xcode. The available `xcodebuild` stopped at startup because only Command Line Tools were installed. Linking against the actual Meta SDK, building for iOS, running in the simulator, signing, and testing on a device remain unverified.

See the [hardware verification checklist](../README.md#verify-on-physical-hardware) for the remaining checks.
