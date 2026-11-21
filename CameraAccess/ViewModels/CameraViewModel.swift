/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

import AVFoundation
import MWDATCamera
import MWDATCore
import Observation
import SwiftUI
import UIKit

extension TimeInterval {
  /// Formats a duration as `mm:ss`.
  var formattedRecordingTime: String {
    let minutes = Int(self) / 60
    let seconds = Int(self) % 60
    return String(format: "%02d:%02d", minutes, seconds)
  }
}

/// A capture awaiting preview/share — a still photo or a recorded video file.
/// Drives a single `.sheet(item:)` so photo and video share one presentation.
enum CapturePreview: Identifiable {
  case photo(UIImage)
  case video(URL)

  var id: String {
    switch self {
    case .photo: return "photo"
    case .video(let url): return "video:\(url.absoluteString)"
    }
  }
}

/// Drives the camera screen by exercising the SDK's camera lifecycle as explicit
/// steps: create/start a `DeviceSession`, add and start a `Stream`, capture or
/// record, stop the stream, end the session. Owns the screen-scoped pieces —
/// active-device monitoring, the `DeviceSession`, and its `Stream`. The UI binds to
/// the SDK's own `DeviceSessionState` / `StreamState` so the sample shows the real
/// state machine.
@Observable
@MainActor
final class CameraViewModel {
  // MARK: - Device

  /// Whether an eligible device is currently active. Drives the "Put on your
  /// glasses" placeholder. Sourced from the auto device selector.
  private(set) var hasActiveDevice: Bool = false

  // MARK: - Session state (bound directly to the SDK)

  /// The device session's state, mirrored from `DeviceSession.statePublisher`.
  /// `.idle`/`.stopped` mean no live session.
  var sessionState: DeviceSessionState = .idle

  // MARK: - Preview

  /// The latest decoded preview frame. Read only by the isolated `LivePreviewView`
  /// so the ~24fps updates don't re-render the whole screen.
  var currentVideoFrame: UIImage?
  var hasReceivedFirstFrame: Bool = false
  /// Pixel size of the first decoded frame of the current stream, for the summary
  /// column. Set once per stream (not per frame) so readers don't re-render at 24fps.
  private(set) var firstFrameSize: CGSize?
  /// Frames per second the SDK actually delivered, measured over the last second.
  /// Published once a second while a stream exists; `nil` without a stream.
  private(set) var measuredStreamFps: Double?

  /// The stream's state, mirrored from `Stream.statePublisher`. `.stopped` means
  /// no live stream.
  var streamState: StreamState = .stopped

  // MARK: - Capture preview

  /// The capture currently shown in the preview/share sheet (photo or video).
  var activePreview: CapturePreview?
  var isCapturingPhoto: Bool = false

  // MARK: - Video recording

  var isRecording: Bool = false
  var isPreparingRecording: Bool = false
  var isFinalizingRecording: Bool = false
  let sensorManager = SensorManager()
  @ObservationIgnored private var recordingStore: RecordingStore?
  @ObservationIgnored private var recordingErrorTask: Task<Void, Never>?
  @ObservationIgnored private var recordingGeneration = UUID()
  @ObservationIgnored private var sessionGeneration = UUID()
  @ObservationIgnored private var sensorLifecycleTask: Task<Void, Never>?
  @ObservationIgnored private var sensorAttachedSession: DeviceSession?
  @ObservationIgnored private var backgroundTask: UIBackgroundTaskIdentifier = .invalid
  /// Wall-clock start of the active recording (first written frame), mirrored from the
  /// recorder. Drives the isolated `RecordingTimerLabel`'s `TimelineView`, so the timer
  /// counts continuously — including through a stream pause. `nil` when not recording.
  var recordingStartDate: Date?
  /// Whether to record the glasses-microphone (Bluetooth HFP) audio into videos
  /// (sound-in-video). Toggled from the mic button shown while streaming.
  var includeAudioInStream: Bool = true
  /// Mic denied — iOS won't re-prompt. Drives the toggle's disabled look + tap-to-Settings.
  /// Still required for glasses audio: iOS gates HFP capture behind mic permission too.
  var micDenied: Bool { AVAudioApplication.shared.recordPermission == .denied }

  // MARK: - Errors

  /// Drives the single error alert. Uses the SDK's own `localizedDescription`
  /// rather than an app-maintained error map.
  var showError: Bool = false
  var errorMessage: String = ""

  /// Drives the confirm prompt shown before the camera-permission redirect to the
  /// Meta AI app — surfaced only when permission isn't already granted.
  var showCameraPermissionRedirectConfirm: Bool = false

  // MARK: - Derived session state

  var isSessionStarting: Bool { sessionState == .starting }
  var isSessionActive: Bool { sessionState == .started }
  /// A session exists and is connected (or connecting); a stream can be started.
  var hasSession: Bool {
    switch sessionState {
    case .starting, .started, .paused, .stopping: return true
    case .idle, .stopped: return false
    }
  }

  /// Human-readable session state for the status line.
  var sessionStateText: String { sessionState.description }

  // MARK: - Derived stream state

  var isStreaming: Bool { streamState == .streaming }
  /// A stream is attached (in any non-terminal state).
  var hasStream: Bool { streamState != .stopped }

  /// A session or stream start/stop is in flight. The UI shows a spinner and
  /// disables the controls until the SDK settles on a stable state, so transient
  /// states never need their own controls or labels.
  var isBusy: Bool {
    isSessionStarting || sessionState == .stopping
      || streamState == .starting || streamState == .waitingForDevice || streamState == .stopping
  }

  /// Human-readable stream state for the status line. `StreamState` has no
  /// `description`, so map it here.
  var streamStateText: String {
    switch streamState {
    case .streaming: return "streaming"
    case .starting: return "starting"
    case .waitingForDevice: return "waiting for device"
    case .stopping: return "stopping"
    case .stopped: return "stopped"
    case .paused: return "paused"
    }
  }

  /// The stream is paused (device-initiated, e.g. a single cap-touch tap). The
  /// last frame is retained, so the preview can stay on screen frozen.
  var isPaused: Bool { streamState == .paused }

  /// The preview surface (live frames, or the last held frame) should fill the
  /// screen: while streaming, while paused (frozen on the last frame), or while
  /// tearing down with a last frame so the stop loader overlays the preview.
  var showsLivePreview: Bool {
    isStreaming
      || ((streamState == .paused || streamState == .stopping) && currentVideoFrame != nil)
  }

  // MARK: - Private

  @ObservationIgnored private let wearables: WearablesInterface
  @ObservationIgnored private let deviceSelector: AutoDeviceSelector
  @ObservationIgnored private var deviceSession: DeviceSession?
  @ObservationIgnored private var camera: MWDATCamera.Camera?
  @ObservationIgnored private var isStoppingForBackground: Bool = false
  @ObservationIgnored private var backgroundStopSuppressionTask: Task<Void, Never>?

  // MARK: - Computer link

  /// How the phone reaches the Mac receiver: off, USB (the Mac connects through the
  /// cable) or Wi-Fi (the phone connects to the paired Mac). Persisted.
  private(set) var linkMode: LinkMode
  private(set) var linkStatus: LinkStatus = .off
  /// The Mac paired for Wi-Fi, from the Keychain.
  private(set) var linkPairing: LinkPairing?
  /// The glasses stream resolution. Persisted; can't change while a stream exists.
  private(set) var streamResolution: StreamingResolution
  /// Independent of the glasses session and preview, so the Mac can connect first.
  @ObservationIgnored private let link: ComputerLink
  @ObservationIgnored private let linkHello: LinkHello
  @ObservationIgnored private let defaults: UserDefaults

  static let linkModeKey = "linkMode"
  private static let streamResolutionKey = "streamResolution"
  private static let streamFrameRate = ComputerLink.defaultFrameRate
  /// UI tests move the USB listener off the default port, so a receiver running on the
  /// Mac (the simulator shares its loopback address) can't connect to the app under test.
  private static let linkPortEnvironmentKey = "GLASSESDATA_LINK_PORT"

  // MARK: - Computer recording

  /// The computer's recorder as last reported, and any command awaiting its answer.
  private(set) var computer = ComputerRecorderTracker()
  /// Whether the computer is recording. Kept apart from `computer` (whose time and frame
  /// count change every second) so views that only need this don't re-render each second.
  private(set) var isComputerRecording = false

  /// The phone, the computer, or both are recording.
  var isAnyRecordingActive: Bool { isRecording || isComputerRecording }

  /// What the Record button shows and does.
  var recordControl: RecordControl {
    RecordControl.evaluate(
      RecordControl.Inputs(
        isStreaming: isStreaming, linkConnected: linkStatus.phase == .connected, saveOnPhone: saveOnPhone,
        phoneRecording: isRecording, phoneBusy: isPreparingRecording || isFinalizingRecording,
        computer: computer.recorder, pending: computer.pending?.command))
  }

  // MARK: - Recording on this iPhone

  /// Record on this iPhone too. Off by default, since recordings use a lot of space. Persisted.
  private(set) var saveOnPhone: Bool
  /// The destination folder for phone recordings, with its free space.
  private(set) var recordingFolder = RecordingFolderInfo(
    name: "Recordings", location: RecordingLocation.appFolderLocation, freeBytes: nil, isAppFolder: true, problem: nil)
  /// The outcome of the latest phone recording, for the Storage card.
  private(set) var lastSaveMessage: String?
  private(set) var lastSaveFailed = false
  /// A picked folder's bookmark; nil for the app's own Documents/Recordings.
  @ObservationIgnored private var folderBookmark: Data?
  /// The picked folder the current phone recording will be moved into when it ends.
  @ObservationIgnored private var recordingDestination: URL?
  /// Keeps a picked folder readable while one of its videos is in the capture preview.
  @ObservationIgnored private var previewAccess: SecurityScopedAccess?
  private static var didRecoverStagedRecordings = false

  func setSaveOnPhone(_ enabled: Bool) {
    guard enabled != saveOnPhone else { return }
    saveOnPhone = enabled
    defaults.set(enabled, forKey: RecordingLocation.saveOnPhoneKey)
    if enabled { refreshRecordingFolder() }
  }

  /// Remembers a folder picked in Files. Returns a message when it can't be used.
  func chooseRecordingFolder(_ url: URL) -> String? {
    do {
      let bookmark = try RecordingLocation.makeBookmark(for: url)
      folderBookmark = bookmark
      defaults.set(bookmark, forKey: RecordingLocation.folderBookmarkKey)
    } catch {
      return "Couldn't use that folder: \(error.localizedDescription)"
    }
    refreshRecordingFolder()
    return nil
  }

  /// Goes back to the app's own Recordings folder.
  func useAppRecordingFolder() {
    folderBookmark = nil
    defaults.removeObject(forKey: RecordingLocation.folderBookmarkKey)
    refreshRecordingFolder()
  }

  /// Re-reads the destination folder's name and free space, off the main actor.
  func refreshRecordingFolder() {
    let bookmark = folderBookmark
    Task { [weak self] in
      let (info, refreshed) = await Task.detached(priority: .utility) {
        Self.describeRecordingFolder(bookmark)
      }.value
      guard let self, self.folderBookmark == bookmark else { return }
      if let refreshed { self.storeRefreshedBookmark(refreshed) }
      if info != self.recordingFolder { self.recordingFolder = info }
    }
  }

  private func storeRefreshedBookmark(_ bookmark: Data) {
    folderBookmark = bookmark
    defaults.set(bookmark, forKey: RecordingLocation.folderBookmarkKey)
  }

  nonisolated private static func describeRecordingFolder(_ bookmark: Data?) -> (RecordingFolderInfo, Data?) {
    let appFolder = (try? RecordingLocation.appFolder())
      ?? FileManager.default.temporaryDirectory.appendingPathComponent("Recordings")
    guard let bookmark else { return (RecordingLocation.describe(appFolder, isAppFolder: true), nil) }
    switch RecordingLocation.resolve(bookmark) {
    case .folder(let url, let refreshed):
      return (RecordingLocation.describe(url, isAppFolder: false), refreshed)
    case .unavailable(let reason):
      var info = RecordingLocation.describe(appFolder, isAppFolder: true)
      info.problem = "\(reason) Until then, recordings go here."
      return (info, nil)
    }
  }

  /// UI tests can't drive the Files folder picker, so `--ui-test-recording-folder <name>`
  /// stands in for it: a folder of that name in Documents, bookmarked for this run only.
  private static func uiTestFolderBookmark() -> Data? {
    #if DEBUG
    let arguments = ProcessInfo.processInfo.arguments
    guard arguments.contains("--ui-testing"), let flag = arguments.firstIndex(of: "--ui-test-recording-folder"),
      arguments.indices.contains(flag + 1),
      let documents = try? FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
    else { return nil }
    let folder = documents.appendingPathComponent(arguments[flag + 1], isDirectory: true)
    try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    return try? folder.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil)
    #else
    return nil
    #endif
  }

  func setLinkMode(_ mode: LinkMode) {
    guard mode != linkMode else { return }
    linkMode = mode
    defaults.set(mode.rawValue, forKey: Self.linkModeKey)
    restartLink()
  }

  /// Parses and stores a scanned `gdpair://` code, then switches the link to Wi-Fi.
  /// Returns a message for the UI when the code can't be used.
  func applyPairing(uri: String) -> String? {
    guard let pairing = LinkPairing(uri: uri) else {
      return "That isn't a valid GlassesData pairing code."
    }
    do {
      try LinkPairingStore.save(pairing)
    } catch {
      return "Couldn't save the pairing securely."
    }
    linkPairing = pairing
    linkMode = .wifi
    defaults.set(LinkMode.wifi.rawValue, forKey: Self.linkModeKey)
    restartLink()
    return nil
  }

  /// Removes the paired Mac. Returns a message for the UI when the Keychain refuses.
  func forgetPairing() -> String? {
    do {
      try LinkPairingStore.remove()
    } catch {
      return "Couldn't remove the pairing."
    }
    linkPairing = nil
    if linkMode == .wifi { restartLink() }
    return nil
  }

  func setStreamResolution(_ resolution: StreamingResolution) {
    guard !hasStream, resolution != streamResolution else { return }
    streamResolution = resolution
    defaults.set(String(describing: resolution), forKey: Self.streamResolutionKey)
  }

  private func restartLink() {
    guard linkMode != .off else {
      link.stop()
      applyLinkStatus(.off)
      return
    }
    link.start(mode: linkMode, pairing: linkPairing, hello: linkHello) { [weak self] status in
      // The main queue keeps status updates in the order the link produced them.
      DispatchQueue.main.async { [weak self] in
        MainActor.assumeIsolated { self?.applyLinkStatus(status) }
      }
    }
  }

  private func applyLinkStatus(_ status: LinkStatus) {
    if status != linkStatus { linkStatus = status }
    var tracker = computer
    let outcome = tracker.apply(status)
    updateComputer(tracker)
    if let outcome { handleComputerOutcome(outcome) }
  }

  /// Assigns only real changes, so the once-a-second link status doesn't re-render views.
  private func updateComputer(_ tracker: ComputerRecorderTracker) {
    if tracker != computer { computer = tracker }
    let recording = tracker.recorder.isRecording
    if recording != isComputerRecording { isComputerRecording = recording }
  }

  /// Sends a CONTROL command; the computer answers with a RECORDER_STATE, or the command
  /// expires after `ComputerRecorderTracker.answerTimeout`.
  private func sendComputerCommand(_ command: LinkControlCommand) {
    guard link.sendControl(command) else { return }
    var tracker = computer
    let id = tracker.sent(command, afterUpdate: linkStatus.recorderUpdates)
    updateComputer(tracker)
    Task { [weak self] in
      try? await Task.sleep(for: ComputerRecorderTracker.answerTimeout)
      guard let self else { return }
      var tracker = self.computer
      let outcome = tracker.expire(id)
      self.updateComputer(tracker)
      if let outcome { self.handleComputerOutcome(outcome) }
    }
  }

  private func handleComputerOutcome(_ outcome: ComputerRecorderTracker.Outcome) {
    // The note shows in the header either way; an alert only when the user's tap
    // recorded nothing at all.
    guard !isRecording, !isPreparingRecording else { return }
    switch outcome {
    case .startRefused(let note):
      showError("The computer didn't start recording: \(note)")
    case .noAnswer(.recordStart):
      showError("The computer didn't answer the record request. Check that the receiver is up to date.")
    case .noAnswer(.recordStop):
      showError("The computer didn't answer the stop request. Its recording may still be running.")
    }
  }

  private func wireSensorLink() {
    let link = self.link
    sensorManager.onLinkMotion = { sample, time in link.submitMotion(sample, phoneTimeNs: time) }
    sensorManager.onLinkInput = { event, time in link.submitInput(event, phoneTimeNs: time) }
    sensorManager.onLinkDeviceState = { state, time in link.submitDeviceState(state, phoneTimeNs: time) }
  }

  private static func makeLinkHello() -> LinkHello {
    let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown"
    var system = utsname()
    uname(&system)
    let machine = withUnsafeBytes(of: &system.machine) { raw in
      String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
    }
    return LinkHello(app: version, device: machine.isEmpty ? UIDevice.current.model : machine, name: UIDevice.current.name)
  }

  @ObservationIgnored private let videoRecorder = VideoRecorder()
  /// Counts frames on the SDK's frame thread; drained once a second by `fpsTask`.
  @ObservationIgnored private let frameCounter = FrameCounter()
  @ObservationIgnored private var fpsTask: Task<Void, Never>?
  // Decodes compressed hvc1 frames for on-screen preview. Not required for
  // recording — the passthrough writer handles compressed frames directly.
  @ObservationIgnored private let videoFrameDecoder = VideoFrameDecoder()

  @ObservationIgnored private var deviceMonitorTask: Task<Void, Never>?
  @ObservationIgnored private var appLifecycleTask: Task<Void, Never>?
  @ObservationIgnored private let sessionTokenBag = ListenerTokenBag()
  @ObservationIgnored private let streamTokenBag = ListenerTokenBag()

  @ObservationIgnored private let photoCaptureFailureMessage =
    "Couldn't capture photo. There may not be enough storage, or another capture is in progress. Try again in a few moments."
  @ObservationIgnored private let backgroundStopErrorSuppressionTimeout: Duration

  // MARK: - Init

  init(
    wearables: WearablesInterface,
    backgroundStopErrorSuppressionTimeout: Duration = .seconds(5),
    defaults: UserDefaults = .standard
  ) {
    self.wearables = wearables
    self.deviceSelector = AutoDeviceSelector(wearables: wearables)
    self.backgroundStopErrorSuppressionTimeout = backgroundStopErrorSuppressionTimeout
    self.defaults = defaults
    linkMode = defaults.string(forKey: Self.linkModeKey).flatMap(LinkMode.init(rawValue:)) ?? .usb
    let storedResolution = defaults.string(forKey: Self.streamResolutionKey)
    streamResolution = StreamingResolution.allCases.first { String(describing: $0) == storedResolution } ?? .high
    saveOnPhone = defaults.bool(forKey: RecordingLocation.saveOnPhoneKey)
    folderBookmark = Self.uiTestFolderBookmark() ?? defaults.data(forKey: RecordingLocation.folderBookmarkKey)
    linkPairing = try? LinkPairingStore.load()
    linkHello = Self.makeLinkHello()
    let port = ProcessInfo.processInfo.environment[Self.linkPortEnvironmentKey].flatMap(UInt16.init)
    link = ComputerLink(usbPort: port ?? LinkProtocol.defaultPort)
    startDeviceMonitoring()
    startAppLifecycleMonitoring()
    if linkMode != .off { restartLink() }
    refreshRecordingFolder()
    if !Self.didRecoverStagedRecordings {
      Self.didRecoverStagedRecordings = true
      Task.detached(priority: .utility) { RecordingLocation.recoverStagedRecordings() }
    }
  }

  isolated deinit {
    deviceMonitorTask?.cancel()
    appLifecycleTask?.cancel()
    backgroundStopSuppressionTask?.cancel()
    recordingErrorTask?.cancel()
    fpsTask?.cancel()
    link.stop()
    deviceSession?.stop()
  }

  // MARK: - Lifecycle step 1: session

  /// Creates and starts a `DeviceSession` (no stream yet). This is the first
  /// explicit step — a session can stay connected with no active stream. The
  /// state observer drives `sessionState`; we only seed an optimistic
  /// `.starting` so the UI reflects the in-flight request immediately.
  func startSession() {
    guard deviceSession == nil else { return }
    do throws(DeviceSessionError) {
      sessionGeneration = UUID()
      let session = try wearables.createSession(deviceSelector: deviceSelector)
      deviceSession = session
      // Subscribe before start() so no initial state transitions are missed.
      observeSession(session)
      // Reflect the in-flight request immediately; the observer drives it onward.
      sessionState = .starting
      try session.start()
    } catch {
      handleSessionError(error)
      sessionState = .idle
      cleanupSession()
    }
  }

  /// Stops the stream (if any) and ends the device session. `stop()` is terminal;
  /// the SDK cascades the stop to the stream and the state observers clean up
  /// when `.stopped` arrives.
  func endSession() {
    guard let session = deviceSession else { return }
    guard sessionState != .stopping else { return }
    // Ending the session ends the data the computer records, so stop its recording too.
    if isComputerRecording, computer.pending?.command != .recordStop { sendComputerCommand(.recordStop) }
    if isRecording {
      beginRecordingBackgroundTask()
      videoRecorder.prepareToStop()
      Task { await stopVideoRecording() }
    }
    // Reflect the in-flight stop immediately so the controls disable at once; the
    // state observer drives it onward to `.stopped`.
    sessionState = .stopping
    let previous = sensorLifecycleTask
    sensorAttachedSession = nil
    sensorLifecycleTask = Task {
      await previous?.value
      await sensorManager.detach()
      session.stop()
    }
  }

  // MARK: - Lifecycle step 2: stream

  /// Adds a `Stream` to the started session and starts it. Requires a session in
  /// `.started`. Camera permission is checked first (a query, no redirect); if it
  /// isn't already granted, the actual request — which redirects to the Meta AI
  /// app — is deferred to `confirmCameraPermissionRedirect()` so the app-switch is
  /// confirmed, not a surprise.
  func startStreaming() async {
    guard let session = deviceSession, session.state == .started else {
      showError("Start a session before previewing.")
      return
    }
    guard camera == nil else { return }

    do {
      if try await wearables.checkPermissionStatus(.camera) == .granted {
        beginStream(on: session)
      } else {
        showCameraPermissionRedirectConfirm = true
      }
    } catch {
      showError(error.localizedDescription)
    }
  }

  /// Confirmed from the permission prompt: requests camera access — this opens the
  /// Meta AI app and resumes when the user returns — then starts the stream if
  /// access was granted.
  func confirmCameraPermissionRedirect() async {
    showCameraPermissionRedirectConfirm = false
    guard let session = deviceSession, session.state == .started, camera == nil else { return }
    do {
      guard try await wearables.requestPermission(.camera) == .granted else {
        showError("Camera permission denied")
        return
      }
    } catch {
      showError(error.localizedDescription)
      return
    }
    beginStream(on: session)
  }

  /// Creates the camera `Stream` on the started session and starts it. Callers
  /// must have ensured camera permission is granted first.
  private func beginStream(on session: DeviceSession) {
    guard camera == nil else { return }
    // hvc1 (compressed HEVC) so frames can be written to file in passthrough mode.
    // Glasses-mic (Bluetooth HFP) audio for sound-in-video is captured app-side by
    // AudioCaptureHandler, so the SDK stream needs only the public video config (no audio codec).
    let config = StreamConfiguration(
      videoCodec: VideoCodec.hvc1,
      resolution: streamResolution,
      frameRate: UInt(Self.streamFrameRate)
    )

    do {
      // Add the consolidated `Camera` capability and stream through `camera.stream`.
      guard let newCamera = try session.addCamera(config: config) else {
        showError("Couldn't create the camera. Try again.")
        return
      }
      camera = newCamera
      // Subscribe before start() so no initial state transitions are missed.
      setupStreamListeners(for: newCamera.stream)
      // Reflect the in-flight request immediately; the observer drives it onward.
      streamState = .starting
      startFpsMeasurement()
      newCamera.stream.start()
    } catch {
      camera = nil
      showError(error.localizedDescription)
    }
  }

  /// Stops the camera stream but keeps the `DeviceSession` connected, so
  /// streaming can be started again without re-creating the session. The stream
  /// `.stopped` observer performs the teardown — we don't eagerly reset here.
  func stopStreaming() {
    guard let activeCamera = camera else { return }
    if isRecording { Task { await stopVideoRecording() } }
    videoRecorder.prepareToStop()
    // Reflect the in-flight request immediately; the observer drives it onward.
    streamState = .stopping
    // Stopping the camera cascades to its stream child and detaches the camera from the
    // session, so streaming can be started again with a fresh `addCamera()`.
    activeCamera.stop()
  }

  // MARK: - Capture

  func capturePhoto() {
    guard !isCapturingPhoto, isStreaming else {
      showError(photoCaptureFailureMessage)
      return
    }
    isCapturingPhoto = true
    let success = camera?.stream.capturePhoto(format: .jpeg) ?? false
    if !success {
      isCapturingPhoto = false
      showError(photoCaptureFailureMessage)
    }
  }

  /// The Record button: starts or stops "a recording", which is the computer's recording
  /// while the link is connected plus this iPhone's when Save on this iPhone is on.
  func toggleRecording() {
    let control = recordControl
    guard control.isEnabled else { return }
    switch control.mode {
    case .start:
      if control.startsComputer { sendComputerCommand(.recordStart) }
      if control.startsPhone { startVideoRecording() }
    case .stop:
      // Also when the computer hasn't confirmed a start yet: it may be about to record.
      if isComputerRecording || computer.pending?.command == .recordStart { sendComputerCommand(.recordStop) }
      if isRecording { Task { await stopVideoRecording() } }
    }
  }

  /// Stops the computer's recording alone, from the Computer card. Works without a glasses
  /// session, for a recording started on the computer.
  func stopComputerRecording() {
    guard isComputerRecording, computer.pending?.command != .recordStop else { return }
    sendComputerCommand(.recordStop)
  }

  func startVideoRecording() {
    guard isStreaming, !isRecording, !isPreparingRecording, !isFinalizingRecording else { return }
    isPreparingRecording = true
    let rate = sensorManager.samplingRateHz
    let startedAt = Date()
    let startedPhoneTimeNs = MonotonicClock.nowNanoseconds()
    let session = deviceSession
    let bookmark = folderBookmark
    Task {
      await prepareVideoRecording(
        rate: rate, startedAt: startedAt, startedPhoneTimeNs: startedPhoneTimeNs, session: session, bookmark: bookmark)
    }
  }

  private func prepareVideoRecording(
    rate: Int, startedAt: Date, startedPhoneTimeNs: UInt64, session: DeviceSession?, bookmark: Data?
  ) async {
    defer { isPreparingRecording = false }
    let store: RecordingStore
    let destination: URL?
    do {
      // A recording for a picked folder is written to the temporary folder and moved
      // into place when it ends; one for the app's folder is written there directly.
      let prepared = try await Task.detached(priority: .userInitiated) { () throws -> (RecordingStore, URL?, Data?) in
        var destination: URL?
        var refreshed: Data?
        if let bookmark, case .folder(let url, let newBookmark) = RecordingLocation.resolve(bookmark) {
          destination = url
          refreshed = newBookmark
        }
        let root = destination == nil ? nil : try RecordingLocation.stagingFolder()
        let store = try RecordingStore(
          samplingRateHz: rate, rootURL: root, startedAt: startedAt, startedPhoneTimeNs: startedPhoneTimeNs)
        return (store, destination, refreshed)
      }.value
      store = prepared.0
      destination = prepared.1
      if let refreshed = prepared.2, folderBookmark == bookmark { storeRefreshedBookmark(refreshed) }
    } catch {
      showError("Couldn't create recording folder: \(error.localizedDescription)")
      return
    }
    guard isStreaming, sessionState == .started, deviceSession === session,
      UIApplication.shared.applicationState != .background else {
      _ = try? await store.finish(videoSourceURL: nil)
      if let destination { _ = await Self.deliver(store.directoryURL, to: destination) }
      return
    }
    recordingStore = store
    recordingDestination = destination
    recordingErrorTask?.cancel()
    recordingErrorTask = Task { [weak self, store] in
      while !Task.isCancelled {
        try? await Task.sleep(for: .seconds(1))
        guard !Task.isCancelled else { return }
        if let error = await store.currentError() {
          self?.showError("Recording log write failed: \(error.localizedDescription)")
          return
        }
      }
    }
    recordingGeneration = UUID()
    sensorManager.onMotion = { [store] sample, time in store.appendMotion(sample, phoneTimeNs: time) }
    sensorManager.onInput = { [store] event, time in store.appendInput(event, phoneTimeNs: time) }
    sensorManager.onDeviceState = { [store] state, time in store.appendDeviceState(state, phoneTimeNs: time) }
    if let state = sensorManager.latestDeviceState {
      store.appendDeviceState(state, phoneTimeNs: MonotonicClock.nowNanoseconds())
    }
    // Flip the intent flag synchronously so the toggle and UI don't lag the
    // frame-driven recorder; the file starts on the next frame.
    isRecording = true
    // If the glasses mic isn't available, flip the mic off so the button reflects that this
    // recording is video-only (the phone mic is never used). Left off afterward — the user can
    // re-tap the mic to turn it back on.
    videoRecorder.prepareToStart(
      includeAudio: includeAudioInStream,
      onGlassesAudioUnavailable: { [weak self] in
        Task { @MainActor in self?.includeAudioInStream = false }
      },
      onFirstFrameWritten: { [store] time, pts, sourcePTS, width, height in
        store.noteFirstVideoFrame(phoneTimeNs: time, ptsSeconds: pts, sourcePTSSeconds: sourcePTS, width: width, height: height)
      },
      onAudioWritten: { [store] in store.noteHFPAudioWritten() }
    )
  }

  func stopVideoRecording() async {
    guard isRecording, !isFinalizingRecording else { return }
    isFinalizingRecording = true
    let generation = recordingGeneration
    let store = recordingStore
    let destination = recordingDestination
    recordingDestination = nil
    recordingErrorTask?.cancel()
    recordingErrorTask = nil
    let endedAt = Date()
    let endedPhoneTimeNs = MonotonicClock.nowNanoseconds()
    recordingStore = nil
    sensorManager.onMotion = nil
    sensorManager.onInput = nil
    sensorManager.onDeviceState = nil
    beginRecordingBackgroundTask()
    isRecording = false
    recordingStartDate = nil
    videoRecorder.prepareToStop()
    // A stop before the first accepted keyframe creates a metadata/log bundle
    // without video and reports that no video was captured.
    // Show a preview/share sheet (mirrors the photo flow) for the finalized file.
    let result = await videoRecorder.stopRecording()
    let sourceURL: URL?
    var nominalFrameRate: Float?
    switch result {
    case .completed(let url):
      sourceURL = url
      let asset = AVURLAsset(url: url)
      if let track = (try? await asset.loadTracks(withMediaType: .video))?.first,
        let rate = try? await track.load(.nominalFrameRate), rate.isFinite, rate > 0 {
        nominalFrameRate = rate
      }
    case .noRecording, .failed: sourceURL = nil
    }
    var archivedURL: URL?
    var archiveError: Error?
    do {
      archivedURL = try await store?.finish(
        videoSourceURL: sourceURL, endedAt: endedAt, endedPhoneTimeNs: endedPhoneTimeNs,
        videoNominalFrameRate: nominalFrameRate
      )
    } catch {
      archiveError = error
    }
    // Move a recording for a picked folder into place, even when archiving reported an
    // error: whatever was written must not stay in the temporary folder.
    var deliveryFailure: String?
    if let store {
      let name = store.directoryURL.lastPathComponent
      if let destination {
        let delivery = await Self.deliver(store.directoryURL, to: destination)
        archivedURL = archivedURL.map { delivery.directory.appendingPathComponent($0.lastPathComponent) }
        if let failure = delivery.failure {
          deliveryFailure = "\(failure) \(Self.keptMessage(delivery.directory))"
          lastSaveMessage = "Couldn't save \(name) to \(destination.lastPathComponent): \(deliveryFailure ?? failure)"
          lastSaveFailed = true
        } else {
          previewAccess = SecurityScopedAccess(destination)
          lastSaveMessage = "Saved \(name) to \(destination.lastPathComponent)."
          lastSaveFailed = false
        }
      } else {
        lastSaveMessage = "Saved \(name) to Recordings."
        lastSaveFailed = false
      }
      if let archiveError, !lastSaveFailed {
        lastSaveMessage = "\(name) was saved, but writing it failed: \(archiveError.localizedDescription)"
        lastSaveFailed = true
      }
      refreshRecordingFolder()
    }
    if case .completed = result, let archivedURL { activePreview = .video(archivedURL) }
    if let archiveError {
      showError("Recording archive failed: \(archiveError.localizedDescription)")
    }
    switch result {
    case .completed: break
    case .noRecording:
      showError("No video frame was saved. The sensor logs were saved.")
    case .failed:
      showError("Recording couldn't be saved. Try again.")
    }
    if let deliveryFailure, let destination {
      showError("Couldn't save the recording to “\(destination.lastPathComponent)”: \(deliveryFailure)")
    }
    if recordingGeneration == generation { isFinalizingRecording = false }
    endRecordingBackgroundTask()
  }

  /// Where a recording that missed its folder ended up, in words.
  private static func keptMessage(_ directory: URL) -> String {
    let staging = (try? RecordingLocation.stagingFolder())?.standardizedFileURL.path
    if let staging, directory.standardizedFileURL.path.hasPrefix(staging) {
      return "It is still in the app's temporary folder and moves to Recordings the next time the app starts."
    }
    return "It was kept in \(RecordingLocation.appFolderLocation) › Recordings."
  }

  /// Moves a finished recording into a picked folder (falling back to the app's folder).
  nonisolated private static func deliver(_ directory: URL, to destination: URL) async -> RecordingDelivery {
    await Task.detached(priority: .userInitiated) {
      RecordingLocation.deliver(directory, to: destination, fallback: try? RecordingLocation.appFolder())
    }.value
  }

  // MARK: - Dismissers

  func dismissError() {
    showError = false
  }

  func dismissCapturePreview() {
    activePreview = nil
    previewAccess = nil
  }

  // MARK: - Device monitoring

  /// Monitors device availability only — does NOT create sessions. Session
  /// creation is deferred to `startSession()` to avoid races.
  private func startDeviceMonitoring() {
    deviceMonitorTask = Task { [weak self] in
      guard let selector = self?.deviceSelector else { return }
      for await deviceId in selector.activeDeviceStream() {
        guard let self else { return }
        if deviceId == nil, hasActiveDevice, isRecording {
          await stopVideoRecording()
        }
        hasActiveDevice = deviceId != nil
      }
    }
  }

  private func startAppLifecycleMonitoring() {
    appLifecycleTask = Task { [weak self] in
      for await _ in NotificationCenter.default.notifications(named: UIApplication.didEnterBackgroundNotification) {
        self?.handleDidEnterBackground()
      }
    }
  }

  private func handleDidEnterBackground() {
    guard hasSession, hasStream, sessionState != .stopping else { return }

    beginBackgroundStopErrorSuppression()
    endSession()
  }

  private func beginRecordingBackgroundTask() {
    guard backgroundTask == .invalid else { return }
    backgroundTask = UIApplication.shared.beginBackgroundTask(withName: "Finalize glasses recording") { [weak self] in
      Task { @MainActor in self?.endRecordingBackgroundTask() }
    }
  }

  private func endRecordingBackgroundTask() {
    guard backgroundTask != .invalid else { return }
    UIApplication.shared.endBackgroundTask(backgroundTask)
    backgroundTask = .invalid
  }

  private func beginBackgroundStopErrorSuppression() {
    isStoppingForBackground = true
    backgroundStopSuppressionTask?.cancel()
    let timeout = backgroundStopErrorSuppressionTimeout
    backgroundStopSuppressionTask = Task { @MainActor [weak self] in
      try? await Task.sleep(for: timeout)
      guard let self, !Task.isCancelled else { return }
      // Bound the suppression window so a stalled teardown cannot hide
      // unrelated later errors indefinitely.
      self.isStoppingForBackground = false
      self.backgroundStopSuppressionTask = nil
    }
  }

  private func endBackgroundStopErrorSuppression() {
    backgroundStopSuppressionTask?.cancel()
    backgroundStopSuppressionTask = nil
    isStoppingForBackground = false
  }

  // MARK: - Session observation

  private func observeSession(_ session: DeviceSession) {
    let generation = sessionGeneration
    session.statePublisher.listen { [weak self] state in
      Task { @MainActor in
        guard let self, self.sessionGeneration == generation else { return }
        self.handleSessionStateChange(state)
      }
    }.store(in: sessionTokenBag)
    session.errorPublisher.listen { [weak self] error in
      Task { @MainActor in
        guard let self, self.sessionGeneration == generation else { return }
        self.handleSessionError(error)
      }
    }.store(in: sessionTokenBag)
  }

  private func handleSessionStateChange(_ state: DeviceSessionState) {
    sessionState = state
    if state == .started, let session = deviceSession {
      let previous = sensorLifecycleTask
      sensorLifecycleTask = Task {
        await previous?.value
        guard deviceSession === session, sessionState == .started, session.state == .started else { return }
        if sensorAttachedSession === session {
          sensorManager.applySamplingRate()
        } else {
          // Wire before attaching so the first device state also reaches the link.
          wireSensorLink()
          await sensorManager.attach(to: session)
          sensorAttachedSession = session
        }
      }
    }
    if state == .stopped {
      endBackgroundStopErrorSuppression()
      let stoppedSession = deviceSession
      sensorAttachedSession = nil
      let previous = sensorLifecycleTask
      sensorLifecycleTask = Task {
        await previous?.value
        if isRecording { await stopVideoRecording() }
        await sensorManager.detach()
        if deviceSession === stoppedSession { cleanupSession() }
      }
    }
  }

  private func handleSessionError(_ error: DeviceSessionError) {
    // Backgrounding the app intentionally ends the preview session; the SDK can
    // still emit teardown errors while that stop converges, and surfacing those
    // would show a false "something went wrong" alert on return.
    guard !isStoppingForBackground else { return }
    // All session errors surface through the single alert, including
    // `datAppOnTheGlassesUpdateRequired`, which the SDK delivers as a one-shot event.
    showError(error.localizedDescription)
  }

  /// Terminal session cleanup — drop the reference and observers so the next
  /// `startSession()` creates a fresh session.
  private func cleanupSession() {
    sessionGeneration = UUID()
    sessionTokenBag.clear()
    deviceSession = nil
  }


  // MARK: - Stream observation

  private func setupStreamListeners(for stream: MWDATCamera.Stream) {
    stream.statePublisher.listen { [weak self] state in
      Task { @MainActor in self?.handleStreamStateChange(state) }
    }.store(in: streamTokenBag)

    stream.videoFramePublisher.listen { [weak self] frame in
      let phoneTimeNs = MonotonicClock.nowNanoseconds()
      guard let self else { return }
      self.frameCounter.increment()

      // Append to the recorder (no-op unless recording) off the main actor so it
      // keeps writing while the app is backgrounded.
      self.appendVideoFrame(frame)

      // Hand the compressed frame to the computer link (returns at once when no Mac is connected).
      self.link.submitVideo(frame.sampleBuffer, phoneTimeNs: phoneTimeNs)

      // Decode the compressed hvc1 frame for preview off the main actor.
      let previewImage = self.videoFrameDecoder.decode(frame.sampleBuffer)

      Task { @MainActor [weak self] in
        guard let self else { return }
        // Keep the live surface stable while a capture preview is onscreen.
        if UIApplication.shared.applicationState != .background,
          activePreview == nil,
          !isCapturingPhoto,
          let image = previewImage
        {
          self.currentVideoFrame = image
          if !self.hasReceivedFirstFrame {
            self.hasReceivedFirstFrame = true
            self.firstFrameSize = image.size
          }
        }
        // Capture the actual recording start (first written frame) once, so the timer
        // anchors to the clip's start and matches its duration.
        if self.isRecording, self.recordingStartDate == nil {
          self.recordingStartDate = self.videoRecorder.recordingStartDate
        }
      }
    }.store(in: streamTokenBag)

    stream.errorPublisher.listen { [weak self] error in
      Task { @MainActor in self?.handleStreamError(error) }
    }.store(in: streamTokenBag)

    stream.photoDataPublisher.listen { [weak self] data in
      Task { @MainActor in self?.handlePhotoData(data) }
    }.store(in: streamTokenBag)
  }

  private func handleStreamStateChange(_ state: StreamState) {
    streamState = state
    if state == .stopped {
      // Terminal — finalize any recording, then converge teardown. Covers both
      // user-initiated stop and SDK auto-stop (disconnect, error, session end).
      Task { await stopVideoRecording() }
      clearStreamResources()
    }
  }

  /// Single stream-teardown convergence point.
  private func clearStreamResources() {
    streamTokenBag.clear()
    // Detach the camera (idempotent) so a subsequent addCamera() can register a new one. On the
    // user-stop and session-end paths the camera is already stopped, so this is a no-op; it is
    // load-bearing only when the stream terminates on its own (a stream-level error while the
    // session stays connected), since teardown cascades parent->child but not child->parent.
    camera?.stop()
    camera = nil
    streamState = .stopped
    currentVideoFrame = nil
    hasReceivedFirstFrame = false
    firstFrameSize = nil
    stopFpsMeasurement()
  }

  // MARK: - Frame rate

  /// Publishes the delivered frame rate once a second until the stream is torn down.
  /// The rate is computed from the real elapsed time, so a late wake-up doesn't skew it.
  private func startFpsMeasurement() {
    stopFpsMeasurement()
    _ = frameCounter.drain()
    measuredStreamFps = 0
    let counter = frameCounter
    fpsTask = Task { [weak self] in
      let clock = ContinuousClock()
      var last = clock.now
      while !Task.isCancelled {
        try? await Task.sleep(for: .seconds(1))
        guard !Task.isCancelled, let self else { return }
        let now = clock.now
        let elapsed = last.duration(to: now)
        last = now
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        guard seconds > 0 else { continue }
        self.measuredStreamFps = Double(counter.drain()) / seconds
      }
    }
  }

  private func stopFpsMeasurement() {
    fpsTask?.cancel()
    fpsTask = nil
    measuredStreamFps = nil
  }

  /// Appends a frame to the recorder. `nonisolated` so it keeps writing while the
  /// app is backgrounded (the recorder is thread-safe).
  nonisolated private func appendVideoFrame(_ frame: VideoFrame) {
    videoRecorder.appendVideoFrame(frame.sampleBuffer)
  }

  private func handleStreamError(_ error: StreamError) {
    // Same bounded suppression window as `handleSessionError`: hide only the
    // expected teardown noise from an intentional background-driven stop.
    guard !isStoppingForBackground else { return }
    showError(error.localizedDescription)
  }

  private func handlePhotoData(_ data: PhotoData) {
    isCapturingPhoto = false
    if let image = UIImage(data: data.data) {
      activePreview = .photo(image)
    }
  }

  private func showError(_ message: String) {
    errorMessage = message
    showError = true
  }
}

/// A frame count shared between the SDK's frame thread (increments) and the main actor
/// (drains once a second). A lock keeps the per-frame cost to a few nanoseconds.
private final class FrameCounter: @unchecked Sendable {
  private let lock = NSLock()
  private var count = 0

  func increment() {
    lock.withLock { count += 1 }
  }

  /// Returns the frames counted since the last drain and resets the count.
  func drain() -> Int {
    lock.withLock {
      defer { count = 0 }
      return count
    }
  }
}
