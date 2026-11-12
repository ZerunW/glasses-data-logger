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

  var pairingStatus: String?
  var showPairingCode = false
  var isPairing = false
  var pairingCodeInput = ""
  @ObservationIgnored private let pairingHost = PairingHost()

  func beginPairing() {
    guard !isBroadcasting, !isPairing else { return }
    isPairing = true
    pairingStatus = "Waiting for a Mac…"
    pairingHost.start { [weak self] event in
      Task { @MainActor in
        guard let self, self.isPairing else { return }
        switch event {
        case .ready: self.pairingStatus = "Select this iPhone in GlassesData on your Mac."
        case .request:
          self.pairingCodeInput = ""
          self.showPairingCode = true
          self.pairingStatus = "Enter the code shown on your Mac."
        case .paired(let key):
          do {
            try PairingTrustStore.save(key)
            self.pairingStatus = "Paired. Start encrypted Wi-Fi broadcasting."
          } catch { self.pairingStatus = "Could not save pairing securely." }
          self.isPairing = false
          self.showPairingCode = false
        case .failed(let message):
          self.pairingStatus = message
          self.isPairing = false
          self.showPairingCode = false
        }
      }
    }
  }

  func confirmPairingCode() {
    showPairingCode = false
    pairingHost.confirm(pairingCodeInput)
    pairingCodeInput = ""
  }

  func cancelPairing() {
    pairingHost.stop()
    isPairing = false
    showPairingCode = false
    pairingCodeInput = ""
    pairingStatus = "Pairing cancelled."
  }

  func forgetPairing() {
    do { try PairingTrustStore.remove(); pairingStatus = "Pairing removed." }
    catch { pairingStatus = "Could not remove pairing." }
  }

  var broadcastStatus: String?
  var isBroadcasting = false
  var broadcastWiFiEnabled = false
  var broadcastPairingKey: String?
  var showBroadcastApproval = false
  var broadcastPeer = ""
  private var broadcastApprovalID: UUID?
  @ObservationIgnored private let broadcast = CameraBroadcast()

  func toggleBroadcast() {
    if isBroadcasting {
      stopBroadcast()
    } else {
      guard isStreaming else { return }
      let key: Data?
      do {
        if broadcastWiFiEnabled { key = try PairingTrustStore.load() ?? SecureTransport.makeKey() }
        else { key = nil }
      }
      catch { broadcastStatus = "Could not generate a secure pairing key."; return }
      broadcastPairingKey = (try? PairingTrustStore.load()) == nil ? key.map(SecureTransport.hex) : nil
      isBroadcasting = true
      broadcastStatus = "Starting…"
      broadcast.start(allowWiFi: broadcastWiFiEnabled, key: key, onStatus: { [weak self] status in
        Task { @MainActor in
          guard let self, self.isBroadcasting else { return }
          self.broadcastStatus = status
          if status.hasPrefix("Broadcast failed:") {
            self.isBroadcasting = false
            self.broadcastWiFiEnabled = false
            self.broadcastPairingKey = nil
          }
          if status.hasPrefix("Connection approval timed out.") || status.hasPrefix("Disconnected.") {
            self.showBroadcastApproval = false
            self.broadcastApprovalID = nil
          }
        }
      }, onApproval: { [weak self] id, peer in
        Task { @MainActor in
          guard let self, self.isBroadcasting else { return }
          self.broadcastApprovalID = id
          self.broadcastPeer = peer
          self.showBroadcastApproval = true
        }
      })
    }
  }

  private func stopBroadcast() {
    if isPairing { cancelPairing() }
    broadcast.stop()
    showBroadcastApproval = false
    broadcastApprovalID = nil
    isBroadcasting = false
    broadcastWiFiEnabled = false
    broadcastPairingKey = nil
    broadcastStatus = nil
  }

  func resolveBroadcastConnection(allow: Bool) {
    guard let id = broadcastApprovalID else { return }
    broadcastApprovalID = nil
    showBroadcastApproval = false
    broadcast.resolveConnection(id, allow: allow)
  }

  @ObservationIgnored private let videoRecorder = VideoRecorder()
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
    backgroundStopErrorSuppressionTimeout: Duration = .seconds(5)
  ) {
    self.wearables = wearables
    self.deviceSelector = AutoDeviceSelector(wearables: wearables)
    self.backgroundStopErrorSuppressionTimeout = backgroundStopErrorSuppressionTimeout
    startDeviceMonitoring()
    startAppLifecycleMonitoring()
  }

  isolated deinit {
    deviceMonitorTask?.cancel()
    appLifecycleTask?.cancel()
    backgroundStopSuppressionTask?.cancel()
    recordingErrorTask?.cancel()
    broadcast.stop()
    pairingHost.stop()
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
      resolution: StreamingResolution.low,
      frameRate: 24
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

  func toggleRecording() {
    guard !isFinalizingRecording, !isPreparingRecording else { return }
    if isRecording {
      Task { await stopVideoRecording() }
    } else {
      startVideoRecording()
    }
  }

  func startVideoRecording() {
    guard isStreaming, !isRecording, !isPreparingRecording, !isFinalizingRecording else { return }
    isPreparingRecording = true
    let rate = sensorManager.samplingRateHz
    let startedAt = Date()
    let startedPhoneTimeNs = MonotonicClock.nowNanoseconds()
    let session = deviceSession
    Task {
      await prepareVideoRecording(rate: rate, startedAt: startedAt, startedPhoneTimeNs: startedPhoneTimeNs, session: session)
    }
  }

  private func prepareVideoRecording(rate: Int, startedAt: Date, startedPhoneTimeNs: UInt64, session: DeviceSession?) async {
    defer { isPreparingRecording = false }
    let store: RecordingStore
    do {
      store = try await Task.detached(priority: .userInitiated) {
        try RecordingStore(samplingRateHz: rate, startedAt: startedAt, startedPhoneTimeNs: startedPhoneTimeNs)
      }.value
    } catch {
      showError("Couldn't create recording folder: \(error.localizedDescription)")
      return
    }
    guard isStreaming, sessionState == .started, deviceSession === session,
      UIApplication.shared.applicationState != .background else {
      _ = try? await store.finish(videoSourceURL: nil)
      return
    }
    recordingStore = store
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
    do {
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
      let archivedURL = try await store?.finish(
        videoSourceURL: sourceURL, endedAt: endedAt, endedPhoneTimeNs: endedPhoneTimeNs,
        videoNominalFrameRate: nominalFrameRate
      )
      if case .completed = result, let archivedURL { activePreview = .video(archivedURL) }
    } catch {
      showError("Recording archive failed: \(error.localizedDescription)")
    }
    switch result {
    case .completed: break
    case .noRecording:
      showError("No video frame was saved. Sensor logs are in Recordings.")
    case .failed:
      showError("Recording couldn't be saved. Try again.")
    }
    if recordingGeneration == generation { isFinalizingRecording = false }
    endRecordingBackgroundTask()
  }

  // MARK: - Dismissers

  func dismissError() {
    showError = false
  }

  func dismissCapturePreview() {
    activePreview = nil
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
      guard let self else { return }

      // Append to the recorder (no-op unless recording) off the main actor so it
      // keeps writing while the app is backgrounded.
      self.appendVideoFrame(frame)

      // Decode the compressed hvc1 frame for preview off the main actor.
      let previewImage = self.videoFrameDecoder.decode(frame.sampleBuffer)
      self.broadcast.submit(frame.sampleBuffer)

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
      stopBroadcast()
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
