import Foundation
import Observation
import MWDATCore
import MWDATInputs
import MWDATMotion


private final class SensorCallbacks: @unchecked Sendable {
  private let lock = NSLock()
  private var motion: (@Sendable (MotionSample, UInt64) -> Void)?
  private var input: (@Sendable (InputEvent, UInt64) -> Void)?
  private var device: (@Sendable (DeviceState, UInt64) -> Void)?
  func setMotion(_ callback: (@Sendable (MotionSample, UInt64) -> Void)?) { lock.lock(); defer { lock.unlock() }; motion = callback }
  func setInput(_ callback: (@Sendable (InputEvent, UInt64) -> Void)?) { lock.lock(); defer { lock.unlock() }; input = callback }
  func setDevice(_ callback: (@Sendable (DeviceState, UInt64) -> Void)?) { lock.lock(); defer { lock.unlock() }; device = callback }
  func emitMotion(_ sample: MotionSample, _ time: UInt64) { lock.lock(); let callback = motion; lock.unlock(); callback?(sample, time) }
  func emitInput(_ event: InputEvent, _ time: UInt64) { lock.lock(); let callback = input; lock.unlock(); callback?(event, time) }
  func emitDevice(_ state: DeviceState, _ time: UInt64) { lock.lock(); let callback = device; lock.unlock(); callback?(state, time) }
}

@MainActor @Observable
final class SensorManager {
  var samplingRateHz = 30 {
    didSet {
      if samplingRateHz != oldValue { applySamplingRate() }
    }
  }
  var latestDeviceState: DeviceState?
  var motionState = "Disconnected"
  var inputsState = "Disconnected"
  var motionError: String?
  var inputsError: String?
  var latestSample: MotionSample?
  var acceleration: [SensorPoint] = []
  var gyroscope: [SensorPoint] = []
  var magnetometer: [SensorPoint] = []
  var inputs: [SensorInput] = []
  var samplesPerSecond = 0
  var onMotion: (@Sendable (MWDATMotion.MotionSample, UInt64) -> Void)? { didSet { callbacks.setMotion(onMotion) } }
  var onInput: (@Sendable (MWDATInputs.InputEvent, UInt64) -> Void)? { didSet { callbacks.setInput(onInput) } }
  var onDeviceState: (@Sendable (MWDATCore.DeviceState, UInt64) -> Void)? { didSet { callbacks.setDevice(onDeviceState) } }
  /// A second, independent set of receivers for the computer link. Called at the same
  /// points as the recording callbacks above, with the same phone timestamp.
  var onLinkMotion: (@Sendable (MWDATMotion.MotionSample, UInt64) -> Void)? { didSet { linkCallbacks.setMotion(onLinkMotion) } }
  var onLinkInput: (@Sendable (MWDATInputs.InputEvent, UInt64) -> Void)? { didSet { linkCallbacks.setInput(onLinkInput) } }
  var onLinkDeviceState: (@Sendable (MWDATCore.DeviceState, UInt64) -> Void)? { didSet { linkCallbacks.setDevice(onLinkDeviceState) } }

  var windowNowNs: UInt64 = MonotonicClock.nowNanoseconds()
  @ObservationIgnored private let callbacks = SensorCallbacks()
  @ObservationIgnored private let linkCallbacks = SensorCallbacks()
  @ObservationIgnored private var timerTask: Task<Void, Never>?
  @ObservationIgnored private var motionGeneration = UUID()
  @ObservationIgnored private var session: DeviceSession?
  @ObservationIgnored private var motion: Motion?
  @ObservationIgnored private var inputsCapability: Inputs?
  @ObservationIgnored private var rateTask: Task<Void, Never>?
  @ObservationIgnored private var motionTask: Task<Void, Never>?
  @ObservationIgnored private var inputsTask: Task<Void, Never>?
  @ObservationIgnored private var motionTokens: [any AnyListenerToken] = []
  @ObservationIgnored private var inputsTokens: [any AnyListenerToken] = []
  @ObservationIgnored private var deviceToken: (any AnyListenerToken)?
  @ObservationIgnored private var recentSampleTimes: [UInt64] = []
  @ObservationIgnored private var lastDisplayUpdateNs: UInt64 = 0
  @ObservationIgnored private var generation = UUID()

  func attach(to session: DeviceSession) async {
    await detach()
    self.session = session
    inputsError = nil; motionError = nil; latestSample = nil; latestDeviceState = nil
    acceleration.removeAll(); gyroscope.removeAll(); magnetometer.removeAll(); inputs.removeAll()
    recentSampleTimes.removeAll(); samplesPerSecond = 0
    lastDisplayUpdateNs = 0
    let current = generation
    let callbacks = self.callbacks
    let linkCallbacks = self.linkCallbacks
    timerTask = Task { [weak self] in
      while !Task.isCancelled {
        try? await Task.sleep(for: .seconds(1))
        guard !Task.isCancelled, let self, self.generation == current else { break }
        self.refreshWindow(MonotonicClock.nowNanoseconds())
      }
    }
    if let device = session.device {
      deviceToken = device.addDeviceStateListener { [weak self] state in
        let time = MonotonicClock.nowNanoseconds()
        callbacks.emitDevice(state, time)
        linkCallbacks.emitDevice(state, time)
        Task { @MainActor [weak self] in
          guard let self, self.generation == current else { return }
          self.latestDeviceState = state
        }
      }
    }
    startMotion()
    do {
      guard let capability = try session.addInputs(configuration: InputsConfiguration(consumeBack: false)) else {
        inputsError = "Unable to start Inputs"; return
      }
      inputsCapability = capability
      inputsState = capability.state.description
      inputsTokens = [
        capability.statePublisher.listen { [weak self] state in
          Task { @MainActor [weak self] in guard let self, self.generation == current else { return }; self.inputsState = state.description }
        },
        capability.errorPublisher.listen { [weak self] error in
          Task { @MainActor [weak self] in
            guard let self, self.generation == current else { return }
            self.inputsError = error == .permissionDenied ? "This experimental feature is not authorized" : error.description
          }
        }
      ]
      let events = capability.events
      inputsTask = Task.detached { [weak self] in
        for await event in events {
          guard !Task.isCancelled else { break }
          let time = MonotonicClock.nowNanoseconds()
          callbacks.emitInput(event, time)
          linkCallbacks.emitInput(event, time)
          Task { @MainActor [weak self] in
            guard let self, self.generation == current else { return }
            self.inputs.insert(SensorInput(event, time: Date()), at: 0)
            if self.inputs.count > 20 { self.inputs.removeLast(self.inputs.count - 20) }
          }
        }
      }
    } catch { inputsError = error.localizedDescription }
  }

  private var rate: MotionSamplingRate {
    switch samplingRateHz {
    case 5: .hz5
    case 10: .hz10
    case 15: .hz15
    case 24: .hz24
    case 60: .hz60
    default: .hz30
    }
  }

  func applySamplingRate() {
    guard session?.state == .started else { return }
    let current = generation
    let previous = rateTask
    rateTask = Task { [weak self] in
      await previous?.value
      guard let self, generation == current, session?.state == .started,
        motion?.configuration.samplingRate != rate else { return }
      await restartMotion()
    }
  }

  private func startMotion() {
    guard let session, session.state == .started else { return }
    let current = generation
    let motionCurrent = motionGeneration
    let callbacks = self.callbacks
    let linkCallbacks = self.linkCallbacks
    do {
      guard let capability = try session.addMotion(configuration: MotionConfiguration(samplingRate: rate)) else {
        motionError = "Unable to start Motion"; return
      }
      motion = capability
      motionError = nil
      motionState = capability.state.description
      motionTokens = [
        capability.statePublisher.listen { [weak self] state in
          Task { @MainActor [weak self] in guard let self, self.generation == current, self.motionGeneration == motionCurrent else { return }; self.motionState = state.description }
        },
        capability.errorPublisher.listen { [weak self] error in
          Task { @MainActor [weak self] in guard let self, self.generation == current, self.motionGeneration == motionCurrent else { return }; self.motionError = error.description }
        }
      ]
      let samples = capability.samples
      motionTask = Task.detached { [weak self] in
        for await sample in samples {
          guard !Task.isCancelled else { break }
          let time = MonotonicClock.nowNanoseconds()
          callbacks.emitMotion(sample, time)
          linkCallbacks.emitMotion(sample, time)
          Task { @MainActor [weak self] in
            guard let self, self.generation == current, self.motionGeneration == motionCurrent else { return }
            self.accept(sample, time: time)
          }
        }
      }
      capability.start()
    } catch { motionError = error.localizedDescription }
  }
  private func accept(_ sample: MotionSample, time: UInt64) {
    recentSampleTimes.append(time)
    // Recording receives every sample on the consumer task. Limit chart and
    // observable updates to 10 Hz so high-rate streams do not redraw the entire panel.
    guard time >= lastDisplayUpdateNs,
      time - lastDisplayUpdateNs >= 100_000_000 || lastDisplayUpdateNs == 0 else { return }
    lastDisplayUpdateNs = time
    latestSample = sample
    func append(_ vector: MWDATMotion.Vector3?, to points: inout [SensorPoint]) {
      if let vector { points.append(SensorPoint(timeNs: time, x: Double(vector.x), y: Double(vector.y), z: Double(vector.z))) }
    }
    append(sample.accelerometer, to: &acceleration)
    append(sample.gyroscope, to: &gyroscope)
    append(sample.magnetometer, to: &magnetometer)
    refreshWindow(MonotonicClock.nowNanoseconds())
  }
  private func refreshWindow(_ now: UInt64) {
    windowNowNs = now
    let oneSecondAgo = now > 1_000_000_000 ? now - 1_000_000_000 : 0
    let tenSecondsAgo = now > 10_000_000_000 ? now - 10_000_000_000 : 0
    recentSampleTimes.removeAll { $0 < oneSecondAgo }
    samplesPerSecond = recentSampleTimes.count
    acceleration.removeAll { $0.timeNs < tenSecondsAgo }
    gyroscope.removeAll { $0.timeNs < tenSecondsAgo }
    magnetometer.removeAll { $0.timeNs < tenSecondsAgo }
  }
  @discardableResult private func stopMotion() async -> Bool {
    motionGeneration = UUID()
    let task = motionTask
    motionTask = nil
    task?.cancel()
    await task?.value
    for token in motionTokens { await token.cancel() }; motionTokens.removeAll()
    if motion != nil {
      do { try session?.removeMotion() } catch { motionError = "Failed to remove Motion: \(error.localizedDescription)"; return false }
    }
    motion = nil
    return true
  }
  private func restartMotion() async { if await stopMotion() { startMotion() } }
  func detach() async {
    generation = UUID()
    timerTask?.cancel(); timerTask = nil
    let task = inputsTask
    inputsTask = nil
    task?.cancel()
    await rateTask?.value
    rateTask = nil
    await stopMotion()
    motion = nil
    await task?.value
    for token in inputsTokens { await token.cancel() }; inputsTokens.removeAll()
    if inputsCapability != nil {
      do { try session?.removeInputs() } catch { inputsError = "Failed to remove Inputs: \(error.localizedDescription)" }
    }
    inputsCapability = nil
    await deviceToken?.cancel(); deviceToken = nil
    session = nil
    latestSample = nil
    recentSampleTimes.removeAll()
    lastDisplayUpdateNs = 0
    samplesPerSecond = 0
    acceleration.removeAll()
    gyroscope.removeAll()
    magnetometer.removeAll()
    motionState = "Disconnected"; inputsState = "Disconnected"
  }
}
