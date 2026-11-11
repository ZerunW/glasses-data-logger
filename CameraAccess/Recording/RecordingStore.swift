import Foundation
import MWDATCore
import MWDATInputs
import MWDATMotion

/// One durable, local recording bundle. Queue order defines append/close order.
final class RecordingStore: @unchecked Sendable {
  struct Info: Codable {
    let startedAt: Date
    let startedPhoneTimeNs: UInt64
    var endedAt: Date?
    var endedPhoneTimeNs: UInt64?
    let samplingRateHz: Int
    let configuredFrameRate: Int
    var videoNominalFrameRate: Float?
    var videoWidth: Int?
    var videoHeight: Int?
    var wroteHFPAudio: Bool = false
    var firstVideoFramePhoneTimeNs: UInt64?
    var firstVideoFramePTSSeconds: Double?
    var firstVideoFrameSourcePTSSeconds: Double?
    var clockNote = "Phone timestamps use mach_continuous_time. Glasses timestamps use the device clock without conversion. First video frame phone time is captured after a successful writer append; PTS correspondence is approximate, not hardware synchronization."
  }

  let directoryURL: URL
  var videoURL: URL { directoryURL.appendingPathComponent("video.mov") }
  private let queue = DispatchQueue(label: "GlassesData.RecordingStore", qos: .utility)
  private var handles: [String: FileHandle] = [:]
  private var buffers: [String: Data] = [:]
  private var closed = false
  private var pendingError: Error?
  private var flushTimer: DispatchSourceTimer?
  private var info: Info
  private static let flushBytes = 32_768
  private static let headers = [
    "motion.csv": "phone_time_ns,glasses_time_ns,ax,ay,az,gx,gy,gz,mx,my,mz,qw,qx,qy,qz,source\n",
    "inputs.csv": "phone_time_ns,glasses_time_ms,source,type,detail\n",
    "device_state.csv": "phone_time_ns,battery,charging,don,hinge,thermal,link\n"
  ]

  init(samplingRateHz: Int, configuredFrameRate: Int = 24, rootURL: URL? = nil, startedAt: Date = Date(), startedPhoneTimeNs: UInt64 = MonotonicClock.nowNanoseconds()) throws {
    let fileManager = FileManager.default
    let root: URL
    if let rootURL {
      root = rootURL
    } else {
      let documents = try fileManager.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
      root = documents.appendingPathComponent("Recordings", isDirectory: true)
    }
    try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.timeZone = .current
    formatter.dateFormat = "yyyy-MM-dd_HHmmss"
    let base = formatter.string(from: startedAt)
    var candidate = root.appendingPathComponent(base, isDirectory: true)
    var suffix = 2
    while fileManager.fileExists(atPath: candidate.path) {
      candidate = root.appendingPathComponent("\(base)_\(suffix)", isDirectory: true)
      suffix += 1
    }
    try fileManager.createDirectory(at: candidate, withIntermediateDirectories: false)
    directoryURL = candidate
    info = Info(startedAt: startedAt, startedPhoneTimeNs: startedPhoneTimeNs, samplingRateHz: samplingRateHz, configuredFrameRate: configuredFrameRate)
    do {
      for (name, header) in Self.headers {
        let url = candidate.appendingPathComponent(name)
        try Data(header.utf8).write(to: url)
        handles[name] = try FileHandle(forWritingTo: url)
        try handles[name]?.seekToEnd()
        buffers[name] = Data()
      }
      try writeInfo()
      let timer = DispatchSource.makeTimerSource(queue: queue)
      timer.schedule(deadline: .now() + 1, repeating: 1)
      timer.setEventHandler { [weak self] in
        guard let self, !self.closed else { return }
        for name in Self.headers.keys {
          self.flush(name)
        }
      }
      flushTimer = timer
      timer.resume()
    } catch {
      for handle in handles.values { try? handle.close() }
      throw error
    }
  }

  func appendMotion(_ sample: MotionSample, phoneTimeNs: UInt64) {
    let values: [String] = [
      String(phoneTimeNs), String(sample.timestampNs),
      number(sample.accelerometer?.x), number(sample.accelerometer?.y), number(sample.accelerometer?.z),
      number(sample.gyroscope?.x), number(sample.gyroscope?.y), number(sample.gyroscope?.z),
      number(sample.magnetometer?.x), number(sample.magnetometer?.y), number(sample.magnetometer?.z),
      number(sample.orientation?.w), number(sample.orientation?.x), number(sample.orientation?.y), number(sample.orientation?.z),
      csv(String(describing: sample.source))
    ]
    append(values.joined(separator: ",") + "\n", to: "motion.csv")
  }

  func appendInput(_ event: InputEvent, phoneTimeNs: UInt64) {
    let type: String
    let detail: String
    switch event {
    case .nav(let direction, _, _): type = "nav"; detail = String(describing: direction)
    case .select: type = "select"; detail = ""
    case .back: type = "back"; detail = ""
    case .button(let button, _, _): type = "button"; detail = String(describing: button)
    case .capture(let press, _, _): type = "capture"; detail = String(describing: press)
    case .drag(let action, let x, let y, let dx, let dy, _, _):
      type = "drag"; detail = "\(action) x=\(x) y=\(y) dx=\(dx) dy=\(dy)"
    @unknown default:
      type = "unknown"; detail = String(describing: event)
    }
    append("\(phoneTimeNs),\(event.timestampMs),\(csv(String(describing: event.source))),\(type),\(csv(detail))\n", to: "inputs.csv")
  }

  func appendDeviceState(_ state: DeviceState, phoneTimeNs: UInt64) {
    let row = [String(phoneTimeNs), state.batteryLevel.map(String.init) ?? "", String(describing: state.chargingState), String(describing: state.donState), String(describing: state.hingeState), String(describing: state.thermalLevel), String(describing: state.linkState)]
    append(row.map(csv).joined(separator: ",") + "\n", to: "device_state.csv")
  }

  func noteFirstVideoFrame(phoneTimeNs: UInt64, ptsSeconds: Double, sourcePTSSeconds: Double, width: Int, height: Int) {
    queue.async {
      guard !self.closed, self.info.firstVideoFramePhoneTimeNs == nil else { return }
      self.info.firstVideoFramePhoneTimeNs = phoneTimeNs
      self.info.firstVideoFramePTSSeconds = ptsSeconds.isFinite ? ptsSeconds : nil
      self.info.firstVideoFrameSourcePTSSeconds = sourcePTSSeconds.isFinite ? sourcePTSSeconds : nil
      self.info.videoWidth = width
      self.info.videoHeight = height
    }
  }

  func noteHFPAudioWritten() {
    queue.async { if !self.closed { self.info.wroteHFPAudio = true } }
  }

  func currentError() async -> Error? {
    await withCheckedContinuation { continuation in
      queue.async { continuation.resume(returning: self.pendingError) }
    }
  }

  /// Drains accepted rows, writes info.json, closes handles, then moves the finalized video.
  func finish(videoSourceURL: URL?, endedAt: Date = Date(), endedPhoneTimeNs: UInt64 = MonotonicClock.nowNanoseconds(), videoNominalFrameRate: Float? = nil) async throws -> URL? {
    try await withCheckedThrowingContinuation { continuation in
      queue.async {
        guard !self.closed else {
          continuation.resume(throwing: StoreError.alreadyClosed)
          return
        }
        self.closed = true
        self.flushTimer?.cancel()
        self.flushTimer = nil
        self.info.endedAt = endedAt
        self.info.endedPhoneTimeNs = endedPhoneTimeNs
        self.info.videoNominalFrameRate = videoNominalFrameRate
        for name in Self.headers.keys { self.flush(name) }
        for handle in self.handles.values {
          do { try handle.close() } catch { self.pendingError = self.pendingError ?? error }
        }
        self.handles.removeAll()
        do { try self.writeInfo() } catch { self.pendingError = self.pendingError ?? error }
        if let videoSourceURL {
          do { try FileManager.default.moveItem(at: videoSourceURL, to: self.videoURL) }
          catch { self.pendingError = self.pendingError ?? error }
        }
        if let error = self.pendingError {
          continuation.resume(throwing: error)
        } else {
          continuation.resume(returning: videoSourceURL == nil ? nil : self.videoURL)
        }
      }
    }
  }

  private func append(_ line: String, to name: String) {
    queue.async {
      guard !self.closed, self.pendingError == nil else { return }
      self.buffers[name, default: Data()].append(contentsOf: line.utf8)
      if self.buffers[name, default: Data()].count >= Self.flushBytes {
        self.flush(name)
      }
    }
  }

  private func flush(_ name: String) {
    guard let handle = handles[name], let data = buffers[name], !data.isEmpty else { return }
    buffers[name] = Data()
    do { try handle.write(contentsOf: data) } catch { pendingError = pendingError ?? error }
  }

  private func writeInfo() throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    encoder.dateEncodingStrategy = .iso8601
    try encoder.encode(info).write(to: directoryURL.appendingPathComponent("info.json"), options: .atomic)
  }

  private func number(_ value: Float?) -> String { value.map(String.init(describing:)) ?? "" }
  private func csv(_ value: String) -> String {
    guard value.contains(where: { $0 == "," || $0 == "\"" || $0 == "\n" }) else { return value }
    return "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\""
  }

  enum StoreError: Error { case alreadyClosed }
}
