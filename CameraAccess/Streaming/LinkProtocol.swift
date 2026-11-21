import Foundation

/// GDL1 message types. Must match `Receiver/glassesdata_receiver/protocol.py`.
enum LinkMessageType: UInt8, Sendable {
  case hello = 0x01
  case helloAck = 0x02
  case videoFormat = 0x10
  case video = 0x11
  case motion = 0x20
  case input = 0x21
  case deviceState = 0x22
  case stats = 0x30
  case ping = 0x40
  case pong = 0x41
  /// Phone to computer: start or stop the computer's recording.
  case control = 0x50
  /// Computer to phone: the computer's recorder, after every change and once a second while recording.
  case recorderState = 0x51

  /// Messages whose payload is a UTF-8 JSON object.
  var isJSON: Bool {
    switch self {
    case .hello, .helloAck, .motion, .input, .deviceState, .stats, .control, .recorderState: return true
    case .videoFormat, .video, .ping, .pong: return false
    }
  }
}

/// A CONTROL command. The receiver ignores commands it doesn't know.
enum LinkControlCommand: String, Sendable {
  case recordStart = "record_start"
  case recordStop = "record_stop"
}

/// The computer's recorder, from RECORDER_STATE. Every message is the complete state.
struct LinkRecorderState: Equatable, Sendable {
  var recording: Bool
  /// The folder of the current recording, or of the last one after a stop.
  var name: String?
  /// Length and frames written of that recording (0 before the first).
  var seconds: Double
  var frames: Int
  /// Why a recording rotated, failed or a command was refused, otherwise nil.
  var note: String?
}

/// One complete frame read from the wire. `type` stays raw so unknown types can be skipped.
struct LinkFrame: Equatable, Sendable {
  let type: UInt8
  let payload: Data

  var messageType: LinkMessageType? { LinkMessageType(rawValue: type) }
}

/// GDL1 framing and message codecs: `u32 BE length | u8 type | payload`, where the
/// length counts the type byte plus the payload. Pure functions; Foundation only.
enum LinkProtocol {
  static let version = 1
  static let defaultPort: UInt16 = 7900
  static let maxFrameLength = 4 * 1024 * 1024
  static let bonjourType = "_glassesdata._tcp"
  static let keyframeFlag: UInt8 = 0x01
  /// `flags | pts_us | phone_time_ns` ahead of the sample bytes in VIDEO.
  static let videoHeaderLength = 17

  enum EncodingError: Error, Equatable {
    case frameTooLarge(Int)
    case notJSONType(LinkMessageType)
    case invalidJSONObject
  }

  // MARK: - Encoders

  static func frame(_ type: LinkMessageType, _ payload: Data) throws(EncodingError) -> Data {
    let length = payload.count + 1
    guard length <= maxFrameLength else { throw .frameTooLarge(length) }
    var data = Data(capacity: 4 + length)
    appendBigEndian(UInt32(length), to: &data)
    data.append(type.rawValue)
    data.append(payload)
    return data
  }

  /// Sorted keys, no whitespace, unescaped slashes. Rejects NaN and infinity, which
  /// `JSONSerialization` would otherwise abort on.
  static func json(_ type: LinkMessageType, _ body: [String: Any]) throws(EncodingError) -> Data {
    guard type.isJSON else { throw .notJSONType(type) }
    guard JSONSerialization.isValidJSONObject(body),
      let payload = try? JSONSerialization.data(withJSONObject: body, options: [.sortedKeys, .withoutEscapingSlashes])
    else { throw .invalidJSONObject }
    return try frame(type, payload)
  }

  static func videoFormat(width: UInt16, height: UInt16, fps: UInt16, hvcC: Data) throws(EncodingError) -> Data {
    var payload = Data(capacity: 6 + hvcC.count)
    appendBigEndian(width, to: &payload)
    appendBigEndian(height, to: &payload)
    appendBigEndian(fps, to: &payload)
    payload.append(hvcC)
    return try frame(.videoFormat, payload)
  }

  /// `sample` holds 4-byte length-prefixed NAL units, as in an MP4 sample.
  static func video(keyframe: Bool, ptsMicroseconds: Int64, phoneTimeNs: UInt64, sample: Data) throws(EncodingError) -> Data {
    var payload = Data(capacity: videoHeaderLength + sample.count)
    payload.append(keyframe ? keyframeFlag : 0)
    appendBigEndian(UInt64(bitPattern: ptsMicroseconds), to: &payload)
    appendBigEndian(phoneTimeNs, to: &payload)
    payload.append(sample)
    return try frame(.video, payload)
  }

  static func pong(macTimeNs: UInt64, phoneTimeNs: UInt64) -> Data {
    var data = Data(capacity: 21)
    appendBigEndian(UInt32(17), to: &data)
    data.append(LinkMessageType.pong.rawValue)
    appendBigEndian(macTimeNs, to: &data)
    appendBigEndian(phoneTimeNs, to: &data)
    return data
  }

  static func helloBody(app: String, device: String, name: String, token: String?) -> [String: Any] {
    var body: [String: Any] = ["protocol": version, "app": app, "device": device, "name": name]
    if let token { body["token"] = token }
    return body
  }

  /// `queue_frames`, `record_priority` and `gaps` were added after the first receiver;
  /// receivers ignore keys they don't know.
  static func statsBody(
    phoneTimeNs: UInt64, glassesFrames: UInt64, sentFrames: UInt64, droppedFrames: UInt64,
    motionSamples: UInt64, queueBytes: Int, queueFrames: Int = 0, recordPriority: Bool = false, gaps: UInt64 = 0
  ) -> [String: Any] {
    [
      "t": phoneTimeNs, "glasses_frames": glassesFrames, "sent_frames": sentFrames,
      "dropped_frames": droppedFrames, "motion_samples": motionSamples, "queue_bytes": queueBytes,
      "queue_frames": queueFrames, "record_priority": recordPriority, "gaps": gaps,
    ]
  }

  static func controlBody(_ command: LinkControlCommand, phoneTimeNs: UInt64) -> [String: Any] {
    ["cmd": command.rawValue, "t": phoneTimeNs]
  }

  // MARK: - Decoders

  struct HelloAck: Equatable, Sendable {
    let ok: Bool
    let error: String?
    /// The receiver's machine name, when it sends one.
    let name: String?
    let receiver: String?
    let code: String?
  }

  static func decodeHelloAck(_ payload: Data) -> HelloAck? {
    guard let object = try? JSONSerialization.jsonObject(with: payload),
      let body = object as? [String: Any],
      let ok = body["ok"] as? Bool
    else { return nil }
    return HelloAck(
      ok: ok, error: body["error"] as? String, name: body["name"] as? String,
      receiver: body["receiver"] as? String, code: body["code"] as? String)
  }

  /// Nil unless `recording` is a JSON boolean and `seconds` and `frames` are numbers;
  /// `name` and `note` may be null or missing.
  static func decodeRecorderState(_ payload: Data) -> LinkRecorderState? {
    guard let object = try? JSONSerialization.jsonObject(with: payload),
      let body = object as? [String: Any],
      let recording = jsonBool(body["recording"]),
      let seconds = jsonNumber(body["seconds"])?.doubleValue, seconds.isFinite,
      let frames = jsonNumber(body["frames"])?.int64Value
    else { return nil }
    return LinkRecorderState(
      recording: recording, name: body["name"] as? String, seconds: max(0, seconds),
      frames: Int(clamping: max(0, frames)), note: (body["note"] as? String).flatMap { $0.isEmpty ? nil : $0 })
  }

  /// JSONSerialization turns both `true` and `1` into an NSNumber; these tell them apart.
  private static func jsonBool(_ value: Any?) -> Bool? {
    guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else { return nil }
    return number.boolValue
  }

  private static func jsonNumber(_ value: Any?) -> NSNumber? {
    guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
    return number
  }

  /// PING carries exactly one `u64` (the Mac's monotonic time).
  static func decodePing(_ payload: Data) -> UInt64? {
    guard payload.count == 8 else { return nil }
    return readBigEndianUInt64(payload, at: payload.startIndex)
  }

  // MARK: - HEVC sample helpers

  /// Re-frames length-prefixed NAL units that use `lengthSize`-byte prefixes (1, 2 or 4)
  /// with 4-byte prefixes. Returns nil when the data does not parse exactly.
  static func fourByteLengthPrefixed(_ data: Data, lengthSize: Int) -> Data? {
    guard [1, 2, 4].contains(lengthSize) else { return nil }
    if lengthSize == 4 { return isWellFormedFourByte(data) ? data : nil }
    var output = Data(capacity: data.count + data.count / 8)
    var index = data.startIndex
    while index < data.endIndex {
      guard data.endIndex - index >= lengthSize else { return nil }
      var length = 0
      for offset in 0..<lengthSize { length = length << 8 | Int(data[index + offset]) }
      index += lengthSize
      guard length > 0, data.endIndex - index >= length else { return nil }
      appendBigEndian(UInt32(length), to: &output)
      output.append(data[index..<(index + length)])
      index += length
    }
    return output.isEmpty ? nil : output
  }

  /// Whether any NAL unit in 4-byte length-prefixed data is an IRAP picture (BLA, IDR or CRA).
  static func containsHEVCKeyframe(_ data: Data) -> Bool {
    var index = data.startIndex
    while data.endIndex - index > 4 {
      let length = Int(readBigEndianUInt32(data, at: index))
      index += 4
      guard length > 0, data.endIndex - index >= length else { return false }
      let nalType = (data[index] >> 1) & 0x3F
      if (16...21).contains(nalType) { return true }
      index += length
    }
    return false
  }

  private static func isWellFormedFourByte(_ data: Data) -> Bool {
    var index = data.startIndex
    guard index < data.endIndex else { return false }
    while index < data.endIndex {
      guard data.endIndex - index >= 4 else { return false }
      let length = Int(readBigEndianUInt32(data, at: index))
      index += 4
      guard length > 0, data.endIndex - index >= length else { return false }
      index += length
    }
    return true
  }

  // MARK: - Byte helpers

  static func appendBigEndian<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
    withUnsafeBytes(of: value.bigEndian) { data.append(contentsOf: $0) }
  }

  static func readBigEndianUInt32(_ data: Data, at index: Data.Index) -> UInt32 {
    data[index..<(index + 4)].reduce(0) { $0 << 8 | UInt32($1) }
  }

  static func readBigEndianUInt64(_ data: Data, at index: Data.Index) -> UInt64 {
    data[index..<(index + 8)].reduce(0) { $0 << 8 | UInt64($1) }
  }
}

/// Incremental GDL1 decoder: feed bytes as they arrive and get complete frames back.
/// Rejects a zero length or one over `LinkProtocol.maxFrameLength` before buffering the
/// payload, so the buffer stays bounded.
struct LinkFrameReader: Sendable {
  enum ReadError: Error, Equatable {
    case invalidLength(UInt32)
  }

  private var buffer = Data()

  var pendingByteCount: Int { buffer.count }

  mutating func feed(_ data: Data) throws(ReadError) -> [LinkFrame] {
    buffer.append(data)
    var frames: [LinkFrame] = []
    var index = buffer.startIndex
    while buffer.endIndex - index >= 4 {
      let length = LinkProtocol.readBigEndianUInt32(buffer, at: index)
      guard length >= 1, length <= UInt32(LinkProtocol.maxFrameLength) else {
        buffer.removeAll()
        throw .invalidLength(length)
      }
      let frameEnd = index + 4 + Int(length)
      guard buffer.endIndex >= frameEnd else { break }
      let type = buffer[index + 4]
      frames.append(LinkFrame(type: type, payload: Data(buffer[(index + 5)..<frameEnd])))
      index = frameEnd
    }
    buffer.removeSubrange(buffer.startIndex..<index)
    return frames
  }
}

/// How much the link may queue before it drops video or gives up on the connection.
struct LinkBufferPolicy: Equatable, Sendable {
  /// Video is dropped while more than this many bytes are waiting to be sent.
  let videoBudgetBytes: Int
  /// Video is dropped while more than this many frames are waiting; nil for no frame limit.
  let maxOutstandingFrames: Int?
  /// The connection is closed as stalled when more than this many bytes are waiting.
  let closeThresholdBytes: Int

  /// Low latency for live viewing: a stall drops video at once, and the picture resumes
  /// at the next keyframe.
  static let lowLatency = LinkBufferPolicy(
    videoBudgetBytes: 2 * 1024 * 1024, maxOutstandingFrames: 12, closeThresholdBytes: 8 * 1024 * 1024)
  /// While the computer records: buffer through short Wi-Fi stalls rather than leave gaps
  /// in the recording. Bounded only by bytes.
  static let recordPriority = LinkBufferPolicy(
    videoBudgetBytes: 64 * 1024 * 1024, maxOutstandingFrames: nil, closeThresholdBytes: 96 * 1024 * 1024)
}

/// Chooses the buffer policy for one connection from the computer's recording state.
/// Switching back to low latency never throws away what is already queued: new frames
/// get the low limits at once, but the connection keeps the record-priority close
/// threshold until the queue has drained below the low one.
struct LinkBackpressure: Equatable, Sendable {
  private(set) var recordPriority = false
  /// Left record priority with more queued than the low-latency close threshold.
  private(set) var draining = false

  /// The limits for admitting a new video frame.
  var admission: LinkBufferPolicy { recordPriority ? .recordPriority : .lowLatency }

  mutating func setRecordPriority(_ enabled: Bool, outstandingBytes: Int) {
    if recordPriority, !enabled {
      draining = outstandingBytes > LinkBufferPolicy.lowLatency.closeThresholdBytes
    } else if enabled {
      draining = false
    }
    recordPriority = enabled
  }

  /// Whether a connection with this much queued has stalled and should be closed.
  mutating func shouldClose(outstandingBytes: Int) -> Bool {
    if draining, outstandingBytes <= LinkBufferPolicy.lowLatency.closeThresholdBytes {
      draining = false
    }
    let limit = recordPriority || draining
      ? LinkBufferPolicy.recordPriority.closeThresholdBytes : LinkBufferPolicy.lowLatency.closeThresholdBytes
    return outstandingBytes > limit
  }
}

/// Video admission for one connection: VIDEO_FORMAT before the first frame and after
/// every hvcC change, a keyframe after any gap, and drops under backpressure.
struct LinkVideoGate: Sendable {
  enum Decision: Equatable, Sendable {
    case drop
    /// Send the frame; when `withFormat` is true, send VIDEO_FORMAT for its hvcC first.
    case send(withFormat: Bool)
  }

  static let maxOutstandingBytes = LinkBufferPolicy.lowLatency.videoBudgetBytes
  static let maxOutstandingFrames = LinkBufferPolicy.lowLatency.maxOutstandingFrames ?? Int.max

  private(set) var needsKeyframe = true
  private(set) var sentFormat: Data?
  /// Times video stopped flowing after it had started: each run of dropped or lost
  /// frames that follows a sent frame counts once.
  private(set) var gaps: UInt64 = 0
  private var lastFrameSent = false

  mutating func admit(
    keyframe: Bool, hvcC: Data, outstandingBytes: Int, outstandingFrames: Int, policy: LinkBufferPolicy = .lowLatency
  ) -> Decision {
    let overFrames = policy.maxOutstandingFrames.map { outstandingFrames > $0 } ?? false
    if outstandingBytes > policy.videoBudgetBytes || overFrames {
      needsKeyframe = true
      return drop()
    }
    let formatChanged = hvcC != sentFormat
    if needsKeyframe || formatChanged {
      guard keyframe else {
        needsKeyframe = true
        return drop()
      }
    }
    needsKeyframe = false
    lastFrameSent = true
    if formatChanged {
      sentFormat = hvcC
      return .send(withFormat: true)
    }
    return .send(withFormat: false)
  }

  /// A frame or its VIDEO_FORMAT was lost after admission: start over at the next keyframe.
  mutating func noteDropped() {
    needsKeyframe = true
    sentFormat = nil
    _ = drop()
  }

  private mutating func drop() -> Decision {
    if lastFrameSent { gaps &+= 1 }
    lastFrameSent = false
    return .drop
  }
}
