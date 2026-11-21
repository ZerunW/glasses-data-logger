import Foundation
import XCTest
@testable import GlassesData

final class LinkProtocolTests: XCTestCase {
  // Byte fixtures shared with Receiver/tests/test_protocol.py. Change both together.
  private static let videoFixture = hex(
    "0000001a"  // length = 1 type + 17 header + 8 data
      + "11"  // VIDEO
      + "01"  // keyframe
      + "fffffffffffffc18"  // pts_us = -1000
      + "000000000000007b"  // phone_time_ns = 123
      + "00000004deadbeef")  // one 4-byte NAL unit
  private static let videoFormatFixture = hex("0000000a" + "10" + "02d0" + "0500" + "0018" + "0102" + "03")
  private static let pongFixture = hex("00000011" + "41" + "0000000000000001" + "0000000000000002")
  private static let statsFixture = hex("00000014") + Data([0x30]) + Data(#"{"dropped":0,"t":5}"#.utf8)
  private static let controlFixture = hex("00000025") + Data([0x50]) + Data(#"{"cmd":"record_start","t":123456789}"#.utf8)
  private static let recorderStateFixture = hex("00000054") + Data([0x51])
    + Data(#"{"frames":42,"name":"2026-11-20_212051","note":null,"recording":true,"seconds":1.5}"#.utf8)

  // MARK: - Fixtures

  func testVideoFixture() throws {
    let encoded = try LinkProtocol.video(
      keyframe: true, ptsMicroseconds: -1000, phoneTimeNs: 123, sample: Self.hex("00000004deadbeef"))
    XCTAssertEqual(encoded, Self.videoFixture)
  }

  func testVideoFormatFixture() throws {
    let encoded = try LinkProtocol.videoFormat(width: 720, height: 1280, fps: 24, hvcC: Self.hex("010203"))
    XCTAssertEqual(encoded, Self.videoFormatFixture)
  }

  func testPongFixture() {
    XCTAssertEqual(LinkProtocol.pong(macTimeNs: 1, phoneTimeNs: 2), Self.pongFixture)
  }

  func testStatsFixture() throws {
    XCTAssertEqual(try LinkProtocol.json(.stats, ["t": 5, "dropped": 0]), Self.statsFixture)
  }

  func testControlFixture() throws {
    let encoded = try LinkProtocol.json(.control, LinkProtocol.controlBody(.recordStart, phoneTimeNs: 123_456_789))
    XCTAssertEqual(encoded, Self.controlFixture)
    let stop = try LinkProtocol.json(.control, LinkProtocol.controlBody(.recordStop, phoneTimeNs: 1))
    XCTAssertEqual(String(decoding: stop.dropFirst(5), as: UTF8.self), #"{"cmd":"record_stop","t":1}"#)
  }

  func testRecorderStateFixture() throws {
    var reader = LinkFrameReader()
    let frames = try reader.feed(Self.recorderStateFixture)
    XCTAssertEqual(frames.map(\.messageType), [.recorderState])
    XCTAssertEqual(
      frames.first.flatMap { LinkProtocol.decodeRecorderState($0.payload) },
      LinkRecorderState(recording: true, name: "2026-11-20_212051", seconds: 1.5, frames: 42, note: nil))
    // The same body encodes to the receiver's bytes (sorted keys, no whitespace).
    let body: [String: Any] = [
      "recording": true, "name": "2026-11-20_212051", "seconds": 1.5, "frames": 42, "note": NSNull(),
    ]
    XCTAssertEqual(try LinkProtocol.json(.recorderState, body), Self.recorderStateFixture)
  }

  func testRecorderStateDecoding() {
    func decode(_ json: String) -> LinkRecorderState? { LinkProtocol.decodeRecorderState(Data(json.utf8)) }
    XCTAssertEqual(
      decode(#"{"frames":0,"name":null,"note":"no video yet","recording":false,"seconds":0}"#),
      LinkRecorderState(recording: false, name: nil, seconds: 0, frames: 0, note: "no video yet"))
    // Missing optional fields, an integer `seconds`, an empty note.
    XCTAssertEqual(
      decode(#"{"frames":7,"recording":true,"seconds":2,"note":""}"#),
      LinkRecorderState(recording: true, name: nil, seconds: 2, frames: 7, note: nil))
    XCTAssertNil(decode(#"{"frames":1,"recording":1,"seconds":0}"#), "recording must be a boolean")
    XCTAssertNil(decode(#"{"frames":true,"recording":true,"seconds":0}"#), "frames must be a number")
    XCTAssertNil(decode(#"{"frames":1,"recording":true,"seconds":"1"}"#))
    XCTAssertNil(decode(#"{"recording":true,"seconds":1}"#))
    XCTAssertNil(decode(#"[true]"#))
    XCTAssertNil(decode("not json"))
  }

  func testStatsKeepTheirKeysAndAddQueueAndGaps() throws {
    let body = LinkProtocol.statsBody(
      phoneTimeNs: 1, glassesFrames: 2, sentFrames: 3, droppedFrames: 4, motionSamples: 5, queueBytes: 6,
      queueFrames: 7, recordPriority: true, gaps: 8)
    let frame = try LinkProtocol.json(.stats, body)
    XCTAssertEqual(
      String(decoding: frame.dropFirst(5), as: UTF8.self),
      #"{"dropped_frames":4,"gaps":8,"glasses_frames":2,"motion_samples":5,"queue_bytes":6,"queue_frames":7,"#
        + #""record_priority":true,"sent_frames":3,"t":1}"#)
  }

  // MARK: - Frame reader

  func testReaderHandlesOneByteAtATimeAcrossFrames() throws {
    let stream = Self.videoFixture + Self.pongFixture + Self.videoFormatFixture
    var reader = LinkFrameReader()
    var frames: [LinkFrame] = []
    for byte in stream {
      frames += try reader.feed(Data([byte]))
    }
    XCTAssertEqual(frames.map(\.messageType), [.video, .pong, .videoFormat])
    XCTAssertEqual(frames[0].payload, Self.videoFixture.dropFirst(5))
    XCTAssertEqual(frames[1].payload, Self.pongFixture.dropFirst(5))
    XCTAssertEqual(frames[2].payload, Self.hex("02d0050000180102" + "03"))
    XCTAssertEqual(reader.pendingByteCount, 0)
  }

  func testReaderReturnsSeveralFramesFromOneChunk() throws {
    var reader = LinkFrameReader()
    let frames = try reader.feed(Self.pongFixture + Self.statsFixture + Self.pongFixture.prefix(3))
    XCTAssertEqual(frames.map(\.messageType), [.pong, .stats])
    XCTAssertEqual(reader.pendingByteCount, 3)
  }

  func testReaderRejectsBadLengths() {
    for length in [UInt32(0), UInt32(LinkProtocol.maxFrameLength + 1), UInt32.max] {
      var reader = LinkFrameReader()
      var data = Data()
      LinkProtocol.appendBigEndian(length, to: &data)
      data.append(0x11)
      XCTAssertThrowsError(try reader.feed(data), "length \(length)") { error in
        XCTAssertEqual(error as? LinkFrameReader.ReadError, .invalidLength(length))
      }
    }
  }

  func testReaderKeepsUnknownTypes() throws {
    var reader = LinkFrameReader()
    let frames = try reader.feed(Self.hex("00000002" + "7f" + "aa"))
    XCTAssertEqual(frames, [LinkFrame(type: 0x7F, payload: Self.hex("aa"))])
    XCTAssertNil(frames[0].messageType)
  }

  func testEncoderRejectsOversizedFrames() {
    XCTAssertThrowsError(try LinkProtocol.frame(.video, Data(count: LinkProtocol.maxFrameLength)))
    XCTAssertNoThrow(try LinkProtocol.frame(.video, Data(count: LinkProtocol.maxFrameLength - 1)))
  }

  // MARK: - Decoders

  func testDecodesPing() {
    XCTAssertEqual(LinkProtocol.decodePing(Self.hex("0102030405060708")), 0x0102_0304_0506_0708)
    XCTAssertNil(LinkProtocol.decodePing(Self.hex("01020304050607")))
    XCTAssertNil(LinkProtocol.decodePing(Self.hex("010203040506070809")))
  }

  func testDecodesPingFromReaderFrame() throws {
    var reader = LinkFrameReader()
    let frames = try reader.feed(Self.hex("00000009" + "40" + "00000000075bcd15"))
    XCTAssertEqual(frames.first?.messageType, .ping)
    XCTAssertEqual(frames.first.flatMap { LinkProtocol.decodePing($0.payload) }, 123_456_789)
  }

  func testDecodesHelloAck() {
    XCTAssertEqual(
      LinkProtocol.decodeHelloAck(Data(#"{"ok":true,"protocol":1,"receiver":"glassesdata-receiver 0.1"}"#.utf8)),
      LinkProtocol.HelloAck(ok: true, error: nil, name: nil, receiver: "glassesdata-receiver 0.1", code: nil))
    XCTAssertEqual(
      LinkProtocol.decodeHelloAck(Data(#"{"error":"not paired with this Mac","ok":false}"#.utf8)),
      LinkProtocol.HelloAck(ok: false, error: "not paired with this Mac", name: nil, receiver: nil, code: nil))
    XCTAssertNil(LinkProtocol.decodeHelloAck(Data(#"{"error":"x"}"#.utf8)))
    XCTAssertNil(LinkProtocol.decodeHelloAck(Data(#"[true]"#.utf8)))
    XCTAssertNil(LinkProtocol.decodeHelloAck(Data("not json".utf8)))
  }

  // MARK: - JSON

  func testHelloIsCompactAndSorted() throws {
    let body = LinkProtocol.helloBody(app: "1.0", device: "iPhone15,2", name: "Phone", token: "abc/def")
    let frame = try LinkProtocol.json(.hello, body)
    XCTAssertEqual(frame.prefix(5), Self.hex("00000052" + "01"))
    XCTAssertEqual(
      String(decoding: frame.dropFirst(5), as: UTF8.self),
      #"{"app":"1.0","device":"iPhone15,2","name":"Phone","protocol":1,"token":"abc/def"}"#)
    let usb = try LinkProtocol.json(.hello, LinkProtocol.helloBody(app: "1.0", device: "d", name: "n", token: nil))
    XCTAssertFalse(String(decoding: usb.dropFirst(5), as: UTF8.self).contains("token"))
  }

  func testJSONRejectsNonFiniteNumbersAndBinaryTypes() {
    XCTAssertThrowsError(try LinkProtocol.json(.motion, ["a": [Double.nan, 0, 0]]))
    XCTAssertThrowsError(try LinkProtocol.json(.motion, ["a": [Double.infinity, 0, 0]]))
    XCTAssertThrowsError(try LinkProtocol.json(.video, ["t": 1]))
  }

  // MARK: - HEVC samples

  func testConvertsNALLengthsToFourBytes() {
    XCTAssertEqual(
      LinkProtocol.fourByteLengthPrefixed(Self.hex("0002" + "2601" + "0001" + "02"), lengthSize: 2),
      Self.hex("00000002" + "2601" + "00000001" + "02"))
    XCTAssertEqual(
      LinkProtocol.fourByteLengthPrefixed(Self.hex("01" + "26"), lengthSize: 1), Self.hex("00000001" + "26"))
    let fourByte = Self.hex("00000002" + "0201")
    XCTAssertEqual(LinkProtocol.fourByteLengthPrefixed(fourByte, lengthSize: 4), fourByte)
    XCTAssertNil(LinkProtocol.fourByteLengthPrefixed(Self.hex("0003" + "2601"), lengthSize: 2))
    XCTAssertNil(LinkProtocol.fourByteLengthPrefixed(Self.hex("00000000"), lengthSize: 4))
    XCTAssertNil(LinkProtocol.fourByteLengthPrefixed(Data(), lengthSize: 4))
    XCTAssertNil(LinkProtocol.fourByteLengthPrefixed(Self.hex("0001" + "26"), lengthSize: 3))
  }

  func testDetectsHEVCKeyframes() {
    // NAL header byte 0 is (type << 1): 0x26 = IDR_W_RADL (19), 0x02 = TRAIL_R (1), 0x40 = VPS (32).
    XCTAssertTrue(LinkProtocol.containsHEVCKeyframe(Self.hex("00000002" + "4001" + "00000002" + "2601")))
    XCTAssertFalse(LinkProtocol.containsHEVCKeyframe(Self.hex("00000002" + "0201")))
    XCTAssertFalse(LinkProtocol.containsHEVCKeyframe(Self.hex("00000009" + "2601")))
  }

  // MARK: - Video gate

  func testGateWaitsForKeyframeAndSendsFormatFirst() {
    var gate = LinkVideoGate()
    let format = Self.hex("01")
    XCTAssertEqual(gate.admit(keyframe: false, hvcC: format, outstandingBytes: 0, outstandingFrames: 0), .drop)
    XCTAssertEqual(gate.admit(keyframe: true, hvcC: format, outstandingBytes: 0, outstandingFrames: 0), .send(withFormat: true))
    XCTAssertEqual(gate.admit(keyframe: false, hvcC: format, outstandingBytes: 0, outstandingFrames: 0), .send(withFormat: false))
    XCTAssertEqual(gate.admit(keyframe: true, hvcC: format, outstandingBytes: 0, outstandingFrames: 0), .send(withFormat: false))
  }

  func testGateResendsFormatAtNextKeyframeAfterChange() {
    var gate = LinkVideoGate()
    _ = gate.admit(keyframe: true, hvcC: Self.hex("01"), outstandingBytes: 0, outstandingFrames: 0)
    XCTAssertEqual(gate.admit(keyframe: false, hvcC: Self.hex("02"), outstandingBytes: 0, outstandingFrames: 0), .drop)
    XCTAssertEqual(gate.admit(keyframe: false, hvcC: Self.hex("02"), outstandingBytes: 0, outstandingFrames: 0), .drop)
    XCTAssertEqual(gate.admit(keyframe: true, hvcC: Self.hex("02"), outstandingBytes: 0, outstandingFrames: 0), .send(withFormat: true))
    XCTAssertEqual(gate.sentFormat, Self.hex("02"))
  }

  func testGateDropsUnderBackpressureUntilNextKeyframe() {
    var gate = LinkVideoGate()
    let format = Self.hex("01")
    _ = gate.admit(keyframe: true, hvcC: format, outstandingBytes: 0, outstandingFrames: 0)
    XCTAssertEqual(
      gate.admit(keyframe: false, hvcC: format, outstandingBytes: LinkVideoGate.maxOutstandingBytes, outstandingFrames: 0),
      .send(withFormat: false))
    XCTAssertEqual(
      gate.admit(keyframe: false, hvcC: format, outstandingBytes: LinkVideoGate.maxOutstandingBytes + 1, outstandingFrames: 0),
      .drop)
    // The queue has drained, but a non-keyframe after a drop would not decode.
    XCTAssertEqual(gate.admit(keyframe: false, hvcC: format, outstandingBytes: 0, outstandingFrames: 0), .drop)
    XCTAssertEqual(gate.admit(keyframe: true, hvcC: format, outstandingBytes: 0, outstandingFrames: 0), .send(withFormat: false))
    XCTAssertEqual(
      gate.admit(keyframe: true, hvcC: format, outstandingBytes: 0, outstandingFrames: LinkVideoGate.maxOutstandingFrames + 1),
      .drop)
    XCTAssertTrue(gate.needsKeyframe)
  }

  func testGateStartsOverAfterLostFrame() {
    var gate = LinkVideoGate()
    _ = gate.admit(keyframe: true, hvcC: Self.hex("01"), outstandingBytes: 0, outstandingFrames: 0)
    gate.noteDropped()
    XCTAssertEqual(gate.admit(keyframe: false, hvcC: Self.hex("01"), outstandingBytes: 0, outstandingFrames: 0), .drop)
    XCTAssertEqual(gate.admit(keyframe: true, hvcC: Self.hex("01"), outstandingBytes: 0, outstandingFrames: 0), .send(withFormat: true))
  }

  // MARK: - Helpers

  private static func hex(_ string: String) -> Data {
    var data = Data()
    var index = string.startIndex
    while index < string.endIndex {
      let next = string.index(index, offsetBy: 2)
      guard let byte = UInt8(string[index..<next], radix: 16) else { preconditionFailure("bad hex \(string)") }
      data.append(byte)
      index = next
    }
    return data
  }
}
