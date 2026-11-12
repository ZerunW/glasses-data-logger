import Foundation
import XCTest
@testable import GlassesData

final class TetherCamProtocolTests: XCTestCase {
  func testFragmentedMessagesPreserveOrderAndPayload() throws {
    let messages: [IucmMessage] = [
      .hello(version: 0x0102, deviceName: "GlassesData", appVersion: "1.0",
             cameras: [CameraDescriptor(id: 0, position: .back, name: "Meta Glasses")]),
      .ping(timestampUs: 123_456),
      .video(ptsUs: 654_321, keyframe: true, nalUnits: Data([0, 0, 0, 1, 0x26]))
    ]
    let wire = messages.reduce(into: Data()) { $0.append(IucmCodec.encode($1)) }
    let parser = IucmFrameParser()
    var decoded: [IucmMessage] = []
    for byte in wire {
      decoded.append(contentsOf: try parser.feedMessages(Data([byte])))
    }
    XCTAssertEqual(decoded, messages)
    XCTAssertEqual(parser.bufferedByteCount, 0)
  }

  func testOversizedFrameIsRejected() throws {
    let length = UInt32(Iucm.maxPayload + 1)
    var header = Data("IUCM".utf8)
    header.append(contentsOf: [0x11, 0, 0, 0])
    header.append(contentsOf: (0..<4).map { UInt8(truncatingIfNeeded: length >> (8 * $0)) })
    XCTAssertThrowsError(try IucmFrameParser().feed(header))
  }

  func testMalformedNalLengthIsRejected() {
    XCTAssertNil(HvccNal.split(Data([0, 0, 0, 8, 0x26])))
  }
}
