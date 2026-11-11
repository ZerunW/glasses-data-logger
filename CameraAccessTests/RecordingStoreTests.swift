import Foundation
import MWDATCore
import MWDATInputs
import MWDATMotion
import XCTest
@testable import GlassesData

final class RecordingStoreTests: XCTestCase {
  func testFinishPreservesHeadersAndArchivesVideo() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = try RecordingStore(samplingRateHz: 30, rootURL: root, startedPhoneTimeNs: 100)
    let video = root.appendingPathComponent("temporary.mov")
    try Data([1, 2, 3]).write(to: video)

    store.appendMotion(MotionSample(timestampNs: 40, source: .glasses), phoneTimeNs: 101)
    store.appendInput(.capture(.doublePress, source: .captureButton, timestampMs: 7), phoneTimeNs: 102)
    store.appendDeviceState(DeviceState(batteryLevel: 72), phoneTimeNs: 103)
    store.noteFirstVideoFrame(phoneTimeNs: 104, ptsSeconds: 0, sourcePTSSeconds: 12, width: 360, height: 640)
    store.noteHFPAudioWritten()

    let archived = try await store.finish(videoSourceURL: video, endedPhoneTimeNs: 105)
    let archivedURL = try XCTUnwrap(archived)
    XCTAssertEqual(try Data(contentsOf: archivedURL), Data([1, 2, 3]))
    XCTAssertFalse(FileManager.default.fileExists(atPath: video.path))

    func rows(_ name: String) throws -> [Substring] {
      try String(contentsOf: store.directoryURL.appendingPathComponent(name), encoding: .utf8).split(separator: "\n")
    }
    let motion = try rows("motion.csv")
    XCTAssertEqual(motion.first, "phone_time_ns,glasses_time_ns,ax,ay,az,gx,gy,gz,mx,my,mz,qw,qx,qy,qz,source")
    let fields = try XCTUnwrap(motion.last).split(separator: ",", omittingEmptySubsequences: false)
    XCTAssertEqual(fields.count, 16)
    XCTAssertEqual(Array(fields.prefix(2)), ["101", "40"])
    XCTAssertTrue(fields.dropFirst(2).dropLast().allSatisfy(\.isEmpty))
    XCTAssertEqual(fields.last, "glasses")
    XCTAssertEqual(try rows("inputs.csv"), ["phone_time_ns,glasses_time_ms,source,type,detail", "102,7,captureButton,capture,doublePress"])
    XCTAssertEqual(try rows("device_state.csv").first, "phone_time_ns,battery,charging,don,hinge,thermal,link")
    XCTAssertTrue(try XCTUnwrap(rows("device_state.csv").last).hasPrefix("103,72,"))

    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    let info = try decoder.decode(RecordingStore.Info.self, from: Data(contentsOf: store.directoryURL.appendingPathComponent("info.json")))
    XCTAssertEqual(info.firstVideoFramePhoneTimeNs, 104)
    XCTAssertEqual(info.endedPhoneTimeNs, 105)
    XCTAssertEqual(info.videoWidth, 360)
    XCTAssertEqual(info.videoHeight, 640)
    XCTAssertTrue(info.wroteHFPAudio)
  }
}
