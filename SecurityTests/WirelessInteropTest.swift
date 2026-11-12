import CoreVideo
import Foundation
import XCTest
@testable import TetherCamCore

final class GlassesSenderInteropTests: XCTestCase {
    func testGlassesSenderDecodesInTetherCamReceiver() throws {
        let environment = ProcessInfo.processInfo.environment
        let sender = Process()
        sender.executableURL = URL(fileURLWithPath: try XCTUnwrap(environment["GLASSESDATA_WIRELESS_SENDER"]))
        sender.arguments = [try XCTUnwrap(environment["GLASSESDATA_SAMPLE_VIDEO"])]
        let output = Pipe()
        sender.standardOutput = output
        sender.standardError = FileHandle.nullDevice
        try sender.run()
        defer { if sender.isRunning { sender.terminate(); sender.waitUntilExit() } }
        let ready = output.fileHandleForReading.availableData
        XCTAssertTrue(String(decoding: ready, as: UTF8.self).contains("Ready for encrypted relay"))
        let receiver = Receiver(config: ReceiverConfig(endpoint: .tcp(host: "127.0.0.1", port: 7879)))
        let frames = Locked<Int>(0)
        receiver.onFrame = { _, _ in frames.with { $0 += 1 } }
        receiver.start()
        defer { receiver.stop() }
        Thread.sleep(forTimeInterval: 4)
        XCTAssertEqual(frames.value, 0, "No video may arrive before approval")
        let deadline = Date().addingTimeInterval(11)
        while frames.value < 10 && Date() < deadline { usleep(20_000) }
        XCTAssertGreaterThanOrEqual(frames.value, 10)
    }
}
