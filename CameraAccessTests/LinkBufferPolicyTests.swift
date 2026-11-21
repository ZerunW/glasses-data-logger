import Foundation
import XCTest
@testable import GlassesData

/// The link's buffering: low latency normally, record priority while the computer records.
final class LinkBufferPolicyTests: XCTestCase {
  private let format = Data([0x01])
  private let mib = 1024 * 1024

  // MARK: - Budgets

  func testLowLatencyDropsOverTwoMiBOrTwelveFrames() {
    var gate = startedGate()
    let policy = LinkBufferPolicy.lowLatency
    XCTAssertEqual(admit(&gate, outstandingBytes: 2 * mib, frames: 12, policy: policy), .send(withFormat: false))
    XCTAssertEqual(admit(&gate, outstandingBytes: 0, frames: 13, policy: policy), .drop)
    gate = startedGate()
    XCTAssertEqual(admit(&gate, outstandingBytes: 2 * mib + 1, frames: 0, policy: policy), .drop)
  }

  func testRecordPriorityBuffersUpToSixtyFourMiBWithoutAFrameLimit() {
    var gate = startedGate()
    let policy = LinkBufferPolicy.recordPriority
    // Far past the low-latency limits, but within the record-priority budget.
    XCTAssertEqual(admit(&gate, outstandingBytes: 3 * mib, frames: 13, policy: policy), .send(withFormat: false))
    XCTAssertEqual(admit(&gate, outstandingBytes: 64 * mib, frames: 5_000, policy: policy), .send(withFormat: false))
    XCTAssertEqual(admit(&gate, outstandingBytes: 64 * mib + 1, frames: 0, policy: policy), .drop)
  }

  func testBackpressureSwitchesBudgetWithTheComputersRecording() {
    var backpressure = LinkBackpressure()
    XCTAssertEqual(backpressure.admission, .lowLatency)
    backpressure.setRecordPriority(true, outstandingBytes: 0)
    XCTAssertEqual(backpressure.admission, .recordPriority)
    backpressure.setRecordPriority(false, outstandingBytes: 0)
    XCTAssertEqual(backpressure.admission, .lowLatency)
  }

  // MARK: - Keyframes

  func testResumesAtKeyframeAfterADropInEitherPolicy() {
    for policy in [LinkBufferPolicy.lowLatency, .recordPriority] {
      var gate = startedGate()
      XCTAssertEqual(admit(&gate, outstandingBytes: policy.videoBudgetBytes + 1, frames: 0, policy: policy), .drop)
      // The queue drained, but a frame after a gap would not decode without its keyframe.
      XCTAssertEqual(admit(&gate, outstandingBytes: 0, frames: 0, policy: policy), .drop)
      XCTAssertEqual(admit(&gate, keyframe: true, outstandingBytes: 0, frames: 0, policy: policy), .send(withFormat: false))
      XCTAssertEqual(admit(&gate, outstandingBytes: 0, frames: 0, policy: policy), .send(withFormat: false))
    }
  }

  // MARK: - Closing a stalled connection

  func testCloseThresholds() {
    var backpressure = LinkBackpressure()
    XCTAssertFalse(backpressure.shouldClose(outstandingBytes: 8 * mib))
    XCTAssertTrue(backpressure.shouldClose(outstandingBytes: 8 * mib + 1))
    backpressure.setRecordPriority(true, outstandingBytes: 0)
    XCTAssertFalse(backpressure.shouldClose(outstandingBytes: 96 * mib))
    XCTAssertTrue(backpressure.shouldClose(outstandingBytes: 96 * mib + 1))
  }

  /// Leaving record priority keeps what is queued: new frames get the low limits at once,
  /// but the connection isn't closed for the backlog until it has drained.
  func testLeavingRecordPriorityKeepsTheQueue() {
    var backpressure = LinkBackpressure()
    backpressure.setRecordPriority(true, outstandingBytes: 0)
    backpressure.setRecordPriority(false, outstandingBytes: 50 * mib)
    XCTAssertEqual(backpressure.admission, .lowLatency, "New frames get the low budget")
    XCTAssertTrue(backpressure.draining)
    XCTAssertFalse(backpressure.shouldClose(outstandingBytes: 50 * mib), "The backlog is not dropped")
    XCTAssertFalse(backpressure.shouldClose(outstandingBytes: 90 * mib))
    XCTAssertTrue(backpressure.shouldClose(outstandingBytes: 96 * mib + 1), "A real stall still ends the connection")

    var drained = LinkBackpressure()
    drained.setRecordPriority(true, outstandingBytes: 0)
    drained.setRecordPriority(false, outstandingBytes: 50 * mib)
    XCTAssertFalse(drained.shouldClose(outstandingBytes: 7 * mib))
    XCTAssertFalse(drained.draining, "Once below the low threshold, the low threshold applies again")
    XCTAssertTrue(drained.shouldClose(outstandingBytes: 9 * mib))
  }

  func testLeavingRecordPriorityWithASmallQueueUsesTheLowThresholdAtOnce() {
    var backpressure = LinkBackpressure()
    backpressure.setRecordPriority(true, outstandingBytes: 0)
    backpressure.setRecordPriority(false, outstandingBytes: mib)
    XCTAssertFalse(backpressure.draining)
    XCTAssertTrue(backpressure.shouldClose(outstandingBytes: 9 * mib))
  }

  func testRecordingAgainWhileDrainingRestoresRecordPriority() {
    var backpressure = LinkBackpressure()
    backpressure.setRecordPriority(true, outstandingBytes: 0)
    backpressure.setRecordPriority(false, outstandingBytes: 50 * mib)
    backpressure.setRecordPriority(true, outstandingBytes: 50 * mib)
    XCTAssertFalse(backpressure.draining)
    XCTAssertEqual(backpressure.admission, .recordPriority)
    XCTAssertFalse(backpressure.shouldClose(outstandingBytes: 96 * mib))
  }

  // MARK: - Gaps

  func testGapsCountEachRunOfDroppedVideoAfterItStarted() {
    var gate = LinkVideoGate()
    // Waiting for the first keyframe is not a gap.
    XCTAssertEqual(admit(&gate, outstandingBytes: 0, frames: 0), .drop)
    XCTAssertEqual(gate.gaps, 0)
    XCTAssertEqual(admit(&gate, keyframe: true, outstandingBytes: 0, frames: 0), .send(withFormat: true))
    // One stall: several drops, one gap.
    XCTAssertEqual(admit(&gate, outstandingBytes: 3 * mib, frames: 0), .drop)
    XCTAssertEqual(admit(&gate, outstandingBytes: 0, frames: 0), .drop)
    XCTAssertEqual(gate.gaps, 1)
    XCTAssertEqual(admit(&gate, keyframe: true, outstandingBytes: 0, frames: 0), .send(withFormat: false))
    // A frame lost after admission is a gap too.
    gate.noteDropped()
    XCTAssertEqual(gate.gaps, 2)
    XCTAssertEqual(admit(&gate, keyframe: true, outstandingBytes: 0, frames: 0), .send(withFormat: true))
    XCTAssertEqual(gate.gaps, 2)
  }

  // MARK: - Helpers

  private func startedGate() -> LinkVideoGate {
    var gate = LinkVideoGate()
    _ = gate.admit(keyframe: true, hvcC: format, outstandingBytes: 0, outstandingFrames: 0)
    return gate
  }

  private func admit(
    _ gate: inout LinkVideoGate, keyframe: Bool = false, outstandingBytes: Int, frames: Int,
    policy: LinkBufferPolicy = .lowLatency
  ) -> LinkVideoGate.Decision {
    gate.admit(keyframe: keyframe, hvcC: format, outstandingBytes: outstandingBytes, outstandingFrames: frames, policy: policy)
  }
}
