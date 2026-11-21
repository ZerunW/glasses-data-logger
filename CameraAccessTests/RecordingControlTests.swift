import Foundation
import XCTest
@testable import GlassesData

/// The Record button's logic: the computer's recorder as the phone tracks it, and what the
/// button offers for each combination of computer and phone.
final class RecordingControlTests: XCTestCase {
  private let recording = LinkRecorderState(recording: true, name: "2026-11-21_093001", seconds: 3, frames: 90, note: nil)
  private let stopped = LinkRecorderState(recording: false, name: "2026-11-21_093001", seconds: 9, frames: 270, note: nil)

  // MARK: - Tracker

  func testFollowsTheComputersState() {
    var tracker = ComputerRecorderTracker()
    tracker.apply(status(.waitingForComputer))
    XCTAssertEqual(tracker.recorder, .unavailable)
    tracker.apply(status(.connected))
    XCTAssertEqual(tracker.recorder, .idle(last: nil), "Connected before the first RECORDER_STATE")
    // Started on the computer's web page: shows as recording without any phone command.
    tracker.apply(status(.connected, recorder: recording, updates: 1))
    XCTAssertEqual(tracker.recorder, .recording(recording))
    tracker.apply(status(.connected, recorder: stopped, updates: 2))
    XCTAssertEqual(tracker.recorder, .idle(last: stopped))
    XCTAssertNil(tracker.note)
  }

  func testLinkLossWhileRecordingIsUnknownUntilTheComputerReports() {
    var tracker = ComputerRecorderTracker()
    tracker.apply(status(.connected, recorder: recording, updates: 1))
    tracker.apply(status(.waitingForComputer))
    XCTAssertEqual(tracker.recorder, .unknown(last: recording))
    XCTAssertNotNil(tracker.note)
    // Reconnected, but no state yet: still unknown.
    tracker.apply(status(.connected))
    XCTAssertEqual(tracker.recorder, .unknown(last: recording))
    tracker.apply(status(.connected, recorder: stopped, updates: 1))
    XCTAssertEqual(tracker.recorder, .idle(last: stopped))
    XCTAssertNil(tracker.note)
  }

  func testLinkLossWhileIdleIsUnavailable() {
    var tracker = ComputerRecorderTracker()
    tracker.apply(status(.connected, recorder: stopped, updates: 1))
    tracker.apply(status(.failed("x")))
    XCTAssertEqual(tracker.recorder, .unavailable)
  }

  func testStartIsAnsweredByTheNextState() {
    var tracker = ComputerRecorderTracker()
    tracker.apply(status(.connected, recorder: stopped, updates: 1))
    _ = tracker.sent(.recordStart, afterUpdate: 1)
    XCTAssertEqual(tracker.pending?.command, .recordStart)
    XCTAssertNil(tracker.apply(status(.connected, recorder: recording, updates: 2)))
    XCTAssertNil(tracker.pending)
  }

  /// The link republishes its status every second with the last state it has. Before
  /// the computer answers, that repeat still says "not recording", and must not read as
  /// a refusal.
  func testARepeatOfTheOldStateIsNotAnAnswer() {
    var tracker = ComputerRecorderTracker()
    tracker.apply(status(.connected, recorder: stopped, updates: 3))
    _ = tracker.sent(.recordStart, afterUpdate: 3)
    XCTAssertNil(tracker.apply(status(.connected, recorder: stopped, updates: 3)))
    XCTAssertEqual(tracker.pending?.command, .recordStart, "Still waiting for the answer")
    XCTAssertNil(tracker.apply(status(.connected, recorder: recording, updates: 4)))
    XCTAssertNil(tracker.pending)
    XCTAssertEqual(tracker.recorder, .recording(recording))
  }

  func testRefusedStartReportsTheComputersNote() {
    var tracker = ComputerRecorderTracker()
    tracker.apply(status(.connected, recorder: stopped, updates: 1))
    _ = tracker.sent(.recordStart, afterUpdate: 1)
    var refused = stopped
    refused.note = "no video yet: start the glasses preview in the app"
    XCTAssertEqual(tracker.apply(status(.connected, recorder: refused, updates: 2)), .startRefused(refused.note!))
    XCTAssertNil(tracker.pending)
    XCTAssertEqual(tracker.note, refused.note)
  }

  func testStopWaitsPastAStaleRecordingUpdate() {
    var tracker = ComputerRecorderTracker()
    tracker.apply(status(.connected, recorder: recording, updates: 5))
    _ = tracker.sent(.recordStop, afterUpdate: 5)
    var tick = recording
    tick.seconds = 4
    tracker.apply(status(.connected, recorder: tick, updates: 6))
    XCTAssertEqual(tracker.pending?.command, .recordStop, "A once-a-second update sent before the stop arrived")
    tracker.apply(status(.connected, recorder: stopped, updates: 7))
    XCTAssertNil(tracker.pending)
  }

  /// Stop pressed before the computer confirmed a start: the stop replaces the start, and
  /// the computer's late "recording" doesn't end the wait for its stop.
  func testStopDuringAPendingStart() {
    var tracker = ComputerRecorderTracker()
    tracker.apply(status(.connected, recorder: stopped, updates: 1))
    _ = tracker.sent(.recordStart, afterUpdate: 1)
    _ = tracker.sent(.recordStop, afterUpdate: 1)
    XCTAssertNil(tracker.apply(status(.connected, recorder: recording, updates: 2)))
    XCTAssertEqual(tracker.pending?.command, .recordStop)
    tracker.apply(status(.connected, recorder: stopped, updates: 3))
    XCTAssertNil(tracker.pending)
    XCTAssertEqual(tracker.recorder, .idle(last: stopped))
  }

  func testUnansweredCommandExpires() {
    var tracker = ComputerRecorderTracker()
    tracker.apply(status(.connected))
    let first = tracker.sent(.recordStart, afterUpdate: 0)
    _ = tracker.apply(status(.connected, recorder: recording, updates: 1))
    XCTAssertNil(tracker.expire(first), "An answered command doesn't expire")
    let second = tracker.sent(.recordStop, afterUpdate: 1)
    XCTAssertEqual(tracker.expire(second), .noAnswer(.recordStop))
    XCTAssertNil(tracker.pending)
    XCTAssertNotNil(tracker.note)
    XCTAssertNil(tracker.expire(second))
  }

  func testLinkLossClearsAPendingCommand() {
    var tracker = ComputerRecorderTracker()
    tracker.apply(status(.connected))
    _ = tracker.sent(.recordStart, afterUpdate: 0)
    tracker.apply(status(.waitingForComputer))
    XCTAssertNil(tracker.pending)
    XCTAssertEqual(tracker.recorder, .unavailable)
  }

  // MARK: - Record button

  func testNeedsAStreamToStart() {
    let control = evaluate(isStreaming: false, linkConnected: true, saveOnPhone: true)
    XCTAssertEqual(control.mode, .start)
    XCTAssertFalse(control.isEnabled)
  }

  func testDisabledWithAReasonWhenThereIsNowhereToSave() {
    let control = evaluate(linkConnected: false, saveOnPhone: false)
    XCTAssertFalse(control.isEnabled)
    XCTAssertEqual(control.caption, "nowhere to save")
    XCTAssertTrue(control.captionIsWarning)
    XCTAssertNotNil(control.accessibilityHint)
  }

  func testStartsWhereverItCan() {
    let computerOnly = evaluate(linkConnected: true, saveOnPhone: false)
    XCTAssertTrue(computerOnly.isEnabled)
    XCTAssertEqual(computerOnly.caption, "computer")
    XCTAssertTrue(computerOnly.startsComputer)
    XCTAssertFalse(computerOnly.startsPhone)

    let phoneOnly = evaluate(linkConnected: false, saveOnPhone: true)
    XCTAssertEqual(phoneOnly.caption, "iPhone")
    XCTAssertFalse(phoneOnly.startsComputer)
    XCTAssertTrue(phoneOnly.startsPhone)

    let both = evaluate(linkConnected: true, saveOnPhone: true)
    XCTAssertEqual(both.caption, "computer + iPhone")
    XCTAssertTrue(both.startsComputer && both.startsPhone)
  }

  func testStopsWhateverIsRecording() {
    let web = evaluate(linkConnected: true, saveOnPhone: true, computer: .recording(recording))
    XCTAssertEqual(web.mode, .stop)
    XCTAssertTrue(web.isEnabled)
    XCTAssertEqual(web.caption, "computer", "A recording started on the computer doesn't start the phone's")

    let phone = evaluate(isStreaming: false, linkConnected: false, saveOnPhone: true, phoneRecording: true)
    XCTAssertEqual(phone.mode, .stop)
    XCTAssertTrue(phone.isEnabled, "Stop works while paused")
    XCTAssertEqual(phone.caption, "iPhone")

    let both = evaluate(linkConnected: true, saveOnPhone: true, phoneRecording: true, computer: .recording(recording))
    XCTAssertEqual(both.caption, "computer + iPhone")

    // The link dropped mid-recording; the phone's own recording can still be stopped.
    let lost = evaluate(linkConnected: false, saveOnPhone: true, phoneRecording: true, computer: .unknown(last: recording))
    XCTAssertEqual(lost.mode, .stop)
    XCTAssertEqual(lost.caption, "iPhone")
  }

  func testWaitsForTheComputersAnswer() {
    let starting = evaluate(linkConnected: true, saveOnPhone: false, pending: .recordStart)
    XCTAssertFalse(starting.isEnabled)
    XCTAssertEqual(starting.caption, "starting…")
    let stopping = evaluate(linkConnected: true, saveOnPhone: false, computer: .recording(recording), pending: .recordStop)
    XCTAssertEqual(stopping.mode, .stop)
    XCTAssertFalse(stopping.isEnabled)
  }

  func testBusyWhileThePhonesRecordingIsSaved() {
    let control = evaluate(linkConnected: true, saveOnPhone: true, phoneBusy: true)
    XCTAssertFalse(control.isEnabled)
    XCTAssertEqual(control.caption, "saving…")
    // A computer recording can still be stopped while the phone's take is saved.
    let computer = evaluate(linkConnected: true, saveOnPhone: true, phoneBusy: true, computer: .recording(recording))
    XCTAssertEqual(computer.mode, .stop)
    XCTAssertTrue(computer.isEnabled)
  }

  // MARK: - Helpers

  private func status(_ phase: LinkStatus.Phase, recorder: LinkRecorderState? = nil, updates: UInt64 = 0) -> LinkStatus {
    var status = LinkStatus(mode: .usb, phase: phase, peerName: nil, message: "")
    status.recorder = recorder
    status.recorderUpdates = updates
    return status
  }

  private func evaluate(
    isStreaming: Bool = true, linkConnected: Bool, saveOnPhone: Bool, phoneRecording: Bool = false,
    phoneBusy: Bool = false, computer: ComputerRecorder? = nil, pending: LinkControlCommand? = nil
  ) -> RecordControl {
    RecordControl.evaluate(
      RecordControl.Inputs(
        isStreaming: isStreaming, linkConnected: linkConnected, saveOnPhone: saveOnPhone,
        phoneRecording: phoneRecording, phoneBusy: phoneBusy,
        computer: computer ?? (linkConnected ? .idle(last: nil) : .unavailable), pending: pending))
  }
}
