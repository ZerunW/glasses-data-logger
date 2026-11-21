//
// RecordingControl.swift
//
// The pure logic behind the Record button. "A recording" is the computer's recording
// (whenever the link is connected) plus a recording on this iPhone (only when Save on this
// iPhone is on). The computer's side follows its RECORDER_STATE messages, so a recording
// started from the receiver's web page shows here too and can be stopped from here.
//

import Foundation

/// What the phone knows about the computer's recorder.
enum ComputerRecorder: Equatable, Sendable {
  /// No computer connected and no recording outstanding.
  case unavailable
  /// Connected and not recording. `last` is the latest recording, when there was one.
  case idle(last: LinkRecorderState?)
  case recording(LinkRecorderState)
  /// The link went down while the computer was recording. The receiver ends a recording
  /// when its phone disconnects, but the phone can't confirm that until it reconnects.
  case unknown(last: LinkRecorderState)

  var isRecording: Bool {
    if case .recording = self { return true }
    return false
  }

  var isUnknown: Bool {
    if case .unknown = self { return true }
    return false
  }
}

/// Follows the computer's recorder through link status updates and the phone's own
/// CONTROL commands. Pure state, so it is unit tested without a socket.
struct ComputerRecorderTracker: Equatable, Sendable {
  /// A command was sent and the computer hasn't answered it yet.
  struct Pending: Equatable, Sendable {
    let command: LinkControlCommand
    let id: Int
    /// `LinkStatus.recorderUpdates` when the command was sent: only a later RECORDER_STATE
    /// can answer it, not the once-a-second repeat of the state from before.
    let afterUpdate: UInt64
  }

  /// Reported to the view model when a start the phone asked for didn't happen.
  enum Outcome: Equatable, Sendable {
    case startRefused(String)
    case noAnswer(LinkControlCommand)
  }

  static let answerTimeout: Duration = .seconds(4)

  private(set) var recorder: ComputerRecorder = .unavailable
  private(set) var pending: Pending?
  /// The computer's latest note (why a command was refused, a rotation, a failure), or the
  /// phone's own note about the link.
  private(set) var note: String?
  private var nextID = 0

  /// Applies a link status. Status updates arrive about once a second while connected,
  /// so this is idempotent for an unchanged status.
  @discardableResult
  mutating func apply(_ status: LinkStatus) -> Outcome? {
    guard status.phase == .connected else {
      switch recorder {
      case .recording(let state):
        recorder = .unknown(last: state)
        note = "Link lost while the computer was recording. It may have stopped."
      case .unknown:
        break
      case .unavailable, .idle:
        recorder = .unavailable
        note = pending == nil ? nil : "The link went down before the computer answered."
      }
      pending = nil
      return nil
    }
    guard let state = status.recorder else {
      // Connected, but the receiver hasn't reported yet (it does right after HELLO_ACK;
      // an older receiver never does). An unknown recording stays unknown until it does.
      if case .unknown = recorder { return nil }
      if case .recording = recorder { return nil }
      recorder = .idle(last: nil)
      return nil
    }
    recorder = state.recording ? .recording(state) : .idle(last: state.name == nil ? nil : state)
    note = state.note
    guard let current = pending, status.recorderUpdates > current.afterUpdate else { return nil }
    switch current.command {
    case .recordStart:
      pending = nil
      if !state.recording {
        return .startRefused(state.note ?? "The computer didn't start recording.")
      }
    case .recordStop:
      // A once-a-second update sent before the stop arrived still says recording.
      if !state.recording { pending = nil }
    }
    return nil
  }

  /// Notes a command the phone just sent, with the link's `recorderUpdates` at that
  /// moment; returns its id for `expire`.
  mutating func sent(_ command: LinkControlCommand, afterUpdate: UInt64) -> Int {
    nextID &+= 1
    pending = Pending(command: command, id: nextID, afterUpdate: afterUpdate)
    if command == .recordStart { note = nil }
    return nextID
  }

  /// Called `answerTimeout` after a command; clears it if it is still unanswered.
  mutating func expire(_ id: Int) -> Outcome? {
    guard let current = pending, current.id == id else { return nil }
    pending = nil
    note = "The computer didn't answer. Is the receiver up to date?"
    return .noAnswer(current.command)
  }
}

/// What the Record button shows and does.
struct RecordControl: Equatable, Sendable {
  enum Mode: Equatable, Sendable { case start, stop }

  struct Inputs: Equatable, Sendable {
    var isStreaming: Bool
    var linkConnected: Bool
    var saveOnPhone: Bool
    var phoneRecording: Bool
    /// The phone's recording is being set up or saved.
    var phoneBusy: Bool
    var computer: ComputerRecorder
    var pending: LinkControlCommand?
  }

  var mode: Mode
  var isEnabled: Bool
  /// Under the button: where a recording goes or what is recording, or why it can't start.
  var caption: String?
  /// The caption explains why the button is disabled.
  var captionIsWarning: Bool = false
  /// When the caption names destinations: their SF Symbols, for a compact form of it.
  var destinationSymbols: [String] = []
  /// Start on the computer (sends record_start).
  var startsComputer = false
  /// Start on this iPhone.
  var startsPhone = false

  static func evaluate(_ inputs: Inputs) -> RecordControl {
    let computerRecording = inputs.computer.isRecording
    if inputs.phoneRecording || computerRecording {
      let caption = destinations(computer: computerRecording, phone: inputs.phoneRecording)
      if inputs.pending == .recordStop {
        return RecordControl(mode: .stop, isEnabled: false, caption: "stopping…")
      }
      // Saving the phone's previous take doesn't hold up stopping the computer.
      return RecordControl(
        mode: .stop, isEnabled: computerRecording || !inputs.phoneBusy, caption: caption,
        destinationSymbols: symbols(computer: computerRecording, phone: inputs.phoneRecording))
    }
    if inputs.pending == .recordStart {
      return RecordControl(mode: .start, isEnabled: false, caption: "starting…")
    }
    if inputs.phoneBusy {
      return RecordControl(mode: .start, isEnabled: false, caption: "saving…")
    }
    guard inputs.isStreaming else {
      return RecordControl(mode: .start, isEnabled: false, caption: nil)
    }
    guard inputs.linkConnected || inputs.saveOnPhone else {
      return RecordControl(mode: .start, isEnabled: false, caption: "nowhere to save", captionIsWarning: true)
    }
    return RecordControl(
      mode: .start, isEnabled: true, caption: destinations(computer: inputs.linkConnected, phone: inputs.saveOnPhone),
      destinationSymbols: symbols(computer: inputs.linkConnected, phone: inputs.saveOnPhone),
      startsComputer: inputs.linkConnected, startsPhone: inputs.saveOnPhone)
  }

  private static func symbols(computer: Bool, phone: Bool) -> [String] {
    (computer ? ["laptopcomputer"] : []) + (phone ? ["iphone"] : [])
  }

  /// The longer explanation for VoiceOver.
  var accessibilityHint: String? {
    switch caption {
    case "nowhere to save":
      return "Connect a computer, or turn on Save on this iPhone in the Storage card."
    case nil where mode == .start && !isEnabled:
      return "Start the preview to record."
    default:
      return nil
    }
  }

  private static func destinations(computer: Bool, phone: Bool) -> String {
    switch (computer, phone) {
    case (true, true): return "computer + iPhone"
    case (true, false): return "computer"
    case (false, true): return "iPhone"
    case (false, false): return ""
    }
  }
}
