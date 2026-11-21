//
// StatusHeaderView.swift
//
// The camera screen's status header. The first line has a coloured dot and a few words
// each for the glasses session and the camera stream. The second line is the computer: a
// tinted capsule with the link state, which becomes the computer's recording readout
// (name, elapsed time, frames) while it records, followed by the computer's latest note.
// Each item is its own view and reads only the state it shows, so the once-a-second fps
// and link updates re-render that item alone.
//

import MWDATCamera
import MWDATCore
import SwiftUI

/// What a status dot says: green = active/connected, yellow = in progress or waiting,
/// red = failed, grey = off.
enum StatusTone {
  case active, pending, failed, off, recording, warning

  var color: Color {
    switch self {
    case .active: return .green
    case .pending: return .yellow
    case .failed: return .red
    case .off: return .gray
    case .recording: return .recordAccent
    case .warning: return .orange
    }
  }
}

struct StatusHeaderView: View {
  var viewModel: CameraViewModel
  /// Tapping the Computer item selects the Computer card.
  let onSelectComputer: () -> Void

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      HStack(spacing: 14) {
        GlassesStatusItem(viewModel: viewModel)
        StreamStatusItem(viewModel: viewModel)
      }
      ComputerStatusItem(viewModel: viewModel, action: onSelectComputer)
    }
    // The header never shrinks; at large text sizes its words scale down instead.
    .dynamicTypeSize(...DynamicTypeSize.xxLarge)
  }
}

// MARK: - Items

/// The glasses session: whether the app holds a live `DeviceSession`.
private struct GlassesStatusItem: View {
  var viewModel: CameraViewModel

  var body: some View {
    let state = viewModel.sessionState
    StatusItem(title: "Glasses", value: Self.text(for: state), tone: Self.tone(for: state))
      .accessibilityIdentifier("glasses_status")
  }

  private static func text(for state: DeviceSessionState) -> String {
    switch state {
    case .idle, .stopped: return "no session"
    case .starting: return "starting"
    case .started: return "started"
    case .paused: return "paused"
    case .stopping: return "stopping"
    }
  }

  private static func tone(for state: DeviceSessionState) -> StatusTone {
    switch state {
    case .started: return .active
    case .starting, .paused, .stopping: return .pending
    case .idle, .stopped: return .off
    }
  }
}

/// The camera stream, with the measured frame rate while it streams.
private struct StreamStatusItem: View {
  var viewModel: CameraViewModel

  var body: some View {
    StatusItem(title: "Stream", value: text, tone: tone)
      .accessibilityIdentifier("stream_status")
  }

  private var text: String {
    switch viewModel.streamState {
    case .streaming:
      guard let fps = viewModel.measuredStreamFps else { return "streaming" }
      return String(format: "%.0f fps", fps)
    case .starting: return "starting"
    case .waitingForDevice: return "waiting"
    case .paused: return "paused"
    case .stopping: return "stopping"
    case .stopped: return "off"
    }
  }

  private var tone: StatusTone {
    switch viewModel.streamState {
    case .streaming: return .active
    case .starting, .waitingForDevice, .paused, .stopping: return .pending
    case .stopped: return .off
    }
  }
}

/// The computer: a tinted capsule with the link's mode and phase ("USB · connected"), or
/// the computer's recording while it records, then its latest note. A button that selects
/// the Computer card, where the link is configured.
private struct ComputerStatusItem: View {
  var viewModel: CameraViewModel
  let action: () -> Void

  var body: some View {
    let status = viewModel.linkStatus
    let computer = viewModel.computer
    Button(action: action) {
      HStack(spacing: 6) {
        ComputerPill(status: status, computer: computer)
          .layoutPriority(1)
        // While the state is unknown the pill already says why.
        if let note = computer.note, !computer.recorder.isUnknown {
          NoteChip(text: note)
        }
      }
      .contentShape(Rectangle())
    }
    .buttonStyle(.plain)
    // On the button itself, so it stays one Button element with this label.
    .accessibilityLabel(Text(verbatim: Self.accessibilityText(status: status, computer: computer)))
    .accessibilityHint("Shows the Computer card")
    .accessibilityIdentifier("computer_status_button")
  }

  static func accessibilityText(status: LinkStatus, computer: ComputerRecorderTracker) -> String {
    var text = "Computer: " + ComputerPill.linkText(for: status)
    switch computer.recorder {
    case .recording(let state):
      text += ", recording \(state.seconds.formattedRecordingTime), \(state.frames) frames"
      if let name = state.name { text += ", \(name)" }
    case .unknown: text += ", recording state unknown"
    case .idle, .unavailable: break
    }
    if let note = computer.note { text += ". \(note)" }
    return text
  }
}

/// The capsule itself. While the computer records it reads "REC 01:23 · 2,041 frames ·
/// 2026-11-20_212051", dropping words before it truncates the name.
private struct ComputerPill: View {
  let status: LinkStatus
  let computer: ComputerRecorderTracker

  var body: some View {
    let tone = self.tone
    HStack(spacing: 6) {
      Circle()
        .fill(tone.color)
        .frame(width: 7, height: 7)
      content
        .font(.footnote.weight(.semibold))
        .foregroundStyle(.white)
        .monospacedDigit()
        .lineLimit(1)
    }
    .padding(.horizontal, 10)
    .frame(height: 26)
    .background(Capsule().fill(tone.color.opacity(0.2)))
    .overlay(Capsule().strokeBorder(tone.color.opacity(0.45), lineWidth: 0.5))
  }

  @ViewBuilder
  private var content: some View {
    switch computer.recorder {
    case .recording(let state):
      let time = state.seconds.formattedRecordingTime
      let frames = state.frames.formatted(.number)
      ViewThatFits(in: .horizontal) {
        recordingLine(time: time, frames: "\(frames) frames", name: state.name)
        recordingLine(time: time, frames: "\(frames) fr", name: state.name)
        HStack(spacing: 4) {
          Text(verbatim: "REC \(time) · \(frames) fr")
            .fixedSize()
          if let name = state.name {
            Text(verbatim: "· \(name)")
              .truncationMode(.middle)
              .foregroundStyle(Color.white.opacity(0.75))
          }
        }
      }
    case .unknown:
      Text(verbatim: "Link lost · recording unknown")
        .minimumScaleFactor(0.8)
    case .idle, .unavailable:
      Text(verbatim: Self.linkText(for: status) + pendingSuffix)
        .minimumScaleFactor(0.8)
    }
  }

  private func recordingLine(time: String, frames: String, name: String?) -> some View {
    HStack(spacing: 4) {
      Text(verbatim: "REC \(time) · \(frames)")
      if let name {
        Text(verbatim: "· \(name)")
          .foregroundStyle(Color.white.opacity(0.75))
      }
    }
    .fixedSize()
  }

  private var pendingSuffix: String {
    switch computer.pending?.command {
    case .recordStart: return " · starting…"
    case .recordStop: return " · stopping…"
    case nil: return ""
    }
  }

  private var tone: StatusTone {
    switch computer.recorder {
    case .recording: return .recording
    case .unknown: return .warning
    case .idle, .unavailable: return Self.tone(for: status.phase)
    }
  }

  static func linkText(for status: LinkStatus) -> String {
    let phase: String
    switch status.phase {
    case .off: return "Computer off"
    case .waitingForComputer: phase = "waiting for computer"
    case .searching: phase = "searching"
    case .connecting: phase = "connecting"
    case .connected: phase = "connected"
    case .needsPairing: phase = "pairing needed"
    case .failed: phase = "not connected"
    }
    let mode: String
    switch status.mode {
    case .off: return "Computer off"
    case .usb: mode = "USB"
    case .wifi: mode = "Wi-Fi"
    }
    return "\(mode) · \(phase)"
  }

  static func tone(for phase: LinkStatus.Phase) -> StatusTone {
    switch phase {
    case .off: return .off
    case .connected: return .active
    // Pairing is waiting on the user rather than broken, so it reads as in progress.
    case .waitingForComputer, .searching, .connecting, .needsPairing: return .pending
    case .failed: return .failed
    }
  }
}

/// The computer's latest note, shortened to the room left on the line. The Computer card
/// shows it in full.
private struct NoteChip: View {
  let text: String

  var body: some View {
    HStack(spacing: 4) {
      Image(systemName: "exclamationmark.triangle.fill")
        .font(.caption2)
        .foregroundStyle(.yellow)
      Text(verbatim: text)
        .font(.caption)
        .foregroundStyle(Color.white.opacity(0.85))
        .lineLimit(1)
        .truncationMode(.tail)
    }
    .padding(.horizontal, 8)
    .frame(height: 26)
    .background(Capsule().fill(Color.yellow.opacity(0.12)))
    .accessibilityIdentifier("computer_note")
  }
}

/// A dot plus "Title value", read by VoiceOver as one element.
private struct StatusItem: View {
  let title: String
  let value: String
  let tone: StatusTone

  var body: some View {
    HStack(spacing: 6) {
      Circle()
        .fill(tone.color)
        .frame(width: 8, height: 8)
        .accessibilityHidden(true)
      HStack(spacing: 4) {
        Text(verbatim: title)
          .foregroundStyle(.white)
        Text(verbatim: value)
          .foregroundStyle(Color.white.opacity(0.7))
          .monospacedDigit()
      }
      .font(.footnote.weight(.medium))
      .lineLimit(1)
      .minimumScaleFactor(0.75)
    }
    .accessibilityElement(children: .ignore)
    .accessibilityLabel(Text(verbatim: "\(title): \(value)"))
  }
}
