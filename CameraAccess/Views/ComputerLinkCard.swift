//
// ComputerLinkCard.swift
//
// The link to the computer's receiver and the computer's recording: the link state first
// (large, coloured), then what the computer is recording, the connection mode, the glasses
// stream resolution, Wi-Fi pairing and the link counters. Draws no padding or background
// of its own: it sits inside the Computer card on the camera screen, which supplies both.
// Owns its own QR scanner sheet, so it can be placed anywhere.
//

import AVFoundation
import MWDATCamera
import SwiftUI
import VisionKit

struct ComputerLinkCard: View {
  var viewModel: CameraViewModel
  @State private var showScanner = false
  @State private var confirmForget = false
  @State private var notice: String?

  private var status: LinkStatus { viewModel.linkStatus }

  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      statusRow
      if viewModel.linkMode != .off {
        ComputerRecordingSection(viewModel: viewModel)
      }
      Picker("Connection", selection: modeBinding) {
        Text("Off").tag(LinkMode.off)
        Text("USB").tag(LinkMode.usb)
        Text("Wi-Fi").tag(LinkMode.wifi)
      }
      .pickerStyle(.segmented)
      .accessibilityIdentifier("link_mode_picker")

      resolutionRow
      if viewModel.linkMode == .wifi {
        pairingRow
      }
      if status.phase == .connected {
        countersRow
      }
      if let notice {
        Text(notice)
          .font(.caption)
          .foregroundStyle(.red)
          .lineLimit(2)
      }
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .sheet(isPresented: $showScanner) { scannerSheet }
    .confirmationDialog("Forget this computer?", isPresented: $confirmForget, titleVisibility: .visible) {
      Button("Forget this computer", role: .destructive) { notice = viewModel.forgetPairing() }
    } message: {
      Text("To use Wi-Fi again, scan the pairing code on your computer.")
    }
  }

  // MARK: - Rows

  /// The link state, large and coloured, with the link's own explanation under it.
  private var statusRow: some View {
    HStack(alignment: .top, spacing: 12) {
      Image(systemName: phaseSymbol)
        .font(.system(size: 17, weight: .semibold))
        .foregroundStyle(phaseColor)
        .frame(width: 38, height: 38)
        .background(Circle().fill(phaseColor.opacity(0.18)))
        .accessibilityHidden(true)
      VStack(alignment: .leading, spacing: 2) {
        Text(verbatim: phaseTitle)
          .font(.headline)
          .lineLimit(1)
          .minimumScaleFactor(0.8)
        Text(verbatim: statusDetail)
          .font(.caption)
          .foregroundStyle(.secondary)
          .lineLimit(2)
          .fixedSize(horizontal: false, vertical: true)
      }
    }
    .accessibilityElement(children: .combine)
    .accessibilityIdentifier("link_status")
  }

  @ViewBuilder
  private var pairingRow: some View {
    if let pairing = viewModel.linkPairing {
      HStack(spacing: 8) {
        Label(pairing.displayName, systemImage: "wifi")
          .font(.footnote)
          .lineLimit(1)
          .truncationMode(.middle)
        Spacer(minLength: 8)
        Menu {
          Button("Pair over Wi-Fi", systemImage: "qrcode.viewfinder") { startScan() }
          Button("Forget this computer", systemImage: "trash", role: .destructive) { confirmForget = true }
        } label: {
          Image(systemName: "ellipsis.circle")
            .imageScale(.large)
        }
        .accessibilityLabel("Pairing options")
      }
    } else {
      Button {
        startScan()
      } label: {
        Label("Pair over Wi-Fi", systemImage: "qrcode.viewfinder")
          .font(.footnote.weight(.semibold))
      }
      .buttonStyle(.bordered)
      .controlSize(.small)
      .accessibilityIdentifier("pair_wifi_button")
    }
  }

  /// The requested quality on one line, and the size the glasses actually deliver once
  /// frames arrive (the glasses may send less than they were asked for).
  private var resolutionRow: some View {
    HStack(spacing: 8) {
      Text("Video")
        .font(.footnote)
      if viewModel.hasStream {
        Image(systemName: "lock.fill")
          .font(.caption2)
          .foregroundStyle(.secondary)
          .accessibilityLabel("Locked while previewing")
      }
      Spacer(minLength: 8)
      if let size = viewModel.firstFrameSize {
        Text(verbatim: "receiving \(Int(size.width))×\(Int(size.height))")
          .font(.caption.monospacedDigit())
          .foregroundStyle(.secondary)
          .lineLimit(1)
          .accessibilityIdentifier("delivered_resolution")
      }
      Menu {
        Picker("Resolution", selection: resolutionBinding) {
          ForEach(StreamingResolution.allCases, id: \.self) { resolution in
            Text(verbatim: Self.title(for: resolution)).tag(resolution)
          }
        }
      } label: {
        HStack(spacing: 3) {
          // Once frames arrive their real size is shown beside the menu, so the label
          // keeps just the quality name there.
          Text(verbatim: viewModel.firstFrameSize == nil
            ? Self.title(for: viewModel.streamResolution) : Self.name(for: viewModel.streamResolution))
          Image(systemName: "chevron.up.chevron.down")
            .imageScale(.small)
        }
        .font(.footnote.weight(.semibold))
        .lineLimit(1)
        .fixedSize()
      }
      .disabled(viewModel.hasStream)
      .opacity(viewModel.hasStream ? 0.5 : 1)
      .accessibilityLabel("Video resolution")
      .accessibilityValue(Text(verbatim: Self.title(for: viewModel.streamResolution)))
      .accessibilityIdentifier("stream_resolution_picker")
    }
  }

  /// What the link has sent, and how much is waiting, while connected.
  private var countersRow: some View {
    VStack(alignment: .leading, spacing: 4) {
      Text(verbatim: "Sent \(status.sentFrames.formatted()) · Dropped \(status.droppedFrames.formatted()) · Gaps \(status.gaps)")
        .font(.caption.monospacedDigit())
        // Frames skipped while waiting for the first keyframe are normal; a gap isn't.
        .foregroundStyle(status.gaps > 0 ? Color.orange : Color.secondary)
        .lineLimit(1)
        .minimumScaleFactor(0.8)
        .accessibilityIdentifier("link_frame_counts")
      if status.recordPriority || status.queuedBytes > 256 * 1024 {
        HStack(spacing: 6) {
          if status.recordPriority {
            Label("Buffering for the recording", systemImage: "tray.full")
              .font(.caption2.weight(.semibold))
              .foregroundStyle(.white)
              .padding(.horizontal, 8)
              .padding(.vertical, 3)
              .background(Capsule().fill(Color.recordAccent.opacity(0.35)))
          }
          Text(verbatim: "Queued \(Int64(status.queuedBytes).formattedStorage)")
            .font(.caption2.monospacedDigit())
            .foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("link_buffer")
      }
    }
  }

  private var scannerSheet: some View {
    NavigationStack {
      PairingScanner { uri in
        showScanner = false
        notice = viewModel.applyPairing(uri: uri)
      }
      .ignoresSafeArea(edges: .bottom)
      .navigationTitle("Scan pairing code")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar { Button("Cancel") { showScanner = false } }
    }
  }

  // MARK: - Actions and bindings

  private var modeBinding: Binding<LinkMode> {
    Binding(get: { viewModel.linkMode }, set: { viewModel.setLinkMode($0) })
  }

  private var resolutionBinding: Binding<StreamingResolution> {
    Binding(get: { viewModel.streamResolution }, set: { viewModel.setStreamResolution($0) })
  }

  private func startScan() {
    notice = nil
    Task {
      guard DataScannerViewController.isSupported else {
        notice = "QR scanning isn't supported on this device."
        return
      }
      let authorized = await AVCaptureDevice.requestAccess(for: .video)
      guard authorized, DataScannerViewController.isAvailable else {
        notice = "Allow camera access in Settings to scan the pairing code."
        return
      }
      showScanner = true
    }
  }

  // MARK: - Presentation

  /// The link's own explanation, except once connected, where it would only repeat the
  /// title: then how the phone is connected.
  private var statusDetail: String {
    guard status.phase == .connected else { return status.message }
    return status.mode == .wifi ? "Over Wi-Fi, encrypted" : "Over the USB cable"
  }

  private var phaseTitle: String {
    let title: String
    switch status.phase {
    case .off: title = "Off"
    case .waitingForComputer: title = "Waiting for computer"
    case .searching: title = "Searching"
    case .connecting: title = "Connecting"
    case .connected: title = "Connected"
    case .needsPairing: title = "Pairing needed"
    case .failed: title = "Not connected"
    }
    guard let peer = status.peerName else { return title }
    return "\(title) · \(peer)"
  }

  private var phaseColor: Color {
    switch status.phase {
    case .off: return .gray
    case .connected: return .green
    case .waitingForComputer, .searching, .connecting: return .yellow
    case .needsPairing: return .orange
    case .failed: return .red
    }
  }

  private var phaseSymbol: String {
    switch status.phase {
    case .off: return "laptopcomputer.slash"
    case .connected: return status.mode == .wifi ? "wifi" : "cable.connector"
    case .waitingForComputer, .searching, .connecting: return "laptopcomputer"
    case .needsPairing: return "qrcode.viewfinder"
    case .failed: return "exclamationmark.triangle"
    }
  }

  private static func name(for resolution: StreamingResolution) -> String {
    switch resolution {
    case .high: return "High"
    case .medium: return "Medium"
    case .low: return "Low"
    @unknown default: return String(describing: resolution).capitalized
    }
  }

  private static func title(for resolution: StreamingResolution) -> String {
    let size = resolution.videoFrameSize
    return "\(name(for: resolution)) \(size.width)×\(size.height)"
  }

}

/// What the computer is recording, from its RECORDER_STATE: name, elapsed time, frames,
/// and its note (why a command was refused, a rotation, a failure).
private struct ComputerRecordingSection: View {
  var viewModel: CameraViewModel

  var body: some View {
    let computer = viewModel.computer
    VStack(alignment: .leading, spacing: 4) {
      HStack(spacing: 6) {
        Circle()
          .fill(dotColor(computer.recorder))
          .frame(width: 8, height: 8)
        Text(verbatim: headline(computer))
          .font(.subheadline.weight(.semibold))
          .monospacedDigit()
          .lineLimit(1)
          .minimumScaleFactor(0.8)
        Spacer(minLength: 6)
        if computer.recorder.isRecording {
          // Also reachable without a glasses session, for a recording started on the computer.
          Button {
            viewModel.stopComputerRecording()
          } label: {
            Label("Stop", systemImage: "stop.fill")
              .font(.caption.weight(.semibold))
              .foregroundStyle(.white)
              .padding(.horizontal, 10)
              .padding(.vertical, 5)
              .background(Capsule().fill(Color.recordAccent.opacity(0.6)))
          }
          .buttonStyle(.plain)
          .disabled(computer.pending?.command == .recordStop)
          .accessibilityLabel("Stop the computer's recording")
          .accessibilityIdentifier("stop_computer_recording_button")
        }
      }
      if let name = recordingName(computer.recorder) {
        Text(verbatim: name)
          .font(.caption.monospaced())
          .foregroundStyle(.secondary)
          .lineLimit(1)
          .truncationMode(.middle)
      }
      if let last = lastRecording(computer.recorder) {
        Text(verbatim: last)
          .font(.caption.monospacedDigit())
          .foregroundStyle(.secondary)
          .lineLimit(1)
      }
      if let note = computer.note {
        Label {
          Text(verbatim: note)
        } icon: {
          Image(systemName: "exclamationmark.triangle.fill")
            .foregroundStyle(.yellow)
        }
        .font(.caption)
        .foregroundStyle(Color.white.opacity(0.85))
        .fixedSize(horizontal: false, vertical: true)
        .accessibilityIdentifier("computer_recording_note")
      }
    }
    .padding(10)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    .accessibilityElement(children: .contain)
    .accessibilityIdentifier("computer_recording")
  }

  private func headline(_ computer: ComputerRecorderTracker) -> String {
    switch computer.pending?.command {
    case .recordStart: return "Starting the computer's recording…"
    case .recordStop: return "Stopping the computer's recording…"
    case nil: break
    }
    switch computer.recorder {
    case .recording(let state):
      return "Recording · \(state.seconds.formattedRecordingTime) · \(state.frames.formatted()) frames"
    case .unknown:
      return "Recording state unknown"
    case .idle:
      return "Computer not recording"
    case .unavailable:
      return "Connect to record on the computer"
    }
  }

  private func recordingName(_ recorder: ComputerRecorder) -> String? {
    switch recorder {
    case .recording(let state): return state.name
    case .idle(let last?), .unknown(let last): return last.name
    case .idle(nil), .unavailable: return nil
    }
  }

  /// Length and frames of the last recording, when not recording.
  private func lastRecording(_ recorder: ComputerRecorder) -> String? {
    switch recorder {
    case .idle(let last?):
      return "Last recording: \(last.seconds.formattedRecordingTime) · \(last.frames.formatted()) frames"
    case .unknown(let last):
      return "Last known: \(last.seconds.formattedRecordingTime) · \(last.frames.formatted()) frames"
    case .recording, .idle(nil), .unavailable:
      return nil
    }
  }

  private func dotColor(_ recorder: ComputerRecorder) -> Color {
    switch recorder {
    case .recording: return .recordAccent
    case .unknown: return .orange
    case .idle: return .green
    case .unavailable: return .gray
    }
  }
}
