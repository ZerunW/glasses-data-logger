//
// VideoPanelView.swift
//
// The glasses video and everything drawn on it: the live (or frozen) frame, the paused
// badge, the start/stop loader and the status placeholder. The same stage renders in
// two sizes: a ~190 pt wide 9:16 box on the camera screen, and full screen. Only
// states of the video itself are ever drawn over it.
//

import SwiftUI

/// Sizes for the two places the stage is shown, so the overlays scale with the box.
enum VideoStageStyle {
  /// The 9:16 box beside the summary column, as narrow as ~120 pt on an iPhone SE.
  case compact
  /// The full-screen cover.
  case full
}

struct VideoStageView: View {
  var viewModel: CameraViewModel
  let style: VideoStageStyle
  let isUpdateRequired: Bool

  var body: some View {
    ZStack {
      Color.black
      if viewModel.showsLivePreview {
        // Live preview, the frozen last frame while paused, or the last frame held
        // while the stream tears down — so the paused badge / stop loader overlays
        // the preview rather than the placeholder.
        LivePreviewView(viewModel: viewModel, style: style)
        if viewModel.isPaused {
          // Dim the frozen frame so the badge reads as intentionally paused.
          Color.black.opacity(0.35)
          PausedOverlay(style: style)
        }
      } else if !viewModel.isBusy {
        StatusPlaceholder(viewModel: viewModel, style: style, isUpdateRequired: isUpdateRequired)
      }

      // Loader overlays whatever is behind it during any in-flight transition.
      if viewModel.isBusy {
        ZStack {
          Color.black.opacity(0.35)
          ProgressView()
            .scaleEffect(style == .full ? 1.5 : 1)
            .tint(.white)
        }
      }
    }
  }
}

// MARK: - Live preview

/// The streaming preview image, aspect-fit. Isolated so the ~24fps
/// `currentVideoFrame` updates re-render only this view, not the screen around it.
private struct LivePreviewView: View {
  var viewModel: CameraViewModel
  let style: VideoStageStyle

  var body: some View {
    if let videoFrame = viewModel.currentVideoFrame, viewModel.hasReceivedFirstFrame {
      Image(uiImage: videoFrame)
        .resizable()
        .aspectRatio(contentMode: .fit)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityLabel("Live video")
        .accessibilityHint(style == .compact ? "Opens the video full screen" : "Closes full screen")
        .accessibilityAddTraits(.isButton)
        .accessibilityIdentifier("live_video")
    } else {
      ProgressView()
        .scaleEffect(style == .full ? 1.5 : 1)
        .tint(.white)
    }
  }
}

// MARK: - Overlays

/// Label over the frozen preview while paused, so a held frame reads as
/// intentionally paused (with how to resume) rather than a stalled feed. A
/// `pause.fill` status glyph (not a circled/button shape) — resume is a glasses
/// gesture, not an on-screen control. In the narrow box it is a small frosted badge
/// inset from the edges, so the frame stays visible around it.
private struct PausedOverlay: View {
  let style: VideoStageStyle

  var body: some View {
    if style == .full {
      VStack(spacing: 8) {
        Image(systemName: "pause.fill")
          .font(.system(size: 36))
          .foregroundStyle(.white)
        Text("Paused")
          .font(.system(size: 20, weight: .semibold))
          .foregroundStyle(.white)
        Text("Tap your glasses to resume.")
          .font(.system(size: 15))
          .multilineTextAlignment(.center)
          .foregroundStyle(Color.white.opacity(0.7))
      }
      .padding(.horizontal, 24)
      .dynamicTypeSize(...DynamicTypeSize.xLarge)
      .accessibilityIdentifier("paused_overlay")
    } else {
      VStack(spacing: 4) {
        HStack(spacing: 5) {
          Image(systemName: "pause.fill")
            .font(.caption.weight(.bold))
          Text("Paused")
            .font(.subheadline.weight(.semibold))
        }
        .foregroundStyle(.white)
        Text("Tap your glasses to resume.")
          .font(.caption2)
          .multilineTextAlignment(.center)
          .foregroundStyle(Color.white.opacity(0.75))
          .lineLimit(3)
          .fixedSize(horizontal: false, vertical: true)
      }
      .padding(.horizontal, 12)
      .padding(.vertical, 10)
      .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
      .padding(.horizontal, 14)
      .dynamicTypeSize(...DynamicTypeSize.large)
      .accessibilityIdentifier("paused_overlay")
    }
  }
}

/// What the video area says when there is no preview to show.
private struct StatusPlaceholder: View {
  var viewModel: CameraViewModel
  let style: VideoStageStyle
  let isUpdateRequired: Bool

  var body: some View {
    if !viewModel.hasActiveDevice {
      placeholder(title: "Put on your glasses", subtitle: nil, showWaitingRow: true)
    } else if isUpdateRequired {
      placeholder(title: LocalizedStringKey(updateRequiredTitle), subtitle: "Apply the update to start your camera.", showWaitingRow: false)
    } else if !viewModel.hasSession {
      placeholder(title: "Ready", subtitle: "Start a session to connect to your glasses.", showWaitingRow: false)
    } else if viewModel.isPaused {
      placeholder(title: "Paused", subtitle: "Tap your glasses to resume.", showWaitingRow: false)
    } else {
      placeholder(title: "Session started", subtitle: "Start the preview to see the live camera feed.", showWaitingRow: false)
    }
  }

  private var isFull: Bool { style == .full }

  private func placeholder(title: LocalizedStringKey, subtitle: LocalizedStringKey?, showWaitingRow: Bool) -> some View {
    VStack(spacing: isFull ? 12 : 6) {
      Image(.cameraAccessIcon)
        .resizable()
        .renderingMode(.template)
        .foregroundStyle(.white)
        .aspectRatio(contentMode: .fit)
        .frame(width: isFull ? 88 : 36)
        .accessibilityHidden(true)

      Text(title)
        .font(isFull ? .system(size: 20, weight: .semibold) : .subheadline.weight(.semibold))
        .foregroundStyle(.white)
        .multilineTextAlignment(.center)
        .lineLimit(2)
        .minimumScaleFactor(0.8)

      if let subtitle {
        Text(subtitle)
          .font(isFull ? .system(size: 15) : .caption)
          .multilineTextAlignment(.center)
          .foregroundStyle(Color.white.opacity(0.7))
          .lineLimit(4)
          .minimumScaleFactor(0.8)
      }

      if showWaitingRow {
        HStack(alignment: .firstTextBaseline, spacing: isFull ? 8 : 4) {
          Image(systemName: "hourglass")
            .foregroundStyle(Color.white.opacity(0.7))
            .accessibilityHidden(true)
          Text("Waiting for an active device")
            .foregroundStyle(Color.white.opacity(0.7))
            .multilineTextAlignment(.leading)
            .lineLimit(2)
            .minimumScaleFactor(0.8)
        }
        .font(isFull ? .system(size: 14) : .caption2)
      }
    }
    .padding(.horizontal, isFull ? 24 : 8)
    .dynamicTypeSize(...DynamicTypeSize.xLarge)
  }
}

// MARK: - Full screen

/// The video alone on black, aspect-fit. Tapping anywhere or the close button returns
/// to the camera screen. Shows the recording clock while recording, since the Record
/// button isn't visible here.
struct FullScreenVideoView: View {
  var viewModel: CameraViewModel
  let isUpdateRequired: Bool
  let onClose: () -> Void

  var body: some View {
    ZStack(alignment: .top) {
      VideoStageView(viewModel: viewModel, style: .full, isUpdateRequired: isUpdateRequired)
        .ignoresSafeArea()
        .contentShape(Rectangle())
        .onTapGesture(perform: onClose)

      HStack(alignment: .center) {
        FullScreenRecordingBadge(viewModel: viewModel)
        Spacer()
        CircleButton(icon: "xmark", text: nil, action: onClose)
          .accessibilityLabel("Close full screen")
          .accessibilityIdentifier("close_fullscreen_button")
      }
      .padding(.horizontal, 20)
      .padding(.top, 8)
    }
    .background(Color.black.ignoresSafeArea())
    .environment(\.colorScheme, .dark)
    .statusBarHidden()
    // The camera screen can't present its error alert while this cover is up, so the
    // cover shows it instead (`CameraView` suppresses its own copy meanwhile).
    .alert("Something went wrong", isPresented: errorBinding) {
      Button("OK") { viewModel.dismissError() }
    } message: {
      Text(viewModel.errorMessage)
    }
  }

  private var errorBinding: Binding<Bool> {
    Binding(get: { viewModel.showError }, set: { viewModel.showError = $0 })
  }
}

/// A red dot and the recording clock, shown on the full-screen video while recording:
/// this iPhone's clock when it records, otherwise the computer's.
private struct FullScreenRecordingBadge: View {
  var viewModel: CameraViewModel

  var body: some View {
    if viewModel.isRecording || viewModel.isComputerRecording {
      HStack(spacing: 6) {
        Circle()
          .fill(Color.recordAccent)
          .frame(width: 8, height: 8)
        Group {
          if viewModel.isRecording {
            RecordingClockText(start: viewModel.recordingStartDate, identifier: "fullscreen_recording_indicator")
          } else if case .recording(let state) = viewModel.computer.recorder {
            Text(verbatim: state.seconds.formattedRecordingTime)
              .accessibilityIdentifier("fullscreen_recording_indicator")
          }
        }
        .font(.subheadline.weight(.semibold))
        .monospacedDigit()
        .foregroundStyle(.white)
      }
      .padding(.horizontal, 12)
      .padding(.vertical, 6)
      .background(Color.black.opacity(0.55), in: Capsule())
    }
  }
}
