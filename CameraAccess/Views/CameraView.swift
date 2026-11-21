/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// CameraView.swift
//
// The capture screen, top to bottom with nothing stacked over the video: a status
// header (glasses session, stream, computer link), the portrait glasses video beside
// a column of live numbers, paged sensor cards, and the capture controls. Walks the
// SDK's camera lifecycle as explicit steps (Start Session → Start Streaming → Capture
// / Record → Stop Streaming → End Session).
//
// Performance: the video frame changes ~24 times a second, the fps once a second and
// the sensors up to 10 times a second. This view reads none of those; each lives in a
// small child view (`VideoStageView`, `LiveSummaryView`, `SensorCardsView`,
// `StatusHeaderView`, `RecordingTimerLabel`) so an update re-renders only that child.
//

import MWDATCore
import SwiftUI
import UIKit

let updateRequiredTitle = "Update required"

struct CameraView: View {
  @State private var viewModel: CameraViewModel
  @Bindable var wearablesVM: WearablesViewModel
  @State private var showSettingsMenu: Bool = false
  @State private var isLaunchingUpdate: Bool = false
  @State private var showFullScreenVideo = false
  @State private var selectedCard: SensorCard = .motion

  init(wearables: WearablesInterface, wearablesVM: WearablesViewModel) {
    self._viewModel = State(wrappedValue: CameraViewModel(wearables: wearables))
    self._wearablesVM = Bindable(wearablesVM)
  }

  private var isUpdateRequired: Bool {
    wearablesVM.requiresFirmwareUpdate
  }

  /// Whether the preview is live or tearing down — drives the in-row Preview
  /// toggle's icon, id, and action. (Recording implies a live preview.)
  private var previewIsActive: Bool {
    viewModel.isStreaming || viewModel.isRecording || viewModel.streamState == .stopping
  }

  var body: some View {
    ZStack {
      Color.black
        .ignoresSafeArea()

      VStack(spacing: 0) {
        header
          .fixedSize(horizontal: false, vertical: true)
        // Everything between the header and the controls flexes; the header and the
        // controls keep their natural height on every phone.
        GeometryReader { proxy in
          middle(MiddleLayout(size: proxy.size))
        }
        bottomBar
          .fixedSize(horizontal: false, vertical: true)
      }
      .overlay { settingsDismissLayer }
      .overlayPreferenceValue(SettingsButtonAnchorKey.self) { anchor in
        settingsMenu(below: anchor)
      }

      if let preview = viewModel.activePreview {
        CapturePreviewView(
          preview: preview,
          onDismiss: { viewModel.dismissCapturePreview() }
        )
        .transition(.opacity)
        .zIndex(1)
      }
    }
    .environment(\.colorScheme, .dark)
    .animation(.spring(response: 0.42, dampingFraction: 0.86), value: viewModel.hasSession)
    .animation(.easeInOut(duration: 0.2), value: previewIsActive)
    .animation(.easeInOut(duration: 0.2), value: viewModel.isStreaming)
    .animation(.easeInOut(duration: 0.2), value: viewModel.isPaused)
    .animation(.easeInOut(duration: 0.2), value: viewModel.includeAudioInStream)
    .animation(.easeInOut(duration: 0.2), value: viewModel.isRecording)
    .animation(.easeInOut(duration: 0.2), value: viewModel.isComputerRecording)
    // While the full-screen video is up it shows this alert itself (a covered view
    // can't present), so this copy stands down.
    .alert("Something went wrong", isPresented: errorAlertBinding) {
      Button("OK") { viewModel.dismissError() }
    } message: {
      Text(viewModel.errorMessage)
    }
    .alert("Allow camera access", isPresented: $viewModel.showCameraPermissionRedirectConfirm) {
      Button("Continue") { Task { await viewModel.confirmCameraPermissionRedirect() } }
      Button("Cancel", role: .cancel) {}
    } message: {
      Text("To preview your glasses camera, you'll be taken to the Meta AI app to grant access, then returned here.")
    }
    .fullScreenCover(isPresented: $showFullScreenVideo) {
      FullScreenVideoView(viewModel: viewModel, isUpdateRequired: isUpdateRequired) {
        showFullScreenVideo = false
      }
    }
    // A finished recording's preview opens on this screen, so leave full screen for it.
    .onChange(of: viewModel.activePreview != nil) { _, hasPreview in
      if hasPreview { showFullScreenVideo = false }
    }
    .navigationBarHidden(true)
  }

  private var errorAlertBinding: Binding<Bool> {
    Binding(
      get: { viewModel.showError && !showFullScreenVideo },
      set: { viewModel.showError = $0 }
    )
  }

  // MARK: - Header

  private var header: some View {
    HStack(alignment: .center, spacing: 8) {
      StatusHeaderView(viewModel: viewModel) {
        withAnimation(.easeInOut(duration: 0.25)) { selectedCard = .computer }
      }
      Spacer(minLength: 0)
      settingsButton
    }
    .padding(.leading, 16)
    .padding(.trailing, 6)
    .padding(.vertical, 6)
  }

  /// App-level settings (Disconnect). The menu itself is drawn by `settingsMenu`, over
  /// the whole screen, so it isn't hidden behind the video below the header.
  private var settingsButton: some View {
    Button {
      withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
        showSettingsMenu.toggle()
      }
    } label: {
      Image(systemName: "gearshape")
        .font(.system(size: 20))
        .foregroundStyle(.white)
        .frame(width: 44, height: 44)
        .contentShape(Rectangle())
    }
    .accessibilityLabel("Settings")
    .accessibilityIdentifier("settings_button")
    .anchorPreference(key: SettingsButtonAnchorKey.self, value: .bounds) { $0 }
  }

  /// Tap outside the open settings menu dismisses it.
  @ViewBuilder
  private var settingsDismissLayer: some View {
    if showSettingsMenu {
      Color.clear
        .contentShape(Rectangle())
        .ignoresSafeArea()
        .onTapGesture {
          withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
            showSettingsMenu = false
          }
        }
    }
  }

  /// The Disconnect button, dropped below the gear's right edge.
  private func settingsMenu(below anchor: Anchor<CGRect>?) -> some View {
    GeometryReader { proxy in
      if showSettingsMenu, let anchor {
        let gear = proxy[anchor]
        CustomButton(
          title: "Disconnect",
          style: .destructive,
          isDisabled: wearablesVM.registrationState != .registered
        ) {
          wearablesVM.disconnectGlasses()
          withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) { showSettingsMenu = false }
        }
        .frame(width: 120)
        .offset(x: gear.maxX - 120 - 10, y: gear.maxY)
        .transition(.scale(scale: 0.01, anchor: .topTrailing).combined(with: .opacity))
      }
    }
  }

  // MARK: - Video, summary and cards

  private func middle(_ layout: MiddleLayout) -> some View {
    VStack(spacing: MiddleLayout.spacing) {
      HStack(alignment: .center, spacing: MiddleLayout.columnSpacing) {
        videoBox
          .frame(width: layout.videoWidth, height: layout.videoWidth * 16 / 9)
        LiveSummaryView(viewModel: viewModel)
          .frame(maxWidth: .infinity, maxHeight: .infinity)
      }
      .padding(.horizontal, MiddleLayout.horizontalInset)
      .frame(height: layout.topHeight)

      SensorCardsView(viewModel: viewModel, selection: $selectedCard)
        .frame(height: layout.cardsHeight)
    }
    .padding(.top, 4)
  }

  /// The glasses video (portrait, 9:16). Tapping it while there is a picture opens it
  /// full screen.
  private var videoBox: some View {
    VideoStageView(viewModel: viewModel, style: .compact, isUpdateRequired: isUpdateRequired)
      .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
      .contentShape(Rectangle())
      .onTapGesture {
        if viewModel.showsLivePreview { showFullScreenVideo = true }
      }
  }

  // MARK: - Bottom bar

  private var bottomBar: some View {
    VStack(spacing: 12) {
      if isUpdateRequired {
        updateControls
      } else {
        // Reserved capture-row space + a single persistent button hold the button at
        // a fixed Y across every state.
        captureRow
        anchoredPrimaryButton
      }
    }
    .frame(maxWidth: .infinity)
    .padding(.horizontal, 16)
    .padding(.top, 12)
    .padding(.bottom, 12)
    .background(Color.black)
    .overlay(alignment: .top) {
      Rectangle()
        .fill(Color.white.opacity(0.12))
        .frame(height: 0.5)
    }
  }

  /// In-session toolbar (Preview / Photo / Record / Mic), kept laid out so the
  /// session button below holds a fixed Y. Hidden without a session; Preview leads as
  /// the gateway and is the lone enabled control in session-ready, the rest enabling
  /// once streaming.
  private var captureRow: some View {
    HStack(spacing: 10) {
      previewPill
      photoButton
      recordPill
      micToggle
    }
    .opacity(viewModel.hasSession ? 1 : 0)
  }

  /// Photo needs a live stream.
  private var captureControlsEnabled: Bool { viewModel.isStreaming }
  /// Mic additionally locks once recording starts — the audio setting can't change
  /// mid-take.
  private var micEnabled: Bool { viewModel.isStreaming && !viewModel.isRecording }

  /// Photo capture as a compact icon (no label) so the Preview and Record pills keep
  /// room for text. Enabled while streaming, including during a recording.
  private var photoButton: some View {
    Button {
      viewModel.capturePhoto()
    } label: {
      Image(systemName: "camera.fill")
        .font(.system(size: 16, weight: .semibold))
        .foregroundStyle(.white)
        .frame(width: 50, height: 50)
        .background(.ultraThinMaterial)
        .clipShape(Circle())
    }
    .accessibilityLabel("Take photo")
    .accessibilityIdentifier("capture_button")
    .disabled(!captureControlsEnabled)
    .opacity(captureControlsEnabled ? 1 : 0.35)
  }

  /// Preview on/off — the gateway for capture. The eye icon flips and dims when off;
  /// locked mid-recording so a recording can't lose its feed.
  private var previewPill: some View {
    capturePill(
      icon: previewIsActive ? "eye.fill" : "eye.slash.fill",
      title: "Preview",
      contentTint: .white,
      id: previewIsActive ? "stop_preview_button" : "start_preview_button"
    ) {
      if previewIsActive {
        viewModel.stopStreaming()
      } else {
        Task { await viewModel.startStreaming() }
      }
    }
    .disabled(previewToggleDisabled)
    .opacity(previewToggleDisabled ? 0.35 : 1)
  }

  private var previewToggleDisabled: Bool {
    if previewIsActive {
      // A recording on either side needs the feed until it is stopped.
      return viewModel.isAnyRecordingActive || viewModel.isBusy
    }
    return !viewModel.isSessionActive || viewModel.isBusy
  }

  /// The single full-width session button: Start Session, or End Session
  /// (destructive) once a session exists. One instance, never branched, so it never
  /// remounts and holds a fixed Y — only its title / style / id / action change.
  private var anchoredPrimaryButton: some View {
    CustomButton(
      title: viewModel.hasSession ? "End session" : "Start session",
      style: viewModel.hasSession ? .destructive : .primary,
      isDisabled: primaryDisabled
    ) {
      if viewModel.hasSession {
        viewModel.endSession()
      } else {
        viewModel.startSession()
      }
    }
    .accessibilityIdentifier(viewModel.hasSession ? "end_session_button" : "start_session_button")
  }

  /// Start needs an active device; End stays available mid-stream/record — the SDK
  /// cascades the stop, so only an in-flight transition disables it.
  private var primaryDisabled: Bool {
    if viewModel.hasSession {
      return viewModel.isBusy
    }
    return viewModel.isBusy || !viewModel.hasActiveDevice
  }

  private func primaryButton(_ title: String, id: String, disabled: Bool = false, action: @escaping () -> Void) -> some View {
    CustomButton(title: title, style: .primary, isDisabled: disabled, action: action)
      .accessibilityIdentifier(id)
  }

  /// Frosted icon+label capsule for the capture actions. `contentTint` colors both
  /// the icon and label so on/off dims them together (see `previewPill`).
  private func capturePill(icon: String, title: String, contentTint: Color = .white, id: String, action: @escaping () -> Void) -> some View {
    Button(action: action) {
      HStack(spacing: 8) {
        Image(systemName: icon)
          .font(.system(size: 15, weight: .semibold))
          .foregroundStyle(contentTint)
        Text(title)
          .font(.system(size: 15, weight: .semibold))
          .foregroundStyle(contentTint)
          .lineLimit(1)
          .minimumScaleFactor(0.85)
      }
      .frame(maxWidth: .infinity)
      .frame(height: 50)
      .background(.ultraThinMaterial)
      .clipShape(Capsule())
    }
    .accessibilityIdentifier(id)
  }

  /// Record/Stop capsule, in its own view: it reads the computer's recorder, which
  /// changes every second while the computer records.
  private var recordPill: some View {
    RecordPill(viewModel: viewModel)
  }

  /// Sound-in-video toggle. Brightens on, dims/slashes when muted; locked until
  /// streaming and during recording. Denied reads as off and, since iOS won't
  /// re-prompt, tapping it opens Settings instead of toggling.
  private var micToggle: some View {
    let isOn = !viewModel.micDenied && viewModel.includeAudioInStream
    return Button {
      if viewModel.micDenied {
        if let url = URL(string: UIApplication.openSettingsURLString) {
          UIApplication.shared.open(url)
        }
      } else {
        viewModel.includeAudioInStream.toggle()
      }
    } label: {
      Image(systemName: isOn ? "mic.fill" : "mic.slash.fill")
        .font(.system(size: 16, weight: .semibold))
        .foregroundStyle(isOn ? .white : Color.white.opacity(0.45))
        .frame(width: 50, height: 50)
        .background(.ultraThinMaterial)
        .clipShape(Circle())
    }
    .disabled(!micEnabled)
    .opacity(micEnabled ? 1 : 0.35)
    .accessibilityLabel(isOn ? "Sound on" : "Sound off")
    .accessibilityIdentifier("mic_toggle")
  }

  // MARK: - Update controls

  @ViewBuilder
  private var updateControls: some View {
    UpdateRequiredMessage()
    // Gate the button while the external update flow launches so a double-tap can't
    // fire it twice; re-enabled once the await returns.
    primaryButton("Update firmware", id: "update_firmware_button", disabled: isLaunchingUpdate) {
      isLaunchingUpdate = true
      Task {
        await wearablesVM.openFirmwareUpdate()
        isLaunchingUpdate = false
      }
    }
  }
}

// MARK: - Layout

/// Splits the height between the header and the controls into the video row and the
/// cards. The video row aims for about 42% of a tall phone's screen; on a short phone
/// it gives way first, so the cards keep at least ~170 pt and nothing scrolls.
private struct MiddleLayout {
  static let spacing: CGFloat = 10
  static let horizontalInset: CGFloat = 16
  static let columnSpacing: CGFloat = 12
  /// The tab row (32 pt + 8 pt gap) plus a card of at least ~170 pt.
  static let minCardsHeight: CGFloat = 210
  static let minTopHeight: CGFloat = 150
  /// Share of the middle area for the video row: on a 430×932 pt phone the middle is
  /// ~640 pt, so this is ~350 pt, about 42% of the screen.
  static let topShare: CGFloat = 0.55
  /// Keeps the summary column readable when the box would otherwise be wide.
  static let minSummaryWidth: CGFloat = 130

  let topHeight: CGFloat
  let cardsHeight: CGFloat
  let videoWidth: CGFloat

  init(size: CGSize) {
    let available = max(0, size.height - 4 - Self.spacing)
    let top = max(Self.minTopHeight, min(available * Self.topShare, available - Self.minCardsHeight))
    topHeight = top
    cardsHeight = max(0, available - top)
    let widthForVideo = size.width - 2 * Self.horizontalInset - Self.columnSpacing - Self.minSummaryWidth
    videoWidth = max(0, min(top * 9 / 16, widthForVideo))
  }
}

/// Where the gear sits, so the Disconnect menu can be drawn over the whole screen.
private struct SettingsButtonAnchorKey: PreferenceKey {
  static var defaultValue: Anchor<CGRect>? { nil }

  static func reduce(value: inout Anchor<CGRect>?, nextValue: () -> Anchor<CGRect>?) {
    value = value ?? nextValue()
  }
}

// MARK: - Record button

/// Record/Stop capsule. Morphs in place while recording (red fill and the live timer),
/// so the bar keeps its shape. A caption under the title says where a recording goes
/// ("computer + iPhone") or, when there is nowhere to record, why it is disabled.
/// Stop stays available while recording even if the stream is paused, so a pause can't
/// trap a recording; starting needs a live stream.
private struct RecordPill: View {
  var viewModel: CameraViewModel

  var body: some View {
    let control = viewModel.recordControl
    Button {
      viewModel.toggleRecording()
    } label: {
      HStack(spacing: 8) {
        Image(systemName: control.mode == .stop ? "stop.fill" : "video.fill")
          .font(.system(size: 15, weight: .semibold))
          .foregroundStyle(Color.recordAccent)
        VStack(alignment: .leading, spacing: 0) {
          title(control)
            .font(.system(size: 15, weight: .semibold))
            .monospacedDigit()
            .foregroundStyle(.white)
          if let caption = control.caption {
            // The words when they fit, otherwise a laptop and a phone glyph.
            ViewThatFits(in: .horizontal) {
              Text(verbatim: caption)
              if !control.destinationSymbols.isEmpty {
                Text(symbolCaption(control.destinationSymbols))
              }
            }
            .font(.caption2.weight(.medium))
            .foregroundStyle(control.captionIsWarning ? Color.orange : Color.white.opacity(0.65))
            .accessibilityLabel(Text(verbatim: caption))
            .accessibilityIdentifier("record_caption")
          }
        }
        .lineLimit(1)
        .minimumScaleFactor(0.75)
        // Dim the title of a disabled button, but keep a reason readable.
        .opacity(control.isEnabled || control.captionIsWarning ? 1 : 0.6)
      }
      .padding(.horizontal, 6)
      .frame(maxWidth: .infinity)
      .frame(height: 50)
      .background {
        if control.mode == .stop {
          ZStack {
            Capsule().fill(.ultraThinMaterial)
            Capsule().fill(Color.recordAccent.opacity(0.50))
          }
        } else {
          Capsule().fill(.ultraThinMaterial)
        }
      }
    }
    .accessibilityIdentifier("record_button")
    .accessibilityHint(control.accessibilityHint ?? "")
    .disabled(!control.isEnabled)
    .opacity(control.isEnabled || control.captionIsWarning ? 1 : 0.35)
  }

  private func symbolCaption(_ symbols: [String]) -> LocalizedStringKey {
    if symbols.count == 2 {
      return "\(Image(systemName: symbols[0])) + \(Image(systemName: symbols[1]))"
    }
    return "\(Image(systemName: symbols[0]))"
  }

  /// "Record", or while recording the elapsed time: this iPhone's clock when it records,
  /// otherwise the computer's.
  @ViewBuilder
  private func title(_ control: RecordControl) -> some View {
    if control.mode == .start {
      Text("Record")
    } else if viewModel.isRecording {
      // Counts from the actual recording start (first written frame), so it tracks the
      // saved clip and keeps advancing through a pause; shows 00:00 until that frame.
      // Identifier on the Text so UI tests find it regardless of the time value.
      RecordingClockText(start: viewModel.recordingStartDate, identifier: "recording_indicator")
    } else if case .recording(let state) = viewModel.computer.recorder {
      Text(verbatim: state.seconds.formattedRecordingTime)
        .accessibilityIdentifier("recording_indicator")
    } else {
      Text("Stop")
    }
  }
}

/// The elapsed recording time as `mm:ss`, ticking once a second from `start` (00:00
/// until the first frame is written). `identifier` goes on the `Text` itself, so UI
/// tests find it whatever the time.
struct RecordingClockText: View {
  let start: Date?
  let identifier: String

  var body: some View {
    if let start {
      TimelineView(.periodic(from: start, by: 1)) { context in
        text(context.date.timeIntervalSince(start))
      }
    } else {
      text(0)
    }
  }

  private func text(_ elapsed: TimeInterval) -> some View {
    Text(elapsed.formattedRecordingTime)
      .accessibilityIdentifier(identifier)
  }
}

// MARK: - Update banner

struct UpdateRequiredMessage: View {
  private let message = "Your glasses need a firmware update before Camera Access can start."

  var body: some View {
    HStack(alignment: .top, spacing: 12) {
      Image(systemName: "exclamationmark.triangle.fill")
        .resizable()
        .aspectRatio(contentMode: .fit)
        .foregroundStyle(Color.updateRequiredForeground)
        .frame(width: 24, height: 24)
        .accessibilityHidden(true)

      VStack(alignment: .leading, spacing: 4) {
        Text(updateRequiredTitle)
          .font(.system(size: 16, weight: .semibold))
          .foregroundStyle(Color.updateRequiredForeground)

        Text(message)
          .font(.system(size: 15))
          .foregroundStyle(Color.updateRequiredForeground)
          .fixedSize(horizontal: false, vertical: true)
      }

      Spacer(minLength: 0)
    }
    .padding(.all, 16)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(Color.updateRequiredBackground)
    .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
  }
}
