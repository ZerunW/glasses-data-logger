//
// LiveSummaryView.swift
//
// The column beside the video: the handful of live numbers worth watching while
// capturing, in four captioned groups (video, recording, head, glasses) with fixed
// spacing, top-aligned with the video. Every row group is its own view reading only what
// it shows, so a 10 Hz orientation update or the once-a-second fps update re-renders a
// few labels, never the video or the controls. Unavailable values read "—".
//

import MWDATCamera
import MWDATCore
import SwiftUI

struct LiveSummaryView: View {
  var viewModel: CameraViewModel

  var body: some View {
    // At large text sizes the rows may outgrow the column; scroll inside the column
    // then, rather than clipping or pushing the rest of the screen.
    ViewThatFits(in: .vertical) {
      rows
      ScrollView { rows }
        .scrollIndicators(.hidden)
    }
  }

  private var rows: some View {
    VStack(alignment: .leading, spacing: 12) {
      SummarySection("Video") {
        ResolutionRow(viewModel: viewModel)
        FrameRateRow(viewModel: viewModel)
        LinkFramesRow(viewModel: viewModel)
      }
      SummarySection("Recording") {
        PhoneRecordingRow(viewModel: viewModel)
        ComputerRecordingRow(viewModel: viewModel)
      }
      SummarySection("Head") {
        MotionSummaryRows(sensorManager: viewModel.sensorManager)
      }
      SummarySection("Glasses") {
        DeviceSummaryRows(sensorManager: viewModel.sensorManager)
      }
    }
    .padding(.vertical, 2)
    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    .dynamicTypeSize(...DynamicTypeSize.accessibility1)
  }
}

/// A small caption over a group of rows.
private struct SummarySection<Content: View>: View {
  let title: String
  @ViewBuilder let content: Content

  init(_ title: String, @ViewBuilder content: () -> Content) {
    self.title = title
    self.content = content()
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 3) {
      Text(verbatim: title.uppercased())
        .font(.caption2.weight(.semibold))
        .tracking(0.8)
        .foregroundStyle(Color.white.opacity(0.38))
        .accessibilityAddTraits(.isHeader)
      VStack(alignment: .leading, spacing: 2) {
        content
      }
    }
  }
}

/// "—", shown wherever a value isn't available yet.
let unavailableText = "—"

// MARK: - Row

/// A small grey label on the left and a monospaced-digit value on the right. The value
/// scales down rather than truncating when the column is narrow.
private struct SummaryRow: View {
  let label: String
  let value: String
  var valueColor: Color = .white
  var identifier: String?

  var body: some View {
    SummaryRowLayout(label: label) {
      Text(verbatim: value)
        .foregroundStyle(valueColor)
    }
    .accessibilityIdentifier(identifier ?? "")
  }
}

private struct SummaryRowLayout<Value: View>: View {
  let label: String
  @ViewBuilder let value: Value

  var body: some View {
    HStack(alignment: .firstTextBaseline, spacing: 6) {
      Text(verbatim: label)
        .font(.caption2)
        .foregroundStyle(Color.white.opacity(0.6))
        .lineLimit(1)
        .fixedSize()
      Spacer(minLength: 2)
      value
        .font(.footnote.weight(.semibold))
        .monospacedDigit()
        .lineLimit(1)
        .minimumScaleFactor(0.6)
    }
    .accessibilityElement(children: .combine)
  }
}

// MARK: - Video

/// The size of the first decoded frame, or the configured resolution (dimmed) before
/// any frame arrives.
private struct ResolutionRow: View {
  var viewModel: CameraViewModel

  var body: some View {
    if let size = viewModel.firstFrameSize {
      SummaryRow(label: "Size", value: "\(Int(size.width))×\(Int(size.height))", identifier: "summary_resolution")
    } else {
      let configured = viewModel.streamResolution.videoFrameSize
      SummaryRow(
        label: "Size", value: "\(configured.width)×\(configured.height)",
        valueColor: Color.white.opacity(0.5), identifier: "summary_resolution")
    }
  }
}

/// Frames per second the glasses actually delivered over the last second.
private struct FrameRateRow: View {
  var viewModel: CameraViewModel

  var body: some View {
    let value = viewModel.measuredStreamFps.map { String(format: "%.1f fps", $0) } ?? unavailableText
    SummaryRow(label: "Rate", value: value, identifier: "summary_fps")
  }
}

/// This iPhone's recording: its clock while recording, "saving…" while the file is
/// finalized, "off" when Save on this iPhone is off.
private struct PhoneRecordingRow: View {
  var viewModel: CameraViewModel

  var body: some View {
    if viewModel.isRecording {
      SummaryRowLayout(label: "iPhone") {
        HStack(spacing: 4) {
          Circle()
            .fill(Color.recordAccent)
            .frame(width: 7, height: 7)
          // Not `recording_indicator`: that identifier stays unique to the Record button.
          RecordingClockText(start: viewModel.recordingStartDate, identifier: "summary_recording_time")
            .foregroundStyle(Color.recordAccent)
        }
      }
    } else if viewModel.isFinalizingRecording {
      SummaryRow(label: "iPhone", value: "saving…", valueColor: .yellow, identifier: "summary_recording_state")
    } else if viewModel.isPreparingRecording {
      SummaryRow(label: "iPhone", value: "starting…", valueColor: .yellow, identifier: "summary_recording_state")
    } else {
      SummaryRow(
        label: "iPhone", value: viewModel.saveOnPhone ? "ready" : "off", valueColor: Color.white.opacity(0.5),
        identifier: "summary_recording_state")
    }
  }
}

/// The computer's recording, from its RECORDER_STATE.
private struct ComputerRecordingRow: View {
  var viewModel: CameraViewModel

  var body: some View {
    switch viewModel.computer.recorder {
    case .recording(let state):
      SummaryRowLayout(label: "Computer") {
        HStack(spacing: 4) {
          Circle()
            .fill(Color.recordAccent)
            .frame(width: 7, height: 7)
          Text(verbatim: state.seconds.formattedRecordingTime)
            .foregroundStyle(Color.recordAccent)
        }
      }
      .accessibilityIdentifier("summary_computer_recording")
    case .unknown:
      SummaryRow(label: "Computer", value: "unknown", valueColor: .orange, identifier: "summary_computer_recording")
    case .idle:
      SummaryRow(
        label: "Computer", value: "ready", valueColor: Color.white.opacity(0.5), identifier: "summary_computer_recording")
    case .unavailable:
      SummaryRow(label: "Computer", value: unavailableText, identifier: "summary_computer_recording")
    }
  }
}

// MARK: - Motion

/// Head orientation from the latest motion sample, plus the received sample rate.
/// Both update at most 10 times a second (`SensorManager` limits display updates).
private struct MotionSummaryRows: View {
  var sensorManager: SensorManager

  var body: some View {
    let sample = sensorManager.latestSample
    // The SDK's orientation when the glasses send one; otherwise the phone's own estimate from
    // the accelerometer and gyroscope, whose yaw counts from the start (no magnetometer).
    let reported = eulerDegrees(sample?.orientation)
    let angles = reported ?? sensorManager.estimatedHeadAngles
    let relativeYaw = reported == nil && angles != nil
    VStack(alignment: .leading, spacing: 2) {
      SummaryRow(label: "Roll", value: Self.degrees(angles?.roll))
      SummaryRow(label: "Pitch", value: Self.degrees(angles?.pitch))
      SummaryRow(label: relativeYaw ? "Yaw (rel.)" : "Yaw", value: Self.degrees(angles?.yaw))
      SummaryRow(
        label: "IMU", value: sample == nil ? unavailableText : "\(sensorManager.samplesPerSecond)/s",
        identifier: "summary_imu_rate")
    }
    .accessibilityIdentifier("summary_orientation")
  }

  private static func degrees(_ value: Double?) -> String {
    guard let value else { return unavailableText }
    return String(format: "%+.1f°", value)
  }
}

// MARK: - Glasses and link

/// Battery and worn state from the latest `DeviceState`.
private struct DeviceSummaryRows: View {
  var sensorManager: SensorManager

  var body: some View {
    let state = sensorManager.latestDeviceState
    SummaryRowLayout(label: "Battery") {
      HStack(spacing: 2) {
        if state?.chargingState == .charging {
          Image(systemName: "bolt.fill")
            .imageScale(.small)
            .foregroundStyle(.green)
            .accessibilityLabel("Charging")
        }
        Text(verbatim: state?.batteryLevel.map { "\($0)%" } ?? unavailableText)
          .foregroundStyle(.white)
      }
    }
    .accessibilityIdentifier("summary_battery")
    SummaryRow(label: "Worn", value: state.map { $0.donState.wornText } ?? unavailableText, identifier: "summary_worn")
  }
}

/// Frames sent to and dropped by the computer link, while it is connected.
private struct LinkFramesRow: View {
  var viewModel: CameraViewModel

  var body: some View {
    let status = viewModel.linkStatus
    if status.phase == .connected {
      SummaryRow(
        label: "Sent/drop", value: "\(status.sentFrames) / \(status.droppedFrames)",
        // Frames skipped while waiting for the first keyframe are normal; a gap isn't.
        valueColor: status.gaps > 0 ? .orange : .white, identifier: "summary_link_frames")
    } else {
      SummaryRow(
        label: "Sent/drop", value: unavailableText, valueColor: Color.white.opacity(0.5), identifier: "summary_link_frames")
    }
  }
}
