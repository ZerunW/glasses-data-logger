//
// SensorCardsView.swift
//
// The paged cards under the video: Motion (one chart at a time, at full card
// height), Inputs (recent glasses taps and presses), Glasses (device state), Computer
// (the link and the computer's recording) and Storage (recording on this iPhone). A row of capsule tabs and a paged `TabView` share one
// selection, so both tapping a tab and swiping change the card. Each card, and each
// fast-changing part of a card, is its own view so sensor updates re-render only it.
//

import Charts
import MWDATCore
import MWDATMotion
import SwiftUI

enum SensorCard: String, CaseIterable, Identifiable {
  case motion, inputs, glasses, computer, storage

  var id: Self { self }

  var title: String {
    switch self {
    case .motion: return "Motion"
    case .inputs: return "Inputs"
    case .glasses: return "Glasses"
    case .computer: return "Computer"
    case .storage: return "Storage"
    }
  }
}

struct SensorCardsView: View {
  var viewModel: CameraViewModel
  @Binding var selection: SensorCard

  var body: some View {
    VStack(spacing: 8) {
      CardTabBar(selection: $selection)
        .padding(.horizontal, 16)

      TabView(selection: $selection) {
        MotionCard(viewModel: viewModel)
          .cardChrome()
          .tag(SensorCard.motion)
        InputsCard(sensorManager: viewModel.sensorManager)
          .cardChrome()
          .tag(SensorCard.inputs)
        GlassesCard(sensorManager: viewModel.sensorManager)
          .cardChrome()
          .tag(SensorCard.glasses)
        ComputerCard(viewModel: viewModel)
          .cardChrome()
          .tag(SensorCard.computer)
        StorageCard(viewModel: viewModel)
          .cardChrome()
          .tag(SensorCard.storage)
      }
      .tabViewStyle(.page(indexDisplayMode: .never))
    }
  }
}

// MARK: - Tabs

private struct CardTabBar: View {
  @Binding var selection: SensorCard

  var body: some View {
    HStack(spacing: 5) {
      ForEach(SensorCard.allCases) { card in
        let isSelected = card == selection
        Button {
          withAnimation(.easeInOut(duration: 0.25)) { selection = card }
        } label: {
          Text(verbatim: card.title)
            .font(.footnote.weight(.semibold))
            .lineLimit(1)
            .minimumScaleFactor(0.7)
            .foregroundStyle(isSelected ? Color.white : Color.white.opacity(0.65))
            .padding(.horizontal, 4)
            .frame(maxWidth: .infinity)
            .frame(height: 32)
            .background(
              Capsule().fill(isSelected ? Color.appPrimary : Color.white.opacity(0.1))
            )
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
        .accessibilityIdentifier("card_tab_\(card.rawValue)")
      }
    }
    .dynamicTypeSize(...DynamicTypeSize.xxLarge)
  }
}

private extension View {
  /// The shared card surface: inset from the screen edges so neighbouring pages show
  /// a gap while swiping.
  func cardChrome() -> some View {
    self
      .padding(12)
      .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
      .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
      .padding(.horizontal, 16)
  }
}

// MARK: - Motion

private enum MotionChannel: String, CaseIterable, Identifiable {
  case accelerometer, gyroscope, magnetometer

  var id: Self { self }

  var title: String {
    switch self {
    case .accelerometer: return "Accel"
    case .gyroscope: return "Gyro"
    case .magnetometer: return "Mag"
    }
  }

  var unit: String {
    switch self {
    case .accelerometer: return "m/s²"
    case .gyroscope: return "rad/s"
    case .magnetometer: return "µT"
    }
  }
}

/// One chart at a time, chosen with a segmented control, so the chart gets the whole
/// card instead of three 90 pt strips.
private struct MotionCard: View {
  var viewModel: CameraViewModel
  @State private var channel: MotionChannel = .accelerometer

  /// Mag appears only once the glasses have sent a magnetometer value.
  private var channels: [MotionChannel] {
    viewModel.sensorManager.hasMagnetometer ? MotionChannel.allCases : [.accelerometer, .gyroscope]
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack(spacing: 8) {
        Picker("Sensor", selection: $channel) {
          ForEach(channels) { channel in
            Text(verbatim: channel.title).tag(channel)
          }
        }
        .pickerStyle(.segmented)
        .frame(maxWidth: 200)
        .accessibilityIdentifier("motion_channel_picker")

        Spacer(minLength: 4)
        SamplingRateMenu(viewModel: viewModel)
        SamplesPerSecondText(sensorManager: viewModel.sensorManager)
      }

      MotionChart(sensorManager: viewModel.sensorManager, channel: channel)
        .frame(maxHeight: .infinity)

      MotionFooter(sensorManager: viewModel.sensorManager, channel: channel)
    }
    .onChange(of: viewModel.sensorManager.hasMagnetometer) { _, hasMagnetometer in
      if !hasMagnetometer, channel == .magnetometer { channel = .accelerometer }
    }
  }
}

/// The IMU sampling rate. Locked while recording so one recording has one rate.
private struct SamplingRateMenu: View {
  var viewModel: CameraViewModel
  private static let rates = [5, 10, 15, 24, 30, 60]

  var body: some View {
    @Bindable var sensorManager = viewModel.sensorManager
    // One recording has one rate, on the phone and on the computer alike.
    let locked = viewModel.isAnyRecordingActive || viewModel.isPreparingRecording
    Menu {
      Picker("Sampling rate", selection: $sensorManager.samplingRateHz) {
        ForEach(Self.rates, id: \.self) { rate in
          Text(verbatim: "\(rate) Hz").tag(rate)
        }
      }
    } label: {
      HStack(spacing: 3) {
        Text(verbatim: "\(sensorManager.samplingRateHz) Hz")
          .monospacedDigit()
        Image(systemName: "chevron.up.chevron.down")
          .imageScale(.small)
      }
      .font(.footnote.weight(.semibold))
      .lineLimit(1)
      .fixedSize()
    }
    .disabled(locked)
    .opacity(locked ? 0.5 : 1)
    .accessibilityLabel("Sampling rate")
    .accessibilityValue(Text(verbatim: "\(sensorManager.samplingRateHz) Hz"))
    .accessibilityIdentifier("sampling_rate_menu")
  }
}

/// Samples actually received in the last second, next to the requested rate.
private struct SamplesPerSecondText: View {
  var sensorManager: SensorManager

  var body: some View {
    Text(verbatim: "\(sensorManager.samplesPerSecond)/s")
      .font(.caption.monospacedDigit())
      .foregroundStyle(.secondary)
      .lineLimit(1)
      .fixedSize()
      .accessibilityLabel(Text(verbatim: "\(sensorManager.samplesPerSecond) samples per second"))
      .accessibilityIdentifier("motion_samples_per_second")
  }
}

/// The last 10 seconds of one sensor, X/Y/Z in red/green/blue.
private struct MotionChart: View {
  var sensorManager: SensorManager
  let channel: MotionChannel

  var body: some View {
    let points = self.points
    let now = Double(sensorManager.windowNowNs)
    Chart {
      ForEach(points) { point in
        let seconds = (Double(point.timeNs) - now) / 1_000_000_000
        LineMark(x: .value("Time relative to now (s)", seconds), y: .value("Value", point.x), series: .value("Axis", "X"))
          .foregroundStyle(.red)
        LineMark(x: .value("Time relative to now (s)", seconds), y: .value("Value", point.y), series: .value("Axis", "Y"))
          .foregroundStyle(.green)
        LineMark(x: .value("Time relative to now (s)", seconds), y: .value("Value", point.z), series: .value("Axis", "Z"))
          .foregroundStyle(.blue)
      }
    }
    .chartXScale(domain: -10.0...0.0)
    .chartXAxis {
      AxisMarks(values: [-10.0, -5.0, 0.0]) { value in
        AxisGridLine()
        AxisValueLabel {
          if let seconds = value.as(Double.self) {
            Text(verbatim: seconds == 0 ? "now" : String(format: "%.0f s", seconds))
          }
        }
      }
    }
    .overlay {
      if points.isEmpty {
        Text("No motion data yet")
          .font(.caption)
          .foregroundStyle(.secondary)
      }
    }
    .accessibilityIdentifier("motion_chart")
  }

  private var points: [SensorPoint] {
    switch channel {
    case .accelerometer: return sensorManager.acceleration
    case .gyroscope: return sensorManager.gyroscope
    case .magnetometer: return sensorManager.magnetometer
    }
  }
}

/// The latest X/Y/Z (coloured like the chart lines, so they double as its legend),
/// the Motion capability's state and any error.
private struct MotionFooter: View {
  var sensorManager: SensorManager
  let channel: MotionChannel

  var body: some View {
    let vector = latest
    VStack(alignment: .leading, spacing: 2) {
      ViewThatFits(in: .horizontal) {
        HStack(spacing: 8) {
          values(vector)
          Spacer(minLength: 4)
          state
        }
        VStack(alignment: .leading, spacing: 2) {
          values(vector)
          state
        }
      }
      if let error = sensorManager.motionError {
        Text(verbatim: error)
          .font(.caption)
          .foregroundStyle(.red)
          .lineLimit(2)
          .accessibilityIdentifier("motion_error")
      }
    }
  }

  private var latest: MWDATMotion.Vector3? {
    switch channel {
    case .accelerometer: return sensorManager.latestSample?.accelerometer
    case .gyroscope: return sensorManager.latestSample?.gyroscope
    case .magnetometer: return sensorManager.latestSample?.magnetometer
    }
  }

  private func values(_ vector: MWDATMotion.Vector3?) -> some View {
    HStack(spacing: 8) {
      axis("X", vector.map { Double($0.x) }, .red)
      axis("Y", vector.map { Double($0.y) }, .green)
      axis("Z", vector.map { Double($0.z) }, .blue)
      Text(verbatim: channel.unit)
        .foregroundStyle(.secondary)
    }
    .font(.caption.monospacedDigit())
    .lineLimit(1)
    .fixedSize()
    .accessibilityElement(children: .combine)
    .accessibilityIdentifier("motion_latest_values")
  }

  private func axis(_ name: String, _ value: Double?, _ color: Color) -> some View {
    HStack(spacing: 3) {
      Text(verbatim: name)
        .fontWeight(.bold)
        .foregroundStyle(color)
      Text(verbatim: value.map { String(format: "%+.2f", $0) } ?? unavailableText)
    }
  }

  private var state: some View {
    Text(verbatim: "Motion \(sensorManager.motionState)")
      .font(.caption)
      .foregroundStyle(.secondary)
      .lineLimit(1)
      .accessibilityIdentifier("motion_state")
  }
}

// MARK: - Inputs

/// Recent taps, swipes and presses on the glasses, newest first.
private struct InputsCard: View {
  var sensorManager: SensorManager

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack(alignment: .firstTextBaseline) {
        Text("Recent inputs")
          .font(.subheadline.weight(.semibold))
        Spacer(minLength: 8)
        Text(verbatim: sensorManager.inputsState)
          .font(.caption)
          .foregroundStyle(.secondary)
          .lineLimit(1)
          .accessibilityIdentifier("inputs_state")
      }
      if let error = sensorManager.inputsError {
        Text(verbatim: error)
          .font(.caption)
          .foregroundStyle(.red)
          .lineLimit(2)
          .accessibilityIdentifier("inputs_error")
      }
      if sensorManager.inputs.isEmpty {
        Text("Taps and button presses on the glasses appear here.")
          .font(.footnote)
          .foregroundStyle(.secondary)
          .lineLimit(2)
          .minimumScaleFactor(0.85)
      } else {
        ScrollView {
          LazyVStack(alignment: .leading, spacing: 8) {
            ForEach(sensorManager.inputs) { event in
              InputRow(input: event)
            }
          }
        }
        .scrollIndicators(.hidden)
        .accessibilityIdentifier("inputs_list")
      }
    }
  }
}

/// One input: an icon for its kind, what happened, where it came from, and the phone's
/// clock time it arrived.
private struct InputRow: View {
  let input: SensorInput

  var body: some View {
    HStack(spacing: 10) {
      Image(systemName: input.symbol)
        .font(.caption.weight(.semibold))
        .foregroundStyle(Color.appPrimary)
        .frame(width: 26, height: 26)
        .background(Circle().fill(Color.appPrimary.opacity(0.18)))
        .accessibilityHidden(true)
      VStack(alignment: .leading, spacing: 1) {
        Text(verbatim: input.title)
          .font(.footnote.weight(.semibold))
        Text(verbatim: input.source)
          .font(.caption2)
          .foregroundStyle(.secondary)
      }
      .lineLimit(1)
      Spacer(minLength: 8)
      Text(input.time, format: .dateTime.hour().minute().second())
        .font(.caption.monospacedDigit())
        .foregroundStyle(.secondary)
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .accessibilityElement(children: .combine)
  }
}

// MARK: - Glasses

/// The glasses' own state, from the latest `DeviceState`, as six tiles that share the
/// card's height.
private struct GlassesCard: View {
  var sensorManager: SensorManager

  var body: some View {
    ViewThatFits(in: .vertical) {
      grid(minTileHeight: 46)
      ScrollView { grid(minTileHeight: 40) }
        .scrollIndicators(.hidden)
    }
    .accessibilityIdentifier("glasses_card")
  }

  private func grid(minTileHeight: CGFloat) -> some View {
    let state = sensorManager.latestDeviceState
    let battery = state?.batteryLevel
    let charging = state?.chargingState == .charging
    return Grid(horizontalSpacing: 8, verticalSpacing: 8) {
      GridRow {
        tile(
          "Battery", battery.map { "\($0)%" } ?? unavailableText, symbol: Self.batterySymbol(battery),
          tint: charging ? .green : (battery ?? 100) <= 20 ? .orange : .white, minHeight: minTileHeight)
        tile(
          "Charging", state.map { $0.chargingState.displayText } ?? unavailableText,
          symbol: charging ? "bolt.fill" : "bolt.slash", tint: charging ? .green : .white, minHeight: minTileHeight)
      }
      GridRow {
        tile(
          "Worn", state.map { $0.donState.wornText } ?? unavailableText, symbol: "eyeglasses", tint: .white,
          minHeight: minTileHeight)
        tile(
          "Hinge", state.map { $0.hingeState.displayText } ?? unavailableText, symbol: "arrow.left.and.right",
          tint: .white, minHeight: minTileHeight)
      }
      GridRow {
        tile(
          "Thermal", state.map { $0.thermalLevel.displayText } ?? unavailableText, symbol: "thermometer.medium",
          tint: Self.thermalTint(state?.thermalLevel), minHeight: minTileHeight)
        tile(
          "Link", state.map { $0.linkState.displayText } ?? unavailableText,
          symbol: "antenna.radiowaves.left.and.right", tint: .white, minHeight: minTileHeight)
      }
    }
  }

  private func tile(_ label: String, _ value: String, symbol: String, tint: Color, minHeight: CGFloat) -> some View {
    HStack(spacing: 10) {
      Image(systemName: symbol)
        .font(.system(size: 15, weight: .semibold))
        .foregroundStyle(tint.opacity(0.9))
        .frame(width: 22)
        .accessibilityHidden(true)
      VStack(alignment: .leading, spacing: 1) {
        Text(verbatim: label)
          .font(.caption2)
          .foregroundStyle(.secondary)
        Text(verbatim: value)
          .font(.subheadline.weight(.semibold))
          .monospacedDigit()
          .lineLimit(1)
          .minimumScaleFactor(0.75)
      }
      Spacer(minLength: 0)
    }
    .padding(.horizontal, 10)
    .frame(maxWidth: .infinity, minHeight: minHeight, maxHeight: .infinity, alignment: .leading)
    .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    .accessibilityElement(children: .combine)
  }

  private static func batterySymbol(_ level: Int?) -> String {
    guard let level else { return "battery.0percent" }
    switch level {
    case ..<13: return "battery.0percent"
    case ..<38: return "battery.25percent"
    case ..<63: return "battery.50percent"
    case ..<88: return "battery.75percent"
    default: return "battery.100percent"
    }
  }

  private static func thermalTint(_ level: ThermalLevel?) -> Color {
    switch level {
    case .moderate: return .yellow
    case .severe, .critical, .emergency, .shutdown: return .red
    default: return .white
    }
  }
}

// MARK: - Computer

/// The link controls. `ComputerLinkCard` draws no chrome of its own, so it sits in this
/// card without a second border; it scrolls when the card is short.
private struct ComputerCard: View {
  var viewModel: CameraViewModel

  var body: some View {
    ScrollView {
      ComputerLinkCard(viewModel: viewModel)
    }
    .scrollBounceBehavior(.basedOnSize)
  }
}

// MARK: - Readable device state

extension DonState {
  /// "worn" / "not worn", for the summary column and the Glasses card.
  var wornText: String {
    switch self {
    case .donned: return "worn"
    case .doffed: return "not worn"
    case .unknown: return "unknown"
    }
  }
}

private extension ChargingState {
  var displayText: String {
    switch self {
    case .charging: return "charging"
    case .notCharging: return "not charging"
    case .unknown: return "unknown"
    }
  }
}

private extension HingeState {
  var displayText: String {
    switch self {
    case .open: return "open"
    case .closed: return "closed"
    case .unknown: return "unknown"
    }
  }
}

private extension ThermalLevel {
  var displayText: String {
    switch self {
    case .none: return "normal"
    case .light: return "light"
    case .moderate: return "moderate"
    case .severe: return "severe"
    case .critical: return "critical"
    case .emergency: return "emergency"
    case .shutdown: return "shutdown"
    case .unknown: return "unknown"
    }
  }
}

private extension MWDATCore.LinkState {
  var displayText: String {
    switch self {
    case .connected: return "connected"
    case .connecting: return "connecting"
    case .disconnected: return "disconnected"
    }
  }
}
