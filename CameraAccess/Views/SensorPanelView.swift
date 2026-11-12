import Charts
import MWDATCore
import SwiftUI
import MWDATMotion

struct SensorPanelView: View {
  @Bindable var sensorManager: SensorManager
  let isRecording: Bool
  @State private var expanded = false

  var body: some View {
    DisclosureGroup("Sensor Panel", isExpanded: $expanded) {
      ScrollView {
        VStack(alignment: .leading, spacing: 12) {
          Picker("Sampling rate", selection: $sensorManager.samplingRateHz) {
            ForEach([5, 10, 15, 24, 30, 60], id: \.self) { Text("\($0) Hz").tag($0) }
          }.disabled(isRecording)
          Text("Motion: \(sensorManager.motionState) · \(sensorManager.samplesPerSecond) samples/s")
          if let error = sensorManager.motionError { Text(error).foregroundStyle(.red) }
          graph("Acceleration (m/s²)", sensorManager.acceleration, sensorManager.latestSample?.accelerometer)
          graph("Gyroscope (rad/s)", sensorManager.gyroscope, sensorManager.latestSample?.gyroscope)
          graph("Magnetometer (µT)", sensorManager.magnetometer, sensorManager.latestSample?.magnetometer)
          Text("Orientation (degrees): \(orientationText(sensorManager.latestSample?.orientation))")
          Text("Inputs: \(sensorManager.inputsState)")
          if let error = sensorManager.inputsError { Text(error).foregroundStyle(.red) }
          ForEach(sensorManager.inputs) { event in Text("\(event.time.formatted(date: .omitted, time: .standard))  \(event.text)").font(.caption) }
          if let state = sensorManager.latestDeviceState {
            Text("Device: \(state.linkState)")
            Text("Battery: \(state.batteryLevel.map(String.init) ?? "None") · Charging: \(state.chargingState)")
            Text("Worn: \(state.donState) · Hinge: \(state.hingeState) · Thermal: \(state.thermalLevel)")
          } else { Text("Device status: None") }
        }.font(.caption).frame(maxWidth: .infinity, alignment: .leading)
      }.frame(maxHeight: 280)
    }
    .padding(10)
    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    .environment(\.colorScheme, .dark)
  }
  private func graph(_ title: String, _ points: [SensorPoint], _ latest: MWDATMotion.Vector3?) -> some View {
    let now = Double(sensorManager.windowNowNs)
    return VStack(alignment: .leading) {
      Text("\(title): \(vectorText(latest))")
      Chart {
        ForEach(points) { point in
          let seconds = (Double(point.timeNs) - now) / 1_000_000_000
          LineMark(x: .value("Time relative to now (s)", seconds), y: .value("Value", point.x), series: .value("Axis", "X")).foregroundStyle(.red)
          LineMark(x: .value("Time relative to now (s)", seconds), y: .value("Value", point.y), series: .value("Axis", "Y")).foregroundStyle(.green)
          LineMark(x: .value("Time relative to now (s)", seconds), y: .value("Value", point.z), series: .value("Axis", "Z")).foregroundStyle(.blue)
        }
      }.chartXScale(domain: -10.0...0.0).frame(height: 90)
      Text("X red · Y green · Z blue").foregroundStyle(.secondary)
    }
  }
  private func vectorText(_ value: MWDATMotion.Vector3?) -> String {
    guard let value else { return "None" }
    return String(format: "X %.2f  Y %.2f  Z %.2f", value.x, value.y, value.z)
  }
  private func orientationText(_ q: MWDATMotion.Quaternion?) -> String {
    guard let angles = eulerDegrees(q) else { return "None" }
    return String(format: "roll %.1f  pitch %.1f  yaw %.1f", angles.roll, angles.pitch, angles.yaw)
  }
}
