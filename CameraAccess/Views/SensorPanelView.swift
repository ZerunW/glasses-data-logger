import Charts
import MWDATCore
import SwiftUI
import MWDATMotion

struct SensorPanelView: View {
  @Bindable var sensorManager: SensorManager
  let isRecording: Bool
  @State private var expanded = false

  var body: some View {
    DisclosureGroup("传感器面板", isExpanded: $expanded) {
      ScrollView {
        VStack(alignment: .leading, spacing: 12) {
          Picker("采样率", selection: $sensorManager.samplingRateHz) {
            ForEach([5, 10, 15, 24, 30, 60], id: \.self) { Text("\($0) Hz").tag($0) }
          }.disabled(isRecording)
          Text("Motion: \(sensorManager.motionState) · \(sensorManager.samplesPerSecond) 样本/秒")
          if let error = sensorManager.motionError { Text(error).foregroundStyle(.red) }
          graph("加速度 (m/s²)", sensorManager.acceleration, sensorManager.latestSample?.accelerometer)
          graph("陀螺仪 (rad/s)", sensorManager.gyroscope, sensorManager.latestSample?.gyroscope)
          graph("磁力计 (µT)", sensorManager.magnetometer, sensorManager.latestSample?.magnetometer)
          Text("姿态 (度): \(orientationText(sensorManager.latestSample?.orientation))")
          Text("Inputs: \(sensorManager.inputsState)")
          if let error = sensorManager.inputsError { Text(error).foregroundStyle(.red) }
          ForEach(sensorManager.inputs) { event in Text("\(event.time.formatted(date: .omitted, time: .standard))  \(event.text)").font(.caption) }
          if let state = sensorManager.latestDeviceState {
            Text("设备: \(state.linkState)")
            Text("电量: \(state.batteryLevel.map(String.init) ?? "无") · 充电: \(state.chargingState)")
            Text("佩戴: \(state.donState) · 镜腿: \(state.hingeState) · 温度: \(state.thermalLevel)")
          } else { Text("设备状态: 无") }
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
          LineMark(x: .value("相对当前时间 (秒)", seconds), y: .value("值", point.x), series: .value("轴", "X")).foregroundStyle(.red)
          LineMark(x: .value("相对当前时间 (秒)", seconds), y: .value("值", point.y), series: .value("轴", "Y")).foregroundStyle(.green)
          LineMark(x: .value("相对当前时间 (秒)", seconds), y: .value("值", point.z), series: .value("轴", "Z")).foregroundStyle(.blue)
        }
      }.chartXScale(domain: -10.0...0.0).frame(height: 90)
      Text("X 红 · Y 绿 · Z 蓝").foregroundStyle(.secondary)
    }
  }
  private func vectorText(_ value: MWDATMotion.Vector3?) -> String {
    guard let value else { return "无" }
    return String(format: "X %.2f  Y %.2f  Z %.2f", value.x, value.y, value.z)
  }
  private func orientationText(_ q: MWDATMotion.Quaternion?) -> String {
    guard let angles = eulerDegrees(q) else { return "无" }
    return String(format: "roll %.1f  pitch %.1f  yaw %.1f", angles.roll, angles.pitch, angles.yaw)
  }
}
