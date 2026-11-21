import Foundation
import MWDATCore
import MWDATInputs
import MWDATMotion

/// GDL1 JSON bodies for the glasses sensors. The fields mirror the phone's own CSV logs
/// in `RecordingStore`. Kept apart from `ComputerLink` so the transport builds without MWDAT.
extension ComputerLink {
  func submitMotion(_ sample: MotionSample, phoneTimeNs: UInt64) {
    guard isConnected else { return }
    noteMotionSample()
    var body: [String: Any] = [
      "t": phoneTimeNs, "gt": sample.timestampNs, "src": String(describing: sample.source),
    ]
    if let value = sample.accelerometer, let numbers = Self.jsonNumbers([value.x, value.y, value.z]) { body["a"] = numbers }
    if let value = sample.gyroscope, let numbers = Self.jsonNumbers([value.x, value.y, value.z]) { body["g"] = numbers }
    if let value = sample.magnetometer, let numbers = Self.jsonNumbers([value.x, value.y, value.z]) { body["m"] = numbers }
    if let value = sample.orientation, let numbers = Self.jsonNumbers([value.w, value.x, value.y, value.z]) {
      body["q"] = numbers
    }
    sendJSON(.motion, body)
  }

  func submitInput(_ event: InputEvent, phoneTimeNs: UInt64) {
    guard isConnected else { return }
    let classification = inputClassification(event)
    sendJSON(.input, [
      "t": phoneTimeNs, "gt_ms": event.timestampMs, "src": String(describing: event.source),
      "type": classification.type, "detail": classification.detail,
    ])
  }

  /// Always encoded, even while disconnected, so the latest state is ready for the next handshake.
  func submitDeviceState(_ state: DeviceState, phoneTimeNs: UInt64) {
    var body: [String: Any] = [
      "t": phoneTimeNs,
      "charging": String(describing: state.chargingState),
      "don": String(describing: state.donState),
      "hinge": String(describing: state.hingeState),
      "thermal": String(describing: state.thermalLevel),
      "link": String(describing: state.linkState),
    ]
    if let battery = state.batteryLevel { body["battery"] = battery }
    sendJSON(.deviceState, body)
  }

  /// Converts through the shortest decimal form so a Float like 0.1 is sent as 0.1, as in
  /// the CSV logs, not as 0.10000000149011612. Nil when any value is NaN or infinite.
  private static func jsonNumbers(_ values: [Float]) -> [Double]? {
    guard values.allSatisfy(\.isFinite) else { return nil }
    return values.map { Double(String(describing: $0)) ?? Double($0) }
  }
}
