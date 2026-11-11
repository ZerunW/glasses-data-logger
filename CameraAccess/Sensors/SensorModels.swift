import Foundation
import MWDATInputs
import MWDATMotion

struct SensorPoint: Identifiable {
  let id = UUID()
  let timeNs: UInt64
  let x: Double
  let y: Double
  let z: Double
}

struct SensorInput: Identifiable {
  let id = UUID()
  let text: String
  let time: Date
}

struct EulerDegrees: Equatable {
  let roll: Double
  let pitch: Double
  let yaw: Double
}

func eulerDegrees(_ q: MWDATMotion.Quaternion?) -> EulerDegrees? {
  guard let q else { return nil }
  var x = Double(q.x), y = Double(q.y), z = Double(q.z), w = Double(q.w)
  let norm = sqrt(x*x + y*y + z*z + w*w)
  guard norm.isFinite, norm > 0 else { return nil }
  x /= norm; y /= norm; z /= norm; w /= norm
  let roll = atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
  let pitch = asin(max(-1, min(1, 2 * (w*y - z*x))))
  let yaw = atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
  return EulerDegrees(roll: roll * 180 / .pi, pitch: pitch * 180 / .pi, yaw: yaw * 180 / .pi)
}

func inputDescription(_ event: InputEvent) -> String {
  let detail: String
  switch event {
  case .nav(let direction, _, _): detail = "导航 \(direction)"
  case .select: detail = "选择"
  case .back: detail = "返回"
  case .button(let button, _, _): detail = "按钮 \(button)"
  case .capture(let press, _, _):
    switch press {
    case .shortPress: detail = "拍摄 短按"
    case .hold: detail = "拍摄 长按"
    case .doublePress: detail = "拍摄 双击"
    @unknown default: detail = "拍摄 未知"
    }
  case .drag(let action, let x, let y, let dx, let dy, _, _):
    detail = "拖动 \(action) x:\(x) y:\(y) dx:\(dx) dy:\(dy)"
  @unknown default: detail = "未知输入"
  }
  return "\(detail) · \(event.source) · \(event.timestampMs) ms"
}
