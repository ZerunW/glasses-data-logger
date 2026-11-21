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

/// One glasses input for the Inputs card: what happened, where, and when on the phone's
/// clock (the glasses' own millisecond timestamp goes to the logs, not the screen).
struct SensorInput: Identifiable {
  let id = UUID()
  let title: String
  let source: String
  /// An SF Symbol for the kind of input.
  let symbol: String
  let time: Date

  init(_ event: InputEvent, time: Date) {
    title = inputTitle(event)
    source = inputSourceName(event.source)
    symbol = inputSymbol(event)
    self.time = time
  }
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

/// "Navigation right · touchpad": the event and where it came from, for VoiceOver and logs.
func inputDescription(_ event: InputEvent) -> String {
  "\(inputTitle(event)) · \(inputSourceName(event.source))"
}

/// What happened, in words: "Navigation right", "Capture long press", "Drag move".
func inputTitle(_ event: InputEvent) -> String {
  switch event {
  case .nav(let direction, _, _): return "Navigation \(direction)"
  case .select: return "Select"
  case .back: return "Back"
  case .button(let button, _, _): return "Button \(button)"
  case .capture(let press, _, _):
    switch press {
    case .shortPress: return "Capture short press"
    case .hold: return "Capture long press"
    case .doublePress: return "Capture double press"
    @unknown default: return "Capture"
    }
  case .drag(let action, _, _, _, _, _, _): return "Drag \(action)"
  @unknown default: return "Unknown input"
  }
}

func inputSourceName(_ source: InputSource) -> String {
  switch source {
  case .captouch: return "touchpad"
  case .neuralBand: return "Neural Band"
  case .captureButton: return "capture button"
  case .actionButton: return "action button"
  case .neuralBandDrag: return "Neural Band drag"
  case .unknown: return "unknown"
  @unknown default: return String(describing: source)
  }
}

private func inputSymbol(_ event: InputEvent) -> String {
  switch event {
  case .nav(let direction, _, _):
    switch direction {
    case .up: return "arrow.up"
    case .down: return "arrow.down"
    case .left: return "arrow.left"
    case .right: return "arrow.right"
    @unknown default: return "arrow.up.and.down.and.arrow.left.and.right"
    }
  case .select: return "hand.tap"
  case .back: return "arrow.uturn.backward"
  case .button: return "button.programmable"
  case .capture: return "camera"
  case .drag: return "hand.draw"
  @unknown default: return "questionmark"
  }
}

/// The `type` and `detail` of an input event, shared by the phone's `inputs.csv` and the
/// computer link's INPUT message so both describe events the same way.
func inputClassification(_ event: InputEvent) -> (type: String, detail: String) {
  switch event {
  case .nav(let direction, _, _): return ("nav", String(describing: direction))
  case .select: return ("select", "")
  case .back: return ("back", "")
  case .button(let button, _, _): return ("button", String(describing: button))
  case .capture(let press, _, _): return ("capture", String(describing: press))
  case .drag(let action, let x, let y, let dx, let dy, _, _):
    return ("drag", "\(action) x=\(x) y=\(y) dx=\(dx) dy=\(dy)")
  @unknown default:
    return ("unknown", String(describing: event))
  }
}
