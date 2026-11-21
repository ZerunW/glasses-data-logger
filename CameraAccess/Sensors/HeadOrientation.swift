import Foundation
import simd

/// How the glasses' IMU axes sit in the head frame (x forward, y left, z up).
///
/// The SDK documents units (m/s², rad/s) but not the axes, so this is measured from a recording
/// with known head motions: still, nods, side tilts, turns (see `Docs/PIPELINE.md`, IMU axes).
struct ImuAxes: Equatable {
  /// Rotation taking an IMU-frame vector into the head frame (a proper rotation, det +1).
  var headFromImu: simd_double3x3
  /// +1 when the accelerometer reads +g along "up" at rest (specific force), -1 when it reports
  /// the direction of gravity instead.
  var accelSign: Double

  static let identity = ImuAxes(headFromImu: matrix_identity_double3x3, accelSign: 1)
  /// The Ray-Ban Meta glasses. Placeholder until measured from the first real recording.
  static let glasses = identity
}

/// Head orientation from the accelerometer and gyroscope alone (Mahony complementary filter),
/// for glasses that send neither a magnetometer nor a fused orientation.
///
/// Roll and pitch are absolute (the accelerometer sees gravity). Yaw has no reference without a
/// magnetometer: it starts at 0 when the filter starts and drifts slowly with the gyroscope bias.
/// The offline pipeline estimates orientation again, more carefully, from the recorded raw data;
/// this is only for the live display.
struct HeadOrientationFilter {
  var axes: ImuAxes
  /// Proportional gain (rad/s per unit of gravity-direction error): how fast roll and pitch
  /// follow the accelerometer. Higher follows faster but lets head accelerations through.
  var kp = 1.0
  /// Integral gain: slowly learns the gyroscope bias on the roll and pitch axes.
  var ki = 0.05

  /// Head-to-world rotation (world z up, yaw 0 at start).
  private(set) var attitude: simd_quatd?
  private var integral = SIMD3<Double>(repeating: 0)
  private var lastTimeNs: Int64?
  /// Accelerometer units, fixed from the first sample: 9.81 for m/s², 1 for g.
  private var gravity: Double?

  init(axes: ImuAxes = .glasses) {
    self.axes = axes
  }

  mutating func reset() {
    attitude = nil
    integral = .zero
    lastTimeNs = nil
    gravity = nil
  }

  /// Feeds one sample. `timeNs` must increase; a gap over 0.25 s restarts the integration
  /// (the attitude stays and the accelerometer pulls it back).
  mutating func update(accelerometer: SIMD3<Double>?, gyroscope: SIMD3<Double>?, timeNs: Int64) {
    let accel = accelerometer.flatMap { a -> SIMD3<Double>? in
      let v = axes.accelSign * (axes.headFromImu * a)
      return v.allFinite ? v : nil
    }
    let gyro = gyroscope.flatMap { w -> SIMD3<Double>? in
      let v = axes.headFromImu * w
      return v.allFinite ? v : nil
    }
    if gravity == nil, let accel, length(accel) > 0.3 {
      gravity = length(accel) < 3 ? 1 : 9.81
    }

    guard let current = attitude else {
      // Start level with the measured gravity, yaw 0.
      if let accel, let gravity, Self.nearGravity(accel, gravity) {
        attitude = Self.levelAttitude(up: normalize(accel))
        lastTimeNs = timeNs
      }
      return
    }
    defer { lastTimeNs = timeNs }
    guard let last = lastTimeNs, timeNs > last else { return }
    let dt = Double(timeNs - last) / 1e9
    guard dt <= 0.25 else { return }

    var omega = gyro ?? .zero
    if let accel, let gravity, Self.nearGravity(accel, gravity) {
      let measuredUp = normalize(accel)
      let estimatedUp = current.inverse.act(SIMD3(0, 0, 1))
      let error = cross(measuredUp, estimatedUp)
      integral += ki * error * dt
      omega += kp * error + integral
    } else if gyro == nil {
      return
    }
    let rate = length(omega)
    var next = current
    if rate > 0 {
      next = current * simd_quatd(angle: rate * dt, axis: omega / rate)
    }
    attitude = simd_normalize(next)
  }

  /// Roll (+ = right ear down), pitch (+ = chin up), yaw (+ = turned left), in degrees.
  var angles: EulerDegrees? {
    guard let attitude else { return nil }
    let r = simd_matrix3x3(attitude)  // columns: head x, y, z axes in the world
    let forward = r.columns.0, left = r.columns.1, up = r.columns.2
    let roll = atan2(left.z, up.z)
    let pitch = asin(max(-1, min(1, forward.z)))
    let yaw = atan2(forward.y, forward.x)
    return EulerDegrees(roll: roll * 180 / .pi, pitch: pitch * 180 / .pi, yaw: yaw * 180 / .pi)
  }

  /// Ignore the accelerometer while the head accelerates hard (more than 30 % off gravity).
  private static func nearGravity(_ accel: SIMD3<Double>, _ gravity: Double) -> Bool {
    abs(length(accel) / gravity - 1) < 0.3
  }

  /// The head-to-world rotation with zero yaw whose world "up" is `up` in the head frame.
  private static func levelAttitude(up: SIMD3<Double>) -> simd_quatd {
    let roll = atan2(up.y, up.z)
    let pitchNoseDown = asin(max(-1, min(1, -up.x)))
    return simd_quatd(angle: pitchNoseDown, axis: SIMD3(0, 1, 0)) * simd_quatd(angle: roll, axis: SIMD3(1, 0, 0))
  }
}

private extension SIMD3 where Scalar == Double {
  var allFinite: Bool { x.isFinite && y.isFinite && z.isFinite }
}
