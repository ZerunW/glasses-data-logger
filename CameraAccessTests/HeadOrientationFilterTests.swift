import XCTest
import simd
@testable import GlassesData

/// Simulates the glasses' IMU for a known head motion and checks the live head-angle filter.
final class HeadOrientationFilterTests: XCTestCase {
  private let g = 9.81
  private let stepNs: Int64 = 10_000_000  // 100 Hz

  /// What an IMU fixed to the head reads at rest in `attitude` (head-to-world): +g along up.
  private func restingAccel(_ attitude: simd_quatd) -> SIMD3<Double> {
    attitude.inverse.act(SIMD3(0, 0, g))
  }

  /// Head-to-world rotation from roll (+ right ear down), pitch (+ chin up), yaw (+ left), degrees.
  private func attitude(roll: Double = 0, pitch: Double = 0, yaw: Double = 0) -> simd_quatd {
    let r = roll * .pi / 180, p = -pitch * .pi / 180, y = yaw * .pi / 180
    return simd_quatd(angle: y, axis: SIMD3(0, 0, 1)) * simd_quatd(angle: p, axis: SIMD3(0, 1, 0))
      * simd_quatd(angle: r, axis: SIMD3(1, 0, 0))
  }

  private func feedStill(_ filter: inout HeadOrientationFilter, _ q: simd_quatd, seconds: Double, from t0: Int64 = 0) -> Int64 {
    var t = t0
    for _ in 0..<Int(seconds * 100) {
      filter.update(accelerometer: restingAccel(q), gyroscope: .zero, timeNs: t)
      t += stepNs
    }
    return t
  }

  func testNoAnglesBeforeAnySample() {
    XCTAssertNil(HeadOrientationFilter(axes: .identity).angles)
  }

  func testStartsFromGravityWithZeroYaw() throws {
    for (roll, pitch) in [(0.0, 0.0), (20.0, 0.0), (0.0, -25.0), (15.0, 30.0)] {
      var filter = HeadOrientationFilter(axes: .identity)
      filter.update(accelerometer: restingAccel(attitude(roll: roll, pitch: pitch, yaw: 40)), gyroscope: .zero, timeNs: 1)
      let a = try XCTUnwrap(filter.angles)
      XCTAssertEqual(a.roll, roll, accuracy: 0.01)
      XCTAssertEqual(a.pitch, pitch, accuracy: 0.01)
      XCTAssertEqual(a.yaw, 0, accuracy: 0.01)  // no magnetometer: yaw counts from the start
    }
  }

  func testSignConventions() throws {
    // Chin up, right ear down, turning left: each positive, through the gyroscope alone.
    let cases: [(SIMD3<Double>, KeyPath<EulerDegrees, Double>)] = [
      (SIMD3(0, -0.5, 0), \.pitch),  // about the left axis, negative = nose up
      (SIMD3(0.5, 0, 0), \.roll),    // about the forward axis, positive = left ear up = right ear down
      (SIMD3(0, 0, 0.5), \.yaw),     // about up, positive = turn left
    ]
    for (rate, angle) in cases {
      var filter = HeadOrientationFilter(axes: .identity)
      filter.kp = 0; filter.ki = 0  // gyroscope only
      filter.update(accelerometer: SIMD3(0, 0, g), gyroscope: .zero, timeNs: 0)
      var t: Int64 = 0
      for _ in 0..<50 {  // 0.5 s at 0.5 rad/s = 14.3°
        t += stepNs
        filter.update(accelerometer: SIMD3(0, 0, g), gyroscope: rate, timeNs: t)
      }
      XCTAssertEqual(try XCTUnwrap(filter.angles)[keyPath: angle], 14.32, accuracy: 0.05)
    }
  }

  func testTurningIntegratesYawWhileRollAndPitchStayLevel() throws {
    var filter = HeadOrientationFilter(axes: .identity)
    var t = feedStill(&filter, attitude(), seconds: 0.5)
    for _ in 0..<100 {  // 1 s at 1 rad/s
      filter.update(accelerometer: SIMD3(0, 0, g), gyroscope: SIMD3(0, 0, 1), timeNs: t)
      t += stepNs
    }
    let a = try XCTUnwrap(filter.angles)
    XCTAssertEqual(a.yaw, 57.3, accuracy: 1.0)
    XCTAssertEqual(a.roll, 0, accuracy: 0.1)
    XCTAssertEqual(a.pitch, 0, accuracy: 0.1)
  }

  func testAccelerometerPullsRollAndPitchBackDespiteGyroBias() throws {
    // Starts level, then the head tilts while the gyroscope reports only a bias: the
    // accelerometer must bring roll and pitch to the true tilt.
    var filter = HeadOrientationFilter(axes: .identity)
    var t = feedStill(&filter, attitude(), seconds: 0.2)
    let tilted = attitude(roll: 10, pitch: 20)
    for _ in 0..<1500 {  // 15 s
      filter.update(accelerometer: restingAccel(tilted), gyroscope: SIMD3(0.01, -0.01, 0), timeNs: t)
      t += stepNs
    }
    let a = try XCTUnwrap(filter.angles)
    XCTAssertEqual(a.roll, 10, accuracy: 1.0)
    XCTAssertEqual(a.pitch, 20, accuracy: 1.0)
  }

  func testHardAccelerationIsIgnored() throws {
    var filter = HeadOrientationFilter(axes: .identity)
    var t = feedStill(&filter, attitude(), seconds: 0.2)
    for _ in 0..<50 {  // a 2 g sideways shove must not tilt the estimate
      filter.update(accelerometer: SIMD3(0, 2 * g, g), gyroscope: .zero, timeNs: t)
      t += stepNs
    }
    let a = try XCTUnwrap(filter.angles)
    XCTAssertEqual(a.roll, 0, accuracy: 0.01)
    XCTAssertEqual(a.pitch, 0, accuracy: 0.01)
  }

  func testAxesMapImuVectorsIntoTheHeadFrame() throws {
    // An IMU mounted with x up, y left, z backward (a 90° turn about the left axis), reporting
    // gravity direction instead of specific force.
    let headFromImu = simd_double3x3(columns: (SIMD3(0, 0, 1), SIMD3(0, 1, 0), SIMD3(-1, 0, 0)))
    var filter = HeadOrientationFilter(axes: ImuAxes(headFromImu: headFromImu, accelSign: -1))
    let level = SIMD3<Double>(-g, 0, 0)  // gravity direction (down = -x in this IMU)
    filter.update(accelerometer: level, gyroscope: .zero, timeNs: 0)
    var a = try XCTUnwrap(filter.angles)
    XCTAssertEqual(a.roll, 0, accuracy: 0.01)
    XCTAssertEqual(a.pitch, 0, accuracy: 0.01)
    // Turning left is about head +z = IMU +x.
    var t: Int64 = 0
    for _ in 0..<100 {
      t += stepNs
      filter.update(accelerometer: level, gyroscope: SIMD3(0.5, 0, 0), timeNs: t)
    }
    a = try XCTUnwrap(filter.angles)
    XCTAssertEqual(a.yaw, 28.6, accuracy: 0.5)
  }

  func testGapsAndOutOfOrderSamplesDoNotIntegrate() throws {
    var filter = HeadOrientationFilter(axes: .identity)
    filter.kp = 0; filter.ki = 0
    filter.update(accelerometer: SIMD3(0, 0, g), gyroscope: .zero, timeNs: 1_000_000_000)
    filter.update(accelerometer: SIMD3(0, 0, g), gyroscope: SIMD3(0, 0, 1), timeNs: 2_000_000_000)  // 1 s gap
    filter.update(accelerometer: SIMD3(0, 0, g), gyroscope: SIMD3(0, 0, 1), timeNs: 1_500_000_000)  // backwards
    XCTAssertEqual(try XCTUnwrap(filter.angles).yaw, 0, accuracy: 1e-9)
  }

  func testAccelerometerInGIsRecognised() throws {
    var filter = HeadOrientationFilter(axes: .identity)
    let q = attitude(roll: 12)
    filter.update(accelerometer: restingAccel(q) / g, gyroscope: .zero, timeNs: 0)
    XCTAssertEqual(try XCTUnwrap(filter.angles).roll, 12, accuracy: 0.01)
  }

  func testResetForgetsEverything() {
    var filter = HeadOrientationFilter(axes: .identity)
    filter.update(accelerometer: SIMD3(0, 0, g), gyroscope: .zero, timeNs: 0)
    filter.reset()
    XCTAssertNil(filter.angles)
  }
}
