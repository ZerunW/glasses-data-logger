import XCTest
import MWDATInputs
import MWDATMotion
@testable import GlassesData

final class SensorModelsTests: XCTestCase {
  func testQuaternionIsNormalizedBeforeConversion() throws {
    let unit = try XCTUnwrap(eulerDegrees(Quaternion(x: 0, y: 0, z: 0.70710677, w: 0.70710677)))
    let scaled = try XCTUnwrap(eulerDegrees(Quaternion(x: 0, y: 0, z: 7.0710677, w: 7.0710677)))
    XCTAssertEqual(unit.yaw, 90, accuracy: 0.001)
    XCTAssertEqual(scaled.yaw, unit.yaw, accuracy: 0.001)
  }

  func testInvalidQuaternionHasNoOrientation() {
    XCTAssertNil(eulerDegrees(Quaternion(x: 0, y: 0, z: 0, w: 0)))
    XCTAssertNil(eulerDegrees(Quaternion(x: .nan, y: 0, z: 0, w: 1)))
    XCTAssertNil(eulerDegrees(nil))
  }

  func testInputsReadWithoutRawTimestamps() {
    let event = InputEvent.nav(direction: .right, source: .captouch, timestampMs: 1_795_240_066_761)
    XCTAssertEqual(inputDescription(event), "Navigation right · touchpad")
    let input = SensorInput(event, time: Date())
    XCTAssertEqual(input.title, "Navigation right")
    XCTAssertEqual(input.source, "touchpad")
    XCTAssertFalse(input.title.contains("1795240066761"))
  }

  func testCapturePressesHaveDistinctLabels() {
    XCTAssertTrue(inputDescription(.capture(.shortPress, source: .captureButton, timestampMs: 1)).contains("short press"))
    XCTAssertTrue(inputDescription(.capture(.hold, source: .captureButton, timestampMs: 2)).contains("long press"))
    XCTAssertTrue(inputDescription(.capture(.doublePress, source: .captureButton, timestampMs: 3)).contains("double press"))
  }
}
