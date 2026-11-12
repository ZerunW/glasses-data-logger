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

  func testCapturePressesHaveDistinctLabels() {
    XCTAssertTrue(inputDescription(.capture(.shortPress, source: .captureButton, timestampMs: 1)).contains("Short press"))
    XCTAssertTrue(inputDescription(.capture(.hold, source: .captureButton, timestampMs: 2)).contains("Long press"))
    XCTAssertTrue(inputDescription(.capture(.doublePress, source: .captureButton, timestampMs: 3)).contains("Double press"))
  }
}
