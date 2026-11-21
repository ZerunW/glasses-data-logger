//
// ScreenshotTourTests.swift
//
// Walks the camera screen through its main states with a mock device and a fake
// computer receiver (`FakeReceiver`), and attaches a full-screen screenshot of each, so
// the layout can be reviewed on any simulator: export them with `xcrun xcresulttool
// export attachments --path <xcresult> --output-path <dir>`. Asserts that each state is
// reached, including the record control flow with the computer.
//

import MWDATMockDeviceTestClient
import XCTest

@MainActor
final class ScreenshotTourTests: XCTestCase {
  private static let linkPort: UInt16 = 7913
  private let portFilePath = NSTemporaryDirectory() + "mwdat_test_server_port_\(UUID().uuidString).txt"
  private let app = XCUIApplication()
  // swiftlint:disable implicitly_unwrapped_optional
  private var mockClient: MockDeviceTestClient!
  // swiftlint:enable implicitly_unwrapped_optional
  private var deviceId: String?
  private var receiver: FakeReceiver?
  private var shotIndex = 0

  override func setUp() async throws {
    continueAfterFailure = false
    addTeardownBlock { [portFilePath] in
      try? FileManager.default.removeItem(atPath: portFilePath)
    }
    // Start from Save on this iPhone off (the default), with a picked folder standing in
    // for one chosen in Files, and the link in USB mode on the fake receiver's port.
    app.launchArguments = [
      "--ui-testing", "-saveOnThisPhone", "NO", "-linkMode", "usb", "--ui-test-recording-folder", "Glasses Exports",
    ]
    app.launchEnvironment["MWDAT_TEST_SERVER_PORT_FILE"] = portFilePath
    app.launchEnvironment[UITestLink.environmentKey] = String(Self.linkPort)
    app.launch()
    mockClient = MockDeviceTestClient(portFilePath: portFilePath)
    let reachable = await mockClient.waitForServer(timeout: 10)
    XCTAssertTrue(reachable, "Test server should be running")
  }

  /// Synchronous on purpose; see `CameraAccessUITests.tearDown`.
  override func tearDown() {
    nonisolated(unsafe) let test = self
    MainActor.assumeIsolated { test.tearDownOnMainActor() }
    super.tearDown()
  }

  private func tearDownOnMainActor() {
    receiver?.disconnect()
    receiver = nil
    if let deviceId, let client = mockClient {
      waitForTeardownWork(timeout: 15) { _ = await client.unpairDevice(deviceId: deviceId) }
    }
    deviceId = nil
    app.terminate()
    mockClient = nil
  }

  func testScreenshotTour() async throws {
    // 1. Home screen (not registered).
    let connect = app.buttons["Connect my glasses"]
    XCTAssertTrue(connect.waitForExistence(timeout: 10))
    shoot("home")

    // 2. Registered, waiting for a device.
    connect.tap()
    XCTAssertTrue(app.staticTexts["Waiting for an active device"].waitForExistence(timeout: 15))
    shoot("waiting-for-device")

    // 3. Device ready.
    let pairedDevice = await mockClient.pairDevice()
    let paired = try XCTUnwrap(pairedDevice)
    deviceId = paired
    _ = await mockClient.setCameraFeed(deviceId: paired, resourceName: "plant", ext: "mp4")
    _ = await mockClient.setCapturedImage(deviceId: paired, resourceName: "plant", ext: "png")
    let startSession = app.buttons["start_session_button"]
    XCTAssertTrue(startSession.waitForExistence(timeout: 15))
    XCTAssertTrue(waitUntilEnabled(startSession))
    shoot("device-ready")

    // 4. Session started, no preview yet.
    startSession.tap()
    let startPreview = app.buttons["start_preview_button"]
    XCTAssertTrue(startPreview.waitForExistence(timeout: 15))
    XCTAssertTrue(waitUntilEnabled(startPreview))
    settle(1.5)
    shoot("session-started")

    // 5. Streaming, no computer and Save on this iPhone off: Record is disabled and says why.
    startPreview.tap()
    let allow = app.alerts.buttons["Continue"]
    if allow.waitForExistence(timeout: 10) { allow.tap() }
    XCTAssertTrue(app.buttons["stop_preview_button"].waitForExistence(timeout: 20))
    XCTAssertTrue(app.buttons["live_video"].waitForExistence(timeout: 15), "The video should be visible")
    let record = app.buttons["record_button"]
    let caption = app.staticTexts["record_caption"]
    XCTAssertTrue(waitForLabel(caption, containing: "nowhere to save"))
    XCTAssertFalse(record.isEnabled)
    settle(2)
    shoot("streaming-nowhere-to-save")

    // 6. The computer connects; not recording.
    let receiver = FakeReceiver(port: Self.linkPort)
    self.receiver = receiver
    receiver.start()
    XCTAssertTrue(receiver.waitUntilConnected(timeout: 20), "The app should accept the receiver")
    let computer = app.buttons["computer_status_button"]
    XCTAssertTrue(waitForLabel(computer, containing: "connected"), "label: \(computer.label)")
    XCTAssertTrue(waitUntilEnabled(record))
    settle(1.5)
    shoot("computer-connected")

    // 7. The cards, with inputs flowing.
    for _ in 0..<2 {
      _ = await mockClient.inputNav(deviceId: paired, direction: .right)
      _ = await mockClient.inputSelect(deviceId: paired)
      _ = await mockClient.inputBack(deviceId: paired)
    }
    _ = await mockClient.inputNav(deviceId: paired, direction: .down)
    for card in ["motion", "inputs", "glasses", "computer", "storage"] {
      let tab = app.buttons["card_tab_\(card)"]
      XCTAssertTrue(tab.waitForExistence(timeout: 5))
      tab.tap()
      settle(1.5)
      shoot("card-\(card)")
    }

    // 8. Save on this iPhone on, into the picked folder.
    let toggle = app.switches["save_on_phone_toggle"]
    XCTAssertTrue(toggle.waitForExistence(timeout: 5))
    toggle.tap()
    if (toggle.value as? String) != "1" { toggle.switches.firstMatch.tap() }
    XCTAssertTrue(waitForLabel(caption, containing: "computer + iPhone"))
    XCTAssertTrue(app.staticTexts["recording_folder_free_space"].waitForExistence(timeout: 10))
    settle(1)
    shoot("storage-on")

    // 9. Recording on both: Record sends record_start; the computer's state comes back.
    app.buttons["card_tab_computer"].tap()
    record.tap()
    XCTAssertTrue(receiver.waitForCommand("record_start", timeout: 10), "Record should send record_start")
    XCTAssertTrue(waitForLabel(computer, containing: ", recording "))
    let timer = app.staticTexts["recording_indicator"]
    XCTAssertTrue(timer.waitForExistence(timeout: 10))
    XCTAssertTrue(waitUntilTimerCounting(timer))
    settle(2)
    shoot("recording")

    // 10. Paused (single cap-touch tap), still recording.
    app.buttons["card_tab_motion"].tap()
    _ = await mockClient.captouchTap(deviceId: paired)
    XCTAssertTrue(app.staticTexts["Paused"].waitForExistence(timeout: 10))
    settle(1)
    shoot("paused")
    _ = await mockClient.captouchTap(deviceId: paired)
    XCTAssertTrue(waitForNonexistence(app.staticTexts["Paused"], timeout: 10))

    // 11. Full-screen video while recording.
    let video = app.buttons["live_video"]
    XCTAssertTrue(video.waitForExistence(timeout: 10))
    video.tap()
    let closeFull = app.buttons["close_fullscreen_button"]
    XCTAssertTrue(closeFull.waitForExistence(timeout: 10))
    settle(1.5)
    shoot("fullscreen")
    closeFull.tap()
    XCTAssertTrue(waitForNonexistence(closeFull, timeout: 10))

    // 12. Photo capture preview (during the recording).
    app.buttons["capture_button"].tap()
    let closePreview = app.buttons["close_preview_button"]
    XCTAssertTrue(closePreview.waitForExistence(timeout: 15))
    settle(1)
    shoot("photo-preview")
    closePreview.tap()
    XCTAssertTrue(waitForNonexistence(closePreview, timeout: 10))

    // 13. Stop: record_stop to the computer; the phone's clip, moved into the picked
    // folder, opens in the preview.
    record.tap()
    XCTAssertTrue(receiver.waitForCommand("record_stop", timeout: 10), "Stop should send record_stop")
    XCTAssertTrue(closePreview.waitForExistence(timeout: 20))
    settle(1.5)
    shoot("video-preview")
    closePreview.tap()
    XCTAssertTrue(waitForNonexistence(closePreview, timeout: 10))
    app.buttons["card_tab_storage"].tap()
    XCTAssertTrue(waitForLabel(app.descendants(matching: .any)["last_save_message"].firstMatch, containing: "Saved"))
    settle(1)
    shoot("storage-saved")

    // 14. A recording started on the computer: it shows here, the phone doesn't record.
    receiver.startRecordingFromComputer()
    app.buttons["card_tab_computer"].tap()
    XCTAssertTrue(waitForLabel(computer, containing: ", recording "))
    XCTAssertTrue(waitForLabel(caption, containing: "computer"))
    XCTAssertFalse(app.staticTexts["summary_recording_time"].exists)
    settle(2.5)
    shoot("recording-from-computer")

    // The Computer card's own Stop stops it too (it works without a glasses session).
    let cardStop = app.buttons["stop_computer_recording_button"]
    XCTAssertTrue(cardStop.waitForExistence(timeout: 5))
    cardStop.tap()
    XCTAssertTrue(receiver.waitForCommand("record_stop", count: 2, timeout: 10), "The card's Stop should send record_stop")
    XCTAssertTrue(waitForNonexistence(cardStop, timeout: 10))

    // 15. The link drops mid-recording: the computer's state is unknown.
    receiver.startRecordingFromComputer()
    XCTAssertTrue(waitForLabel(computer, containing: ", recording "))
    receiver.disconnect()
    XCTAssertTrue(waitForLabel(computer, containing: "recording state unknown"))
    settle(1.5)
    shoot("link-lost")

    // 16. End the session.
    app.buttons["end_session_button"].tap()
    XCTAssertTrue(app.buttons["start_session_button"].waitForExistence(timeout: 15))
    settle(1)
    shoot("session-ended")
  }

  // MARK: - Helpers

  private func shoot(_ name: String) {
    shotIndex += 1
    let attachment = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
    attachment.name = String(format: "%02d-%@", shotIndex, name)
    attachment.lifetime = .keepAlways
    add(attachment)
  }

  /// Lets animations and once-a-second readouts settle before a screenshot.
  private func settle(_ seconds: TimeInterval) {
    _ = XCTWaiter.wait(for: [XCTestExpectation(description: "settle")], timeout: seconds)
  }

  private func waitUntilEnabled(_ element: XCUIElement, timeout: TimeInterval = 15) -> Bool {
    let expectation = XCTNSPredicateExpectation(predicate: NSPredicate(format: "isEnabled == true"), object: element)
    return XCTWaiter.wait(for: [expectation], timeout: timeout) == .completed
  }

  private func waitForNonexistence(_ element: XCUIElement, timeout: TimeInterval) -> Bool {
    let expectation = XCTNSPredicateExpectation(predicate: NSPredicate(format: "exists == false"), object: element)
    return XCTWaiter.wait(for: [expectation], timeout: timeout) == .completed
  }

  private func waitForLabel(_ element: XCUIElement, containing text: String, timeout: TimeInterval = 10) -> Bool {
    let predicate = NSPredicate { object, _ in (object as? XCUIElement)?.label.contains(text) ?? false }
    let expectation = XCTNSPredicateExpectation(predicate: predicate, object: element)
    return XCTWaiter.wait(for: [expectation], timeout: timeout) == .completed
  }

  private func waitUntilTimerCounting(_ timer: XCUIElement) -> Bool {
    let predicate = NSPredicate { element, _ in
      guard let label = (element as? XCUIElement)?.label else { return false }
      let parts = label.split(separator: ":").compactMap { Int($0) }
      return parts.count == 2 && parts[0] * 60 + parts[1] > 0
    }
    let expectation = XCTNSPredicateExpectation(predicate: predicate, object: timer)
    return XCTWaiter.wait(for: [expectation], timeout: 10) == .completed
  }
}
