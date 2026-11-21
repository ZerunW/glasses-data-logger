import Foundation
import XCTest
@testable import GlassesData

/// Moving finished recordings into a picked folder, the fallback when that fails, and the
/// folder bookmark.
final class RecordingLocationTests: XCTestCase {
  private var root: URL!

  override func setUpWithError() throws {
    root = FileManager.default.temporaryDirectory.appendingPathComponent("RecordingLocationTests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: root)
  }

  func testDeliverMovesTheRecordingIntoTheFolder() throws {
    let staged = try makeRecording("2026-11-21_093001")
    let folder = try makeFolder("Picked")
    let delivery = RecordingLocation.deliver(staged, to: folder, fallback: try makeFolder("App"))
    XCTAssertNil(delivery.failure)
    XCTAssertEqual(delivery.directory.lastPathComponent, "2026-11-21_093001")
    XCTAssertEqual(delivery.directory.deletingLastPathComponent().standardizedFileURL, folder.standardizedFileURL)
    XCTAssertEqual(try Data(contentsOf: delivery.directory.appendingPathComponent("video.mov")), Data([1, 2, 3]))
    XCTAssertFalse(FileManager.default.fileExists(atPath: staged.path), "Nothing stays in the temporary folder")
  }

  func testDeliverNeverOverwritesAnExistingRecording() throws {
    let folder = try makeFolder("Picked")
    try FileManager.default.createDirectory(
      at: folder.appendingPathComponent("2026-11-21_093001"), withIntermediateDirectories: false)
    let delivery = RecordingLocation.deliver(try makeRecording("2026-11-21_093001"), to: folder, fallback: nil)
    XCTAssertNil(delivery.failure)
    XCTAssertEqual(delivery.directory.lastPathComponent, "2026-11-21_093001_2")
  }

  func testDeliverKeepsTheRecordingInTheAppFolderWhenTheMoveFails() throws {
    let staged = try makeRecording("2026-11-21_093001")
    let missing = root.appendingPathComponent("Unplugged drive")
    let appFolder = try makeFolder("App")
    let delivery = RecordingLocation.deliver(staged, to: missing, fallback: appFolder)
    XCTAssertNotNil(delivery.failure, "The failure is reported")
    XCTAssertEqual(delivery.directory.deletingLastPathComponent().standardizedFileURL, appFolder.standardizedFileURL)
    XCTAssertTrue(FileManager.default.fileExists(atPath: delivery.directory.appendingPathComponent("video.mov").path))
  }

  func testDeliverLeavesTheRecordingInPlaceWithoutAFallback() throws {
    let staged = try makeRecording("2026-11-21_093001")
    let delivery = RecordingLocation.deliver(staged, to: root.appendingPathComponent("missing"), fallback: nil)
    XCTAssertNotNil(delivery.failure)
    XCTAssertEqual(delivery.directory, staged)
    XCTAssertTrue(FileManager.default.fileExists(atPath: staged.path))
  }

  func testBookmarkResolvesToTheSameFolder() throws {
    let folder = try makeFolder("Picked")
    let bookmark = try RecordingLocation.makeBookmark(for: folder)
    guard case .folder(let url, _) = RecordingLocation.resolve(bookmark) else {
      return XCTFail("The bookmark should resolve")
    }
    XCTAssertEqual(url.resolvingSymlinksInPath().path, folder.resolvingSymlinksInPath().path)
    XCTAssertEqual(RecordingLocation.resolve(Data([1, 2, 3])), .unavailable("The chosen folder can't be found. Choose it again."))
  }

  func testDescribeReportsNameAndFreeSpace() throws {
    let folder = try makeFolder("Glasses Exports")
    let info = RecordingLocation.describe(folder, isAppFolder: false)
    XCTAssertEqual(info.name, "Glasses Exports")
    XCTAssertNil(info.problem)
    XCTAssertGreaterThan(try XCTUnwrap(info.freeBytes), 0)
    XCTAssertFalse(info.isAppFolder)

    let gone = RecordingLocation.describe(root.appendingPathComponent("gone"), isAppFolder: false)
    XCTAssertNotNil(gone.problem)
  }

  func testRecoversRecordingsLeftInTheStagingFolder() throws {
    let staging = try RecordingLocation.stagingFolder()
    let name = "orphan-\(UUID().uuidString)"
    let orphan = staging.appendingPathComponent(name)
    try FileManager.default.createDirectory(at: orphan, withIntermediateDirectories: true)
    RecordingLocation.recoverStagedRecordings()
    let recovered = try RecordingLocation.appFolder().appendingPathComponent(name)
    defer { try? FileManager.default.removeItem(at: recovered) }
    XCTAssertFalse(FileManager.default.fileExists(atPath: orphan.path))
    XCTAssertTrue(FileManager.default.fileExists(atPath: recovered.path))
  }

  // MARK: - Helpers

  private func makeFolder(_ name: String) throws -> URL {
    let folder = root.appendingPathComponent(name, isDirectory: true)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    return folder
  }

  private func makeRecording(_ name: String) throws -> URL {
    let staged = try makeFolder("Staging").appendingPathComponent(name, isDirectory: true)
    try FileManager.default.createDirectory(at: staged, withIntermediateDirectories: false)
    try Data([1, 2, 3]).write(to: staged.appendingPathComponent("video.mov"))
    try Data("{}".utf8).write(to: staged.appendingPathComponent("info.json"))
    return staged
  }
}
