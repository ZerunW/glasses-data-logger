//
// RecordingLocation.swift
//
// Where finished phone recordings are kept: the app's own Documents/Recordings folder
// (shown in Files under On My iPhone › GlassesData), or a folder the user picked in Files
// (on the iPhone, in iCloud Drive or on a USB drive), remembered as a bookmark.
//
// A recording bound for a picked folder is written to the app's temporary folder first and
// moved into the folder when it ends, so a slow or removable destination never holds up
// the writer. When that move fails, the recording is kept in the app's own folder instead.
//

import Foundation

/// What the Storage card shows about the destination folder.
struct RecordingFolderInfo: Equatable, Sendable {
  var name: String
  /// Where the folder is: "On My iPhone › GlassesData", "iCloud Drive", a drive's name or "Files".
  var location: String
  var freeBytes: Int64?
  var isAppFolder: Bool
  /// Set when the picked folder can't be reached; recordings then go to the app's folder.
  var problem: String?
}

/// Where a finished recording ended up.
struct RecordingDelivery: Sendable {
  let directory: URL
  /// Nil when the recording reached the chosen folder; otherwise why it didn't.
  let failure: String?
}

enum RecordingLocation {
  static let saveOnPhoneKey = "saveOnThisPhone"
  static let folderBookmarkKey = "recordingFolderBookmark"

  // MARK: - Folders

  /// Documents/Recordings, created on demand.
  static func appFolder() throws -> URL {
    let documents = try FileManager.default.url(
      for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
    let folder = documents.appendingPathComponent("Recordings", isDirectory: true)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    return folder
  }

  /// Where recordings bound for a picked folder are written until they end.
  static func stagingFolder() throws -> URL {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("StagedRecordings", isDirectory: true)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    return folder
  }

  static var appFolderLocation: String {
    let name = Bundle.main.object(forInfoDictionaryKey: "CFBundleDisplayName") as? String
      ?? Bundle.main.object(forInfoDictionaryKey: "CFBundleName") as? String ?? "GlassesData"
    return "On My iPhone › \(name)"
  }

  // MARK: - Bookmarks

  /// A bookmark for a folder from the document picker, created while holding its scope.
  static func makeBookmark(for url: URL) throws -> Data {
    let accessing = url.startAccessingSecurityScopedResource()
    defer { if accessing { url.stopAccessingSecurityScopedResource() } }
    return try url.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil)
  }

  enum Resolution: Equatable, Sendable {
    /// `refreshedBookmark` replaces the stored one when it had gone stale.
    case folder(URL, refreshedBookmark: Data?)
    case unavailable(String)
  }

  static func resolve(_ bookmark: Data) -> Resolution {
    var isStale = false
    let url: URL
    do {
      url = try URL(resolvingBookmarkData: bookmark, options: [], relativeTo: nil, bookmarkDataIsStale: &isStale)
    } catch {
      return .unavailable("The chosen folder can't be found. Choose it again.")
    }
    guard isStale else { return .folder(url, refreshedBookmark: nil) }
    let accessing = url.startAccessingSecurityScopedResource()
    defer { if accessing { url.stopAccessingSecurityScopedResource() } }
    let refreshed = try? url.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil)
    return .folder(url, refreshedBookmark: refreshed)
  }

  // MARK: - Describing

  /// Name, kind and free space of a folder, read while holding its security scope.
  static func describe(_ url: URL, isAppFolder: Bool) -> RecordingFolderInfo {
    let accessing = isAppFolder ? false : url.startAccessingSecurityScopedResource()
    defer { if accessing { url.stopAccessingSecurityScopedResource() } }
    let keys: Set<URLResourceKey> = [
      .volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey, .isUbiquitousItemKey,
      .volumeIsInternalKey, .volumeLocalizedNameKey, .isDirectoryKey,
    ]
    let values = try? url.resourceValues(forKeys: keys)
    let free = values?.volumeAvailableCapacityForImportantUsage ?? values?.volumeAvailableCapacity.map(Int64.init)
    var info = RecordingFolderInfo(
      name: isAppFolder ? "Recordings" : url.lastPathComponent, location: appFolderLocation, freeBytes: free,
      isAppFolder: isAppFolder, problem: nil)
    if !isAppFolder {
      if values?.isUbiquitousItem == true {
        info.location = "iCloud Drive"
      } else if values?.volumeIsInternal == false {
        info.location = values?.volumeLocalizedName ?? "External drive"
      } else {
        info.location = "Files"
      }
      if values?.isDirectory != true {
        info.problem = "The chosen folder isn't available. Recordings go to \(appFolderLocation) › Recordings."
      }
    }
    return info
  }

  // MARK: - Moving

  /// Moves a finished recording into `folder` (a picked, security-scoped folder) with file
  /// coordination. If that fails it moves it into `fallback` instead (or leaves it where it
  /// is if there is none or even that fails) and reports why.
  static func deliver(_ item: URL, to folder: URL, fallback: URL?) -> RecordingDelivery {
    let accessing = folder.startAccessingSecurityScopedResource()
    defer { if accessing { folder.stopAccessingSecurityScopedResource() } }
    do {
      return RecordingDelivery(directory: try coordinatedMove(item, into: folder), failure: nil)
    } catch {
      let reason = error.localizedDescription
      if let fallback, let kept = try? coordinatedMove(item, into: fallback) {
        return RecordingDelivery(directory: kept, failure: reason)
      }
      return RecordingDelivery(directory: item, failure: reason)
    }
  }

  /// Moves any recordings left in the staging folder (the app was closed or crashed during
  /// a recording) into the app's folder, so they are never lost with the temporary folder.
  static func recoverStagedRecordings() {
    guard let staging = try? stagingFolder(), let destination = try? appFolder(),
      let items = try? FileManager.default.contentsOfDirectory(at: staging, includingPropertiesForKeys: nil)
    else { return }
    for item in items {
      _ = try? coordinatedMove(item, into: destination)
    }
  }

  private static func coordinatedMove(_ item: URL, into folder: URL) throws -> URL {
    let fileManager = FileManager.default
    var isDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: folder.path, isDirectory: &isDirectory), isDirectory.boolValue else {
      throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: folder.path])
    }
    let destination = uniqueDestination(for: item.lastPathComponent, in: folder)
    let coordinator = NSFileCoordinator(filePresenter: nil)
    var coordinationError: NSError?
    var outcome: Result<URL, Error> = .failure(CocoaError(.fileWriteUnknown))
    coordinator.coordinate(
      writingItemAt: item, options: .forMoving, writingItemAt: destination, options: .forReplacing,
      error: &coordinationError
    ) { source, target in
      do {
        try fileManager.moveItem(at: source, to: target)
        coordinator.item(at: source, didMoveTo: target)
        outcome = .success(target)
      } catch {
        // A move across volumes copies first; don't leave half a copy behind.
        if fileManager.fileExists(atPath: source.path), fileManager.fileExists(atPath: target.path) {
          try? fileManager.removeItem(at: target)
        }
        outcome = .failure(error)
      }
    }
    if let coordinationError { throw coordinationError }
    return try outcome.get()
  }

  private static func uniqueDestination(for name: String, in folder: URL) -> URL {
    var candidate = folder.appendingPathComponent(name, isDirectory: true)
    var suffix = 2
    while FileManager.default.fileExists(atPath: candidate.path) {
      candidate = folder.appendingPathComponent("\(name)_\(suffix)", isDirectory: true)
      suffix += 1
    }
    return candidate
  }
}

/// Keeps a picked folder's security scope open, for example while one of its videos plays
/// in the capture preview. Released when the holder goes away.
final class SecurityScopedAccess: @unchecked Sendable {
  private let url: URL
  private let accessing: Bool

  init(_ url: URL) {
    self.url = url
    accessing = url.startAccessingSecurityScopedResource()
  }

  deinit {
    if accessing { url.stopAccessingSecurityScopedResource() }
  }
}

extension Int64 {
  /// "12.3 GB free"-style size, in the units Files uses.
  var formattedStorage: String {
    ByteCountFormatter.string(fromByteCount: self, countStyle: .file)
  }
}
