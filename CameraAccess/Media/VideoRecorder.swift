/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

import AVFoundation
import CoreMedia
import os

/// Outcome of finalizing a recording: a finished file, nothing recorded, or a
/// writer failure the caller can surface to the user.
enum RecordingStopResult {
  case completed(URL)
  case noRecording
  case failed
}

/// Records the glasses camera stream to a `.mov` in the temporary directory. Owns
/// the `AVAssetWriter` and delegates compressed video to `VideoCaptureHandler` and
/// (when enabled) glasses-mic (Bluetooth HFP) audio to `AudioCaptureHandler` on the same writer.
/// Thread-safe via `OSAllocatedUnfairLock`, which guards the non-Sendable AVFoundation state.
final class VideoRecorder: Sendable {
  private struct State {
    var assetWriter: AVAssetWriter?
    var audioCaptureHandler: AudioCaptureHandler?
    var videoCaptureHandler: VideoCaptureHandler?
    var outputURL: URL?
    var recordingStartTime: Date?
    var isRecording: Bool = false
    var isStarting: Bool = false
    var takeID = UUID()
    // Recording is gated by the shutter: frames are written only while this is true.
    var shouldAcceptNewFrames: Bool = false
    // Whether to record glasses-microphone (Bluetooth HFP) audio alongside video.
    var includeAudio: Bool = true
    // Forwarded to the audio handler; fires when the glasses mic isn't available at capture.
    var onGlassesAudioUnavailable: (@Sendable () -> Void)?
    var onFirstFrameWritten: (@Sendable (UInt64, Double, Double, Int, Int) -> Void)?
    var onAudioWritten: (@Sendable () -> Void)?
  }

  private let state = OSAllocatedUnfairLock(uncheckedState: State())

  private static let logger = Logger(subsystem: "com.meta.wearables.CameraAccess", category: "Recording")

  var isRecording: Bool {
    state.withLockUnchecked { $0.isRecording }
  }

  /// Wall-clock time the recording actually began (first written frame), or nil when
  /// not recording. Drives the on-screen timer so it counts continuously, including
  /// through a stream pause.
  var recordingStartDate: Date? {
    state.withLockUnchecked { $0.recordingStartTime }
  }

  var isAudioStreaming: Bool {
    state.withLockUnchecked { $0.audioCaptureHandler?.isAudioStreaming ?? false }
  }

  var isVideoStreaming: Bool {
    state.withLockUnchecked { $0.videoCaptureHandler?.isVideoStreaming ?? false }
  }

  init() {}

  // MARK: - Recording Control

  /// Stops accepting frames so the recording can be finalized.
  func prepareToStop() {
    state.withLockUnchecked { $0.shouldAcceptNewFrames = false }
  }

  /// Allows the next frame to auto-start a new recording, recording glasses-mic (Bluetooth
  /// HFP) audio alongside video when `includeAudio` is true. `onGlassesAudioUnavailable` fires
  /// if the glasses mic isn't available at capture, so the caller can reflect mic-off.
  func prepareToStart(
    includeAudio: Bool,
    onGlassesAudioUnavailable: @escaping @Sendable () -> Void,
    onFirstFrameWritten: @escaping @Sendable (UInt64, Double, Double, Int, Int) -> Void,
    onAudioWritten: @escaping @Sendable () -> Void
  ) {
    state.withLockUnchecked {
      $0.takeID = UUID()
      $0.shouldAcceptNewFrames = true
      $0.includeAudio = includeAudio
      $0.onGlassesAudioUnavailable = onGlassesAudioUnavailable
      $0.onFirstFrameWritten = onFirstFrameWritten
      $0.onAudioWritten = onAudioWritten
    }
  }

  private func startRecording(with sampleBuffer: CMSampleBuffer, includeAudio: Bool) {
    let takeID = state.withLockUnchecked { state -> UUID? in
      guard state.shouldAcceptNewFrames, !state.isRecording, !state.isStarting else { return nil }
      state.isStarting = true
      return state.takeID
    }

    guard let takeID else {
      return
    }
    defer { state.withLockUnchecked { $0.isStarting = false } }

    guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer) else {
      return
    }

    let dimensions = CMVideoFormatDescriptionGetDimensions(formatDescription)
    let width = Int(dimensions.width)
    let height = Int(dimensions.height)

    guard width > 0, height > 0 else { return }

    let timestamp = Date().timeIntervalSince1970
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("recording_\(timestamp).mov")

    do {
      let writer = try AVAssetWriter(outputURL: url, fileType: .mov)

      let startTime = Date()

      let firstFrameCallback = state.withLockUnchecked { $0.onFirstFrameWritten }
      let videoHandler = VideoCaptureHandler(writer: writer, width: width, height: height, sourceFormatHint: formatDescription) { phoneNs, pts, sourcePTS in
        firstFrameCallback?(phoneNs, pts, sourcePTS, width, height)
      }

      var audioHandler: AudioCaptureHandler?
      if includeAudio {
        let noopUnavailable: @Sendable () -> Void = {}
        let onGlassesAudioUnavailable =
          state.withLockUnchecked { $0.onGlassesAudioUnavailable } ?? noopUnavailable
        audioHandler = AudioCaptureHandler(
          writer: writer,
          recordingStartTime: startTime,
          onGlassesAudioUnavailable: onGlassesAudioUnavailable,
          onAudioWritten: state.withLockUnchecked { $0.onAudioWritten } ?? {}
        )
      }

      if writer.startWriting() {
        writer.startSession(atSourceTime: .zero)

        let accepted = state.withLockUnchecked { state -> Bool in
          guard state.shouldAcceptNewFrames, state.takeID == takeID else { return false }
          state.assetWriter = writer
          state.videoCaptureHandler = videoHandler
          state.audioCaptureHandler = audioHandler
          state.outputURL = url
          state.recordingStartTime = startTime
          state.isRecording = true
          return true
        }
        if accepted {
          videoHandler.start()
          audioHandler?.start()
        } else {
          writer.cancelWriting()
          try? FileManager.default.removeItem(at: url)
        }
      } else {
        Self.logger.error("Asset writer failed to start writing")
        cleanup(takeID: takeID)
      }
    } catch {
      Self.logger.error("Failed to create asset writer: \(error.localizedDescription)")
      cleanup(takeID: takeID)
    }
  }

  /// Finalizes the recording: `.completed(url)`, `.noRecording`, or `.failed`. Returns
  /// the result directly (rather than stashing the URL) to avoid a read race.
  func stopRecording() async -> RecordingStopResult {
    // A keyframe can be constructing the writer while the user presses Stop.
    // Let that bounded setup finish before taking the writer snapshot.
    for _ in 0..<100 {
      if !state.withLockUnchecked({ $0.isStarting }) { break }
      try? await Task.sleep(for: .milliseconds(10))
    }
    let (shouldStop, url, videoHandler, audioHandler, writer) = state.withLockUnchecked { state -> (Bool, URL?, VideoCaptureHandler?, AudioCaptureHandler?, AVAssetWriter?) in
      guard state.isRecording else {
        return (false, nil, nil, nil, nil)
      }
      // Clear both flags atomically so a late frame can't auto-start a new recording
      // while this one finalizes.
      state.isRecording = false
      state.shouldAcceptNewFrames = false
      return (true, state.outputURL, state.videoCaptureHandler, state.audioCaptureHandler, state.assetWriter)
    }

    guard shouldStop else { return .noRecording }

    // Finish inputs under the same lock the frame thread appends on, so no late append
    // slips through to fail the writer. finishWriting() flushes the queue — no drain
    // delay, since passthrough appends are synchronous.
    videoHandler?.stop()
    audioHandler?.stop()

    guard let writer, let url else {
      cleanup()
      return .noRecording
    }

    await writer.finishWriting()
    let completed = writer.status == .completed
    if !completed {
      Self.logger.error("Asset writer did not finish cleanly (status: \(writer.status.rawValue))")
    }
    // Clear only if a newer recording hasn't replaced this one (a quick stop→start can).
    state.withLockUnchecked { state in
      guard state.assetWriter === writer else { return }
      state.assetWriter = nil
      state.videoCaptureHandler = nil
      state.audioCaptureHandler = nil
      state.outputURL = nil
      state.recordingStartTime = nil
    }
    return completed ? .completed(url) : .failed
  }

  // MARK: - Video Frame Handling

  func appendVideoFrame(_ sampleBuffer: CMSampleBuffer) {
    // Start on the first keyframe (IDR) so the video track opens on a decodable frame;
    // leading P-frames otherwise play back black.
    if !isRecording {
      let (shouldAccept, includeAudio) = state.withLockUnchecked { ($0.shouldAcceptNewFrames, $0.includeAudio) }
      guard shouldAccept, sampleBuffer.isHEVCKeyframe() else { return }
      startRecording(with: sampleBuffer, includeAudio: includeAudio)
    }
    let videoHandler = state.withLockUnchecked { $0.shouldAcceptNewFrames ? $0.videoCaptureHandler : nil }
    videoHandler?.appendVideoFrame(sampleBuffer)
  }

  // MARK: - Helpers

  private func cleanup(takeID: UUID? = nil) {
    state.withLockUnchecked { state in
      if let takeID, state.takeID != takeID { return }
      state.assetWriter = nil
      state.videoCaptureHandler = nil
      state.audioCaptureHandler = nil
      state.outputURL = nil
      state.recordingStartTime = nil
      // Stop accepting frames so a failed start doesn't retry on every frame.
      state.shouldAcceptNewFrames = false
    }
  }
}
