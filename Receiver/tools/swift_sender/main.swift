// A macOS stand-in for the iPhone app: drives the app's real ComputerLink with the
// compressed samples of an HEVC file, so the Python receiver can be tested end to end
// without a phone. Built by run.sh from the app's own sources.
//
//   swift_sender --usb [--port P] [--seconds N] VIDEO
//   swift_sender --wifi 'gdpair://v1?...' [--host H] [--seconds N] VIDEO

import AVFoundation
import CoreMedia
import Foundation
import os

struct Options {
  var mode = LinkMode.usb
  var port = LinkProtocol.defaultPort
  var uri: String?
  var host: String?
  var seconds = 10.0
  var video: String?

  static func parse(_ arguments: [String]) -> Options? {
    var options = Options()
    var iterator = arguments.dropFirst().makeIterator()
    while let argument = iterator.next() {
      switch argument {
      case "--usb": options.mode = .usb
      case "--wifi":
        options.mode = .wifi
        guard let uri = iterator.next() else { return nil }
        options.uri = uri
      case "--port":
        guard let value = iterator.next().flatMap(UInt16.init), value > 0 else { return nil }
        options.port = value
      case "--host": options.host = iterator.next()
      case "--seconds":
        guard let value = iterator.next().flatMap(Double.init), value > 0 else { return nil }
        options.seconds = value
      default:
        guard !argument.hasPrefix("-"), options.video == nil else { return nil }
        options.video = argument
      }
    }
    return options.video == nil ? nil : options
  }
}

struct Report {
  var everConnected = false
  var last = LinkStatus.off
}

func log(_ message: String) {
  print("[swift_sender] \(message)")
  fflush(stdout)
}

func describe(_ status: LinkStatus) -> String {
  let peer = status.peerName.map { " peer=\($0)" } ?? ""
  return "\(status.phase)\(peer) sent=\(status.sentFrames) dropped=\(status.droppedFrames): \(status.message)"
}

func loadSamples(_ path: String) async throws -> (samples: [CMSampleBuffer], frameRate: Double) {
  let asset = AVURLAsset(url: URL(fileURLWithPath: path))
  guard let track = try await asset.loadTracks(withMediaType: .video).first else {
    throw CocoaError(.fileReadCorruptFile)
  }
  let frameRate = Double(try await track.load(.nominalFrameRate))
  let reader = try AVAssetReader(asset: asset)
  let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
  reader.add(output)
  guard reader.startReading() else { throw reader.error ?? CocoaError(.fileReadUnknown) }
  var samples: [CMSampleBuffer] = []
  while let sample = output.copyNextSampleBuffer() {
    if CMSampleBufferGetNumSamples(sample) > 0 { samples.append(sample) }
  }
  guard reader.status == .completed, !samples.isEmpty else { throw reader.error ?? CocoaError(.fileReadCorruptFile) }
  return (samples, frameRate > 0 ? frameRate : 30)
}

/// The same sample with its presentation time moved by `offset`, for seamless looping.
func retimed(_ sample: CMSampleBuffer, by offset: CMTime) -> CMSampleBuffer? {
  var timing = CMSampleTimingInfo()
  guard CMSampleBufferGetSampleTimingInfo(sample, at: 0, timingInfoOut: &timing) == noErr else { return nil }
  timing.presentationTimeStamp = CMTimeAdd(timing.presentationTimeStamp, offset)
  if timing.decodeTimeStamp.isValid { timing.decodeTimeStamp = CMTimeAdd(timing.decodeTimeStamp, offset) }
  var copy: CMSampleBuffer?
  let status = CMSampleBufferCreateCopyWithNewTiming(
    allocator: kCFAllocatorDefault, sampleBuffer: sample, sampleTimingEntryCount: 1,
    sampleTimingArray: &timing, sampleBufferOut: &copy)
  return status == noErr ? copy : nil
}

guard let options = Options.parse(CommandLine.arguments) else {
  log("usage: swift_sender (--usb [--port P] | --wifi URI [--host H]) [--seconds N] VIDEO")
  exit(2)
}

var pairing: LinkPairing?
if options.mode == .wifi {
  guard let uri = options.uri, let parsed = LinkPairing(uri: uri) else {
    log("invalid pairing URI")
    exit(2)
  }
  pairing = parsed
  if let host = options.host {
    pairing = LinkPairing(
      name: parsed.name, hosts: [host], port: parsed.port, fingerprint: parsed.fingerprint, token: parsed.token)
  }
}

/// Feeds the samples to a real `ComputerLink` at the file's frame rate for `options.seconds`.
/// Returns the exit status: 0 when the link reached the connected state.
func stream(_ samples: [CMSampleBuffer], frameRate: Double, options: Options, pairing: LinkPairing?) -> Int32 {
  let interval = 1 / frameRate
  let loopDuration = CMTime(seconds: Double(samples.count) * interval, preferredTimescale: 1_000_000)
  log("\(samples.count) samples at \(frameRate) fps from \(options.video ?? "")")

  let report = OSAllocatedUnfairLock(initialState: Report())
  let link = ComputerLink(usbPort: options.port, videoFrameRate: UInt16(frameRate.rounded()))

  // Sent before the link connects: the link must cache it and send it after the handshake.
  link.sendJSON(.deviceState, [
    "t": MonotonicClock.nowNanoseconds(), "battery": 87, "charging": "notCharging", "don": "donned",
    "hinge": "open", "thermal": "normal", "link": "connected",
  ])

  link.start(
    mode: options.mode, pairing: pairing,
    hello: LinkHello(app: "swift_sender", device: "Mac", name: "Swift sender")
  ) { status in
    report.withLock {
      $0.last = status
      if status.phase == .connected { $0.everConnected = true }
    }
    log(describe(status))
  }

  let started = Date()
  var index = 0
  var loop: Int64 = 0
  var submitted = 0
  var nextFrame = started
  var nextMotion = started
  var motionCount = 0
  while Date().timeIntervalSince(started) < options.seconds {
    let now = Date()
    if now >= nextFrame {
      let offset = CMTimeMultiply(loopDuration, multiplier: Int32(loop))
      if let sample = retimed(samples[index], by: offset) {
        link.submitVideo(sample, phoneTimeNs: MonotonicClock.nowNanoseconds())
        submitted += 1
      }
      index += 1
      if index == samples.count {
        index = 0
        loop += 1
      }
      nextFrame += interval
    }
    if now >= nextMotion {
      let t = now.timeIntervalSince(started)
      if link.isConnected { link.noteMotionSample() }
      link.sendJSON(.motion, [
        "t": MonotonicClock.nowNanoseconds(), "gt": Int64(t * 1e9), "src": "glasses",
        "a": [sin(t), 9.81, cos(t)], "g": [0.01, 0.2, 0.0], "m": [20.0, -5.0, 40.0], "q": [1.0, 0.0, 0.0, 0.0],
      ])
      motionCount += 1
      if motionCount % 50 == 0 {
        link.sendJSON(.deviceState, [
          "t": MonotonicClock.nowNanoseconds(), "battery": 86, "charging": "notCharging", "don": "donned",
          "hinge": "open", "thermal": "normal", "link": "connected",
        ])
      }
      nextMotion += 0.1
    }
    let wait = min(nextFrame, nextMotion).timeIntervalSince(Date())
    if wait > 0 { Thread.sleep(forTimeInterval: wait) }
  }

  // Let the last STATS go out before closing.
  Thread.sleep(forTimeInterval: 1.2)
  let final = report.withLock { $0 }
  log("finished: submitted=\(submitted) glasses frames; last status: \(describe(final.last))")
  link.stop()
  Thread.sleep(forTimeInterval: 0.3)
  return final.everConnected ? 0 : 1
}

let (samples, frameRate) = try await loadSamples(options.video ?? "")
exit(stream(samples, frameRate: frameRate, options: options, pairing: pairing))
