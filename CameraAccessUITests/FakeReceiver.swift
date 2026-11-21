//
// FakeReceiver.swift
//
// A stand-in for the computer's receiver, for UI tests. The app under test listens for
// the computer on 127.0.0.1 (USB mode); the test runner, also in the simulator, connects
// to it there and speaks enough GDL1 to complete the handshake, keep the link alive with
// PINGs, read the phone's CONTROL commands and send RECORDER_STATE, like the receiver
// does: right after HELLO_ACK, on every change, and once a second while recording.
//
// Also holds the shared teardown helper for the UI test classes.
//

import Foundation
import Network
import XCTest

/// The USB port the app listens on in UI tests (see `GLASSESDATA_LINK_PORT` in the app),
/// kept off the default 7900 so a receiver running on the Mac can't connect to the app.
enum UITestLink {
  static let environmentKey = "GLASSESDATA_LINK_PORT"
}

final class FakeReceiver: @unchecked Sendable {
  struct Recorder: Equatable {
    var recording = false
    var name: String?
    var seconds = 0.0
    var frames = 0
    var note: String?
  }

  private let port: NWEndpoint.Port
  private let queue = DispatchQueue(label: "GlassesDataUITests.FakeReceiver")
  private let lock = NSLock()
  // Touched only on `queue`.
  private var connection: NWConnection?
  private var buffer = Data()
  private var timer: DispatchSourceTimer?
  private var recordingCount = 0
  // Guarded by `lock`.
  private var handshaken = false
  private var closed = false
  private var receivedCommands: [String] = []
  private var recorder = Recorder()

  init(port: UInt16) {
    self.port = NWEndpoint.Port(rawValue: port)!
  }

  var isConnected: Bool { lock.withLock { handshaken && !closed } }
  var commands: [String] { lock.withLock { receivedCommands } }
  var currentRecorder: Recorder { lock.withLock { recorder } }

  /// Connects to the app, retrying until its listener accepts.
  func start() {
    queue.async { self.attempt() }
  }

  /// Waits (spinning the run loop) until the phone has completed the handshake.
  func waitUntilConnected(timeout: TimeInterval) -> Bool {
    let predicate = NSPredicate { object, _ in (object as? FakeReceiver)?.isConnected == true }
    let expectation = XCTNSPredicateExpectation(predicate: predicate, object: self)
    return XCTWaiter.wait(for: [expectation], timeout: timeout) == .completed
  }

  /// Waits until the phone has sent `command` at least `count` times.
  func waitForCommand(_ command: String, count: Int = 1, timeout: TimeInterval) -> Bool {
    let predicate = NSPredicate { object, _ in
      ((object as? FakeReceiver)?.commands.filter { $0 == command }.count ?? 0) >= count
    }
    let expectation = XCTNSPredicateExpectation(predicate: predicate, object: self)
    return XCTWaiter.wait(for: [expectation], timeout: timeout) == .completed
  }

  /// A recording started on the computer itself (from its web page), not by the phone.
  func startRecordingFromComputer() {
    queue.async {
      self.beginRecording()
      self.sendState()
    }
  }

  /// Drops the connection, as when the cable is pulled or Wi-Fi goes away.
  func disconnect() {
    lock.withLock { closed = true }
    queue.sync {
      timer?.cancel()
      timer = nil
      connection?.cancel()
      connection = nil
    }
  }

  // MARK: - Connection

  private func attempt() {
    guard !lock.withLock({ closed || handshaken }) else { return }
    let next = NWConnection(host: "127.0.0.1", port: port, using: .tcp)
    connection = next
    next.stateUpdateHandler = { [weak self, weak next] state in
      guard let self, let next, self.connection === next else { return }
      switch state {
      case .ready:
        self.receive(on: next)
      case .failed, .waiting:
        next.cancel()
        self.connection = nil
        guard !self.lock.withLock({ self.handshaken }) else { return }
        self.queue.asyncAfter(deadline: .now() + 0.5) { self.attempt() }
      default:
        break
      }
    }
    next.start(queue: queue)
  }

  private func receive(on current: NWConnection) {
    current.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, isComplete, error in
      guard let self, self.connection === current else { return }
      if let data { self.buffer.append(data) }
      self.drainFrames()
      if error != nil || isComplete {
        current.cancel()
        self.connection = nil
        self.timer?.cancel()
        self.timer = nil
        let wasHandshaken = self.lock.withLock { () -> Bool in
          let was = self.handshaken
          self.handshaken = false
          return was
        }
        // Before the handshake the app may simply not be listening yet.
        if !wasHandshaken { self.queue.asyncAfter(deadline: .now() + 0.5) { self.attempt() } }
        return
      }
      self.receive(on: current)
    }
  }

  private func drainFrames() {
    while buffer.count >= 4 {
      let start = buffer.startIndex
      let length = buffer[start..<(start + 4)].reduce(0) { $0 << 8 | Int($1) }
      guard length >= 1, buffer.count >= 4 + length else { return }
      let type = buffer[start + 4]
      let payload = Data(buffer[(start + 5)..<(start + 4 + length)])
      buffer.removeSubrange(start..<(start + 4 + length))
      handle(type: type, payload: payload)
    }
  }

  private func handle(type: UInt8, payload: Data) {
    switch type {
    case 0x01:  // HELLO
      send(json: 0x02, ["ok": true, "protocol": 1, "receiver": "fake-receiver", "name": "Test Computer"])
      lock.withLock { handshaken = true }
      sendState()
      startTimer()
    case 0x50:  // CONTROL
      guard let body = try? JSONSerialization.jsonObject(with: payload) as? [String: Any],
        let command = body["cmd"] as? String
      else { return }
      lock.withLock { receivedCommands.append(command) }
      switch command {
      case "record_start":
        if currentRecorder.recording {
          lock.withLock { recorder.note = "already recording" }
        } else {
          beginRecording()
        }
        sendState()
        lock.withLock { recorder.note = nil }
      case "record_stop":
        lock.withLock { recorder.recording = false }
        sendState()
      default:
        break
      }
    default:
      break  // VIDEO, sensors, STATS, PONG: read and dropped.
    }
  }

  private func beginRecording() {
    recordingCount += 1
    lock.withLock {
      recorder = Recorder(
        recording: true, name: String(format: "2026-11-21_09300%d", recordingCount), seconds: 0, frames: 0, note: nil)
    }
  }

  /// PING every second, as the receiver does, so the phone's idle timeout never fires,
  /// and RECORDER_STATE every second while recording.
  private func startTimer() {
    let source = DispatchSource.makeTimerSource(queue: queue)
    source.schedule(deadline: .now() + 1, repeating: 1)
    source.setEventHandler { [weak self] in
      guard let self else { return }
      var ping = Data([0, 0, 0, 9, 0x40])
      let now = DispatchTime.now().uptimeNanoseconds
      withUnsafeBytes(of: now.bigEndian) { ping.append(contentsOf: $0) }
      self.connection?.send(content: ping, completion: .idempotent)
      let recording = self.lock.withLock { () -> Bool in
        guard self.recorder.recording else { return false }
        self.recorder.seconds += 1
        self.recorder.frames += 30
        return true
      }
      if recording { self.sendState() }
    }
    source.resume()
    timer = source
  }

  private func sendState() {
    let state = currentRecorder
    send(json: 0x51, [
      "recording": state.recording, "name": state.name ?? NSNull(), "seconds": state.seconds,
      "frames": state.frames, "note": state.note ?? NSNull(),
    ])
  }

  private func send(json type: UInt8, _ body: [String: Any]) {
    guard let payload = try? JSONSerialization.data(withJSONObject: body, options: [.sortedKeys]) else { return }
    var frame = Data()
    withUnsafeBytes(of: UInt32(payload.count + 1).bigEndian) { frame.append(contentsOf: $0) }
    frame.append(type)
    frame.append(payload)
    connection?.send(content: frame, completion: .idempotent)
  }
}

/// Runs async work that doesn't need the main actor, from a synchronous tearDown, and
/// waits for it with `XCTWaiter`, which keeps the run loop turning. A synchronous tearDown
/// is what keeps a failed UI test from hanging: XCTest runs tearDown from inside the
/// interrupted test's main-actor job, so an async tearDown (which needs that same actor)
/// could never start, and the runner waited forever.
func waitForTeardownWork(timeout: TimeInterval, _ work: @escaping @Sendable () async -> Void) {
  let done = XCTestExpectation(description: "teardown work")
  Task.detached {
    await work()
    done.fulfill()
  }
  _ = XCTWaiter().wait(for: [done], timeout: timeout)
}
