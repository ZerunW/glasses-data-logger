import CoreMedia
import Darwin
import Foundation
import Network
import os

/// TetherCam IUCM server. All protocol and connection state is confined to queue.
final class CameraBroadcast: @unchecked Sendable {
  private let queue = DispatchQueue(label: "GlassesData.TetherCam", qos: .userInitiated)
  private var listener: NWListener?
  private var client: NWConnection?
  private var parser = IucmFrameParser()
  private var streaming = false
  private var waitingForKeyframe = true
  private var sending = false
  private var currentHVCC: Data?
  private var status: (@Sendable (String) -> Void)?
  private var approvalHandler: (@Sendable (UUID, String) -> Void)?
  private var pendingApproval: UUID?
  private var approved = false
  private var lastReceivedNs: UInt64 = 0
  private let wantsFrames = OSAllocatedUnfairLock(initialState: false)
  private struct FrameGate {
    var busy = false
    var dropped = false
  }
  private let frameGate = OSAllocatedUnfairLock(initialState: FrameGate())

  func start(allowWiFi: Bool, onStatus: @escaping @Sendable (String) -> Void, onApproval: @escaping @Sendable (UUID, String) -> Void) {
    queue.async { [self] in
      guard listener == nil else { return }
      status = onStatus
      approvalHandler = onApproval
      do {
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        if !allowWiFi {
          parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: 7878)
        }
        (parameters.defaultProtocolStack.transportProtocol as? NWProtocolTCP.Options)?.noDelay = true
        let listener = allowWiFi
          ? try NWListener(using: parameters, on: NWEndpoint.Port(rawValue: 7878)!)
          : try NWListener(using: parameters)
        self.listener = listener
        listener.stateUpdateHandler = { [weak self, weak listener] state in
          guard let self, let listener, self.listener === listener else { return }
          switch state {
          case .ready:
            let addresses = allowWiFi ? Self.localAddresses().map { "\($0):7878" }.joined(separator: "\n") : ""
            onStatus(addresses.isEmpty ? "Ready for USB" : "Ready for USB. Wi-Fi addresses:\n\(addresses)")
          case .failed(let error):
            onStatus("Broadcast failed: \(error.localizedDescription)")
            self.stopOnQueue()
          default: break
          }
        }
        listener.newConnectionHandler = { [weak self] connection in self?.accept(connection) }
        listener.start(queue: queue)
      } catch { onStatus("Broadcast failed: \(error.localizedDescription)") }
    }
  }

  func stop() { queue.async { self.stopOnQueue() } }

  private func stopOnQueue() {
    wantsFrames.withLock { $0 = false }
    client?.cancel(); client = nil
    listener?.cancel(); listener = nil
    pendingApproval = nil; approved = false; approvalHandler = nil
    streaming = false; sending = false; waitingForKeyframe = true; currentHVCC = nil
  }

  private func accept(_ connection: NWConnection) {
    guard listener != nil, client == nil else { connection.cancel(); return }
    client = connection
    approved = false
    let approvalID = UUID()
    pendingApproval = approvalID
    parser = IucmFrameParser()
    streaming = false; waitingForKeyframe = true; currentHVCC = nil; sending = false
    connection.stateUpdateHandler = { [weak self, weak connection] state in
      guard let self, let connection, self.client === connection else { return }
      switch state {
      case .failed, .cancelled:
        self.client = nil; self.streaming = false; self.sending = false; self.pendingApproval = nil; self.approved = false
        self.wantsFrames.withLock { $0 = false }
        self.status?("Disconnected. Waiting for TetherCam on port 7878.")
      default: break
      }
    }
    connection.start(queue: queue)
    approvalHandler?(approvalID, String(describing: connection.endpoint))
    // TetherCam reconnects if it does not receive HELLO within five seconds.
    queue.asyncAfter(deadline: .now() + 4.5) { [weak self, weak connection] in
      guard let self, let connection, self.client === connection, self.pendingApproval == approvalID else { return }
      self.pendingApproval = nil
      self.client = nil
      connection.cancel()
      self.status?("Connection approval timed out.")
    }
  }

  func resolveConnection(_ id: UUID, allow: Bool) {
    queue.async { [self] in
      guard pendingApproval == id, let connection = client else { return }
      pendingApproval = nil
      guard allow else {
        client = nil
        connection.cancel()
        status?("Connection rejected.")
        return
      }
      approved = true
      lastReceivedNs = MonotonicClock.nowNanoseconds()
      sendControl(.hello(version: 0x0102, deviceName: "GlassesData", appVersion: "1.0", cameras: [CameraDescriptor(id: 0, position: .back, name: "Meta Glasses")]), to: connection)
      receive(connection)
      checkActivity(connection)
    }
  }

  private func checkActivity(_ connection: NWConnection) {
    queue.asyncAfter(deadline: .now() + 10) { [weak self, weak connection] in
      guard let self, let connection, self.client === connection, self.approved else { return }
      let now = MonotonicClock.nowNanoseconds()
      if now >= self.lastReceivedNs, now - self.lastReceivedNs >= 10_000_000_000 {
        self.client = nil; self.streaming = false; self.approved = false
        self.wantsFrames.withLock { $0 = false }
        connection.cancel()
        self.status?("Connection timed out. Waiting for TetherCam.")
      } else {
        self.checkActivity(connection)
      }
    }
  }

  private func receive(_ connection: NWConnection) {
    connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self, weak connection] data, _, complete, error in
      guard let self, let connection, self.client === connection else { return }
      guard error == nil, !complete, let data, !data.isEmpty else {
        connection.cancel(); self.client = nil; self.streaming = false
        self.wantsFrames.withLock { $0 = false }; return
      }
      self.lastReceivedNs = MonotonicClock.nowNanoseconds()
      do {
        for message in try self.parser.feedMessages(data) {
          switch message {
          case .start:
            self.streaming = true; self.waitingForKeyframe = true; self.currentHVCC = nil
            self.wantsFrames.withLock { $0 = true }
            self.status?("TetherCam connected. Waiting for a camera keyframe.")
          case .stop:
            self.streaming = false; self.waitingForKeyframe = true
            self.wantsFrames.withLock { $0 = false }
          case .ping(let timestamp): self.sendControl(.pong(timestampUs: timestamp), to: connection)
          default: break
          }
        }
      } catch {
        connection.cancel(); self.client = nil; self.streaming = false
        self.wantsFrames.withLock { $0 = false }; return
      }
      self.receive(connection)
    }
  }

  private func sendControl(_ message: IucmMessage, to connection: NWConnection) {
    connection.send(content: IucmCodec.encode(message), completion: .contentProcessed { error in
      if error != nil { connection.cancel() }
    })
  }

  func submit(_ sample: CMSampleBuffer) {
    guard wantsFrames.withLock({ $0 }) else { return }
    let admission = frameGate.withLock { gate -> (Bool, Bool) in
      guard !gate.busy else { gate.dropped = true; return (false, false) }
      gate.busy = true
      let dropped = gate.dropped
      gate.dropped = false
      return (true, dropped)
    }
    guard admission.0 else { return }
    var enqueued = false
    defer {
      if !enqueued { frameGate.withLock { $0.busy = false; $0.dropped = true } }
    }
    guard let format = CMSampleBufferGetFormatDescription(sample),
      let atoms = CMFormatDescriptionGetExtension(format, extensionKey: kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms) as? [String: Any],
      let hvcc = atoms["hvcC"] as? Data,
      let buffer = CMSampleBufferGetDataBuffer(sample) else { return }
    let length = CMBlockBufferGetDataLength(buffer)
    guard length > 0, length <= Iucm.maxPayload - 8 else { return }
    var bytes = Data(count: length)
    let result = bytes.withUnsafeMutableBytes { raw in
      CMBlockBufferCopyDataBytes(buffer, atOffset: 0, dataLength: length, destination: raw.baseAddress!)
    }
    guard result == noErr, let units = HvccNal.split(bytes) else { return }
    let nal = HvccNal.join(units.filter { !$0.isEmpty && (($0[$0.startIndex] >> 1) & 0x3f) <= 31 })
    guard !nal.isEmpty else { return }
    let dimensions = CMVideoFormatDescriptionGetDimensions(format)
    let keyframe = sample.isHEVCKeyframe()
    let time = CMSampleBufferGetPresentationTimeStamp(sample).seconds
    guard time.isFinite, time >= 0, time < Double(UInt64.max) / 1_000_000 else { return }
    let pts = UInt64(time * 1_000_000)
    queue.async { [self] in
      defer { frameGate.withLock { $0.busy = false } }
      guard let client, approved, streaming else { return }
      if admission.1 { waitingForKeyframe = true }
      if sending { waitingForKeyframe = true; return }
      if currentHVCC != hvcc { waitingForKeyframe = true }
      if waitingForKeyframe {
        guard keyframe, dimensions.width > 0, dimensions.height > 0,
          dimensions.width <= UInt16.max, dimensions.height <= UInt16.max else { return }
        currentHVCC = hvcc
        waitingForKeyframe = false
        sendControl(.config(width: UInt16(dimensions.width), height: UInt16(dimensions.height), fps: 24, hvcC: hvcc), to: client)
        status?("TetherCam connected. Streaming video.")
      }
      sending = true
      client.send(content: IucmCodec.encode(.video(ptsUs: pts, keyframe: keyframe, nalUnits: nal)), completion: .contentProcessed { [weak self, weak client] error in
        guard let self, let client, self.client === client else { return }
        self.sending = false
        if error != nil {
          client.cancel(); self.client = nil; self.streaming = false
          self.wantsFrames.withLock { $0 = false }
        }
      })
    }
    enqueued = true
  }

  private static func localAddresses() -> [String] {
    var first: UnsafeMutablePointer<ifaddrs>?
    guard getifaddrs(&first) == 0 else { return [] }
    defer { freeifaddrs(first) }
    var addresses = [String]()
    var current = first
    while let entry = current {
      defer { current = entry.pointee.ifa_next }
      let name = String(cString: entry.pointee.ifa_name)
      guard name.hasPrefix("en") || name.hasPrefix("bridge") || name.hasPrefix("ap"),
        let address = entry.pointee.ifa_addr, address.pointee.sa_family == UInt8(AF_INET),
        (entry.pointee.ifa_flags & UInt32(IFF_LOOPBACK)) == 0,
        (entry.pointee.ifa_flags & UInt32(IFF_UP)) != 0 else { continue }
      var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
      if getnameinfo(address, socklen_t(address.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0 {
        addresses.append(String(cString: host))
      }
    }
    return Array(Set(addresses)).sorted()
  }
}
