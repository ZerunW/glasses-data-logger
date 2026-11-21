import CoreMedia
import Foundation
import Network

enum LinkMode: String, Sendable, CaseIterable {
  case off, usb, wifi
}

/// What the UI shows about the link. Frame counters refresh once a second.
struct LinkStatus: Sendable, Equatable {
  enum Phase: Sendable, Equatable {
    case off
    /// USB: listening on the phone's loopback address for the Mac.
    case waitingForComputer
    /// Wi-Fi: looking for the paired Mac.
    case searching
    case connecting
    case connected
    case needsPairing
    case failed(String)
  }

  var mode: LinkMode
  var phase: Phase
  var peerName: String?
  var sentFrames: UInt64 = 0
  var droppedFrames: UInt64 = 0
  /// Times video stopped flowing since the connection opened.
  var gaps: UInt64 = 0
  /// Bytes handed to the network and not yet sent.
  var queuedBytes: Int = 0
  /// The computer is recording, so the link buffers video instead of dropping it.
  var recordPriority = false
  /// The computer's recorder, once it has reported it on this connection; nil while not
  /// connected and for receivers that don't report it.
  var recorder: LinkRecorderState?
  /// RECORDER_STATE messages received on this connection. Status updates repeat the last
  /// state every second; this tells a new message (an answer to a command) from a repeat.
  var recorderUpdates: UInt64 = 0
  var message: String

  static let off = LinkStatus(mode: .off, phase: .off, peerName: nil, message: "The computer link is off.")
}

/// What the phone says about itself in HELLO. The Wi-Fi token comes from the pairing.
struct LinkHello: Sendable, Equatable {
  let app: String
  let device: String
  let name: String
}

/// The parts of a glasses `CMSampleBuffer` that GDL1 sends, copied out on the caller's thread.
struct LinkVideoSample: Sendable {
  let hvcC: Data
  let width: UInt16
  let height: UInt16
  let keyframe: Bool
  let ptsMicroseconds: Int64
  /// 4-byte length-prefixed NAL units.
  let data: Data

  init?(_ sample: CMSampleBuffer) {
    guard let format = CMSampleBufferGetFormatDescription(sample),
      let atoms = CMFormatDescriptionGetExtension(
        format, extensionKey: kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms) as? [String: Any],
      let hvcC = atoms["hvcC"] as? Data, !hvcC.isEmpty,
      let buffer = CMSampleBufferGetDataBuffer(sample)
    else { return nil }
    let dimensions = CMVideoFormatDescriptionGetDimensions(format)
    guard let width = UInt16(exactly: dimensions.width), let height = UInt16(exactly: dimensions.height),
      width > 0, height > 0
    else { return nil }
    let pts = CMTimeConvertScale(
      CMSampleBufferGetPresentationTimeStamp(sample), timescale: 1_000_000, method: .roundHalfAwayFromZero)
    guard pts.isNumeric else { return nil }

    let length = CMBlockBufferGetDataLength(buffer)
    guard length > 0, length <= LinkProtocol.maxFrameLength - 1 - LinkProtocol.videoHeaderLength else { return nil }
    var bytes = Data(count: length)
    let copied = bytes.withUnsafeMutableBytes { raw -> OSStatus in
      guard let base = raw.baseAddress else { return kCMBlockBufferBadPointerParameterErr }
      return CMBlockBufferCopyDataBytes(buffer, atOffset: 0, dataLength: length, destination: base)
    }
    guard copied == kCMBlockBufferNoErr else { return nil }

    var lengthSize: Int32 = 0
    let lengthStatus = CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
      format, parameterSetIndex: 0, parameterSetPointerOut: nil, parameterSetSizeOut: nil,
      parameterSetCountOut: nil, nalUnitHeaderLengthOut: &lengthSize)
    let nalLengthSize = lengthStatus == noErr && lengthSize > 0 ? Int(lengthSize) : 4
    guard let data = LinkProtocol.fourByteLengthPrefixed(bytes, lengthSize: nalLengthSize) else { return nil }

    self.hvcC = hvcC
    self.width = width
    self.height = height
    self.keyframe = LinkProtocol.containsHEVCKeyframe(data)
    self.ptsMicroseconds = pts.value
    self.data = data
  }
}

/// The phone's end of the GDL1 link to the Mac receiver.
///
/// USB: listens on `127.0.0.1:7900`; the Mac reaches it through usbmuxd. Wi-Fi: connects
/// out to the paired Mac over TLS 1.3 with a pinned certificate, found through Bonjour or
/// the addresses in the pairing. Either way the phone sends HELLO, waits for HELLO_ACK,
/// then streams video, sensor messages and STATS, and answers every PING. While the
/// computer reports that it is recording, video is buffered through short stalls instead
/// of dropped (see `LinkBufferPolicy`).
// @unchecked Sendable: all mutable state is confined to `queue`, except `gate`, which is only touched under `gateLock`.
final class ComputerLink: @unchecked Sendable {
  static let defaultFrameRate: UInt16 = 24

  private static let connectTimeout: TimeInterval = 5
  private static let handshakeTimeout: TimeInterval = 5
  private static let idleTimeoutNs: UInt64 = 6_000_000_000
  private static let browseTimeout: TimeInterval = 3
  private static let wifiRetryDelay: TimeInterval = 2
  private static let listenerRetryDelay: TimeInterval = 1

  private let queue = DispatchQueue(label: "GlassesData.ComputerLink", qos: .userInitiated)
  private let usbPort: UInt16
  private let videoFrameRate: UInt16

  /// Read on the frame and sensor threads so they can return at once while no Mac is connected.
  private struct Gate {
    var connected = false
    var glassesFrames: UInt64 = 0
    var motionSamples: UInt64 = 0
  }
  private let gateLock = NSLock()
  private var gate = Gate()

  private enum Stage { case connecting, handshaking, connected }

  private var mode: LinkMode = .off
  private var pairing: LinkPairing?
  private var hello: LinkHello?
  private var onStatus: (@Sendable (LinkStatus) -> Void)?
  private var status = LinkStatus.off
  /// Bumped by `start` and `stop`; delayed work from an older configuration is dropped.
  private var generation: UInt64 = 0
  private var listener: NWListener?
  private var browser: NWBrowser?
  /// Bumped for every Wi-Fi search, so a stale browse timeout or retry does nothing.
  private var attempt: UInt64 = 0
  private var pendingHosts: [String] = []
  /// This search already chose a route: the Bonjour result, or the hosts from the pairing.
  private var searchSettled = false
  /// The Mac refused our pairing token; wait for the user to pair again.
  private var pairingRejected = false

  private var connection: NWConnection?
  /// Bumped whenever a connection opens or closes, so stale callbacks do nothing.
  private var connectionID: UInt64 = 0
  private var stage = Stage.connecting
  private var peerName: String?
  private var reader = LinkFrameReader()
  private var lastReceiveNs: UInt64 = 0
  private var heartbeat: DispatchSourceTimer?
  private var videoGate = LinkVideoGate()
  private var backpressure = LinkBackpressure()
  private var recorderState: LinkRecorderState?
  private var recorderUpdates: UInt64 = 0
  private var outstandingBytes = 0
  private var outstandingVideoFrames = 0
  private var sentFrames: UInt64 = 0
  private var droppedFrames: UInt64 = 0
  private var latestDeviceState: Data?

  init(usbPort: UInt16 = LinkProtocol.defaultPort, videoFrameRate: UInt16 = defaultFrameRate) {
    self.usbPort = usbPort
    self.videoFrameRate = videoFrameRate
  }

  // MARK: - Public API

  /// Starts (or restarts) the link. `onStatus` is called on the link's queue.
  func start(mode: LinkMode, pairing: LinkPairing?, hello: LinkHello, onStatus: @escaping @Sendable (LinkStatus) -> Void) {
    queue.async { [self] in
      teardown()
      generation &+= 1
      self.mode = mode
      self.pairing = pairing
      self.hello = hello
      self.onStatus = onStatus
      pairingRejected = false
      status = LinkStatus(mode: mode, phase: .off, peerName: nil, message: "")
      switch mode {
      case .off: publish(.off, LinkStatus.off.message)
      case .usb: startListener()
      case .wifi: beginWifiSearch()
      }
    }
  }

  func stop() {
    queue.async { [self] in
      teardown()
      generation &+= 1
      mode = .off
      publish(.off, LinkStatus.off.message)
      onStatus = nil
    }
  }

  /// Called on the SDK's frame thread for every glasses frame. Returns at once while no
  /// Mac is connected; otherwise copies the sample out and hands it to the link queue.
  func submitVideo(_ sample: CMSampleBuffer, phoneTimeNs: UInt64) {
    let connected = gateLock.withLock {
      if gate.connected { gate.glassesFrames &+= 1 }
      return gate.connected
    }
    guard connected else { return }
    let prepared = LinkVideoSample(sample)
    queue.async { [self] in handleVideo(prepared, phoneTimeNs: phoneTimeNs) }
  }

  /// Sends one JSON message. Sensor messages are never dropped for backpressure. The
  /// latest DEVICE_STATE is kept and re-sent after every handshake.
  func sendJSON(_ type: LinkMessageType, _ body: [String: Any]) {
    let keep = type == .deviceState
    guard keep || isConnected, let frame = try? LinkProtocol.json(type, body) else { return }
    queue.async { [self] in
      if keep { latestDeviceState = frame }
      guard connection != nil, stage == .connected else { return }
      send(frame, isVideo: false)
    }
  }

  /// Asks the computer to start or stop recording. Returns false when no computer is
  /// connected; the answer arrives as a RECORDER_STATE in the status.
  @discardableResult
  func sendControl(_ command: LinkControlCommand, phoneTimeNs: UInt64 = MonotonicClock.nowNanoseconds()) -> Bool {
    guard isConnected, let frame = try? LinkProtocol.json(.control, LinkProtocol.controlBody(command, phoneTimeNs: phoneTimeNs))
    else { return false }
    queue.async { [self] in
      guard connection != nil, stage == .connected else { return }
      send(frame, isVideo: false)
    }
    return true
  }

  /// Counts one IMU sample for STATS. Only counts while connected.
  func noteMotionSample() {
    gateLock.withLock { if gate.connected { gate.motionSamples &+= 1 } }
  }

  var isConnected: Bool {
    gateLock.withLock { gate.connected }
  }

  // MARK: - USB

  private func startListener() {
    guard mode == .usb, listener == nil else { return }
    let parameters = NWParameters.tcp
    (parameters.defaultProtocolStack.transportProtocol as? NWProtocolTCP.Options)?.noDelay = true
    parameters.allowLocalEndpointReuse = true
    guard let port = NWEndpoint.Port(rawValue: usbPort) else {
      publish(.failed("Invalid port"), "USB port \(usbPort) is not valid.")
      return
    }
    parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: port)
    let newListener: NWListener
    do {
      newListener = try NWListener(using: parameters)
    } catch {
      listenerFailed(error.localizedDescription)
      return
    }
    listener = newListener
    newListener.stateUpdateHandler = { [weak self, weak newListener] state in
      guard let self, let newListener, self.listener === newListener else { return }
      switch state {
      case .ready:
        if self.connection == nil { self.publishWaitingForComputer(nil) }
      case .failed(let error), .waiting(let error):
        self.listenerFailed(Self.describe(error))
      default: break
      }
    }
    newListener.newConnectionHandler = { [weak self] connection in self?.acceptUSB(connection) }
    newListener.start(queue: queue)
  }

  private func listenerFailed(_ reason: String) {
    listener?.cancel()
    listener = nil
    if connection == nil {
      publish(.failed(reason), "USB listener stopped (\(reason)). Restarting…")
    }
    after(Self.listenerRetryDelay) { $0.startListener() }
  }

  private func acceptUSB(_ newConnection: NWConnection) {
    guard mode == .usb, connection == nil else {
      newConnection.cancel()
      return
    }
    open(newConnection)
  }

  private func publishWaitingForComputer(_ reason: String?) {
    let waiting = "Waiting for the computer. Connect the cable and run the receiver."
    publish(.waitingForComputer, reason.map { "\($0) \(waiting)" } ?? waiting)
  }

  // MARK: - Wi-Fi

  private func beginWifiSearch() {
    guard mode == .wifi, connection == nil, browser == nil, !pairingRejected else { return }
    guard let pairing else {
      publish(.needsPairing, "Pair with your computer: run the receiver with --wifi --pair and scan its code.")
      return
    }
    attempt &+= 1
    let current = attempt
    pendingHosts = []
    searchSettled = false
    publish(.searching, "Looking for \(pairing.displayName)…")

    let newBrowser = NWBrowser(
      for: .bonjourWithTXTRecord(type: LinkProtocol.bonjourType, domain: nil), using: NWParameters())
    browser = newBrowser
    newBrowser.browseResultsChangedHandler = { [weak self, weak newBrowser] results, _ in
      guard let self, let newBrowser, self.browser === newBrowser, let pairing = self.pairing else { return }
      for result in results {
        guard case .bonjour(let record) = result.metadata,
          record["fp"]?.lowercased() == pairing.fingerprint
        else { continue }
        self.stopBrowser()
        self.searchSettled = true
        // Keep the pairing's addresses as the fallback for this round: if the Bonjour
        // endpoint fails (a stale record, a network that resolves but doesn't route),
        // `end` moves on to them before scheduling a retry.
        self.pendingHosts = pairing.hosts
        self.connectWifi(to: result.endpoint, label: pairing.displayName)
        return
      }
    }
    newBrowser.stateUpdateHandler = { [weak self, weak newBrowser] state in
      guard let self, let newBrowser, self.browser === newBrowser else { return }
      if case .failed = state { self.tryPairingHosts(attempt: current) }
    }
    newBrowser.start(queue: queue)
    after(Self.browseTimeout) { $0.tryPairingHosts(attempt: current) }
  }

  /// Falls back to the addresses in the pairing when Bonjour did not find the Mac. (When
  /// Bonjour found it but the connection failed, `end` falls back to the same addresses.)
  private func tryPairingHosts(attempt current: UInt64) {
    guard attempt == current, mode == .wifi, connection == nil, !searchSettled, let pairing else { return }
    searchSettled = true
    stopBrowser()
    pendingHosts = pairing.hosts
    connectNextHost()
  }

  private func stopBrowser() {
    browser?.cancel()
    browser = nil
  }

  private func connectNextHost() {
    guard mode == .wifi, connection == nil, let pairing else { return }
    guard !pendingHosts.isEmpty, let port = NWEndpoint.Port(rawValue: pairing.port) else {
      scheduleWifiRetry(status.message)
      return
    }
    let host = pendingHosts.removeFirst()
    connectWifi(to: .hostPort(host: NWEndpoint.Host(host), port: port), label: "\(pairing.displayName) at \(host)")
  }

  private func connectWifi(to endpoint: NWEndpoint, label: String) {
    guard let pairing else { return }
    let parameters = LinkTLS.clientParameters(pinnedFingerprint: pairing.fingerprint, queue: queue)
    publish(.connecting, "Connecting to \(label)…", peer: pairing.displayName)
    open(NWConnection(to: endpoint, using: parameters))
  }

  private func scheduleWifiRetry(_ reason: String) {
    let current = attempt
    publish(.failed(reason), "\(reason) Retrying…")
    after(Self.wifiRetryDelay) { link in
      guard link.attempt == current, link.connection == nil else { return }
      link.beginWifiSearch()
    }
  }

  // MARK: - Connection

  private func open(_ newConnection: NWConnection) {
    connectionID &+= 1
    let id = connectionID
    connection = newConnection
    stage = .connecting
    reader = LinkFrameReader()
    newConnection.stateUpdateHandler = { [weak self] state in
      guard let self, self.connectionID == id else { return }
      switch state {
      case .ready: self.connectionReady(id)
      case .waiting(let error), .failed(let error): self.end(id, reason: Self.describe(error))
      case .cancelled: self.end(id, reason: "The connection was closed.")
      default: break
      }
    }
    newConnection.start(queue: queue)
    after(Self.connectTimeout) { link in
      guard link.connectionID == id, link.stage == .connecting else { return }
      link.end(id, reason: "The connection timed out.")
    }
  }

  private func connectionReady(_ id: UInt64) {
    guard stage == .connecting, let hello else { return }
    stage = .handshaking
    lastReceiveNs = MonotonicClock.nowNanoseconds()
    let token = mode == .wifi ? pairing?.token : nil
    let body = LinkProtocol.helloBody(app: hello.app, device: hello.device, name: hello.name, token: token)
    guard let frame = try? LinkProtocol.json(.hello, body) else {
      end(id, reason: "Could not encode HELLO.")
      return
    }
    if mode == .usb { publish(.connecting, "Computer connected. Waiting for the receiver to accept…") }
    send(frame, isVideo: false)
    receive(id)
    after(Self.handshakeTimeout) { link in
      guard link.connectionID == id, link.stage == .handshaking else { return }
      link.end(id, reason: "The receiver did not answer.")
    }
  }

  private func receive(_ id: UInt64) {
    connection?.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { [weak self] data, _, isComplete, error in
      guard let self, self.connectionID == id else { return }
      if let data, !data.isEmpty {
        self.lastReceiveNs = MonotonicClock.nowNanoseconds()
        let frames: [LinkFrame]
        do {
          frames = try self.reader.feed(data)
        } catch {
          self.end(id, reason: "The receiver sent invalid data.")
          return
        }
        for frame in frames {
          self.handle(frame, id: id)
          guard self.connectionID == id else { return }
        }
      }
      if let error {
        self.end(id, reason: Self.describe(error))
      } else if isComplete {
        self.end(id, reason: "The receiver closed the connection.")
      } else {
        self.receive(id)
      }
    }
  }

  private func handle(_ frame: LinkFrame, id: UInt64) {
    switch frame.messageType {
    case .ping:
      guard let macTime = LinkProtocol.decodePing(frame.payload) else {
        end(id, reason: "The receiver sent an invalid PING.")
        return
      }
      send(LinkProtocol.pong(macTimeNs: macTime, phoneTimeNs: MonotonicClock.nowNanoseconds()), isVideo: false)
    case .helloAck:
      guard stage == .handshaking else { return }
      guard let ack = LinkProtocol.decodeHelloAck(frame.payload) else {
        end(id, reason: "The receiver sent an invalid HELLO_ACK.")
        return
      }
      if ack.ok {
        handshakeCompleted(ack, id: id)
      } else {
        handshakeRejected(ack, id: id)
      }
    case .recorderState:
      // A malformed state is ignored rather than ending the connection: the video and
      // sensor stream matter more than the recorder readout.
      guard stage == .connected, let state = LinkProtocol.decodeRecorderState(frame.payload) else { return }
      recorderState = state
      recorderUpdates &+= 1
      backpressure.setRecordPriority(state.recording, outstandingBytes: outstandingBytes)
      publish(.connected, status.message, peer: peerName)
    default:
      break
    }
  }

  private func handshakeCompleted(_ ack: LinkProtocol.HelloAck, id: UInt64) {
    stage = .connected
    // This round found its route; a later disconnect starts a fresh search.
    pendingHosts = []
    peerName = mode == .wifi ? pairing?.displayName : (ack.name ?? "computer")
    sentFrames = 0
    droppedFrames = 0
    videoGate = LinkVideoGate()
    backpressure = LinkBackpressure()
    recorderState = nil
    recorderUpdates = 0
    gateLock.withLock { gate = Gate(connected: true) }
    if let latestDeviceState { send(latestDeviceState, isVideo: false) }
    let timer = DispatchSource.makeTimerSource(queue: queue)
    timer.schedule(deadline: .now() + 1, repeating: 1)
    timer.setEventHandler { [weak self] in self?.tick(id) }
    timer.resume()
    heartbeat = timer
    publish(.connected, "Connected to \(peerName ?? "the computer").", peer: peerName)
  }

  private func handshakeRejected(_ ack: LinkProtocol.HelloAck, id: UInt64) {
    let error = ack.error ?? "the receiver refused the connection"
    let isPairingError = ack.code == "pairing" || error.localizedCaseInsensitiveContains("pair")
    if mode == .wifi, isPairingError {
      pairingRejected = true
    }
    end(id, reason: "Refused by the computer: \(error).")
  }

  private func tick(_ id: UInt64) {
    guard connectionID == id, stage == .connected else { return }
    let now = MonotonicClock.nowNanoseconds()
    if now > lastReceiveNs, now - lastReceiveNs > Self.idleTimeoutNs {
      end(id, reason: "The computer stopped responding.")
      return
    }
    let counters = gateLock.withLock { gate }
    let body = LinkProtocol.statsBody(
      phoneTimeNs: now, glassesFrames: counters.glassesFrames, sentFrames: sentFrames,
      droppedFrames: droppedFrames, motionSamples: counters.motionSamples, queueBytes: outstandingBytes,
      queueFrames: outstandingVideoFrames, recordPriority: backpressure.recordPriority, gaps: videoGate.gaps)
    if let frame = try? LinkProtocol.json(.stats, body) { send(frame, isVideo: false) }
    guard connectionID == id else { return }
    publish(.connected, status.message, peer: peerName)
  }

  /// Ends connection `id` (if it is still the current one) and moves on according to the mode.
  private func end(_ id: UInt64, reason: String) {
    guard connectionID == id, connection != nil else { return }
    let wasConnected = stage == .connected
    closeConnection()
    switch mode {
    case .off:
      break
    case .usb:
      publishWaitingForComputer(wasConnected ? "Disconnected: \(reason)" : reason)
    case .wifi:
      if pairingRejected {
        publish(.needsPairing, "\(reason) Scan the pairing code on your computer again.")
      } else if !wasConnected, !pendingHosts.isEmpty {
        publish(.connecting, reason)
        connectNextHost()
      } else {
        pendingHosts = []
        scheduleWifiRetry(wasConnected ? "Disconnected: \(reason)" : reason)
      }
    }
  }

  private func closeConnection() {
    connectionID &+= 1
    connection?.stateUpdateHandler = nil
    connection?.cancel()
    connection = nil
    heartbeat?.cancel()
    heartbeat = nil
    stage = .connecting
    peerName = nil
    reader = LinkFrameReader()
    outstandingBytes = 0
    outstandingVideoFrames = 0
    backpressure = LinkBackpressure()
    recorderState = nil
    recorderUpdates = 0
    gateLock.withLock { gate.connected = false }
  }

  private func teardown() {
    closeConnection()
    listener?.cancel()
    listener = nil
    stopBrowser()
    attempt &+= 1
    pendingHosts = []
  }

  // MARK: - Sending

  private func handleVideo(_ sample: LinkVideoSample?, phoneTimeNs: UInt64) {
    guard connection != nil, stage == .connected else { return }
    guard let sample,
      let frame = try? LinkProtocol.video(
        keyframe: sample.keyframe, ptsMicroseconds: sample.ptsMicroseconds, phoneTimeNs: phoneTimeNs, sample: sample.data)
    else {
      droppedFrames &+= 1
      videoGate.noteDropped()
      return
    }
    let decision = videoGate.admit(
      keyframe: sample.keyframe, hvcC: sample.hvcC,
      outstandingBytes: outstandingBytes, outstandingFrames: outstandingVideoFrames, policy: backpressure.admission)
    switch decision {
    case .drop:
      droppedFrames &+= 1
    case .send(let withFormat):
      if withFormat {
        guard let format = try? LinkProtocol.videoFormat(
          width: sample.width, height: sample.height, fps: videoFrameRate, hvcC: sample.hvcC)
        else {
          droppedFrames &+= 1
          videoGate.noteDropped()
          return
        }
        send(format, isVideo: false)
      }
      sentFrames &+= 1
      send(frame, isVideo: true)
    }
  }

  /// Tracks bytes handed to Network and not yet processed; closes a connection that stalls.
  /// Sensor messages are always queued; only video is ever dropped (in `handleVideo`).
  private func send(_ frame: Data, isVideo: Bool) {
    guard let connection else { return }
    let id = connectionID
    let size = frame.count
    outstandingBytes += size
    if isVideo { outstandingVideoFrames += 1 }
    connection.send(content: frame, completion: .contentProcessed { [weak self] error in
      guard let self, self.connectionID == id else { return }
      self.outstandingBytes -= size
      if isVideo { self.outstandingVideoFrames -= 1 }
      if let error { self.end(id, reason: Self.describe(error)) }
    })
    if backpressure.shouldClose(outstandingBytes: outstandingBytes) {
      end(id, reason: "The connection stalled.")
    }
  }

  // MARK: - Helpers

  private func publish(_ phase: LinkStatus.Phase, _ message: String, peer: String? = nil) {
    let connected = phase == .connected
    let next = LinkStatus(
      mode: mode, phase: phase, peerName: peer, sentFrames: sentFrames, droppedFrames: droppedFrames,
      gaps: connected ? videoGate.gaps : 0, queuedBytes: connected ? outstandingBytes : 0,
      recordPriority: connected && backpressure.recordPriority, recorder: connected ? recorderState : nil,
      recorderUpdates: connected ? recorderUpdates : 0, message: message)
    guard next != status else { return }
    status = next
    onStatus?(next)
  }

  private func after(_ delay: TimeInterval, _ work: @escaping @Sendable (ComputerLink) -> Void) {
    let scheduled = generation
    queue.asyncAfter(deadline: .now() + delay) { [weak self] in
      guard let self, self.generation == scheduled else { return }
      work(self)
    }
  }

  private static func describe(_ error: NWError) -> String {
    switch error {
    case .tls(let status):
      return "Secure connection failed (TLS \(status)). The computer's certificate may not match the pairing."
    case .posix(let code) where code == .ECONNREFUSED:
      return "No receiver is listening."
    default:
      return error.localizedDescription
    }
  }
}
