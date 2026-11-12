import CoreMedia
import Foundation
import Network

// This control-plane check does not inject camera frames.
extension CMSampleBuffer { func isHEVCKeyframe() -> Bool { false } }
let mode = CommandLine.arguments[1]
let key = Data(repeating: 0x42, count: 32)
let server = CameraBroadcast()
let queue = DispatchQueue(label: "approval-test")
let lock = NSLock()
var gotApproval = false
var gotHello = false
var gotStartStatus = false
var closed = false
let parser = IucmFrameParser()
func fail(_ message: String) -> Never { fputs("FAIL: \(message)\n", stderr); exit(1) }
func receive(_ connection: NWConnection) {
  connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { data, _, complete, error in
    if let data, !data.isEmpty {
      do {
        for message in try parser.feedMessages(data) {
          switch message {
          case .hello:
            lock.lock(); gotHello = true; lock.unlock()
            let start = StartParams(cameraId: 0, width: 1280, height: 720, fps: 24, bitrateKbps: 4000)
            connection.send(content: IucmCodec.encode(.start(start)), completion: .contentProcessed { _ in })
          case .config, .video: fail("Unexpected video data without submitted camera frames")
          default: break
          }
        }
      } catch { fail("Invalid response framing") }
    }
    if complete || error != nil || data == nil || data!.isEmpty {
      lock.lock(); closed = true; lock.unlock()
      return
    }
    receive(connection)
  }
}
server.start(allowWiFi: true, key: key, onStatus: { state in
  if state.hasPrefix("Broadcast failed:") { fail(state) }
  if state == "Connection approved. Starting video." {
    lock.lock(); gotStartStatus = true; lock.unlock()
  }
  if state.hasPrefix("Ready for nearby") {
    let clientKey = mode == "wrong" ? Data(repeating: 0x43, count: 32) : key
    let connection = NWConnection(host: "127.0.0.1", port: 7878, using: SecureTransport.parameters(key: clientKey))
    connection.stateUpdateHandler = { state in
      if case .ready = state { receive(connection) }
    }
    connection.start(queue: queue)
  }
}, onApproval: { id, _ in
  lock.lock(); gotApproval = true; lock.unlock()
  if mode == "wrong" { fail("Wrong key prompted for approval") }
  queue.asyncAfter(deadline: .now() + 1) {
    server.resolveConnection(id, allow: mode == "allow")
  }
})
queue.asyncAfter(deadline: .now() + 4) {
  lock.lock(); let approval = gotApproval; let hello = gotHello; let started = gotStartStatus; let disconnected = closed; lock.unlock()
  switch mode {
  case "allow":
    guard approval && hello && started else { fail("HELLO/START/approval did not complete") }
    print("PASS: HELLO and START complete while video awaits approval")
  case "reject":
    guard approval && hello && disconnected else { fail("Rejected client did not disconnect") }
    print("PASS: rejected client disconnected")
  case "wrong":
    guard !approval && !hello else { fail("Wrong key reached the protocol or approval prompt") }
    print("PASS: wrong TLS key cannot reach the protocol")
  default: fail("Unknown scenario")
  }
  exit(0)
}
dispatchMain()
