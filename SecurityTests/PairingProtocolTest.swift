import Foundation
import Network
let mode = CommandLine.arguments[1]
let queue = DispatchQueue(label: "pairing-test")
let listener = try NWListener(using: .tcp, on: 19879)
var phone: PairingConnection?
var mac: PairingConnection?
var phoneKey: Data?
var macKey: Data?
var phoneCode: String?
var macCode: String?
var phoneFailure = false
func report() {
  if mode == "allow", let phoneKey, let macKey {
    guard phoneKey == macKey, phoneCode == macCode else { fputs("FAIL: mismatch\n", stderr); exit(1) }
    print("PASS: correct code pairs both sides"); exit(0)
  }
  if (mode == "wrong" || mode == "tamper"), phoneFailure {
    guard phoneKey == nil, macKey == nil else { fputs("FAIL: wrong code paired\n", stderr); exit(1) }
    print("PASS: \(mode) attempt fails pairing"); exit(0)
  }
}
listener.newConnectionHandler = { connection in
  phone = PairingConnection(connection: connection, role: .phone, queue: queue) { event in
    switch event {
    case .code(let code):
      if mode == "tamper" { fputs("FAIL: invalid commitment accepted\n", stderr); exit(1) }
      phoneCode = code
      if mode == "pending" { return }
      queue.asyncAfter(deadline: .now() + 0.3) { phone?.confirm(code: mode == "wrong" ? "not-a-code" : code) }
    case .paired(let key): phoneKey = key; report()
    case .failed: phoneFailure = true; report()
    }
  }
  phone?.start()
}
listener.stateUpdateHandler = { state in
  if case .ready = state {
    let connection = NWConnection(host: "127.0.0.1", port: 19879, using: .tcp)
    if mode == "tamper" {
      connection.stateUpdateHandler = { state in
        guard case .ready = state else { return }
        connection.send(content: Data(repeating: 0, count: 32), completion: .contentProcessed { _ in
          connection.receive(minimumIncompleteLength: 32, maximumLength: 32) { _, _, _, _ in
            connection.receive(minimumIncompleteLength: 64, maximumLength: 64) { _, _, _, _ in
              connection.send(content: Data(repeating: 0, count: 64), completion: .contentProcessed { _ in })
            }
          }
        })
      }
      connection.start(queue: queue)
      return
    }
    mac = PairingConnection(connection: connection, role: .mac, queue: queue) { event in
      switch event {
      case .code(let code): macCode = code
      case .paired(let key): macKey = key; report()
      case .failed: break
      }
    }
    mac?.start()
  }
}
listener.start(queue: queue)
queue.asyncAfter(deadline: .now() + 2) {
  if mode == "pending" {
    guard phoneCode != nil, macCode != nil, phoneKey == nil, macKey == nil else {
      fputs("FAIL: pairing completed without user confirmation\n", stderr); exit(1)
    }
    print("PASS: no pairing before user confirmation"); exit(0)
  }
}
queue.asyncAfter(deadline: .now() + 8) { fputs("FAIL: timeout\n", stderr); exit(1) }
dispatchMain()
