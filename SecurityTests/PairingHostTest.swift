import Foundation
import Network
let queue = DispatchQueue(label: "pairing-host-test")
let host = PairingHost()
var client: PairingConnection?
var phoneKey: Data?
var macKey: Data?
var macCode: String?
var requested = false
var submitted = false
func maybeConfirm() {
  guard requested, let macCode, !submitted else { return }
  submitted = true
  host.confirm(macCode)
}
func report() {
  guard let phoneKey, let macKey else { return }
  guard phoneKey == macKey else { fputs("FAIL: paired keys differ\n", stderr); exit(1) }
  print("PASS: pairing host and client agree after user confirmation"); exit(0)
}
host.start { event in
  queue.async {
    switch event {
    case .ready:
      let connection = NWConnection(host: "127.0.0.1", port: 7877, using: .tcp)
      client = PairingConnection(connection: connection, role: .mac, queue: queue) { event in
        switch event {
        case .code(let code): macCode = code; maybeConfirm()
        case .paired(let key): macKey = key; report()
        case .failed(let message): fputs("FAIL: \(message)\n", stderr); exit(1)
        }
      }
      client?.start()
    case .request: requested = true; maybeConfirm()
    case .paired(let key): phoneKey = key; report()
    case .failed(let message): fputs("FAIL: \(message)\n", stderr); exit(1)
    }
  }
}
queue.asyncAfter(deadline: .now() + 8) { fputs("FAIL: host timeout\n", stderr); exit(1) }
dispatchMain()
