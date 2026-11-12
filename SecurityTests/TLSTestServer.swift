import Foundation
import Network

let key = Data(repeating: 0x42, count: 32) // Public test fixture, never a real pairing key.
let queue = DispatchQueue(label: "GlassesData.SecurityTests")
let parameters = SecureTransport.parameters(key: key)
parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: UInt16(CommandLine.arguments[1])!)!)
let listener = try NWListener(using: parameters)
listener.stateUpdateHandler = { state in
  if case .ready = state { print("READY"); fflush(stdout) }
  if case .failed(let error) = state { fputs("Listener failed: \(error)\n", stderr); exit(1) }
}
listener.newConnectionHandler = { connection in
  connection.stateUpdateHandler = { state in
    if case .ready = state {
      connection.send(content: Data("GLASSESDATA_SECRET_TEST_FRAME".utf8), completion: .contentProcessed { _ in })
    }
  }
  connection.start(queue: queue)
}
listener.start(queue: queue)
dispatchMain()
