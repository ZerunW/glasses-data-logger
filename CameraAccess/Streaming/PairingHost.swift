import Foundation
import Network

final class PairingHost: @unchecked Sendable {
  enum Event { case ready, request, paired(Data), failed(String) }
  private let queue = DispatchQueue(label: "GlassesData.PairingHost")
  private var listener: NWListener?
  private var attempt: PairingConnection?
  private var update: (@Sendable (Event) -> Void)?

  func start(update: @escaping @Sendable (Event) -> Void) {
    queue.async { [self] in
      stopOnQueue(); self.update = update
      do {
        let parameters = NWParameters.tcp
        parameters.includePeerToPeer = true
        let listener = try NWListener(using: parameters, on: 7877)
        listener.service = NWListener.Service(name: "GlassesData", type: "_gdpair._tcp")
        self.listener = listener
        listener.stateUpdateHandler = { [weak self, weak listener] state in
          guard let self, let listener, self.listener === listener else { return }
          switch state {
          case .ready: self.update?(.ready)
          case .failed: self.update?(.failed("Could not start pairing.")); self.stopOnQueue()
          default: break
          }
        }
        listener.newConnectionHandler = { [weak self] connection in
          guard let self, self.attempt == nil else { connection.cancel(); return }
          let attempt = PairingConnection(connection: connection, role: .phone, queue: self.queue) { [weak self] event in
            guard let self else { return }
            switch event {
            case .code: self.update?(.request)
            case .paired(let key): self.update?(.paired(key)); self.stopOnQueue()
            case .failed(let message): self.update?(.failed(message)); self.stopOnQueue()
            }
          }
          self.attempt = attempt
          attempt.start()
        }
        listener.start(queue: queue)
        queue.asyncAfter(deadline: .now() + 90) { [weak self, weak listener] in
          guard let self, let listener, self.listener === listener else { return }
          self.update?(.failed("Pairing timed out.")); self.stopOnQueue()
        }
      } catch { update(.failed("Could not start pairing.")); stopOnQueue() }
    }
  }
  func confirm(_ code: String) { queue.async { self.attempt?.confirm(code: code) } }
  func stop() { queue.async { self.stopOnQueue() } }
  private func stopOnQueue() {
    let active = attempt; attempt = nil
    listener?.cancel(); listener = nil; update = nil
    active?.cancel()
  }
}
