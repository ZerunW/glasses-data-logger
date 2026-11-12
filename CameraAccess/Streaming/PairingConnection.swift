import CryptoKit
import Foundation
import Network

/// One bounded pairing attempt over an untrusted connection. The six-digit
/// code must be independently transferred and confirmed by the user before
/// the phone sends its cryptographic confirmation.
final class PairingConnection: @unchecked Sendable {
  enum Event {
    case code(String)
    case paired(Data)
    case failed(String)
  }
  private let queue: DispatchQueue
  private let connection: NWConnection
  private let role: PairingCryptography.Role
  private let crypto = PairingCryptography()
  private let nonce = SymmetricKey(size: .bits256).withUnsafeBytes { Data($0) }
  private let callback: @Sendable (Event) -> Void
  private var agreement: PairingCryptography.Agreement?
  private var finished = false
  private var startedExchange = false
  private var verificationPending = false

  init(connection: NWConnection, role: PairingCryptography.Role,
       queue: DispatchQueue, callback: @escaping @Sendable (Event) -> Void) {
    self.connection = connection; self.role = role; self.queue = queue; self.callback = callback
  }

  func start() {
    connection.stateUpdateHandler = { [weak self] state in
      guard let self else { return }
      switch state {
      case .ready:
        guard !self.startedExchange else { return }
        self.startedExchange = true
        self.exchangeCommitments()
      case .failed, .cancelled: self.fail("Pairing connection ended.")
      default: break
      }
    }
    connection.start(queue: queue)
    queue.asyncAfter(deadline: .now() + 60) { [weak self] in self?.fail("Pairing timed out.") }
  }

  /// Only the phone accepts user input; an incorrect code ends the attempt.
  func confirm(code: String) {
    queue.async { [self] in
      guard !finished, role == .phone, verificationPending, let agreement else { return }
      verificationPending = false
      guard code == agreement.displayCode else { fail("Incorrect pairing code."); return }
      send(agreement.confirmation(for: .phone)) { [weak self] in self?.readConfirmation(from: .mac) }
    }
  }

  func cancel() { queue.async { self.fail("Pairing cancelled.") } }

  // Both peers commit before either reveals its public key. This prevents an
  // active attacker from grinding candidate keys against the other side's
  // already-revealed key to force a matching short authentication string.
  private func exchangeCommitments() {
    let payload = crypto.publicKey + nonce
    let localCommitment = commitment(for: payload)
    send(localCommitment) { [weak self] in
      guard let self else { return }
      self.receiveExactly(32) { [weak self] peerCommitment in
        guard let self else { return }
        self.send(payload) { [weak self] in
          guard let self else { return }
          self.receiveExactly(64) { [weak self] peerPayload in
            guard let self, !self.finished else { return }
            guard self.commitment(for: peerPayload) == peerCommitment else {
              self.fail("Invalid pairing commitment."); return
            }
            do {
              let agreement = try self.crypto.agree(with: Data(peerPayload.prefix(32)), role: self.role)
              self.agreement = agreement
              if self.role == .phone { self.verificationPending = true }
              self.callback(.code(agreement.displayCode))
              if self.role == .mac { self.readConfirmation(from: .phone) }
            } catch { self.fail("Invalid pairing public key.") }
          }
        }
      }
    }
  }

  private func commitment(for payload: Data) -> Data {
    var message = Data("GlassesData pairing commitment v1".utf8)
    message.append(payload)
    return Data(SHA256.hash(data: message))
  }

  private func readConfirmation(from peer: PairingCryptography.Role) {
    receiveExactly(32) { [weak self] bytes in
      guard let self, let agreement = self.agreement, !self.finished else { return }
      guard agreement.verifyConfirmation(bytes, from: peer) else { self.fail("Pairing verification failed."); return }
      if self.role == .mac {
        self.send(agreement.confirmation(for: .mac)) { [weak self] in self?.complete(agreement.streamKey) }
      } else { self.complete(agreement.streamKey) }
    }
  }

  private func receiveExactly(_ count: Int, collected: Data = Data(), completion: @escaping @Sendable (Data) -> Void) {
    guard !finished else { return }
    connection.receive(minimumIncompleteLength: 1, maximumLength: count - collected.count) { [weak self] data, _, complete, error in
      guard let self, !self.finished else { return }
      guard error == nil, let data, !data.isEmpty else { self.fail("Pairing connection interrupted."); return }
      var result = collected; result.append(data)
      if result.count == count { completion(result) }
      else if complete { self.fail("Incomplete pairing message.") }
      else { self.receiveExactly(count, collected: result, completion: completion) }
    }
  }

  private func send(_ bytes: Data, completion: @escaping @Sendable () -> Void) {
    guard !finished else { return }
    connection.send(content: bytes, completion: .contentProcessed { [weak self] error in
      guard let self, !self.finished else { return }
      guard error == nil else { self.fail("Could not send pairing message."); return }
      completion()
    })
  }
  private func complete(_ key: Data) {
    guard !finished else { return }
    finished = true; callback(.paired(key)); connection.cancel()
  }
  private func fail(_ message: String) {
    guard !finished else { return }
    finished = true; callback(.failed(message)); connection.cancel()
  }
}
