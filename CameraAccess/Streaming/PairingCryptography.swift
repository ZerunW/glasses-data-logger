import CryptoKit
import Foundation

/// Cryptographic material for a single, short-lived pairing attempt.
/// The caller must verify the displayed code through an out-of-band channel
/// and authenticate the subsequent confirmation before persisting any key.
struct PairingCryptography {
  enum Error: Swift.Error { case invalidPublicKey, identicalPublicKeys }
  private let privateKey = Curve25519.KeyAgreement.PrivateKey()
  var publicKey: Data { privateKey.publicKey.rawRepresentation }

  struct Agreement {
    let displayCode: String
    let streamKey: Data
    private let authenticationKey: SymmetricKey
    private let transcript: Data

    fileprivate init(shared: SharedSecret, transcript: Data) {
      self.transcript = transcript
      let salt = Data(SHA256.hash(data: transcript))
      let verificationKey = shared.hkdfDerivedSymmetricKey(using: SHA256.self, salt: salt, sharedInfo: Data("GlassesData pairing verification v1".utf8), outputByteCount: 32)
      authenticationKey = shared.hkdfDerivedSymmetricKey(using: SHA256.self, salt: salt, sharedInfo: Data("GlassesData pairing confirmation v1".utf8), outputByteCount: 32)
      let stream = shared.hkdfDerivedSymmetricKey(using: SHA256.self, salt: salt, sharedInfo: Data("GlassesData streaming key v1".utf8), outputByteCount: 32)
      streamKey = stream.withUnsafeBytes { Data($0) }
      // Rejection sampling avoids bias from reducing a uniformly random 32-bit
      // number modulo one million. A domain-separated counter supplies retries.
      var counter: UInt32 = 0
      var number: UInt32 = 0
      repeat {
        var be = counter.bigEndian
        var message = Data("GlassesData display code v1".utf8)
        withUnsafeBytes(of: &be) { message.append(contentsOf: $0) }
        let digest = HMAC<SHA256>.authenticationCode(for: message, using: verificationKey)
        number = digest.prefix(4).reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
        counter &+= 1
      } while number >= 4_294_000_000
      displayCode = String(format: "%06u", number % 1_000_000)
    }

    func confirmation(for role: Role) -> Data {
      var message = Data("GlassesData pairing confirmation v1:".utf8)
      message.append(contentsOf: role.rawValue.utf8)
      message.append(transcript)
      return Data(HMAC<SHA256>.authenticationCode(for: message, using: authenticationKey))
    }

    func verifyConfirmation(_ data: Data, from role: Role) -> Bool {
      var message = Data("GlassesData pairing confirmation v1:".utf8)
      message.append(contentsOf: role.rawValue.utf8)
      message.append(transcript)
      return HMAC<SHA256>.isValidAuthenticationCode(data, authenticating: message, using: authenticationKey)
    }
  }

  enum Role: String { case phone, mac }

  func agree(with peerBytes: Data, role: Role) throws -> Agreement {
    guard peerBytes.count == 32,
      let peer = try? Curve25519.KeyAgreement.PublicKey(rawRepresentation: peerBytes) else { throw Error.invalidPublicKey }
    guard peerBytes != publicKey else { throw Error.identicalPublicKeys }
    let shared = try privateKey.sharedSecretFromKeyAgreement(with: peer)
    var transcript = Data("GlassesData pairing transcript v1".utf8)
    switch role {
    case .phone: transcript.append(publicKey); transcript.append(peerBytes)
    case .mac: transcript.append(peerBytes); transcript.append(publicKey)
    }
    return Agreement(shared: shared, transcript: transcript)
  }
}
