import CryptoKit
import Foundation
import Network
import Security

/// TLS settings for the Wi-Fi link. The receiver's certificate is self-signed, so trust
/// comes only from the fingerprint pinned by the pairing QR code: there is no system
/// trust evaluation and no host-name check.
enum LinkTLS {
  static func clientParameters(pinnedFingerprint: String, queue: DispatchQueue) -> NWParameters {
    let pinned = pinnedFingerprint.lowercased()
    let tls = NWProtocolTLS.Options()
    let options = tls.securityProtocolOptions
    sec_protocol_options_set_min_tls_protocol_version(options, .TLSv13)
    sec_protocol_options_set_verify_block(options, { _, trust, complete in
      complete(leafFingerprint(trust) == pinned)
    }, queue)
    let tcp = NWProtocolTCP.Options()
    tcp.noDelay = true
    return NWParameters(tls: tls, tcp: tcp)
  }

  static func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  private static func leafFingerprint(_ trust: sec_trust_t) -> String? {
    let secTrust = sec_trust_copy_ref(trust).takeRetainedValue()
    guard let chain = SecTrustCopyCertificateChain(secTrust) as? [SecCertificate], let leaf = chain.first else {
      return nil
    }
    return sha256Hex(SecCertificateCopyData(leaf) as Data)
  }
}
