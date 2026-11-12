import Foundation
import Network
import Security

enum SecureTransport {
  static func makeKey() throws -> Data {
    var bytes = [UInt8](repeating: 0, count: 32)
    let result = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
    guard result == errSecSuccess else { throw KeyError.randomFailed(result) }
    return Data(bytes)
  }

  static func keyFromHex(_ string: String) -> Data? {
    guard string.utf8.count == 64, string.utf8.allSatisfy({ ($0 >= 48 && $0 <= 57) || ($0 >= 97 && $0 <= 102) || ($0 >= 65 && $0 <= 70) }) else { return nil }
    var result = Data()
    var index = string.startIndex
    for _ in 0..<32 {
      let end = string.index(index, offsetBy: 2)
      guard let byte = UInt8(string[index..<end], radix: 16) else { return nil }
      result.append(byte)
      index = end
    }
    return result
  }

  static func hex(_ key: Data) -> String {
    key.map { String(format: "%02x", $0) }.joined()
  }

  static func parameters(key: Data) -> NWParameters {
    let tls = NWProtocolTLS.Options()
    let options = tls.securityProtocolOptions
    let keyBytes = key.withUnsafeBytes { DispatchData(bytes: $0) }
    let identity = Data("GlassesData-TetherCam-v1".utf8).withUnsafeBytes { DispatchData(bytes: $0) }
    sec_protocol_options_set_min_tls_protocol_version(options, .TLSv12)
    sec_protocol_options_set_max_tls_protocol_version(options, .TLSv12)
    sec_protocol_options_append_tls_ciphersuite(options, tls_ciphersuite_t(rawValue: 0x00A8)!)
    sec_protocol_options_add_pre_shared_key(options, keyBytes as __DispatchData, identity as __DispatchData)
    let tcp = NWProtocolTCP.Options()
    tcp.noDelay = true
    return NWParameters(tls: tls, tcp: tcp)
  }

  enum KeyError: Error { case randomFailed(OSStatus); case missingKey }
}
