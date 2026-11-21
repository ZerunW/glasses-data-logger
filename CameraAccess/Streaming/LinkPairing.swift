import Foundation

/// The Mac an iPhone is paired with for the Wi-Fi link, read from the QR code the
/// receiver shows: `gdpair://v1?name=<mac>&hosts=<ip,ip>&port=7900&fp=<64 hex>&token=<token>`.
struct LinkPairing: Codable, Equatable, Sendable {
  static let scheme = "gdpair"

  let name: String
  let hosts: [String]
  let port: UInt16
  /// Lowercase hex SHA-256 of the receiver certificate's DER encoding.
  let fingerprint: String
  let token: String

  /// Validates every field; `fingerprint` may be any case and is stored lowercased.
  init?(name: String, hosts: [String], port: UInt16, fingerprint: String, token: String) {
    guard Self.isValidName(name), Self.isValidHosts(hosts), port > 0,
      Self.isValidFingerprint(fingerprint), Self.isValidToken(token)
    else { return nil }
    self.name = name
    self.hosts = hosts
    self.port = port
    self.fingerprint = fingerprint.lowercased()
    self.token = token
  }

  init?(uri: String) {
    let trimmed = uri.trimmingCharacters(in: .whitespacesAndNewlines)
    guard let components = URLComponents(string: trimmed),
      components.scheme?.lowercased() == Self.scheme,
      components.host?.lowercased() == "v1",
      components.path.isEmpty || components.path == "/",
      let items = components.queryItems
    else { return nil }
    var values: [String: String] = [:]
    for item in items {
      // A repeated field is ambiguous; refuse it rather than pick one.
      guard values[item.name] == nil else { return nil }
      values[item.name] = item.value ?? ""
    }
    guard let hostList = values["hosts"], let portText = values["port"],
      let fingerprint = values["fp"], let token = values["token"],
      !portText.isEmpty, portText.utf8.count <= 5, portText.utf8.allSatisfy({ $0 >= 48 && $0 <= 57 }),
      let port = UInt16(portText)
    else { return nil }
    let hosts = hostList.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
    self.init(name: values["name"] ?? "", hosts: hosts, port: port, fingerprint: fingerprint, token: token)
  }

  init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    guard let pairing = LinkPairing(
      name: try container.decode(String.self, forKey: .name),
      hosts: try container.decode([String].self, forKey: .hosts),
      port: try container.decode(UInt16.self, forKey: .port),
      fingerprint: try container.decode(String.self, forKey: .fingerprint),
      token: try container.decode(String.self, forKey: .token))
    else {
      throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "Invalid pairing"))
    }
    self = pairing
  }

  /// A name for the UI; the receiver may send an empty one.
  var displayName: String { name.isEmpty ? "Mac" : name }

  // MARK: - Validation

  private static func isValidName(_ name: String) -> Bool {
    name.count <= 64 && !name.unicodeScalars.contains { CharacterSet.controlCharacters.contains($0) }
  }

  private static func isValidHosts(_ hosts: [String]) -> Bool {
    (1...8).contains(hosts.count)
      && hosts.allSatisfy { host in
        (1...64).contains(host.count)
          && !host.unicodeScalars.contains {
            CharacterSet.whitespacesAndNewlines.contains($0) || CharacterSet.controlCharacters.contains($0)
          }
      }
  }

  private static func isValidFingerprint(_ fingerprint: String) -> Bool {
    fingerprint.utf8.count == 64 && fingerprint.utf8.allSatisfy(isHexDigit)
  }

  private static func isValidToken(_ token: String) -> Bool {
    (16...128).contains(token.utf8.count)
      && token.utf8.allSatisfy { isASCIIAlphanumeric($0) || $0 == UInt8(ascii: "-") || $0 == UInt8(ascii: "_") }
  }

  private static func isHexDigit(_ byte: UInt8) -> Bool {
    (48...57).contains(byte) || (65...70).contains(byte) || (97...102).contains(byte)
  }

  private static func isASCIIAlphanumeric(_ byte: UInt8) -> Bool {
    (48...57).contains(byte) || (65...90).contains(byte) || (97...122).contains(byte)
  }
}
