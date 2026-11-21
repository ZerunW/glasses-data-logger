import Foundation
import XCTest
@testable import GlassesData

final class LinkPairingTests: XCTestCase {
  private let fingerprint = String(repeating: "0123456789abcdef", count: 4)
  private let token = "Ab0_-xYz12345678QRstuvWX"

  private func uri(
    scheme: String = "gdpair", host: String = "v1", name: String? = "Studio%20Mac",
    hosts: String? = "192.168.1.20%2C10.0.0.5", port: String? = "7900",
    fp: String? = nil, token: String? = nil, extra: String = ""
  ) -> String {
    var fields: [String] = []
    if let name { fields.append("name=\(name)") }
    if let hosts { fields.append("hosts=\(hosts)") }
    if let port { fields.append("port=\(port)") }
    fields.append("fp=\(fp ?? fingerprint)")
    fields.append("token=\(token ?? self.token)")
    return "\(scheme)://\(host)?" + fields.joined(separator: "&") + extra
  }

  func testParsesValidURI() throws {
    let pairing = try XCTUnwrap(LinkPairing(uri: uri()))
    XCTAssertEqual(pairing.name, "Studio Mac")
    XCTAssertEqual(pairing.hosts, ["192.168.1.20", "10.0.0.5"])
    XCTAssertEqual(pairing.port, 7900)
    XCTAssertEqual(pairing.fingerprint, fingerprint)
    XCTAssertEqual(pairing.token, token)
  }

  func testAcceptsWhatTheReceiverPrints() throws {
    // Shape of `wifi.pairing_uri`: urlencode with quote, so commas and spaces are percent-encoded.
    let printed = "gdpair://v1?name=Zerun%E2%80%99s%20MacBook&hosts=192.168.0.2%2C172.16.0.9&port=7900&fp="
      + fingerprint + "&token=" + token
    let pairing = try XCTUnwrap(LinkPairing(uri: "  \(printed)\n"))
    XCTAssertEqual(pairing.name, "Zerun\u{2019}s MacBook")
    XCTAssertEqual(pairing.hosts, ["192.168.0.2", "172.16.0.9"])
  }

  func testLowercasesUppercaseFingerprint() throws {
    let pairing = try XCTUnwrap(LinkPairing(uri: uri(fp: fingerprint.uppercased())))
    XCTAssertEqual(pairing.fingerprint, fingerprint)
  }

  func testMissingNameIsAllowed() throws {
    let pairing = try XCTUnwrap(LinkPairing(uri: uri(name: nil)))
    XCTAssertEqual(pairing.name, "")
    XCTAssertEqual(pairing.displayName, "Mac")
  }

  func testRejectsWrongSchemeOrVersion() {
    XCTAssertNil(LinkPairing(uri: uri(scheme: "glassesdata")))
    XCTAssertNil(LinkPairing(uri: uri(scheme: "https")))
    XCTAssertNil(LinkPairing(uri: uri(host: "v2")))
    XCTAssertNil(LinkPairing(uri: uri(host: "pair")))
    XCTAssertNil(LinkPairing(uri: "gdpair://v1/extra?" + uri().split(separator: "?")[1]))
    XCTAssertNil(LinkPairing(uri: "not a uri"))
    XCTAssertNil(LinkPairing(uri: ""))
  }

  func testRejectsBadFingerprints() {
    XCTAssertNil(LinkPairing(uri: uri(fp: String(fingerprint.dropLast()))))
    XCTAssertNil(LinkPairing(uri: uri(fp: fingerprint + "0")))
    XCTAssertNil(LinkPairing(uri: uri(fp: String(fingerprint.dropLast()) + "g")))
    XCTAssertNil(LinkPairing(uri: uri(fp: "")))
  }

  func testRejectsBadTokens() {
    XCTAssertNil(LinkPairing(uri: uri(token: String(repeating: "a", count: 15))))
    XCTAssertNotNil(LinkPairing(uri: uri(token: String(repeating: "a", count: 16))))
    XCTAssertNotNil(LinkPairing(uri: uri(token: String(repeating: "a", count: 128))))
    XCTAssertNil(LinkPairing(uri: uri(token: String(repeating: "a", count: 129))))
    XCTAssertNil(LinkPairing(uri: uri(token: "abcdefghijklmnop%2B")))
    XCTAssertNil(LinkPairing(uri: uri(token: "abcdefghijklmnop%3D")))
    XCTAssertNil(LinkPairing(uri: uri(token: "abcdefghijklmnop%20")))
  }

  func testRejectsBadHosts() {
    XCTAssertNil(LinkPairing(uri: uri(hosts: nil)))
    XCTAssertNil(LinkPairing(uri: uri(hosts: "")))
    XCTAssertNil(LinkPairing(uri: uri(hosts: "10.0.0.1%2C%2C10.0.0.2")))
    XCTAssertNil(LinkPairing(uri: uri(hosts: "10.0.0.1%20")))
    XCTAssertNil(LinkPairing(uri: uri(hosts: String(repeating: "h", count: 65))))
    XCTAssertNotNil(LinkPairing(uri: uri(hosts: String(repeating: "h", count: 64))))
    let eight = (1...8).map { "10.0.0.\($0)" }.joined(separator: "%2C")
    XCTAssertNotNil(LinkPairing(uri: uri(hosts: eight)))
    XCTAssertNil(LinkPairing(uri: uri(hosts: eight + "%2C10.0.0.9")))
  }

  func testRejectsBadPorts() {
    XCTAssertNil(LinkPairing(uri: uri(port: nil)))
    XCTAssertNil(LinkPairing(uri: uri(port: "0")))
    XCTAssertNil(LinkPairing(uri: uri(port: "65536")))
    XCTAssertNil(LinkPairing(uri: uri(port: "-1")))
    XCTAssertNil(LinkPairing(uri: uri(port: "%2B7900")))
    XCTAssertNil(LinkPairing(uri: uri(port: "79a0")))
    XCTAssertNotNil(LinkPairing(uri: uri(port: "65535")))
    XCTAssertNotNil(LinkPairing(uri: uri(port: "1")))
  }

  func testRejectsLongOrControlCharacterNames() {
    XCTAssertNotNil(LinkPairing(uri: uri(name: String(repeating: "n", count: 64))))
    XCTAssertNil(LinkPairing(uri: uri(name: String(repeating: "n", count: 65))))
    XCTAssertNil(LinkPairing(uri: uri(name: "Mac%0A")))
  }

  func testRejectsRepeatedFields() {
    XCTAssertNil(LinkPairing(uri: uri(extra: "&token=\(token)")))
    XCTAssertNil(LinkPairing(uri: uri(extra: "&port=7901")))
  }

  func testCodableRoundTripRevalidates() throws {
    let pairing = try XCTUnwrap(LinkPairing(uri: uri()))
    let data = try JSONEncoder().encode(pairing)
    XCTAssertEqual(try JSONDecoder().decode(LinkPairing.self, from: data), pairing)

    var object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    object["token"] = "short"
    let tampered = try JSONSerialization.data(withJSONObject: object)
    XCTAssertThrowsError(try JSONDecoder().decode(LinkPairing.self, from: tampered))
  }
}
