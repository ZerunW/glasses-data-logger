import Foundation
import XCTest
@testable import GlassesData

final class PairingCryptographyTests: XCTestCase {
  func testBothPeersDeriveTheSameCodeKeyAndConfirmations() throws {
    let phone = PairingCryptography()
    let mac = PairingCryptography()
    let phoneAgreement = try phone.agree(with: mac.publicKey, role: .phone)
    let macAgreement = try mac.agree(with: phone.publicKey, role: .mac)
    XCTAssertEqual(phoneAgreement.displayCode, macAgreement.displayCode)
    XCTAssertEqual(phoneAgreement.displayCode.count, 6)
    XCTAssertEqual(phoneAgreement.streamKey, macAgreement.streamKey)
    XCTAssertTrue(phoneAgreement.verifyConfirmation(macAgreement.confirmation(for: .mac), from: .mac))
    XCTAssertTrue(macAgreement.verifyConfirmation(phoneAgreement.confirmation(for: .phone), from: .phone))
  }

  func testWrongPeerAndReflectedConfirmationAreRejected() throws {
    let phone = PairingCryptography()
    let mac = PairingCryptography()
    let attacker = PairingCryptography()
    let agreement = try phone.agree(with: mac.publicKey, role: .phone)
    let attackerAgreement = try attacker.agree(with: phone.publicKey, role: .mac)
    XCTAssertFalse(agreement.verifyConfirmation(attackerAgreement.confirmation(for: .mac), from: .mac))
    XCTAssertFalse(agreement.verifyConfirmation(agreement.confirmation(for: .phone), from: .mac))
    XCTAssertThrowsError(try phone.agree(with: phone.publicKey, role: .phone))
    XCTAssertThrowsError(try phone.agree(with: Data([1, 2, 3]), role: .phone))
  }
}
