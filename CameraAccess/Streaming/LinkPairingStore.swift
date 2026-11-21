import Foundation
import Security

/// Keychain storage for the one paired Mac. Device-local: never synchronized to iCloud
/// and readable only while the device is unlocked.
enum LinkPairingStore {
  private static let service = "com.glassesdata.link.pairing.v1"
  private static let account = "paired-mac"
  private static var query: [CFString: Any] {
    [
      kSecClass: kSecClassGenericPassword,
      kSecAttrService: service,
      kSecAttrAccount: account,
      kSecAttrSynchronizable: false,
    ]
  }

  static func load() throws -> LinkPairing? {
    var request = query
    request[kSecReturnData] = true
    request[kSecMatchLimit] = kSecMatchLimitOne
    var item: CFTypeRef?
    let status = SecItemCopyMatching(request as CFDictionary, &item)
    if status == errSecItemNotFound { return nil }
    guard status == errSecSuccess else { throw StoreError.keychain(status) }
    guard let data = item as? Data, let pairing = try? JSONDecoder().decode(LinkPairing.self, from: data) else {
      throw StoreError.invalidData
    }
    return pairing
  }

  static func save(_ pairing: LinkPairing) throws {
    let data = try JSONEncoder().encode(pairing)
    let status = SecItemUpdate(query as CFDictionary, [kSecValueData: data] as CFDictionary)
    if status == errSecSuccess { return }
    guard status == errSecItemNotFound else { throw StoreError.keychain(status) }
    var attributes = query
    attributes[kSecValueData] = data
    #if !os(macOS)
    attributes[kSecAttrAccessible] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
    #endif
    let added = SecItemAdd(attributes as CFDictionary, nil)
    guard added == errSecSuccess else { throw StoreError.keychain(added) }
  }

  static func remove() throws {
    let status = SecItemDelete(query as CFDictionary)
    guard status == errSecSuccess || status == errSecItemNotFound else { throw StoreError.keychain(status) }
  }

  enum StoreError: Error { case keychain(OSStatus), invalidData }
}
