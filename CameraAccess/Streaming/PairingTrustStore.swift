import Foundation
import Security

/// Single-device trust. Values are device-local and never synchronize to iCloud.
enum PairingTrustStore {
  private static let service = "com.glassesdata.camera.pairing.v1"
  private static let account = "trusted-peer"
  private static var query: [CFString: Any] {
    let value: [CFString: Any] = [kSecClass: kSecClassGenericPassword,
      kSecAttrService: service, kSecAttrAccount: account,
      kSecAttrSynchronizable: false]
    return value
  }
  static func load() throws -> Data? {
    var request = query
    request[kSecReturnData] = true
    request[kSecMatchLimit] = kSecMatchLimitOne
    var item: CFTypeRef?
    let status = SecItemCopyMatching(request as CFDictionary, &item)
    if status == errSecItemNotFound { return nil }
    guard status == errSecSuccess else { throw StoreError.keychain(status) }
    guard let key = item as? Data, key.count == 32 else { throw StoreError.invalidKey }
    return key
  }
  static func save(_ key: Data) throws {
    guard key.count == 32 else { throw StoreError.invalidKey }
    let status = SecItemUpdate(query as CFDictionary, [kSecValueData: key] as CFDictionary)
    if status == errSecSuccess { return }
    guard status == errSecItemNotFound else { throw StoreError.keychain(status) }
    var attributes = query
    attributes[kSecValueData] = key
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
  enum StoreError: Error { case keychain(OSStatus), invalidKey }
}
