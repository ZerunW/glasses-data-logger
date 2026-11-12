import AVFoundation
import SwiftUI
import VisionKit

/// Scans the QR code containing the Mac's six-digit pairing code.
struct PairingScanner: UIViewControllerRepresentable {
  let onCode: @MainActor (String) -> Void
  func makeCoordinator() -> Coordinator { Coordinator(onCode: onCode) }
  func makeUIViewController(context: Context) -> DataScannerViewController {
    let controller = DataScannerViewController(
      recognizedDataTypes: [.barcode(symbologies: [.qr])], qualityLevel: .accurate,
      recognizesMultipleItems: true, isHighFrameRateTrackingEnabled: false,
      isPinchToZoomEnabled: true, isGuidanceEnabled: true, isHighlightingEnabled: true)
    controller.delegate = context.coordinator
    DispatchQueue.main.async { try? controller.startScanning() }
    return controller
  }
  func updateUIViewController(_ controller: DataScannerViewController, context: Context) {}
  static func dismantleUIViewController(_ controller: DataScannerViewController, coordinator: Coordinator) {
    controller.stopScanning()
  }
  @MainActor final class Coordinator: NSObject, DataScannerViewControllerDelegate {
    private let onCode: @MainActor (String) -> Void
    private var handled = false
    init(onCode: @escaping @MainActor (String) -> Void) { self.onCode = onCode }
    func dataScanner(_ dataScanner: DataScannerViewController, didAdd addedItems: [RecognizedItem], allItems: [RecognizedItem]) {
      accept(allItems, scanner: dataScanner)
    }
    func dataScanner(_ dataScanner: DataScannerViewController, didUpdate updatedItems: [RecognizedItem], allItems: [RecognizedItem]) {
      accept(allItems, scanner: dataScanner)
    }
    private func accept(_ items: [RecognizedItem], scanner: DataScannerViewController) {
      guard !handled else { return }
      for item in items {
        guard case .barcode(let barcode) = item,
          let payload = barcode.payloadStringValue else { continue }
        let value = payload.trimmingCharacters(in: .whitespacesAndNewlines)
        guard value.utf8.count == 6, value.utf8.allSatisfy({ $0 >= 48 && $0 <= 57 }) else { continue }
        handled = true
        scanner.stopScanning()
        onCode(value)
        return
      }
    }
  }
}
