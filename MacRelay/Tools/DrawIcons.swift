import AppKit
import Foundation

struct IconPainter {
  static let side = 1024
  static let macBlue = NSColor(calibratedRed: 0.08, green: 0.24, blue: 0.43, alpha: 1)

  static func makeIcon(sourceURL: URL, foreground: NSColor) throws -> NSBitmapImageRep {
    guard let source = NSImage(contentsOf: sourceURL) else {
      throw NSError(domain: "IconPainter", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not open icon reference image"])
    }

    // Center-crop only the empty side margins so the original artwork keeps its proportions.
    let cropSide = min(source.size.width, source.size.height) * 0.91
    let sourceRect = NSRect(x: (source.size.width - cropSide) / 2, y: 0, width: cropSide, height: cropSide)
    guard let sourceBitmap = NSBitmapImageRep(
      bitmapDataPlanes: nil,
      pixelsWide: side,
      pixelsHigh: side,
      bitsPerSample: 8,
      samplesPerPixel: 4,
      hasAlpha: true,
      isPlanar: false,
      colorSpaceName: .deviceRGB,
      bytesPerRow: 0,
      bitsPerPixel: 0
    ) else {
      throw NSError(domain: "IconPainter", code: 2)
    }

    NSGraphicsContext.saveGraphicsState()
    guard let context = NSGraphicsContext(bitmapImageRep: sourceBitmap) else {
      NSGraphicsContext.restoreGraphicsState()
      throw NSError(domain: "IconPainter", code: 3)
    }
    NSGraphicsContext.current = context
    context.imageInterpolation = .high
    NSColor.black.setFill()
    NSBezierPath(rect: NSRect(x: 0, y: 0, width: side, height: side)).fill()
    source.draw(
      in: NSRect(x: 0, y: 0, width: side, height: side),
      from: sourceRect,
      operation: .copy,
      fraction: 1,
      respectFlipped: false,
      hints: nil
    )
    context.flushGraphics()
    NSGraphicsContext.restoreGraphicsState()

    guard let output = NSBitmapImageRep(
      bitmapDataPlanes: nil,
      pixelsWide: side,
      pixelsHigh: side,
      bitsPerSample: 8,
      samplesPerPixel: 4,
      hasAlpha: true,
      isPlanar: false,
      colorSpaceName: .deviceRGB,
      bytesPerRow: 0,
      bitsPerPixel: 0
    ) else {
      throw NSError(domain: "IconPainter", code: 4)
    }

    let fg = foreground.usingColorSpace(.deviceRGB)!
    for y in 0..<side {
      for x in 0..<side {
        let sample = sourceBitmap.colorAt(x: x, y: y)!.usingColorSpace(.deviceRGB)!
        // The source is white artwork on black; its luminance becomes color coverage.
        let coverage = (sample.redComponent + sample.greenComponent + sample.blueComponent) / 3
        output.setColor(
          NSColor(
            calibratedRed: 1 - (1 - fg.redComponent) * coverage,
            green: 1 - (1 - fg.greenComponent) * coverage,
            blue: 1 - (1 - fg.blueComponent) * coverage,
            alpha: 1
          ),
          atX: x,
          y: y
        )
      }
    }
    return output
  }

  static func write(_ bitmap: NSBitmapImageRep, to url: URL) throws {
    guard let png = bitmap.representation(using: .png, properties: [:]) else {
      throw NSError(domain: "IconPainter", code: 5)
    }
    try png.write(to: url, options: .atomic)
  }
}

@main struct DrawIcons {
  static func main() throws {
    let args = CommandLine.arguments
    guard args.count == 3 else {
      throw NSError(domain: "DrawIcons", code: 1, userInfo: [NSLocalizedDescriptionKey: "Usage: DrawIcons output-directory reference-image"])
    }
    let output = URL(fileURLWithPath: args[1], isDirectory: true)
    let source = URL(fileURLWithPath: args[2])
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    try IconPainter.write(IconPainter.makeIcon(sourceURL: source, foreground: .black), to: output.appendingPathComponent("GlassesData-iPhone.png"))
    try IconPainter.write(IconPainter.makeIcon(sourceURL: source, foreground: IconPainter.macBlue), to: output.appendingPathComponent("GlassesData-Mac.png"))
  }
}
