import AVFoundation
import Foundation
import Network

let key = Data(repeating: 0x42, count: 32)
let server = CameraBroadcast()
let relay = Relay()
let ready = DispatchSemaphore(value: 0)
server.start(allowWiFi: true, key: key, onStatus: { status in
  if status.hasPrefix("Ready for nearby") { ready.signal() }
  if status.hasPrefix("Broadcast failed:") { fputs("\(status)\n", stderr); exit(1) }
}, onApproval: { id, _ in
  DispatchQueue.global().asyncAfter(deadline: .now() + 6) { server.resolveConnection(id, allow: true) }
})
guard ready.wait(timeout: .now() + 8) == .success else { fatalError("Broadcast did not start") }
relay.start(endpoint: .hostPort(host: "127.0.0.1", port: 7878), key: key) { state in
  switch state {
  case .ready: print("Ready for encrypted relay"); fflush(stdout)
  case .failed(let message): fputs("Relay: \(message)\n", stderr); exit(1)
  default: break
  }
}
let asset = AVURLAsset(url: URL(fileURLWithPath: CommandLine.arguments[1]))
let track = asset.tracks(withMediaType: .video).first!
let deadline = Date().addingTimeInterval(16)
while Date() < deadline {
  let reader = try AVAssetReader(asset: asset)
  let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
  reader.add(output)
  reader.startReading()
  while Date() < deadline, let frame = output.copyNextSampleBuffer() {
    server.submit(frame)
    Thread.sleep(forTimeInterval: 0.045)
  }
}
relay.stop()
server.stop()
