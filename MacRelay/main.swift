import AppKit
import CoreImage
import CoreImage.CIFilterBuiltins
import Network
import SwiftUI

@MainActor
final class CameraRelayModel: ObservableObject {
  struct NearbyDevice: Identifiable {
    let endpoint: NWEndpoint
    var id: String { String(describing: endpoint) }
    var name: String {
      if case .service(let name, _, _, _) = endpoint { return name }
      return "GlassesData"
    }
  }
  @Published var devices: [NearbyDevice] = []
  @Published var selectedDevice = ""
  @Published var discoveryStatus = "Looking for nearby iPhones…"
  @Published var address = ""
  @Published var pairingDevices: [NearbyDevice] = []
  @Published var selectedPairingDevice = ""
  @Published var pairingCode: String?
  @Published var pairingMessage = "On your iPhone, open Mac camera and select Pair a Mac."
  @Published var isPairing = false
  @Published var hasTrust = (try? PairingTrustStore.load()) != nil
  private var pairingBrowser: NWBrowser?
  private let pairingQueue = DispatchQueue(label: "GlassesData.PairingClient")
  private var pairingAttempt: PairingConnection?
  private var autoConnectEligible = true
  private var browser: NWBrowser?
  private let discoveryQueue = DispatchQueue(label: "GlassesData.Discovery")

  init() {
    let parameters = NWParameters.tcp
    parameters.includePeerToPeer = true
    let browser = NWBrowser(for: .bonjour(type: "_glassesdata._tcp", domain: nil), using: parameters)
    self.browser = browser
    browser.browseResultsChangedHandler = { [weak self] results, _ in
      let devices = results.map { NearbyDevice(endpoint: $0.endpoint) }.sorted { $0.id < $1.id }
      DispatchQueue.main.async {
        guard let self else { return }
        self.devices = devices
        if !devices.contains(where: { $0.id == self.selectedDevice }) {
          self.selectedDevice = devices.first?.id ?? ""
        }
        self.discoveryStatus = devices.isEmpty ? "Looking for nearby iPhones…" : "Choose an iPhone below."
        if !devices.isEmpty, self.hasTrust, self.autoConnectEligible, !self.useUSB, !self.isSwitchingMode, !self.isPairing, !self.isRunning {
          self.autoConnectEligible = false
          self.start()
        }
      }
    }
    browser.stateUpdateHandler = { [weak self] state in
      if case .waiting = state {
        DispatchQueue.main.async { self?.discoveryStatus = "Discovery unavailable. Check local network permission." }
      }
    }
    browser.start(queue: discoveryQueue)
    let pairingParameters = NWParameters.tcp
    pairingParameters.includePeerToPeer = true
    let pairingBrowser = NWBrowser(for: .bonjour(type: "_gdpair._tcp", domain: nil), using: pairingParameters)
    self.pairingBrowser = pairingBrowser
    pairingBrowser.browseResultsChangedHandler = { [weak self] results, _ in
      let devices = results.map { NearbyDevice(endpoint: $0.endpoint) }.sorted { $0.id < $1.id }
      DispatchQueue.main.async {
        guard let self else { return }
        self.pairingDevices = devices
        if !devices.contains(where: { $0.id == self.selectedPairingDevice }) {
          self.selectedPairingDevice = devices.first?.id ?? ""
        }
      }
    }
    pairingBrowser.start(queue: discoveryQueue)
    _ = startReceiverIfNeeded()
  }

  func beginPairing() {
    guard !isRunning, !isPairing, let device = pairingDevices.first(where: { $0.id == selectedPairingDevice }) else {
      pairingMessage = "Start pairing on your iPhone first."
      return
    }
    autoConnectEligible = false
    isPairing = true
    pairingCode = nil
    pairingMessage = "Connecting to iPhone…"
    let parameters = NWParameters.tcp
    parameters.includePeerToPeer = true
    let connection = NWConnection(to: device.endpoint, using: parameters)
    let attempt = PairingConnection(connection: connection, role: .mac, queue: pairingQueue) { [weak self] event in
      DispatchQueue.main.async {
        guard let self, self.isPairing else { return }
        switch event {
        case .code(let code):
          self.pairingCode = code
          self.pairingMessage = "Scan this QR code on your iPhone, or enter the code manually."
        case .paired(let key):
          do {
            try PairingTrustStore.save(key)
            self.hasTrust = true
            self.autoConnectEligible = true
            self.pairingMessage = "Paired. Start encrypted Wi-Fi on your iPhone."
          } catch { self.pairingMessage = "Could not save pairing securely." }
          self.isPairing = false; self.pairingCode = nil; self.pairingAttempt = nil
        case .failed(let message):
          self.pairingMessage = message
          self.isPairing = false; self.pairingCode = nil; self.pairingAttempt = nil
        }
      }
    }
    pairingAttempt = attempt
    attempt.start()
  }

  func cancelPairing() {
    isPairing = false
    pairingAttempt?.cancel(); pairingAttempt = nil; pairingCode = nil
    pairingMessage = "Pairing cancelled."
  }

  func forgetPairing() {
    do { try PairingTrustStore.remove(); hasTrust = false; pairingMessage = "Pairing removed." }
    catch { pairingMessage = "Could not remove pairing." }
  }
  @Published var pairingKey = ""
  @Published var status = "Pair your iPhone to connect."
  @Published var isRunning = false
  @Published var isConnected = false
  @Published private(set) var useUSB = false
  @Published private(set) var isSwitchingMode = false
  private let relay = Relay()
  private var tetherCam: Process?

  func start() {
    guard !useUSB, !isSwitchingMode else { return }
    let host = address.trimmingCharacters(in: .whitespacesAndNewlines)
    let endpoint = devices.first(where: { $0.id == selectedDevice })?.endpoint ?? (host.isEmpty ? nil : NWEndpoint.hostPort(host: NWEndpoint.Host(host), port: 7878))
    let key = (try? PairingTrustStore.load()) ?? SecureTransport.keyFromHex(pairingKey.trimmingCharacters(in: .whitespacesAndNewlines))
    guard let endpoint, let key else {
      status = "Select a nearby iPhone and pair it first."
      return
    }
    guard startReceiverIfNeeded() else { return }
    isRunning = true
    isConnected = false
    status = "Starting secure connection…"
    relay.start(endpoint: endpoint, key: key) { [weak self] state in
      DispatchQueue.main.async {
        guard let self = self, self.isRunning else { return }
        switch state {
        case .ready:
          self.status = "Waiting for TetherCam. Approve the connection on iPhone."
        case .connected:
          self.isConnected = true
          self.status = "Encrypted connection established. Approve it on iPhone."
        case .disconnected(let message):
          self.isConnected = false
          self.status = message
        case .failed(let message):
          self.status = message
          self.stopRelay()
        }
      }
    }
  }

  private func startReceiverIfNeeded() -> Bool {
    if let process = tetherCam, process.isRunning { return true }
    let executable = Bundle.main.bundleURL.appendingPathComponent("Contents/Helpers/TetherCam.app/Contents/MacOS/TetherCam").path
    guard FileManager.default.isExecutableFile(atPath: executable) else {
      status = "The built-in camera receiver is missing. Reinstall GlassesData."
      return false
    }
    guard NSWorkspace.shared.runningApplications.allSatisfy({ $0.bundleIdentifier != "at.gotzendorfer.tethercam.mac" }) else {
      status = "Quit TetherCam first, then connect again."
      return false
    }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = useUSB ? ["--headless"] : ["--headless", "--debug-tcp", "127.0.0.1:7879"]
    process.terminationHandler = { [weak self] process in
      DispatchQueue.main.async {
        guard let self, self.tetherCam === process else { return }
        self.tetherCam = nil
        self.isConnected = false
        self.status = "TetherCam exited (code \(process.terminationStatus)). Check that its camera extension is enabled in System Settings."
        self.stopRelay()
      }
    }
    do {
      try process.run()
      tetherCam = process
      return true
    } catch {
      status = "Could not launch TetherCam: \(error.localizedDescription)"
      return false
    }
  }

  func stop() {
    autoConnectEligible = false
    stopRelay()
    status = "Disconnected."
  }

  func selectUSB(_ selected: Bool) {
    guard selected != useUSB, !isSwitchingMode else { return }
    if isPairing { cancelPairing() }
    stopRelay()
    useUSB = selected
    autoConnectEligible = !selected
    isSwitchingMode = true
    status = "Switching connection…"
    let previous = tetherCam
    tetherCam = nil
    if let previous, previous.isRunning { previous.terminate() }
    DispatchQueue.global(qos: .utility).async { [weak self] in
      if let previous { previous.waitUntilExit() }
      DispatchQueue.main.async {
        guard let self else { return }
        self.isSwitchingMode = false
        guard self.startReceiverIfNeeded() else { return }
        if selected {
          self.status = "USB receiver ready. Start USB broadcasting on your iPhone and approve the connection."
        } else if self.hasTrust && !self.devices.isEmpty {
          self.autoConnectEligible = false
          self.start()
        } else {
          self.status = "Pair your iPhone to connect."
        }
      }
    }
  }

  private func stopRelay() {
    isRunning = false
    isConnected = false
    relay.stop()
    pairingKey = ""
  }

  func shutdown() {
    stopRelay()
    let process = tetherCam
    tetherCam = nil
    if let process, process.isRunning { process.terminate() }
  }
}

final class Relay: @unchecked Sendable {
  enum State { case ready, connected, disconnected(String), failed(String) }
  private let queue = DispatchQueue(label: "GlassesData.MacRelay")
  private var listener: NWListener?
  private var pair: (NWConnection, NWConnection)?
  private var update: ((State) -> Void)?

  func start(endpoint: NWEndpoint, key: Data, update: @escaping (State) -> Void) {
    queue.async { [self] in
      stopOnQueue()
      self.update = update
      let parameters = NWParameters.tcp
      parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: 7879)
      parameters.allowLocalEndpointReuse = true
      do {
        let listener = try NWListener(using: parameters)
        self.listener = listener
        listener.stateUpdateHandler = { [weak self, weak listener] state in
          guard let self, let listener, self.listener === listener else { return }
          switch state {
          case .ready: self.update?(.ready)
          case .failed(let error): self.update?(.failed("Local connection failed: \(error.localizedDescription)")); self.stopOnQueue()
          default: break
          }
        }
        listener.newConnectionHandler = { [weak self] local in
          guard let self, self.pair == nil else { local.cancel(); return }
          let parameters = SecureTransport.parameters(key: key)
          parameters.includePeerToPeer = true
          let remote = NWConnection(to: endpoint, using: parameters)
          self.pair = (local, remote)
          local.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed(let error):
              self?.update?(.disconnected("TetherCam receiver connection failed: \(error.localizedDescription)"))
              self?.close(local, remote)
            case .cancelled: self?.close(local, remote)
            default: break
            }
          }
          remote.stateUpdateHandler = { [weak self] state in
            guard let self, let pair = self.pair, pair.0 === local else { return }
            switch state {
            case .ready:
              self.update?(.connected)
              self.forward(local, remote); self.forward(remote, local)
            case .failed(let error):
              self.update?(.disconnected("Encrypted iPhone connection failed: \(error.localizedDescription)"))
              self.close(local, remote)
            case .cancelled: self.close(local, remote)
            default: break
            }
          }
          local.start(queue: self.queue)
          remote.start(queue: self.queue)
          self.queue.asyncAfter(deadline: .now() + 10) { [weak self] in
            guard let self, let pair = self.pair, pair.0 === local, remote.state != .ready else { return }
            self.update?(.disconnected("Connection timed out. Check the iPhone and network."))
            self.close(local, remote)
          }
        }
        listener.start(queue: queue)
      } catch { self.update?(.failed("Could not start local connection: \(error.localizedDescription)")); self.stopOnQueue() }
    }
  }

  func stop() { queue.async { self.stopOnQueue() } }
  private func stopOnQueue() {
    if let pair { pair.0.cancel(); pair.1.cancel() }
    pair = nil
    listener?.cancel(); listener = nil; update = nil
  }
  private func close(_ local: NWConnection, _ remote: NWConnection) {
    guard let pair, (pair.0 === local || pair.1 === local) else { return }
    self.pair = nil
    local.cancel(); remote.cancel()
  }
  private func forward(_ from: NWConnection, _ to: NWConnection) {
    from.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, complete, error in
      guard let self else { return }
      guard error == nil, !complete, let data, !data.isEmpty else {
        self.close(from, to)
        self.update?(.disconnected("Disconnected. Waiting for TetherCam to reconnect."))
        return
      }
      to.send(content: data, completion: .contentProcessed { [weak self] error in
        guard let self else { return }
        if error != nil { self.close(from, to) } else { self.forward(from, to) }
      })
    }
  }
}

struct ContentView: View {
  @ObservedObject var model: CameraRelayModel
  var body: some View {
    VStack(alignment: .leading, spacing: 20) {
      HStack(spacing: 12) {
        Image(systemName: "video.badge.waveform")
          .font(.system(size: 22, weight: .medium))
          .frame(width: 46, height: 46)
          .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
        VStack(alignment: .leading, spacing: 3) {
          Text("Mac camera").font(.title2.weight(.semibold))
          Text("GlassesData").foregroundStyle(.secondary)
        }
      }
      Text("Connect your iPhone and use the glasses camera in Mac apps.")
        .foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
      GroupBox("iPhone connection") {
        VStack(alignment: .leading, spacing: 12) {
          Picker("Connection", selection: Binding(get: { model.useUSB }, set: { model.selectUSB($0) })) {
            Text("Encrypted Wi-Fi").tag(false)
            Text("USB").tag(true)
          }.disabled(model.isSwitchingMode)
          if model.useUSB {
            Text("Connect your iPhone with a data cable, trust this Mac, and start USB broadcasting on your iPhone.")
              .font(.caption).foregroundStyle(.secondary)
          } else {
          if model.devices.isEmpty {
            Text(model.discoveryStatus).font(.callout).foregroundStyle(.secondary)
          } else {
            Picker("Nearby iPhone", selection: $model.selectedDevice) {
              ForEach(model.devices) { device in Text(device.name).tag(device.id) }
            }.disabled(model.isRunning)
          }
          DisclosureGroup("Manual connection") {
            TextField("iPhone IP address", text: $model.address)
              .textFieldStyle(.roundedBorder).disabled(model.isRunning)
            if !model.hasTrust {
              SecureField("Pairing key", text: $model.pairingKey)
                .textFieldStyle(.roundedBorder).disabled(model.isRunning)
            }
          }
          Text("Start encrypted Wi-Fi in the Mac camera panel on your iPhone.")
            .font(.caption).foregroundStyle(.secondary)
          }
        }.padding(6)
      }
      if !model.useUSB {
      GroupBox("Pair a device") {
        VStack(alignment: .leading, spacing: 10) {
          Text(model.pairingMessage).font(.callout).foregroundStyle(.secondary)
          if !model.pairingDevices.isEmpty && !model.isPairing {
            Picker("iPhone", selection: $model.selectedPairingDevice) {
              ForEach(model.pairingDevices) { device in Text(device.name).tag(device.id) }
            }
          }
          if let code = model.pairingCode {
            VStack(spacing: 8) {
              PairingQRCode(code: code)
              Text(code).font(.system(size: 22, weight: .semibold, design: .monospaced))
            }
            .frame(maxWidth: .infinity)
          }
          HStack {
            Button(model.isPairing ? "Cancel" : "Pair with iPhone") {
              model.isPairing ? model.cancelPairing() : model.beginPairing()
            }.disabled(model.isRunning || (!model.isPairing && model.pairingDevices.isEmpty))
            if model.hasTrust {
              Button("Forget device") { model.forgetPairing() }.disabled(model.isRunning || model.isPairing)
            }
          }
        }.padding(6).frame(maxWidth: .infinity, alignment: .leading)
      }
      }
      HStack(alignment: .top, spacing: 8) {
        Circle().fill(model.isConnected ? Color.green : Color.secondary).frame(width: 8, height: 8).padding(.top, 5)
        Text(model.status).font(.callout).foregroundStyle(.secondary)
      }.frame(maxWidth: .infinity, alignment: .leading)
      HStack {
        Link("Camera extension settings", destination: URL(string: "x-apple.systempreferences:com.apple.LoginItems-Settings.extension")!)
        Spacer()
        if !model.useUSB {
          Button(model.isRunning ? "Disconnect" : "Connect") {
            model.isRunning ? model.stop() : model.start()
          }.keyboardShortcut(.defaultAction).disabled(model.isPairing || model.isSwitchingMode)
        }
      }
      Text("TetherCam receives video locally on this Mac. Only connect on a Mac you trust.")
        .font(.caption).foregroundStyle(.secondary)
    }
    .padding(24)
    .frame(width: 460)
  }
}

@main
struct GlassesDataMacApp: App {
  @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
  @StateObject private var model = CameraRelayModel()

  var body: some Scene {
    WindowGroup("GlassesData") {
      ContentView(model: model)
        .onAppear { appDelegate.stopReceiver = { [model] in model.shutdown() } }
    }
      .windowResizability(.contentSize)
  }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
  var stopReceiver: (() -> Void)?
  func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
  func applicationWillTerminate(_ notification: Notification) { stopReceiver?() }
}


/// QR code carrying the short authentication code shown to both peers.
struct PairingQRCode: View {
  let code: String

  var body: some View {
    Group {
      if let image = Self.makeImage(for: code) {
        Image(nsImage: image)
          .interpolation(.none)
          .resizable()
          .scaledToFit()
          .padding(12)
          .background(Color.white, in: RoundedRectangle(cornerRadius: 12))
          .accessibilityLabel("QR code for pairing code \(code)")
      } else {
        Text(code)
          .font(.system(size: 31, weight: .bold, design: .monospaced))
          .accessibilityLabel("Pairing code \(code)")
      }
    }
    .frame(width: 230, height: 230)
  }

  private static func makeImage(for code: String) -> NSImage? {
    let filter = CIFilter.qrCodeGenerator()
    filter.message = Data(code.utf8)
    filter.correctionLevel = "M"
    guard let output = filter.outputImage,
      let cgImage = CIContext().createCGImage(output, from: output.extent) else { return nil }
    return NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
  }
}
