//
// StorageCard.swift
//
// Recording on this iPhone: the "Save on this iPhone" switch (off by default, since
// recordings take a lot of space), the destination folder with its free space, and the
// outcome of the latest recording. The folder is the app's own Recordings folder or one
// picked in Files: On My iPhone, iCloud Drive or a USB drive. Draws no chrome of its own;
// it sits inside a card on the camera screen.
//

import SwiftUI
import UIKit
import UniformTypeIdentifiers

struct StorageCard: View {
  var viewModel: CameraViewModel
  @State private var showPicker = false
  @State private var notice: String?

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 12) {
        toggleRow
        folderRow
        if let problem = viewModel.recordingFolder.problem {
          message(problem, symbol: "exclamationmark.triangle.fill", tint: .orange)
            .accessibilityIdentifier("recording_folder_problem")
        }
        if let notice {
          message(notice, symbol: "exclamationmark.triangle.fill", tint: .red)
        }
        if let last = viewModel.lastSaveMessage {
          message(
            last, symbol: viewModel.lastSaveFailed ? "xmark.octagon.fill" : "checkmark.circle.fill",
            tint: viewModel.lastSaveFailed ? .red : .green
          )
          .accessibilityIdentifier("last_save_message")
        }
        Text(viewModel.recordingFolder.isAppFolder
          ? "Find recordings in Files › On My iPhone › GlassesData › Recordings."
          : "Each recording is written on this iPhone first, then moved into this folder when it ends.")
          .font(.caption)
          .foregroundStyle(.secondary)
          .fixedSize(horizontal: false, vertical: true)
      }
      .frame(maxWidth: .infinity, alignment: .leading)
    }
    .scrollBounceBehavior(.basedOnSize)
    .scrollIndicators(.hidden)
    .onAppear { viewModel.refreshRecordingFolder() }
    .sheet(isPresented: $showPicker) {
      FolderPicker { url in
        showPicker = false
        guard let url else { return }
        notice = viewModel.chooseRecordingFolder(url)
      }
      .ignoresSafeArea()
    }
  }

  // MARK: - Rows

  private var toggleRow: some View {
    HStack(alignment: .center, spacing: 12) {
      Image(systemName: "iphone")
        .font(.system(size: 17, weight: .semibold))
        .foregroundStyle(viewModel.saveOnPhone ? Color.appPrimary : Color.gray)
        .frame(width: 38, height: 38)
        .background(Circle().fill((viewModel.saveOnPhone ? Color.appPrimary : Color.gray).opacity(0.18)))
        .accessibilityHidden(true)
      VStack(alignment: .leading, spacing: 2) {
        Text("Save on this iPhone")
          .font(.headline)
          .lineLimit(1)
          .minimumScaleFactor(0.8)
        Text(viewModel.saveOnPhone
          ? "Record also writes video and sensor logs here."
          : "Off: Record saves on the computer only.")
          .font(.caption)
          .foregroundStyle(.secondary)
          .fixedSize(horizontal: false, vertical: true)
      }
      Spacer(minLength: 8)
      Toggle("Save on this iPhone", isOn: saveBinding)
        .labelsHidden()
        .tint(Color.appPrimary)
        .disabled(viewModel.isRecording || viewModel.isPreparingRecording)
        .accessibilityIdentifier("save_on_phone_toggle")
    }
  }

  private var folderRow: some View {
    let folder = viewModel.recordingFolder
    return HStack(spacing: 10) {
      Image(systemName: folder.isAppFolder ? "folder.fill" : folderSymbol(folder))
        .font(.system(size: 15, weight: .semibold))
        .foregroundStyle(Color.appPrimary)
        .frame(width: 22)
        .accessibilityHidden(true)
      VStack(alignment: .leading, spacing: 1) {
        Text(verbatim: folder.name)
          .font(.subheadline.weight(.semibold))
          .lineLimit(1)
          .truncationMode(.middle)
        HStack(spacing: 0) {
          Text(verbatim: folder.location)
            .lineLimit(1)
            .truncationMode(.middle)
          if let free = folder.freeBytes {
            Text(verbatim: " · \(free.formattedStorage) free")
              .monospacedDigit()
              .foregroundStyle(free < 2_000_000_000 ? Color.orange : Color.secondary)
              .lineLimit(1)
              .fixedSize()
              .accessibilityIdentifier("recording_folder_free_space")
          }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
      }
      Spacer(minLength: 8)
      Menu {
        Button("Choose a folder…", systemImage: "folder.badge.plus") { showPicker = true }
        if !folder.isAppFolder {
          Button("Use the app's Recordings folder", systemImage: "arrow.uturn.backward") {
            notice = nil
            viewModel.useAppRecordingFolder()
          }
        }
      } label: {
        Text("Change")
          .font(.footnote.weight(.semibold))
          .padding(.horizontal, 10)
          .padding(.vertical, 5)
          .background(Capsule().fill(Color.white.opacity(0.1)))
          .fixedSize()
      }
      .disabled(viewModel.isRecording || viewModel.isPreparingRecording || viewModel.isFinalizingRecording)
      .accessibilityLabel("Change folder")
      .accessibilityIdentifier("recording_folder_menu")
    }
    .padding(10)
    .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    .opacity(viewModel.saveOnPhone ? 1 : 0.55)
    .accessibilityElement(children: .contain)
    .accessibilityIdentifier("recording_folder")
  }

  private func message(_ text: String, symbol: String, tint: Color) -> some View {
    Label {
      Text(verbatim: text)
        .foregroundStyle(Color.white.opacity(0.85))
    } icon: {
      Image(systemName: symbol)
        .foregroundStyle(tint)
    }
    .font(.caption)
    .fixedSize(horizontal: false, vertical: true)
  }

  private func folderSymbol(_ folder: RecordingFolderInfo) -> String {
    switch folder.location {
    case "iCloud Drive": return "icloud.fill"
    case "Files": return "folder.fill"
    default: return "externaldrive.fill"
    }
  }

  private var saveBinding: Binding<Bool> {
    Binding(get: { viewModel.saveOnPhone }, set: { viewModel.setSaveOnPhone($0) })
  }
}

/// The Files folder picker. Calls back with the picked folder, or nil when cancelled.
struct FolderPicker: UIViewControllerRepresentable {
  let onPick: (URL?) -> Void

  func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
    let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.folder])
    picker.allowsMultipleSelection = false
    picker.delegate = context.coordinator
    return picker
  }

  func updateUIViewController(_ controller: UIDocumentPickerViewController, context: Context) {}

  func makeCoordinator() -> Coordinator { Coordinator(onPick: onPick) }

  @MainActor
  final class Coordinator: NSObject, UIDocumentPickerDelegate {
    let onPick: (URL?) -> Void

    init(onPick: @escaping (URL?) -> Void) {
      self.onPick = onPick
    }

    func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
      onPick(urls.first)
    }

    func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
      onPick(nil)
    }
  }
}
