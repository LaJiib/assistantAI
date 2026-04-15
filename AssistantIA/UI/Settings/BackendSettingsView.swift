// BackendSettingsView.swift
// AssistantIA
//
// Vue de monitoring et reconnexion au backend Python.
// Le backend est lancé indépendamment (VSCode / terminal).

import SwiftUI
internal import Combine

struct BackendSettingsView: View {
    @Environment(BackendManager.self) private var backendManager
    @State private var errorExpanded = false
    @State private var connectingAt: Date? = nil

    var body: some View {
        Form {
            statusSection
            actionsSection
            if case .error(let msg) = backendManager.state {
                errorSection(msg)
            }
            infoSection
        }
        .formStyle(.grouped)
        .navigationTitle("Backend Python")
        .onChange(of: backendManager.state) { _, newState in
            if case .connecting = newState { connectingAt = Date() }
            else { connectingAt = nil }
        }
    }

    // MARK: - Status Section

    private var statusSection: some View {
        Section("État") {
            HStack(spacing: 12) {
                statusIndicator
                VStack(alignment: .leading, spacing: 2) {
                    Text(backendManager.state.displayText)
                        .font(.body)
                        .fontWeight(.medium)
                    if case .connecting = backendManager.state, let t = connectingAt {
                        ElapsedTimeView(since: t)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                Spacer()
                if case .connecting = backendManager.state {
                    ProgressView()
                        .controlSize(.small)
                }
            }
            .padding(.vertical, 4)
        }
    }

    @ViewBuilder
    private var statusIndicator: some View {
        switch backendManager.state {
        case .stopped:
            Image(systemName: "circle")
                .foregroundStyle(.secondary)
                .font(.title2)
        case .connecting:
            Image(systemName: "circle.dotted")
                .foregroundStyle(.orange)
                .font(.title2)
                .symbolEffect(.pulse)
        case .running:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .font(.title2)
        case .error:
            Image(systemName: "xmark.circle.fill")
                .foregroundStyle(.red)
                .font(.title2)
        }
    }

    // MARK: - Actions Section

    private var actionsSection: some View {
        Section("Connexion") {
            Button {
                Task { await backendManager.reconnect() }
            } label: {
                Label("Reconnecter", systemImage: "arrow.clockwise")
            }
            .disabled({
                if case .connecting = backendManager.state { return true }
                return false
            }())
        }
    }

    // MARK: - Error Section

    @ViewBuilder
    private func errorSection(_ message: String) -> some View {
        Section {
            DisclosureGroup("Détail de l'erreur", isExpanded: $errorExpanded) {
                ScrollView(.vertical) {
                    Text(message)
                        .font(.system(.footnote, design: .monospaced))
                        .foregroundStyle(.primary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .padding(8)
                }
                .frame(maxHeight: 180)
                .background(Color(NSColor.textBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }

            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(message, forType: .string)
            } label: {
                Label("Copier l'erreur", systemImage: "doc.on.doc")
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.secondary)
        } header: {
            Label("Erreur de connexion", systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
        }
    }

    // MARK: - Info Section

    private var infoSection: some View {
        Section("Informations") {
            LabeledContent("URL") {
                Text(backendManager.baseURL.absoluteString)
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }

            if let pid = backendManager.pid {
                LabeledContent("PID") {
                    Text(String(pid))
                        .font(.system(.body, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
            }

            if case .running(let port) = backendManager.state {
                LabeledContent("Port") {
                    Text(String(port))
                        .font(.system(.body, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
            }

            LabeledContent("Lancer le backend") {
                Text("cd backend && python main.py")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.tertiary)
                    .textSelection(.enabled)
            }
        }
    }
}

// MARK: - Elapsed Time Subview

/// Affiche le temps écoulé depuis une date (ex: "20s") et se met à jour chaque seconde.
private struct ElapsedTimeView: View {
    let since: Date
    @State private var elapsed: Int = 0
    private let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()

    var body: some View {
        Text("En cours depuis \(elapsed)s...")
            .onReceive(timer) { _ in
                elapsed = Int(Date().timeIntervalSince(since))
            }
            .onAppear {
                elapsed = Int(Date().timeIntervalSince(since))
            }
    }
}

// MARK: - Preview

#Preview {
    NavigationStack {
        BackendSettingsView()
    }
    .environment(BackendManager())
    .frame(width: 420, height: 460)
}
