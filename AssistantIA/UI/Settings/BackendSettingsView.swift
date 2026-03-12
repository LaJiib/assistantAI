// BackendSettingsView.swift
// AssistantIA
//
// Vue de monitoring et contrôle du backend Python.
// Accessible via Settings / Debug — permet Start/Stop manuel et diagnostic.

import SwiftUI
internal import Combine

struct BackendSettingsView: View {
    @Environment(BackendManager.self) private var backendManager
    @State private var errorExpanded = false
    @State private var isRestarting = false
    @State private var startedAt: Date? = nil

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
            if case .starting = newState { startedAt = Date() }
            if case .running = newState  { startedAt = nil }
            if case .stopped = newState  { startedAt = nil }
            if case .error   = newState  { startedAt = nil }
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
                    if case .starting = backendManager.state, let t = startedAt {
                        ElapsedTimeView(since: t)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                Spacer()
                if case .starting = backendManager.state {
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
        case .starting:
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
        Section("Contrôles") {
            switch backendManager.state {
            case .stopped, .error:
                Button {
                    Task { try? await backendManager.start() }
                } label: {
                    Label("Démarrer le backend", systemImage: "play.fill")
                }

            case .starting:
                Button(role: .destructive) {
                    backendManager.stop()
                } label: {
                    Label("Annuler le démarrage", systemImage: "stop.fill")
                }

            case .running:
                Button(role: .destructive) {
                    backendManager.stop()
                } label: {
                    Label("Arrêter le backend", systemImage: "stop.fill")
                }

                Button {
                    Task {
                        isRestarting = true
                        backendManager.stop()
                        // Attendre que l'arrêt soit effectif
                        try? await Task.sleep(nanoseconds: 500_000_000)
                        try? await backendManager.start()
                        isRestarting = false
                    }
                } label: {
                    HStack {
                        Label("Redémarrer", systemImage: "arrow.clockwise")
                        if isRestarting { ProgressView().controlSize(.mini) }
                    }
                }
                .disabled(isRestarting)
            }
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
            Label("Erreur de démarrage", systemImage: "exclamationmark.triangle.fill")
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

            LabeledContent("Logs") {
                Text("Console Xcode — filtre \"[Backend]\"")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
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
    .frame(width: 420, height: 520)
}
