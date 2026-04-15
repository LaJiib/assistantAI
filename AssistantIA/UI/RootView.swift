//
//  RootView.swift
//  AssistantIA
//
//  Created by JB SENET on 01/02/2026.
//

import SwiftUI

struct RootView: View {
    @Bindable var conversationManager: ConversationManager
    @Environment(BackendManager.self) private var backendManager
    @State private var showBackendSettings = false

    init(conversationManager: ConversationManager) {
        self.conversationManager = conversationManager
    }

    var body: some View {
        NavigationSplitView {
            ConversationListView(manager: conversationManager)
                .navigationSplitViewColumnWidth(min: 250, ideal: 300)
        } detail: {
            if let viewModel = conversationManager.activeViewModel {
                ChatView(viewModel: viewModel)
            } else {
                ContentUnavailableView(
                    "No Conversation Selected",
                    systemImage: "bubble.left.and.bubble.right"
                )
            }
        }
        .toolbar {
            backendStatusToolbar
        }
        .sheet(isPresented: $showBackendSettings) {
            NavigationStack {
                BackendSettingsView()
                    .toolbar {
                        ToolbarItem(placement: .cancellationAction) {
                            Button("Fermer") { showBackendSettings = false }
                        }
                    }
            }
            .environment(backendManager)
            .frame(minWidth: 400, minHeight: 460)
        }
    }

    @ToolbarContentBuilder
    private var backendStatusToolbar: some ToolbarContent {
        ToolbarItem(placement: .navigation) {
            Button {
                showBackendSettings = true
            } label: {
                backendStatusLabel
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
            .help("État backend Python — cliquez pour les détails")
        }
    }

    @ViewBuilder
    private var backendStatusLabel: some View {
        switch backendManager.state {
        case .stopped:
            Label("Backend arrêté", systemImage: "circle")
                .foregroundStyle(.secondary)
                .labelStyle(.iconOnly)
                .font(.title3)
        case .connecting:
            Label("Backend en connexion", systemImage: "circle.dotted")
                .foregroundStyle(.orange)
                .labelStyle(.iconOnly)
                .symbolEffect(.pulse)
                .font(.title3)
        case .running:
            Label("Backend actif", systemImage: "circle.fill")
                .foregroundStyle(.green)
                .labelStyle(.iconOnly)
                .font(.title3)
        case .error:
            Label("Erreur backend", systemImage: "exclamationmark.circle.fill")
                .foregroundStyle(.red)
                .labelStyle(.iconOnly)
                .font(.title3)
        }
    }
    
}
