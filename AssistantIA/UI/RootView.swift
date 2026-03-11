//
//  RootView.swift
//  AssistantIA
//
//  Created by JB SENET on 01/02/2026.
//

import SwiftUI

struct RootView: View {
    @Bindable var modelStateManager: ModelStateManager
    @Bindable var conversationManager: ConversationManager
    @Environment(BackendManager.self) private var backendManager
    @State private var showBackendSettings = false

    init(modelStateManager: ModelStateManager, conversationManager: ConversationManager) {
        self.modelStateManager = modelStateManager
        self.conversationManager = conversationManager
    }
    
    var body: some View {
        switch modelStateManager.modelState {
        case .notConfigured, .error:
            SetupView(modelStateManager: modelStateManager)
            
        case .configured, .loading, .loaded:
            mainView
        }
    }
    
    private var mainView: some View {
        VStack(spacing: 0) {
            // Banner si modèle pas chargé
            if case .configured = modelStateManager.modelState {
                HStack {
                    Image(systemName: "info.circle")
                    Text("Model not loaded. Click toolbar button to load.")
                    Spacer()
                }
                .padding()
                .background(Color.orange.opacity(0.2))
            }
            
            // Navigation principale
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
                modelControlToolbar
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
    }
    
    @ToolbarContentBuilder
    private var modelControlToolbar: some ToolbarContent {
        ToolbarItem(placement: .primaryAction) {
            modelControlButton
        }
    }

    @ToolbarContentBuilder
    private var backendStatusToolbar: some ToolbarContent {
        ToolbarItem(placement: .status) {
            Button {
                showBackendSettings = true
            } label: {
                backendStatusLabel
            }
            .buttonStyle(.borderless)
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
        case .starting:
            Label("Backend en démarrage", systemImage: "circle.dotted")
                .foregroundStyle(.orange)
                .labelStyle(.iconOnly)
                .symbolEffect(.pulse)
        case .running:
            Label("Backend actif", systemImage: "circle.fill")
                .foregroundStyle(.green)
                .labelStyle(.iconOnly)
        case .error:
            Label("Erreur backend", systemImage: "exclamationmark.circle.fill")
                .foregroundStyle(.red)
                .labelStyle(.iconOnly)
        }
    }
    
    @ViewBuilder
    private var modelControlButton: some View {
        switch modelStateManager.modelState {
        case .notConfigured:
            EmptyView()
            
        case .configured:
            Button {
                Task {
                    try? await modelStateManager.loadModel()
                }
            } label: {
                Label("Load Model", systemImage: "arrow.down.circle.fill")
            }
            
        case .loading:
            ProgressView()
            
        case .loaded:
            Button {
                modelStateManager.unloadModel()
            } label: {
                Label("Unload Model", systemImage: "arrow.up.circle.fill")
            }
            
        case .error(let message):
            Menu {
                Button("Retry") {
                    Task {
                        try? await modelStateManager.loadModel()
                    }
                }
                Text(message)
            } label: {
                Label("Error", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
            }
        }
    }
}
