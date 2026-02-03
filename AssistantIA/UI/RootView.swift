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
    
    init(modelStateManager: ModelStateManager, conversationManager: ConversationManager) {
        self.modelStateManager = modelStateManager
        self.conversationManager = conversationManager
    }
    
    var body: some View {
        switch modelStateManager.modelState {
        case .notConfigured:
            SetupView(modelStateManager: modelStateManager)
            
        case .configured, .loading, .loaded, .error:
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
            }
        }
    }
    
    @ToolbarContentBuilder
    private var modelControlToolbar: some ToolbarContent {
        ToolbarItem(placement: .primaryAction) {
            modelControlButton
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
