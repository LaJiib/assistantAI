//
//  ChatView.swift
//  AssistantIA
//
//  Created by JB SENET on 31/01/2026.
//

import SwiftUI

/// Main chat interface view that manages the conversation UI and user interactions.
/// Displays messages, handles media attachments, and provides input controls.
struct ChatView: View {
    /// View model that manages the chat state and business logic
    @Bindable private var vm: ConversationViewModel
    @Environment(BackendManager.self) private var backendManager

    /// Initializes the chat view with a view model
    /// - Parameter viewModel: The view model to manage chat state
    init(viewModel: ConversationViewModel) {
        self.vm = viewModel
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Display conversation history
                ConversationView(messages: vm.messages)

                Divider()

                // Input field with send and media attachment buttons
                PromptField(
                    prompt: $vm.prompt,
                    options: $vm.messageOptions,
                    sendButtonAction: vm.generate,
                    canSend: canSend
                )
                .padding()
            }
            .navigationTitle("AssistantIA")
            .toolbar {
                ChatToolbarView(vm: vm)
            }
        }
    }

    private var canSend: Bool {
        if case .running = backendManager.state { return true }
        return false
    }
}

#Preview {
    // 1. Création d'un ID de test
    let testID = UUID()
    
    // 2. Création d'objets de test valides (Mocks)
    let conversation = Conversation(id: testID)
    
    let metadata = ConversationMetadata(
        id: testID,
        title: "Conversation de test",
        createdAt: Date(),
        updatedAt: Date(),
        messageCount: 0,
        specificInstruction: nil // On initialise avec rien
    )
    
    // 3. Initialisation du ViewModel avec les deux arguments requis
    let viewModel = ConversationViewModel(
        conversation: conversation,
        metadata: metadata,
        onDelete: { }
    )
    
    ChatView(viewModel: viewModel)
}
