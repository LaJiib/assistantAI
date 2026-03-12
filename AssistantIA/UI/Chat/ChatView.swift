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
    let conversation = Conversation(id: UUID(), systemPrompt: "Preview conversation")
    let viewModel = ConversationViewModel(
        conversation: conversation,
        onDelete: { }
    )
    ChatView(viewModel: viewModel)
}
