//
//  ChatView.swift
//  AssistantIA
//
//  Created by JB SENET on 31/01/2026.
//

import AVFoundation
import AVKit
import SwiftUI

/// Main chat interface view that manages the conversation UI and user interactions.
/// Displays messages, handles media attachments, and provides input controls.
struct ChatView: View {
    /// View model that manages the chat state and business logic
    @Bindable private var vm: ConversationViewModel

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
                )
                .padding()
            }
            .navigationTitle("MLX Chat Example")
            .toolbar {
                ChatToolbarView(vm: vm)
            }
        }
    }
}

#Preview {
    let service = MLXService()
    let conversation = Conversation(systemPrompt: "Preview conversation")
    let viewModel = ConversationViewModel(conversation: conversation, mlxService: service)
    return ChatView(viewModel: viewModel)
}
