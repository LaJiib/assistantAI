//
//  ChatToolbarView.swift
//  AssistantIA
//
//  Created by JB SENET on 01/02/2026.
//

import SwiftUI

/// Toolbar view for the chat interface that displays error messages and conversation controls.
struct ChatToolbarView: View {
    /// View model containing the chat state and controls
    @Bindable var vm: ConversationViewModel
    @State private var isConfirmingDelete = false

    var body: some View {
        // Display error message if present
        if let errorMessage = vm.errorMessage {
            ErrorView(errorMessage: errorMessage)
        }

        // Button to clear chat history
        Button {
            vm.clear([.chat, .meta])
        } label: {
            Label("Effacer", systemImage: "eraser")
        }
        
        Button {
            isConfirmingDelete = true
        } label: {
            Label("Supprimer", systemImage: "trash")
                .foregroundStyle(.red)
        }
        .confirmationDialog("Supprimer cette conversation ?",
                             isPresented: $isConfirmingDelete,
                             titleVisibility: .visible) {
            Button("Supprimer", role: .destructive) {
                vm.deleteConversation()
            }
            Button("Annuler", role: .cancel) { }
        }
    }
}
