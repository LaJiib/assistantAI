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
    
    @State private var showingSettings = false
    @State private var draftspecificInstruction = ""

    var body: some View {
        // Display error message if present
        if let errorMessage = vm.errorMessage {
            ErrorView(errorMessage: errorMessage)
        }
        Button {
            draftspecificInstruction = vm.metadata.specificInstruction ?? ""
            showingSettings = true
        } label: {
            Label("Paramètres", systemImage: "gearshape")
        }
        .sheet(isPresented: $showingSettings) {
            NavigationStack {
                Form {
                    Section(header: Text("Instructions Spécifiques")) {
                        TextEditor(text: $draftspecificInstruction)
                            .frame(minHeight: 150)
                    }
                    Section(footer: Text("Ces instructions seront ajoutées à l'identité de base d'Iris pour cette conversation uniquement.")) {}
                }
                .navigationTitle("Paramètres")
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) {
                        Button("Annuler") { showingSettings = false }
                    }
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Enregistrer") {
                            Task {
                                let newPrompt = draftspecificInstruction.trimmingCharacters(in: .whitespacesAndNewlines)
                                let finalPrompt = newPrompt.isEmpty ? nil : newPrompt
                                                
                                // Sauvegarde via l'API
                                try? await ConversationAPI.shared.updateConversation(
                                    id: vm.conversation.id,
                                    specificInstruction: finalPrompt
                                )
                                // Mise à jour locale
                                vm.metadata.specificInstruction = finalPrompt
                                showingSettings = false
                            }
                        }
                    }
                }
            }
            .frame(minWidth: 400, minHeight: 300)
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
