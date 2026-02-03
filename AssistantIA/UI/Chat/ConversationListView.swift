//
//  ConversationListView.swift
//  AssistantIA
//
//  Created by JB SENET on 02/02/2026.
//

import SwiftUI

struct ConversationListView: View {
    @Bindable var manager: ConversationManager
    @State private var editingConversationID: UUID? = nil
    
    var body: some View{
        List(selection: $manager.activeConversationID) {
            ForEach(manager.metadata.sorted { $0.updatedAt > $1.updatedAt }) { meta in
                NavigationLink(value: meta.id) {
                    VStack(alignment: .leading) {
                        Text(meta.title)
                            .font(.headline)
                        Text("\(meta.messageCount) messages")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .contextMenu {
                    Button {
                        editingConversationID = meta.id
                    } label: {
                        Label("Renommer", systemImage: "pencil")
                    }

                    Button(role: .destructive) {
                        manager.deleteConversation(id: meta.id)
                    } label: {
                        Label("Supprimer", systemImage: "trash")
                    }
                }
            }
            .onDelete(perform: deleteConversations)
        }
        .navigationTitle("Conversations")
        .toolbar{
            Button {
                manager.createConversation()
            } label: {
                Label("New Conversation", systemImage: "plus")
            }
        }
    }
    
    func deleteConversations(at offsets: IndexSet) {
        let sorted = manager.metadata.sorted { $0.updatedAt > $1.updatedAt }
        for index in offsets {
            manager.deleteConversation(id: sorted[index].id)
        }
    }
}

/// Options for clearing different aspects of the chat state
struct ClearOption: RawRepresentable, OptionSet {
    let rawValue: Int

    /// Clears current prompt and media selection
    static let prompt = ClearOption(rawValue: 1 << 0)
    /// Clears chat history and cancels generation
    static let chat = ClearOption(rawValue: 1 << 1)
    /// Clears generation metadata
    static let meta = ClearOption(rawValue: 1 << 2)
}
