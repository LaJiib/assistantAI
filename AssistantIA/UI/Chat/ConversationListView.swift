//
//  ConversationListView.swift
//  AssistantIA
//
//  Created by JB SENET on 02/02/2026.
//

import SwiftUI

struct ConversationListView: View {
    @Bindable var manager: ConversationManager
    
    var body: some View{
        List(selection: $manager.activeConversationID) {
            ForEach(manager.conversations) { conversation in
                NavigationLink(value: conversation.id) {
                    VStack(alignment: .leading) {
                        Text(conversation.title)
                            .font(.headline)
                        Text("\(conversation.messages.count) messages")
                            .font(.caption)
                            .foregroundStyle(.secondary)
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
        for index in offsets {
            let conversation = manager.conversations[index]
            manager.deleteConversation(id: conversation.id)
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
