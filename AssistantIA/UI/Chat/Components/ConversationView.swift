//
//  ConversationView.swift
//  AssistantIA
//
//  Created by JB SENET on 01/02/2026.
//

import SwiftUI

/// Displays the chat conversation as a scrollable list of messages.
struct ConversationView: View {
    /// Array of messages to display in the conversation
    let messages: [Message]

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                ForEach(messages) { message in
                    MessageView(message)
                        .padding(.horizontal, 12)
                }
            }
        }
        .padding(.vertical, 8)
        .defaultScrollAnchor(.bottom, for: .sizeChanges)
    }
}

#Preview {
    // Display sample conversation in preview
    ConversationView(messages: [])
}
