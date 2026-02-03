//
//  MessageView.swift
//  AssistantIA
//
//  Created by JB SENET on 01/02/2026.
//

import AVKit
import SwiftUI

/// A view that displays a single message in the chat interface.
/// Supports different message roles (user, assistant, system) and media attachments.
struct MessageView: View {
    /// The message to be displayed
    let message: Message

    /// Creates a message view
    /// - Parameter message: The message model to display
    init(_ message: Message) {
        self.message = message
    }

    var body: some View {
        switch message.role {
        case .user:
            // User messages are right-aligned with blue background
            HStack {
                Spacer()
                VStack(alignment: .trailing, spacing: 8) {
                    // Message content with tinted background.
                    // LocalizedStringKey used to trigger default handling of markdown content.
                    Text(LocalizedStringKey(message.content))
                        .padding(.vertical, 8)
                        .padding(.horizontal, 12)
                        .background(.tint, in: .rect(cornerRadius: 16))
                        .textSelection(.enabled)
                }
            }

        case .assistant:
            // Assistant messages are left-aligned without background
            // LocalizedStringKey used to trigger default handling of markdown content.
            HStack {
                Text(LocalizedStringKey(message.content))
                    .textSelection(.enabled)

                Spacer()
            }

        case .system:
            // System messages are centered with computer icon
            Label(message.content, systemImage: "desktopcomputer")
                .font(.headline)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .center)
        }
    }
}

#Preview {
    VStack(spacing: 20) {
        MessageView(.system(Constants.defaultSystemPrompt))

        MessageView(
            .user(
                "Salut"
            )
        )

        MessageView(.assistant("Bonjour, comment puis-je vous aider aujourd'hui ?"))
    }
    .padding()
}
