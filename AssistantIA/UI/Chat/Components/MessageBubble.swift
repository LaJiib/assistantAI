//
//  MessageBubble.swift
//  AssistantIA
//
//  Created by JB SENET on 31/01/2026.
//

import SwiftUI

struct MessageBubble: View {
    let message: Message
    
    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            if message.role == .user {
                Spacer(minLength: 60)
            }
            
            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                Text(message.content)
                    .padding(12)
                    .background(backgroundColor)
                    .foregroundColor(textColor)
                    .cornerRadius(16)
                
                Text(message.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 4)
            }
            
            if message.role == .assistant {
                Spacer(minLength: 60)
            }
        }
    }
    
    private var backgroundColor: Color {
        message.role == .user ? .blue : Color.secondary.opacity(0.15)
    }
    
    private var textColor: Color {
        message.role == .user ? .white : .primary
    }
}

#Preview {
    VStack(spacing: 12) {
        MessageBubble(message: Message(role: .user, content: "Bonjour !"))
        MessageBubble(message: Message(role: .assistant, content: "Bonjour ! Comment puis-je vous aider ?"))
    }
    .padding()
}
