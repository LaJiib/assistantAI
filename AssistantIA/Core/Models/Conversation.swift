//
//  Conversation.swift
//  AssistantIA
//
//  Created by JB SENET on 02/02/2026.
//

import Foundation

struct Conversation: Identifiable, Codable, Equatable {
    let id: UUID
    var title: String
    let systemPrompt: String
    
    var messages: [Message]
    
    let createdAt: Date
    var updatedAt: Date?
    
    init(systemPrompt: String, title: String = "Nouvelle Conversation" ) {
        self.id = UUID()
        self.title = title
        self.systemPrompt = systemPrompt
        self.messages = [.system(systemPrompt)]
        let now = Date()
        self.createdAt = now
        self.updatedAt = now
    }
    
    init(
        id: UUID,
        title: String,
        systemPrompt: String,
        messages: [Message],
        createdAt: Date,
        updatedAt: Date
    ) {
        self.id = id
        self.title = title
        self.systemPrompt = systemPrompt
        self.messages = messages
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
    
    mutating func touch () {
        self.updatedAt = Date()
    }
    
    mutating func addMessage(_ message: Message) {
        messages.append(message)
        touch()
    }
}
