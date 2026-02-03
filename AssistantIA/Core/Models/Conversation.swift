//
//  Conversation.swift
//  AssistantIA
//
//  Created by JB SENET on 02/02/2026.
//

import Foundation

struct Conversation: Identifiable, Codable, Equatable {
    let id: UUID
    let systemPrompt: String
    var messages: [Message]

    init(id: UUID, systemPrompt: String) {
        self.id = id
        self.systemPrompt = systemPrompt
        self.messages = [.system(systemPrompt)]
    }
}
