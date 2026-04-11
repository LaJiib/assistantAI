//
//  Conversation.swift
//  AssistantIA
//
//  Created by JB SENET on 02/02/2026.
//

import Foundation

struct Conversation: Identifiable, Codable, Equatable {
    let id: UUID
    var messages: [Message]

    init(id: UUID) {
        self.id = id
        self.messages = []
    }

    init(id: UUID, messages: [Message]) {
        self.id = id
        self.messages = messages
    }
}
