//
//  Message.swift
//  AssistantIA
//
//  Created by JB SENET on 31/01/2026.
//

import Foundation

struct Message: Identifiable, Equatable, Decodable {
    let id: UUID
    let role: Role
    var parts: [MessagePart]
    let timestamp: Date

    enum Role: String, Decodable {
        case user, assistant, system, tool
    }

    // MARK: - Factory methods (usage local / previews)

    static func user(_ content: String) -> Message {
        Message(id: UUID(), role: .user, parts: [.text(content)], timestamp: Date())
    }

    static func assistant(_ text: String = "") -> Message {
        let parts: [MessagePart] = text.isEmpty ? [] : [.text(text)]
        return Message(id: UUID(), role: .assistant, parts: parts, timestamp: Date())
    }

    static func system(_ content: String) -> Message {
        Message(id: UUID(), role: .system, parts: [.text(content)], timestamp: Date())
    }

    // MARK: - Helpers

    /// Concatène le contenu de toutes les parts de type text.
    /// Utilisé pour la génération du titre et l'annulation propre.
    var textContent: String {
        parts.filter { $0.type == .text }.compactMap { $0.content }.joined()
    }
}

struct MessagePart: Equatable, Decodable {
    let type: PartType
    var content: String?
    var toolCallId: String?
    var toolName: String?

    enum PartType: String, Decodable {
        case text, reasoning, toolCall, toolResult
    }

    // MARK: - Factory methods

    static func text(_ content: String) -> MessagePart {
        MessagePart(type: .text, content: content)
    }

    static func reasoning(_ content: String = "") -> MessagePart {
        MessagePart(type: .reasoning, content: content)
    }

    static func toolCall(id: String, name: String) -> MessagePart {
        MessagePart(type: .toolCall, content: nil, toolCallId: id, toolName: name)
    }

    static func toolResult(id: String, content: String) -> MessagePart {
        MessagePart(type: .toolResult, content: content, toolCallId: id)
    }
}
