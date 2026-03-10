//
//  Message.swift
//  AssistantIA
//
//  Created by JB SENET on 31/01/2026.
//

import Foundation

struct Message: Identifiable, Equatable, Codable {
    let id: UUID
    let role: Role
    var content: String
    /// Array of image URLs attached to the message
    var images: [URL]
    let timestamp: Date
    
    enum Role: Codable {
        case user
        case assistant
        case system
    }
    
    init(role: Role, content: String, images: [URL] = []) {
        self.id = UUID()
        self.role = role
        self.content = content
        self.images = images
        self.timestamp = Date()
    }
}

extension Message {
    /// Creates a user message with optional media attachments
    /// - Parameters:
    ///   - content: The text content of the message
    /// - Returns: A new Message instance with user role
    static func user(_ content: String, images: [URL] = []) -> Message {
        Message(role: .user, content: content, images: images)
    }

    /// Creates an assistant message
    /// - Parameter content: The text content of the message
    /// - Returns: A new Message instance with assistant role
    static func assistant(_ content: String) -> Message {
        Message(role: .assistant, content: content)
    }

    /// Creates a system message
    /// - Parameter content: The text content of the message
    /// - Returns: A new Message instance with system role
    static func system(_ content: String) -> Message {
        Message(role: .system, content: content)
    }
}
