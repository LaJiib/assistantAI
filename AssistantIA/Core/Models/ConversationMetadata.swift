//
//  ConversationMetadata.swift
//  AssistantIA
//
//  Created by JB SENET on 03/02/2026.
//

import Foundation

struct ConversationMetadata: Identifiable, Codable, Equatable {
    let id: UUID
    var title: String
    let systemPrompt: String
    let createdAt: Date
    var updatedAt: Date
    var messageCount: Int          // pour afficher "X messages" sans charger le fichier
}
