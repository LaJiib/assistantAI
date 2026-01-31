//
//  Constants.swift
//  AssistantIA
//
//  Created by JB SENET on 31/01/2026.
//

import Foundation

enum Constants {
    // Path vers le modèle sur votre SSD externe
    // À MODIFIER selon votre configuration
    static let modelPath = "/Volumes/AISSD/Models/mistral-nemo-instruct"
    
    // System prompt par défaut
    static let defaultSystemPrompt = "Tu t'appelles Iris, Tu es mon assistante personnelle IA serviable et précise."
    
    // Paramètres par défaut
    static let defaultTemperature: Float = 0.3
    static let defaultMaxTokens: Int = 512
}
