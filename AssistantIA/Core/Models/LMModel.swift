//
//  LMModel.swift
//  AssistantIA
//
//  Created by JB SENET on 01/02/2026.
//

import MLXLMCommon

/// Represents a language model configuration with its associated properties and type.
/// Can represent either a large language model (LLM) or a vision-language model (VLM).
struct LMModel {
    /// Name of the model
    let name: String

    /// Configuration settings for model initialization
    let configuration: ModelConfiguration
    
}

// MARK: - Helpers

extension LMModel {
    /// Display name with additional "(Vision)" suffix for vision models
    var displayName: String {name}
}


extension LMModel: Identifiable, Hashable {
    var id: String {
        name
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(name)
    }
}
