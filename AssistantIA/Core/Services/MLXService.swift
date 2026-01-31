//
//  MLXService.swift
//  AssistantIA
//
//  Created by JB SENET on 31/01/2026.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Service gérant le modèle MLX pour la génération de texte
actor MLXService {
    
    // MARK: - Properties
    
    /// Cache pour éviter de recharger le modèle
    private let modelCache = NSCache<NSString, ModelContainer>()
    
    /// Configuration du modèle
    private let modelConfiguration: ModelConfiguration
    
    /// Clé de cache pour notre modèle unique
    private let cacheKey = "mistral-nemo-local" as NSString
    
    /// Statut du chargement
    private(set) var isLoaded = false
    
    // MARK: - Initialization
    
    init() {
        // Créer la configuration depuis le directory local
        let modelURL = URL(fileURLWithPath: Constants.modelPath)
        
        self.modelConfiguration = ModelConfiguration(
            directory: modelURL,
            tokenizerId: nil,
            overrideTokenizer: nil,
            defaultPrompt: Constants.defaultSystemPrompt,
            extraEOSTokens: []
        )
    }
    
    // MARK: - Model Loading
    
    /// Charge le modèle (ou le récupère du cache)
    private func loadModel() async throws -> ModelContainer {
        // Limiter la mémoire GPU pour éviter les crashes
        // Note: Limitation de mémoire GPU non disponible via `Memory.cacheLimit` ici.
        // Si nécessaire, configurez les paramètres du modèle/contexte fournis par MLX ou ajustez les tailles de tensors.
        
        // Vérifier le cache
        if let container = modelCache.object(forKey: cacheKey) {
            print("📦 Modèle récupéré du cache")
            return container
        }
        
        print("📦 Chargement du modèle depuis : \(Constants.modelPath)")
        
        // Charger le modèle via LLMModelFactory
        let container = try await LLMModelFactory.shared.loadContainer(
            hub: defaultHubApi,
            configuration: modelConfiguration
        ) { progress in
            let percentage = Int(progress.fractionCompleted * 100)
            print("⏳ Chargement : \(percentage)%")
        }
        
        // Mettre en cache
        modelCache.setObject(container, forKey: cacheKey)
        isLoaded = true
        
        print("✅ Modèle chargé et prêt")
        return container
    }
    
    /// Generates text based on the provided messages using the specified model.
    /// - Parameters:
    ///   - messages: Array of chat messages including user, assistant, and system messages
    ///   - model: The language model to use for generation
    /// - Returns: An AsyncStream of generated text tokens
    /// - Throws: Errors that might occur during generation
    func generate(messages: [Message]) async throws -> AsyncStream<Generation> {
        // Load or retrieve model from cache
        let modelContainer = try await loadModel()

        // Map app-specific Message type to Chat.Message for model input
        let chatMessages = messages.map { message in
            let role: Chat.Message.Role =
                switch message.role {
                case .assistant:
                    .assistant
                case .user:
                    .user
                case .system:
                    .system
                }

            return Chat.Message(
                role: role, content: message.content)
        }

        // Prepare input for model processing
        let userInput = UserInput(
            chat: chatMessages, processing: .init())

        // Generate response using the model
        return try await modelContainer.perform { (context: ModelContext) in
            let lmInput = try await context.processor.prepare(input: userInput)
            let parameters = await GenerateParameters(maxTokens: Constants.defaultMaxTokens,
                temperature: Constants.defaultTemperature,)

            return try MLXLMCommon.generate(
                input: lmInput, parameters: parameters, context: context)
        }
    }
}

