//
//  MLXService.swift
//  AssistantIA
//
//  Created by JB SENET on 31/01/2026.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
internal import Tokenizers


/// A service class that manages machine learning models for text and vision-language tasks.
/// This class handles model loading, caching, and text generation using various LLM and VLM models.
@Observable
class MLXService {

    /// Cache to store loaded model containers to avoid reloading.
    private var loadedContainer: ModelContainer? = nil

    /// Tracks the current model download progress.
    /// Access this property to monitor model download status.
    @MainActor
    private(set) var modelDownloadProgress: Progress?
    
    private(set) var currentModel: LMModel?
    
    
    init() {}

    /// Loads the current model container, or returns the already loaded one.
    /// - Returns: The active ModelContainer
    /// - Throws: Errors that might occur during model loading
    private func loadContainer() async throws -> ModelContainer {
        if let container = loadedContainer {
            return container
        }
        guard let model = currentModel else {
            throw MLXServiceError.modelNotLoaded
        }
        let container = try await LLMModelFactory.shared.loadContainer(
            hub: defaultHubApi, configuration: model.configuration
        ) { progress in
            Task { @MainActor in
                self.modelDownloadProgress = progress
            }
        }
        loadedContainer = container
        return container
    }

    
    func loadLocalModel(at path: String) async throws {
        let configuration = ModelConfiguration(directory: URL(filePath: path))
        currentModel = LMModel(name: "Ministral3-14B-instruct", configuration: configuration)
        let container = try await loadContainer()
        print("🔍 === DIAGNOSTIC PROCESSOR ===")
        print("🔍 Type du processor: \(type(of: container.processor))")
        print("🔍 Nom du type: \(String(describing: type(of: container.processor)))")
    }

    
    func unloadModel() {
        loadedContainer = nil
        currentModel = nil
        Memory.clearCache()
    }

    /// Generates text based on the provided messages using the specified model.
    /// - Parameters:
    ///   - messages: Array of chat messages including user, assistant, and system messages
    ///   - model: The language model to use for generation
    /// - Returns: An AsyncStream of generated text tokens
    /// - Throws: Errors that might occur during generation
    func generate(messages: [Message]) async throws -> AsyncStream<Generation> {
        // Load or retrieve model from cache
        guard currentModel != nil else {
                    throw MLXServiceError.modelNotLoaded
                }

        let modelContainer = try await loadContainer()

        // Map app-specific Message type to Chat.Message for model input
        let chat = messages.map { message in
            let role: Chat.Message.Role =
                switch message.role {
                case .assistant:
                    .assistant
                case .user:
                    .user
                case .system:
                    .system
                }
        let images: [UserInput.Image] = message.images.map { imageURL in .url(imageURL) }
            
            return Chat.Message(
                role: role, content: message.content, images: images)
        }

        // Prepare input for model processing
        let userInput = UserInput(
            chat: chat, processing: .init(resize: .init(width: 1540, height: 1540)))
        // DEBUG 1 : Avant prepare
        print("📥 UserInput AVANT prepare:")
        print("   Prompt: \(userInput.prompt)")
        print("   Chat: \(chat)")
        
        return try await modelContainer.perform { (context: ModelContext) in
            let lmInput = try await context.processor.prepare(input: userInput)
            
            // DEBUG 2 : Après prepare
            print("📤 LMInput APRÈS prepare:")
            print("   Type: \(type(of: lmInput))")
            // Si tu peux accéder aux tokens :
            print("   Tokens: \(lmInput.text.tokens)")
            
            // DÉCODE les tokens pour voir le texte
            let tokenIds = lmInput.text.tokens.asArray(Int.self)
            let decodedText = context.tokenizer.decode(tokens: tokenIds)
            print("   📝 Texte décodé: \(decodedText)")
            let parameters = await GenerateParameters(maxTokens: Constants.defaultMaxTokens,
                temperature: Constants.defaultTemperature)

            return try MLXLMCommon.generate(
                input: lmInput, parameters: parameters, context: context)
        }
    }
}

enum MLXServiceError: LocalizedError {
    case modelNotLoaded
    case modelLoadFailed(String)
    
    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Aucun modèle n'est chargé. Veuillez charger un modèle d'abord."
        case .modelLoadFailed(let reason):
            return "Échec du chargement du modèle: \(reason)"
        }
    }
}
