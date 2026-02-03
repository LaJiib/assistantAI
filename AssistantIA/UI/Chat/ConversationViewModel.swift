//
//  ConversationView.swift
//  AssistantIA
//
//  Created by JB SENET on 02/02/2026.
//

import MLXLMCommon
import SwiftUI
import UniformTypeIdentifiers

@Observable
@MainActor
class ConversationViewModel {
    var conversation: Conversation
    private let mlxService: MLXService
    
    
    /// Current user input text
    var prompt: String = ""
    
    // Indicates if text generation is in progress
    var isGenerating = false
    
    /// Current generation task, used for cancellation
    private var generateTask: Task<Void, any Error>?

    /// Stores performance metrics from the current generation
    private var generateCompletionInfo: GenerateCompletionInfo?
    
    var errorMessage: String? = nil
    
    var messages: [Message] {
        get { conversation.messages }
        set { conversation.messages = newValue }
    }
    
    var tokensPerSecond: Double {
        generateCompletionInfo?.tokensPerSecond ?? 0
    }

    var modelDownloadProgress: Progress? {
        mlxService.modelDownloadProgress
    }
    
    init(conversation: Conversation, mlxService: MLXService) {
        self.conversation = conversation
        self.mlxService = mlxService
    }
    
    /// Generates response for the current prompt and media attachments
    func generate() async {
        // Cancel any existing generation task
        if let existingTask = generateTask {
            existingTask.cancel()
            generateTask = nil
        }

        isGenerating = true

        // Add user message with any media attachments
        messages.append(.user(prompt))

        // Clear the input after sending
        clear(.prompt)
        guard mlxService.currentModel != nil else {
            errorMessage = "Model not loaded. Please load the model first."
            isGenerating = false
            return
        }
        
        generateTask = Task {
            // Process generation chunks and update UI
            for await generation in try await mlxService.generate(
                messages: messages)
            {
                switch generation {
                case .chunk(let chunk):
                    // Vérifier si dernier message est assistant
                    if let lastIndex = messages.indices.last,
                       messages[lastIndex].role == .assistant {
                        // C'est un assistant existant, append
                        messages[lastIndex].content += chunk
                    } else {
                        // Pas d'assistant ou dernier est user, créer
                        messages.append(.assistant(chunk))
                    }
                case .info(let info):
                    // Update performance metrics
                    generateCompletionInfo = info
                case .toolCall(let call):
                    break
                }
            }
        }

        do {
            // Handle task completion and cancellation
            try await withTaskCancellationHandler {
                try await generateTask?.value
            } onCancel: {
                Task { @MainActor in
                    generateTask?.cancel()
                    
                    if let lastIndex = messages.indices.last,
                       messages[lastIndex].role == .assistant {
                        messages[lastIndex].content += "\n[Cancelled]"
                    }
                }
            }
        } catch {
            errorMessage = error.localizedDescription
        }

        isGenerating = false
        generateTask = nil
    }


    /// Clears various aspects of the chat state based on provided options
    func clear(_ options: ClearOption) {
        if options.contains(.prompt) {
            prompt = ""
        }

        if options.contains(.chat) {
            conversation.messages = [.system(conversation.systemPrompt)]
            generateTask?.cancel()
        }

        if options.contains(.meta) {
            generateCompletionInfo = nil
        }

        errorMessage = nil
    }
    
    func updateTitle(_ newTitle: String) {
        conversation.title = newTitle
    }
    
    private func touch() {
        conversation.updatedAt = Date()
    }
    
    //nouvelle fonction pour éventuellement arréter les tâches lors d'un unload
    func cancelGenerationTaskIfNeeded() {
        generateTask?.cancel()
        generateTask = nil
    }

}
