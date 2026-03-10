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
    
    /// Manages image and video attachments for the current message
    var mediaSelection = MediaSelection()
    
    // Indicates if text generation is in progress
    var isGenerating = false
    
    /// Current generation task, used for cancellation
    private var generateTask: Task<Void, any Error>?

    /// Stores performance metrics from the current generation
    private var generateCompletionInfo: GenerateCompletionInfo?
    
    private var titleGenerated: Bool = false
    
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
    private let onConversationUpdated: @Sendable (Conversation, String?) -> Void

    private let onDelete: @Sendable () -> Void
    
    init(conversation: Conversation, mlxService: MLXService, onConversationUpdated: @escaping @Sendable (Conversation, String?) -> Void, onDelete: @escaping @Sendable () -> Void) {
        self.conversation = conversation
        self.mlxService = mlxService
        self.onConversationUpdated = onConversationUpdated
        self.titleGenerated = conversation.messages.count > 3
        self.onDelete = onDelete
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
        messages.append(.user(prompt, images: mediaSelection.images))

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
        onConversationUpdated(conversation, nil)
        await generateTitleWithAI()
        isGenerating = false
        generateTask = nil
    }
    
    private func generateTitleWithAI() async {
        // On ne lance ça que si le modèle est chargé et qu'on est au 1er échange
        guard mlxService.currentModel != nil else { return }
        guard messages.count == 3 && !titleGenerated else { return }

        let firstUserMessage = messages.first { $0.role == .user }?.content ?? ""
        guard !firstUserMessage.isEmpty else { return }

        // Prompt spécifique pour la génération de titre — court, direct, pas de bavardage
        let titleMessages: [Message] = [
            .system("Your only task is to generate a very short title (max 10 words) that summarizes the user's request and the purpose of the chat. Respond with only the title, nothing else."),
            .user(firstUserMessage)
        ]

        do {
            var title = ""
            for await generation in try await mlxService.generate(messages: titleMessages) {
                switch generation {
                case .chunk(let chunk):
                    title += chunk
                case .info, .toolCall:
                    break
                }
            }
            // Nettoyer le titre (retirer les guillemets, trim, etc.)
            let cleanedTitle = title
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
            if !cleanedTitle.isEmpty {
                onConversationUpdated(conversation, cleanedTitle)  // sauvegarder avec le nouveau titre
                titleGenerated = true
            }
        } catch {
            // En cas d'erreur, on garde "Nouvelle Conversation" — pas de crash
            print("⚠️ Erreur génération titre: \(error)")
        }
    }
    
    /// Processes and adds media attachments to the current message
    func addMedia(_ result: Result<URL, any Error>) {
        do {
            let url = try result.get()

            // Determine media type and add to appropriate collection
            if let mediaType = UTType(filenameExtension: url.pathExtension) {
                if mediaType.conforms(to: .image) {
                    mediaSelection.images = [url]
                }
            }
        } catch {
            errorMessage = "Failed to load media item.\n\nError: \(error)"
        }
    }
    
    func deleteConversation() {
        onDelete()
    }


    /// Clears various aspects of the chat state based on provided options
    func clear(_ options: ClearOption) {
        if options.contains(.prompt) {
            prompt = ""
            mediaSelection = .init()
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
    
    //nouvelle fonction pour éventuellement arréter les tâches lors d'un unload
    func cancelGenerationTaskIfNeeded() {
        generateTask?.cancel()
        generateTask = nil
    }

}

/// Manages the state of media attachments in the chat
@Observable
class MediaSelection {
    /// Controls visibility of media selection UI
    var isShowing = false

    /// Currently selected image URLs
    var images: [URL] = [] {
        didSet {
            didSetURLs(oldValue, images)
        }
    }

    /// Currently selected video URLs
    var videos: [URL] = [] {
        didSet {
            didSetURLs(oldValue, videos)
        }
    }

    /// Whether any media is currently selected
    var isEmpty: Bool {
        images.isEmpty && videos.isEmpty
    }

    private func didSetURLs(_ old: [URL], _ new: [URL]) {
        // the urls we get from fileImporter require SSB calls to access
        new.filter { !old.contains($0) }.forEach { _ = $0.startAccessingSecurityScopedResource() }
        old.filter { !new.contains($0) }.forEach { $0.stopAccessingSecurityScopedResource() }
    }
}
