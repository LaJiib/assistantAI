// ConversationViewModel.swift
// AssistantIA
//
// Created by JB SENET on 02/02/2026.
//
// ── Phase 2 : DÉSACTIVÉ MLXService → ChatAPI ──────────────────────────────────
// Toute génération passe maintenant par le backend Python via ChatAPI.shared.
// MLXService est conservé dans l'init pour compatibilité avec ConversationManager
// (rollback : décommenter le code marqué ROLLBACK et recommenter le code PHASE 2).
// ─────────────────────────────────────────────────────────────────────────────

// ROLLBACK : remettre ces imports si retour MLXService
// import MLXLMCommon
import SwiftUI
import UniformTypeIdentifiers

@Observable
@MainActor
class ConversationViewModel {
    var conversation: Conversation

    // ROLLBACK : décommenter si retour MLXService
    // private let mlxService: MLXService

    // PHASE 2 : mlxService conservé dans l'init mais ignoré (compat ConversationManager)
    // Génération via ChatAPI.shared

    /// Current user input text
    var prompt: String = ""

    /// Indicates if text generation is in progress
    var isGenerating = false

    /// Current generation task, used for cancellation
    private var generateTask: Task<Void, any Error>?

    // ROLLBACK : réactiver si retour MLXService
    // private var generateCompletionInfo: GenerateCompletionInfo?

    private var titleGenerated: Bool = false

    var errorMessage: String? = nil

    var messages: [Message] {
        get { conversation.messages }
        set { conversation.messages = newValue }
    }

    // PHASE 2 : tok/s non disponible via ChatAPI (loggué server-side par le backend)
    // ROLLBACK : retourner generateCompletionInfo?.tokensPerSecond ?? 0
    var tokensPerSecond: Double { 0 }

    // PHASE 2 : aucun téléchargement MLX local
    // ROLLBACK : retourner mlxService.modelDownloadProgress
    var modelDownloadProgress: Progress? { nil }

    private let onConversationUpdated: @Sendable (Conversation, String?) -> Void
    private let onDelete: @Sendable () -> Void

    // Signature inchangée pour compatibilité ConversationManager (mlxService ignoré en Phase 2)
    init(
        conversation: Conversation,
        mlxService: MLXService,
        onConversationUpdated: @escaping @Sendable (Conversation, String?) -> Void,
        onDelete: @escaping @Sendable () -> Void
    ) {
        self.conversation = conversation
        // PHASE 2 : mlxService ignoré — génération via ChatAPI.shared
        // self.mlxService = mlxService  // ROLLBACK : décommenter
        self.onConversationUpdated = onConversationUpdated
        self.titleGenerated = conversation.messages.count > 3
        self.onDelete = onDelete
    }

    // MARK: - Generate

    /// Génère une réponse pour le prompt courant via le backend Python (streaming SSE).
    func generate() async {
        // Annuler toute génération en cours
        if let existingTask = generateTask {
            existingTask.cancel()
            generateTask = nil
        }

        isGenerating = true
        errorMessage = nil

        // Ajouter le message utilisateur
        messages.append(.user(prompt))
        clear(.prompt)

        // PHASE 2 : plus de guard sur mlxService.currentModel
        // Si le backend est down, l'erreur sera catchée ci-dessous.
        // ROLLBACK :
        // guard mlxService.currentModel != nil else {
        //     errorMessage = "Model not loaded. Please load the model first."
        //     isGenerating = false
        //     return
        // }

        let builtPrompt = buildPrompt(from: messages)

        generateTask = Task {
            // PHASE 2 : ChatAPI.shared.streamMessage
            // ROLLBACK : for await generation in try await mlxService.generate(messages: messages)
            //            { switch generation { case .chunk(let c): ...; case .info(let i):
            //              generateCompletionInfo = i; case .toolCall: break } }
            for try await chunk in ChatAPI.shared.streamMessage(builtPrompt) {
                if let lastIndex = messages.indices.last,
                   messages[lastIndex].role == .assistant {
                    messages[lastIndex].content += chunk
                } else {
                    messages.append(.assistant(chunk))
                }
            }
        }

        do {
            try await withTaskCancellationHandler {
                try await generateTask?.value
            } onCancel: {
                Task { @MainActor in
                    self.generateTask?.cancel()
                    if let lastIndex = self.messages.indices.last,
                       self.messages[lastIndex].role == .assistant {
                        self.messages[lastIndex].content += "\n[Annulé]"
                    }
                }
            }
        } catch is CancellationError {
            // Annulation volontaire — message [Annulé] déjà ajouté dans onCancel
        } catch {
            errorMessage = error.localizedDescription
        }

        onConversationUpdated(conversation, nil)
        await generateTitleWithAI()
        isGenerating = false
        generateTask = nil
    }

    // MARK: - Title Generation

    private func generateTitleWithAI() async {
        // Génèrer le titre uniquement au 1er échange (système + user + assistant = 3 messages)
        guard messages.count == 3, !titleGenerated else { return }

        let firstUserMessage = messages.first { $0.role == .user }?.content ?? ""
        guard !firstUserMessage.isEmpty else { return }

        // PHASE 2 : sendMessage non-streaming (court, titre)
        // ROLLBACK : utiliser mlxService.generate(messages: titleMessages) avec switch
        let titlePrompt = """
        System: Your only task is to generate a very short title (max 10 words) \
        that summarizes the user's request. Respond with only the title, nothing else.

        User: \(firstUserMessage)
        """

        do {
            let rawTitle = try await ChatAPI.shared.sendMessage(
                titlePrompt,
                maxTokens: 30,
                temperature: 0.1
            )
            let cleanedTitle = rawTitle
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
            if !cleanedTitle.isEmpty {
                onConversationUpdated(conversation, cleanedTitle)
                titleGenerated = true
            }
        } catch {
            // En cas d'erreur (backend down, etc.), on garde "Nouvelle Conversation"
            print("[ConversationViewModel] ⚠️ Erreur génération titre: \(error)")
        }
    }

    // MARK: - Prompt Builder

    /// Formate l'historique de la conversation en texte structuré pour le backend.
    ///
    /// Le backend accepte `prompt: str` (Phase 2). Le modèle comprend ce format
    /// naturellement. Phase 3 mettra à jour l'API pour prendre `messages: List[Dict]`.
    private func buildPrompt(from messages: [Message]) -> String {
        messages.map { message in
            switch message.role {
            case .system:    "System: \(message.content)"
            case .user:      "User: \(message.content)"
            case .assistant: "Assistant: \(message.content)"
            }
        }
        .joined(separator: "\n\n")
    }

    // MARK: - Helpers

    func deleteConversation() {
        onDelete()
    }

    /// Clears various aspects of the chat state based on provided options.
    func clear(_ options: ClearOption) {
        if options.contains(.prompt) {
            prompt = ""
        }
        if options.contains(.chat) {
            conversation.messages = [.system(conversation.systemPrompt)]
            generateTask?.cancel()
        }
        if options.contains(.meta) {
            // PHASE 2 : generateCompletionInfo supprimé
            // ROLLBACK : generateCompletionInfo = nil
        }
        errorMessage = nil
    }

    /// Annule la génération en cours si active (appelé lors d'un unload).
    func cancelGenerationTaskIfNeeded() {
        generateTask?.cancel()
        generateTask = nil
    }
}
