// ConversationViewModel.swift
// AssistantIA
//
// Orchestrateur léger Phase 2bis : UI ↔ ConversationAPI.
//
// Responsabilités :
//   - Append local immédiat des messages (UI réactive via @Observable)
//   - Délègue génération + persistance à ConversationAPI (streaming SSE)
//   - Génère le titre au 1er échange via ChatAPI (stateless, sans persistance message)
//   - Notifie ConversationManager via onTitleGenerated (cache local)
//   - Gère l'annulation mid-génération avec message partiel [Annulé]

import SwiftUI

@Observable
@MainActor
class ConversationViewModel {

    // MARK: - State

    var conversation: Conversation
    var prompt: String = ""
    var isGenerating = false
    var errorMessage: String? = nil

    var messages: [Message] {
        get { conversation.messages }
        set { conversation.messages = newValue }
    }

    // MARK: - Private

    private let conversationAPI: ConversationAPI
    private var generateTask: Task<Void, any Error>?
    private var titleGenerated: Bool

    /// Callback léger : notifie ConversationManager quand le titre a changé.
    /// Le Manager met à jour son cache local (metadata[idx].title).
    /// La persistance backend est gérée directement par le ViewModel via conversationAPI.
    private let onTitleGenerated: (@Sendable (String) -> Void)?
    private let onDelete: @Sendable () -> Void

    // MARK: - Init

    init(
        conversation: Conversation,
        conversationAPI: ConversationAPI = .shared,
        onTitleGenerated: (@Sendable (String) -> Void)? = nil,
        onDelete: @escaping @Sendable () -> Void
    ) {
        self.conversation = conversation
        self.conversationAPI = conversationAPI
        self.onTitleGenerated = onTitleGenerated
        self.onDelete = onDelete
        // Titre déjà généré si l'échange initial (system+user+assistant) est passé
        self.titleGenerated = conversation.messages.count > 3
    }

    // MARK: - Generate

    /// Envoie le texte brut au backend (Agent Iris) et affiche la réponse.
    ///
    /// Flux :
    ///   1. Append user message localement (UI immédiate)
    ///   2. POST /agent/chat — backend orchestre l'agent loop (tool calling inclus)
    ///   3. Réponse finale appended en une fois (l'agent n'est pas streamable)
    func generate() async {
        if let existingTask = generateTask {
            existingTask.cancel()
            generateTask = nil
        }

        isGenerating = true
        errorMessage = nil

        let userContent = prompt
        messages.append(.user(userContent))
        clear(.prompt)

        generateTask = Task {
            // Délègue à ConversationAPI → /api/conversations/{id}/messages/stream
            // L'agent Iris (Pydantic AI) orchestre la génération côté backend.
            // La persistance user + assistant est gérée par le backend.
            for try await chunk in conversationAPI.sendMessage(
                conversationID: conversation.id,
                content: userContent
            ) {
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
            // Annulation volontaire
        } catch {
            errorMessage = error.localizedDescription
        }

        await generateTitleWithAI()
        isGenerating = false
        generateTask = nil
    }

    // MARK: - Title Generation

    /// Génère un titre court au 1er échange via POST /agent/title.
    ///
    /// Le frontend envoie uniquement le texte brut ; le backend gère le prompt engineering.
    private func generateTitleWithAI() async {
        guard messages.count == 3, !titleGenerated else { return }
        let firstUserMessage = messages.first(where: { $0.role == .user })?.content ?? ""
        guard !firstUserMessage.isEmpty else { return }

        do {
            let title = try await ChatAPI.shared.generateTitle(for: firstUserMessage)
            try? await conversationAPI.updateConversation(id: conversation.id, title: title)
            onTitleGenerated?(title)
            titleGenerated = true
        } catch {
            print("[ConversationViewModel] ⚠️ Titre: \(error)")
        }
    }

    // MARK: - Helpers

    func deleteConversation() {
        onDelete()
    }

    func clear(_ options: ClearOption) {
        if options.contains(.prompt) { prompt = "" }
        if options.contains(.chat) {
            conversation.messages = []
            generateTask?.cancel()
        }
        errorMessage = nil
    }

    func cancelGenerationTaskIfNeeded() {
        generateTask?.cancel()
        generateTask = nil
    }
}
