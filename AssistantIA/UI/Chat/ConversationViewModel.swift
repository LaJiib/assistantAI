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

    /// Envoie le prompt courant au backend et stream la réponse.
    ///
    /// Flux :
    ///   1. Append user message localement (UI immédiate)
    ///   2. `conversationAPI.sendMessage()` → backend sauvegarde user + génère assistant
    ///   3. Chunks SSE appended localement au fur et à mesure
    ///   4. Backend sauvegarde l'assistant complet (ou partiel si cancel) dans `finally`
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
            // Appel direct au nouvel endpoint Agent
            for try await chunk in ChatAPI.shared.streamAgentChat(prompt: userContent) {
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
                    // Message partiel conservé — cohérent avec ce que le backend a sauvegardé
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
            // Message user conservé localement même si erreur réseau :
            // on ne sait pas si le backend l'a reçu ou non.
        }

        await generateTitleWithAI()
        isGenerating = false
        generateTask = nil
    }

    // MARK: - Title Generation

    /// Génère un titre court au 1er échange (system + user + assistant = 3 messages).
    ///
    /// Utilise ChatAPI (endpoint /chat stateless) pour ne pas polluer l'historique.
    /// Met à jour le titre côté backend + notifie le Manager pour son cache local.
    private func generateTitleWithAI() async {
        guard messages.count == 3, !titleGenerated else { return }
        let firstUserMessage = messages.first(where: { $0.role == .user })?.content ?? ""
        guard !firstUserMessage.isEmpty else { return }

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
            guard !cleanedTitle.isEmpty else { return }

            // Persistance backend
            try? await conversationAPI.updateConversation(id: conversation.id, title: cleanedTitle)
            // Cache Manager
            onTitleGenerated?(cleanedTitle)
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
            conversation.messages = [.system(conversation.systemPrompt)]
            generateTask?.cancel()
        }
        errorMessage = nil
    }

    func cancelGenerationTaskIfNeeded() {
        generateTask?.cancel()
        generateTask = nil
    }
}
