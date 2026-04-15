// ConversationViewModel.swift
// AssistantIA
//
// Orchestrateur UI ↔ ConversationAPI.
//
// Responsabilités :
//   - Append local immédiat des messages (UI réactive via @Observable)
//   - Délègue génération + persistance à ConversationAPI (streaming SSE typé)
//   - Route les StreamEvent vers les parts du message assistant en cours
//   - Génère le titre au 1er échange via ChatAPI (stateless, sans persistance message)
//   - Notifie ConversationManager via onTitleGenerated (cache local)
//   - Gère l'annulation mid-génération

import SwiftUI

@Observable
@MainActor
class ConversationViewModel {

    // MARK: - State

    var conversation: Conversation
    var metadata: ConversationMetadata
    var prompt: String = ""
    var isGenerating = false
    var messageOptions: [String: Bool] = ["think": false]
    var errorMessage: String? = nil

    var messages: [Message] {
        get { conversation.messages }
        set { conversation.messages = newValue }
    }

    // MARK: - Private

    private let conversationAPI: ConversationAPI
    private var generateTask: Task<Void, any Error>?
    private var titleGenerated: Bool

    private let onTitleGenerated: (@Sendable (String) -> Void)?
    private let onDelete: @Sendable () -> Void

    // MARK: - Init

    init(
        conversation: Conversation,
        metadata: ConversationMetadata,
        conversationAPI: ConversationAPI = .shared,
        onTitleGenerated: (@Sendable (String) -> Void)? = nil,
        onDelete: @escaping @Sendable () -> Void
    ) {
        self.conversation = conversation
        self.metadata = metadata
        self.conversationAPI = conversationAPI
        self.onTitleGenerated = onTitleGenerated
        self.onDelete = onDelete
        self.titleGenerated = conversation.messages.count > 3
    }

    // MARK: - Generate

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

        // Coquille vide pour l'assistant — parts se rempliront via handleStreamEvent
        messages.append(.assistant())
        let assistantIndex = messages.count - 1

        generateTask = Task {
            for try await event in conversationAPI.sendMessage(
                conversationID: conversation.id,
                content: userContent,
                options: messageOptions
            ) {
                await MainActor.run {
                    self.handleStreamEvent(event, at: assistantIndex)
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
                        self.messages[lastIndex].parts.append(.text("\n[Annulé]"))
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

    // MARK: - Stream Event Routing

    /// Route un StreamEvent vers les parts du message assistant à l'index donné.
    /// Appelé sur MainActor depuis la boucle de streaming.
    private func handleStreamEvent(_ event: StreamEvent, at index: Int) {
        guard messages.indices.contains(index) else { return }

        switch event.type {
        case .textDelta:
            guard let content = event.content, !content.isEmpty else { return }
            appendOrUpdateLastPart(type: .text, content: content, at: index)

        case .reasoningDelta:
            guard let content = event.content, !content.isEmpty else { return }
            appendOrUpdateLastPart(type: .reasoning, content: content, at: index)

        case .toolCallStart:
            guard let id = event.toolCallId, let name = event.toolName else { return }
            messages[index].parts.append(.toolCall(id: id, name: name))

        case .toolCallResult:
            guard let id = event.toolCallId else { return }
            // Marquer la part toolCall comme complétée et attacher l'aperçu
            if let partIdx = messages[index].parts.firstIndex(where: {
                $0.type == .toolCall && $0.toolCallId == id
            }) {
                messages[index].parts[partIdx].isCompleted = true
                messages[index].parts[partIdx].preview = event.preview
            } else {
                // Fallback : part toolCall absente (ne devrait pas arriver)
                messages[index].parts.append(MessagePart(
                    type: .toolResult, content: nil, toolCallId: id, toolName: nil,
                    preview: event.preview, isCompleted: true
                ))
            }

        case .start, .done, .error, .unknown:
            break
        }
    }

    /// Ajoute un delta au dernier part du type spécifié, ou crée un nouveau part
    /// si le dernier part est d'un type différent (ex: passage texte → reasoning).
    private func appendOrUpdateLastPart(type: MessagePart.PartType, content: String, at index: Int) {
        if let lastIdx = messages[index].parts.indices.last,
           messages[index].parts[lastIdx].type == type {
            messages[index].parts[lastIdx].content =
                (messages[index].parts[lastIdx].content ?? "") + content
        } else {
            messages[index].parts.append(MessagePart(type: type, content: content))
        }
    }

    // MARK: - Title Generation

    private func generateTitleWithAI() async {
        guard messages.count == 3, !titleGenerated else { return }
        let firstUserMessage = messages.first(where: { $0.role == .user })?.textContent ?? ""
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
