// ConversationManager.swift
// AssistantIA
//
// ── Phase 2bis-B : Wrapper REST API ───────────────────────────────────────────
// ConversationManager est un cache UI + proxy vers ConversationAPI.
// Toute persistence est gérée par le backend Python.
//
// Responsabilités :
//   - Cache local metadata[] pour rendu SwiftUI réactif
//   - Cache viewModelCache[UUID] pour préserver l'état UI (isGenerating, etc.)
//   - Délégation de toutes les opérations CRUD à ConversationAPI
// ─────────────────────────────────────────────────────────────────────────────

import Foundation

@Observable
@MainActor
class ConversationManager {

    // MARK: - State

    private(set) var metadata: [ConversationMetadata] = []
    var activeConversationID: UUID?

    /// Erreur du dernier loadConversations() — nil si succès.
    private(set) var loadError: String?

    // MARK: - Dependencies

    private let conversationAPI: ConversationAPI
    private var viewModelCache: [UUID: ConversationViewModel] = [:]

    // MARK: - Computed

    var activeViewModel: ConversationViewModel? {
        guard let id = activeConversationID else { return nil }
        return getViewModel(for: id)
    }

    // MARK: - Init

    /// - Parameter conversationAPI: client REST backend — défaut = shared singleton
    init(conversationAPI: ConversationAPI = .shared) {
        self.conversationAPI = conversationAPI
    }

    // MARK: - Chargement initial

    /// Charge la liste des conversations depuis le backend.
    ///
    /// Appelé par AssistantIAApp.startBackend() après que le backend soit prêt.
    /// Si la liste est vide → crée automatiquement une première conversation.
    func loadConversations() async throws {
        do {
            let loaded = try await conversationAPI.listConversations()
            metadata = loaded
            loadError = nil

            if activeConversationID == nil {
                activeConversationID = loaded.first?.id
            }
            if metadata.isEmpty {
                try await createConversation()
            }
        } catch let err as ConversationAPIError {
            loadError = err.errorDescription
            throw err
        }
    }

    // MARK: - CRUD

    /// Crée une nouvelle conversation via l'API.
    /// Async obligatoire : l'UUID est généré par le backend.
    @discardableResult
    func createConversation(
        title: String = "Nouvelle Conversation"
    ) async throws -> ConversationMetadata {
        let meta = try await conversationAPI.createConversation(
            title: title
        )
        metadata.append(meta)
        activeConversationID = meta.id
        return meta
    }

    /// Supprime une conversation : optimistic local update + sync backend.
    func deleteConversation(id: UUID) {
        metadata.removeAll { $0.id == id }
        viewModelCache.removeValue(forKey: id)

        if activeConversationID == id {
            activeConversationID = metadata.first?.id
        }

        Task {
            do {
                try await conversationAPI.deleteConversation(id: id)
            } catch {
                print("[ConversationManager] ⚠️ Erreur suppression \(id) : \(error)")
            }
        }
    }

    /// Renomme une conversation : optimistic local update + sync backend.
    func renameConversation(id: UUID, newTitle: String) {
        let trimmed = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard let idx = metadata.firstIndex(where: { $0.id == id }) else { return }

        metadata[idx].title = trimmed
        metadata[idx].updatedAt = Date()

        Task {
            do {
                try await conversationAPI.updateConversation(id: id, title: trimmed)
            } catch {
                print("[ConversationManager] ⚠️ Erreur renommage \(id) : \(error)")
            }
        }
    }

    func setActiveConversation(id: UUID) {
        guard metadata.contains(where: { $0.id == id }) else { return }
        activeConversationID = id
    }

    // MARK: - ViewModel

    /// Retourne le ViewModel (sync) avec chargement async des messages.
    ///
    /// Pattern :
    ///   1. Cache hit → retourne immédiatement (état UI préservé)
    ///   2. Cache miss → crée ViewModel avec Conversation placeholder
    ///                   → Task charge les vrais messages depuis l'API
    ///                   → @Observable déclenche le re-render SwiftUI automatiquement
    func getViewModel(for conversationID: UUID) -> ConversationViewModel? {
        if let cachedVM = viewModelCache[conversationID] {
            return cachedVM
        }

        guard let meta = metadata.first(where: { $0.id == conversationID }) else {
            return nil
        }

        let placeholder = Conversation(id: conversationID)

        let viewModel = ConversationViewModel(
            conversation: placeholder,
            metadata: meta,
            conversationAPI: conversationAPI,
            onTitleGenerated: { [weak self] title in
                Task { @MainActor in
                    self?.titleUpdated(for: conversationID, title: title)
                }
            },
            onDelete: { [weak self] in
                Task { @MainActor in
                    self?.deleteConversation(id: conversationID)
                }
            }
        )

        viewModelCache[conversationID] = viewModel

        Task {
            do {
                let conversation = try await conversationAPI.getConversation(id: conversationID)
                viewModel.conversation = conversation
            } catch ConversationAPIError.notFound {
                metadata.removeAll { $0.id == conversationID }
                viewModelCache.removeValue(forKey: conversationID)
                if activeConversationID == conversationID {
                    activeConversationID = metadata.first?.id
                }
                print("[ConversationManager] ⚠️ Conversation \(conversationID) absente (auto-cleanup)")
            } catch {
                print("[ConversationManager] ⚠️ Chargement messages \(conversationID) : \(error)")
            }
        }

        return viewModel
    }

    // MARK: - Callbacks internes

    /// Met à jour le cache local quand le ViewModel a généré un titre.
    /// La persistance backend est déjà faite par le ViewModel directement.
    private func titleUpdated(for id: UUID, title: String) {
        guard let idx = metadata.firstIndex(where: { $0.id == id }) else { return }
        metadata[idx].title = title
        metadata[idx].updatedAt = Date()
    }

    func cancelAllActiveGenerations() {
        viewModelCache.values.forEach { $0.cancelGenerationTaskIfNeeded() }
    }
}
