//
//  ConversationManager.swift
//  AssistantIA
//
//  Created by JB SENET on 02/02/2026.
//
import Foundation

enum ConversationStorageError: LocalizedError {
    case notFound(UUID)
    case saveFailed(String)

    var errorDescription: String? {
        switch self {
        case .notFound(let id):
            return "Conversation \(id) introuvable sur le disque"
        case .saveFailed(let reason):
            return "Erreur sauvegarde: \(reason)"
        }
    }
}

@Observable
@MainActor
class ConversationManager {
    private(set) var metadata: [ConversationMetadata] = []
    
    var activeConversationID: UUID?
    
    private let mlxService: MLXService
    
    private var viewModelCache: [UUID: ConversationViewModel] = [:]
    
    private var dataFolderURL: URL?
    
    var activeViewModel: ConversationViewModel? {
        guard let id = activeConversationID else {
            return nil
        }
        return getViewModel(for: id)
    }
    
    init(mlxService: MLXService) {
        self.mlxService = mlxService
        loadIndex()
        if metadata.isEmpty {
            createConversation()
        }
    }

    func setDataFolderURL(_ url: URL) {
        dataFolderURL = url
        loadIndex()
    }
    
    func loadMessages(for id: UUID) throws -> Conversation {
        guard metadata.contains(where: { $0.id == id }) else {
            throw ConversationStorageError.notFound(id)
        }
        guard let url = conversationFileURL(for: id) else {
            throw ConversationStorageError.notFound(id)
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Conversation.self, from: data)
    }

    private func saveMessages(_ conversation: Conversation) throws {
        guard let url = conversationFileURL(for: conversation.id) else {
            throw ConversationStorageError.saveFailed("Pas de dossier de données configuré")
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(conversation)
        try data.write(to: url)
    }
    
    @discardableResult
    func createConversation(systemPrompt: String = Constants.defaultSystemPrompt,
                            title: String = "Nouvelle Conversation") -> ConversationMetadata {
        let id = UUID()
        let now = Date()

        // 1. Créer les métadonnées dans l'index
        let meta = ConversationMetadata(
            id: id,
            title: title,
            systemPrompt: systemPrompt,
            createdAt: now,
            updatedAt: now,
            messageCount: 1          // le system prompt compte comme un message
        )
        metadata.append(meta)
        activeConversationID = id

        // 2. Créer le fichier de messages sur le disque
        let conversation = Conversation(id: id, systemPrompt: systemPrompt)
        try? saveMessages(conversation)

        // 3. Sauvegarder l'index
        saveIndex()

        return meta
    }
    
    func deleteConversation(id: UUID) {
        metadata.removeAll { $0.id == id }
        viewModelCache.removeValue(forKey: id)

        // Supprimer le fichier de messages du disque
        if let url = conversationFileURL(for: id) {
            try? FileManager.default.removeItem(at: url)
        }

        if activeConversationID == id {
            activeConversationID = metadata.first?.id
        }
        saveIndex()
    }
    
    func renameConversation(id: UUID, newTitle: String) {
        guard let index = metadata.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }   // on refuse un titre vide
        metadata[index].title = trimmed
        metadata[index].updatedAt = Date()
        saveIndex()
    }
    
    func setActiveConversation(id: UUID) {
        guard metadata.contains(where: { $0.id == id }) else { return }
        activeConversationID = id
    }
    
    func getViewModel(for conversationID: UUID) -> ConversationViewModel? {
        if let cachedVM = viewModelCache[conversationID] {
            return cachedVM
        }

        // Charger les messages depuis le fichier individuel
        guard let conversation = try? loadMessages(for: conversationID) else {
            return nil
        }

        let viewModel = ConversationViewModel(
            conversation: conversation,
            mlxService: mlxService,
            onConversationUpdated: { [weak self] updatedConversation, newTitle in
                Task { @MainActor in
                    self?.conversationDidUpdate(updatedConversation, newTitle: newTitle)
                }
            },
            onDelete: { [weak self] in
                Task { @MainActor in
                    self?.deleteConversation(id: conversationID)   // l'ID est capturé ici
                }
            }
        )

        viewModelCache[conversationID] = viewModel
        return viewModel
    }
    
    private func saveIndex() {
        guard let url = indexFileURL() else {
            print("ℹ️ Pas de dossier de données configuré, pas de sauvegarde")
            return
        }
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            let data = try encoder.encode(metadata)
            try data.write(to: url)
            print("✅ Index sauvegardé")
        } catch {
            print("❌ Erreur sauvegarde index: \(error)")
        }
    }
    
    private func loadIndex() {
        guard let url = indexFileURL() else {
            print("ℹ️ Pas de dossier de données configuré")
            return
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            print("ℹ️ Pas d'index existant")
            return
        }
        do {
            let data = try Data(contentsOf: url)
            metadata = try JSONDecoder().decode([ConversationMetadata].self, from: data)
            print("✅ \(metadata.count) conversations dans l'index")
            if activeConversationID == nil {
                activeConversationID = metadata.first?.id
            }
        } catch {
            print("❌ Erreur chargement index: \(error)")
        }
    }
    
    private func indexFileURL() -> URL? {
        guard let dataFolder = dataFolderURL else { return nil }
        let conversationsFolder = dataFolder.appendingPathComponent("conversations")
        try? FileManager.default.createDirectory(at: conversationsFolder, withIntermediateDirectories: true)
        return conversationsFolder.appendingPathComponent("conversations.json")
    }
    
    private func conversationFileURL(for id: UUID) -> URL? {
        guard let dataFolder = dataFolderURL else { return nil }
        let conversationsFolder = dataFolder.appendingPathComponent("conversations")
        try? FileManager.default.createDirectory(at: conversationsFolder, withIntermediateDirectories: true)
        return conversationsFolder.appendingPathComponent("\(id).json")
    }
    
    func cancelAllActiveGenerations() {
        viewModelCache.values.forEach { $0.cancelGenerationTaskIfNeeded() }
    }
    
    func conversationDidUpdate(_ updated: Conversation, newTitle: String?) {
        // 1. Sauvegarder les messages sur le disque
        try? saveMessages(updated)

        // 2. Mettre à jour le metadata dans l'index
        if let index = metadata.firstIndex(where: { $0.id == updated.id }) {
            if let title = newTitle {
                metadata[index].title = title
            }
            metadata[index].updatedAt = Date()
            metadata[index].messageCount = updated.messages.count
        }

        // 3. Sauvegarder l'index
        saveIndex()
    }
}
