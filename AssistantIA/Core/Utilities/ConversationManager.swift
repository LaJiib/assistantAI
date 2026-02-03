//
//  ConversationManager.swift
//  AssistantIA
//
//  Created by JB SENET on 02/02/2026.
//
import Foundation

@Observable
@MainActor
class ConversationManager {
    private(set) var conversations: [Conversation] = []
    
    var activeConversationID: UUID?
    
    private let mlxService: MLXService
    
    private var viewModelCache: [UUID: ConversationViewModel] = [:]
    
    var activeConversation: Conversation? {
        guard let id = activeConversationID else {
            return nil
        }
        return conversations.first { $0.id == id }
    }
    
    var activeViewModel: ConversationViewModel? {
        guard let id = activeConversationID else {
            return nil
        }
        return getViewModel(for: id)
    }
    
    init(mlxService: MLXService) {
        self.mlxService = mlxService
        loadConversations()
        
        if conversations.isEmpty {
            createConversation()
        }
    }
    
    @discardableResult
    func createConversation(systemPrompt: String = Constants.defaultSystemPrompt,
                            title: String = "Nouvelle Conversation")-> Conversation {
        let conversation = Conversation(systemPrompt: systemPrompt, title: title)
        conversations.append(conversation)
        
        activeConversationID = conversation.id
        
        saveConversations()
        return conversation
    }
    
    func deleteConversation(id: UUID) {
        conversations.removeAll { $0.id == id }
        
        viewModelCache.removeValue(forKey: id)
        
        
        if activeConversationID == id {
            activeConversationID = conversations.first?.id
        }
         saveConversations()
    }
    
    func setActiveConversation(id: UUID) {
        guard conversations.contains(where: { $0.id == id }) else {
            return // Conversation inexistante
        }
        
        activeConversationID = id
    }
    
    func getViewModel(for conversationID: UUID) -> ConversationViewModel? {
        // Vérifier le cache d'abord
        if let cachedVM = viewModelCache[conversationID] {
            return cachedVM
        }
        
        // Trouver la conversation
        guard let conversation = conversations.first(where: { $0.id == conversationID }) else {
            return nil
        }
        
        // Créer un nouveau ViewModel
        let viewModel = ConversationViewModel(
            conversation: conversation,
            mlxService: mlxService
        )
        
        // Mettre en cache
        viewModelCache[conversationID] = viewModel
        
        return viewModel
    }
    
    func saveConversations() {
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted // Pour debug
            
            let data = try encoder.encode(conversations)
            
            let url = getConversationsFileURL()
            try data.write(to: url)
            
            print("✅ Conversations sauvegardées")
        } catch {
            print("❌ Erreur sauvegarde: \(error)")
        }
    }
    
    func loadConversations() {
        do {
            let url = getConversationsFileURL()
            
            // Vérifier que le fichier existe
            guard FileManager.default.fileExists(atPath: url.path) else {
                print("ℹ️ Pas de fichier de conversations")
                return
            }
            
            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            
            conversations = try decoder.decode([Conversation].self, from: data)
            
            print("✅ \(conversations.count) conversations chargées")
            
            // Activer la première si aucune n'est active
            if activeConversationID == nil {
                activeConversationID = conversations.first?.id
            }
        } catch {
            print("❌ Erreur chargement: \(error)")
        }
    }
    private func getConversationsFileURL() -> URL {
        let documentsDirectory = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first!
        
        return documentsDirectory.appendingPathComponent("conversations.json")
    }
    
    func cancelAllActiveGenerations() {
        viewModelCache.values.forEach { $0.cancelGenerationTaskIfNeeded() }
    }
}
