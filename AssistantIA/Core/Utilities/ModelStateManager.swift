//
//  ModelStateManager.swift
//  AssistantIA
//
//  Created by JB SENET on 02/02/2026.
//

import Foundation

enum ModelStateError: LocalizedError {
    case notConfigured
    case alreadyLoading
    
    var errorDescription: String? {
        switch self {
        case .notConfigured:
            return "Model folder not configured"
        case .alreadyLoading:
            return "Model is already loading"
        }
    }
}

enum ModelLoadState: Equatable {
    case notConfigured
    case configured(URL)
    case loading
    case loaded(URL)
    case error(String)
}


@Observable
@MainActor
class ModelStateManager {
    private(set) var modelState: ModelLoadState = .notConfigured
    private let mlxService: MLXService
    private let conversationManager: ConversationManager
    private let bookmarkManager = BookmarkManager()
    
    var isLoaded: Bool {
        if case .loaded = modelState {
            return true
        }
        return false
    }
    
    var currentModelURL: URL? {
        switch modelState {
        case .configured(let url), .loaded(let url):
            return url
        default:
            return nil
        }
    }
    
    var stateDescription: String {
        switch modelState {
        case .notConfigured:
            return "Not configured"
        case .configured:
            return "Ready to load"
        case .loading:
            return "Loading..."
        case .loaded:
            return "Loaded"
        case .error(let message):
            return "Error: \(message)"
        }
    }
    
    init(mlxService: MLXService, conversationManager: ConversationManager) {
        self.mlxService = mlxService
        self.conversationManager = conversationManager
        if let url = try? bookmarkManager.loadBookmark() {
            modelState = .configured(url)
            conversationManager.setDataFolderURL(url)
        }
    }
    
    @MainActor
    func requestModelFolder() -> URL? {
        guard let url = bookmarkManager.requestDataFolder() else {
            return nil
        }
        try? bookmarkManager.saveBookmark(for: url)
        modelState = .configured(url)
        conversationManager.setDataFolderURL(url)
        return url
    }
    
    func loadModel() async throws {
        guard case .configured(let url) = modelState else {
            throw ModelStateError.notConfigured
        }
        modelState = .loading
        do {
            let modelPath = url.appendingPathComponent("models/Ministral-3-14B-Instruct-2512-8bit")
            try await mlxService.loadLocalModel(at: modelPath.path)
            modelState = .loaded(url)
        } catch {
            // Invalider le bookmark qui a conduit à cette erreur
            UserDefaults.standard.removeObject(forKey: "modelFolderBookmark")
            modelState = .error(error.localizedDescription)
            throw error
        }
    }
    
    func unloadModel() {
        conversationManager.cancelAllActiveGenerations()   // 1. couper les tâches
        mlxService.unloadModel()                           // 2. décharger le modèle
        if case .loaded(let url) = modelState {            // 3. mettre à jour l'état
            modelState = .configured(url)
        }
    }
}
