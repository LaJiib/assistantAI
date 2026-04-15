//
//  AssistantIAApp.swift
//  AssistantIA
//

import SwiftUI

@main
struct AssistantIAApp: App {
    @State private var conversationManager = ConversationManager()
    @State private var backendManager = BackendManager()

    var body: some Scene {
        WindowGroup {
            RootView(conversationManager: conversationManager)
                // Rend backendManager accessible à toutes les vues enfants
                .environment(backendManager)
                // Connexion au backend asynchrone : UI s'affiche immédiatement,
                // le manager poll /health jusqu'à ce que le backend soit prêt.
                // .task est annulé automatiquement si la fenêtre se ferme.
                .task {
                    await connectToBackend()
                }
        }
    }

    /// Se connecte au backend puis charge les conversations depuis l'API.
    /// Les erreurs sont absorbées — l'état est visible via backendManager.state
    /// et conversationManager.loadError.
    @MainActor
    private func connectToBackend() async {
        await backendManager.connect()

        guard backendManager.state.isRunning else { return }

        do {
            try await conversationManager.loadConversations()
        } catch {
            print("[App] ⚠️ Chargement conversations échoué : \(error.localizedDescription)")
        }
    }
}
