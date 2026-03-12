//
//  AssistantIAApp.swift
//  AssistantIA
//
//  Created by JB SENET on 31/01/2026.
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
                // Démarrage backend asynchrone : UI s'affiche immédiatement,
                // backend charge en arrière-plan (~20s pour le modèle).
                // .task est annulé automatiquement si la fenêtre se ferme.
                .task {
                    await startBackend()
                }
                // Backstop : NSApplication.willTerminate garantit l'arrêt
                // même si le .task n'est pas encore annulé au moment du quit.
                .onReceive(
                    NotificationCenter.default.publisher(
                        for: NSApplication.willTerminateNotification
                    )
                ) { _ in
                    backendManager.stop()
                }
        }
    }

    /// Lance le backend puis charge les conversations depuis l'API.
    /// Les erreurs sont absorbées — l'état est visible via backendManager.state
    /// et conversationManager.loadError.
    @MainActor
    private func startBackend() async {
        do {
            try await backendManager.start()
        } catch is CancellationError {
            // App quittée pendant le démarrage : arrêter le process si encore actif
            backendManager.stop()
            return
        } catch {
            // Échec startup : backendManager.state est déjà .error avec le message
            print("[App] ⚠️ Backend startup échoué : \(error.localizedDescription)")
            return
        }

        // Backend prêt → charger les conversations depuis l'API
        do {
            try await conversationManager.loadConversations()
        } catch {
            // Erreur exposée via conversationManager.loadError — pas de crash
            print("[App] ⚠️ Chargement conversations échoué : \(error.localizedDescription)")
        }
    }
}
