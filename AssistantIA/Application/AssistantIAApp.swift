//
//  AssistantIAApp.swift
//  AssistantIA
//
//  Created by JB SENET on 31/01/2026.
//

import SwiftUI

@main
struct AssistantIAApp: App {
    @State private var mlxService: MLXService
    @State private var conversationManager: ConversationManager
    @State private var modelStateManager: ModelStateManager

    // Phase 2 : BackendManager géré ici — lifecycle lié à l'app entière.
    // @State garantit une instance unique initialisée une seule fois.
    @State private var backendManager = BackendManager()

    init() {
        let service = MLXService()
        let manager = ConversationManager(mlxService: service)

        _mlxService = State(initialValue: service)
        _conversationManager = State(initialValue: manager)
        _modelStateManager = State(initialValue: ModelStateManager(mlxService: service, conversationManager: manager))
    }

    var body: some Scene {
        WindowGroup {
            RootView(modelStateManager: modelStateManager,
                     conversationManager: conversationManager)
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

    /// Lance le backend et absorbe les erreurs — l'état est visible via backendManager.state.
    @MainActor
    private func startBackend() async {
        do {
            try await backendManager.start()
        } catch is CancellationError {
            // App quittée pendant le démarrage : arrêter le process si encore actif
            backendManager.stop()
        } catch {
            // Échec startup : backendManager.state est déjà .error avec le message
            print("[App] ⚠️ Backend startup échoué : \(error.localizedDescription)")
        }
    }
}
