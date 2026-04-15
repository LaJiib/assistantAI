// BackendManager.swift
// AssistantIA
//
// Moniteur de connectivité vers le backend Python FastAPI.
// Le backend est lancé indépendamment (VSCode / terminal) ; ce manager
// se contente de poller /health et d'exposer l'état à l'UI.

import Foundation
import Observation

@Observable
@MainActor
final class BackendManager {

    // MARK: - State Machine

    enum State: Equatable {
        case stopped                 // Pas encore tenté de connexion
        case connecting              // Polling /health en cours
        case running(port: Int)      // Backend prêt (model_loaded=true)
        case error(String)           // Timeout ou backend introuvable

        var isRunning: Bool {
            if case .running = self { return true }
            return false
        }

        var displayText: String {
            switch self {
            case .stopped:           return "Non connecté"
            case .connecting:        return "Connexion..."
            case .running(let port): return "Actif (port \(port))"
            case .error(let msg):    return "Erreur : \(msg)"
            }
        }
    }

    // MARK: - Public State

    private(set) var state: State = .stopped

    /// PID rapporté par le backend dans /health. Nul si non connecté.
    private(set) var pid: Int? = nil

    let baseURL: URL = URL(string: "http://127.0.0.1:8000")!

    // MARK: - Config

    private let port = 8000
    private let connectTimeout: TimeInterval = 180   // Le modèle met ~90s à charger
    @ObservationIgnored private let adminAPI: BackendAdminAPI

    // MARK: - Init

    init() {
        adminAPI = BackendAdminAPI(baseURL: URL(string: "http://127.0.0.1:8000")!)
    }

    // MARK: - Public API

    /// Tente de joindre le backend déjà en cours d'exécution.
    /// Poll /health jusqu'à model_loaded=true ou timeout.
    func connect() async {
        switch state {
        case .connecting, .running:
            return  // Déjà actif ou en cours
        case .stopped, .error:
            break
        }
        state = .connecting
        pid = nil
        print("[BackendManager] 🔌 Connexion au backend sur \(baseURL.absoluteString)...")
        await pollUntilReady(timeout: connectTimeout)
    }

    /// Réinitialise l'état et retente la connexion.
    func reconnect() async {
        state = .stopped
        pid = nil
        await connect()
    }

    /// Health check ponctuel : GET /health → retourne l'état backend typé.
    func checkHealth() async throws -> BackendHealth {
        do {
            return try await adminAPI.health()
        } catch {
            throw BackendError.unhealthy
        }
    }

    // MARK: - Private: Health Polling

    private func pollUntilReady(timeout: TimeInterval) async {
        let deadline = Date().addingTimeInterval(timeout)
        let pollInterval: UInt64 = 2_000_000_000  // 2s

        while Date() < deadline {
            // Annulation depuis l'extérieur (reconnect() ou Task.cancel)
            if case .stopped = state { return }

            do {
                let health = try await checkHealth()
                if health.modelLoaded {
                    pid = health.pid
                    state = .running(port: port)
                    print("[BackendManager] ✅ Connecté — model_loaded=true, pid:\(health.pid)")
                    return
                } else {
                    print("[BackendManager] ⏳ Backend répond, modèle en chargement...")
                }
            } catch {
                // Backend pas encore disponible — on continue
            }

            do {
                try await Task.sleep(nanoseconds: pollInterval)
            } catch {
                return  // Tâche annulée
            }
        }

        let errMsg = "Backend introuvable sur \(baseURL.absoluteString) après \(Int(timeout))s.\nLancez le backend : cd backend && python main.py"
        state = .error(errMsg)
        print("[BackendManager] ❌ Timeout connexion backend.")
    }
}

// MARK: - Errors

enum BackendError: LocalizedError {
    case unhealthy
    case connectionFailed(String)

    var errorDescription: String? {
        switch self {
        case .unhealthy:                   return "Backend ne répond pas."
        case .connectionFailed(let msg):   return msg
        }
    }
}
