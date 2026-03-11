// BackendManager.swift
// AssistantIA
//
// Gère le cycle de vie du backend Python FastAPI :
// démarrage subprocess, health check polling, capture logs, arrêt propre.

import Foundation
import Observation

@Observable
@MainActor
final class BackendManager {

    // MARK: - State Machine

    enum State: Equatable {
        case stopped
        case starting
        case running(port: Int)
        case error(String)

        var isRunning: Bool {
            if case .running = self { return true }
            return false
        }

        var displayText: String {
            switch self {
            case .stopped:           return "Arrêté"
            case .starting:          return "Démarrage..."
            case .running(let port): return "Actif (port \(port))"
            case .error(let msg):    return "Erreur : \(msg)"
            }
        }
    }

    // MARK: - Public State

    private(set) var state: State = .stopped

    // MARK: - Config

    private let backendPath: String
    private let pythonPath: String
    private let host = "127.0.0.1"
    private let port = 8000
    private let startupTimeout: TimeInterval = 30

    /// Nombre de redémarrages automatiques tentés après crash.
    private var restartAttempts = 0
    private let maxRestartAttempts = 1

    // MARK: - Process

    private var process: Process?
    /// Tampon stderr pour diagnostiquer les échecs au démarrage.
    private var stderrBuffer = ""

    // MARK: - Init

    init() {
        let repoRoot = Self.findRepoRoot()
        backendPath = "\(repoRoot)/backend"
        pythonPath  = "\(repoRoot)/backend/venv/bin/python"
    }

    /// Remonte l'arborescence pour trouver la racine du repo (contient backend/main.py).
    private static func findRepoRoot() -> String {
        let candidates = [
            "\(NSHomeDirectory())/AssistantIA/AssistantIA",
            "\(NSHomeDirectory())/Developer/AssistantIA",
        ]
        let fm = FileManager.default
        for candidate in candidates {
            if fm.fileExists(atPath: "\(candidate)/backend/main.py") {
                return candidate
            }
        }
        // Fallback : chemin connu du projet
        return "\(NSHomeDirectory())/AssistantIA/AssistantIA"
    }

    // MARK: - Public API

    /// Démarre le backend Python et attend qu'il soit prêt (health check polling).
    /// Peut être relancé depuis l'état .error.
    func start() async throws {
        switch state {
        case .starting, .running:
            return  // Déjà actif ou en cours de démarrage
        case .stopped, .error:
            break   // Procéder au démarrage
        }

        state = .starting
        stderrBuffer = ""
        restartAttempts = 0
        print("[BackendManager] 🚀 Démarrage backend...")

        try await launchProcess()
        try await waitForBackendReady(timeout: startupTimeout)
    }

    /// Arrête le backend proprement (SIGTERM + attente terminaison).
    func stop() {
        // Signaler arrêt immédiatement pour bloquer terminationHandler
        state = .stopped

        guard let process = process, process.isRunning else {
            self.process = nil
            return
        }
        print("[BackendManager] 🛑 Arrêt backend (PID: \(process.processIdentifier))...")
        process.terminate()

        // waitUntilExit bloque le thread — exécuté hors MainActor
        let capturedProcess = process
        self.process = nil
        Task.detached(priority: .utility) {
            capturedProcess.waitUntilExit()
            print("[BackendManager] ✅ Backend arrêté (code: \(capturedProcess.terminationStatus))")
        }
    }

    /// Health check ponctuel : GET /health → retourne le JSON ou lance une erreur.
    func checkHealth() async throws -> [String: Any] {
        let url = URL(string: "http://\(host):\(port)/health")!
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw BackendError.unhealthy
        }
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        return json
    }

    // MARK: - Private: Process Launch

    private func launchProcess() async throws {
        let fm = FileManager.default

        guard fm.fileExists(atPath: "\(backendPath)/main.py") else {
            let msg = "main.py introuvable dans \(backendPath)"
            state = .error(msg)
            throw BackendError.missingFiles(msg)
        }
        guard fm.fileExists(atPath: pythonPath) else {
            let msg = "Python venv introuvable : \(pythonPath)"
            state = .error(msg)
            throw BackendError.missingFiles(msg)
        }

        let p = Process()
        p.executableURL = URL(fileURLWithPath: pythonPath)
        p.arguments = ["main.py"]
        p.currentDirectoryURL = URL(fileURLWithPath: backendPath)

        // Hériter de l'environnement système + forcer HOST/PORT localhost
        var env = ProcessInfo.processInfo.environment
        env["HOST"] = host
        env["PORT"] = String(port)
        p.environment = env

        // Stdout → console Xcode (non-bloquant via readabilityHandler)
        let stdoutPipe = Pipe()
        p.standardOutput = stdoutPipe
        stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            text.components(separatedBy: "\n")
                .filter { !$0.isEmpty }
                .forEach { print("[Backend] \($0)") }
        }

        // Stderr → console Xcode + buffer pour diagnostic erreurs startup
        let stderrPipe = Pipe()
        p.standardError = stderrPipe
        stderrPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            text.components(separatedBy: "\n")
                .filter { !$0.isEmpty }
                .forEach { print("[Backend stderr] \($0)") }
            // Accumuler dans le buffer sur MainActor
            Task { @MainActor [weak self] in
                self?.stderrBuffer += text
            }
        }

        // Terminaison inattendue → crash handler
        p.terminationHandler = { [weak self] proc in
            Task { @MainActor [weak self] in
                self?.handleProcessTermination(proc)
            }
        }

        do {
            try p.run()
        } catch {
            let msg = "Impossible de lancer Python : \(error.localizedDescription)"
            state = .error(msg)
            throw BackendError.launchFailed(error)
        }

        process = p
        print("[BackendManager] Process lancé (PID: \(p.processIdentifier))")
    }

    // MARK: - Private: Health Check Polling

    private func waitForBackendReady(timeout: TimeInterval) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        let pollInterval: UInt64 = 500_000_000  // 500ms en nanosecondes

        while Date() < deadline {
            // Vérifier si le process a crashé entre deux polls
            if case .error = state {
                throw BackendError.startupFailed(stderrBuffer)
            }

            do {
                let health = try await checkHealth()
                let modelLoaded = health["model_loaded"] as? Bool ?? false

                if modelLoaded {
                    state = .running(port: port)
                    print("[BackendManager] ✅ Backend prêt — model_loaded=true, port:\(port)")
                    return
                } else {
                    print("[BackendManager] ⏳ Backend répond, chargement modèle en cours...")
                }
            } catch {
                // Backend pas encore démarré, on continue à poller
            }

            try await Task.sleep(nanoseconds: pollInterval)
        }

        // Timeout dépassé sans succès
        stop()
        let errMsg = "Timeout \(Int(timeout))s — backend non disponible.\n\(stderrBuffer.suffix(500))"
        state = .error(errMsg)
        throw BackendError.timeout
    }

    // MARK: - Private: Crash Handler

    private func handleProcessTermination(_ proc: Process) {
        let code = proc.terminationStatus

        // Si on a nous-mêmes arrêté le process (stop() set state = .stopped avant), ignorer
        guard case .running = state else { return }

        print("[BackendManager] ⚠️ Backend terminé inopinément (code: \(code))")

        if restartAttempts < maxRestartAttempts {
            restartAttempts += 1
            print("[BackendManager] 🔄 Auto-restart (\(restartAttempts)/\(maxRestartAttempts))...")
            state = .stopped
            stderrBuffer = ""
            Task {
                do {
                    try await self.start()
                } catch {
                    self.state = .error("Auto-restart échoué : \(error.localizedDescription)")
                }
            }
        } else {
            state = .error("Backend crashé (code: \(code)). Relancez l'application.")
            print("[BackendManager] ❌ Crashes répétés, abandon auto-restart.")
        }
    }
}

// MARK: - Errors

enum BackendError: LocalizedError {
    case missingFiles(String)
    case launchFailed(Error)
    case unhealthy
    case startupFailed(String)
    case timeout

    var errorDescription: String? {
        switch self {
        case .missingFiles(let msg):    return msg
        case .launchFailed(let e):      return "Échec lancement : \(e.localizedDescription)"
        case .unhealthy:                return "Backend ne répond pas correctement."
        case .startupFailed(let log):   return "Échec démarrage backend.\n\(log)"
        case .timeout:                  return "Timeout démarrage backend (30s)."
        }
    }
}
