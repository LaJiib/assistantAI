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

    /// PID du process Python actif, nil si arrêté. Pour debug / Activity Monitor.
    var pid: Int? {
        guard let p = process, p.isRunning else { return nil }
        return Int(p.processIdentifier)
    }

    var baseURL: URL {
        URL(string: "http://\(host):\(port)")!
    }

    // MARK: - Config

    private let backendPath: String
    private let pythonPath: String
    private let host = "127.0.0.1"
    private let port = 8000
    private let startupTimeout: TimeInterval = 30
    @ObservationIgnored private let adminAPI: BackendAdminAPI

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
        adminAPI = BackendAdminAPI(baseURL: URL(string: "http://127.0.0.1:8000")!)
        // Résoudre le double symlink python → python3.14 → /opt/homebrew/...
        // Process.run() échoue si l'exécutable est un symlink vers un chemin hors sandbox.
        let symlinkPath = "\(repoRoot)/backend/venv/bin/python"
        pythonPath = URL(fileURLWithPath: symlinkPath)
            .resolvingSymlinksInPath().path
    }

    /// Remonte l'arborescence pour trouver la racine du repo (contient backend/main.py).
    ///
    /// NSHomeDirectory() retourne le répertoire sandbox dans une app macOS sandboxée
    /// (/Users/xxx/Library/Containers/…). On utilise NSUserName() pour construire
    /// le vrai répertoire home de l'utilisateur (/Users/xxx).
    private static func findRepoRoot() -> String {
        let realHome = "/Users/\(NSUserName())"
        let candidates = [
            "\(realHome)/AssistantIA/AssistantIA",
            "\(realHome)/Developer/AssistantIA",
        ]
        let fm = FileManager.default
        for candidate in candidates {
            if fm.fileExists(atPath: "\(candidate)/backend/main.py") {
                return candidate
            }
        }
        // Fallback : chemin connu du projet
        return "\(realHome)/AssistantIA/AssistantIA"
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

    /// Health check ponctuel : GET /health → retourne l'état backend typé.
    func checkHealth() async throws -> BackendHealth {
        do {
            return try await adminAPI.health()
        } catch {
            throw BackendError.unhealthy
        }
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

        // Construire l'environnement venv manuellement.
        //
        // Problème : pythonPath est résolu via resolvingSymlinksInPath() pour que
        // Process.run() trouve le binaire Homebrew. Mais Python résolu depuis
        // /opt/homebrew/... ne détecte pas le pyvenv.cfg → venv non activé → ImportError.
        //
        // Solution : recréer ce que `source venv/bin/activate` fait, en posant
        // VIRTUAL_ENV, PYTHONPATH et PATH directement dans l'environnement du process.
        let venvPath = "\(backendPath)/venv"
        var env = ProcessInfo.processInfo.environment

        // VIRTUAL_ENV : signale à Python qu'il tourne dans ce venv
        env["VIRTUAL_ENV"] = venvPath

        // PYTHONPATH : chemin vers les packages installés dans le venv
        // Recherche dynamique du dossier python3.x pour ne pas hardcoder la version
        let libPath = "\(venvPath)/lib"
        if let dirs = try? FileManager.default.contentsOfDirectory(atPath: libPath),
           let pyDir = dirs.first(where: { $0.hasPrefix("python") }) {
            env["PYTHONPATH"] = "\(libPath)/\(pyDir)/site-packages"
        }

        // PATH : ajouter venv/bin en tête pour que les scripts du venv (uvicorn, etc.)
        // soient trouvés en priorité par les sous-processus Python
        let currentPath = env["PATH"] ?? "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        env["PATH"] = "\(venvPath)/bin:\(currentPath)"

        // Paramètres backend
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
        let pollInterval: UInt64 = 2_000_000_000  // 2s — réduit le bruit de logs réseau

        while Date() < deadline {
            // Arrêt explicite demandé (stop() appelé pendant le démarrage)
            if case .stopped = state { return }
            // Process crashé entre deux polls
            if case .error = state {
                throw BackendError.startupFailed(stderrBuffer)
            }
            // Process mort pendant startup: ne pas valider un autre backend déjà sur le port.
            guard let currentProcess = process, currentProcess.isRunning else {
                let errMsg = "Backend arrêté pendant le démarrage.\n\(stderrBuffer.suffix(500))"
                state = .error(errMsg)
                throw BackendError.startupFailed(errMsg)
            }

            do {
                let health = try await checkHealth()
                let modelLoaded = health.modelLoaded
                let healthPID = health.pid
                let expectedPID = Int(currentProcess.processIdentifier)

                // Si un autre process répond sur 8000, on attend notre propre instance.
                guard healthPID == expectedPID else {
                    print("[BackendManager] ⏳ Port \(port) occupé par un autre backend (pid \(healthPID)), attente de l'instance pid \(expectedPID)...")
                    try await Task.sleep(nanoseconds: pollInterval)
                    continue
                }

                if modelLoaded {
                    restartAttempts = 0
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
        if case .stopped = state { return }
        if case .starting = state {
            state = .error("Backend arrêté pendant le démarrage (code: \(code)).")
            print("[BackendManager] ❌ Backend arrêté pendant startup (code: \(code))")
            return
        }
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
