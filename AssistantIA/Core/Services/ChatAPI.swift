// ChatAPI.swift
// AssistantIA
//
// Client HTTP vers le backend Python FastAPI.
// Supporte génération complète (POST /chat) et streaming SSE (POST /chat/stream).
//
// Thread-safe par construction : état entièrement immuable (let).

import Foundation

final class ChatAPI {

    // MARK: - Config

    let baseURL: URL
    private let session: URLSession

    static let shared = ChatAPI()

    init(baseURL: URL = URL(string: "http://127.0.0.1:8000")!) {
        self.baseURL = baseURL

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest  = 60   // délai entre deux paquets
        config.timeoutIntervalForResource = 300  // durée totale max (génération longue)
        config.waitsForConnectivity = false      // échouer immédiatement si backend down
        self.session = URLSession(configuration: config)
    }

    // MARK: - Private: SSE Parser

    private func handleSSELine(
        _ line: String,
        continuation: AsyncThrowingStream<String, Error>.Continuation
    ) {
        // SSE : ignorer lignes vides et commentaires
        guard line.hasPrefix("data: ") else { return }
        let payload = String(line.dropFirst(6))

        // Fin de stream
        guard payload != "[DONE]" else {
            continuation.finish()
            return
        }

        // Décodage JSON {"text": "..."} ou {"error": "..."}
        guard let data = payload.data(using: .utf8) else {
            continuation.finish(throwing: ChatAPIError.decodingError("Ligne SSE non-UTF8"))
            return
        }

        do {
            let chunk = try JSONDecoder().decode(SSEChunk.self, from: data)
            if let errorMsg = chunk.error {
                continuation.finish(throwing: ChatAPIError.streamError(errorMsg))
            } else if let text = chunk.text {
                continuation.yield(text)
            }
        } catch {
            continuation.finish(throwing: ChatAPIError.decodingError("JSON invalide : \(payload)"))
        }
    }

    // MARK: - Private: HTTP Helpers

    private func makeRequest(path: String) -> URLRequest {
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        return request
    }

    /// Valide la réponse HTTP. `data` optionnel (nil en mode streaming avant lecture).
    private func validateHTTPResponse(_ response: URLResponse, data: Data?) throws {
        guard let http = response as? HTTPURLResponse else { return }

        switch http.statusCode {
        case 200...299:
            return
        case 503:
            let detail = data.flatMap { Self.extractDetail(from: $0) } ?? "Modèle non chargé"
            throw ChatAPIError.backendBusy(detail)
        default:
            let detail = data.flatMap { Self.extractDetail(from: $0) } ?? "Erreur inconnue"
            throw ChatAPIError.httpError(http.statusCode, detail)
        }
    }

    /// Extrait le champ "detail" d'une réponse d'erreur FastAPI.
    private static func extractDetail(from data: Data) -> String? {
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        return json?["detail"] as? String
    }

    /// Transforme une URLError en ChatAPIError typé.
    private func classify(_ error: Error) -> ChatAPIError {
        guard let urlError = error as? URLError else {
            return .httpError(0, error.localizedDescription)
        }
        switch urlError.code {
        case .timedOut:
            return .timeout
        case .cannotConnectToHost, .networkConnectionLost,
             .notConnectedToInternet:
            return .backendUnavailable
        default:
            return .httpError(urlError.errorCode, urlError.localizedDescription)
        }
    }

    /// Génération de titre : POST /agent/title → titre court ≤ 10 mots.
    /// Le frontend envoie uniquement le texte brut ; le prompt engineering est côté backend.
    func generateTitle(for message: String) async throws -> String {
        var request = makeRequest(path: "/agent/title")
        let body = TitleRequest(message: message)
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await session.data(for: request)
        try validateHTTPResponse(response, data: data)

        let decoded = try JSONDecoder().decode(TitleResponse.self, from: data)
        return decoded.title
    }
}

// MARK: - Request / Response Models

private struct ChatRequest: Encodable {
    let prompt: String
    let max_tokens: Int?
    let temperature: Float?
}

private struct ChatResponse: Decodable {
    let response: String
}

/// Aligné sur AgentChatRequest (Python) : champ `message`.
private struct AgentChatRequest: Encodable {
    let message: String
    let max_tokens: Int?
    let temperature: Float?
}

private struct AgentChatResponse: Decodable {
    let response: String
    let model: String
}

private struct TitleRequest: Encodable {
    let message: String
}

private struct TitleResponse: Decodable {
    let title: String
}

/// Chunk SSE : {"text": "..."} ou {"error": "..."}
private struct SSEChunk: Decodable {
    let text: String?
    let error: String?
}

// MARK: - Errors

enum ChatAPIError: LocalizedError {
    /// Backend non démarré ou connexion refusée.
    case backendUnavailable
    /// HTTP 503 — backend up mais modèle pas encore chargé.
    case backendBusy(String)
    /// Autre code HTTP d'erreur.
    case httpError(Int, String)
    /// Délai de 60s dépassé sans réponse complète.
    case timeout
    /// Backend a renvoyé `{"error": "..."}` en milieu de stream.
    case streamError(String)
    /// JSON ou SSE malformé (bug protocole).
    case decodingError(String)
    /// Annulation volontaire (user a interrompu la génération).
    case cancelled

    var errorDescription: String? {
        switch self {
        case .backendUnavailable:
            return "Backend non disponible. Relancez l'application."
        case .backendBusy(let detail):
            return "Backend occupé : \(detail)"
        case .httpError(let code, let detail):
            return "Erreur HTTP \(code) : \(detail)"
        case .timeout:
            return "Délai dépassé (60s). Réessayez avec un prompt plus court."
        case .streamError(let msg):
            return "Erreur génération : \(msg)"
        case .decodingError(let msg):
            return "Erreur protocole : \(msg)"
        case .cancelled:
            return "Génération annulée."
        }
    }

    /// Vrai si l'erreur est récupérable (retry possible).
    var isRetryable: Bool {
        switch self {
        case .timeout, .backendBusy: return true
        default: return false
        }
    }
}
