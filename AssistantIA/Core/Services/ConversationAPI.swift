// ConversationAPI.swift
// AssistantIA
//
// Client REST vers le backend Python FastAPI.
// Gère toutes les opérations conversations et messages.
//
// Thread-safe par isolation actor.
// Streaming SSE via AsyncThrowingStream avec propagation de cancellation.
//
// Les modèles Swift existants (ConversationMetadata, Conversation, Message)
// sont décodés directement depuis les réponses JSON — leurs champs correspondent
// exactement à ce que le backend retourne.

import Foundation

// MARK: - Errors

enum ConversationAPIError: LocalizedError {
    /// Connexion refusée — backend non démarré ou crashé.
    case networkError(URLError)
    /// HTTP 404 — conversation supprimée (auto-cleanup) ou ID invalide.
    case notFound
    /// HTTP 503 — backend up mais modèle en cours de chargement.
    case modelLoading(String)
    /// Autre code HTTP d'erreur (4xx, 5xx).
    case serverError(Int, String)
    /// JSON malformé ou types inattendus.
    case decodingError(String)
    /// Erreur émise par le backend en milieu de stream SSE.
    case streamError(String)
    /// Annulation volontaire.
    case cancelled

    var errorDescription: String? {
        switch self {
        case .networkError:
            return "Backend non disponible. Relancez l'application."
        case .notFound:
            return "Conversation introuvable."
        case .modelLoading(let detail):
            return "Backend démarre : \(detail). Patientez…"
        case .serverError(let code, let detail):
            return "Erreur serveur \(code) : \(detail)"
        case .decodingError(let msg):
            return "Erreur protocole : \(msg)"
        case .streamError(let msg):
            return "Erreur génération : \(msg)"
        case .cancelled:
            return "Génération annulée."
        }
    }

    /// Vrai si l'erreur est temporaire et qu'un retry a du sens.
    var isTransient: Bool {
        switch self {
        case .modelLoading, .networkError: return true
        default: return false
        }
    }
}

// MARK: - Private API response types
// DTO réseau (contrat backend) séparés des modèles métier Swift.

private enum APIRole: String, Decodable {
    case user
    case assistant
    case system
}

private struct APIMessage: Decodable {
    let id: UUID
    let role: APIRole
    let content: String
    let timestamp: Date
}

private struct APIConversation: Decodable {
    let id: UUID
    let systemPrompt: String
    let messages: [APIMessage]
}

private struct APIConversationMetadata: Decodable {
    let id: UUID
    let title: String
    let createdAt: Date
    let updatedAt: Date
    let messageCount: Int
}

private struct APISendMessageResponse: Decodable {
    let userMessage: APIMessage
    let assistantMessage: APIMessage
}

/// Chunk SSE — {"text":"..."}, {"error":"..."} ou {"userMessage":{...}}
private struct SSEChunk: Decodable {
    let text:        String?
    let error:       String?
    // Premier event du stream : userMessage inclus (ignoré dans le stream texte)
}

// MARK: - Request bodies

private struct CreateConversationBody: Encodable {
    let title:        String
}

private struct UpdateConversationBody: Encodable {
    let title: String
}

private struct SendMessageBody: Encodable {
    let content:     String
    let max_tokens:  Int?
    let temperature: Float?
}

// MARK: - ConversationAPI

actor ConversationAPI {

    static let shared = ConversationAPI()

    private let baseURL: URL
    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder

    init(baseURL: URL = URL(string: "http://127.0.0.1:8000")!) {
        self.baseURL = baseURL

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest  = 60    // délai entre paquets (génération longue)
        config.timeoutIntervalForResource = 300   // durée totale max
        config.waitsForConnectivity = false       // échouer immédiatement si backend down
        self.session = URLSession(configuration: config)

        self.decoder = Self.makeDecoder()
        self.encoder = JSONEncoder()
    }

    // MARK: - Conversations CRUD

    /// GET /api/conversations/ → liste triée par updatedAt desc.
    func listConversations() async throws -> [ConversationMetadata] {
        let request = makeRequest(method: "GET", path: "/api/conversations/")
        let data = try await perform(request)
        let items = try decode([APIConversationMetadata].self, from: data)
        return items.map(toDomain)
    }

    /// GET /api/conversations/{id} → Conversation complète (messages inclus).
    /// Lève `.notFound` si la conversation a été nettoyée par le backend.
    func getConversation(id: UUID) async throws -> Conversation {
        let request = makeRequest(method: "GET", path: "/api/conversations/\(id)")
        let data = try await perform(request)
        let conv = try decode(APIConversation.self, from: data)
        return toDomain(conv)
    }

    /// POST /api/conversations/ → crée et retourne les métadonnées.
    @discardableResult
    func createConversation(title: String) async throws -> ConversationMetadata {
        var request = makeRequest(method: "POST", path: "/api/conversations/")
        request.httpBody = try encoder.encode(
            CreateConversationBody(title: title)
        )
        let data = try await perform(request, expectedStatuses: [201])
        let meta = try decode(APIConversationMetadata.self, from: data)
        return toDomain(meta)
    }

    /// PUT /api/conversations/{id} → modifie le titre, retourne les métadonnées à jour.
    @discardableResult
    func updateConversation(id: UUID, title: String) async throws -> ConversationMetadata {
        var request = makeRequest(method: "PUT", path: "/api/conversations/\(id)")
        request.httpBody = try encoder.encode(UpdateConversationBody(title: title))
        let data = try await perform(request)
        let meta = try decode(APIConversationMetadata.self, from: data)
        return toDomain(meta)
    }

    /// DELETE /api/conversations/{id} → 204, pas de corps retourné.
    func deleteConversation(id: UUID) async throws {
        let request = makeRequest(method: "DELETE", path: "/api/conversations/\(id)")
        _ = try await perform(request, expectedStatuses: [204])
    }

    // MARK: - Messages

    /// GET /api/conversations/{id}/messages/ → historique user+assistant (system exclu).
    func getMessages(conversationID: UUID) async throws -> [Message] {
        let path = "/api/conversations/\(conversationID)/messages/"
        let request = makeRequest(method: "GET", path: path)
        let data = try await perform(request)
        let messages = try decode([APIMessage].self, from: data)
        return messages.map(toDomain)
    }

    /// POST /api/conversations/{id}/messages/ → génération complète (bloquant).
    /// Retourne les deux messages créés : user et assistant.
    func sendMessageSync(
        conversationID: UUID,
        content: String,
        maxTokens: Int? = nil,
        temperature: Float? = nil
    ) async throws -> (user: Message, assistant: Message) {
        var request = makeRequest(
            method: "POST",
            path: "/api/conversations/\(conversationID)/messages/"
        )
        request.httpBody = try encoder.encode(SendMessageBody(
            content: content,
            max_tokens: maxTokens,
            temperature: temperature
        ))
        let data = try await perform(request, expectedStatuses: [201])
        let resp = try decode(APISendMessageResponse.self, from: data)
        return (toDomain(resp.userMessage), toDomain(resp.assistantMessage))
    }

    /// POST /api/conversations/{id}/messages/stream → stream SSE de chunks texte.
    ///
    /// Utilisation :
    /// ```swift
    /// for try await chunk in api.sendMessage(conversationID: id, content: "Bonjour") {
    ///     text += chunk
    /// }
    /// ```
    ///
    /// Annulation : `Task.cancel()` ferme la connexion HTTP.
    /// Le backend sauvegarde alors le texte partiel accumulé.
    nonisolated func sendMessage(
        conversationID: UUID,
        content: String,
        maxTokens: Int? = nil,
        temperature: Float? = nil
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var request = URLRequest(
                        url: self.baseURL.appendingPathComponent(
                            "/api/conversations/\(conversationID)/messages/stream"
                        )
                    )
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    request.timeoutInterval = 60
                    request.httpBody = try JSONEncoder().encode(SendMessageBody(
                        content: content,
                        max_tokens: maxTokens,
                        temperature: temperature
                    ))

                    let config = URLSessionConfiguration.default
                    config.timeoutIntervalForRequest  = 60
                    config.timeoutIntervalForResource = 300
                    config.waitsForConnectivity = false
                    let streamSession = URLSession(configuration: config)

                    let (bytes, response) = try await streamSession.bytes(for: request)
                    try Self.validateHTTPResponse(response, data: nil)

                    for try await line in bytes.lines {
                        try Task.checkCancellation()
                        guard line.hasPrefix("data: ") else { continue }
                        let payload = String(line.dropFirst(6))

                        if payload == "[DONE]" {
                            continuation.finish()
                            return
                        }

                        guard let raw = payload.data(using: .utf8) else { continue }

                        // Décoder uniquement text/error, ignorer userMessage du premier event
                        if let json = try? JSONSerialization.jsonObject(with: raw) as? [String: Any] {
                            if let errorMsg = json["error"] as? String {
                                continuation.finish(throwing: ConversationAPIError.streamError(errorMsg))
                                return
                            }
                            if let text = json["text"] as? String {
                                continuation.yield(text)
                            }
                            // json["userMessage"] → ignoré dans le stream texte
                        }
                    }

                    continuation.finish()

                } catch is CancellationError {
                    continuation.finish(throwing: ConversationAPIError.cancelled)
                } catch let err as ConversationAPIError {
                    continuation.finish(throwing: err)
                } catch let urlErr as URLError {
                    continuation.finish(throwing: ConversationAPIError.networkError(urlErr))
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            // Propager la cancellation du stream vers la URLSession task
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Private: HTTP

    private func perform(
        _ request: URLRequest,
        expectedStatuses: Set<Int> = [200, 201]
    ) async throws -> Data {
        do {
            let (data, response) = try await session.data(for: request)
            try Self.validateHTTPResponse(response, data: data, expectedStatuses: expectedStatuses)
            return data
        } catch let err as ConversationAPIError {
            throw err
        } catch let urlErr as URLError {
            throw ConversationAPIError.networkError(urlErr)
        }
    }

    private func makeRequest(method: String, path: String) -> URLRequest {
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        return request
    }

    private static func validateHTTPResponse(
        _ response: URLResponse,
        data: Data?,
        expectedStatuses: Set<Int> = [200, 201]
    ) throws {
        guard let http = response as? HTTPURLResponse else { return }
        if expectedStatuses.contains(http.statusCode) { return }

        let detail = data.flatMap { extractDetail(from: $0) } ?? "Erreur inconnue"

        switch http.statusCode {
        case 404: throw ConversationAPIError.notFound
        case 503: throw ConversationAPIError.modelLoading(detail)
        default:  throw ConversationAPIError.serverError(http.statusCode, detail)
        }
    }

    private static func extractDetail(from data: Data) -> String? {
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        return json?["detail"] as? String
    }

    // MARK: - Private: Decoding

    private func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        do {
            return try decoder.decode(type, from: data)
        } catch {
            throw ConversationAPIError.decodingError(
                "\(T.self): \(error.localizedDescription)"
            )
        }
    }

    // MARK: - Private: DTO -> Domain

    private func toDomain(_ role: APIRole) -> Message.Role {
        switch role {
        case .user: return .user
        case .assistant: return .assistant
        case .system: return .system
        }
    }

    private func toDomain(_ message: APIMessage) -> Message {
        Message(
            id: message.id,
            role: toDomain(message.role),
            content: message.content,
            timestamp: message.timestamp
        )
    }

    private func toDomain(_ conversation: APIConversation) -> Conversation {
        Conversation(
            id: conversation.id,
            messages: conversation.messages.map(toDomain)
        )
    }

    private func toDomain(_ meta: APIConversationMetadata) -> ConversationMetadata {
        ConversationMetadata(
            id: meta.id,
            title: meta.title,
            createdAt: meta.createdAt,
            updatedAt: meta.updatedAt,
            messageCount: meta.messageCount
        )
    }


    /// JSONDecoder configuré pour ISO 8601 avec microsecondes.
    ///
    /// Le backend sérialise `datetime.isoformat()` :
    ///   → `"2026-03-12T10:30:00.123456+00:00"` (avec microsecondes)
    ///
    /// `dateDecodingStrategy = .iso8601` échoue sur les microsecondes.
    /// Solution : formatters en cascade — avec microsecondes en premier, fallback sans.
    private static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()

        let withMicros = ISO8601DateFormatter()
        withMicros.formatOptions = [.withInternetDateTime, .withFractionalSeconds]

        let withoutMicros = ISO8601DateFormatter()
        withoutMicros.formatOptions = [.withInternetDateTime]

        decoder.dateDecodingStrategy = .custom { dec in
            let container = try dec.singleValueContainer()
            let string = try container.decode(String.self)

            if let date = withMicros.date(from: string)    { return date }
            if let date = withoutMicros.date(from: string) { return date }

            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Date ISO 8601 invalide : \(string)"
            )
        }

        return decoder
    }
}
