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

// MARK: - Stream Events (protocole BFF)

/// Événement typé émis par le backend SSE.
/// Le client consomme ces événements pour construire le message assistant en temps réel.
struct StreamEvent {
    enum EventType: String {
        case start
        case textDelta
        case reasoningDelta
        case toolCallStart
        case toolCallResult
        case error
        case done
        case unknown
    }
    let type: EventType
    var content: String?
    var toolCallId: String?
    var toolName: String?
    var preview: ToolPreview?   // aperçu structuré pour toolCallResult
}

// MARK: - Private API response types
// DTO réseau (contrat backend) séparés des modèles métier Swift.

private enum APIRole: String, Decodable {
    case user
    case assistant
    case system
}

/// Part d'un message tel que retourné par GET /messages/
private struct APIMessagePart: Decodable {
    let type:       String
    let content:    String?
    let toolCallId: String?
    let toolName:   String?
}

private struct APIMessage: Decodable {
    let id:        UUID
    let role:      APIRole
    let parts:     [APIMessagePart]
    let timestamp: Date
}

private struct APIConversation: Decodable {
    let id:       UUID
    let messages: [APIMessage]
}

private struct APIConversationMetadata: Decodable {
    let id:                  UUID
    let title:               String
    let createdAt:           Date
    let updatedAt:           Date
    let messageCount:        Int
    let specificInstruction: String?
}

private struct APISendMessageResponse: Decodable {
    let userMessage:      APIMessage
    let assistantMessage: APIMessage
}

/// Enveloppe brute d'un event SSE BFF — décodage du JSON dans la ligne "data: ..."
private struct RawSSEEvent: Decodable {
    let type:       String?
    let content:    String?
    let toolCallId: String?
    let toolName:   String?
    let error:      String?
    let preview:    ToolPreview?    // aperçu structuré (toolCallResult uniquement)

    enum CodingKeys: String, CodingKey {
        case type, content, toolCallId, toolName, error, preview
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        type       = try c.decodeIfPresent(String.self, forKey: .type)
        content    = try c.decodeIfPresent(String.self, forKey: .content)
        toolCallId = try c.decodeIfPresent(String.self, forKey: .toolCallId)
        toolName   = try c.decodeIfPresent(String.self, forKey: .toolName)
        error      = try c.decodeIfPresent(String.self, forKey: .error)
        // Décodage tolérant : si le preview est malformé, on ignore sans planter l'event
        do {
            preview = try c.decodeIfPresent(ToolPreview.self, forKey: .preview)
        } catch {
            preview = nil
        }
    }
}

// MARK: - Request bodies

private struct CreateConversationBody: Encodable {
    let title:        String
}

private struct UpdateConversationBody: Encodable {
    let title: String
    let specificInstruction: String?
}

private struct SendMessageBody: Encodable {
    let content:     String
    let temperature: Float?
    let options:     [String: Bool]?
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
    func updateConversation(id: UUID, title: String? = nil, specificInstruction: String? = nil) async throws -> ConversationMetadata {
        var request = makeRequest(method: "PUT", path: "/api/conversations/\(id)")
        request.httpBody = try encoder.encode(UpdateConversationBody(title: title!, specificInstruction: specificInstruction!))
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
    /// Chaque message contient un tableau de parts (text, reasoning, toolCall…).
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
        options: [String: Bool]? = nil,
        temperature: Float? = nil
    ) async throws -> (user: Message, assistant: Message) {
        var request = makeRequest(
            method: "POST",
            path: "/api/conversations/\(conversationID)/messages/"
        )
        request.httpBody = try encoder.encode(SendMessageBody(
            content: content,
            temperature: temperature,
            options: options
        ))
        let data = try await perform(request, expectedStatuses: [201])
        let resp = try decode(APISendMessageResponse.self, from: data)
        return (toDomain(resp.userMessage), toDomain(resp.assistantMessage))
    }

    /// POST /api/conversations/{id}/messages/stream → stream SSE d'événements typés.
    ///
    /// Utilisation :
    /// ```swift
    /// for try await event in api.sendMessage(conversationID: id, content: "Bonjour") {
    ///     switch event.type {
    ///     case .textDelta: accumuler event.content
    ///     case .reasoningDelta: afficher en DisclosureGroup
    ///     case .toolCallStart: montrer un indicateur de chargement
    ///     ...
    ///     }
    /// }
    /// ```
    ///
    /// Annulation : `Task.cancel()` ferme la connexion HTTP.
    nonisolated func sendMessage(
        conversationID: UUID,
        content: String,
        options: [String: Bool]? = nil,
        temperature: Float? = nil
    ) -> AsyncThrowingStream<StreamEvent, Error> {
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
                        temperature: temperature,
                        options: options
                    ))

                    let config = URLSessionConfiguration.default
                    config.timeoutIntervalForRequest  = 60
                    config.timeoutIntervalForResource = 300
                    config.waitsForConnectivity = false
                    let streamSession = URLSession(configuration: config)

                    let (bytes, response) = try await streamSession.bytes(for: request)
                    try Self.validateHTTPResponse(response, data: nil)

                    let decoder = JSONDecoder()

                    for try await line in bytes.lines {
                        try Task.checkCancellation()
                        guard line.hasPrefix("data: ") else { continue }
                        let payload = String(line.dropFirst(6))

                        // [DONE] n'est PAS du JSON — intercepter avant tout décodage
                        if payload == "[DONE]" {
                            continuation.finish()
                            return
                        }

                        guard let raw = payload.data(using: .utf8) else { continue }

                        guard let evt = try? decoder.decode(RawSSEEvent.self, from: raw) else { continue }

                        let eventType = StreamEvent.EventType(rawValue: evt.type ?? "") ?? .unknown

                        switch eventType {
                        case .error:
                            let msg = evt.content ?? evt.error ?? "Erreur inconnue"
                            continuation.finish(throwing: ConversationAPIError.streamError(msg))
                            return
                        case .done:
                            // Le backend émet {"type":"done"} puis [DONE] — on attend [DONE]
                            continue
                        case .start, .unknown:
                            continue
                        default:
                            continuation.yield(StreamEvent(
                                type: eventType,
                                content: evt.content,
                                toolCallId: evt.toolCallId,
                                toolName: evt.toolName,
                                preview: evt.preview
                            ))
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

    private func toDomain(_ part: APIMessagePart) -> MessagePart {
        let partType = MessagePart.PartType(rawValue: part.type) ?? .text
        // Les tool calls persistés en historique sont par définition terminés.
        // isCompleted = nil (défaut) déclencherait le spinner au lieu du checkmark.
        let isCompleted: Bool? = (partType == .toolCall) ? true : nil
        return MessagePart(
            type: partType,
            content: part.content,
            toolCallId: part.toolCallId,
            toolName: part.toolName,
            isCompleted: isCompleted
        )
    }

    private func toDomain(_ message: APIMessage) -> Message {
        Message(
            id: message.id,
            role: toDomain(message.role),
            parts: message.parts.map(toDomain),
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
            messageCount: meta.messageCount,
            specificInstruction: meta.specificInstruction
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
