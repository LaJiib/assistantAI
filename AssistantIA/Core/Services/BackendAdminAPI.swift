// BackendAdminAPI.swift
// AssistantIA
//
// Client d'administration minimal du backend.
// V1: health check typé (base pour futures actions admin).

import Foundation

struct BackendHealth: Decodable {
    let status: String
    let modelLoaded: Bool
    let modelName: String
    let pid: Int
}

enum BackendAdminError: LocalizedError {
    case unhealthyStatus(Int)
    case network(URLError)
    case invalidResponse

    var errorDescription: String? {
        switch self {
        case .unhealthyStatus(let code):
            return "Backend unhealthy (HTTP \(code))."
        case .network(let err):
            return err.localizedDescription
        case .invalidResponse:
            return "Réponse backend invalide."
        }
    }
}

final class BackendAdminAPI {
    private let baseURL: URL
    private let session: URLSession
    private let decoder: JSONDecoder

    init(baseURL: URL, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.session = session

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        self.decoder = decoder
    }

    func health() async throws -> BackendHealth {
        let url = baseURL.appendingPathComponent("/health")
        do {
            let (data, response) = try await session.data(from: url)
            guard let http = response as? HTTPURLResponse else {
                throw BackendAdminError.invalidResponse
            }
            guard http.statusCode == 200 else {
                throw BackendAdminError.unhealthyStatus(http.statusCode)
            }
            return try decoder.decode(BackendHealth.self, from: data)
        } catch let err as URLError {
            throw BackendAdminError.network(err)
        }
    }
}

