import Foundation
import LLMkit
import SwiftUI

class OllamaService: ObservableObject {
    static let defaultBaseURL = "http://localhost:11434"

    // MARK: - Published Properties
    @Published var baseURL: String {
        didSet {
            UserDefaults.standard.set(baseURL, forKey: "ollamaBaseURL")
        }
    }

    @Published var selectedModel: String {
        didSet {
            UserDefaults.standard.set(selectedModel, forKey: "ollamaSelectedModel")
        }
    }

    @Published var availableModels: [OllamaModel] = []
    @Published var isConnected: Bool = false
    @Published var isLoadingModels: Bool = false

    private let defaultTemperature: Double = 0.3

    init() {
        self.baseURL = UserDefaults.standard.string(forKey: "ollamaBaseURL") ?? Self.defaultBaseURL
        self.selectedModel = UserDefaults.standard.string(forKey: "ollamaSelectedModel") ?? "llama2"
    }

    private var baseURLValue: URL? {
        URL(string: baseURL)
    }

    @MainActor
    func checkConnection() async {
        guard let url = baseURLValue else {
            isConnected = false
            return
        }
        isConnected = await OllamaClient.checkConnection(baseURL: url)
    }

    @MainActor
    func refreshModels() async {
        _ = await refreshConnectionAndModels()
    }

    @MainActor
    func refreshConnectionAndModels() async -> Result<[OllamaModel], Error> {
        isLoadingModels = true
        defer { isLoadingModels = false }

        guard let url = baseURLValue else {
            isConnected = false
            availableModels = []
            return .failure(LocalAIError.invalidURL)
        }

        do {
            let models = try await OllamaClient.fetchModels(baseURL: url)
            isConnected = true
            availableModels = models

            if !models.contains(where: { $0.name == selectedModel }) && !models.isEmpty {
                selectedModel = models[0].name
            }

            return .success(models)
        } catch {
            isConnected = false
            availableModels = []
            return .failure(error)
        }
    }

    func enhance(
        _ text: String, withSystemPrompt systemPrompt: String? = nil, model: String? = nil, timeout: TimeInterval = 30
    ) async throws -> String {
        guard let systemPrompt = systemPrompt else {
            throw LocalAIError.invalidRequest
        }

        guard let url = baseURLValue else {
            throw LocalAIError.invalidURL
        }

        let trimmedModel = model?.trimmingCharacters(in: .whitespacesAndNewlines)
        let requestModel = (trimmedModel?.isEmpty == false ? trimmedModel : nil) ?? selectedModel

        // Posted directly rather than through OllamaClient.generate, which sends temperature at the top
        // level (ignored by Ollama), no num_ctx, no keep_alive, and decodes only `response` — so the
        // model unloaded between dictations (38 s cold vs 3 s warm against a 12 s rung) and a truncated
        // reply was pasted as the user's text. keep_alive is per-request only; a Modelfile rejects it.
        // num_predict is deliberately absent: a cap truncates, while the caller's timeout already bounds
        // generation and a timeout descends the fallback ladder instead of pasting a fragment.
        let body = OllamaGenerateRequest(
            model: requestModel,
            prompt: text,
            system: systemPrompt,
            stream: false,
            think: false,
            keepAlive: "30m",
            options: .init(temperature: defaultTemperature, numCtx: 8192)
        )

        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        guard let bodyData = try? encoder.encode(body) else {
            throw LocalAIError.invalidRequest
        }

        var request = URLRequest(url: url.appendingPathComponent("api/generate"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = timeout
        request.httpBody = bodyData

        try Task.checkCancellation()

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch let error as URLError {
            switch error.code {
            case .timedOut:
                throw LocalAIError.timeout
            case .cancelled:
                // A cancelled recording must not read as an outage and push the ladder to the next rung.
                throw CancellationError()
            default:
                throw LocalAIError.serviceUnavailable
            }
        }

        guard let http = response as? HTTPURLResponse else {
            throw LocalAIError.invalidResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 404 { throw LocalAIError.modelNotFound }
            if http.statusCode == 500 { throw LocalAIError.serverError }
            throw LocalAIError.invalidResponse
        }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        guard let reply = try? decoder.decode(OllamaGenerateReply.self, from: data) else {
            throw LocalAIError.invalidResponse
        }
        // A reply cut off at the context limit is a fragment of the rewrite; pasting it would replace
        // the user's words with half of them.
        guard reply.doneReason != "length" else {
            throw LocalAIError.invalidResponse
        }
        return reply.response
    }

    private func mapLLMKitError(_ error: LLMKitError) -> LocalAIError {
        switch error {
        case .invalidURL:
            return .invalidURL
        case .httpError(let statusCode, _):
            if statusCode == 404 { return .modelNotFound }
            if statusCode == 500 { return .serverError }
            return .invalidResponse
        case .networkError:
            return .serviceUnavailable
        case .noResultReturned, .decodingError:
            return .invalidResponse
        case .encodingError:
            return .invalidRequest
        case .missingAPIKey:
            return .invalidResponse
        case .timeout:
            return .timeout
        }
    }
}

// MARK: - Generate Wire Types
// Coded with snake_case key strategies, so `keepAlive` goes over the wire as `keep_alive`.
private struct OllamaGenerateRequest: Encodable {
    struct Options: Encodable {
        let temperature: Double
        let numCtx: Int
    }

    let model: String
    let prompt: String
    let system: String
    let stream: Bool
    let think: Bool
    let keepAlive: String
    let options: Options
}

private struct OllamaGenerateReply: Decodable {
    let response: String
    let doneReason: String?
}

// MARK: - Error Types
enum LocalAIError: Error, LocalizedError {
    case invalidURL
    case serviceUnavailable
    case invalidResponse
    case modelNotFound
    case serverError
    case invalidRequest
    case timeout

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return String(localized: "Invalid Ollama server URL")
        case .serviceUnavailable:
            return String(localized: "Ollama service is not available")
        case .invalidResponse:
            return String(localized: "Invalid response from Ollama server")
        case .modelNotFound:
            return String(localized: "Selected model not found")
        case .serverError:
            return String(localized: "Ollama server error")
        case .invalidRequest:
            return String(localized: "System prompt is required")
        case .timeout:
            return String(localized: "Ollama request timed out")
        }
    }
}
