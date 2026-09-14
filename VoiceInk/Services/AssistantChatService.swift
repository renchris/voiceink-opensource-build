import Foundation
import LLMkit
import SwiftData

@MainActor
final class AssistantChatService {
    struct Reply {
        let text: String
        let duration: TimeInterval
        let systemPrompt: String?
        let requestLog: String
        /// Who actually answered — a ladder rung when the configured model was
        /// cooling — so the saved row names the model that wrote the text.
        let provider: AIProvider
        let modelName: String
    }

    private let modelContext: ModelContext
    private let aiService: AIService
    private let enhancementService: AIEnhancementService?

    private var requestTimeout: TimeInterval {
        let stored = UserDefaults.standard.integer(forKey: "EnhancementTimeoutSeconds")
        return stored > 0 ? TimeInterval(stored) : 7
    }

    init(modelContext: ModelContext, aiService: AIService, enhancementService: AIEnhancementService?) {
        self.modelContext = modelContext
        self.aiService = aiService
        self.enhancementService = enhancementService
    }

    func requestAssistantReply(
        provider: AIProvider,
        modelName: String?,
        systemPrompt: String?,
        messages: [AssistantDisplayMessage]
    ) async throws -> Reply {
        let chatMessages = messages.map { message in
            switch message.role {
            case .user:
                return ChatMessage.user(message.content)
            case .assistant:
                return ChatMessage.assistant(message.content)
            }
        }

        // Follow-ups bypass enhance(), so they consult the cooldown map here: a
        // model already refusing on quota would only refuse again. When every
        // rung is cooling, ask the configured model anyway — for an assistant the
        // answer IS the output, so a wait beats silence (the pipeline's exemption).
        let textLength = messages.reduce(0) { $0 + $1.content.count }
        let target = enhancementService?.resolveModel(
            provider: provider,
            modelName: modelName,
            textLength: textLength
        ) ?? (provider: provider, modelName: modelName ?? provider.defaultModel)

        let startTime = Date()
        let text: String
        do {
            text = try await aiService.completeChat(
                provider: target.provider,
                modelName: target.modelName,
                messages: chatMessages,
                systemPrompt: systemPrompt,
                timeout: requestTimeout
            )
        } catch {
            // A cancelled turn says nothing about the model's quota.
            if !(error is CancellationError) {
                enhancementService?.recordOutcome(
                    provider: target.provider, modelName: target.modelName, error: error)
            }
            throw error
        }
        enhancementService?.recordOutcome(provider: target.provider, modelName: target.modelName, error: nil)

        return Reply(
            text: text,
            duration: Date().timeIntervalSince(startTime),
            systemPrompt: systemPrompt,
            requestLog: Self.requestLog(from: messages),
            provider: target.provider,
            modelName: target.modelName
        )
    }

    func applyAssistantTurn(
        transcription: Transcription,
        response: Reply,
        promptName: String?
    ) {
        transcription.enhancedText = response.text
        transcription.aiEnhancementModelName = response.modelName
        transcription.promptName = promptName
        transcription.enhancementDuration = response.duration
        transcription.aiRequestSystemMessage = response.systemPrompt
        transcription.aiRequestUserMessage = response.requestLog
        transcription.transcriptionStatus = TranscriptionStatus.completed.rawValue
    }

    func saveTypedAssistantTurn(
        input: String,
        response: Reply,
        promptName: String?,
        modeName: String?,
        modeEmoji: String?
    ) throws {
        let transcription = Transcription(
            text: input,
            duration: 0,
            enhancedText: response.text,
            aiEnhancementModelName: response.modelName,
            promptName: promptName,
            enhancementDuration: response.duration,
            aiRequestSystemMessage: response.systemPrompt,
            aiRequestUserMessage: response.requestLog,
            modeName: modeName,
            modeEmoji: modeEmoji,
            transcriptionStatus: .completed
        )

        modelContext.insert(transcription)
        try modelContext.save()
        NotificationCenter.default.post(name: .transcriptionCreated, object: transcription)
        NotificationCenter.default.post(name: .transcriptionCompleted, object: transcription)
    }

    private static func requestLog(from messages: [AssistantDisplayMessage]) -> String {
        messages.map { message in
            let label: String
            switch message.role {
            case .assistant:
                label = "Assistant"
            case .user:
                label = "User"
            }
            return "\(label):\n\(message.content)"
        }
        .joined(separator: "\n\n")
    }
}
