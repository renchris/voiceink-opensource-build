import Foundation
import os

/// Encapsulates a single recording-to-transcription lifecycle (streaming or file-based).
@MainActor
protocol TranscriptionSession: AnyObject {
    /// Prepares the session. Returns an audio chunk callback for streaming, or nil for file-based.
    func prepare(configuration: TranscriptionRuntimeConfiguration) async throws -> ((Data) -> Void)?

    /// Called after recording stops. Returns the final transcribed text.
    func transcribe(audioURL: URL) async throws -> String

    /// Cancel the session and clean up resources.
    func cancel()
}

// MARK: - File-Based Session

/// File-based session: records to file, uploads after stop.
@MainActor
final class FileTranscriptionSession: TranscriptionSession {
    private let service: TranscriptionService
    private var model: (any TranscriptionModel)?
    private var context: TranscriptionRequestContext = .currentDefaults

    init(service: TranscriptionService) {
        self.service = service
    }

    func prepare(configuration: TranscriptionRuntimeConfiguration) async throws -> ((Data) -> Void)? {
        self.model = configuration.model
        self.context = configuration.requestContext.scoped(to: configuration.model)
        return nil
    }

    func transcribe(audioURL: URL) async throws -> String {
        guard let model = model else {
            throw VoiceInkEngineError.transcriptionFailed
        }
        return try await service.transcribe(audioURL: audioURL, model: model, context: context)
    }

    func cancel() {
        // No-op for file-based transcription
    }
}

// MARK: - Streaming Session

/// Streaming session with automatic fallback to file-based upload on failure.
@MainActor
final class StreamingTranscriptionSession: TranscriptionSession {
    private let streamingService: StreamingTranscriptionService
    private let fallbackService: TranscriptionService
    private var model: (any TranscriptionModel)?
    private var context: TranscriptionRequestContext = .currentDefaults
    private var streamingFailed = false
    private var startupTask: Task<Void, Never>?
    private var startupTaskID: UUID?
    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "StreamingTranscriptionSession")

    /// Recordings at or above this length warn the user when they drop to batch decoding.
    /// Short clips fall back routinely — the agreement engine needs a few confirmed segments
    /// before it can finalize from the stream — and are not worth interrupting for.
    private static let batchFallbackWarningThreshold: TimeInterval = 120

    init(streamingService: StreamingTranscriptionService, fallbackService: TranscriptionService) {
        self.streamingService = streamingService
        self.fallbackService = fallbackService
    }

    func prepare(configuration: TranscriptionRuntimeConfiguration) async throws -> ((Data) -> Void)? {
        let model = configuration.model
        let context = configuration.requestContext.scoped(to: model)

        self.model = model
        self.context = context
        logger.notice("Streaming session prepare model=\(model.displayName, privacy: .public)")

        // Return callback immediately; WebSocket connects in background
        let service = streamingService
        let callback: (Data) -> Void = { [weak service] data in
            service?.sendAudioChunk(data)
        }

        startupTask?.cancel()
        let taskID = UUID()
        startupTaskID = taskID
        startupTask = Task { [weak self] in
            guard let self = self else { return }
            defer {
                if self.startupTaskID == taskID {
                    self.startupTask = nil
                    self.startupTaskID = nil
                }
            }
            guard !Task.isCancelled else { return }

            do {
                let start = Date()
                try await self.streamingService.startStreaming(model: model, context: context)
                guard !Task.isCancelled else {
                    self.streamingService.cancel()
                    return
                }
                self.logger.notice(
                    "Streaming session connected model=\(model.displayName, privacy: .public) elapsed=\(Date().timeIntervalSince(start), format: .fixed(precision: 3), privacy: .public)s"
                )
            } catch is CancellationError {
                self.streamingService.cancel()
            } catch {
                guard !Task.isCancelled else { return }
                let desc = error.localizedDescription
                self.logger.error("❌ Failed to start streaming, will fall back to batch: \(desc, privacy: .public)")
                self.streamingFailed = true
            }
        }

        return callback
    }

    func transcribe(audioURL: URL) async throws -> String {
        guard let model = model else {
            throw VoiceInkEngineError.transcriptionFailed
        }

        if !streamingFailed {
            do {
                let start = Date()
                logger.notice("Streaming stop/transcribe started model=\(model.displayName, privacy: .public)")
                let result = try await streamingService.stopAndFinalize()
                switch result {
                case .finalized(let text):
                    logger.notice(
                        "Streaming transcript received elapsed=\(Date().timeIntervalSince(start), format: .fixed(precision: 3), privacy: .public)s chars=\(text.count, privacy: .public)"
                    )
                    return text
                case .requiresBatchFallback:
                    logger.notice("Streaming provider requested full batch transcription")
                }
            } catch {
                logger.error("❌ Streaming failed, falling back to batch: \(error, privacy: .public)")
                startupTask?.cancel()
                startupTask = nil
                startupTaskID = nil
                streamingService.cancel()
            }
        } else {
            startupTask?.cancel()
            startupTask = nil
            startupTaskID = nil
            streamingService.cancel()
        }

        let fallbackStart = Date()
        logger.notice(
            "Using batch fallback for \(model.displayName, privacy: .public) file=\(audioURL.lastPathComponent, privacy: .public)"
        )

        // Batch decoding of long-form audio can omit content without raising an error, so a long
        // recording that silently drops to this path yields a transcript that looks complete but
        // is not. Surface it rather than substituting quietly — the WAV is always retained, so the
        // user can re-transcribe if the result looks short.
        let fallbackAudioDuration = await AudioFileMetadata.duration(for: audioURL)
        if fallbackAudioDuration >= Self.batchFallbackWarningThreshold {
            logger.warning(
                "Long recording fell back to batch decoding duration=\(fallbackAudioDuration, format: .fixed(precision: 1), privacy: .public)s — transcript may be incomplete"
            )
            NotificationManager.shared.showNotification(
                title: String(
                    format: String(localized: "Live transcription dropped out — %d min re-decoded in batch, may be incomplete"),
                    max(1, Int(fallbackAudioDuration / 60))
                ),
                type: .warning,
                duration: 10.0
            )
        }

        let text = try await fallbackService.transcribe(audioURL: audioURL, model: model, context: context)
        logger.notice(
            "Batch fallback completed elapsed=\(Date().timeIntervalSince(fallbackStart), format: .fixed(precision: 3), privacy: .public)s chars=\(text.count, privacy: .public)"
        )
        return text
    }

    func cancel() {
        startupTask?.cancel()
        startupTask = nil
        startupTaskID = nil
        streamingService.cancel()
    }
}
