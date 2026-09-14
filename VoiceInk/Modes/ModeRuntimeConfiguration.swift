import Foundation

struct TranscriptionRuntimeConfiguration {
    let mode: ModeConfig
    let model: any TranscriptionModel
    let language: String
    let isRealtimeEnabled: Bool

    var metadata: (name: String?, emoji: String?) {
        guard mode.isEnabled else {
            return (nil, nil)
        }
        return (mode.name, mode.icon.value)
    }

    var requestContext: TranscriptionRequestContext {
        TranscriptionRequestContext(
            language: language,
            prompt: model.provider == .whisper ? UserDefaults.standard.string(forKey: "TranscriptionPrompt") : nil
        )
    }
}

struct TranscriptionFormattingConfiguration {
    let mode: ModeConfig?
    let isTextFormattingEnabled: Bool
}

/// One model of one provider — a rung of the enhancement fallback ladder, and
/// the anchor a substituted configuration remembers.
struct EnhancementModelRef: Hashable {
    let provider: AIProvider
    let modelName: String
}

struct EnhancementRuntimeConfiguration {
    let mode: ModeConfig?
    let isEnabled: Bool
    let prompt: CustomPrompt?
    let provider: AIProvider?
    let modelName: String?
    let useClipboardContext: Bool
    let useSelectedTextContext: Bool
    let useScreenCaptureContext: Bool
    /// The model the mode asked for, when this configuration is a stand-in for
    /// it. The ladder is always built from that anchor: re-deriving it from the
    /// substitute would put the substitute's provider first and lose the order.
    var substitutedFrom: EnhancementModelRef? = nil

    var isSubstitute: Bool { substitutedFrom != nil }

    /// The model this configuration names, with the provider's default standing
    /// in for an unset name — the same identity the quota cooldowns key on.
    var modelRef: EnhancementModelRef? {
        guard let provider else { return nil }
        return EnhancementModelRef(provider: provider, modelName: modelName ?? provider.defaultModel)
    }

    /// The configuration the mode asked for, before any substitution.
    var anchor: EnhancementRuntimeConfiguration {
        guard let substitutedFrom else { return self }
        return EnhancementRuntimeConfiguration(
            mode: mode,
            isEnabled: isEnabled,
            prompt: prompt,
            provider: substitutedFrom.provider,
            modelName: substitutedFrom.modelName,
            useClipboardContext: useClipboardContext,
            useSelectedTextContext: useSelectedTextContext,
            useScreenCaptureContext: useScreenCaptureContext
        )
    }

    /// Same request, aimed at a different model — a rung of the fallback ladder.
    /// Remembers the original anchor however many rungs deep the walk goes, and
    /// aiming back at the anchor itself yields the anchor, not a "substitute".
    func replacingModel(provider: AIProvider, modelName: String) -> EnhancementRuntimeConfiguration {
        let anchorRef = substitutedFrom ?? modelRef
        return EnhancementRuntimeConfiguration(
            mode: mode,
            isEnabled: isEnabled,
            prompt: prompt,
            provider: provider,
            modelName: modelName,
            useClipboardContext: useClipboardContext,
            useSelectedTextContext: useSelectedTextContext,
            useScreenCaptureContext: useScreenCaptureContext,
            substitutedFrom: anchorRef == EnhancementModelRef(provider: provider, modelName: modelName)
                ? nil : anchorRef
        )
    }

    func replacingPrompt(_ prompt: CustomPrompt) -> EnhancementRuntimeConfiguration {
        EnhancementRuntimeConfiguration(
            mode: mode,
            isEnabled: true,
            prompt: prompt,
            provider: provider,
            modelName: modelName,
            useClipboardContext: useClipboardContext,
            useSelectedTextContext: useSelectedTextContext,
            useScreenCaptureContext: useScreenCaptureContext,
            substitutedFrom: substitutedFrom
        )
    }
}

struct OutputRuntimeConfiguration {
    let mode: ModeConfig?
    let outputMode: ModeOutputMode
    let autoSendKey: AutoSendKey
    let customCommand: ModeCustomCommand?
}

enum ModeTranscriptionModelResolution {
    case noMode
    case noSelection(mode: ModeConfig)
    case modelNotFound(mode: ModeConfig)
    case unavailable(mode: ModeConfig, model: any TranscriptionModel)
    case available(mode: ModeConfig, model: any TranscriptionModel)
}

@MainActor
enum ModeRuntimeResolver {
    static func transcriptionModelResolution(
        mode: ModeConfig? = nil,
        transcriptionModelManager: TranscriptionModelManager
    ) -> ModeTranscriptionModelResolution {
        guard let mode = mode ?? ModeManager.shared.currentEffectiveConfiguration else {
            return .noMode
        }

        guard let modelName = mode.selectedTranscriptionModelName,
            !modelName.isEmpty
        else {
            return .noSelection(mode: mode)
        }

        guard
            let model = transcriptionModelManager.allAvailableModels.first(where: {
                $0.name == modelName
            })
        else {
            return .modelNotFound(mode: mode)
        }

        guard transcriptionModelManager.usableModels.contains(where: { $0.name == modelName }) else {
            return .unavailable(mode: mode, model: model)
        }

        return .available(mode: mode, model: model)
    }

    static func transcriptionConfiguration(
        mode: ModeConfig? = nil,
        transcriptionModelManager: TranscriptionModelManager
    ) -> TranscriptionRuntimeConfiguration? {
        transcriptionConfiguration(
            from: transcriptionModelResolution(
                mode: mode,
                transcriptionModelManager: transcriptionModelManager
            )
        )
    }

    static func transcriptionConfiguration(
        from resolution: ModeTranscriptionModelResolution
    ) -> TranscriptionRuntimeConfiguration? {
        guard
            case .available(let mode, let model) = resolution
        else {
            return nil
        }

        let language = TranscriptionLanguageSupport.validLanguageOrFallback(
            mode.selectedLanguage,
            for: model,
            realtimeEnabled: mode.isRealtimeTranscriptionEnabled
        )

        return TranscriptionRuntimeConfiguration(
            mode: mode,
            model: model,
            language: language,
            isRealtimeEnabled: TranscriptionRealtimeSupport.isEnabled(
                for: model, modeValue: mode.isRealtimeTranscriptionEnabled)
        )
    }

    static func transcriptionFormattingConfiguration(mode: ModeConfig? = nil) -> TranscriptionFormattingConfiguration {
        let mode = mode ?? ModeManager.shared.currentEffectiveConfiguration

        return TranscriptionFormattingConfiguration(
            mode: mode,
            isTextFormattingEnabled: mode?.isTextFormattingEnabled
                ?? UserDefaults.standard.bool(forKey: "IsTextFormattingEnabled")
        )
    }

    static func currentEnhancementConfiguration(
        mode: ModeConfig? = nil,
        enhancementService: AIEnhancementService,
        aiService: AIService
    ) -> EnhancementRuntimeConfiguration {
        let mode = mode ?? ModeManager.shared.currentEffectiveConfiguration
        let prompt = resolvedPrompt(
            promptId: mode?.selectedPrompt,
            enhancementService: enhancementService
        )
        let provider = resolvedProvider(
            providerName: mode?.selectedAIProvider,
            aiService: aiService
        )
        let modelName = resolvedEnhancementModelName(
            provider: provider,
            configuredModelName: mode?.selectedAIModel,
            aiService: aiService
        )

        return EnhancementRuntimeConfiguration(
            mode: mode,
            isEnabled: mode?.isAIEnhancementEnabled ?? false,
            prompt: prompt,
            provider: provider,
            modelName: modelName,
            useClipboardContext: mode?.useClipboardContext ?? false,
            useSelectedTextContext: mode?.useSelectedTextContext ?? true,
            useScreenCaptureContext: mode?.useScreenCapture ?? false
        )
    }

    static func outputConfiguration(mode: ModeConfig? = nil) -> OutputRuntimeConfiguration {
        let mode = mode ?? ModeManager.shared.currentEffectiveConfiguration

        return OutputRuntimeConfiguration(
            mode: mode,
            outputMode: mode?.outputMode ?? .paste,
            autoSendKey: mode?.autoSendKey ?? .none,
            customCommand: mode?.customCommand
        )
    }

    private static func resolvedPrompt(
        promptId: String?,
        enhancementService: AIEnhancementService
    ) -> CustomPrompt? {
        guard let promptId,
            let uuid = UUID(uuidString: promptId)
        else {
            return nil
        }

        return enhancementService.allPrompts.first { $0.id == uuid }
    }

    private static func resolvedProvider(
        providerName: String?,
        aiService: AIService
    ) -> AIProvider? {
        if let providerName,
            let provider = AIProvider(rawValue: providerName),
            aiService.connectedProviders.contains(provider)
        {
            return provider
        }

        return aiService.connectedProviders.first
    }

    private static func resolvedEnhancementModelName(
        provider: AIProvider?,
        configuredModelName: String?,
        aiService: AIService
    ) -> String? {
        guard let provider else { return nil }

        if provider == .localCLI {
            return nil
        }

        let models = aiService.availableModels(for: provider)
        if let configuredModelName,
            !configuredModelName.isEmpty,
            (models.isEmpty || models.contains(configuredModelName))
        {
            return configuredModelName
        }

        if let firstModel = models.first {
            return firstModel
        }

        return provider.defaultModel
    }
}
