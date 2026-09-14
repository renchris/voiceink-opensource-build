import Foundation
import Testing
@testable import VoiceInk

/// The fallback ladder's pure decisions. Nothing here boots `AIEnhancementService`
/// or touches `UserDefaults.standard`: the test host shares its bundle id with the
/// copy of VoiceInk the user is running.
struct EnhancementLadderPolicyTests {

    private typealias Policy = EnhancementLadderPolicy

    private func ref(_ provider: AIProvider, _ modelName: String) -> EnhancementModelRef {
        EnhancementModelRef(provider: provider, modelName: modelName)
    }

    private func configuration(_ provider: AIProvider, _ modelName: String) -> EnhancementRuntimeConfiguration {
        EnhancementRuntimeConfiguration(
            mode: nil,
            isEnabled: true,
            prompt: nil,
            provider: provider,
            modelName: modelName,
            useClipboardContext: false,
            useSelectedTextContext: false,
            useScreenCaptureContext: false
        )
    }

    // MARK: - Ladder order

    @Test func ladderOrdersModeSelectionsThenProviderOrderThenOtherCloudThenLocalLast() {
        let anchor = ref(.gemini, "gemini-3.8-flash")
        let otherProviders = [
            ref(.ollama, "gemma4:26b"),
            ref(.anthropic, "claude-haiku-4-5"),
            ref(.gemini, "gemini-9-flash"),
            ref(.openAI, "gpt-5.5"),
            ref(.groq, ""),
            ref(.anthropic, "claude-haiku-4-5"),
        ]
        let modeSelections = ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.5-flash"]
        let providerOrder = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.6-flash"]
        let excluded = [ref(.gemini, "gemini-3.7-flash"), ref(.openAI, "gpt-5.5")]

        let armed = Policy.ladder(
            anchor: anchor,
            modeSelections: modeSelections,
            providerOrder: providerOrder,
            otherProviders: otherProviders,
            includeLocal: true,
            excluding: excluded
        )
        #expect(armed == [
            ref(.gemini, "gemini-3.5-flash"),
            ref(.gemini, "gemini-3.6-flash"),
            ref(.anthropic, "claude-haiku-4-5"),
            ref(.ollama, "gemma4:26b"),
        ])

        let unarmed = Policy.ladder(
            anchor: anchor,
            modeSelections: modeSelections,
            providerOrder: providerOrder,
            otherProviders: otherProviders,
            includeLocal: false,
            excluding: excluded
        )
        #expect(unarmed == Array(armed.dropLast()))
        #expect(!unarmed.contains { Policy.isLocal($0.provider) })

        for ladder in [armed, unarmed] {
            #expect(!ladder.contains(anchor))
            #expect(ladder.allSatisfy { !excluded.contains($0) })
            #expect(Set(ladder).count == ladder.count)
        }
    }

    @Test func geminiLadderPutsFlashBeforeLiteAndLeavesProPreviewInThePickerOnly() {
        let fallbackOrder = AIProvider.gemini.fallbackOrder
        let availableModels = AIProvider.gemini.availableModels

        #expect(!fallbackOrder.contains("gemini-3.1-pro-preview"))
        #expect(availableModels.contains("gemini-3.1-pro-preview"))
        // A rung the picker does not know is a typo, and would be a dead rung.
        #expect(fallbackOrder.allSatisfy { availableModels.contains($0) })

        let ladder = Policy.ladder(
            anchor: ref(.gemini, "gemini-3.8-flash"),
            modeSelections: [],
            providerOrder: fallbackOrder,
            otherProviders: [],
            includeLocal: false
        )
        let names = ladder.map(\.modelName)
        let firstLite = names.firstIndex { $0.contains("lite") }
        let lastFlash = names.lastIndex { !$0.contains("lite") }
        #expect(firstLite != nil)
        #expect(lastFlash != nil)
        if let firstLite, let lastFlash {
            #expect(lastFlash < firstLite)
        }
        #expect(names.first == "gemini-3.7-flash")
    }

    // MARK: - Truncation guard

    @Test func rungIsSafeBlocksLongTextOnlyForOpenAICompatibleProviders() {
        let limit = Policy.openAICompatibleFallbackCharacterLimit
        #expect(limit == 6000)

        for provider in [AIProvider.openAI, .groq, .cerebras, .mistral, .openRouter] {
            #expect(Policy.rungIsSafe(ref(provider, "m"), textLength: limit))
            #expect(!Policy.rungIsSafe(ref(provider, "m"), textLength: limit + 1))
        }
        for provider in [AIProvider.gemini, .anthropic, .ollama, .localCLI] {
            #expect(Policy.rungIsSafe(ref(provider, "m"), textLength: limit + 1))
            #expect(Policy.rungIsSafe(ref(provider, "m"), textLength: 50_000))
        }
    }

    // MARK: - Timeouts

    @Test func timeoutOverrideTightensOnlyASubstitutedLocalRungAndNoSubstituteRetriesTimeout() {
        #expect(Policy.timeoutOverride(provider: .ollama, isSubstitute: true) == 12)
        #expect(Policy.timeoutOverride(provider: .localCLI, isSubstitute: true) == 12)
        #expect(Policy.timeoutOverride(provider: .gemini, isSubstitute: true) == nil)
        #expect(Policy.timeoutOverride(provider: .anthropic, isSubstitute: true) == nil)
        // D1: a mode that deliberately runs on a local model keeps the user's timeout.
        #expect(Policy.timeoutOverride(provider: .ollama, isSubstitute: false) == nil)
        #expect(Policy.timeoutOverride(provider: .localCLI, isSubstitute: false) == nil)
        #expect(Policy.timeoutOverride(provider: nil, isSubstitute: true) == nil)

        // D22: the ladder is the retry for a substitute.
        #expect(!Policy.retriesTimeout(isSubstitute: true, retryOnTimeoutEnabled: true))
        #expect(!Policy.retriesTimeout(isSubstitute: true, retryOnTimeoutEnabled: false))
        #expect(Policy.retriesTimeout(isSubstitute: false, retryOnTimeoutEnabled: true))
        #expect(!Policy.retriesTimeout(isSubstitute: false, retryOnTimeoutEnabled: false))
    }

    // MARK: - Substitution anchor

    @Test func replacingModelRemembersTheOriginalAnchorAcrossHops() {
        let configured = configuration(.gemini, "gemini-3.8-flash")
        #expect(!configured.isSubstitute)
        #expect(configured.modelRef == ref(.gemini, "gemini-3.8-flash"))

        let sibling = configured.replacingModel(provider: .gemini, modelName: "gemini-3.7-flash")
        #expect(sibling.isSubstitute)
        #expect(sibling.substitutedFrom == ref(.gemini, "gemini-3.8-flash"))
        #expect(sibling.modelRef == ref(.gemini, "gemini-3.7-flash"))

        let crossProvider = sibling.replacingModel(provider: .anthropic, modelName: "claude-haiku-4-5")
        #expect(crossProvider.substitutedFrom == ref(.gemini, "gemini-3.8-flash"))
        #expect(crossProvider.modelRef == ref(.anthropic, "claude-haiku-4-5"))

        let anchor = crossProvider.anchor
        #expect(!anchor.isSubstitute)
        #expect(anchor.provider == .gemini)
        #expect(anchor.modelName == "gemini-3.8-flash")

        // Aiming back at the anchor is the configured model, not a substitute.
        let back = crossProvider.replacingModel(provider: .gemini, modelName: "gemini-3.8-flash")
        #expect(!back.isSubstitute)
        #expect(back.modelRef == ref(.gemini, "gemini-3.8-flash"))

        let prompt = CustomPrompt(title: "Test", promptText: "Rewrite.")
        let reprompted = crossProvider.replacingPrompt(prompt)
        #expect(reprompted.substitutedFrom == ref(.gemini, "gemini-3.8-flash"))
        #expect(reprompted.prompt == prompt)
        #expect(reprompted.modelRef == ref(.anthropic, "claude-haiku-4-5"))

        // The ladder built from a substitute's anchor equals the one built from
        // the configured model: a walk never re-derives the order from where it landed.
        func ladder(from config: EnhancementRuntimeConfiguration) -> [EnhancementModelRef] {
            guard let anchorRef = config.anchor.modelRef else { return [] }
            return Policy.ladder(
                anchor: anchorRef,
                modeSelections: [],
                providerOrder: AIProvider.gemini.fallbackOrder,
                otherProviders: [ref(.anthropic, "claude-haiku-4-5")],
                includeLocal: false
            )
        }
        #expect(!ladder(from: configured).isEmpty)
        #expect(ladder(from: crossProvider) == ladder(from: configured))
    }

    // MARK: - Failure kinds

    @Test func failureKindClassifiesEveryErrorTheLadderSees() {
        #expect(Policy.failureKind(of: EnhancementError.rateLimitExceeded(detail: nil, retryAfter: nil)) == .quota)
        #expect(Policy.failureKind(of: EnhancementError.timeout) == .transient)
        #expect(Policy.failureKind(of: EnhancementError.serverError) == .transient)
        #expect(Policy.failureKind(of: EnhancementError.networkError) == .transient)
        #expect(Policy.failureKind(of: EnhancementError.notConfigured) == .provider)
        #expect(Policy.failureKind(of: EnhancementError.customError("HTTP 401: invalid key")) == .provider)
        #expect(Policy.failureKind(of: EnhancementError.customError("HTTP 403: forbidden")) == .provider)
        #expect(Policy.failureKind(of: EnhancementError.customError("HTTP 404: model not found")) == .rung)
        #expect(Policy.failureKind(of: EnhancementError.enhancementFailed) == .rung)
        #expect(Policy.failureKind(of: EnhancementError.invalidResponse) == .rung)
        // D23: a cancelled recording must never walk the ladder.
        #expect(Policy.failureKind(of: CancellationError()) == .cancelled)
        #expect(Policy.failureKind(of: NSError(domain: NSURLErrorDomain, code: NSURLErrorCancelled)) == .cancelled)
    }

    // MARK: - Descent

    @Test func descendChoosesTheNextRungByFailureKind() {
        let failed = ref(.gemini, "gemini-3.8-flash")
        let remaining = [
            ref(.gemini, "gemini-3.7-flash"),
            ref(.gemini, "gemini-3.6-flash"),
            ref(.anthropic, "claude-haiku-4-5"),
            ref(.ollama, "gemma4:26b"),
        ]
        let siblingsOnly = Array(remaining.prefix(2))

        // 1e: a sibling behind a down endpoint is a guaranteed second loss.
        #expect(Policy.descend(from: failed, kind: .transient, remaining: remaining, transientDescentUsed: false)
            == .init(next: ref(.anthropic, "claude-haiku-4-5"), remaining: [ref(.ollama, "gemma4:26b")]))
        #expect(Policy.descend(from: failed, kind: .transient, remaining: siblingsOnly, transientDescentUsed: false)
            == .init(next: ref(.gemini, "gemini-3.7-flash"), remaining: [ref(.gemini, "gemini-3.6-flash")]))
        #expect(Policy.descend(from: failed, kind: .transient, remaining: remaining, transientDescentUsed: true) == nil)

        #expect(Policy.descend(from: failed, kind: .provider, remaining: remaining, transientDescentUsed: false)
            == .init(next: ref(.anthropic, "claude-haiku-4-5"), remaining: [ref(.ollama, "gemma4:26b")]))
        #expect(Policy.descend(from: failed, kind: .provider, remaining: siblingsOnly, transientDescentUsed: false)
            == nil)

        #expect(Policy.descend(from: failed, kind: .cancelled, remaining: remaining, transientDescentUsed: false)
            == nil)

        let literalNext = Policy.Descent(next: remaining[0], remaining: Array(remaining.dropFirst()))
        #expect(Policy.descend(from: failed, kind: .quota, remaining: remaining, transientDescentUsed: true)
            == literalNext)
        #expect(Policy.descend(from: failed, kind: .rung, remaining: remaining, transientDescentUsed: false)
            == literalNext)
        #expect(Policy.descend(from: failed, kind: .quota, remaining: [], transientDescentUsed: false) == nil)
    }

    @Test func maximumDescentsCapsALongLadderAtSix() {
        #expect(Policy.maximumDescents(ladderCount: 0) == 0)
        #expect(Policy.maximumDescents(ladderCount: 3) == 3)
        #expect(Policy.maximumDescents(ladderCount: 14) == 6)
    }

    // MARK: - Error mapping

    @Test func mapUntypedErrorSendsOfflineToTheTransientBranchAndKeepsCancellation() {
        let offline = Policy.mapUntypedError(NSError(domain: NSURLErrorDomain, code: NSURLErrorNotConnectedToInternet))
        if case .networkError? = offline as? EnhancementError {} else {
            Issue.record("expected EnhancementError.networkError, got \(offline)")
        }
        // D11: an outage has to reach the transient descent to ever find the local rung.
        #expect(Policy.failureKind(of: offline) == .transient)

        #expect(Policy.mapUntypedError(NSError(domain: NSURLErrorDomain, code: NSURLErrorCancelled))
            is CancellationError)
        #expect(Policy.mapUntypedError(CancellationError()) is CancellationError)

        let other = Policy.mapUntypedError(
            NSError(domain: "LLMkit", code: 7, userInfo: [NSLocalizedDescriptionKey: "decoding failed"]))
        if case .customError(let message)? = other as? EnhancementError {
            #expect(message == "decoding failed")
        } else {
            Issue.record("expected EnhancementError.customError, got \(other)")
        }
    }
}
