import Foundation

/// The fallback ladder's decisions as pure functions over plain values. The
/// service owns the state (cooldowns, keys, the user's modes) and asks these for
/// every choice, so each choice can be tested without booting the service —
/// whose `init` reads, and may rewrite, the live app's preferences.
enum EnhancementLadderPolicy {

    // MARK: - Ladder order

    /// Where enhancement goes when the anchor cannot answer, in order:
    ///   1. models of the anchor's provider the user already chose in another mode
    ///   2. that provider's remaining models, in its fallback order
    ///   3. other enhancement providers they have connected, at their selected model
    ///   4. a local provider, last and only when armed
    /// Always built from the ANCHOR — the mode's configured model — never from a
    /// substitute, so a walk that crossed providers does not re-derive the order
    /// from wherever it landed. The anchor and `excluded` never appear.
    static func ladder(
        anchor: EnhancementModelRef,
        modeSelections: [String],
        providerOrder: [String],
        otherProviders: [EnhancementModelRef],
        includeLocal: Bool,
        excluding excluded: [EnhancementModelRef] = []
    ) -> [EnhancementModelRef] {
        var seen: Set<EnhancementModelRef> = [anchor]
        seen.formUnion(excluded)
        var ladder: [EnhancementModelRef] = []

        func append(_ rung: EnhancementModelRef) {
            guard !rung.modelName.isEmpty, seen.insert(rung).inserted else { return }
            ladder.append(rung)
        }

        // 1. Their own choices elsewhere come before our list order.
        for modelName in modeSelections {
            append(EnhancementModelRef(provider: anchor.provider, modelName: modelName))
        }

        // 2. The rest of this provider's models.
        for modelName in providerOrder {
            append(EnhancementModelRef(provider: anchor.provider, modelName: modelName))
        }

        // 3. Other providers they have actually connected.
        for rung in otherProviders where rung.provider != anchor.provider && !isLocal(rung.provider) {
            append(rung)
        }

        // 4. Local, last, and only when armed.
        if includeLocal {
            for rung in otherProviders where rung.provider != anchor.provider && isLocal(rung.provider) {
                append(rung)
            }
        }

        return ladder
    }

    static func isLocal(_ provider: AIProvider?) -> Bool {
        provider == .ollama || provider == .localCLI
    }

    /// Bounds the worst case for a single dictation. Cooldowns persist between
    /// dictations, so a longer ladder is still walked in full — just across
    /// several recordings instead of making one of them wait for all of it.
    static let maximumDescentsPerRequest = 6

    static func maximumDescents(ladderCount: Int) -> Int {
        min(ladderCount, maximumDescentsPerRequest)
    }

    // MARK: - Truncation guard

    /// LLMkit's OpenAI-compatible client sends no completion-token cap and never
    /// reads `finish_reason`, so a long rewrite comes back cut mid-sentence and is
    /// indistinguishable from a complete one — measured at exactly 2,048 tokens on
    /// a real 11,415-character transcript. Pasting a silent truncation is the one
    /// failure a fallback must never introduce, so keep long transcripts off those
    /// rungs and let the ladder carry them somewhere that reports completion.
    static let openAICompatibleFallbackCharacterLimit = 6000

    static func usesOpenAICompatibleClient(_ provider: AIProvider) -> Bool {
        switch provider {
        case .gemini, .anthropic, .ollama, .localCLI:
            return false
        default:
            return true
        }
    }

    static func rungIsSafe(_ rung: EnhancementModelRef, textLength: Int) -> Bool {
        textLength <= openAICompatibleFallbackCharacterLimit || !usesOpenAICompatibleClient(rung.provider)
    }

    // MARK: - Timeouts

    /// A local rung gets a tighter budget than the user's enhancement timeout: it
    /// is a fallback and may not become the new stall. 12s, not 8s — the measured
    /// warm p90 for the best local model is 5.67s but its max over 132 calls on
    /// real transcripts is 8.94s, so an 8s budget fails exactly the longest
    /// dictations, which are the ones most expensive to redo.
    static let localFallbackTimeout: TimeInterval = 12

    /// Keyed off the configuration itself, not off whether THIS call descended: a
    /// substitute chosen before the first request is just as much a fallback as
    /// one reached mid-walk, and must not run on the cloud budget.
    static func timeoutOverride(provider: AIProvider?, isSubstitute: Bool) -> TimeInterval? {
        isSubstitute && isLocal(provider) ? localFallbackTimeout : nil
    }

    /// The ladder IS the retry for a substitute. Re-asking a rung that just timed
    /// out spends its whole budget again — three times over on the local rung —
    /// before the walk may move on.
    static func retriesTimeout(isSubstitute: Bool, retryOnTimeoutEnabled: Bool) -> Bool {
        retryOnTimeoutEnabled && !isSubstitute
    }

    // MARK: - Descent

    /// What a failed attempt says about the rungs still ahead of it.
    enum FailureKind: Equatable {
        /// Out of quota: cool this model and take the next rung.
        case quota
        /// Timeout, 5xx or unreachable: the provider is down or slow, and its
        /// sibling models sit behind the same endpoint.
        case transient
        /// Missing or rejected credentials: every model of this provider fails
        /// identically, so none of them is worth asking.
        case provider
        /// This model alone refused — decommissioned, or an unsupported parameter.
        case rung
        /// The caller gave up. Never descend: the next rung would answer nobody.
        case cancelled
    }

    static func failureKind(of error: Error) -> FailureKind {
        if isCancellation(error) { return .cancelled }
        guard let error = error as? EnhancementError else { return .rung }
        switch error {
        case .rateLimitExceeded:
            return .quota
        case .timeout, .serverError, .networkError:
            return .transient
        case .notConfigured:
            return .provider
        case .customError(let message):
            // `mapLLMKitError` formats every unmapped HTTP status this way.
            return message.hasPrefix("HTTP 401") || message.hasPrefix("HTTP 403") ? .provider : .rung
        case .enhancementFailed, .invalidResponse:
            return .rung
        }
    }

    struct Descent: Equatable {
        let next: EnhancementModelRef
        let remaining: [EnhancementModelRef]
    }

    /// The next rung after `failed`, or nil when the walk should stop and surface
    /// the error. A transient failure descends only once per request, because
    /// each one costs its full timeout before it fails; it goes to a different
    /// provider when one is left, since a sibling model behind a down endpoint is
    /// a guaranteed second loss.
    static func descend(
        from failed: EnhancementModelRef,
        kind: FailureKind,
        remaining: [EnhancementModelRef],
        transientDescentUsed: Bool
    ) -> Descent? {
        var candidates = remaining
        switch kind {
        case .cancelled:
            return nil
        case .quota, .rung:
            break
        case .provider:
            candidates.removeAll { $0.provider == failed.provider }
        case .transient:
            guard !transientDescentUsed else { return nil }
            if candidates.contains(where: { $0.provider != failed.provider }) {
                candidates.removeAll { $0.provider == failed.provider }
            }
        }
        guard !candidates.isEmpty else { return nil }
        let next = candidates.removeFirst()
        return Descent(next: next, remaining: candidates)
    }

    // MARK: - Quota cooldowns

    /// Which quota bucket a refusal came from. It decides how long the model sits
    /// out: a daily cap is gone until the provider's reset, a per-minute cap is
    /// back within a minute, and one backoff rule for both is wrong in opposite
    /// directions — hours of re-probing a dead model, or benching a live one.
    enum QuotaPeriod: Equatable {
        case perDay
        case perMinute
        case unknown
    }

    struct QuotaCooldown: Equatable {
        let until: Date
        /// What this cooldown lasted; the next refusal of unknown period doubles it.
        let duration: TimeInterval

        func isActive(at now: Date) -> Bool {
            until > now
        }

        /// The duration outlives the cooldown so a model that refuses again soon
        /// after it expired backs off further instead of starting over at 60s —
        /// but not forever: a day after it ended, the refusal is history.
        func isRetained(at now: Date) -> Bool {
            now < until.addingTimeInterval(quotaEscalationMemory)
        }
    }

    static let minimumQuotaCooldown: TimeInterval = 60
    /// Four hours: the ceiling only governs how often a model whose limit period
    /// the provider did not name is re-probed. Each probe costs one refusal.
    static let maximumQuotaCooldown: TimeInterval = 14400
    static let quotaEscalationMemory: TimeInterval = 24 * 3600
    /// Gemini resets daily quotas at midnight Pacific. The margin keeps the first
    /// probe after it from landing on a server whose clock is a few seconds
    /// behind ours, which would spend a refusal and cool the model for a day.
    static let dailyQuotaResetMargin: TimeInterval = 60
    /// A daily cooldown lasts at most a day and a minute, so anything longer read
    /// back from disk was written under a clock that has since been corrected.
    static let maximumRestoredCooldown: TimeInterval = 25 * 3600

    /// The cooldown a fresh refusal opens. `previous` is the model's last cooldown
    /// (active or not); a caller must not open a second one while it is active,
    /// or a burst of concurrent refusals would double it once per request.
    static func cooldown(
        forRefusal period: QuotaPeriod,
        retryAfter: TimeInterval?,
        previous: QuotaCooldown?,
        now: Date
    ) -> QuotaCooldown {
        let duration: TimeInterval
        switch period {
        case .perDay:
            // Nothing earlier can succeed, whatever retryAfter says: Gemini sends
            // a seconds-scale retryDelay even on a daily cap.
            let until = nextDailyQuotaReset(after: now)
            return QuotaCooldown(until: until, duration: until.timeIntervalSince(now))
        case .perMinute:
            // Never doubled: the bucket refills on its own schedule, and a second
            // refusal says the minute is still busy, not that it got longer.
            duration = max(retryAfter ?? minimumQuotaCooldown, minimumQuotaCooldown)
        case .unknown:
            // The provider's own wait is honoured on every refusal, not only the
            // first; doubling is only the floor under it.
            let escalation = previous.flatMap { $0.isRetained(at: now) ? $0.duration * 2 : nil }
            duration = max(retryAfter ?? 0, escalation ?? minimumQuotaCooldown)
        }
        let clamped = min(max(duration, minimumQuotaCooldown), maximumQuotaCooldown)
        return QuotaCooldown(until: now.addingTimeInterval(clamped), duration: clamped)
    }

    /// The next midnight in America/Los_Angeles, plus the margin. Measured from
    /// `now - margin` so a refusal inside the margin — written against the
    /// previous day's count — waits for this reset, not the next one.
    static func nextDailyQuotaReset(after now: Date) -> Date {
        let fallback = now.addingTimeInterval(24 * 3600)
        guard let pacific = TimeZone(identifier: "America/Los_Angeles") else { return fallback }
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = pacific
        guard
            let midnight = calendar.nextDate(
                after: now.addingTimeInterval(-dailyQuotaResetMargin),
                matching: DateComponents(hour: 0, minute: 0, second: 0),
                matchingPolicy: .nextTime
            )
        else { return fallback }
        return midnight.addingTimeInterval(dailyQuotaResetMargin)
    }

    /// A cooldown read back from disk, or nil when it should be forgotten. The
    /// clamp matters because nothing else can end a cooldown early: a cooling
    /// model is dropped from the ladder, so it is never asked, never succeeds,
    /// and a wrong `until` would bench it indefinitely.
    static func restoredCooldown(until: Date, duration: TimeInterval?, now: Date) -> QuotaCooldown? {
        let cooldown = QuotaCooldown(
            until: min(until, now.addingTimeInterval(maximumRestoredCooldown)),
            duration: duration ?? minimumQuotaCooldown
        )
        return cooldown.isRetained(at: now) ? cooldown : nil
    }

    // MARK: - Error mapping

    static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        let nsError = error as NSError
        return nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorCancelled
    }

    /// For errors that reach `makeRequest` untyped. LLMkit rethrows URLSession's
    /// own errors raw (only its timeout is typed), so an offline request arrives
    /// as a bare `NSURLErrorDomain` error; left as `.customError` it would never
    /// reach the transient branch, and an outage would never find the local rung.
    static func mapUntypedError(_ error: Error) -> Error {
        if isCancellation(error) { return CancellationError() }
        if (error as NSError).domain == NSURLErrorDomain { return EnhancementError.networkError }
        return EnhancementError.customError(error.localizedDescription)
    }
}
