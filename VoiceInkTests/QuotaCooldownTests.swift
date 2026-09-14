import Foundation
import Testing
@testable import VoiceInk

/// Cooldown arithmetic. Every `now` is a fixed instant built in UTC, so neither
/// the wall clock nor the machine's time zone can move a result.
struct QuotaCooldownTests {

    private typealias Policy = EnhancementLadderPolicy
    private typealias Cooldown = EnhancementLadderPolicy.QuotaCooldown

    private func utc(_ year: Int, _ month: Int, _ day: Int, _ hour: Int, _ minute: Int, _ second: Int = 0) -> Date {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        return calendar.date(
            from: DateComponents(year: year, month: month, day: day, hour: hour, minute: minute, second: second)
        )!
    }

    private let now = Date(timeIntervalSince1970: 1_789_000_000)

    // MARK: - Per day

    @Test func perDayCooldownRunsToThePacificMidnightResetPlusAMinute() {
        let cases: [(now: Date, until: Date, note: String)] = [
            (utc(2026, 9, 14, 6, 59), utc(2026, 9, 14, 7, 1), "a minute before midnight PDT"),
            (utc(2026, 9, 14, 7, 0, 30), utc(2026, 9, 14, 7, 1), "inside the margin after midnight PDT"),
            (utc(2026, 9, 14, 7, 5), utc(2026, 9, 15, 7, 1), "five minutes after midnight PDT"),
            (utc(2026, 11, 1, 6, 30), utc(2026, 11, 1, 7, 1), "the night before PDT ends"),
            (utc(2026, 11, 1, 19, 0), utc(2026, 11, 2, 8, 1), "the 25-hour day PDT ends"),
            (utc(2026, 3, 8, 7, 30), utc(2026, 3, 8, 8, 1), "the night before PDT starts"),
            (utc(2026, 3, 8, 19, 0), utc(2026, 3, 9, 7, 1), "the 23-hour day PDT starts"),
        ]
        for entry in cases {
            #expect(Policy.nextDailyQuotaReset(after: entry.now) == entry.until, "\(entry.note)")

            // Gemini sends a seconds-scale retryDelay even on a daily cap; it must not shorten it,
            // and an escalated previous must not lengthen it.
            let cooldown = Policy.cooldown(
                forRefusal: .perDay,
                retryAfter: 31,
                previous: Cooldown(until: entry.now, duration: 14400),
                now: entry.now
            )
            #expect(cooldown.until == entry.until, "\(entry.note)")
            #expect(cooldown.duration == entry.until.timeIntervalSince(entry.now), "\(entry.note)")
        }
    }

    // MARK: - Per minute

    @Test func perMinuteCooldownNeverDoublesAndHonoursRetryAfter() {
        let escalated = Cooldown(until: now.addingTimeInterval(-5), duration: 3600)

        let plain = Policy.cooldown(forRefusal: .perMinute, retryAfter: nil, previous: escalated, now: now)
        #expect(plain.duration == 60)
        #expect(plain.until == now.addingTimeInterval(60))

        #expect(Policy.cooldown(forRefusal: .perMinute, retryAfter: 90, previous: escalated, now: now).duration == 90)
        #expect(Policy.cooldown(forRefusal: .perMinute, retryAfter: 60, previous: nil, now: now).duration == 60)
        #expect(Policy.cooldown(forRefusal: .perMinute, retryAfter: 10, previous: nil, now: now).duration == 60)
        #expect(Policy.cooldown(forRefusal: .perMinute, retryAfter: 20_000, previous: nil, now: now).duration == 14400)
    }

    // MARK: - Unknown period

    @Test func unknownPeriodCooldownDoublesHonoursRetryAfterAndClamps() {
        func duration(retryAfter: TimeInterval?, previous: Cooldown?) -> TimeInterval {
            Policy.cooldown(forRefusal: .unknown, retryAfter: retryAfter, previous: previous, now: now).duration
        }
        let recent = Cooldown(until: now.addingTimeInterval(-10), duration: 600)

        #expect(duration(retryAfter: nil, previous: nil) == 60)
        #expect(duration(retryAfter: 300, previous: nil) == 300)
        #expect(duration(retryAfter: 5, previous: nil) == 60)
        #expect(duration(retryAfter: nil, previous: recent) == 1200)
        #expect(duration(retryAfter: 500, previous: recent) == 1200)
        // D14: the provider's wait is honoured on the second refusal too.
        #expect(duration(retryAfter: 3000, previous: recent) == 3000)

        #expect(duration(retryAfter: nil, previous: Cooldown(until: now, duration: 10_000)) == 14400)
        #expect(duration(retryAfter: 99_999, previous: nil) == 14400)

        let dayOld = Cooldown(until: now.addingTimeInterval(-24 * 3600 - 1), duration: 600)
        let justUnderADay = Cooldown(until: now.addingTimeInterval(-24 * 3600 + 1), duration: 600)
        #expect(duration(retryAfter: nil, previous: dayOld) == 60)
        #expect(duration(retryAfter: nil, previous: justUnderADay) == 1200)

        let cooldown = Policy.cooldown(forRefusal: .unknown, retryAfter: nil, previous: recent, now: now)
        #expect(cooldown.until == now.addingTimeInterval(1200))
    }

    // MARK: - Restore

    @Test func restoredCooldownClampsAWrongClockAndForgetsOnlyAfterADay() {
        // D8: a cooling model is never asked, so a far-future `until` would bench it indefinitely.
        let farFuture = Policy.restoredCooldown(until: now.addingTimeInterval(90 * 24 * 3600), duration: 3600, now: now)
        #expect(farFuture == Cooldown(until: now.addingTimeInterval(25 * 3600), duration: 3600))

        let active = Policy.restoredCooldown(until: now.addingTimeInterval(7200), duration: nil, now: now)
        #expect(active == Cooldown(until: now.addingTimeInterval(7200), duration: 60))

        #expect(Policy.restoredCooldown(until: now.addingTimeInterval(-25 * 3600), duration: 1800, now: now) == nil)

        // D19: an expired cooldown keeps its duration, so the next refusal escalates from it.
        let expired = Policy.restoredCooldown(until: now.addingTimeInterval(-3600), duration: 1800, now: now)
        #expect(expired == Cooldown(until: now.addingTimeInterval(-3600), duration: 1800))
        #expect(expired?.isActive(at: now) == false)
        let next = Policy.cooldown(forRefusal: .unknown, retryAfter: nil, previous: expired, now: now)
        #expect(next.duration == 3600)
    }
}
