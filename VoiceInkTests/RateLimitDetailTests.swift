import Foundation
import Testing
@testable import VoiceInk

struct RateLimitDetailTests {

    private func geminiBody(quotaIds: [String]) -> String {
        let violations = quotaIds.map { #"{"quotaId":"\#($0)"}"# }.joined(separator: ",")
        return #"""
        {"error":{"code":429,"message":"You exceeded your current quota, please check your plan and billing \#
        details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits.\#
        \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, \#
        limit: 20, model: gemini-3.8-flash\nPlease retry in 31.43s.","status":"RESOURCE_EXHAUSTED","details":[\#
        {"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[\#(violations)]},\#
        {"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"31s"}]}}
        """#
    }

    @Test func geminiBodyReportsThePeriodFromDetailsEvenWhenTheQuotaClauseWins() throws {
        let parsed = RateLimitDetail.parse(geminiBody(quotaIds: ["GenerateRequestsPerDayPerProjectPerModel-FreeTier"]))

        // D9b: the summary is the readable quota clause, which never names the period.
        let summary = try #require(parsed.summary)
        #expect(summary.contains("gemini-3.8-flash"))
        #expect(summary.contains("limit of 20"))
        #expect(!summary.contains("PerDay"))
        #expect(parsed.quotaPeriod == .perDay)
        let retryAfter = try #require(parsed.retryAfter)
        #expect(abs(retryAfter - 31.43) < 0.001)

        let perMinute = RateLimitDetail.parse(
            geminiBody(quotaIds: ["GenerateRequestsPerMinutePerProjectPerModel-FreeTier"]))
        #expect(perMinute.quotaPeriod == .perMinute)

        // One refusal can breach both buckets; the model is out for the day.
        let both = RateLimitDetail.parse(geminiBody(quotaIds: [
            "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
            "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
        ]))
        #expect(both.quotaPeriod == .perDay)
    }

    @Test func openAIStyleTryAgainParsesSecondsMillisecondsAndMinutes() throws {
        let body = #"""
        {"error":{"message":"Rate limit reached for gpt-5.4-mini in organization org-test on requests per min \#
        (RPM): Limit 3, Used 3, Requested 1. Please try again in 20s. Visit \#
        https://platform.openai.com/account/rate-limits to learn more.","type":"requests","code":"rate_limit_exceeded"}}
        """#
        let parsed = RateLimitDetail.parse(body)
        // D18: without this form every non-Gemini refusal fell back to the 60s floor.
        #expect(parsed.retryAfter == 20)
        #expect(parsed.quotaPeriod == .perMinute)
        #expect(parsed.summary != nil)

        let milliseconds = try #require(RateLimitDetail.parse("Please try again in 820ms.").retryAfter)
        #expect(abs(milliseconds - 0.82) < 0.001)
        let minutes = try #require(RateLimitDetail.parse("Please try again in 1m26.4s.").retryAfter)
        #expect(abs(minutes - 86.4) < 0.001)
    }

    @Test func emptyOrUnstructuredBodies() {
        for body in ["", "  \n "] {
            let parsed = RateLimitDetail.parse(body)
            #expect(parsed.summary == nil)
            #expect(parsed.retryAfter == nil)
            #expect(parsed.quotaPeriod == .unknown)
        }

        let plain = RateLimitDetail.parse("Too Many Requests")
        #expect(plain.summary == "Too Many Requests")
        #expect(plain.retryAfter == nil)
        #expect(plain.quotaPeriod == .unknown)
    }
}
