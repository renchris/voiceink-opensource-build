import Foundation
import os

/// Crash-safe, append-as-you-go journal for a streaming transcription session.
///
/// When "Save Monologues to File" is enabled, every stable (committed) segment emitted during a
/// streaming session is appended to a per-session Markdown file and `fsync`'d to disk immediately.
/// If the app is killed mid-session, everything committed up to that instant is already persisted —
/// only the last few not-yet-confirmed words are lost (those are still in the always-saved WAV).
///
/// Only the incremental LocalAgreement path (`FluidAudioStreamingProvider` + `WordAgreementEngine`,
/// i.e. the Parakeet TDT v2/v3 models) commits segments continuously, so that path gets true
/// incremental crash-safety. Commit-at-stop providers still produce a correct file — just written
/// in one shot at the end.
///
/// All file I/O runs on a private serial queue so it never blocks the main actor.
final class StreamingTranscriptJournal: @unchecked Sendable {
    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "MonologueJournal")
    private let queue = DispatchQueue(label: "com.prakashjoshipax.voiceink.monologueJournal", qos: .utility)
    private let fileURL: URL
    private let startedAt: Date
    private var handle: FileHandle?
    private var opened = false

    /// Directory monologue files are written to: ~/Documents/VoiceInk Monologues
    static var directory: URL {
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Documents")
        return base.appendingPathComponent("VoiceInk Monologues", isDirectory: true)
    }

    /// Creates a journal for one session. `startedAt` names the file and stamps its header.
    init(startedAt: Date) {
        self.startedAt = startedAt
        let stamp = Self.fileStampFormatter.string(from: startedAt)
        self.fileURL = Self.directory.appendingPathComponent("\(stamp).md")
    }

    /// Append a committed (stable) segment and flush it to disk.
    func append(_ segment: String) {
        let text = segment.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        queue.async { [weak self] in
            self?.write(text + " ")
        }
    }

    /// Write a trailing newline, flush, and close the file. Safe to call more than once.
    func finish() {
        queue.async { [weak self] in
            guard let self, let handle = self.handle else { return }
            self.writeRaw("\n")
            try? handle.synchronizeFile()
            try? handle.close()
            self.handle = nil
        }
    }

    // MARK: - Private (all executed on `queue`)

    private func write(_ string: String) {
        if !opened { openFile() }
        writeRaw(string)
        try? handle?.synchronizeFile()  // fsync: survive an app crash / power loss
    }

    private func openFile() {
        opened = true
        do {
            try FileManager.default.createDirectory(at: Self.directory, withIntermediateDirectories: true)
            if !FileManager.default.fileExists(atPath: fileURL.path) {
                FileManager.default.createFile(atPath: fileURL.path, contents: nil)
            }
            let h = try FileHandle(forWritingTo: fileURL)
            h.seekToEndOfFile()
            self.handle = h
            writeRaw("# Monologue — \(Self.headerFormatter.string(from: startedAt))\n\n")
        } catch {
            logger.error("Failed to open monologue journal at \(self.fileURL.path, privacy: .public): \(error, privacy: .public)")
        }
    }

    private func writeRaw(_ string: String) {
        guard let handle, let data = string.data(using: .utf8) else { return }
        do {
            try handle.write(contentsOf: data)
        } catch {
            logger.error("Monologue journal write failed: \(error, privacy: .public)")
        }
    }

    private static let fileStampFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        return f
    }()

    private static let headerFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        return f
    }()
}
