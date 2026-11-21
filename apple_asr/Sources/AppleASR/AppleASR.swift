// apple-asr -- stream raw PCM through Apple's on-device SpeechTranscriber.
//
// Protocol (consumed by livecaption/asr_apple.py):
//   stdin : raw little-endian float32 mono PCM @ 16 kHz, streamed; EOF = end of audio.
//   stdout: one JSON object per line
//     {"type":"ready","locale":"zh-CN"}                       analyzer started, send audio
//     {"type":"partial","text":"…","start":1.23,"end":2.34}   volatile (in-progress) result
//     {"type":"final","text":"…","start":1.23,"end":2.34}     finalized result
//     {"type":"done"}                                          all input processed
//     {"type":"error","message":"…"}                           fatal; process exits 1
//   stderr: human-readable progress (asset download etc.)
//   start/end are seconds on the input timeline (sample 0 of stdin = 0.0 s).
//
// Flags:
//   --locale <id>    e.g. en-US, zh-CN (default en-US)
//   --fast           SpeechTranscriber .fastResults (lower latency, slightly less accurate)
//   --check          resolve locale + install assets, print ready, exit (no audio)
//   --list-locales   print {"supported":[…],"installed":[…]} and exit

import AVFoundation
import CoreMedia
import Foundation
import Speech

let inputSampleRate = 16_000.0

struct BridgeError: Error, CustomStringConvertible {
    let description: String
    init(_ d: String) { description = d }
}

// ---- stdout JSON lines (serialized; results task and main task both write) ----
let outLock = NSLock()

func emit(_ obj: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: obj, options: []) else { return }
    outLock.lock()
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0A]))
    outLock.unlock()
}

func logErr(_ s: String) {
    FileHandle.standardError.write(Data((s + "\n").utf8))
}

func seconds(_ t: CMTime) -> Double {
    let s = CMTimeGetSeconds(t)
    return s.isFinite ? (s * 1000).rounded() / 1000 : 0
}

// ---- args ----
struct Options {
    var locale = "en-US"
    var fast = false
    var check = false
    var listLocales = false
}

func parseArgs() throws -> Options {
    var o = Options()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let a = it.next() {
        switch a {
        case "--locale":
            guard let v = it.next() else { throw BridgeError("--locale needs a value") }
            o.locale = v
        case "--fast": o.fast = true
        case "--check": o.check = true
        case "--list-locales": o.listLocales = true
        default: throw BridgeError("unknown argument: \(a)")
        }
    }
    return o
}

// ---- stdin reader: blocking reads on a plain thread, converted into AnalyzerInput ----
// (AnalyzerInputConverter would do this, but it is macOS 27+; AVAudioConverter keeps 26 working.)
final class StdinPump: @unchecked Sendable {
    let continuation: AsyncStream<AnalyzerInput>.Continuation
    let srcFormat: AVAudioFormat
    let dstFormat: AVAudioFormat
    let converter: AVAudioConverter?  // nil when the analyzer already takes 16 kHz float32 mono

    init(continuation: AsyncStream<AnalyzerInput>.Continuation, analyzerFormat: AVAudioFormat) throws {
        self.continuation = continuation
        guard let f = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: inputSampleRate, channels: 1,
            interleaved: false
        ) else { throw BridgeError("cannot create 16 kHz float32 format") }
        self.srcFormat = f
        self.dstFormat = analyzerFormat
        if analyzerFormat == f {
            self.converter = nil
        } else {
            guard let c = AVAudioConverter(from: f, to: analyzerFormat) else {
                throw BridgeError("cannot convert 16 kHz float32 to \(analyzerFormat)")
            }
            self.converter = c
        }
    }

    /// Convert one 16 kHz buffer into the analyzer format (resampling keeps converter state).
    func convert(_ buf: AVAudioPCMBuffer) throws -> AVAudioPCMBuffer? {
        guard let converter else { return buf }
        let ratio = dstFormat.sampleRate / srcFormat.sampleRate
        let cap = AVAudioFrameCount(Double(buf.frameLength) * ratio) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: dstFormat, frameCapacity: cap) else {
            return nil
        }
        var consumed = false
        var err: NSError?
        let status = converter.convert(to: out, error: &err) { _, inputStatus in
            if consumed {
                inputStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            inputStatus.pointee = .haveData
            return buf
        }
        if status == .error { throw err ?? BridgeError("audio conversion failed") }
        return out.frameLength > 0 ? out : nil
    }

    /// Read stdin until EOF, feeding the analyzer. Runs on its own thread.
    func run() {
        let stdin = FileHandle.standardInput
        var leftover = Data()
        let chunkBytes = Int(inputSampleRate * 0.1) * 4  // ~100 ms
        while true {
            let data = stdin.readData(ofLength: chunkBytes)
            if data.isEmpty { break }  // EOF
            leftover.append(data)
            let usable = leftover.count - leftover.count % 4
            if usable == 0 { continue }
            let frames = usable / 4
            guard let buf = AVAudioPCMBuffer(
                pcmFormat: srcFormat, frameCapacity: AVAudioFrameCount(frames)
            ) else { continue }
            buf.frameLength = AVAudioFrameCount(frames)
            leftover.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: Float.self)
                buf.floatChannelData![0].update(from: src.baseAddress!, count: frames)
            }
            leftover.removeFirst(usable)
            do {
                if let out = try convert(buf) { continuation.yield(AnalyzerInput(buffer: out)) }
            } catch {
                logErr("apple-asr: audio conversion failed: \(error)")
            }
        }
        continuation.finish()
    }
}

// ---- main ----
func makeTranscriber(_ locale: Locale, fast: Bool) -> SpeechTranscriber {
    var reporting: Set<SpeechTranscriber.ReportingOption> = [.volatileResults]
    if fast { reporting.insert(.fastResults) }
    return SpeechTranscriber(
        locale: locale, transcriptionOptions: [], reportingOptions: reporting,
        attributeOptions: [.audioTimeRange]
    )
}

func run(_ opt: Options) async throws {
    guard SpeechTranscriber.isAvailable else {
        throw BridgeError("SpeechTranscriber is not available on this Mac (needs macOS 26+, Apple Silicon)")
    }

    if opt.listLocales {
        let sup = await SpeechTranscriber.supportedLocales.map { $0.identifier(.bcp47) }.sorted()
        let inst = await SpeechTranscriber.installedLocales.map { $0.identifier(.bcp47) }.sorted()
        emit(["supported": sup, "installed": inst])
        return
    }

    guard let locale = await SpeechTranscriber.supportedLocale(
        equivalentTo: Locale(identifier: opt.locale)
    ) else {
        let sup = await SpeechTranscriber.supportedLocales.map { $0.identifier(.bcp47) }.sorted()
        throw BridgeError(
            "locale '\(opt.locale)' is not supported by SpeechTranscriber. Supported: "
                + sup.joined(separator: ", "))
    }
    let localeId = locale.identifier(.bcp47)
    let transcriber = makeTranscriber(locale, fast: opt.fast)

    // Make sure the on-device model for this locale is present (downloads on first use).
    if let req = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
        logErr("apple-asr: downloading speech model for \(localeId) …")
        try await req.downloadAndInstall()
        logErr("apple-asr: speech model for \(localeId) installed")
    }

    if opt.check {
        emit(["type": "ready", "locale": localeId])
        return
    }

    let analyzer = SpeechAnalyzer(
        modules: [transcriber],
        options: .init(priority: .userInitiated, modelRetention: .processLifetime)
    )
    guard let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber])
    else { throw BridgeError("no audio format compatible with SpeechTranscriber") }
    try await analyzer.prepareToAnalyze(in: format)

    let resultsTask = Task {
        for try await r in transcriber.results {
            let text = String(r.text.characters)
            // Prefer the span of the actually-transcribed words (audioTimeRange attributes):
            // a volatile result's range starts where the previous final ended, so it would
            // include the leading silence.
            var start = r.range.start
            var end = r.range.end
            var first = true
            for run in r.text.runs {
                guard let tr = run.audioTimeRange else { continue }
                let s = String(r.text[run.range].characters)
                if s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { continue }
                if first { start = tr.start; first = false }
                end = tr.end
            }
            emit([
                "type": r.isFinal ? "final" : "partial",
                "text": text,
                "start": seconds(start),
                "end": seconds(end),
            ])
        }
    }

    let (stream, continuation) = AsyncStream.makeStream(of: AnalyzerInput.self)
    let pump = try StdinPump(continuation: continuation, analyzerFormat: format)
    try await analyzer.start(inputSequence: stream)
    emit(["type": "ready", "locale": localeId])

    // Pump stdin on a dedicated thread (blocking reads) and wait for EOF.
    await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
        let t = Thread {
            pump.run()
            done.resume()
        }
        t.name = "apple-asr-stdin"
        t.start()
    }

    try await analyzer.finalizeAndFinishThroughEndOfInput()
    try await resultsTask.value
    emit(["type": "done"])
}

@main
struct AppleASR {
    static func main() async {
        do {
            try await run(try parseArgs())
            exit(0)
        } catch {
            emit(["type": "error", "message": "\(error)"])
            logErr("apple-asr: \(error)")
            exit(1)
        }
    }
}
