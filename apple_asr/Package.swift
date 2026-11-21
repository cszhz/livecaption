// swift-tools-version: 5.10
// apple-asr: a tiny stdin->stdout bridge to Apple's on-device SpeechAnalyzer/SpeechTranscriber
// (macOS 26+). Built into ./bin/apple-asr by scripts/build_apple_asr.sh; driven by
// livecaption/asr_apple.py.

import PackageDescription

let package = Package(
    name: "apple-asr",
    platforms: [.macOS("26.0")],
    targets: [
        .executableTarget(name: "apple-asr", path: "Sources/AppleASR")
    ]
)
