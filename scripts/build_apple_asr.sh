#!/usr/bin/env bash
# Build apple-asr (Apple on-device SpeechAnalyzer/SpeechTranscriber bridge) into ./bin/apple-asr.
# Only needed for --asr-model apple. Requires macOS 26+ and a Swift 6 toolchain
# (Xcode or Command Line Tools).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="$HERE/apple_asr"
BIN_DIR="$HERE/bin"

mkdir -p "$BIN_DIR"
echo "==> swift build -c release ($PKG)"
cd "$PKG"
swift build -c release

BIN="$(swift build -c release --show-bin-path)/apple-asr"
cp "$BIN" "$BIN_DIR/apple-asr"
echo "==> Done: $BIN_DIR/apple-asr"
