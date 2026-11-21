"""Alternate ASR backend: Apple's on-device SpeechAnalyzer / SpeechTranscriber (macOS 26+).

The speech model is a system asset (managed by macOS, shared with Dictation; downloaded on
first use of a locale), not an HF repo, and runs outside MLX -- so it doesn't compete with the
translation model for the MLX lock / GPU. The Speech framework is Swift-only, so we drive it
through a tiny bridge binary (apple_asr/ -> ./bin/apple-asr, built by
scripts/build_apple_asr.sh), one subprocess per audio stream:

    stdin : raw float32 LE mono PCM @ 16 kHz (exactly the samples AsrWorker hands us)
    stdout: JSON lines -- ready / partial / final / done / error (see the Swift source)

Endpointing is SpeechTranscriber's own (no Silero VAD here): each *final* result becomes one
utterance, the latest *volatile* result is the partial. Result time ranges are on the stdin
timeline, so they map back to sample offsets exactly.

Speaker diarization (`--diarize`) is utterance-level, like qwen3's fallback: the final's audio
slice is fed to Sortformer and labeled with its majority speaker. There's no two-pass
re-decode (SpeechTranscriber's final already is its best pass), so diff is always None.
"""

from __future__ import annotations

import json
import queue
import re
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable

import numpy as np

from . import config
from .models import resolve_apple_asr
from .runtime import MLX_LOCK as _MLX_LOCK

_SR = config.SAMPLE_RATE
_READY_TIMEOUT_SEC = 120.0  # first use of a locale may download the model
_DONE_TIMEOUT_SEC = 60.0
# Audio kept for diarization slicing. A final rarely spans more than ~30 s; keep headroom.
_AUDIO_KEEP_SEC = 120.0

# SpeechTranscriber's Chinese output puts a space before full-width punctuation ("大家好 ，")
# and between CJK runs; drop whitespace that touches a CJK char / full-width punctuation on
# either side, keeping spaces between Latin words ("Apple 的 API" -> "Apple的API" is fine for
# captions and matches what the other engines emit).
_CJK = r"\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef"
_SPACE_NEAR_CJK = re.compile(rf"(?<=[{_CJK}])\s+|\s+(?=[{_CJK}])")


def _clean_text(text: str) -> str:
    return _SPACE_NEAR_CJK.sub("", text).strip()

# Common tags / names -> BCP-47 locale for SpeechTranscriber. Anything else with a region
# (xx-yy) is normalized to xx-YY and validated by the bridge, which lists supported locales.
_APPLE_LANG_ALIASES = {
    "en": "en-US", "english": "en-US",
    "zh": "zh-CN", "cmn": "zh-CN", "chinese": "zh-CN", "mandarin": "zh-CN",
    "zh-hans": "zh-CN", "zh-hant": "zh-TW",
    "yue": "yue-CN", "cantonese": "yue-CN",
    "ja": "ja-JP", "japanese": "ja-JP",
    "ko": "ko-KR", "korean": "ko-KR",
    "de": "de-DE", "german": "de-DE",
    "fr": "fr-FR", "french": "fr-FR",
    "es": "es-ES", "spanish": "es-ES",
    "it": "it-IT", "italian": "it-IT",
    "pt": "pt-BR", "portuguese": "pt-BR",
}


def _resolve_apple_locale(language: str | None) -> str:
    key = (language or "").strip().lower()
    if not key or key == "auto":
        raise ValueError(
            "The apple ASR engine needs an explicit --asr-lang (e.g. en-us or zh-cn); "
            "SpeechTranscriber does not auto-detect the spoken language."
        )
    if key in _APPLE_LANG_ALIASES:
        return _APPLE_LANG_ALIASES[key]
    parts = key.replace("_", "-").split("-")
    if len(parts) == 2 and len(parts[1]) == 2:
        return f"{parts[0]}-{parts[1].upper()}"
    return language.strip()


def _bridge_error(stderr_tail: str, fallback: str) -> str:
    """Pick the most useful message from the bridge's stderr (it echoes its fatal error)."""
    lines = [ln for ln in stderr_tail.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("apple-asr: "):
            return ln[len("apple-asr: "):]
    return lines[-1] if lines else fallback


class AppleRecognizer:
    """Shared config: bridge binary + resolved locale (+ Sortformer weights if diarizing).
    Each create_stream() spawns its own bridge process."""

    def __init__(
        self,
        language: str = config.ASR_LANGUAGE,
        diarize: bool = False,
        log: Callable[[str], None] | None = None,
        binary: str | None = None,
        fast: bool = config.APPLE_ASR_FAST_RESULTS,
    ):
        say = log or (lambda _m: None)
        self.binary = resolve_apple_asr(binary)
        self.locale = _resolve_apple_locale(language)
        self.fast = fast
        say(f"Loading ASR model Apple SpeechTranscriber ({self.locale}, on-device) …")
        # Resolve the locale and install its system speech asset now (may download on first
        # use), so a bad --asr-lang fails before any audio flows.
        proc = subprocess.run(
            [self.binary, "--locale", self.locale, "--check"],
            capture_output=True, text=True, timeout=_READY_TIMEOUT_SEC * 5,
        )
        if proc.returncode != 0:
            raise ValueError(
                "Apple SpeechTranscriber: "
                + _bridge_error(proc.stderr, f"bridge exited with {proc.returncode}")
            )
        for line in proc.stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "ready":
                self.locale = ev.get("locale", self.locale)
        self.diar = None
        if diarize:
            from mlx_audio.vad import load as load_vad

            say(f"Loading diarization model {config.DIAR_MODEL} …")
            self.diar = load_vad(config.DIAR_MODEL)
            say("Warming up models …")
            with _MLX_LOCK:
                self.diar.feed(
                    np.zeros(_SR // 2, dtype=np.float32),
                    self.diar.init_streaming_state(),
                    sample_rate=_SR,
                    threshold=config.DIAR_THRESHOLD,
                )

    def create_stream(self) -> AppleOnlineStream:
        return AppleOnlineStream(self)


def build_apple_recognizer(
    language: str = config.ASR_LANGUAGE,
    diarize: bool = False,
    log: Callable[[str], None] | None = None,
) -> AppleRecognizer:
    """Validate the locale / install the system speech asset (+ optional Sortformer).

    Raises FileNotFoundError (build hint) if the bridge binary is missing, ValueError for an
    unsupported locale."""
    return AppleRecognizer(language, diarize, log)


class AppleOnlineStream:
    """One audio stream <-> one apple-asr process.

    accept_waveform / flush return events in the OnlineStream shapes AsrWorker expects:
      ("partial", text, None, utterance_samples)   latest volatile result
      ("final", [(speaker, text, None)], utterance_samples)
                                                   one finalized SpeechTranscriber result;
                                                   speaker = Sortformer majority label (None if
                                                   --no-diarize)
    utterance_samples = samples fed so far minus the result's start sample (AsrWorker uses it
    for audio-position timestamps in file mode).
    """

    def __init__(self, rec: AppleRecognizer):
        self._rec = rec
        self.seed_skew_sec = 0.0  # AsrWorker backdates live timestamps by this; n/a here
        # Volatile results carry no word timings (their range starts at the previous final's
        # end, i.e. includes the silence); finals do. Tell AsrWorker to timestamp file-mode
        # utterances from the final's start instead of the first partial.
        self.exact_final_start = True
        cmd = [rec.binary, "--locale", rec.locale]
        if rec.fast:
            cmd.append("--fast")
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._events: queue.Queue[dict | None] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=20)
        threading.Thread(target=self._read_stdout, daemon=True, name="apple-asr-out").start()
        threading.Thread(target=self._read_stderr, daemon=True, name="apple-asr-err").start()
        self._fed = 0  # samples written so far == the bridge's input timeline
        self._last_partial = ""
        self._finished = False
        # diarization audio window: chunks + the sample offset of the first chunk
        self._diar = rec.diar
        self._chunks: deque[np.ndarray] = deque()
        self._chunks_start = 0
        self._chunks_len = 0
        if self._diar is not None:
            self._diar_state = self._diar.init_streaming_state()
        self._wait_ready()

    # ---- bridge I/O ----

    def _read_stdout(self) -> None:
        assert self._proc.stdout is not None
        for raw in self._proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                self._events.put(json.loads(line))
            except json.JSONDecodeError:
                self._stderr_tail.append(line)
        self._events.put(None)  # EOF

    def _read_stderr(self) -> None:
        assert self._proc.stderr is not None
        for raw in self._proc.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                self._stderr_tail.append(line)

    def _fail(self, what: str) -> RuntimeError:
        msg = _bridge_error("\n".join(self._stderr_tail), what)
        return RuntimeError(f"apple-asr: {msg}")

    def _wait_ready(self) -> None:
        try:
            ev = self._events.get(timeout=_READY_TIMEOUT_SEC)
        except queue.Empty:
            self._proc.kill()
            raise self._fail("bridge did not become ready") from None
        if ev is None or ev.get("type") != "ready":
            self._proc.kill()
            if ev and ev.get("type") == "error":
                raise RuntimeError(f"apple-asr: {ev.get('message')}")
            raise self._fail("bridge exited before ready")

    # ---- public API ----

    def accept_waveform(self, samples: np.ndarray) -> list[tuple]:
        pcm = np.ascontiguousarray(samples, dtype="<f4")
        if len(pcm):
            assert self._proc.stdin is not None
            try:
                self._proc.stdin.write(pcm.tobytes())
            except BrokenPipeError:
                raise self._fail("bridge exited unexpectedly") from None
            if self._diar is not None:
                self._remember(pcm)
            self._fed += len(pcm)
        return self._drain(block=False)

    def flush(self) -> list[tuple]:
        """Stream ended: close stdin so the bridge finalizes everything, collect the rest."""
        if self._finished:
            return []
        self._finished = True
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.close()
        except BrokenPipeError:
            pass
        events = self._drain(block=True)
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        return events

    # ---- event conversion ----

    def _drain(self, block: bool) -> list[tuple]:
        out: list[tuple] = []
        while True:
            try:
                ev = self._events.get(timeout=_DONE_TIMEOUT_SEC) if block else (
                    self._events.get_nowait()
                )
            except queue.Empty:
                if block:
                    print("[warn] apple-asr: timed out waiting for final results",
                          file=sys.stderr)
                break
            if ev is None:  # bridge stdout closed
                if not self._finished:
                    raise self._fail(f"bridge exited (code {self._proc.poll()})")
                break
            kind = ev.get("type")
            if kind == "done":
                break
            if kind == "error":
                raise RuntimeError(f"apple-asr: {ev.get('message')}")
            if kind not in ("partial", "final"):
                continue
            text = _clean_text(str(ev.get("text", "")))
            start = max(0, int(round(float(ev.get("start", 0.0)) * _SR)))
            end = max(start, int(round(float(ev.get("end", 0.0)) * _SR)))
            utt_samples = max(0, self._fed - start)
            if kind == "partial":
                if text and text != self._last_partial:
                    self._last_partial = text
                    out.append(("partial", text, None, utt_samples))
                continue
            self._last_partial = ""
            if not text:
                continue
            speaker = self._dominant_speaker(start, end) if self._diar is not None else None
            out.append(("final", [(speaker, text, None)], utt_samples))
        return out

    # ---- diarization ----

    def _remember(self, pcm: np.ndarray) -> None:
        self._chunks.append(pcm)
        self._chunks_len += len(pcm)
        keep = int(_AUDIO_KEEP_SEC * _SR)
        while self._chunks and self._chunks_len - len(self._chunks[0]) >= keep:
            old = self._chunks.popleft()
            self._chunks_len -= len(old)
            self._chunks_start += len(old)

    def _slice(self, start: int, end: int) -> np.ndarray:
        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        buf = np.concatenate(self._chunks)
        a = max(0, start - self._chunks_start)
        b = max(a, min(len(buf), end - self._chunks_start))
        return buf[a:b]

    def _dominant_speaker(self, start: int, end: int) -> int | None:
        """Utterance-level label: feed the final's audio to Sortformer (state persists across
        utterances so S-numbers stay stable) and return the majority frame speaker."""
        import mlx.core as mx

        audio = self._slice(start, end)
        if len(audio) < _SR // 10:
            return None
        with _MLX_LOCK:
            res, self._diar_state = self._diar.feed(
                audio, self._diar_state, sample_rate=_SR, threshold=config.DIAR_THRESHOLD,
            )
            probs = np.array(res.speaker_probs.astype(mx.float32))
        probs = probs.reshape(-1, probs.shape[-1])
        labels = [int(p.argmax()) for p in probs if p.max() >= config.DIAR_THRESHOLD]
        if not labels:
            return None
        return max(set(labels), key=labels.count)
