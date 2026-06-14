# Changelog

## [0.9.1]

### Features

- **Loop / repetition detection & recovery in the segment (offline/`--serve`) path.** The
  offline path previously only *observed* decoder degeneracy; it now detects it (maxed token
  budget, or a tail block repeated ≥ 4 times via a wide period-32 detector AND covering ≥ half
  the output — a coverage gate so a brief legitimate refrain isn't mistaken for a runaway loop)
  and recovers by halving the clip at a word boundary and re-decoding each half, iteratively,
  bounded by `--loop-max-depth` (3) and `--loop-min-split-sec` (8 s). Detection thresholds are
  fixed constants (like the stream path); the only loop flags are `--loop-detect yes|no` (master
  toggle, both paths) and the two recovery bounds. See `docs/loop-detection.md`.
- **`--serve` honors all startup decode overrides** (`-S`/`-W`/`--loop-*`), built from one
  `DecodeSettings` shared verbatim with the one-shot CLI — serve and CLI can no longer diverge.

### Fixes

- **Sync to the C reference**: port `QWEN_STREAM_MAX_REPEAT_TOKEN_RUN` (single-token-run
  suppression) into the Rust `--stream` path, which had dropped it, *and* its companion
  `dropped_repeat_tokens >= 8` recovery-reset trigger — when the guard trims ≥ 8 tokens in a
  chunk the stream now re-anchors (matching `qwen_asr.c:1964`) instead of leaving the decoder
  context poisoned.

## [0.2.3] - 2026-02-23

### Features

- **Live audio capture** (macOS) — `--live` flag captures from audio input devices (microphone, BlackHole) in real time with segmented, streaming, or VAD modes
- **VAD live mode** — `--vad` flag uses energy-based Voice Activity Detection to detect speech segments and transcribe each independently, with cross-segment prompt conditioning for improved accuracy
- **Model download subcommand** — `qwen-asr download --list` and `qwen-asr download <model>` for built-in model management
- **Forced alignment** — `--align` flag produces word-level timestamps for a known transcript using the ForcedAligner model variant

### Performance

- **Lazy encoder re-encoding** — Partial encoder tail is only re-encoded every other chunk in streaming mode, giving near-perfect LCP (Longest Common Prefix) reuse and cutting decoder prefill cost by ~50% on skip chunks
- **Streaming robustness** — Degeneracy detection resets decoder state when stale or repetitive output is detected; periodic re-anchoring prevents unbounded sequence growth

### Changed

- Debug messages (`[stream degen]`, `[stream reanchor]`) now only appear in `--debug` mode
- Project restructured into workspace: `crates/qwen-asr` (library), `crates/qwen-asr-cli` (CLI binary)
- Removed WIP banner from all README files

## [0.2.0] - 2026-02-15

### Performance

- **Reusable BF16→F32 scratch buffer** — Pre-allocated scratch in `DecoderBuffers` eliminates ~140 heap allocations per prefill pass
- **SIMD BF16→F32 bulk conversion** — NEON (`vshll_n_u16`) and AVX2 (`_mm256_cvtepu16_epi32`) paths for 4-8x faster weight conversion
- **Threaded encoder attention** — `bidirectional_attention` parallelized across heads via thread pool for near-linear scaling
- **Cached mel filter bank** — `OnceLock`-based lazy initialization avoids redundant computation in streaming mode
- **SIMD activation functions** — Vectorized `rms_norm`, `gelu`, and `swiglu` with fast polynomial exp approximation (NEON + AVX2)
- **Encoder buffer reuse** — New `EncoderBuffers` struct with persistent scratch avoids per-call allocations in encoder forward pass
- **vDSP integration** (macOS, `--features vdsp`) — `vDSP_dotpr`, `vDSP_vsmul`, `vDSP_vsma`, `vvexpf` leverage Apple AMX coprocessor

### Features

- **Built-in profiling** — `--profile` flag prints per-operation timing breakdown (call count, total/avg time)
- **iOS support** — Static library target with C-FFI API (`src/c_api.rs`): `qwen_asr_load_model`, `qwen_asr_transcribe_file`, `qwen_asr_transcribe_pcm`, `qwen_asr_free`
- **Android support** — Shared library target with JNI API (`src/jni_api.rs`) for `com.qwenasr.QAsrEngine` Java class
- **Feature flags** — `blas` (default), `vdsp`, `ios`, `android` for platform-specific builds
- **Cross-compilation config** — `.cargo/config.toml` with tuned CPU targets for iOS (`apple-a14`) and Android (`cortex-a76`)

### Changed

- Library crate renamed to `qwen_asr` (was `q-asr`) for valid Rust identifier in imports
- Library target now produces `lib`, `staticlib`, and `cdylib` outputs
- Thread pool workers recover from poisoned mutex instead of panicking
- Regression tests serialized via `Mutex` to prevent thread pool race conditions
- README updated with per-platform build instructions (macOS, Linux, iOS, Android)

## [0.1.0] - 2026-02-15

Initial release — pure Rust port of [antirez/qwen-asr](https://github.com/antirez/qwen-asr).

- CPU-only Qwen3-ASR inference (0.6B and 1.7B models)
- Three runtime modes: offline, segmented (`-S`), streaming (`--stream`)
- NEON SIMD (aarch64) and AVX2+FMA (x86_64) acceleration
- BLAS via Accelerate (macOS) / OpenBLAS (Linux)
- Zero runtime Rust crate dependencies (only `libc`)
- 22 tests (kernels, audio, tokenizer, regression)
