# qwen-asr

Pure Rust, CPU-only inference engine for [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) speech-to-text. Zero runtime crate deps (only `libc`). Ported from [antirez/qwen-asr](https://github.com/antirez/qwen-asr).

Supports 0.6B and 1.7B models. Modes: offline, segmented, streaming, live capture, VAD live, forced alignment.


## Auto Research

Performance optimizations were discovered autonomously using the [autoresearch](https://github.com/karpathy/autoresearch) pattern: an AI agent loops over hypothesize-implement-benchmark-keep/revert cycles on the inference code. The experiment protocol is defined in [`program.md`](program.md).

## Benchmark

Offline ASR benchmark on macOS (Apple M5, 10 standalone rounds, 28.2s audio). Implementations are benchmarked sequentially, not in parallel. The primary metric is median inference time, so model loading and process startup do not dominate comparisons.

| Implementation | Commit | Median inference ms | Mean ms | Best ms | RTF |
|---|---:|---:|---:|---:|---:|
| qwen-asr (first) | [`bf52daf`](https://github.com/huanglizhuo/QwenASR/commit/bf52daf) | 1,842 | 1,853 | 1,820 | 15.31x |
| qwen-asr (latest) | [`0f5f065`](https://github.com/huanglizhuo/QwenASR/commit/0f5f065) | 676 | 678 | 668 | 41.69x |
| pure C upstream | [`b00b789`](https://github.com/antirez/qwen-asr/commit/b00b789) | 1,885 | 1,885 | 1,861 | 14.94x |
| second-state MLX GPU | [`3fa6734`](https://github.com/second-state/qwen3_asr_rs/commit/3fa6734) | 2,785 | 2,808 | 2,745 | 10.11x |
| mlx-audio Python MLX | [`0.4.3`](https://github.com/Blaizzy/mlx-audio) | 801 | 820 | 788 | 35.16x |

qwen-asr and pure C use internal inference timers. MLX-based implementations are timed after model load with explicit GPU synchronization. Wall-clock time is still recorded for diagnostics and end-to-end command cost.

<details>
<summary>Wall-clock timing</summary>

| Implementation | Commit | Median wall-clock ms | Mean ms | Best ms | Wall-clock RTF |
|---|---:|---:|---:|---:|---:|
| qwen-asr (first) | [`bf52daf`](https://github.com/huanglizhuo/QwenASR/commit/bf52daf) | 2,171 | 2,205 | 2,150 | 12.99x |
| qwen-asr (latest) | [`0f5f065`](https://github.com/huanglizhuo/QwenASR/commit/0f5f065) | 1,263 | 1,289 | 1,252 | 22.34x |
| pure C upstream | [`b00b789`](https://github.com/antirez/qwen-asr/commit/b00b789) | 2,154 | 2,148 | 2,125 | 13.08x |
| second-state MLX GPU | [`3fa6734`](https://github.com/second-state/qwen3_asr_rs/commit/3fa6734) | 2,982 | 3,049 | 2,940 | 9.44x |
| mlx-audio Python MLX | [`0.4.3`](https://github.com/Blaizzy/mlx-audio) | 1,855 | 1,918 | 1,806 | 15.18x |

</details>

![Latency](bench/charts/benchmark-unified-latency.png)

![Realtime factor](bench/charts/benchmark-unified-rtf.png)

- **Fastest overall**: qwen-asr latest `0f5f065`
- qwen-asr latest is **2.72x** faster than qwen-asr first `bf52daf` by median inference time
- qwen-asr latest is **2.79x** faster than the upstream pure C implementation by median inference time
- qwen-asr latest is **4.12x** faster than second-state MLX Metal GPU by median inference time
- qwen-asr latest is **1.18x** faster than mlx-audio Python MLX (8-bit) by median inference time

Reproduce all results:

```bash
# qwen-asr first + latest + pure C + second-state MLX GPU + mlx-audio
./bench/benchmark-all.sh --runs 10
```

### Why does pure CPU Rust beat GPU?

1. **Hand-optimized NEON kernels** — Custom `vDSP`/`Accelerate`, hand-written `neon_dotprod` matmul, and fused fast-attention specifically tuned for the 0.6B model and Apple Silicon cache hierarchy.
2. **Zero framework overhead** — No tensor dispatch, memory pools, or FFI bridging. 100% Rust end-to-end.
3. **Model too small for GPU** — A 0.6B model can't saturate the Metal GPU. Kernel launch overhead and CPU↔GPU copies dominate. Both MLX backends are ~2–4× slower than the CPU path.
4. **mlx-audio 8-bit overhead** — Quantization saves memory but dequantization during compute adds extra work.

## Quick Start

```bash
# Install
cargo install qwen-asr-cli

# Download model
qwen-asr download qwen3-asr-0.6b

# Transcribe
qwen-asr -d qwen3-asr-0.6b -i audio.wav
```

Or download a pre-built binary from [GitHub Releases](https://github.com/huanglizhuo/QwenASR/releases).

## Usage

```
qwen-asr -d <model_dir> (-i <file> | --stdin | --live) [options]
```

```bash
qwen-asr -d qwen3-asr-0.6b -i audio.wav              # basic
qwen-asr -d qwen3-asr-0.6b -i audio.wav --silent      # transcript only
cat audio.wav | qwen-asr -d qwen3-asr-0.6b --stdin     # pipe from stdin
qwen-asr -d qwen3-asr-0.6b -i long.wav -S 30           # segmented (long audio)
qwen-asr -d qwen3-asr-0.6b -i audio.wav --stream       # streaming
qwen-asr -d qwen3-asr-0.6b --live --device "BlackHole 2ch"         # live capture (macOS)
qwen-asr -d qwen3-asr-0.6b --live --vad --device "BlackHole 2ch"   # VAD live
qwen-asr -d qwen3-aligner-0.6b -i audio.wav --align "Hello world" --align-language English  # alignment
ffmpeg -i video.mp4 -f s16le -ar 16000 -ac 1 - | qwen-asr -d qwen3-asr-0.6b --stdin        # raw PCM
```

<details>
<summary>All options</summary>

| Option | Description | Default |
|--------|-------------|---------|
| `-d <dir>` | Model directory (required) | — |
| `-i <file>` | Input WAV file | — |
| `--stdin` | Read audio from stdin (WAV or raw s16le 16kHz) | off |
| `--live` | Live capture from audio device (macOS) | off |
| `--device <name>` | Input device for live capture | system default |
| `--list-devices` | List audio input devices | — |
| `--vad` | VAD live mode | off |
| `-t <n>` | Thread count | all CPUs |
| `-S <secs>` | Segment target seconds | 0 (full) |
| `--stream` | Streaming mode | off |
| `--stream-chunk-sec <s>` | Chunk size for streaming | 2.0 |
| `--language <lang>` | Force output language (`en`, `zh`, `ja`, ...) | auto |
| `--silent` | Transcript only, no status output | off |
| `--profile` | Print timing breakdown | off |

</details>

## Build

**Always use release mode.** Debug builds are 10-50x slower.

```bash
# macOS
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Linux
sudo apt install libopenblas-dev   # Debian/Ubuntu
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Without BLAS
RUSTFLAGS="-C target-cpu=native" cargo build --release --no-default-features

# iOS (static library + C-FFI)
cargo build --release --target aarch64-apple-ios --features ios

# Android (shared library + JNI)
cargo ndk -t arm64-v8a build --release --features android
```

| Feature | Description |
|---------|-------------|
| `blas` (default) | BLAS linking (Accelerate on macOS, OpenBLAS on Linux) |
| `vdsp` | Accelerate vDSP/vForce for AMX (macOS) |
| `ios` | C-FFI API (`src/c_api.rs`) |
| `android` | JNI API (`src/jni_api.rs`) |

## OpenClaw Skill

One-command install for [OpenClaw](https://github.com/anthropics/openclaw) users:

```bash
bash skills/qwen-asr/scripts/install.sh
bash skills/qwen-asr/scripts/transcribe.sh audio.wav
```

## Acknowledgments

Rust port of [antirez/qwen-asr](https://github.com/antirez/qwen-asr), a pure C implementation of Qwen3-ASR inference by antirez.

## License

Same license as [antirez/qwen-asr](https://github.com/antirez/qwen-asr).
