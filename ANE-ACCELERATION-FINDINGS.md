# Apple Neural Engine acceleration for qwen-asr — findings

Investigation (2026-06-16) into offloading qwen3-asr-0.6b compute to the Apple
Neural Engine (ANE) on an M3 Ultra (macOS 26.5), to (a) run encoder matmuls on
the ANE and (b) get a ≥2× end-to-end speedup at CPU-equivalent accuracy.

**TL;DR**
- ✅ **Accuracy: solved.** ANE encoder offload at **fp32 IO** produces
  CPU-equivalent transcripts (often bit-identical, 5 languages tested).
- ❌ **Speed: ~1.0–1.4×, hard-capped; 2× is unreachable** on this hardware/model.
  The decoder (~70% of compute) is already int8-optimized and ANE-hostile; the
  accurate ANE encoder is break-even with the CPU. Verified by measurement,
  Amdahl, source inspection, and a working-but-failing Jacobi build.

---

## 1. What was built (all behind flags / off by default)

Files (in the `process_asr/qwen-asr` submodule, **uncommitted**):
- `crates/qwen-asr/src/mac_ane.rs` — CoreML model spec emitted as protobuf in
  Rust (no `coremltools`/Python), fp16/fp32 IO, the `AneLinear` wrapper, the
  per-weight ANE model cache for encoder offload, plus `benchmark()`,
  `reconcile()`, `splitk_probe()`.
- `crates/qwen-asr/ane/ane_shim.m` — Objective-C CoreML shim (compile + load with
  `computeUnits = CPUAndNeuralEngine`, run, report planned device via
  `MLComputePlan`). Compiled by `build.rs` via `cc`.
- `crates/qwen-asr/src/decoder.rs` — `decoder_forward_batch` (batched forward
  returning per-position argmax) for lookahead decoding.
- `crates/qwen-asr/src/transcribe.rs` — flag-gated Jacobi/lookahead decode loop.
- `crates/qwen-asr-cli/src/main.rs` — CLI flags below.
- Root `Makefile` — `itest-qwen-asr-ane-reconcile`, `itest-qwen-asr-ane-speed`.

Feature flag: build with `--features mac-ane`.

CLI / env knobs:
| Flag / env | Effect |
|---|---|
| `--mac-ane` | speed benchmark (CPU vs ANE GEMMs + concurrent throughput) |
| `--mac-ane-reconcile` | assert ANE matmul == CPU matmul within tolerance |
| `--mac-ane-splitk-probe` | per-GEMM error vs split-K groups |
| `--mac-ane-encoder` | offload encoder GEMMs (seq≥128) to ANE during transcription |
| `QWEN_ANE_IO16=1` | fp16 IO (fast, lossy) instead of fp32 (accurate) |
| `QWEN_ANE_SPLITK=G`, `QWEN_ANE_COMP=1`, `QWEN_ANE_ICOMP=1` | precision experiments |
| `QWEN_ANE_MAX=N` | cap number of offloaded GEMMs (diagnostic) |
| `QWEN_LOOKAHEAD=N` | Jacobi lookahead decode, window N (0 = off) |

---

## 2. Accuracy — solved (the real deliverable)

The matmul genuinely runs on the ANE (verified via `MLComputePlan`:
`preferredComputeDevice` is `MLNeuralEngineComputeDevice`).

**fp16 IO is fast but wrong:** the ANE runs the conv fully in fp16 → ~6% per-GEMM
error (cosine 0.998 in isolation), which **compounds across the 24-layer encoder
into repetition garbage** ("BIRD BIRD…", "Mau mau…"). Unusable.

**fp32 IO is accurate:** CoreML keeps fp32 accumulation around the conv → per-GEMM
error drops to **2.2e-4, cosine 1.00000** (including small-seq shapes). End-to-end,
`--mac-ane-encoder` (fp32 IO) vs the CPU baseline:

| Language | char diffs vs CPU |
|---|---|
| English | **bit-identical (0)** |
| German | **bit-identical (0)** |
| Chinese | 2 / 171 |
| Korean | 2 / 197 |
| French | 8 / 513 (30s) · 19 / 1780 (120s) |

All residual diffs are cosmetic: proper-noun accents (the ANE even *added* a
correct cedilla `exercant`→`exerçant`), capitalization, `cinq`→`5`, segment-edge
words — within normal ASR run-to-run noise (CPU-BLAS vs ANE differ at ~1e-6 f32
summation order and occasionally flip a borderline argmax). **WER-equivalent.**

Precision experiments (to push toward bit-identical):
- **Weight compensation** (`QWEN_ANE_COMP`, W=W_hi+W_lo): **no effect** — per-GEMM
  error flat at 1.61e-4 → the residual is NOT weight quantization.
- **Input compensation** (`QWEN_ANE_ICOMP`, x=x_hi+x_lo): the residual IS the ANE
  rounding x to fp16 internally; this pushes German+English to bit-identical.
- **Split-K** (`QWEN_ANE_SPLITK`): per-GEMM error EXACTLY flat across G=1..64 →
  the fp16 error is per-element, not K-accumulation; split-K can't reduce it.

---

## 3. Speed — ~1.0–1.4×, hard-capped at ~1.4×

### Pipeline profile (where time goes)
Decoder dominates; the ANE can only touch the encoder.

| Component | ~% of compute | ANE-amenable? |
|---|---|---|
| Decoder matvec (incl. lm_head) | ~45% | ❌ seq=1, memory-bound |
| Decoder attention | ~23% | ❌ |
| **Encoder GEMMs** | ~21% | ✅ batched |
| Encoder conv stem | ~6% | ~ |
| Encoder attention | ~2% | ❌ |

Decoder ≈ **70%**, encoder ≈ **30%** → Amdahl ceiling ≈ `1/(1−0.30) ≈ 1.4×`
even if the encoder were free.

### Measured end-to-end (fp32 IO, CPU-equivalent accuracy)
| Workload | speedup |
|---|---|
| **Pure encoder (120s silence, isolates ANE encoder)** | **1.00×** (2.88s CPU vs 2.89s ANE) |
| French 120s | 1.42× (favorable outlier) |
| German / English / CNA / BBC 120s | 0.88–1.02× |
| Default 30s segments (all langs) | ~1.0× |

**Key finding:** at the accuracy required (fp32 IO), the ANE encoder is
**break-even with the CPU** — the 3.7× ANE throughput only exists in fp16
(inaccurate). So there is *no speedup to harvest*, independent of speech density.
(`make itest-qwen-asr-ane-speed` reproduces the fp16 microbenchmark.)

### The decoder is already int8
`decoder.rs` (aarch64) already runs int8 matvec for decode
(`linear_nobias_int8_qkv`, `int8_addto`, `int8_swiglu`, `argmax_matvec_int8`).
The 70% bottleneck has no remaining CPU headroom and can't use the ANE
(seq=1 matvec, memory-bandwidth-bound; the ANE shares the same unified memory and
carries ~ms per-call dispatch overhead).

---

## 4. Jacobi / lookahead decoding — built, empirically fails for ASR

The only code-level route to 2× would be parallelizing the decoder losslessly.
Built `decoder_forward_batch` + a flag-gated Jacobi loop (`QWEN_LOOKAHEAD=N`).
Two independent killers:

1. **Acceptance rate ≈ 1.10**, flat across window N∈{4,8,16}. ASR transcription
   text is not self-predictable (unlike repetitive code), so no-draft Jacobi
   guesses almost never match the greedy continuation → no speedup ceiling above
   ~1.1×.
2. The batched path uses **f32/bf16 prefill GEMM** (4 bytes/wt), which is both
   *slower* than the already-int8 decode (1 byte) and *diverges* from it (different
   weights → not lossless). A proper batched-int8 GEMM kernel would be needed —
   not worth it given acceptance ≈ 1.1.

Measured: lookahead ran **~5× slower** (35s vs 7.4s). Real speculative decoding
needs a good **draft model** (high acceptance) — none exists for this model.

---

## 4b. Pipeline (encoder ‖ decoder) — SHIPPED, ~1.15× at exact accuracy

`--pipeline` (or `QWEN_PIPELINE=1`) overlaps encoding with decoding across
segments: a background worker encodes segment i+1 while the main thread decodes
segment i. Output is **byte-identical** to serial (same encoding, same past-text).

This required a **re-entrant thread-pool refactor** (`kernels/mod.rs`): the engine
had one global pool with a single dispatch slot, so concurrent `parallel_for` from
two threads corrupted it → deterministic SIGSEGV in the decoder's swiglu. The pool
is now instantiable, with TWO instances selected by a thread-local — a
default/decoder pool and an encoder-lane pool (`use_encoder_pool`, default 8
threads, `QWEN_ENC_THREADS`). Each has its own workers + dispatch atomics, so the
two lanes run concurrently without corruption. Single process; the model weights
stay shared (`Arc<QwenModel>`). Serial path is unchanged (byte-identical output,
tests pass).

Measured (CPU encode ‖ CPU decode, identical output):
- 120s French (4 segments): 5.31s → 4.64s = **1.14×**
- 300s English (10 segments): 10.07s → 8.64s = **1.16×** (scales with segments)

Why ~1.15× and not the ~1.4× Amdahl ceiling: the decoder is **memory-bandwidth-
bound**, and the encoder competes for the same unified-memory bandwidth, so the
overlap is partial. The ANE was meant to sidestep this (encoder compute off the
CPU), **but the accurate fp32-IO ANE encoder is too slow at the real seq (~366):
22s vs 5s** — its per-call transpose + per-segment model rebuild dominate. So the
pipeline uses the **CPU** encoder (not `--mac-ane-encoder`), and the ANE stays
unused in the shipping path.

## 5. Conclusion

- **Accuracy goal: met.** fp32-IO ANE encoder offload = CPU-equivalent transcripts.
- **2× speed goal: not achievable** with CPU+ANE at CPU accuracy on this
  hardware/model. Two hard walls: (1) accurate ANE matmul ≈ CPU speed (no win),
  (2) the 70% decoder is already int8 and ANE-hostile (Amdahl ≤ 1.4×).
- The ANE remains valuable for **throughput on isolated large fp16 matmuls**
  (the 3.7× concurrent benchmark) — just not for this accuracy-sensitive,
  decoder-bound ASR pipeline.

### What would actually deliver 2× (all outside "CPU+ANE, same accuracy")
- **A trained draft model** → real speculative decoding (high acceptance) →
  attacks the 70% decoder; the batched verify could then use the ANE.
- **A distilled/smaller decoder** → directly cuts the bottleneck.
- **Decoder int8→int4** → already int8; int4 would roughly halve its memory
  traffic, small accuracy cost (not "same accuracy").

---

## 6. Reproduce

```bash
cd process_asr/qwen-asr
cargo build --release --features mac-ane -p qwen-asr-cli

# accuracy parity (ANE matmul vs CPU)
./target/release/qwen-asr --mac-ane-reconcile

# fp16 throughput microbenchmark (the 3.7× that fails accuracy)
./target/release/qwen-asr --mac-ane

# end-to-end, accurate encoder offload
./target/release/qwen-asr -d qwen3-asr-0.6b -i clip.wav --language French --silent -S 120 --mac-ane-encoder

# Jacobi acceptance (≈1.10 for ASR)
QWEN_LOOKAHEAD=8 ./target/release/qwen-asr -d qwen3-asr-0.6b -i clip.wav --language French --silent -S 120
```
