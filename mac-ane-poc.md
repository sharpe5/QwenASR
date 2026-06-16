# Apple Neural Engine (ANE) matmul offload — proof of concept

`--mac-ane` (macOS, build with `--features mac-ane`) benchmarks the engine's
encoder GEMMs on the Apple Neural Engine vs the CPU and reports concurrent
CPU+ANE throughput.

```
cargo build --release --features mac-ane -p qwen-asr-cli
./target/release/qwen-asr --mac-ane            # speed benchmark + concurrent throughput
./target/release/qwen-asr --mac-ane-reconcile  # assert ANE output == CPU matmul (fp16 tol)
```

From the mrecord repo root (builds the feature, then runs):

```
make itest-qwen-asr-ane-speed       # CPU vs ANE timing + parallel CPU+ANE throughput
make itest-qwen-asr-ane-reconcile   # parity: ANE output matches CPU matmul within fp16
```

## Result (M3 Ultra, macOS 26.5)

- Matmul **confirmed running on the ANE** — verified via `MLComputePlan`
  (`preferredComputeDevice` is `MLNeuralEngineComputeDevice`).
- Pure ANE inference ≈ full-CPU (Accelerate, all 20 P-cores): ~3–6 GFLOP/s parity
  per op; ANE sustains ~5.5–6.1 TFLOP/s (fp16) when batched (seq ≥ 3000).
- **Concurrent CPU + ANE throughput ≈ 3.7–3.8×** the CPU alone
  (CPU ~4.2 TFLOP/s + ANE ~11.8 TFLOP/s ≈ 16 TFLOP/s combined).
  Target was ≥ 2×.

## How it works (no `coremltools`, no Python)

1. **Model spec built in Rust as protobuf** (`src/mac_ane.rs`). The linear
   `y = x @ Wᵀ` is emitted as a CoreML `NeuralNetwork` with a single **1×1
   convolution** (conv is the ANE's best-supported op; the classic NN validator
   rejects rank-2 IO, so IO is rank-3 `[C,1,seq]`). Weight `[out,in,1,1]` is
   byte-identical to the engine's `[out,in]` and is baked in as **fp16**.
   Spec version 7 (required for fp16 IO).
2. **Objective-C shim** (`ane/ane_shim.m`, compiled by `build.rs` via `cc`,
   links `CoreML`+`Foundation`): writes the spec to a temp `.mlmodel`, compiles
   it with `MLModel compileModelAtURL:`, loads it with
   `computeUnits = CPUAndNeuralEngine`, and runs predictions over reusable fp16
   `MLMultiArray`s. Also reports the planned compute device via `MLComputePlan`.

## Why the win comes from parallelism, not per-op speed

A single ANE prediction has high per-call latency (dispatch + the fp16
pack/transpose, which is CPU-side). For **throughput** that doesn't matter: the
ANE is a *separate* compute unit, so

- the f32↔f16 pack/transpose is done CPU-side and **overlaps** ANE compute
  (`forward_raw` takes pre-packed fp16 in CoreML's channel-major layout), and
- **two driver threads double-buffer** submissions so the ANE never idles
  between predictions (single-threaded it sat ~50% idle → ~2× gain from 2
  threads).

The CPU keeps doing GEMMs at full Accelerate rate while the ANE adds ~2.8× more
matmul throughput on top → ~3.7× combined.

## RESOLVED: fp32 IO makes encoder offload accuracy-viable

The earlier "negative result" below was an artifact of using **fp16 IO** (chosen
for throughput). With fp16 IO the ANE runs the conv fully in fp16 (~6% error →
garbage through the deep stack). Switching the encoder offload to **fp32 IO**
(CoreML then keeps fp32 accumulation around the conv) drops per-GEMM error from
6.4e-2 to **1.6e-4** (cosine 1.00000) — including the small-`seq` shapes that
were 34% before. End-to-end this turns garbage into correct transcripts.

Three approaches tried (goal: match the CPU transcript):

1. **fp32 IO** (the fix; default for `--mac-ane-encoder`). French 120s: 8986
   char-diffs (garbage) → **19/1780 (98.9% identical)**. 5-language 30s clips:
   English identical; fr/de ~8 diffs, zh/ko ~2 — all cosmetic.
2. **Weight compensation** (`QWEN_ANE_COMP=1`, W=W_hi+W_lo each fp16): **no
   effect** — per-GEMM error unchanged (1.616e-4 → 1.614e-4), proving the
   residual is not weight quantization.
3. **Input compensation** (`QWEN_ANE_ICOMP=1`, x=x_hi+x_lo each fp16, summed in
   f32): the residual IS the ANE rounding `x` to fp16 internally. Pushes German +
   English to **bit-identical**; French 120s → 19 diffs.

Residual floor: a handful of ambiguous tokens per clip (proper-noun accents,
numbers, segment-boundary words) where CPU-BLAS vs ANE f32 summation order
differ at ~1e-6 and flip a borderline argmax. This is WER-equivalent to CPU —
the accuracy goal is met. (Throughput of the fp32-IO path is lower than the fp16
microbenchmark; accuracy and speed trade off via the IO dtype.)

## End-to-end encoder offload (`--mac-ane-encoder`) — original (fp16-IO) negative result

`--mac-ane-encoder` routes the encoder's attention + FFN GEMMs (seq ≥ 128)
through the ANE during real transcription, with a per-weight model cache. Tested
on real multilingual broadcast audio (fr/de/zh/ko/en, `datafix/`):

- The integration is **correct**: offloading a single GEMM yields a near-identical
  transcript (≈29 char diffs of ~1800).
- But fp16 error **compounds with depth**. Offloaded GEMMs vs CPU (French, 120s):
  1 → 29 diffs (intact); 6 → 59 (intact); 24 → 518 (degrading); all ~144
  (6 GEMMs × 24 layers) → **collapse into repetition garbage** ("BIRD BIRD…",
  "Mau mau…", "désolé désolé…").
- This holds at BOTH the real seq (~366 per 30 s segment) and large seq (~1560),
  so it is the deep-stack fp16 *accumulation*, not the small-tensor kernel.

Conclusion: the single-GEMM reconcile (cosine 0.998) looks fine, but wholesale
fp16 encoder offload is **not usable** — ~6% per-GEMM error stacked over the
24-layer encoder corrupts the representation and the loop-prone decoder
degenerates.

### Split-K (f32 cross-chunk accumulation) — tried, does NOT help

`QWEN_ANE_SPLITK=<G>` splits each GEMM's K into G chunks, runs each on the ANE,
and sums the partials in f32. Per-GEMM error vs CPU f32 (`--mac-ane-splitk-probe`,
seq=1500, K=896) is **exactly flat**:

```
split-K   cosine    rel_err
   1      0.99791   6.463e-2
   2      0.99791   6.463e-2
   8      0.99791   6.463e-2
  64      0.99791   6.463e-2
```

So the ~6% is **not** K-accumulation error (G=64 → K=14 per chunk changes
nothing); it is per-element and irreducible by decomposition. End-to-end French
at `-S 120` stays garbage for every G. The ANE simply has no f32-accumulation
matmul, so a deep transformer encoder can't get the accuracy it needs.

### What does work

- **Bounded hybrid** (`QWEN_ANE_MAX=<n>`): offloading ≤ ~6 GEMMs (≈1 layer) keeps
  the transcript intact (≈60 char diffs / 1800), but the speedup is negligible.
- **Throughput on isolated large matmuls** — the original `--mac-ane` POC (~3.7×
  parallel CPU+ANE) stands; the ANE is good where a single big GEMM's ~6% error
  is acceptable, just not stacked 144-deep.

## Caveats / next steps before production

- **Precision**: fp16 inputs + fp16 accumulation on the ANE give ~6% relative
  error vs the CPU f32 path on random data (cosine ~0.998 — same operation, not
  bit-exact). Must be WER-evaluated on real audio before wiring into the encoder;
  the engine currently keeps activations f32. `--mac-ane-reconcile` quantifies it.
- **Small tensors**: for small `seq` (≈200) the ANE selects a lower-precision
  kernel and error jumps to ~34% (independent of channel dims). Offload only
  large batched GEMMs (seq ≥ ~1500); that's also where throughput is best.
- Integration would build one cached ANE model per (weight, seq) and route the
  encoder's batched `linear` calls through it, feeding multiple audio segments
  in parallel (throughput-oriented; latency is not a concern for the coproc).
