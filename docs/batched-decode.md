# 16-way batched offline decode (beta)

Status: **working, bit-identical, verified on 4 languages × 15 min.** Single
process, single thread. Opt-in via `--batch <n>` (default `0` = off → unchanged
sequential path). This doc is the resume-context record for the work.

## Goal

> Create a 1-process, 1-thread version of `process_asr/qwen-asr-beta` that has
> 16-way batching (compute on batches of 16 ×30s clips at once; leverage 128-bit
> NEON SIMD — or AMX via BLAS/vDSP if it can stay bit-identical). Iterate until
> output is **bit-identical** to the original `process_asr/qwen-asr` on **≥4
> languages with 15-minute clips**, within 8 hours.

The hard constraint is **bit-identical** output. Everything below is shaped by it.

## TL;DR of the result

- `--batch <n>` decodes up to `n` ~30 s windows concurrently, **byte-identical**
  to the sequential original on the plain-text, `--json`/segmented, and `--serve`
  paths.
- Verified byte-identical on 15-min clips: **Chinese, Korean, German, French**
  (and English broadcast across `--batch 1/5/16/32`). 73/73 unit+integration
  tests pass.
- Performance **knee is `--batch ≈ 4`** (plateau through 32). Uncontended speedup
  on healthy content is modest (~7–9%: zh 79→74 s, de 83→76 s); it grows under
  memory-bandwidth pressure. Degenerate-heavy content sees no benefit (recovery is
  sequential). `--batch 1` is strictly worse than sequential. See the sweep below.

## Why NEON INT8, not AMX/BLAS

The autoregressive single-token decode is the dominant cost (thousands of steps,
each reading the whole ~0.6 B decoder weight set) and on aarch64 it runs as
**INT8 `sdot` matvec**. AMX (via Accelerate `cblas_sgemm`) is **f32 only** — using
it for the decode would not reproduce the INT8 arithmetic, so it would break
bit-identicality. Therefore the decode stays on INT8 NEON.

The encoder + decoder *prefill* already use `cblas_sgemm` (→ AMX) and are left
**per-window** (run sequentially, reused verbatim → trivially bit-identical).
They are a one-pass-per-window cost, not the bottleneck, so batching them was not
needed to satisfy the goal. (Possible future work; see below.)

## The batching idea: weight-stationary, not SIMD-lane-packed

Two ways to "batch":

1. **Pack clips into SIMD lanes** (4 clips in a NEON f32 vector). This changes the
   reduction order of each dot product → **not bit-identical**. Rejected.
2. **Weight-stationary reordering** (used here). The decode matvec computes, for
   each output row `o`, an independent dot product per window. Batching loads each
   weight row **once** and applies it to all `N` windows, reusing the *identical*
   per-window reduction. This only reorders *independent* dot products — it never
   reassociates a single sum — so every value is unchanged.

The win is **weight memory bandwidth**: the big INT8 matrix is streamed from DRAM
once per `N` windows instead of once per window. The decode is bandwidth-bound on
weights, so this is the lever. (It does **not** cut compute — same total `sdot`s —
which is why the ceiling is ~2×, not ~16×.)

### Why it is exactly bit-identical

INT8 `sdot` accumulation is **exact integer arithmetic** (no rounding). The integer
dot product is the same regardless of how lanes/accumulators are grouped, so the
batched single-row reduction yields the *same* `i32` sum as the original's pair-row
reduction. The only float ops are the final `sum as f32 * x_scale * w_scale`
(reproduced verbatim) and a scalar tail that is dead for these dims (all of
`dim=1024, q=2048, kv=1024, inter=3072→gate_up 6144, lm=151936` are multiples of
32). Per-window rms-norm / RoPE / attention (over that window's own KV cache) /
SwiGLU / quantize all call the **identical kernels per lane**.

Segments that finish (EOS) or hit the token cap drop out of the active set; that
never touches a still-active segment's math, so each window's token stream — and
its text — is identical to decoding it alone.

## What was implemented

### New kernels — `crates/qwen-asr/src/kernels/neon.rs`
- `matvec_int8_batched(ys, xs_int8, x_scales, w_int8, w_scales, batch, in_dim, out_dim, accumulate)`
- `argmax_int8_batched(best, xs_int8, x_scales, w_int8, w_scales, batch, in_dim, out_dim)`
- Both dispatch over **const-generic, register-resident lane tiles**
  `matvec_lanes::<N>` / `argmax_lanes::<N>` for `N ∈ {8,4,2,1}`. Each `N` is a
  separately **monomorphized** SIMD function with the lane loop unrolled and the
  loaded weight vectors reused across lanes from registers. Any `--batch n`
  composes from these tiles (16 = 2×8, 32 = 4×8, 64 = 8×8), still a single DRAM
  pass of each weight row per output.
- **Why 8 is the largest tile:** a tile holds `2·N` `int32x4` accumulators live;
  `N=8` (16 vregs) + the 2 weight vectors fit the 32-register NEON file. `N=16`
  would spill. (`argmax_lanes` uses a 2-accumulator reduction — still exact, so
  still bit-identical — to keep register pressure down.)
- Public wrappers in `kernels/mod.rs`: `matvec_int8_batched`, `argmax_int8_batched`
  (aarch64 only; `unimplemented!` elsewhere — the batch module falls back to
  sequential on non-aarch64).

### Orchestration — `crates/qwen-asr/src/batch.rs` (new module)
- `transcribe_audio_batched(ctx, samples, max_batch)` — batched twin of
  `transcribe::transcribe_audio` (plain text).
- `transcribe_segmented_batched` / `transcribe_clips_batched` — batched twins of
  the `--json` / `--serve` segmented + clips paths.
- `batched_decode_step` — one batched single-token step reproducing
  `decoder::decoder_forward`'s aarch64 INT8 path: per-lane rms-norm, batched QKV
  matvec, per-lane q/k-norm + RoPE + KV-write + causal attention, batched o-proj
  (residual add), per-lane post-norm, batched gate_up + per-lane SwiGLU, batched
  down-proj (residual add), per-lane final norm, batched lm-head argmax.
- Per-window encode + prefill reuse the **existing** `Encoder::forward` and
  `decoder::decoder_prefill` verbatim (bit-identical), one window at a time, into
  per-window `KvCache`s. Only the autoregressive loop is batched.
- Falls back to the sequential path when batching can't apply: non-aarch64,
  `--past-text` on (sequential dependency between windows), aligner model, a
  streaming token callback, or `batch < 1`.

### Degeneracy / loop recovery (the 32→16→8 split)
A coarse ~30 s window that the model loops on is, in the original, decoded by
`transcribe::transcribe_with_recovery`, which halves it at word boundaries
(32→16→8 s) and re-decodes. The batched segmented path:
- Decodes **healthy** windows in batches and emits their text directly.
- Hands **degenerate** windows to the **unchanged sequential
  `transcribe_with_recovery`**, which does the bit-identical 32→16→8 halving on the
  unpadded span. Because the batched decode emits identical tokens, the *same*
  windows are flagged degenerate (`transcribe::segment_is_degenerate`, reused
  verbatim), so the splits and every emitted segment are byte-identical.

**Early-abort optimization** (`decode_group(..., loop_detect)`): a window that is
*already* a repetition loop will go to recovery regardless, so the batched loop
aborts that lane the moment its partial output trips the authoritative loop check
(`segment_is_degenerate(partial, maxed=false, true)`, polled every 16 text tokens)
and hands it to recovery, instead of grinding it to the 2048-token cap inside the
batch. Bit-identical because aborting only **adds** a window to the recovery set
(recovery is authoritative); a window emitted directly always completed cleanly
with a healthy final verdict. The plain `transcribe_audio` path has no recovery, so
it passes `loop_detect=false` and never aborts.

### Plumbing — shared CLI/serve default
- `DecodeSettings.batch_size` (in `context.rs`) is the **single source of truth**,
  so the CLI and `--serve` share the same default (0) and a `--batch n` passed on
  the `--serve` launch flows through to every resident ctx (`apply_settings`
  copies it; `serve.rs` routes on `ctx.batch_size`).
- `--batch <n>` flag in `main.rs`, applied to `settings.batch_size`; routes the
  plain, `--json`, and `--srt` paths. Shown in `--help`.

## Model facts (0.6B, `qwen3-asr-0.6b`)
`dec_hidden=1024, dec_layers=28, dec_heads=16, dec_kv_heads=8, dec_head_dim=128,
dec_intermediate=3072, vocab=151936`. So `q_dim=2048, kv_dim=1024, gate_up=6144,
lm_head in=1024 out=151936` — all in/out dims are multiples of 32 (no scalar tail).
Default decode settings: `past_text=Auto→OFF` for offline (windows independent),
`loop_detect=true` (recovery active).

## Verification

Reference = original `process_asr/qwen-asr` (built `--release`,
`-C target-cpu=native`). Compare `--json` output byte-for-byte, single thread
(`-t 1 --silent`).

Test clips (15 min, 16 kHz mono WAV, extracted from `/Users/t/mrecord/datafix/`
6-hour broadcast opus with `ffmpeg -ss 1800 -t 900 -ar 16000 -ac 1`):
- zh — `radio-yicai-first-financial-…` (Chinese)
- ko — `radio-ytn-radio-kr-…` (Korean)
- de — `tv-welt-de-…` (German)
- fr — `radio-france-inter-…` (French)

Result: **all byte-identical (`cmp`), `PASS=4 FAIL=0`.** Also byte-identical on
`bench/samples-compare/broadcast119.wav` (English) for `--batch 1/5/16/32` and on
the plain-text path.

### Batch-size sweep / performance knee

Method: for each batch size the 4 languages run **concurrently** (one process per
language, each `-t 1`), so contention is uniform across batch sizes and the
*relative* comparison is valid (M3 Ultra, 28 cores; 4 single-thread decoders do
not saturate memory bandwidth). Wall-clock seconds for the 15-min `--json` decode,
**byte-identical across every batch size** (md5 single-valued per language):

| `--batch` | zh (healthy) | de (healthy) | ko (mixed) | fr (degenerate-heavy) |
|-----------|:----:|:----:|:----:|:----:|
| 0 (seq) | 79 | 83 | 113 | 197 |
| 1       | 82 | 85 | 116 | 201 |
| 2       | 77 | 77 | 113 | 202 |
| **4**   | **74** | **76** | 114 | 202 |
| 8       | 74 | 76 | 114 | 202 |
| 16      | 74 | 75 | 114 | 203 |
| 32      | 73 | 76 | 114 | 203 |

**Knee = `--batch ≈ 4`.** On healthy content the gain saturates by B=4
(zh 79→74, de 83→76, ~7–9%) and is flat from 4 through 32 — B≥8 buys nothing
measurable. **B=1 is strictly worse than sequential** (batched-module overhead
with no batching benefit; never use it). Degenerate-heavy content (ko, fr) shows
**no benefit**: it is dominated by sequential loop-recovery, which batching cannot
accelerate (fr is even ~1% slower from the abort bookkeeping).

**Why modest here, and bandwidth-dependent.** Batching saves weight *memory
bandwidth*, not *compute* (same total `sdot` count). On an uncontended M3 Ultra the
decode is largely compute-bound, so the win is single-digit %. The win **grows with
memory-bandwidth pressure**: earlier numbers measured while the box was also running
builds + the reference decode showed ~1.5× — partly a contention artifact, but a
real signal that on a bandwidth-starved machine (or with many concurrent decoders)
batching helps substantially more. So the right `--batch` is deployment-dependent.

**Recommended default: `--batch 8`** (safely on the plateau, robust to load); use 4
to minimise per-step working set. Do not bother above ~8 for ~30-window (15-min)
clips, and never use 1.

### GPU (MLX) feasibility — the batch knee is the *opposite* story

The CPU result above (batching barely helps) is because the CPU does the same total
`sdot` compute regardless of batch. The GPU is the mirror image: it turns batch into
throughput. Microbenchmark of the **exact 0.6B decoder matmul stack** (28 layers'
qkv/o/gate_up/down + lm_head) on MLX f16, M3 Ultra GPU, `mx.eval` per step
(autoregressive sync), **matmul-only — attention omitted** (optimistic upper bound):

| batch | ms/step | window-tokens/s | TFLOP/s |
|------:|--------:|----------------:|--------:|
| 1   | 3.3   | 306    | — |
| 16  | 7.3   | 2 203  | 2.6 |
| 32  | 7.4   | 4 314  | 5.1 |
| 64  | 8.7   | 7 339  | 8.7 |
| 128 | 12.6  | 10 153 | 12.1 |
| 256 | 24.0  | 10 686 | 12.7 |
| 512 | 43.0  | 11 914 | 14.2 |
| 1024| 78.5  | 13 050 | 15.6 |
| 2048| 150.7 | 13 589 | 16.2 |
| 4096| 290.1 | 14 122 | 16.8 |

**GPU throughput knee ≈ batch 64–128.** Below it `ms/step` is flat (~7–9 ms) — pure
kernel-dispatch-latency regime — so throughput scales ~linearly with batch. Above
~128 it goes compute-bound (`ms/step` ∝ batch, TFLOP/s saturates ~16–17) and
throughput plateaus (10 k→14 k from 128→4096). **Sweet spot ≈ 128.**

This explains the earlier "mlx-lm slower than qwen-asr" result: that was **batch=1**,
where the GPU is latency-bound (306 win-tok/s matmul-only) and loses to the tuned
CPU INT8 path once real attention + per-token framework overhead are added. The CPU
can't convert batch into speed; the GPU can — by ~50× from batch 1 to 128.

Caveats before reading this as "port to GPU":
1. **Matmul-only.** The real decode is ~40% per-window causal attention over a
   **dynamic KV cache** (ragged across windows). That is the GPU's weak spot and the
   real engineering cost; end-to-end GPU throughput will be well below this ceiling.
2. **batch = concurrent 30 s windows.** The production input is **6-hour .opus files
   ≈ 720 windows each** (21 600 s ÷ 30 s), so batch 64–128 is reachable **from a single
   file** — the GPU's sweet spot is directly in range, no cross-file juggling needed.
   (The windows are independent by default — no past-text — so all 720 are batchable.)
3. **Not bit-identical** (GPU f16 ≠ CPU INT8). It would be a separate production-speed
   engine validated by WER/transcript parity, not `cmp`.

**With attention included** (MLX fused `scaled_dot_product_attention`, GQA, KV=512 —
the second microbench, `tmp/mlx_bench3.py`):

| batch | ms/step | window-tokens/s |
|------:|--------:|----------------:|
| 1   | 3.7  | 271   |
| 16  | 8.7  | 1 846 |
| 32  | 10.3 | 3 118 |
| 64  | 13.9 | 4 610 |
| 128 | 23.1 | 5 536 |
| 256 | 44.3 | 5 783 |

Attention scaling with KV length (batch=128): Lkv 128→7 533, 256→6 969, 512→5 522,
1024→3 299 win-tok/s — i.e. the per-step attention cost grows ∝ context length, so
longer prefill / later decode positions cost more.

Adding attention roughly **halves** the matmul-only ceiling (b128: 10 153→5 536) and
pulls the **knee to ~64–128** (plateau begins ~128). Even so, at batch 128 the GPU
does ~5 500 window-tokens/s vs the CPU's ~40–50 per window — a ~100× *compute* gap.

Read it honestly though: these microbenches are a **compute ceiling**, not an
end-to-end number. They omit per-token framework/Python overhead, real ragged-KV
cache growth + padding/masking, rms-norm/RoPE/q-k-norm, sampling and detokenisation —
exactly the fixed per-step overheads that sank the earlier **batch=1** mlx-lm test
(and that amortise away as batch grows). So:
- **batch 1:** real GPU loses (overhead-bound) — matches the prior result.
- **batch 64–128:** even if real overhead cuts the ceiling 2–3×, that is still
  ~1 000–2 500 window-tok/s, i.e. **~20–50× the CPU** — a decisive win *if* the queue
  can supply that many concurrent windows (batch across many blocks, not one clip).

Bottom line: a **batched MLX GPU engine is worth prototyping for the large-batch
queue regime** (target batch ~64–128). The remaining risk is engineering, not
throughput: efficient **ragged-KV batched attention** (windows at different positions)
and keeping per-step framework overhead low. Reproduce: `tmp/mlx_bench2.py`
(matmul-only) and `tmp/mlx_bench3.py` (with attention).

### ggml / llama.cpp batched-decode spike — **measured** (the de-risk)

Before committing to a ggml port, we measured ggml's batched decode directly, using
**off-the-shelf tooling on a size-identical model** (no ASR code written): stock
`llama-batched-bench` (Homebrew llama.cpp b9690, **Metal**) on a stock **Qwen3-0.6B
Q8_0** LLM — whose decoder is identical to the qwen-asr decoder (1024 hidden, 28
layers, 16/8 GQA heads, 128 head_dim). Flash-attention on, `-ngl 99`, PP=448, TG=256:

| batch | decode t/s (total) | per-window t/s | prefill t/s |
|------:|-------------------:|---------------:|------------:|
| 1   | 247   | 247 | 12 325 |
| 4   | 715   | 179 | 12 491 |
| 16  | 1 232 | 77  | 12 196 |
| 64  | 3 009 | 47  | 12 369 |

**ggml's batched decode works and scales — 12× from batch 1→64** (247→3 009 decode
t/s), the **same order as the MLX ceiling** (3 009 vs 4 610 with attention) and
~60–75× the CPU decoder (~40 t/s/window). Prefill is ~12 k t/s regardless of batch.
Per-sequence latency *drops* with batch (247→47 t/s/window) — the throughput/latency
trade — which is exactly right for **offline batch processing of 6-hour files**, wrong
for low-latency streaming.

Worked estimate for one 6-hour file (~720 windows × ~256 gen tokens ≈ 184 k decode
tokens): CPU ≈ **~77 min** of decode (batch-invariant); ggml/Metal batch-64 ≈
184 k/3 009 ≈ **61 s** decode + ~26 s prefill ≈ **~90 s** — i.e. the dominant decode
path goes from over an hour to ~1 minute.

**Verdict: the ggml path is de-risked.** Its batched decode delivers the throughput
that justifies the port; it's portable (Metal + CUDA + CPU from one source), and Q8_0
keeps it int8 (closest to the reference). The **remaining work is building the
Qwen3-ASR audio-encoder graph in ggml** (the decoder is proven here; the encoder is
the only model piece ggml lacks). NVIDIA: run the same `llama-batched-bench` with a
CUDA build on a 3090 to get the matching number (expect higher per-step compute,
batch capped ~32–64 by 24 GB VRAM). Reproduce: `brew install llama.cpp` then
`llama-batched-bench -m Qwen3-0.6B-Q8_0.gguf -c 46080 -ngl 99 -fa 1 -npp 448 -ntg 256
-npl 1,4,16,64`.

### Reproduce
```
# build both
( cd ../qwen-asr && RUSTFLAGS="-C target-cpu=native" cargo build --release )
RUSTFLAGS="-C target-cpu=native" cargo build --release

M=../qwen-asr/qwen3-asr-0.6b
ORIG=../qwen-asr/target/release/qwen-asr
BETA=target/release/qwen-asr
$ORIG -d $M -i clip.wav --language Chinese -t 1 --silent --json > ref.json
$BETA -d $M -i clip.wav --language Chinese -t 1 --silent --json --batch 16 > beta.json
cmp ref.json beta.json && echo IDENTICAL
```

## Known limitations / future work
- **Encoder + prefill are per-window** (not batched). They use BLAS/AMX f32 and are
  a one-pass cost; batching them would need a bit-identical stacked-M BLAS call
  (must be test-gated — Accelerate is not guaranteed identical across M). Candidate
  next win if encode/prefill becomes a larger fraction after the decode win.
- **Speedup is bandwidth-bound, not large by default.** Batching cuts weight memory
  traffic, not compute (same `sdot` count), so on an uncontended M3 Ultra it is only
  ~7–9% on healthy content (knee at B≈4; see the sweep). It grows under
  memory-bandwidth pressure. Beating it meaningfully means cutting *compute*
  (AMX f32 / int4 / etc.), which changes arithmetic and breaks bit-identicality.
- **Degenerate-heavy content** is recovery-bound; batching is ≈parity there.
- Batched path is **aarch64-only** for now (INT8 NEON); other arches fall back to
  sequential automatically.
- Default is **off** (`--batch 0`); opt-in. Not yet wired into the mrecord coproc
  qwen worker launch (would pass `--batch 16` to the `--serve` invocation).

## Reference branches (local, not pushed)
Snapped in the `qwen-asr-beta` submodule at each verified bit-identical milestone:
- `progress/2026-06-17-235457-batched-16way-bitident` — batched decode + const-generic kernels
- `progress/2026-06-18-095101-early-abort-bitident` — + degenerate early-abort

`main` carries the changes **uncommitted** for review (8 files + new `batch.rs`).

## Map of changed files
- `crates/qwen-asr/src/batch.rs` — **new**, the batched orchestration
- `crates/qwen-asr/src/kernels/neon.rs` — batched INT8 kernels + const-generic tiles
- `crates/qwen-asr/src/kernels/mod.rs` — public dispatch wrappers
- `crates/qwen-asr/src/transcribe.rs` — exposed `pub(crate)` helpers
  (`find_split_point`, `should_insert_boundary_space`, `transcribe_with_recovery`,
  `segment_is_degenerate`, the prompt-token consts)
- `crates/qwen-asr/src/context.rs` — `DecodeSettings.batch_size` + `QwenCtx.batch_size`
- `crates/qwen-asr/src/lib.rs` — `pub mod batch;`
- `crates/qwen-asr-cli/src/main.rs` — `--batch` flag, routing, help
- `crates/qwen-asr-cli/src/serve.rs` — route on `ctx.batch_size`
