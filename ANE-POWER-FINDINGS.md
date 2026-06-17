# qwen-asr `--mac-ane` power-efficiency investigation — findings & resume notes

Branch: `mac-ane-power` (qwen-asr submodule), built on top of the earlier
`main_qwen_asr_experiment_failed_ane` work. Investigation dates: 2026-06-17.
Hardware: **Apple M3 Ultra, 28 cores, 96 GB, macOS 26.5.1**. All numbers at
**thread=1** unless noted (production runs many single-threaded workers).

Companion docs: `ANE-ACCELERATION-FINDINGS.md` (prior 2× speed attempt),
`mac-ane-poc.md`.

---

## TL;DR (the headline)

1. **`--mac-ane` works — the encoder GEMMs really do run on the ANE** (confirmed
   ~40% ANE via `mactop`; the encoder is offloaded). But the ANE carries
   **negligible energy** (avg ~0.02 W, peak 0.09–0.88 W) vs the **CPU at ~50 W**.
2. **There is no power win and no reliable speed win from the ANE.** The decoder
   (~80 % of compute — int8 `seq=1` matvec, memory-bound, **ANE-hostile**) stays
   on the CPU and dominates. The only ANE-amenable work is the encoder (~20 %),
   too small to matter, and its CPU-side support (transpose, conv-stem, dispatch)
   eats the saving. **The goal "power efficiency by shifting work to the ANE,
   ≥1.1× speed" is not physically achievable on this decoder-dominated model.**
3. **Correctness of the ANE path is fine**: batched-ANE vs batched-CPU = **98.3 %
   identical** (fp32 IO; cosmetic argmax flips only).
4. **Real, ANE-independent speed finding** (on clean / VAD-like speech): **smaller
   blocks are faster** — 30 s decodes fastest, throughput falls monotonically as
   blocks grow (decoder KV-context cost dominates). 30 s = 10.8× RT, 150 s = 4.7×.
5. **⚠️ MAJOR CAVEAT — no VAD was applied.** All runs fed *raw* audio (music,
   songs, jingles) straight to qwen-asr. **Production applies VAD first** (sends
   only speech `regions`). The whole degeneracy / stuck-block / loop-recovery
   thread below was triggered by non-speech audio production never sees. Re-run
   with VAD before trusting those parts.

---

## Goal (as it evolved across the session)

Original: "shift as much qwen-asr work as possible onto the ANE, gated by
`--mac-ane`, for **power efficiency**, while **maintaining current speed**;
translate 60-min blocks in 5 languages." Refinements added live:

- ≥1.1× speed became a **requirement** (not bonus).
- Parallelize everything; segments are independent.
- 60/90 s segments OK if no repetition.
- Optimize **throughput, not latency**.
- Measure at **process=1, thread=1** on 30-min blocks (later 15-min) — simulating
  production (many single-threaded processes sharing one ANE).
- Last sub-goal: grid of K∈{1,2,4,8,16} chunks × {30,60,90}s, batched into the ANE.

---

## What was built (working tree on `mac-ane-power`, UNCOMMITTED)

All changes are gated/additive; the CPU path is unchanged when `--mac-ane` /
`QWEN_ANE_BATCH` are absent.

- **`crates/qwen-asr/src/mac_ane.rs`**
  - **seq-bucketing** (`seq_bucket`, env `QWEN_ANE_SEQ_BUCKET`, default 16): rounds
    the GEMM `seq` up to a bucket so the compiled CoreML model is **reused across
    segments** instead of recompiled each one (the prior branch's real-seq 22s
    regression was per-segment recompile). Bit-exact (rows are independent).
  - **vDSP transpose** (`vDSP_mtrans`): replaced the scalar `[seq,in]↔[out,seq]`
    pack/unpack (the CPU work feeding the ANE) with Accelerate's SIMD transpose.
- **`crates/qwen-asr/ane/ane_shim.m`** — made `qwen_ane_run` **re-entrant &
  zero-copy**: wraps caller buffers via `initWithDataPointer` + `outputBackings`
  (no per-call alloc, no shared mutable IO buffers), so **N worker threads can
  share one compiled model concurrently**. Added `contiguous_strides` + IO geometry
  capture in `qwen_ane_create`.
- **`crates/qwen-asr/src/encoder.rs`**
  - Extracted **`conv_stem_forward`** (one segment's Conv2D stem → post-stem seq).
  - Added **`forward_batch`**: encodes **K independent segments at once**, batching
    every per-position GEMM (q/k/v/out, FFN, conv_out, proj1/2) over the
    concatenated `[Σseqᵢ, d_model]` sequence — so the ANE sees one large `seq`
    (≈K·366) instead of K small ones (its efficient regime, K× fewer dispatches).
    Attention stays strictly per-segment. Numerically exact vs per-segment forward.
- **`crates/qwen-asr/src/transcribe.rs`**
  - **`QWEN_ANE_BATCH=K`** path in `transcribe_audio`: a background worker
    `forward_batch`-encodes K segments and streams encodings to the main thread,
    which decodes them (encode‖decode overlap). Past-text conditioning preserved
    on the (sequential) decode side.
  - **Loop-recovery made size-bounded** (see ⚠️ below — candidate for revert).
- **`crates/qwen-asr-cli/src/main.rs`** — `--mac-ane` is now the **production
  offload flag** (enables encoder ANE offload during transcription). Old benchmark
  moved to `--mac-ane-bench`. Help text updated.
- **`crates/qwen-asr/src/context.rs`** — `loop_max_depth` default 3 → **12**
  (⚠️ candidate for revert, see below) + test assertion.

---

## Measurement methodology (reusable, in `/tmp/ane-bench/`)

- **Power**: `powermetrics` ANE *milliwatt* rail is **unreliable on M3 Ultra**
  (reads 0 even for WhisperKit). Use **`mactop --headless --format json`**
  (userland, IOReport) — `soc_metrics.ane_power` / `cpu_power`. The *average* ANE
  power washes bursts to ~0; for utilization use your on-box `mactop` (showed ~40%).
  CPU power is the dependable energy proxy (ANE is genuinely low-power).
- `powermetrics` needs root via `MAC_M3_STUDIO_PASSWORD` from `.env` (`sudo -S`),
  per `s/monitor-gpu.sh`. mactop does not.
- Test audio: 5 foreign-lang 30-min blocks pulled from the Storage Box via
  `s/query-storagebox.sh get` (STORAGEBOX_* in `.env`) — France-Inter (fr),
  Deutschlandfunk (de), Al-Jazeera-Arabic (ar), NHK-R1 (ja), YTN-Radio (ko) — plus
  English `bench/samples-compare/broadcast119.wav`. Converted to 16 kHz mono wav.
- Scripts written: `bench.py` (serve concurrency), `powerbench.sh`, `lang_run.sh`,
  `table5.sh`, `grid.sh`, `knee.sh`/`knee2.sh`, `quality.sh`.

---

## Key data

### A) The ANE is used but carries no energy (mactop, sustained load)
```
qwen --mac-ane batched:  max ANE 0.41W  avg ANE 0.02W  max CPU 53.2W
WhisperKit (base):       max ANE 0.00W  avg ANE 0.00W  max CPU 10.0W  (CPU low ⇒ on ANE)
```

### B) Single 30-min stream (per-worker), thread=1
| Config | Wall | Speed | CPU power | Energy |
|---|---|---|---|---|
| CPU serial (1 thread) — baseline | 124 s | 1.0× | 9.8 W | 1220 J |
| CPU pipeline (2 thread) | 99.5 s | 1.25× | 11.5 W | 1145 J |
| ANE batch K=4 (2 thread) | 108 s | 1.15× | 11.4 W | 1233 J |
*1.15–1.25× is the encode‖decode **overlap**, available on pure CPU too; ANE adds nothing.*

### C) Saturated machine (16 threads)
| Config | Throughput | CPU power | J/audio-s |
|---|---|---|---|
| 16× CPU serial | 69× RT | 58.4 W | 0.85 |
| 8× ANE batch-K4 | 31× RT | 35.7 W (−39%) | 1.16 |
*ANE cools the CPU but the shallow offload bottlenecks the ANE → worse J/audio.*

### D) 5-language A/B (5-min, thread=1): CPU-serial vs batched-ANE
| Lang | CPU-serial | batched-ANE | Speedup | ANE peak | ANE==CPU |
|---|---|---|---|---|---|
| French | 74.6s/10.0W | 77.9s/10.2W | 0.96× | 0.10W | 98.34% |
| German | 30.0s/9.8W | 32.6s/10.0W | 0.92× | 0.57W | 99.44% |
| Arabic | 30.0s/9.6W | 32.3s/11.3W | 0.93× | 0.09W | 80.82%* |
| Japanese | 28.0s/9.2W | 30.2s/11.3W | 0.93× | 0.09W | 97.60% |
| Korean | 32.1s/9.6W | 34.4s/11.2W | 0.93× | 0.09W | 95.68% |
*ANE consistently 0.92–0.96× (slower) and equal-or-higher CPU power. Arabic 80%
and the diffs are batched-vs-serial path differences, not the ANE.*

### E) ANE batching grid (15-min French, batched into ANE) — ANE never loads
ANE peak stayed **0.09–0.88 W across all K∈{1,2,4,8,16} × {30,60,90}s**; throughput
noise-dominated; CPU power flat ~8–11 W. Conclusion: cannot "give the ANE work to do."

### F) Block-size speed knee — German 15-min, **clean speech**, guard ON, thread=1
| Block | 30 | 45 | 60 | 75 | 90 | 120 | 150 | 180 |
|---|---|---|---|---|---|---|---|---|
| xRT | **10.8** | 9.4 | 8.2 | 7.3 | 6.5 | 5.6 | 4.7 | timeout |
| vs30 | 1.00 | 0.87 | 0.76 | 0.68 | 0.60 | 0.52 | 0.44 | — |
**Smaller = faster; monotonic; no knee favoring larger blocks. The decoder is the wall.**
(French had shown a spurious "60s=1.35× faster" — that was the song-loop artifact, see ⚠️.)

### G) Transcript quality 30s vs 60s, guard OFF (`-i` text) vs ON (`--json`)
| Lang | OFF 30s | OFF 60s | ON 30s | ON 60s |
|---|---|---|---|---|
| French | **0.667/1885** ⚠️ | 0.068/681 | 0.057/683 | 0.068/682 |
| German | 0.003/599 | 0.003/599 | 0.003/600 | 0.003/600 |
*(degeneracy score / word count). The split-in-half guard rescues the French song
loop (0.667 → 0.057). Guard ON ⇒ 30s ≈ 60s quality.*

---

## Anti-repetition guard (`transcribe_with_recovery`) — important mechanics

- The guard (halve a degenerate block at a word boundary, re-decode) is **ON only
  in the segmented path**: `transcribe_segmented` / `transcribe_clips` → used by
  **`--serve` (production)**, `-i --json`, `-i --srt`.
- It is **OFF in plain `-i` text mode** (`transcribe_audio`, line ~1156) — which
  the A/B and grid used. So those repetition numbers are worst-case.
- Original behavior was **depth-bounded** (`loop_max_depth=3`) + size floor
  `2·loop_min_split_sec` (16 s). A 75 s degenerate block hit depth 3 at ~16 s,
  still inside the loop → effectively stuck (decoder spins to `max_tokens=2048`).

### ⚠️ The `loop_max_depth=12` / size-bounded change is suspect — REVERT candidate
I changed recovery to **size-bounded** (split until <8 s, depth backstop 12). It
**did not fix French 75 s — it made it worse**: for *un-recoverable* content (a real
song), every split-half is also degenerate, so it recurses all the way and
**re-decodes every level to the 2048-token cap → exponential blowup** (75 s → ~16
sub-blocks × 2048 tokens). The old depth=3 cap was *protecting* against this.
**The real fix is mid-decode loop early-stop** (detect the repetition during
generation and stop, like the streaming path), not more splitting — then small
blocks are cheap and any splitting is cheap. Recommend reverting `loop_max_depth`
to 3 and pursuing early-stop instead.

---

## ⚠️ The VAD confound (must address before trusting degeneracy results)

`skip_silence=false` by default and no `--clip-timestamps` was passed → **raw audio
fed directly**, including music/songs. **Production VADs first** (coproc worker
sends speech `regions`; qwen-asr decodes only those via `transcribe_clips`). So:

- French song-loop degeneracy, stuck-75 s, exponential blowup → **artifacts of
  non-VAD input**; production never sees that audio.
- "60 s = 1.35× faster" → **artifact** (song loop at 30 s).
- German clean-speech speed curve → **valid** (talk radio ≈ VAD-clean).

---

## Open items / how to resume

1. **Revert `loop_max_depth` 12 → 3** in `context.rs` (+ test assertion) — the
   size-bounded recovery is harmful for un-recoverable content.
2. **Re-run the block-size knee WITH VAD** (run the coproc VAD pre-pass → feed
   `--clip-timestamps`/`--serve regions`) to confirm "smaller-is-faster" holds
   production-accurately and quantify degeneracy on real speech only.
3. (Optional) **Mid-decode loop early-stop** in `transcribe_segment` — caps the
   2048-token spin on degenerate blocks; the correct fix for repetition cost.
4. **Decision on the goal**: power-efficiency-via-ANE is not achievable here
   (decoder-dominated, ANE-hostile). The only real ANE lever would be **batched
   cross-stream decoding** (decode N streams' matvecs as one ANE matmul) — major
   rearchitecture, fp16-accuracy-risky, research-grade. Otherwise: drop the ANE
   angle; the actionable speed lever is block size / decoder, not the ANE.
5. Investigate **Japanese** producing ~4–12 words / 5 min (likely the NHK clip is
   mostly music, or a routing issue) — orthogonal to the above.

## What would actually work — batched cross-stream decoding (the real lever)

### Why the ANE accelerates the encoder but not the decoder — it's the M dimension
Same model family, same weights; the only structural difference is the batch
dimension **M** of each `[M, hidden] @ W` op:

| | M (batch) | shape | nature | ANE? |
|---|---|---|---|---|
| Encoder | seq ≈ 366 (whole segment at once) | `[366, h] @ W` | **matmul** (high arithmetic intensity) | ✅ amenable |
| Decoder (today) | **1** (autoregressive, one token) | `[1, h] @ W` | **matvec** (~1 FLOP/byte, memory-bound) | ❌ |
| Decoder (cross-stream batched) | **N** streams | `[N, h] @ W` | **matmul** | ✅ amenable |

A matvec reads the entire weight matrix to produce one output vector → bound by
**memory bandwidth**, and the ANE shares the same unified-memory bus as the CPU,
so moving it there cannot help. The encoder gets its M for free from the seq axis;
the decoder must get its M from **N independent audio streams decoded in lockstep**.

### Why the encode‖decode overlap is a latency trick, not throughput
Total CPU work per block = `encode + decode` regardless of scheduling.
- Serial (1 thread): wall = encode+decode; **CPU-seconds = encode+decode**.
- Overlap (2 threads): wall = max(encode,decode) ≈ decode (the ~1.25×), but two
  cores are busy → **CPU-seconds = encode+decode**, *identical*.

At production **saturation** (all cores pinned by single-threaded workers),
aggregate throughput = core_capacity ÷ CPU-seconds-per-block, and CPU-seconds is
unchanged → **the overlap adds zero aggregate throughput**. It only helps when
cores are idle (the coproc never is). The coproc is throughput-bound, so the
overlap — and likewise the encoder ANE offload — net to ~0 in production.

### The one lever that survives saturation: lower CPU-seconds-per-token
Two ways: (1) smaller blocks (less KV context → cheaper decode — the German knee,
real but modest), or (2) **batch the decoder across streams**. The coproc *already*
runs ~16 concurrent single-threaded decode workers, each re-reading the full int8
weight set per token (16 independent matvecs). Batching them reads each weight
matrix **once per 16 tokens** → ~**16× less memory traffic per token** until
compute-bound (crossover ≈ M=10–25 on this machine, so 16 is well-chosen). This:
- **raises throughput** several× at saturation (genuinely fewer CPU-seconds/token),
- **lowers power** (DRAM traffic is the dominant energy term) — the ORIGINAL goal,
  achieved by cutting the real cost, not by chasing the ANE,
- makes the decode a **compute-bound matmul** that *can then* go to the ANE (optional),
- **preserves correctness** — same int8 math, just batched (no fp16 argmax flips).

This is standard high-throughput LLM serving (vLLM / continuous batching) applied
to the coproc's already-concurrent streams.

### Empirical proof the decoder is bandwidth-bound — and batching is the fix
Measured per-worker throughput collapses as concurrent workers are added (CPU,
thread=1):

| Concurrent workers | Aggregate ×RT | Per-worker ×RT |
|---|---|---|
| 1  | 14.1× | **14.1×** |
| 4  | 49.3× | **12.3×** |
| 16 (saturation) | 69× | **4.3×** |

A worker does ~14× realtime alone but only ~4.3× when 16 run together — a **~3×
collapse**, with cores *not* oversubscribed. That is the textbook **memory-bandwidth**
signature: all 16 workers independently re-read the full int8 weight set every token,
saturating the ~819 GB/s bus, so the cores starve. Batching reads each weight matrix
**once per 16 tokens** → removes the bus wall → per-worker rate climbs back toward
~14× → aggregate goes from **69× toward ~150–190×** (compute-bound ceiling), i.e.
**~2–3× the current saturated CPU throughput**, plus the matching DRAM-energy drop.
This is the concrete justification for the whole batched-decode effort, on hardware
already in place, with no ANE and bit-exact output.

## Concrete plan — batched cross-stream decode

### Existing hooks in this codebase
- `decoder::decoder_forward_batch` ALREADY EXISTS (from the lookahead/Jacobi
  experiment): it batches **positions of one stream** as `[wlen, h] @ W` — i.e. the
  batched-forward plumbing is largely there. Two gaps: it batches positions not
  streams, and it uses a **bf16/f32 GEMM**, not int8 (the prior findings flagged
  "a proper batched-int8 GEMM kernel would be needed"). The plan fills both.
- `--serve` already holds N concurrent decodes via `CtxPool` (serve.rs) — the N
  streams to batch are already in the process.

### Threading model — per-core batching (preferred: simplest, same win)
The bandwidth win is **intra-core**: a single worker that processes M streams reads
each weight matrix **once and reuses it across M tokens** — so escaping the memory
wall needs **no cross-core cooperation**. Keep today's independent single-threaded
`--serve` workers (`CtxPool`, `set_threads(1)`); just make **each worker batch M≈16
streams** locally. 16 workers × M=16 = 256 streams in flight, ~same total DRAM
traffic as today's 16 single streams, but ~**3–4× the throughput** (now compute-bound,
not memory-stalled). No global scheduler, no multi-threaded GEMM, no thread-pool
changes — the simplest path to the win.

| Per-core M | streams in flight | DRAM read / token | regime |
|---|---|---|---|
| 1 (today) | 16 | full weights | memory-bound (~4.3×/core) |
| 4 | 64 | weights ÷ 4 | partial (under crossover) — simple first cut |
| **16** | 256 | weights ÷ 16 | **compute-bound → full ~3–4×** |

NEON is 128-bit (4 int32 lanes), so M=4 maps straight onto a register; M=16 is
reached by register-tiling (`smmla` tiles M in 2s). The width sets the *tile*
granularity, not a batch ceiling.

*(A cooperative model — all cores on ONE batch, sharing the weight read at cache
level — is an alternative, worth it only if you stay bandwidth-bound after batching.
Once M≈16 makes each core compute-bound, it buys nothing extra and adds plumbing.
Prefer per-core.)*

### The batched int8 GEMM kernel (the one genuinely new piece)
Generalize `neon.rs::matvec_int8` (M=1, uses `sdot_s32`) to `matmul_int8`
`[M, in] @ int8[out, in] → [M, out]`. **Use `smmla` (FEAT_I8MM = YES on this M3
Ultra)** as the inner instruction, not `sdot`: `smmla` is an 8-bit **2×8·8×2 → 2×2**
integer matrix-multiply-accumulate ≈ **2× the int8 throughput of `sdot`**, and still
**bit-exact** (int32 accumulate). `sdot` was right for the M=1 matvec; the batched
matmul is exactly the shape `smmla` wants — keep `sdot` as the M=1 fallback.
Specialize per batch with **const generics** (`matmul_int8::<M>`) + a runtime
dispatch. Add batched variants of `kernels::linear_nobias_int8_qkv`,
`linear_nobias_int8_swiglu`, `linear_nobias_int8_addto`, and `argmax_matvec_int8`
(→ per-row argmax over `[M, vocab]`). lm_head `[M, h] @ [h, 151936]` is the biggest
single win. (Apple has no 512-bit vectors / no SVE / no SME on M3 — NEON is 128-bit;
wide throughput comes from ~4 NEON pipes/core + `i8mm`/`bf16` matrix instructions +
the private AMX coprocessor via Accelerate. SME is M4+.)

### Phase 0 — per-core static batching (de-risk), fixed M, padded
Each worker holds a fixed M (start at 4 to validate, then 16), padded, decoded to
EOS together. Crude (wastes the tail when a worker's streams desync) but proves the
kernel + the win with minimal change and **no scheduler**. Per worker:
1. **Batched decode step** `decoder_forward_batch_streams(layers, &mut [KvCache; M],
   rope, bufs, &[input_embed; M]) -> [i32; M]`. Per layer: rms_norm / RoPE / SwiGLU
   per-row; QKV/O/MLP/lm_head via the batched int8 GEMM; **attention stays per-stream**
   (each KvCache its own length — loop M small attentions, the unbatched ~10%).
2. **DecoderBuffers** widened to M rows (`bufs.x` → `[M, dim]`, etc.).
3. Driver: a batched analogue of the `transcribe_segment` decode loop advancing M
   streams, handling per-stream EOS / loop-detect / sampling.

### Phase 1 — per-worker continuous batching (the real throughput)
Within each worker, keep its M-batch full: when one of its streams hits EOS, **evict
it and admit the next block** (ragged positions per stream). This is local to the
worker — **no cross-core coordination** — and removes Phase-0's padding waste. Wire
it into the `--serve` request loop so each worker pulls the next block as a slot
frees, instead of one-block-per-connection.

### Throughput / power estimate (validate, don't trust blindly)
Decode is memory-bound at ~1 FLOP/byte (int8). Batching M streams cuts weight
memory traffic per token ~M× until compute-bound. Expect a **large multiple** of
single-stream aggregate throughput up to the M3 Ultra compute ceiling (crossover
~M=10–25); DRAM energy per token drops similarly. **Prototype the batched int8 GEMM
and measure tokens/s vs M (1,2,4,8,16,32)** to find this machine's crossover before
committing to the scheduler — that one microbenchmark de-risks the whole effort.

### Correctness — batched decode is bit-exact by construction
The key reason this is tractable: a batched GEMM `[N, in] @ W` computes each output
element `(stream r, out o)` as `Σₖ x[r,k]·w[o,k]` — the reduction is over **K, not
N**, so each stream's row depends only on its own input (row-independent; batch-mates
can't perturb it). And the decoder is **int8**: the dot product accumulates in
**int32 — exact integer math**, which is **associative**, so the K-reduction order
(tiling/vectorization of the batched kernel) is irrelevant — the int32 accumulator is
**identical** to the M=1 matvec. The only float step is the final `× scale` per
output element. So batched int8 decode is **bit-identical** to single-stream — unlike
the fp16 ANE path, which changes the numerics and flips ~1–2% of argmaxes.

| Decoder component | batched as | bit-exact? | why |
|---|---|---|---|
| QKV / O / MLP / lm_head (int8) | `[N,h] @ W` | ✅ | int32 accumulation, order-independent |
| RMSNorm / RoPE / SwiGLU | per-row | ✅ | elementwise; same per-row math |
| Attention | per-stream | ✅ | identical to single-stream (own KV) |
| Argmax (greedy) | per-stream over `[N, vocab]` | ✅ | deterministic; ties broken identically |
| Encoder | left **per-stream** (unchanged code) | ✅ | already a matmul via seq; no need to cross-stream batch |

**Probability of 100% correctness**: functionally identical (WER-identical) ~95%;
*bit-exact* ~85–90% — achievable and worth targeting, the only "tax" being discipline
(don't reorder any **f32** reduction — attention/norms must stay per-stream; one
sloppy f32 batch or a state-isolation bug drops you to "functionally identical").

**Enforcement: the single-stream path is a perfect oracle.** Every failure mode here
is a *bug*, not a numerical inevitability — so test it directly: run the same audio
single-stream and in a batch and `assert tokens_batched[r] == tokens_single` (also a
unit test `matmul_int8::<N>` row r == `matvec_int8` for that row). That oracle makes
"100% correct" an enforceable invariant, not a hope.

### Where the ANE fits — structurally yes, but it forces a bad trade (probably skip)
Once decode is a compute-bound batched matmul, the ANE *can* run it. But the ANE is
**fp16** and the decoder is **int8**, and that mismatch poisons all three angles:
1. **It breaks the bit-exactness** above (int8→fp16 reintroduces argmax flips) — you'd
   trade *guaranteed* correctness for a speed *bet*.
2. **The CPU int8 batched GEMM is a high bar.** Compute-bound int8 has ~2× the
   arithmetic density of fp16 and `sdot` is efficient; the ANE would do 2× the work in
   fp16 *plus* a dequant (2× weight memory). Not obviously faster, and only "wins" lossy.
3. **Per-layer CPU↔ANE ping-pong.** Attention stays per-stream on CPU (ragged KV), so
   every layer is CPU-attn → ANE-proj → CPU-attn… 28 layers × per-token-step =
   constant dispatch + data movement. The encoder pays this once per forward pass; the
   decoder would pay it every step.

**Conclusion:** batch **int8 on the CPU** — that cuts DRAM traffic (faster + lower
power, the original goal) and stays bit-exact. The ANE is a red herring even after
batching: the CPU int8 is already compute-bound-fast, and the ANE only "helps" by
going fp16 and breaking correctness. Treat ANE as a *measured* experiment, expecting
it to be marginal-and-lossy — not the goal.

### Risks
- Batched int8 GEMM must stay efficient at M=16 (current kernels are tuned for M=1);
  the microbenchmark settles this.
- Continuous-batching scheduler complexity (ragged positions, eviction/admission).
- KV-cache RAM grows with N (N separate caches) — bounded, fits in 96 GB.
- Accuracy is preserved (same int8 math); no fp16 risk, unlike the ANE encoder path.

### Scale
Weeks of work — it's building a mini batched-inference engine for qwen-asr. But it
is the **only** approach that attacks the 80%, it serves throughput AND the original
power goal, and it is proven, well-trodden territory.

## Reproduce
```bash
cd process_asr/qwen-asr
cargo build --release --features mac-ane -p qwen-asr-cli
BIN=./target/release/qwen-asr MODEL=qwen3-asr-0.6b
# ANE offload (batched), one process:
QWEN_ANE_BATCH=4 $BIN -d $MODEL -i clip.wav --language French -S 30 -t 1 --mac-ane
# ANE utilization (your tool): mactop --headless --format json --count 25 -i 200
# block-size knee: /tmp/ane-bench/knee2.sh  ;  quality A/B: /tmp/ane-bench/quality.sh
```
