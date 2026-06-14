# Loop / repetition detection & recovery

## The problem

Qwen3-ASR, like most autoregressive decoders, **degenerates into repetition loops when it
decodes too long a span at once** — it gets stuck emitting the same phrase over and over
until it hits the token cap. A single 30 s+ span of dense continuous speech (no silence to
break it) can produce thousands of characters of `…أو نتحدث عن الجانب الخاص بال. أو نتحدث عن
الجانب الخاص بال…`. The CLI help has always warned about this; `-S 30` exists precisely to
cap span length.

Two failure modes follow:
1. **Slowness** — a looping segment runs the full token budget (~2048) instead of stopping at
   EOS (~50-200 tokens), so it costs 10-40× a healthy segment and turns a block into a
   multi-hour "stuck" decode.
2. **Garbage output** — the repeated text is useless, and downstream consumers that size
   buffers from cue length can OOM on the monster cue.

The `--stream` path already defends against this (ported from antirez's C): a tail
repeat-block detector (`stream_tail_repeat_blocks`) + prefix rollback + a single-token-run
guard. **The offline/segment path — used by one-shot `-i` and by `--serve` (which is what
production runs) — only *observed* the degeneracy** (`perf_maxed_segments`, logged
`DEGENERATE`) and took no corrective action.

## What we measured

- The loop is **driven by span length**, not content: the *same* audio that loops as a 33 s
  span transcribes cleanly when split to ≤16.5 s. (33 s → 7662-char loop; 2×16.5 s → clean.)
- It overran to 33 s because the silence-seeking splitter (`-W 3`) couldn't find a gap in a
  dense speech run and extended past the 30 s target. So a *soft* 30 s cap isn't enough.
- The looping phrase is **longer than 6 tokens**, so the stream detector's tuned
  `period ≤ 6` would not catch it even if it ran in the offline path.

## The fix (two layers)

### 1. Detection — `transcribe_segment`

A segment is flagged **degenerate** when either:
- it ran to the token cap without EOS (the maxed-token signal — high confidence), **or**
- its tail is a repetition loop per `is_repetition_loop`: a block repeated `≥ loop_min_repeats`
  (4) times, found by reusing antirez's `stream_tail_repeat_blocks` **at the wider
  `loop_max_period` (32)** so it sees sentence-length phrase loops the stream path's period-6
  misses, **AND** that repeat covering at least **half** the decoded tokens.

The coverage gate is the **false-positive guard**: a runaway loop *dominates* the output (the
repeat is ~the whole tail), whereas legitimate brief repetition — a sung refrain, a chant, a
repeated list item — is only a small fraction of the segment, so it is **not** flagged. The
maxed-token signal stays unconditional (it's near-impossible to hit without looping). Token-level
(not text), so it's language-agnostic and exact.

### 2. Recovery — `transcribe_with_recovery` (iterative, not recursive)

On a flagged segment, **halve it at a word boundary and re-decode each half**, repeating
until clean. The split point is `find_split_point(midpoint, ±search_sec)` — the lowest-energy
point near the centre — so the cut lands **between words**, never mid-word (which would
corrupt the ASR; this is why a blunt hard `-W 0` cut was rejected).

It's driven by an **explicit work-stack, not call recursion** (a `Vec<(start,end,depth)>`;
pop → decode → push the two halves or emit; emitted pieces sorted back into time order).
Bounded by:
- `loop_max_depth` (3): at most 3 halvings (caps larger spans; a typical ≤33 s segment stops
  at the floor first — `32 → 16 → 8`).
- `loop_min_split_sec` (8 s): only split a span that is at least 2× this, so short segments
  are never over-split. It gates *whether* to split, not the exact half sizes — the cut
  follows the word gap (lowest-energy point near the midpoint), so halves can be uneven and
  one may fall below this; that half just isn't split again. (A clean word-boundary cut
  matters more than an exact size — forcing the size would push the cut mid-word.)

Mid-word cutting can only happen on a span that is **already pure garbage** (a confirmed
loop) and only when no silence gap exists — at worst one boundary word is clipped versus the
entire span being a loop. Healthy audio is never touched: with `--loop-detect off` nothing is
ever flagged, so decode is byte-for-byte the legacy behavior.

## Stream-path sync to the C reference

The Rust port had **dropped** antirez's `QWEN_STREAM_MAX_REPEAT_TOKEN_RUN` guard (suppress a
single token repeated > 12× in a row at the stream commit). It's restored here
(`suppress_repeat_token_runs`, `--loop-max-token-run`, default 12), applied at both stream
commit sites, so the Rust `--stream` path matches the C again.

## CLI / serve parity

All knobs live in **one** `DecodeSettings` table (`context.rs`), consumed by `from_model`,
the CLI, `--serve`, and the `--help` strings. `main` builds the settings **once** (defaults +
flag overrides) and passes the *same* value to both the one-shot path and `serve::run`, so
**`--serve` and the CLI can never decode differently** for the same flags — including every
`--loop-*` knob and `-S`/`-W`. (This closes the original divergence where `--serve` ignored
the 30 s clipping.) `loop_defaults_are_gold_standard` locks the defaults in CI.

## Options and constants

The detection **thresholds are fixed constants** (in `transcribe.rs`), not runtime flags — the
stream path proves hard-coded values are right, and nobody needs to tune a degeneracy detector
per run:

| constant (`transcribe.rs`) | value | path | role |
|----------------------------|-------|------|------|
| `STREAM_DEGEN_MAX_PERIOD` / `STREAM_DEGEN_MIN_REPEATS` | 6 / 4 | stream | C-reference tail-repeat detector |
| `STREAM_MAX_REPEAT_TOKEN_RUN` | 12 | stream | single-token-run suppression cap |
| `SEGMENT_DEGEN_MAX_PERIOD` / `SEGMENT_DEGEN_MIN_REPEATS` | 32 / 4 | segment | wide-period phrase-loop detector |

Only three things are CLI flags — the on/off switch (both paths) and the segment-recovery
bounds (segment path only):

| flag | default | applies to | purpose |
|------|---------|------------|---------|
| `--loop-detect <yes\|no>` | yes | both | master switch for all detection + recovery |
| `--loop-min-split-sec <s>` | 8 | segment | recovery size floor — only split spans ≥ 2× this (shortest recovered segment = this) |
| `--loop-max-depth <n>` | 3 | segment | recovery depth cap — e.g. 32→16→8 (stops at the 8s floor) |

In short: `--loop-detect` is the only switch that touches both; `--loop-max-token-run` is the
only **stream** tunable; the remaining four are **batch**-only. (The stream path's `period 6 /
min-repeats 4` are intentionally not exposed — they mirror the C reference and stream's short
~8 s chunks rarely loop in the first place.)

## Tests

`crates/qwen-asr/src/transcribe.rs::loop_tests` and `context.rs::tests`:
- phrase loop detected at period 32, **missed at period 6** (proves the wider-period need);
- clean text not flagged;
- single-token-run suppression (incl. run continued from the prefix tail; no-op when off);
- `find_split_point` lands inside a silence gap (word-boundary split);
- loop defaults locked; `DecodeSettings` parity.

The end-to-end "audio that loops" case is deliberately **not** a CI test: the loop is
alignment/context-sensitive and does not survive extraction to a standalone clip (verified
across 33 s/42 s/90 s extracts), so it would be flaky. The deterministic token/sample-level
tests above cover the same logic reliably.
