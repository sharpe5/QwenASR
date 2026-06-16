//! Offline, segmented, and streaming transcription orchestration.

use crate::audio;
use crate::config::*;
use crate::context::{QwenCtx, QwenModel};
use crate::decoder::{self, tok_embed_bf16_to_f32};
use crate::encoder::EncoderBuffers;
use crate::kernels;
use crate::tokenizer::QwenTokenizer;

use std::time::Instant;

/// Mel + encoder forward for one audio slice → `(encoder output, seq_len)`.
/// Pure over `&QwenModel` + its own `EncoderBuffers`, so it runs in a background
/// thread (encoder on the ANE) while the main thread decodes the previous
/// segment on the CPU — see the pipelined path in [`transcribe_audio`].
fn encode_segment(
    model: &QwenModel,
    cfg: &QwenConfig,
    enc_bufs: &mut EncoderBuffers,
    samples: &[f32],
) -> Option<(Vec<f32>, usize)> {
    let (mel, mel_frames) = audio::mel_spectrogram(samples)?;
    model.encoder.forward(cfg, &mel, mel_frames, Some(enc_bufs))
}

// Prompt token sequences
const PREFIX_HEAD: &[i32] = &[151644, 8948, 198];
const PREFIX_TAIL: &[i32] = &[151645, 198, 151644, 872, 198, 151669];
const SUFFIX_BASE: &[i32] = &[151670, 151645, 198, 151644, 77091, 198];

// Streaming robustness constants (matching C reference)
const STREAM_DEGEN_MAX_PERIOD: usize = 6;
const STREAM_DEGEN_MIN_REPEATS: usize = 4;
const STREAM_STALE_CHUNKS: i32 = 4;
// C `dropped_repeat_tokens >= 8` (qwen_asr.c:1964): if the MAX_REPEAT_TOKEN_RUN guard trims
// at least this many tokens in a chunk, force a stream re-anchor so the poisoned context can't
// keep spamming the same token on the next chunk.
const STREAM_DEGEN_DROP_RESET: usize = 8;
const STREAM_RESET_INTERVAL_CHUNKS: i32 = 45;
const STREAM_RESET_CARRY_TOKENS: usize = 24;
const STREAM_MAX_ENC_WINDOWS: usize = 4;
// Single-token-run suppression cap at the stream commit (antirez's QWEN_STREAM_MAX_REPEAT_TOKEN_RUN).
const STREAM_MAX_REPEAT_TOKEN_RUN: i32 = 12;

// Batch (segment) loop detector — see is_repetition_loop / docs/loop-detection.md. A WIDER
// period than the stream path's tuned 6 so it catches sentence-length phrase loops; same 4-reps
// threshold. Not runtime-tunable (the stream path proves these are fine as constants); the only
// loop knob the segment path exposes is the on/off `--loop-detect` and the recovery bounds.
const SEGMENT_DEGEN_MAX_PERIOD: usize = 32;
const SEGMENT_DEGEN_MIN_REPEATS: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq)]
struct PrefillRowKey {
    a: u64,
    b: u64,
}

#[inline]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

#[inline]
fn prefill_token_key(token: i32) -> PrefillRowKey {
    let x = token as u32 as u64;
    PrefillRowKey {
        a: mix64(0x544f_4b45_4e00_0000 ^ x),
        b: mix64(0x454d_4245_4400_0000 ^ x.rotate_left(17)),
    }
}

fn prefill_embed_key(row: &[f32]) -> PrefillRowKey {
    let mut a = 0xcbf29ce484222325u64;
    let mut b = 0x9e3779b97f4a7c15u64 ^ row.len() as u64;
    for &v in row {
        let bits = v.to_bits() as u64;
        a ^= bits;
        a = a.wrapping_mul(0x100000001b3);
        b ^= bits
            .wrapping_add(0x9e3779b97f4a7c15)
            .wrapping_add(b << 6)
            .wrapping_add(b >> 2);
    }
    PrefillRowKey { a, b }
}

fn prefill_embed_keys(data: &[f32], seq_len: usize, dim: usize) -> Vec<PrefillRowKey> {
    let mut keys = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        keys.push(prefill_embed_key(&data[i * dim..(i + 1) * dim]));
    }
    keys
}

fn prefill_lcp_len(prev: &[PrefillRowKey], current: &[PrefillRowKey], prefill_len: usize) -> usize {
    let cmp_len = prefill_len.min(prev.len()).min(current.len());
    let mut reused = 0usize;
    while reused < cmp_len && prev[reused] == current[reused] {
        reused += 1;
    }
    reused
}

fn get_time_ms() -> f64 {
    // Use monotonic clock
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_secs_f64() * 1000.0
}

fn elapsed_ms(t0: f64) -> f64 {
    get_time_ms() - t0
}

/// Returns `(best_reps, best_period)`: how many times a block of `best_period`
/// tokens repeats at the tail of `tokens`. Used for streaming degeneracy detection.
fn stream_tail_repeat_blocks(tokens: &[i32], max_period: usize) -> (usize, usize) {
    let n = tokens.len();
    if n < 2 {
        return (1, 0);
    }
    let period_cap = (n / 2).min(if max_period > 0 { max_period } else { n / 2 });
    let mut best_reps = 1usize;
    let mut best_period = 0usize;
    for p in 1..=period_cap {
        let mut reps = 1usize;
        while (reps + 1) * p <= n {
            let a = &tokens[n - (reps + 1) * p .. n - reps * p];
            let b = &tokens[n - reps * p       .. n - (reps - 1) * p];
            if a != b { break; }
            reps += 1;
        }
        if reps > best_reps { best_reps = reps; best_period = p; }
    }
    (best_reps, best_period)
}

/// Port of antirez's C `QWEN_STREAM_MAX_REPEAT_TOKEN_RUN` guard (the Rust port had dropped
/// it). Drops tokens from `chunk` that extend a run of the SAME token beyond `max_run`,
/// counting the run continued from the already-committed `prefix` tail — so single-token
/// spam is suppressed at the stream commit. In-place; no-op when `max_run < 1`. Returns the
/// number of tokens dropped — the caller forces a stream re-anchor when it reaches
/// `STREAM_DEGEN_DROP_RESET` (C `dropped_repeat_tokens >= 8`, qwen_asr.c:1964).
fn suppress_repeat_token_runs(prefix: &[i32], chunk: &mut Vec<i32>, max_run: i32) -> usize {
    if max_run < 1 || chunk.is_empty() {
        return 0;
    }
    let before = chunk.len();
    let mut prev_tok = -1i32;
    let mut prev_run = 0i32;
    if let Some(&last) = prefix.last() {
        prev_tok = last;
        prev_run = 1;
        for &t in prefix.iter().rev().skip(1) {
            if t != prev_tok {
                break;
            }
            prev_run += 1;
            if prev_run >= max_run {
                break;
            }
        }
    }
    chunk.retain(|&tok| {
        if tok == prev_tok {
            prev_run += 1;
            prev_run <= max_run // drop once the run exceeds max_run
        } else {
            prev_tok = tok;
            prev_run = 1;
            true
        }
    });
    before - chunk.len()
}

/// Transcribe a single segment. Returns (text, n_text_tokens, degenerate) where
/// `degenerate` is true when loop detection (loop_detect) judged the output a repetition
/// loop — maxed token budget without EOS, or a tail block repeated >= loop_min_repeats times.
fn transcribe_segment(
    ctx: &mut QwenCtx,
    samples: &[f32],
    tokenizer: &QwenTokenizer,
    past_tokens: Option<&[i32]>,
    pre_enc: Option<(Vec<f32>, usize)>,
) -> Option<(String, i32, bool)> {
    let mut cfg_owned = ctx.model.config.clone();
    if let Some(w) = ctx.enc_n_window_infer_override {
        cfg_owned.enc_n_window_infer = w;
    }
    let cfg = &cfg_owned;
    let dim = cfg.dec_hidden;
    let seg_t0 = get_time_ms();
    let mut n_text_tokens = 0i32;

    // Mel + Encoder (or reuse a pre-computed encoding from the pipeline worker).
    let t0 = get_time_ms();
    let (enc_output, enc_seq_len) = match pre_enc {
        Some(e) => e,
        None => {
            let model = ctx.model.clone();
            encode_segment(&model, cfg, &mut ctx.enc_bufs, samples)?
        }
    };
    let mel_ms = 0.0;
    let enc_ms = elapsed_ms(t0);

    if kernels::verbose() >= 2 {
        eprintln!("  Encoder: {} tokens ({:.0} ms)", enc_seq_len, enc_ms);
    }

    if !ctx.prepare_prompt_tokens(tokenizer) {
        return None;
    }

    // Build input embeddings
    let n_prompt_tokens = ctx.prompt_tokens.as_ref().map_or(0, |t| t.len());
    let n_force_prompt_tokens = ctx.force_prompt_tokens.as_ref().map_or(0, |t| t.len());
    let n_past = past_tokens.map_or(0, |t| t.len());
    let n_past_prompt_tokens = if n_past > 0 { n_past + 1 } else { 0 }; // +1 for <asr_text>

    let prefix_len = PREFIX_HEAD.len() + n_prompt_tokens + PREFIX_TAIL.len();
    let suffix_len = SUFFIX_BASE.len() + n_force_prompt_tokens;
    let total_seq = prefix_len + enc_seq_len + suffix_len + n_past_prompt_tokens;

    let mut input_embeds = vec![0.0f32; total_seq * dim];
    let tok_emb = ctx.model.decoder.tok_embeddings_bf16;

    // Embed prefix head
    let mut off = 0;
    for &tok in PREFIX_HEAD {
        unsafe {
            tok_embed_bf16_to_f32(
                &mut input_embeds[off * dim..(off + 1) * dim],
                tok_emb,
                tok,
                dim,
            )
        };
        off += 1;
    }

    // Optional prompt
    if let Some(ref ptoks) = ctx.prompt_tokens {
        for &tok in ptoks {
            unsafe {
                tok_embed_bf16_to_f32(
                    &mut input_embeds[off * dim..(off + 1) * dim],
                    tok_emb,
                    tok,
                    dim,
                )
            };
            off += 1;
        }
    }

    // Prefix tail
    for &tok in PREFIX_TAIL {
        unsafe {
            tok_embed_bf16_to_f32(
                &mut input_embeds[off * dim..(off + 1) * dim],
                tok_emb,
                tok,
                dim,
            )
        };
        off += 1;
    }

    // Encoder output
    for i in 0..enc_seq_len {
        input_embeds[(prefix_len + i) * dim..(prefix_len + i + 1) * dim]
            .copy_from_slice(&enc_output[i * dim..(i + 1) * dim]);
    }

    // Suffix base
    let suffix_off = prefix_len + enc_seq_len;
    for (i, &tok) in SUFFIX_BASE.iter().enumerate() {
        unsafe {
            tok_embed_bf16_to_f32(
            &mut input_embeds[(suffix_off + i) * dim..(suffix_off + i + 1) * dim],
                tok_emb,
                tok,
                dim,
            )
        };
    }

    // Force language tokens
    if let Some(ref ftoks) = ctx.force_prompt_tokens {
        for (i, &tok) in ftoks.iter().enumerate() {
            unsafe {
                tok_embed_bf16_to_f32(
                &mut input_embeds[(suffix_off + SUFFIX_BASE.len() + i) * dim
                    ..(suffix_off + SUFFIX_BASE.len() + i + 1) * dim],
                    tok_emb,
                    tok,
                    dim,
                )
            };
        }
    }

    // Past text conditioning tokens
    let past_off = suffix_off + suffix_len;
    if let Some(ptoks) = past_tokens {
        for (i, &tok) in ptoks.iter().enumerate() {
            unsafe {
                tok_embed_bf16_to_f32(
                &mut input_embeds[(past_off + i) * dim..(past_off + i + 1) * dim],
                    tok_emb,
                    tok,
                    dim,
                )
            };
        }
        unsafe {
            tok_embed_bf16_to_f32(
                &mut input_embeds
                    [(past_off + ptoks.len()) * dim..(past_off + ptoks.len() + 1) * dim],
                tok_emb,
                TOKEN_ASR_TEXT,
                dim,
            )
        };
    }

    // Decoder prefill
    let t0 = get_time_ms();
    ctx.kv_cache.len = 0;
    let prefill_len = total_seq - 1;
    decoder::decoder_prefill(
        &ctx.model.decoder,
        cfg,
        &mut ctx.kv_cache,
        &mut ctx.rope_cache,
        &mut ctx.dec_bufs,
        &input_embeds,
        prefill_len,
    );

    // First token from last prefill position
    let last_embed = &input_embeds[prefill_len * dim..(prefill_len + 1) * dim];
    let mut token = decoder::decoder_forward(
        &ctx.model.decoder,
        cfg,
        &mut ctx.kv_cache,
        &mut ctx.rope_cache,
        &mut ctx.dec_bufs,
        last_embed,
    );

    let prefill_ms = elapsed_ms(t0);
    if kernels::verbose() >= 2 {
        eprintln!("  Prefill: {} tokens ({:.0} ms)", total_seq, prefill_ms);
    }

    // Autoregressive decode
    let t0 = get_time_ms();
    let max_tokens = 2048;
    let mut n_generated = 0;
    let mut past_asr_text = n_force_prompt_tokens > 0 || n_past > 0;

    let mut text_bytes: Vec<u8> = Vec::new();
    // Text token ids, kept for loop/degeneracy detection (stream_tail_repeat_blocks).
    let mut text_tokens: Vec<i32> = Vec::new();
    let mut tmp_embed = vec![0.0f32; dim];

    // Lookahead (Jacobi) decoding: QWEN_LOOKAHEAD=N enables it with window N.
    // 0 (default) = plain autoregressive. The output is identical token-for-token
    // either way (every speculated token is verified); only speed differs.
    let lookahead_n: usize = std::env::var("QWEN_LOOKAHEAD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut guesses: Vec<i32> = if lookahead_n > 0 { vec![token; lookahead_n] } else { Vec::new() };
    let mut pending: std::collections::VecDeque<i32> = std::collections::VecDeque::new();
    let mut win_embeds: Vec<f32> = Vec::new();
    let (mut la_batches, mut la_committed) = (0u64, 0u64); // acceptance telemetry

    while n_generated < max_tokens {
        n_generated += 1;

        if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END {
            break;
        }

        if token == TOKEN_ASR_TEXT {
            past_asr_text = true;
        } else if past_asr_text {
            let piece_bytes = tokenizer.decode_bytes(token);
            text_bytes.extend_from_slice(piece_bytes);
            text_tokens.push(token);
            n_text_tokens += 1;

            if let Some(ref cb) = ctx.token_cb {
                // For the callback, provide lossy UTF-8 for display purposes
                cb(&String::from_utf8_lossy(piece_bytes));
            }
        }

        // Advance to the next token.
        if lookahead_n == 0 {
            unsafe { tok_embed_bf16_to_f32(&mut tmp_embed, tok_emb, token, dim) };
            token = decoder::decoder_forward(
                &ctx.model.decoder,
                cfg,
                &mut ctx.kv_cache,
                &mut ctx.rope_cache,
                &mut ctx.dec_bufs,
                &tmp_embed,
            );
        } else {
            if pending.is_empty() {
                // One batched forward over [cur, guess0, guess1, ...] yields the
                // model's correct next-token at every position; accept the longest
                // run of guesses that matched. Lossless: cur's continuation is
                // always preds[0]; each later token is taken only if its guess held.
                let wlen = 1 + guesses.len();
                win_embeds.resize(wlen * dim, 0.0);
                unsafe {
                    tok_embed_bf16_to_f32(&mut win_embeds[0..dim], tok_emb, token, dim);
                    for (i, &g) in guesses.iter().enumerate() {
                        tok_embed_bf16_to_f32(
                            &mut win_embeds[(i + 1) * dim..(i + 2) * dim],
                            tok_emb,
                            g,
                            dim,
                        );
                    }
                }
                let start_pos = ctx.kv_cache.len;
                let preds = decoder::decoder_forward_batch(
                    &ctx.model.decoder,
                    cfg,
                    &mut ctx.kv_cache,
                    &mut ctx.rope_cache,
                    &mut ctx.dec_bufs,
                    &win_embeds,
                    wlen,
                );
                let mut n_accept = 1usize;
                for i in 0..guesses.len() {
                    if guesses[i] == preds[i] {
                        n_accept += 1;
                    } else {
                        break;
                    }
                }
                la_batches += 1;
                la_committed += n_accept as u64;
                // Commit only the verified positions; discard speculative KV.
                ctx.kv_cache.len = start_pos + n_accept;
                for &t in &preds[0..n_accept] {
                    pending.push_back(t);
                }
                // Jacobi update: rejected-tail predictions seed the next guesses.
                let pad = *preds.last().unwrap();
                let mut ng: Vec<i32> = preds[n_accept..].to_vec();
                while ng.len() < lookahead_n {
                    ng.push(pad);
                }
                ng.truncate(lookahead_n);
                guesses = ng;
            }
            token = pending.pop_front().unwrap();
        }
    }
    if lookahead_n > 0 && la_batches > 0 {
        eprintln!(
            "[lookahead] window={lookahead_n} batches={la_batches} committed={la_committed} avg_accept={:.2}",
            la_committed as f64 / la_batches as f64
        );
    }

    let decode_ms = elapsed_ms(t0);
    if kernels::verbose() >= 2 {
        eprintln!(
            "  Decode: {} tokens ({:.0} ms, {:.1} ms/token)",
            n_generated,
            decode_ms,
            if n_generated > 0 {
                decode_ms / n_generated as f64
            } else {
                0.0
            }
        );
    }

    // Trim whitespace — convert accumulated bytes to UTF-8 first
    let text = String::from_utf8_lossy(&text_bytes);
    let trimmed = text.trim().to_string();

    ctx.perf_total_ms += elapsed_ms(seg_t0);
    ctx.perf_text_tokens += n_text_tokens;
    ctx.perf_encode_ms += mel_ms + enc_ms;
    ctx.perf_decode_ms += prefill_ms + decode_ms;

    // Observe-only degeneracy accounting (no behavior change). A segment that ran the
    // full `max_tokens` budget without ever hitting EOS never broke out of the loop
    // above — the model is repeating/degenerate. The offline path has no early-stop
    // (unlike `transcribe_stream`), so each such segment costs ~10-40x a healthy one
    // (~50-200 tokens); that is what turns a ~70-min block into a multi-hour "stuck" one.
    ctx.perf_segments += 1;
    let maxed = n_generated >= max_tokens;
    if maxed {
        ctx.perf_maxed_segments += 1;
    }

    // Loop/degeneracy detection (docs/loop-detection.md). A segment is degenerate if it ran to
    // the token cap without EOS (maxed — the high-confidence observe-only signal), OR its tail is
    // a repetition LOOP per is_repetition_loop (a block repeated >= loop_min_repeats times AND
    // covering most of the output). The coverage gate is what keeps legitimate brief repetition
    // (a refrain repeated a few times) from being mistaken for a runaway loop.
    let degenerate = ctx.loop_detect
        && (maxed
            || is_repetition_loop(&text_tokens, SEGMENT_DEGEN_MAX_PERIOD, SEGMENT_DEGEN_MIN_REPEATS));

    Some((trimmed, n_text_tokens, degenerate))
}

/// True if `tokens` ends in a repetition LOOP: a block of <= `max_period` tokens repeated
/// >= `min_reps` times (via antirez's `stream_tail_repeat_blocks`) AND that repeat covering at
/// least HALF of `tokens`. The coverage gate is the false-positive guard: a runaway loop
/// dominates the output (the repeat is ~the whole tail), whereas legitimate brief repetition (a
/// sung refrain, a chant, a repeated list item) is only a small fraction of the segment and is
/// NOT flagged. See docs/loop-detection.md.
fn is_repetition_loop(tokens: &[i32], max_period: usize, min_reps: usize) -> bool {
    let (reps, period) = stream_tail_repeat_blocks(tokens, max_period);
    reps >= min_reps && 2 * reps.saturating_mul(period) >= tokens.len()
}

// ========================================================================
// Segment-based splitting
// ========================================================================

fn find_split_point(samples: &[f32], target_sample: usize, search_sec: f32) -> usize {
    let search_half = (search_sec * SAMPLE_RATE as f32) as usize;
    let lo = target_sample.saturating_sub(search_half);
    let hi = (target_sample + search_half).min(samples.len());

    let win_samples = 1600; // 100ms at 16kHz
    let mut best_energy = 1e30f32;
    let mut best_center = target_sample;

    let mut pos = lo;
    while pos + win_samples <= hi {
        let end = (pos + win_samples).min(samples.len());
        let mut energy = 0.0f32;
        for &s in samples.iter().take(end).skip(pos) {
            energy += s * s;
        }
        energy /= (end - pos) as f32;
        if energy < best_energy {
            best_energy = energy;
            best_center = pos + (end - pos) / 2;
        }
        pos += win_samples / 2;
    }

    best_center
}

/// Transcribe one segment's audio, recovering from decoder repetition loops WITHOUT
/// recursion (docs/loop-detection.md). Decodes the span; if `transcribe_segment` reports it
/// degenerate, the span is halved at a WORD BOUNDARY — `find_split_point` picks the lowest-
/// energy point within ±`search_sec` of the midpoint, so the cut lands in a gap, not mid-word
/// — and each half is re-decoded. Driven by an explicit work-stack (no call recursion),
/// bounded by `loop_max_depth` halvings and the `loop_min_split_sec` size floor. Emitted
/// sub-segments are sorted back into time order. With `loop_detect` off, nothing is ever
/// flagged degenerate, so this decodes the span exactly once — identical to legacy behavior.
fn transcribe_with_recovery(
    ctx: &mut QwenCtx,
    samples: &[f32],
    tokenizer: &QwenTokenizer,
    base_ms: u64,
    out: &mut Vec<TranscriptSegment>,
) {
    let min_samples = SAMPLE_RATE as usize / 2;
    let min_split = (ctx.loop_min_split_sec.max(0.0) * SAMPLE_RATE as f32) as usize;
    let max_depth = ctx.loop_max_depth.max(0);

    // Work-stack of (start, end, depth) offsets into `samples`. Pop → decode → either emit
    // or push the two halves. LIFO with "push right then left" processes left-first; the
    // final sort by start_ms makes ordering bulletproof regardless.
    let mut work: Vec<(usize, usize, i32)> = vec![(0, samples.len(), 0)];
    let mut done: Vec<TranscriptSegment> = Vec::new();

    while let Some((s, e, depth)) = work.pop() {
        let seg = &samples[s..e];
        let seg_len = e - s;
        // Record this sub-segment's audio duration for the xRT/perf log (NOT a token budget —
        // max_tokens is a flat constant; this only feeds the realtime-ratio reporting).
        ctx.perf_audio_ms = 1000.0 * seg_len as f64 / SAMPLE_RATE as f64;
        // Pad sub-minimum slices to the model floor.
        let seg_buf: Vec<f32>;
        let seg_ptr = if seg_len < min_samples {
            seg_buf = {
                let mut buf = vec![0.0f32; min_samples];
                buf[..seg_len].copy_from_slice(seg);
                buf
            };
            &seg_buf[..]
        } else {
            seg
        };

        // Snapshot the observe-only counters so a DISCARDED (re-split) decode doesn't inflate
        // them — only the FINAL emitted segments should count, so a clip that recovers cleanly
        // never reads back as DEGENERATE and perf_segments stays == the emitted-segment count.
        let segments_before = ctx.perf_segments;
        let maxed_before = ctx.perf_maxed_segments;
        let (text, _n, degenerate) = match transcribe_segment(ctx, seg_ptr, tokenizer, None, None) {
            Some(r) => r,
            None => continue,
        };

        // Recover: split at a WORD BOUNDARY (find_split_point = lowest-energy point within
        // ±search of the midpoint) and re-decode each half — while depth allows AND the span is
        // at least 2*min_split. NOTE: the floor gates WHETHER to split, not the exact half sizes:
        // the cut follows the word gap, so halves can be uneven and one may fall below
        // loop_min_split_sec. That's deliberate — a clean word-boundary cut matters more than
        // hitting an exact size (forcing the size would push the cut mid-word). The undersized
        // half is still a valid short clip; it just won't be split again (it's < 2*min_split).
        if degenerate && depth < max_depth && seg_len >= 2 * min_split.max(1) {
            // This decode is thrown away and replaced by the two halves — roll back its counters.
            ctx.perf_segments = segments_before;
            ctx.perf_maxed_segments = maxed_before;
            let search = ctx.search_sec.min(seg_len as f32 / SAMPLE_RATE as f32 / 2.0);
            let rel = find_split_point(seg, seg_len / 2, search).clamp(1, seg_len - 1);
            let split = s + rel;
            work.push((split, e, depth + 1));
            work.push((s, split, depth + 1));
            continue;
        }

        if !text.is_empty() {
            let start_ms = base_ms + (s as u64 * 1000) / SAMPLE_RATE as u64;
            let end_ms = base_ms + (e as u64 * 1000) / SAMPLE_RATE as u64;
            done.push(TranscriptSegment { start_ms, end_ms, text });
        }
    }

    done.sort_by_key(|t| t.start_ms);
    out.extend(done);
}

fn should_insert_boundary_space(prev_ch: u8, next_ch: u8) -> bool {
    if prev_ch == 0 || next_ch == 0 {
        return false;
    }
    if (prev_ch as char).is_whitespace() {
        return false;
    }
    if (next_ch as char).is_whitespace() {
        return false;
    }
    if (next_ch as char).is_ascii_punctuation() {
        return false;
    }
    true
}

// ========================================================================
// Public API
// ========================================================================

/// A single transcribed segment with wall-clock timestamps.
pub struct TranscriptSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

/// Transcribe one contiguous slice of already-decoded audio into ≤`segment_sec`
/// sub-chunks, appending each transcribed sub-chunk to `out` with timestamps
/// rebased by `base_ms`.
///
/// This is the shared core of [`transcribe_segmented`] (one slice covering the
/// whole file, `base_ms = 0`) and [`transcribe_clips`] (one slice per speech
/// region, `base_ms = region_start_ms`). It performs NO model loading or audio
/// decoding — the caller owns the loaded tokenizer and the decoded `samples`.
///
/// The ~`segment_sec` (default 30 s) sub-chunking is the decoder-repetition
/// guard: long spans make the model loop, so they are split at low-energy
/// boundaries. Sub-chunk timestamps are derived from in-slice sample offsets and
/// shifted by `base_ms`, so a region never produces a segment spanning a skipped
/// gap.
fn segment_slice(
    ctx: &mut QwenCtx,
    samples: &[f32],
    tokenizer: &QwenTokenizer,
    base_ms: u64,
    out: &mut Vec<TranscriptSegment>,
) {
    let segment_sec = if ctx.segment_sec > 0.0 { ctx.segment_sec } else { 30.0 };
    let search_sec = ctx.search_sec.min(segment_sec / 2.0);
    let target_samples = (segment_sec * SAMPLE_RATE as f32) as usize;
    let margin_samples = (search_sec * SAMPLE_RATE as f32) as usize;

    if samples.len() <= target_samples + margin_samples {
        // Whole slice fits in one segment — still run loop-recovery (it may halve a
        // degenerate short file). perf_audio_ms / padding / emit live inside the helper.
        transcribe_with_recovery(ctx, samples, tokenizer, base_ms, out);
        return;
    }

    // Build split points over the slice
    let mut splits = vec![0usize];
    let mut pos = 0;
    while pos + target_samples + margin_samples < samples.len() {
        let split = find_split_point(samples, pos + target_samples, search_sec);
        splits.push(split);
        pos = split;
        if splits.len() >= 127 {
            break;
        }
    }
    let n_splits = splits.len();

    for s in 0..n_splits {
        let seg_start = splits[s];
        let seg_end = if s + 1 < n_splits { splits[s + 1] } else { samples.len() };
        let seg_base_ms = base_ms + (seg_start as u64 * 1000) / SAMPLE_RATE as u64;
        // Per coarse ~30s segment, run the loop-recovery transcriber: it decodes the
        // segment and, if degenerate, halves it at word boundaries and re-decodes (no
        // recursion — explicit work-stack). perf_audio_ms / padding / emit live inside it.
        transcribe_with_recovery(ctx, &samples[seg_start..seg_end], tokenizer, seg_base_ms, out);
    }
}

/// Transcribe audio and return per-segment results with timestamps.
///
/// Unlike [`transcribe_audio`], this function preserves the original audio
/// timeline (no silence compaction) so that `start_ms`/`end_ms` are accurate
/// for subtitle generation.  Each segment's duration is used to set the token
/// budget, so accuracy is not sacrificed for long files.
///
/// Uses `ctx.segment_sec` for splitting; falls back to 30 s if unset.
pub fn transcribe_segmented(ctx: &mut QwenCtx, samples: &[f32]) -> Option<Vec<TranscriptSegment>> {
    ctx.reset_perf();

    let model = ctx.model.clone();
    let tokenizer = &model.tokenizer;
    if !ctx.prepare_prompt_tokens(tokenizer) {
        return None;
    }

    let mut segments: Vec<TranscriptSegment> = Vec::new();
    segment_slice(ctx, samples, tokenizer, 0, &mut segments);
    Some(segments)
}

/// Transcribe only the listed speech regions of already-decoded audio, in a
/// single model load / single decode, returning per-segment timestamps rebased
/// to the ORIGINAL file timeline.
///
/// `regions` are `(start_ms, end_ms)` spans on the original timeline, expected
/// sorted and non-overlapping (as produced by a VAD/music pre-pass). For each
/// region the corresponding slice of `samples` is transcribed via
/// [`segment_slice`] — internal ≤`ctx.segment_sec` sub-chunking is preserved,
/// and region boundaries never trigger a reload or re-decode. Everything outside
/// the listed regions is skipped, and no segment spans a skipped gap.
pub fn transcribe_clips(
    ctx: &mut QwenCtx,
    samples: &[f32],
    regions: &[(u64, u64)],
) -> Option<Vec<TranscriptSegment>> {
    ctx.reset_perf();

    let model = ctx.model.clone();
    let tokenizer = &model.tokenizer;
    if !ctx.prepare_prompt_tokens(tokenizer) {
        return None;
    }

    let n = samples.len();
    let mut segments: Vec<TranscriptSegment> = Vec::new();

    for &(region_start_ms, region_end_ms) in regions {
        // Map the original-timeline region to in-buffer sample offsets, clamped
        // to the decoded audio (a trailing region may extend past EOF).
        let (start_sample, end_sample) = match region_to_samples(region_start_ms, region_end_ms, n) {
            Some(r) => r,
            None => continue,
        };

        // base_ms is the region's original-timeline start; in-slice offsets are
        // added on top, so segment times land on the original file timeline.
        segment_slice(
            ctx,
            &samples[start_sample..end_sample],
            tokenizer,
            region_start_ms,
            &mut segments,
        );
    }

    Some(segments)
}

/// Map an original-timeline `(start_ms, end_ms)` region to clamped in-buffer
/// sample offsets `[start, end)` within a buffer of `n_samples`. Returns `None`
/// when the region is empty after clamping (e.g. entirely past end-of-file).
fn region_to_samples(start_ms: u64, end_ms: u64, n_samples: usize) -> Option<(usize, usize)> {
    let start = ((start_ms.saturating_mul(SAMPLE_RATE as u64)) / 1000) as usize;
    let end = ((end_ms.saturating_mul(SAMPLE_RATE as u64)) / 1000) as usize;
    let start = start.min(n_samples);
    let end = end.min(n_samples);
    if end <= start {
        None
    } else {
        Some((start, end))
    }
}

#[cfg(test)]
mod clip_tests {
    use super::*;

    const SR: usize = SAMPLE_RATE as usize; // samples per second

    #[test]
    fn region_maps_ms_to_samples() {
        // 1500ms..2500ms at 16kHz -> 24000..40000.
        assert_eq!(region_to_samples(1500, 2500, SR * 10), Some((24_000, 40_000)));
    }

    #[test]
    fn region_clamps_to_end_of_buffer() {
        // Buffer is only 1s long; a region from 0.5s..5s clamps its end to 1s.
        assert_eq!(region_to_samples(500, 5000, SR), Some((SR / 2, SR)));
    }

    #[test]
    fn region_entirely_past_eof_is_none() {
        // Buffer 1s long; region starts at 2s -> nothing to transcribe.
        assert_eq!(region_to_samples(2000, 3000, SR), None);
    }

    #[test]
    fn region_zero_width_is_none() {
        assert_eq!(region_to_samples(1000, 1000, SR * 10), None);
    }
}

/// Transcribe audio samples (f32, 16 kHz, mono, range [-1, 1]).
///
/// When `ctx.segment_sec > 0` and the audio exceeds that duration, it is
/// automatically split at low-energy boundaries.
/// Returns `None` if the tokenizer or encoder fails to initialize.
pub fn transcribe_audio(ctx: &mut QwenCtx, samples: &[f32]) -> Option<String> {
    ctx.reset_perf();
    ctx.perf_audio_ms = 1000.0 * samples.len() as f64 / SAMPLE_RATE as f64;

    let audio_samples = if ctx.skip_silence {
        let compacted = if ctx.segment_sec > 0.0 {
            audio::compact_silence_fast(samples)
        } else {
            audio::compact_silence(samples)
        };
        if kernels::verbose() >= 1 {
            let used_pct = 100.0 * compacted.len() as f32 / samples.len().max(1) as f32;
            eprintln!(
                "Silence skip: used {:.1}%, skipped {:.1}% ({} -> {} samples)",
                used_pct,
                100.0 - used_pct,
                samples.len(),
                compacted.len()
            );
        }
        compacted
    } else {
        samples.to_vec()
    };

    if kernels::verbose() >= 2 {
        eprintln!(
            "Audio: {} samples ({:.1} seconds)",
            audio_samples.len(),
            audio_samples.len() as f32 / SAMPLE_RATE as f32
        );
    }

    let model = ctx.model.clone();
    let tokenizer = &model.tokenizer;
    if !ctx.prepare_prompt_tokens(tokenizer) {
        return None;
    }

    let target_samples = (ctx.segment_sec * SAMPLE_RATE as f32) as usize;
    let search = ctx.search_sec.min(ctx.segment_sec / 2.0);
    let margin_samples = (search * SAMPLE_RATE as f32) as usize;

    // No splitting if segment_sec is 0 or audio fits in one segment
    if ctx.segment_sec <= 0.0 || audio_samples.len() <= target_samples + margin_samples {
        let (text, _, _) = transcribe_segment(ctx, &audio_samples, tokenizer, None, None)?;
        return Some(text);
    }

    // Build split points
    let mut splits = vec![0usize];
    let mut pos = 0;
    while pos + target_samples + margin_samples < audio_samples.len() {
        let split = find_split_point(&audio_samples, pos + target_samples, search);
        splits.push(split);
        pos = split;
        if splits.len() >= 127 {
            break;
        }
    }
    let n_splits = splits.len();

    if kernels::verbose() >= 2 {
        eprintln!("Splitting into {} segments", n_splits);
    }

    let mut result = String::new();
    let min_samples = SAMPLE_RATE as usize / 2;
    let use_past_text = ctx.past_text_conditioning;

    // Pipelined encode ‖ decode (QWEN_PIPELINE=1): a background thread encodes the
    // next segment(s) — encoder GEMMs on the ANE when --mac-ane-encoder is set,
    // and CPU work on a SEPARATE thread pool (use_encoder_pool) so it never shares
    // dispatch state with the main decoder's pool. The two overlap, hiding the
    // ~30% encoder under the ~70% decoder. Output is identical to the serial path.
    if std::env::var("QWEN_PIPELINE").map(|v| v == "1").unwrap_or(false) {
        let worker_cfg = {
            let mut c = ctx.model.config.clone();
            if let Some(w) = ctx.enc_n_window_infer_override {
                c.enc_n_window_infer = w;
            }
            c
        };
        let enc_threads: usize = std::env::var("QWEN_ENC_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let ranges: Vec<(usize, usize)> = (0..n_splits)
            .map(|s| {
                let start = splits[s];
                let end = if s + 1 < n_splits { splits[s + 1] } else { audio_samples.len() };
                (start, end)
            })
            .collect();
        let worker_model = ctx.model.clone();
        let audio_ref: &[f32] = &audio_samples;
        std::thread::scope(|scope| {
            let (tx, rx) = std::sync::mpsc::sync_channel::<Option<(Vec<f32>, usize)>>(2);
            scope.spawn(move || {
                // Dedicated encoder-lane pool: re-entrant with the main decoder pool.
                kernels::use_encoder_pool(enc_threads);
                let mut enc_bufs = EncoderBuffers::new();
                for (start, end) in ranges {
                    let seg_samples = end - start;
                    let seg_buf: Vec<f32>;
                    let seg = if seg_samples < min_samples {
                        let mut b = vec![0.0f32; min_samples];
                        b[..seg_samples].copy_from_slice(&audio_ref[start..end]);
                        seg_buf = b;
                        &seg_buf[..]
                    } else {
                        &audio_ref[start..end]
                    };
                    if tx.send(encode_segment(&worker_model, &worker_cfg, &mut enc_bufs, seg)).is_err() {
                        break;
                    }
                }
            });
            for _ in 0..n_splits {
                let enc = match rx.recv() {
                    Ok(Some(e)) => e,
                    _ => continue,
                };
                let past_tokens: Option<Vec<i32>> = if use_past_text && !result.is_empty() {
                    tokenizer.encode(&result)
                } else {
                    None
                };
                let seg_text = match transcribe_segment(
                    ctx, &[], tokenizer, past_tokens.as_deref(), Some(enc),
                ) {
                    Some((t, _, _)) if !t.is_empty() => t,
                    _ => continue,
                };
                if !result.is_empty() {
                    let prev = *result.as_bytes().last().unwrap_or(&0);
                    let next = *seg_text.as_bytes().first().unwrap_or(&0);
                    if should_insert_boundary_space(prev, next) {
                        result.push(' ');
                        if let Some(ref cb) = ctx.token_cb {
                            cb(" ");
                        }
                    }
                }
                if let Some(ref cb) = ctx.token_cb {
                    if ctx.past_text_conditioning {
                        cb(&seg_text);
                    }
                }
                result.push_str(&seg_text);
            }
        });
        return Some(result);
    }

    for s in 0..n_splits {
        let core_start = splits[s];
        let core_end = if s + 1 < n_splits {
            splits[s + 1]
        } else {
            audio_samples.len()
        };
        let seg_start = core_start;
        let seg_end = core_end;
        let seg_samples = seg_end - seg_start;

        if kernels::verbose() >= 2 {
            eprintln!(
                "Segment {}/{}: {:.1}-{:.1}s ({} samples)",
                s + 1,
                n_splits,
                seg_start as f32 / SAMPLE_RATE as f32,
                seg_end as f32 / SAMPLE_RATE as f32,
                seg_samples
            );
        }

        let seg_buf: Vec<f32>;
        let seg_ptr = if seg_samples < min_samples {
            seg_buf = {
                let mut buf = vec![0.0f32; min_samples];
                buf[..seg_samples].copy_from_slice(&audio_samples[seg_start..seg_end]);
                buf
            };
            &seg_buf[..]
        } else {
            &audio_samples[seg_start..seg_end]
        };

        let past_tokens: Option<Vec<i32>> = if use_past_text && !result.is_empty() {
            tokenizer.encode(&result)
        } else {
            None
        };

        let (seg_text, _seg_text_tokens, _degenerate) =
            match transcribe_segment(ctx, seg_ptr, tokenizer, past_tokens.as_deref(), None) {
            Some(r) => r,
            None => continue,
        };

        if seg_text.is_empty() {
            continue;
        }

        let need_space = if !result.is_empty() {
            let prev = *result.as_bytes().last().unwrap_or(&0);
            let next = *seg_text.as_bytes().first().unwrap_or(&0);
            should_insert_boundary_space(prev, next)
        } else {
            false
        };

        if need_space {
            result.push(' ');
            if let Some(ref cb) = ctx.token_cb {
                cb(" ");
            }
        }

        if let Some(ref cb) = ctx.token_cb {
            if ctx.past_text_conditioning {
                cb(&seg_text);
            }
        }

        result.push_str(&seg_text);
    }

    Some(result)
}

/// Convenience wrapper: load a WAV file and transcribe it.
///
/// Equivalent to [`audio::load_wav`] followed by
/// [`transcribe_audio`].
pub fn transcribe(ctx: &mut QwenCtx, wav_path: &str) -> Option<String> {
    let samples = audio::load_wav(wav_path)?;
    transcribe_audio(ctx, &samples)
}

/// Transcribe from stdin.
pub fn transcribe_stdin(ctx: &mut QwenCtx) -> Option<String> {
    let samples = audio::read_pcm_stdin()?;
    transcribe_audio(ctx, &samples)
}

/// Streaming transcription: processes audio in chunks, emitting tokens via
/// `ctx.token_cb` as they become stable.
///
/// Trades throughput for lower latency compared to offline mode. If no
/// `token_cb` is set, falls back to a single offline decode of the full audio.
pub fn transcribe_stream(ctx: &mut QwenCtx, samples: &[f32]) -> Option<String> {
    let mut cfg = ctx.model.config.clone();
    // Fold the per-ctx encoder-window override (CLI --enc-window-sec) into the
    // working config clone, exactly where it used to live on ctx.config.
    if let Some(w) = ctx.enc_n_window_infer_override {
        cfg.enc_n_window_infer = w;
    }
    let dim = cfg.dec_hidden;
    let chunk_samples = (ctx.stream_chunk_sec * SAMPLE_RATE as f32) as usize;
    let rollback = ctx.stream_rollback;
    let unfixed_chunks = ctx.stream_unfixed_chunks;
    let max_new_tokens = if ctx.stream_max_new_tokens > 0 {
        ctx.stream_max_new_tokens
    } else {
        32
    };

    ctx.reset_perf();
    ctx.perf_audio_ms = 1000.0 * samples.len() as f64 / SAMPLE_RATE as f64;

    // If no token callback, fall back to offline decode
    if ctx.token_cb.is_none() {
        if kernels::verbose() >= 2 {
            eprintln!("Streaming: no token callback, using direct final refinement");
        }
        let audio_samples = if ctx.skip_silence {
            audio::compact_silence(samples)
        } else {
            samples.to_vec()
        };
        let model = ctx.model.clone();
        let tokenizer = &model.tokenizer;
        ctx.prepare_prompt_tokens(tokenizer);
        let (text, _, _) = transcribe_segment(ctx, &audio_samples, tokenizer, None, None)?;
        return Some(text);
    }

    let audio_samples = if ctx.skip_silence {
        audio::compact_silence_fast(samples)
    } else {
        samples.to_vec()
    };

    let model = ctx.model.clone();
    let tokenizer = &model.tokenizer;
    if !ctx.prepare_prompt_tokens(tokenizer) {
        return None;
    }

    let enc_window_frames = cfg.enc_n_window_infer.clamp(100, 800);
    let enc_window_samples = enc_window_frames * HOP_LENGTH;

    let tok_emb = ctx.model.decoder.tok_embeddings_bf16;

    let mut raw_tokens: Vec<i32> = Vec::new();
    let mut stable_text_tokens: Vec<i32> = Vec::new();
    let mut result_bytes: Vec<u8> = Vec::new();
    let mut tmp_embed = vec![0.0f32; dim];

    let mut chunk_idx = 0i32;
    let mut audio_cursor = 0usize;

    // Encoder window cache
    struct EncWindow {
        seq_len: usize,
        enc_output: Vec<f32>,
        row_keys: Vec<PrefillRowKey>,
    }
    let mut enc_cache: Vec<EncWindow> = Vec::new();
    let mut enc_cached_seq_total = 0usize;

    // Previous prefill row keys for LCP reuse
    let mut prev_prefill_keys: Vec<PrefillRowKey> = Vec::new();

    // Streaming robustness state
    let mut stale_count = 0i32;
    let mut prev_tail_snapshot: Vec<i32> = Vec::new();
    let mut enc_cache_base_windows = 0usize;

    while audio_cursor < audio_samples.len() {
        let chunk_t0 = get_time_ms();
        audio_cursor = (audio_cursor + chunk_samples).min(audio_samples.len());
        let is_final = audio_cursor >= audio_samples.len();

        // Encoder
        let t0 = get_time_ms();
        let full_end = (audio_cursor / enc_window_samples) * enc_window_samples;

        // Cache completed windows (base offset accounts for windows cleared on re-anchor)
        while (enc_cache_base_windows + enc_cache.len()) * enc_window_samples < full_end {
            let ws = (enc_cache_base_windows + enc_cache.len()) * enc_window_samples;
            let (mel, mel_frames) =
                audio::mel_spectrogram(&audio_samples[ws..ws + enc_window_samples])?;
            let (win_enc, win_seq) =
                ctx.model.encoder
                    .forward(&cfg, &mel, mel_frames, Some(&mut ctx.enc_bufs))?;
            let row_keys = prefill_embed_keys(&win_enc, win_seq, dim);
            enc_cached_seq_total += win_seq;
            enc_cache.push(EncWindow {
                seq_len: win_seq,
                enc_output: win_enc,
                row_keys,
            });
        }

        // Encode partial tail
        let mut partial_seq = 0;
        let mut partial_enc: Vec<f32> = Vec::new();
        let mut partial_keys: Vec<PrefillRowKey> = Vec::new();
        if is_final && full_end < audio_cursor {
            let _partial_samples = audio_cursor - full_end;
            if let Some((mel, mel_frames)) =
                audio::mel_spectrogram(&audio_samples[full_end..audio_cursor])
            {
                if let Some((enc, seq)) =
                    ctx.model.encoder
                        .forward(&cfg, &mel, mel_frames, Some(&mut ctx.enc_bufs))
                {
                    partial_seq = seq;
                    partial_keys = prefill_embed_keys(&enc, seq, dim);
                    partial_enc = enc;
                }
            }
        }

        let enc_seq_len = enc_cached_seq_total + partial_seq;
        if enc_seq_len == 0 {
            chunk_idx += 1;
            continue;
        }

        let enc_ms = elapsed_ms(t0);
        ctx.perf_encode_ms += enc_ms;

        if !is_final && chunk_idx < unfixed_chunks {
            prev_prefill_keys.clear();
            ctx.perf_total_ms += elapsed_ms(chunk_t0);
            chunk_idx += 1;
            continue;
        }

        // Prefix rollback
        let n_prefix_tokens = if ctx.past_text_conditioning
            && chunk_idx >= unfixed_chunks
            && !raw_tokens.is_empty()
        {
            (raw_tokens.len() as i32 - rollback).max(0) as usize
        } else {
            0
        };

        // Build input embeddings
        let n_prompt_tokens = ctx.prompt_tokens.as_ref().map_or(0, |t| t.len());
        let n_force_prompt_tokens = ctx.force_prompt_tokens.as_ref().map_or(0, |t| t.len());
        let prefix_len = PREFIX_HEAD.len() + n_prompt_tokens + PREFIX_TAIL.len();
        let suffix_len = SUFFIX_BASE.len() + n_force_prompt_tokens;
        let total_seq = prefix_len + enc_seq_len + suffix_len + n_prefix_tokens;

        let mut input_embeds = vec![0.0f32; total_seq * dim];
        let mut prefill_keys = vec![PrefillRowKey { a: 0, b: 0 }; total_seq];
        let mut off = 0;

        for &tok in PREFIX_HEAD {
            unsafe {
                tok_embed_bf16_to_f32(
                    &mut input_embeds[off * dim..(off + 1) * dim],
                    tok_emb,
                    tok,
                    dim,
                )
            };
            prefill_keys[off] = prefill_token_key(tok);
            off += 1;
        }
        if let Some(ref ptoks) = ctx.prompt_tokens {
            for &tok in ptoks {
                unsafe {
                    tok_embed_bf16_to_f32(
                        &mut input_embeds[off * dim..(off + 1) * dim],
                        tok_emb,
                        tok,
                        dim,
                    )
                };
                prefill_keys[off] = prefill_token_key(tok);
                off += 1;
            }
        }
        for &tok in PREFIX_TAIL {
            unsafe {
                tok_embed_bf16_to_f32(
                    &mut input_embeds[off * dim..(off + 1) * dim],
                    tok_emb,
                    tok,
                    dim,
                )
            };
            prefill_keys[off] = prefill_token_key(tok);
            off += 1;
        }

        let mut enc_key_off = 0;
        for w in &enc_cache {
            prefill_keys[prefix_len + enc_key_off..prefix_len + enc_key_off + w.seq_len]
                .copy_from_slice(&w.row_keys);
            enc_key_off += w.seq_len;
        }
        if partial_seq > 0 {
            prefill_keys[prefix_len + enc_key_off..prefix_len + enc_key_off + partial_seq]
                .copy_from_slice(&partial_keys);
        }

        let mut enc_embed_off = 0;
        for w in &enc_cache {
            let n = w.seq_len * dim;
            input_embeds
                [(prefix_len + enc_embed_off) * dim..(prefix_len + enc_embed_off) * dim + n]
                .copy_from_slice(&w.enc_output);
            enc_embed_off += w.seq_len;
        }
        if partial_seq > 0 {
            let n = partial_seq * dim;
            input_embeds
                [(prefix_len + enc_embed_off) * dim..(prefix_len + enc_embed_off) * dim + n]
                .copy_from_slice(&partial_enc);
        }

        let suffix_off = prefix_len + enc_seq_len;
        for (i, &tok) in SUFFIX_BASE.iter().enumerate() {
            unsafe {
                tok_embed_bf16_to_f32(
                &mut input_embeds[(suffix_off + i) * dim..(suffix_off + i + 1) * dim],
                    tok_emb,
                    tok,
                    dim,
                )
            };
            prefill_keys[suffix_off + i] = prefill_token_key(tok);
        }
        if let Some(ref ftoks) = ctx.force_prompt_tokens {
            for (i, &tok) in ftoks.iter().enumerate() {
                unsafe {
                    tok_embed_bf16_to_f32(
                    &mut input_embeds[(suffix_off + SUFFIX_BASE.len() + i) * dim
                        ..(suffix_off + SUFFIX_BASE.len() + i + 1) * dim],
                        tok_emb,
                        tok,
                        dim,
                    )
                };
                prefill_keys[suffix_off + SUFFIX_BASE.len() + i] = prefill_token_key(tok);
            }
        }

        let text_off = suffix_off + suffix_len;
        for i in 0..n_prefix_tokens {
            unsafe {
                tok_embed_bf16_to_f32(
                &mut input_embeds[(text_off + i) * dim..(text_off + i + 1) * dim],
                    tok_emb,
                    raw_tokens[i],
                    dim,
                )
            };
            prefill_keys[text_off + i] = prefill_token_key(raw_tokens[i]);
        }

        // Decoder prefill with LCP reuse
        let t0 = get_time_ms();
        let prefill_len = total_seq - 1;

        let reused_prefill = prefill_lcp_len(&prev_prefill_keys, &prefill_keys, prefill_len);

        ctx.kv_cache.len = reused_prefill;
        let delta_prefill = prefill_len - reused_prefill;
        if delta_prefill > 0 {
            decoder::decoder_prefill(
                &ctx.model.decoder,
                &cfg,
                &mut ctx.kv_cache,
                &mut ctx.rope_cache,
                &mut ctx.dec_bufs,
                &input_embeds[reused_prefill * dim..],
                delta_prefill,
            );
        }

        // Save for next chunk
        prev_prefill_keys.clear();
        prev_prefill_keys.extend_from_slice(&prefill_keys[..prefill_len]);

        let prefill_ms = elapsed_ms(t0);
        ctx.perf_decode_ms += prefill_ms;

        let last_embed = &input_embeds[prefill_len * dim..(prefill_len + 1) * dim];
        let mut token = decoder::decoder_forward(
            &ctx.model.decoder,
            &cfg,
            &mut ctx.kv_cache,
            &mut ctx.rope_cache,
            &mut ctx.dec_bufs,
            last_embed,
        );

        // Autoregressive decode
        let t0 = get_time_ms();
        let mut chunk_tokens: Vec<i32> = Vec::new();
        let mut n_generated = 0;

        while n_generated < max_new_tokens {
            n_generated += 1;
            if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END {
                break;
            }
            chunk_tokens.push(token);
            unsafe {
                tok_embed_bf16_to_f32(&mut tmp_embed, tok_emb, token, dim);
            }
            token = decoder::decoder_forward(
                &ctx.model.decoder,
                &cfg,
                &mut ctx.kv_cache,
                &mut ctx.rope_cache,
                &mut ctx.dec_bufs,
                &tmp_embed,
            );
        }

        let decode_ms = elapsed_ms(t0);
        ctx.perf_decode_ms += decode_ms;

        // Update raw token history
        raw_tokens.truncate(n_prefix_tokens);
        let dropped_repeat_tokens = if ctx.loop_detect {
            suppress_repeat_token_runs(&raw_tokens, &mut chunk_tokens, STREAM_MAX_REPEAT_TOKEN_RUN)
        } else {
            0
        };
        raw_tokens.extend_from_slice(&chunk_tokens);

        // Streaming degeneracy detection
        if raw_tokens == prev_tail_snapshot {
            stale_count += 1;
        } else {
            stale_count = 0;
            prev_tail_snapshot = raw_tokens.clone();
        }
        let (best_reps, _) = stream_tail_repeat_blocks(&raw_tokens, STREAM_DEGEN_MAX_PERIOD);
        let is_degen = stale_count >= STREAM_STALE_CHUNKS
            || best_reps >= STREAM_DEGEN_MIN_REPEATS
            || dropped_repeat_tokens >= STREAM_DEGEN_DROP_RESET;

        if is_degen {
            if kernels::verbose() >= 2 {
                eprintln!(
                    "[stream degen] reset at chunk {} (stale={}, reps={})",
                    chunk_idx, stale_count, best_reps
                );
            }
            let carry = stable_text_tokens.len().min(STREAM_RESET_CARRY_TOKENS);
            let carry_start = stable_text_tokens.len() - carry;
            raw_tokens.clear();
            if carry > 0 {
                raw_tokens.push(TOKEN_ASR_TEXT);
                raw_tokens.extend_from_slice(&stable_text_tokens[carry_start..]);
            }
            prev_prefill_keys.clear();
            stale_count = 0;
            prev_tail_snapshot.clear();
        }

        // Periodic re-anchor: reset context every STREAM_RESET_INTERVAL_CHUNKS chunks
        if chunk_idx > 0 && chunk_idx % STREAM_RESET_INTERVAL_CHUNKS == 0 {
            if kernels::verbose() >= 2 {
                eprintln!("[stream reanchor] at chunk {}", chunk_idx);
            }
            let carry = stable_text_tokens.len().min(STREAM_RESET_CARRY_TOKENS);
            let carry_start = stable_text_tokens.len() - carry;
            raw_tokens.clear();
            if carry > 0 {
                raw_tokens.push(TOKEN_ASR_TEXT);
                raw_tokens.extend_from_slice(&stable_text_tokens[carry_start..]);
            }
            prev_prefill_keys.clear();
            stale_count = 0;
            prev_tail_snapshot.clear();
            enc_cache_base_windows += enc_cache.len();
            enc_cache.clear();
            enc_cached_seq_total = 0;
        }

        // Parse text region
        let text_start = if n_force_prompt_tokens == 0 {
            raw_tokens
                .iter()
                .position(|&t| t == TOKEN_ASR_TEXT)
                .map(|p| p + 1)
                .unwrap_or(0)
        } else {
            0
        };
        let n_text_tokens = raw_tokens.len().saturating_sub(text_start);

        // Fixed frontier
        let candidate_len = if is_final {
            n_text_tokens
        } else if chunk_idx >= unfixed_chunks {
            (n_text_tokens as i32 - rollback).max(0) as usize
        } else {
            0
        };

        // Monotonic commit
        let candidate_tokens = &raw_tokens[text_start..];
        let _lcp = stable_text_tokens
            .iter()
            .zip(candidate_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();

        let emit_from = stable_text_tokens.len();
        let emit_to = candidate_len.max(emit_from);

        for i in emit_from..emit_to {
            if i < candidate_tokens.len() {
                if i >= stable_text_tokens.len() {
                    stable_text_tokens.push(candidate_tokens[i]);
                }
                let piece_bytes = tokenizer.decode_bytes(candidate_tokens[i]);
                if let Some(ref cb) = ctx.token_cb {
                    cb(&String::from_utf8_lossy(piece_bytes));
                }
                ctx.perf_text_tokens += 1;
                result_bytes.extend_from_slice(piece_bytes);
            }
        }

        ctx.perf_total_ms += elapsed_ms(chunk_t0);
        chunk_idx += 1;
    }

    Some(String::from_utf8_lossy(&result_bytes).trim().to_string())
}// ========================================================================
// Incremental Streaming API
// ========================================================================

/// Encoder window cached output.
struct EncWindow {
    seq_len: usize,
    enc_output: Vec<f32>,
    row_keys: Vec<PrefillRowKey>,
}

/// Persistent state for incremental streaming transcription.
///
/// Create once, then call [`stream_push_audio`] each time new audio
/// arrives.  The state keeps encoder caches, token history, and decoder
/// prefill row keys so that only *new* work is performed per call.
pub struct StreamState {
    // Encoder
    enc_cache: Vec<EncWindow>,
    enc_cached_seq_total: usize,
    enc_cache_base_windows: usize,

    // Decoder token history
    raw_tokens: Vec<i32>,
    stable_text_tokens: Vec<i32>,
    result_bytes: Vec<u8>,

    // Prefill LCP reuse
    prev_prefill_keys: Vec<PrefillRowKey>,

    // Streaming robustness
    prev_tail_snapshot: Vec<i32>,
    stale_count: i32,

    // Lazy partial encoding: skip re-encoding every other chunk
    last_partial_cursor: usize,
    last_partial_enc: Vec<f32>,
    last_partial_keys: Vec<PrefillRowKey>,
    last_partial_seq: usize,

    // Audio cursor
    audio_cursor: usize,
    chunk_idx: i32,

    prompt_prepared: bool,
}

impl Default for StreamState {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamState {
    /// Create a new empty streaming state.
    pub fn new() -> Self {
        StreamState {
            enc_cache: Vec::new(),
            enc_cached_seq_total: 0,
            enc_cache_base_windows: 0,
            raw_tokens: Vec::new(),
            stable_text_tokens: Vec::new(),
            result_bytes: Vec::new(),
            prev_prefill_keys: Vec::new(),
            prev_tail_snapshot: Vec::new(),
            stale_count: 0,
            last_partial_cursor: 0,
            last_partial_enc: Vec::new(),
            last_partial_keys: Vec::new(),
            last_partial_seq: 0,
            audio_cursor: 0,
            chunk_idx: 0,
            prompt_prepared: false,
        }
    }

    /// Reset state for a new streaming window (e.g., after 30s limit).
    pub fn reset(&mut self) {
        self.enc_cache.clear();
        self.enc_cached_seq_total = 0;
        self.enc_cache_base_windows = 0;
        self.raw_tokens.clear();
        self.stable_text_tokens.clear();
        self.result_bytes.clear();
        self.prev_prefill_keys.clear();
        self.prev_tail_snapshot.clear();
        self.stale_count = 0;
        self.last_partial_cursor = 0;
        self.last_partial_enc.clear();
        self.last_partial_keys.clear();
        self.last_partial_seq = 0;
        self.audio_cursor = 0;
        self.chunk_idx = 0;
        // Keep tokenizer and prompt_prepared
    }

    /// Get the current stable transcription result.
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.result_bytes).into_owned()
    }

    /// Get how many samples have been processed so far.
    pub fn audio_cursor(&self) -> usize {
        self.audio_cursor
    }
}

/// Process all available new audio incrementally.
///
/// `samples` is the **full** audio buffer accumulated so far (16 kHz mono f32).
/// Processes all full chunks from `state.audio_cursor` to end of `samples`.
/// When `finalize` is true, also processes any remaining partial chunk and
/// emits all rollback-buffered tokens.
/// Returns the newly emitted text delta (if any).
pub fn stream_push_audio(
    ctx: &mut QwenCtx,
    samples: &[f32],
    state: &mut StreamState,
    finalize: bool,
) -> Option<String> {
    let mut cfg = ctx.model.config.clone();
    // Fold the per-ctx encoder-window override (CLI --enc-window-sec) into the
    // working config clone, exactly where it used to live on ctx.config.
    if let Some(w) = ctx.enc_n_window_infer_override {
        cfg.enc_n_window_infer = w;
    }
    let dim = cfg.dec_hidden;
    let chunk_samples = (ctx.stream_chunk_sec * SAMPLE_RATE as f32) as usize;
    let rollback = ctx.stream_rollback;
    let unfixed_chunks = ctx.stream_unfixed_chunks;
    let max_new_tokens = if ctx.stream_max_new_tokens > 0 {
        ctx.stream_max_new_tokens
    } else {
        32
    };

    // Tokenizer is resident in the shared model (loaded once); borrow it via a
    // separate Arc handle so the borrow doesn't tie up `ctx` across the &mut calls.
    let model = ctx.model.clone();
    let tokenizer = &model.tokenizer;

    if !state.prompt_prepared {
        if !ctx.prepare_prompt_tokens(tokenizer) {
            return None;
        }
        state.prompt_prepared = true;
    }

    // Check if we have enough audio for at least one chunk (or finalizing)
    let available = samples.len().saturating_sub(state.audio_cursor);
    if available < chunk_samples && !finalize {
        return Some(String::new());
    }
    if available == 0 {
        return Some(String::new());
    }

    let enc_window_frames = cfg.enc_n_window_infer.clamp(100, 800);
    let enc_window_samples = enc_window_frames * HOP_LENGTH;
    let tok_emb = ctx.model.decoder.tok_embeddings_bf16;
    let mut tmp_embed = vec![0.0f32; dim];
    let mut delta_bytes: Vec<u8> = Vec::new();

    // ---- Process full chunks, plus remainder if finalizing ----
    while state.audio_cursor < samples.len() {
        let remaining = samples.len() - state.audio_cursor;
        if remaining < chunk_samples && !finalize {
            break; // Wait for more audio
        }

        let chunk_t0 = get_time_ms();
        state.audio_cursor = (state.audio_cursor + chunk_samples).min(samples.len());
        let is_final = finalize && state.audio_cursor >= samples.len();

    // ---- Encoder: only encode new windows ----
    let t0 = get_time_ms();
    let full_end = (state.audio_cursor / enc_window_samples) * enc_window_samples;

    // Cache newly completed windows (base offset accounts for windows cleared on re-anchor)
        while (state.enc_cache_base_windows + state.enc_cache.len()) * enc_window_samples < full_end
        {
        let ws = (state.enc_cache_base_windows + state.enc_cache.len()) * enc_window_samples;
        let (mel, mel_frames) = audio::mel_spectrogram(&samples[ws..ws + enc_window_samples])?;
            let (win_enc, win_seq) =
                ctx.model.encoder
                    .forward(&cfg, &mel, mel_frames, Some(&mut ctx.enc_bufs))?;
            let row_keys = prefill_embed_keys(&win_enc, win_seq, dim);
        state.enc_cached_seq_total += win_seq;
            state.enc_cache.push(EncWindow {
                seq_len: win_seq,
                enc_output: win_enc,
                row_keys,
            });
    }

    // Encode partial tail — with lazy re-encoding for LCP optimization.
    // Only re-encode when enough new audio has accumulated (every 2 chunks),
    // on the first chunk, or when finalizing. On skip chunks, the reused
    // encoder output gives near-perfect LCP matching, cutting prefill cost.
    let enc_update_threshold = chunk_samples * 2;
    let partial_age = state.audio_cursor.saturating_sub(state.last_partial_cursor);
        let need_encode =
            state.last_partial_cursor == 0 || partial_age >= enc_update_threshold || is_final;

    let partial_seq;
    let partial_enc;
        let partial_keys;
    if need_encode && full_end < state.audio_cursor {
            if let Some((mel, mel_frames)) =
                audio::mel_spectrogram(&samples[full_end..state.audio_cursor])
            {
                if let Some((enc, seq)) =
                    ctx.model.encoder
                        .forward(&cfg, &mel, mel_frames, Some(&mut ctx.enc_bufs))
                {
                partial_seq = seq;
                    partial_keys = prefill_embed_keys(&enc, seq, dim);
                partial_enc = enc;
                state.last_partial_cursor = state.audio_cursor;
                state.last_partial_enc = partial_enc.clone();
                    state.last_partial_keys = partial_keys.clone();
                state.last_partial_seq = partial_seq;
            } else {
                partial_seq = state.last_partial_seq;
                partial_enc = state.last_partial_enc.clone();
                    partial_keys = state.last_partial_keys.clone();
            }
        } else {
            partial_seq = state.last_partial_seq;
            partial_enc = state.last_partial_enc.clone();
                partial_keys = state.last_partial_keys.clone();
        }
    } else if full_end < state.audio_cursor {
        // Reuse previous partial encoding (skip chunk)
        partial_seq = state.last_partial_seq;
        partial_enc = state.last_partial_enc.clone();
            partial_keys = state.last_partial_keys.clone();
    } else {
        partial_seq = 0;
        partial_enc = Vec::new();
            partial_keys = Vec::new();
    }

    let enc_seq_len = state.enc_cached_seq_total + partial_seq;
    if enc_seq_len == 0 {
        state.chunk_idx += 1;
        return Some(String::new());
    }

    let enc_ms = elapsed_ms(t0);
    ctx.perf_encode_ms += enc_ms;

    // ---- Prefix rollback ----
    let n_prefix_tokens = if ctx.past_text_conditioning
        && state.chunk_idx >= unfixed_chunks
        && !state.raw_tokens.is_empty()
    {
        (state.raw_tokens.len() as i32 - rollback).max(0) as usize
    } else {
        0
    };

    // ---- Build input embeddings ----
    let n_prompt_tokens = ctx.prompt_tokens.as_ref().map_or(0, |t| t.len());
    let n_force_prompt_tokens = ctx.force_prompt_tokens.as_ref().map_or(0, |t| t.len());
    let prefix_len = PREFIX_HEAD.len() + n_prompt_tokens + PREFIX_TAIL.len();
    let suffix_len = SUFFIX_BASE.len() + n_force_prompt_tokens;
    let total_seq = prefix_len + enc_seq_len + suffix_len + n_prefix_tokens;

    let mut input_embeds = vec![0.0f32; total_seq * dim];
        let mut prefill_keys = vec![PrefillRowKey { a: 0, b: 0 }; total_seq];
    let mut off = 0;

    for &tok in PREFIX_HEAD {
            unsafe {
                tok_embed_bf16_to_f32(
                    &mut input_embeds[off * dim..(off + 1) * dim],
                    tok_emb,
                    tok,
                    dim,
                );
            }
            prefill_keys[off] = prefill_token_key(tok);
        off += 1;
    }
    if let Some(ref ptoks) = ctx.prompt_tokens {
        for &tok in ptoks {
                unsafe {
                    tok_embed_bf16_to_f32(
                        &mut input_embeds[off * dim..(off + 1) * dim],
                        tok_emb,
                        tok,
                        dim,
                    );
                }
                prefill_keys[off] = prefill_token_key(tok);
            off += 1;
        }
    }
    for &tok in PREFIX_TAIL {
            unsafe {
                tok_embed_bf16_to_f32(
                    &mut input_embeds[off * dim..(off + 1) * dim],
                    tok_emb,
                    tok,
                    dim,
                );
            }
            prefill_keys[off] = prefill_token_key(tok);
        off += 1;
    }

        let mut enc_key_off = 0;
        for w in &state.enc_cache {
            prefill_keys[prefix_len + enc_key_off..prefix_len + enc_key_off + w.seq_len]
                .copy_from_slice(&w.row_keys);
            enc_key_off += w.seq_len;
        }
        if partial_seq > 0 {
            prefill_keys[prefix_len + enc_key_off..prefix_len + enc_key_off + partial_seq]
                .copy_from_slice(&partial_keys);
        }

        let mut enc_embed_off = 0;
        for w in &state.enc_cache {
            let n = w.seq_len * dim;
            input_embeds
                [(prefix_len + enc_embed_off) * dim..(prefix_len + enc_embed_off) * dim + n]
                .copy_from_slice(&w.enc_output);
            enc_embed_off += w.seq_len;
        }
        if partial_seq > 0 {
            let n = partial_seq * dim;
            input_embeds
                [(prefix_len + enc_embed_off) * dim..(prefix_len + enc_embed_off) * dim + n]
                .copy_from_slice(&partial_enc);
    }

    let suffix_off = prefix_len + enc_seq_len;
    for (i, &tok) in SUFFIX_BASE.iter().enumerate() {
            unsafe {
                tok_embed_bf16_to_f32(
            &mut input_embeds[(suffix_off + i) * dim..(suffix_off + i + 1) * dim],
                    tok_emb,
                    tok,
                    dim,
                );
            }
            prefill_keys[suffix_off + i] = prefill_token_key(tok);
    }
    if let Some(ref ftoks) = ctx.force_prompt_tokens {
        for (i, &tok) in ftoks.iter().enumerate() {
                unsafe {
                    tok_embed_bf16_to_f32(
                &mut input_embeds[(suffix_off + SUFFIX_BASE.len() + i) * dim
                    ..(suffix_off + SUFFIX_BASE.len() + i + 1) * dim],
                        tok_emb,
                        tok,
                        dim,
                    );
                }
                prefill_keys[suffix_off + SUFFIX_BASE.len() + i] = prefill_token_key(tok);
        }
    }

    let text_off = suffix_off + suffix_len;
    for i in 0..n_prefix_tokens {
            unsafe {
                tok_embed_bf16_to_f32(
            &mut input_embeds[(text_off + i) * dim..(text_off + i + 1) * dim],
                    tok_emb,
                    state.raw_tokens[i],
                    dim,
                );
            }
            prefill_keys[text_off + i] = prefill_token_key(state.raw_tokens[i]);
    }

    // ---- Decoder prefill with LCP reuse ----
    let t0 = get_time_ms();
    let prefill_len = total_seq - 1;

        let reused_prefill = prefill_lcp_len(&state.prev_prefill_keys, &prefill_keys, prefill_len);

    ctx.kv_cache.len = reused_prefill;
    let delta_prefill = prefill_len - reused_prefill;
    if delta_prefill > 0 {
        decoder::decoder_prefill(
                &ctx.model.decoder,
                &cfg,
                &mut ctx.kv_cache,
                &mut ctx.rope_cache,
            &mut ctx.dec_bufs,
            &input_embeds[reused_prefill * dim..],
            delta_prefill,
        );
    }

    let last_embed = &input_embeds[prefill_len * dim..(prefill_len + 1) * dim];
    let mut token = decoder::decoder_forward(
            &ctx.model.decoder,
            &cfg,
            &mut ctx.kv_cache,
            &mut ctx.rope_cache,
            &mut ctx.dec_bufs,
            last_embed,
    );

    // Save for next chunk
        state.prev_prefill_keys.clear();
        state
            .prev_prefill_keys
            .extend_from_slice(&prefill_keys[..prefill_len]);

    let prefill_ms = elapsed_ms(t0);
    ctx.perf_decode_ms += prefill_ms;

    if kernels::verbose() >= 2 {
        eprintln!(
            "  [stream chunk {}] encoder: {:.0}ms, prefill: {}/{} reused ({:.0}ms, delta={})",
            state.chunk_idx, enc_ms, reused_prefill, prefill_len, prefill_ms, delta_prefill
        );
    }

    // ---- Autoregressive decode ----
    let t0 = get_time_ms();
    let mut chunk_tokens: Vec<i32> = Vec::new();
    let mut n_generated = 0;

    while n_generated < max_new_tokens {
        n_generated += 1;
            if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END {
                break;
            }
        chunk_tokens.push(token);
            unsafe {
                tok_embed_bf16_to_f32(&mut tmp_embed, tok_emb, token, dim);
            }
        token = decoder::decoder_forward(
                &ctx.model.decoder,
                &cfg,
                &mut ctx.kv_cache,
                &mut ctx.rope_cache,
                &mut ctx.dec_bufs,
                &tmp_embed,
        );
    }

    let decode_ms = elapsed_ms(t0);
    ctx.perf_decode_ms += decode_ms;

    // ---- Detect speech end (decoder produced EOT on silence) ----
    // When chunk_tokens is empty, the decoder saw silence/end-of-speech.
    // Commit ALL remaining rollback-buffered tokens BEFORE truncation,
    // since truncate will remove them.
    let speech_ended = chunk_tokens.is_empty()
        && !state.raw_tokens.is_empty()
        && state.chunk_idx >= unfixed_chunks;

    if speech_ended {
        // Emit remaining rollback tokens from current raw_tokens (before truncation)
        let text_start = if n_force_prompt_tokens == 0 {
                state
                    .raw_tokens
                    .iter()
                    .position(|&t| t == TOKEN_ASR_TEXT)
                .map(|p| p + 1)
                .unwrap_or(0)
        } else {
            0
        };
        let candidate_tokens = &state.raw_tokens[text_start..];
        let n_text = candidate_tokens.len();
        let emit_from = state.stable_text_tokens.len();
        for i in emit_from..n_text {
            if i < candidate_tokens.len() {
                if i >= state.stable_text_tokens.len() {
                    state.stable_text_tokens.push(candidate_tokens[i]);
                }
                let piece_bytes = tokenizer.decode_bytes(candidate_tokens[i]);
                if let Some(ref cb) = ctx.token_cb {
                    cb(&String::from_utf8_lossy(piece_bytes));
                }
                ctx.perf_text_tokens += 1;
                state.result_bytes.extend_from_slice(piece_bytes);
                delta_bytes.extend_from_slice(piece_bytes);
            }
        }
    }

    // ---- Update raw token history ----
    state.raw_tokens.truncate(n_prefix_tokens);
    let dropped_repeat_tokens = if ctx.loop_detect {
        suppress_repeat_token_runs(&state.raw_tokens, &mut chunk_tokens, STREAM_MAX_REPEAT_TOKEN_RUN)
    } else {
        0
    };
    state.raw_tokens.extend_from_slice(&chunk_tokens);

    // ---- Streaming degeneracy detection ----
    if !speech_ended {
        if state.raw_tokens == state.prev_tail_snapshot {
            state.stale_count += 1;
        } else {
            state.stale_count = 0;
            state.prev_tail_snapshot = state.raw_tokens.clone();
        }
            let (best_reps, _) =
                stream_tail_repeat_blocks(&state.raw_tokens, STREAM_DEGEN_MAX_PERIOD);
            let is_degen = state.stale_count >= STREAM_STALE_CHUNKS
                || best_reps >= STREAM_DEGEN_MIN_REPEATS
                || dropped_repeat_tokens >= STREAM_DEGEN_DROP_RESET;

        if is_degen {
            if kernels::verbose() >= 2 {
                    eprintln!(
                        "[stream degen] reset at chunk {} (stale={}, reps={})",
                        state.chunk_idx, state.stale_count, best_reps
                    );
            }
                let carry = state
                    .stable_text_tokens
                    .len()
                    .min(STREAM_RESET_CARRY_TOKENS);
            let carry_start = state.stable_text_tokens.len() - carry;
            state.raw_tokens.clear();
            if carry > 0 {
                state.raw_tokens.push(TOKEN_ASR_TEXT);
                    state
                        .raw_tokens
                        .extend_from_slice(&state.stable_text_tokens[carry_start..]);
            }
                state.prev_prefill_keys.clear();
            state.stale_count = 0;
            state.prev_tail_snapshot.clear();
            if state.enc_cache.len() >= STREAM_MAX_ENC_WINDOWS {
                state.enc_cache_base_windows += state.enc_cache.len();
                state.enc_cache.clear();
                state.enc_cached_seq_total = 0;
            }
        }

        // Periodic re-anchor: reset context every STREAM_RESET_INTERVAL_CHUNKS chunks
        if state.chunk_idx > 0 && state.chunk_idx % STREAM_RESET_INTERVAL_CHUNKS == 0 {
            if kernels::verbose() >= 2 {
                eprintln!("[stream reanchor] at chunk {}", state.chunk_idx);
            }
                let carry = state
                    .stable_text_tokens
                    .len()
                    .min(STREAM_RESET_CARRY_TOKENS);
            let carry_start = state.stable_text_tokens.len() - carry;
            state.raw_tokens.clear();
            if carry > 0 {
                state.raw_tokens.push(TOKEN_ASR_TEXT);
                    state
                        .raw_tokens
                        .extend_from_slice(&state.stable_text_tokens[carry_start..]);
            }
                state.prev_prefill_keys.clear();
            state.stale_count = 0;
            state.prev_tail_snapshot.clear();
            if state.enc_cache.len() >= STREAM_MAX_ENC_WINDOWS {
                state.enc_cache_base_windows += state.enc_cache.len();
                state.enc_cache.clear();
                state.enc_cached_seq_total = 0;
            }
        }
    }

    // ---- Parse text region and emit stable tokens (non-speech-ended case) ----
    if !speech_ended {
        let text_start = if n_force_prompt_tokens == 0 {
                state
                    .raw_tokens
                    .iter()
                    .position(|&t| t == TOKEN_ASR_TEXT)
                .map(|p| p + 1)
                .unwrap_or(0)
        } else {
            0
        };
        let n_text_tokens = state.raw_tokens.len().saturating_sub(text_start);

        let candidate_len = if is_final {
            n_text_tokens
        } else if state.chunk_idx >= unfixed_chunks {
            (n_text_tokens as i32 - rollback).max(0) as usize
        } else {
            0
        };

        let candidate_tokens = &state.raw_tokens[text_start..];
        let emit_from = state.stable_text_tokens.len();
        let emit_to = candidate_len.max(emit_from);

        for i in emit_from..emit_to {
            if i < candidate_tokens.len() {
                if i >= state.stable_text_tokens.len() {
                    state.stable_text_tokens.push(candidate_tokens[i]);
                }
                let piece_bytes = tokenizer.decode_bytes(candidate_tokens[i]);
                if let Some(ref cb) = ctx.token_cb {
                    cb(&String::from_utf8_lossy(piece_bytes));
                }
                ctx.perf_text_tokens += 1;
                state.result_bytes.extend_from_slice(piece_bytes);
                delta_bytes.extend_from_slice(piece_bytes);
            }
        }
    }

        ctx.perf_total_ms += elapsed_ms(chunk_t0);
        state.chunk_idx += 1;

        // Stop processing after speech ends — no point encoding more silence
        if speech_ended {
            break;
        }
    } // end while loop

    Some(String::from_utf8_lossy(&delta_bytes).into_owned())
}

#[cfg(test)]
mod loop_tests {
    use super::*;

    // ── Detection: stream_tail_repeat_blocks (ported from antirez's C) ──
    // Mirrors the real Arabic monster loop: a sentence-length PHRASE repeated many times.
    // The phrase is > 6 tokens, so the stream path's tuned period-6 MISSES it while the
    // batch detector's wider period CATCHES it — the exact gap this feature closes.
    #[test]
    fn detects_phrase_loop_at_wide_period_but_misses_at_six() {
        let phrase: Vec<i32> = (100..109).collect(); // 9-token "phrase"
        let mut toks = Vec::new();
        for _ in 0..50 {
            toks.extend_from_slice(&phrase);
        }
        assert!(
            stream_tail_repeat_blocks(&toks, 32).0 >= 4,
            "wide period (batch) must flag a 9-token phrase loop"
        );
        assert!(
            stream_tail_repeat_blocks(&toks, 6).0 < 4,
            "period-6 (stream) misses a >6-token phrase loop — why batch needs the wider period"
        );
    }

    #[test]
    fn clean_text_is_not_flagged() {
        let toks: Vec<i32> = (0..300).collect(); // all distinct, no tail repetition
        assert!(stream_tail_repeat_blocks(&toks, 32).0 < 4);
    }

    // Coverage guard: is_repetition_loop flags a runaway loop (the repeat dominates) but NOT a
    // legitimate brief refrain (the same block repeated a few times within mostly-distinct text).
    #[test]
    fn repetition_loop_flags_dominant_loop_not_brief_refrain() {
        let phrase: Vec<i32> = (100..109).collect(); // 9-token block

        // Dominant loop: phrase × 50 → covers ~100% of the tokens → flagged.
        let mut loopy = Vec::new();
        for _ in 0..50 {
            loopy.extend_from_slice(&phrase);
        }
        assert!(is_repetition_loop(&loopy, 32, 4));

        // Brief refrain: 300 distinct tokens then the phrase ×4 at the tail. The block repeats
        // >= 4 times (so the raw detector fires), but covers only 36/336 ≈ 11% → NOT flagged.
        let mut refrain: Vec<i32> = (1000..1300).collect();
        for _ in 0..4 {
            refrain.extend_from_slice(&phrase);
        }
        assert!(stream_tail_repeat_blocks(&refrain, 32).0 >= 4, "raw detector still fires");
        assert!(!is_repetition_loop(&refrain, 32, 4), "but coverage gate rejects the refrain");
    }

    // ── Sync to C: suppress_repeat_token_runs (QWEN_STREAM_MAX_REPEAT_TOKEN_RUN) ──
    #[test]
    fn suppress_trims_single_token_run_to_max() {
        let mut chunk = vec![9i32; 20];
        let dropped = suppress_repeat_token_runs(&[1, 2, 3], &mut chunk, 12);
        assert_eq!(chunk.len(), 12, "a 20-long run of one token trims to max_run=12");
        assert!(chunk.iter().all(|&t| t == 9));
        // 20 - 12 = 8 dropped, which reaches STREAM_DEGEN_DROP_RESET → forces a re-anchor (C parity).
        assert_eq!(dropped, 8);
        assert!(dropped >= STREAM_DEGEN_DROP_RESET);
    }

    #[test]
    fn suppress_continues_run_from_prefix_tail() {
        // prefix already ends in three 9s, so only 9 more fit before the run hits 12.
        let mut chunk = vec![9i32; 20];
        suppress_repeat_token_runs(&[9, 9, 9], &mut chunk, 12);
        assert_eq!(chunk.len(), 9);
    }

    #[test]
    fn suppress_is_noop_when_max_run_below_one() {
        let mut chunk = vec![9i32; 20];
        suppress_repeat_token_runs(&[], &mut chunk, 0);
        assert_eq!(chunk.len(), 20);
    }

    // ── Word-boundary halving: find_split_point lands in a silence gap ──
    // The recovery halves a degenerate clip at find_split_point(midpoint, ±search) so the cut
    // falls between words (lowest energy), not mid-word.
    #[test]
    fn find_split_point_lands_in_silence_gap() {
        let sr = SAMPLE_RATE as usize;
        let mut samples = vec![1.0f32; 10 * sr]; // 10s of "speech"
        let gap_lo = 5 * sr - sr / 4; // 0.5s silent gap centered at 5s
        let gap_hi = 5 * sr + sr / 4;
        for s in samples[gap_lo..gap_hi].iter_mut() {
            *s = 0.0;
        }
        let split = find_split_point(&samples, 5 * sr, 3.0);
        assert!(
            split >= gap_lo && split <= gap_hi,
            "split {split} should land in the silent gap [{gap_lo},{gap_hi}], not mid-word"
        );
    }
}
