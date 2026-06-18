//! Single-process, single-thread **16-way batched** offline transcription.
//!
//! Decodes up to `max_batch` independent ~30 s segments *concurrently*, streaming
//! each decoder weight from memory **once per batch** (weight-stationary INT8
//! matvec — see [`kernels::matvec_int8_batched`]) instead of once per segment.
//! The autoregressive single-token decode is bandwidth-bound on the ~0.6 B INT8
//! weights, so reusing each weight row across N segments is an ~N× cut in weight
//! traffic — the whole point of batching here.
//!
//! ## Bit-identical guarantee
//! Every segment's arithmetic is UNCHANGED versus running it alone through
//! [`crate::transcribe::transcribe_audio`]:
//!   * The encoder, the input-embedding construction, and the decoder *prefill*
//!     run per-segment through the exact same code (`Encoder::forward`,
//!     `decoder::decoder_prefill`).
//!   * The batched single-token decode reproduces `decoder::decoder_forward`'s
//!     single-thread (aarch64 INT8) path operation-for-operation. The only batched
//!     primitives are the four projection matvecs and the lm-head argmax, whose
//!     INT8 `sdot` reduction is **exact integer arithmetic** — so regrouping it
//!     across a batch cannot change any value (see [`kernels::matvec_int8_batched`]).
//!     Per-segment ops (rms-norm, RoPE, attention over that segment's own KV cache,
//!     SwiGLU) call the identical kernels per lane.
//! Segments that hit EOS (or the token cap) simply drop out of the active set;
//! that never touches a still-active segment's computation, so each segment's
//! token stream — and therefore its text — is identical to the sequential path.
//!
//! Only the default offline configuration is batched (no `--past-text`, no
//! aligner, no streaming token callback, aarch64). Anything else transparently
//! falls back to [`crate::transcribe::transcribe_audio`].

use crate::audio;
use crate::config::*;
use crate::context::QwenCtx;
use crate::decoder::{self, tok_embed_bf16_to_f32, DecoderBuffers, KvCache, RopeCache};
use crate::kernels;
use crate::transcribe::{
    find_split_point, segment_is_degenerate, should_insert_boundary_space, transcribe_with_recovery,
    TranscriptSegment, PREFIX_HEAD, PREFIX_TAIL, SUFFIX_BASE,
};

const MAX_DECODE_TOKENS: i32 = 2048;

/// Result of decoding one segment through the batched loop — the trimmed text plus
/// the signals `transcribe_segment` exposes for loop/degeneracy recovery.
struct SegOutcome {
    text: String,
    text_tokens: Vec<i32>,
    /// True when the segment hit the token cap without emitting EOS (the
    /// high-confidence degeneracy signal; see `transcribe_segment`).
    maxed: bool,
    /// True when the batched decode was aborted early because the window was
    /// already a repetition loop — it must go to sequential recovery (which is
    /// authoritative and re-decodes from scratch), so its partial text is unused.
    aborted: bool,
}

/// Per-segment decode state carried through the batched autoregressive loop.
struct SegState {
    kv: KvCache,
    /// Current decoder input embedding (the residual-stream seed for the next
    /// step): the last prefill embed for the first step, then `tok_embed(token)`.
    x: Vec<f32>,
    token: i32,
    n_generated: i32,
    past_asr_text: bool,
    text_bytes: Vec<u8>,
    /// Text token ids (same set `transcribe_segment` records) for loop detection.
    text_tokens: Vec<i32>,
    aborted: bool,
    done: bool,
}

/// Reusable scratch for one batched decode step, sized for `max_batch` lanes.
struct BatchScratch {
    xs: Vec<f32>,        // B * dim   — residual stream
    x_norm: Vec<f32>,    // B * dim
    q: Vec<f32>,         // B * q_dim
    k: Vec<f32>,         // B * kv_dim
    v: Vec<f32>,         // B * kv_dim
    attn_out: Vec<f32>,  // B * q_dim
    gate_up: Vec<f32>,   // B * 2*intermediate
    ffn: Vec<f32>,       // B * intermediate
    // int8 quantization staging (largest in-dim is intermediate for the down proj)
    xi8: Vec<i8>,        // B * max(dim, q_dim, intermediate)
    xsc: Vec<f32>,       // B
    best: Vec<usize>,    // B
}

impl BatchScratch {
    fn new(b: usize, cfg: &QwenConfig) -> Self {
        let dim = cfg.dec_hidden;
        let q_dim = cfg.dec_heads * cfg.dec_head_dim;
        let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
        let inter = cfg.dec_intermediate;
        let max_in = dim.max(q_dim).max(inter);
        BatchScratch {
            xs: vec![0.0; b * dim],
            x_norm: vec![0.0; b * dim],
            q: vec![0.0; b * q_dim],
            k: vec![0.0; b * kv_dim],
            v: vec![0.0; b * kv_dim],
            attn_out: vec![0.0; b * q_dim],
            gate_up: vec![0.0; b * 2 * inter],
            ffn: vec![0.0; b * inter],
            xi8: vec![0; b * max_in],
            xsc: vec![0.0; b],
            best: vec![0; b],
        }
    }
}

/// Quantize each lane's `in_dim`-vector (stride `src_stride` in `src`) into the
/// contiguous `xi8`/`xsc` staging buffers — exactly `kernels::quantize_f32_to_int8`
/// per lane, which is what the sequential INT8 path does before every matvec.
fn quantize_lanes(src: &[f32], b: usize, in_dim: usize, xi8: &mut [i8], xsc: &mut [f32]) {
    for bi in 0..b {
        let (q, s) = kernels::quantize_f32_to_int8(&src[bi * in_dim..bi * in_dim + in_dim]);
        xi8[bi * in_dim..bi * in_dim + in_dim].copy_from_slice(&q);
        xsc[bi] = s;
    }
}

/// One batched single-token decode step over the `active` segments. Reproduces
/// [`decoder::decoder_forward`]'s aarch64 single-thread path per lane, with the
/// four projection matvecs and the lm-head argmax run weight-stationary across
/// all lanes. Writes each active segment's next `token` and advances its KV len.
#[allow(clippy::too_many_arguments)]
fn batched_decode_step(
    decoder: &decoder::Decoder,
    cfg: &QwenConfig,
    rope: &mut RopeCache,
    segs: &mut [SegState],
    active: &[usize],
    sc: &mut BatchScratch,
) {
    let b = active.len();
    if b == 0 {
        return;
    }
    let dim = cfg.dec_hidden;
    let n_heads = cfg.dec_heads;
    let n_kv_heads = cfg.dec_kv_heads;
    let head_dim = cfg.dec_head_dim;
    let intermediate = cfg.dec_intermediate;
    let eps = cfg.dec_rms_norm_eps;
    let theta = cfg.dec_rope_theta;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Seed the residual stream from each lane's current input embedding, and grow
    // each KV cache / the shared RoPE table for this step's position (mirrors the
    // per-call grow + rope.ensure in decoder_forward).
    let mut max_pos = 0usize;
    for (bi, &si) in active.iter().enumerate() {
        sc.xs[bi * dim..bi * dim + dim].copy_from_slice(&segs[si].x);
        let pos = segs[si].kv.len;
        if pos >= segs[si].kv.max_seq {
            segs[si].kv.grow(pos + 1024);
        }
        max_pos = max_pos.max(pos);
    }
    rope.ensure(max_pos + 1, head_dim, theta);

    for layer_idx in 0..decoder.layers.len() {
        let layer = &decoder.layers[layer_idx];

        // input rms-norm  (per lane)
        for bi in 0..b {
            kernels::rms_norm(
                &mut sc.x_norm[bi * dim..bi * dim + dim],
                &sc.xs[bi * dim..bi * dim + dim],
                &layer.input_norm,
                1,
                dim,
                eps,
            );
        }
        // quantize x_norm once, reuse for q/k/v  (matches linear_nobias_int8_qkv)
        quantize_lanes(&sc.x_norm, b, dim, &mut sc.xi8, &mut sc.xsc);
        kernels::matvec_int8_batched(&mut sc.q, &sc.xi8[..b * dim], &sc.xsc,
            &layer.wq_int8, &layer.wq_int8_scales, b, dim, q_dim, false);
        kernels::matvec_int8_batched(&mut sc.k, &sc.xi8[..b * dim], &sc.xsc,
            &layer.wk_int8, &layer.wk_int8_scales, b, dim, kv_dim, false);
        kernels::matvec_int8_batched(&mut sc.v, &sc.xi8[..b * dim], &sc.xsc,
            &layer.wv_int8, &layer.wv_int8_scales, b, dim, kv_dim, false);

        // per-lane: q/k norm, RoPE, KV write, causal attention over own cache
        for (bi, &si) in active.iter().enumerate() {
            let pos = segs[si].kv.len;
            let qd = &mut sc.q[bi * q_dim..bi * q_dim + q_dim];
            kernels::rms_norm_per_head(qd, &layer.q_norm_weight, 1, n_heads, head_dim, eps);
            let kd = &mut sc.k[bi * kv_dim..bi * kv_dim + kv_dim];
            kernels::rms_norm_per_head(kd, &layer.k_norm_weight, 1, n_kv_heads, head_dim, eps);

            let rope_cos = rope.cos_at(pos);
            let rope_sin = rope.sin_at(pos);
            kernels::apply_rope_neox(
                &mut sc.q[bi * q_dim..bi * q_dim + q_dim], rope_cos, rope_sin, 1, n_heads, head_dim);
            kernels::apply_rope_neox(
                &mut sc.k[bi * kv_dim..bi * kv_dim + kv_dim], rope_cos, rope_sin, 1, n_kv_heads, head_dim);

            segs[si].kv.k_write_pos(layer_idx, pos, &sc.k[bi * kv_dim..bi * kv_dim + kv_dim]);
            segs[si].kv.v_write_pos(layer_idx, pos, &sc.v[bi * kv_dim..bi * kv_dim + kv_dim]);

            let total_seq = pos + 1;
            let k_base = segs[si].kv.k_layer_base(layer_idx);
            let v_base = segs[si].kv.v_layer_base(layer_idx);
            let head_stride = segs[si].kv.head_stride();
            kernels::causal_attention(
                &mut sc.attn_out[bi * q_dim..bi * q_dim + q_dim],
                &sc.q[bi * q_dim..bi * q_dim + q_dim],
                k_base, v_base, head_stride,
                1, total_seq, n_heads, n_kv_heads, head_dim, scale, pos,
            );
        }

        // o-projection with fused residual add: xs += attn_out @ wo
        quantize_lanes(&sc.attn_out, b, q_dim, &mut sc.xi8, &mut sc.xsc);
        kernels::matvec_int8_batched(&mut sc.xs, &sc.xi8[..b * q_dim], &sc.xsc,
            &layer.wo_int8, &layer.wo_int8_scales, b, q_dim, dim, true);

        // post-attention rms-norm
        for bi in 0..b {
            kernels::rms_norm(
                &mut sc.x_norm[bi * dim..bi * dim + dim],
                &sc.xs[bi * dim..bi * dim + dim],
                &layer.post_attn_norm,
                1, dim, eps,
            );
        }
        // gate_up + SwiGLU
        quantize_lanes(&sc.x_norm, b, dim, &mut sc.xi8, &mut sc.xsc);
        kernels::matvec_int8_batched(&mut sc.gate_up, &sc.xi8[..b * dim], &sc.xsc,
            &layer.gate_up_int8, &layer.gate_up_int8_scales, b, dim, 2 * intermediate, false);
        for bi in 0..b {
            let gu = &sc.gate_up[bi * 2 * intermediate..bi * 2 * intermediate + 2 * intermediate];
            let out = &mut sc.ffn[bi * intermediate..bi * intermediate + intermediate];
            for j in 0..intermediate {
                let g = gu[2 * j];
                let u = gu[2 * j + 1];
                out[j] = g / (1.0 + (-g).exp()) * u;
            }
        }
        // down-projection with fused residual add: xs += ffn @ down
        quantize_lanes(&sc.ffn, b, intermediate, &mut sc.xi8, &mut sc.xsc);
        kernels::matvec_int8_batched(&mut sc.xs, &sc.xi8[..b * intermediate], &sc.xsc,
            &layer.down_int8, &layer.down_int8_scales, b, intermediate, dim, true);
    }

    // final rms-norm + lm-head argmax (weight-stationary across lanes)
    for bi in 0..b {
        kernels::rms_norm(
            &mut sc.x_norm[bi * dim..bi * dim + dim],
            &sc.xs[bi * dim..bi * dim + dim],
            &decoder.norm,
            1, dim, eps,
        );
    }
    quantize_lanes(&sc.x_norm, b, dim, &mut sc.xi8, &mut sc.xsc);
    let lm_out_dim = cfg.lm_head_dim();
    let lm_int8 = decoder
        .lm_head_int8
        .as_ref()
        .expect("batched decode requires INT8 lm_head (aarch64)");
    let lm_scales = decoder
        .lm_head_int8_scales
        .as_ref()
        .expect("batched decode requires INT8 lm_head scales (aarch64)");
    kernels::argmax_int8_batched(&mut sc.best, &sc.xi8[..b * dim], &sc.xsc,
        lm_int8, lm_scales, b, dim, lm_out_dim);

    for (bi, &si) in active.iter().enumerate() {
        segs[si].kv.len += 1;
        segs[si].token = sc.best[bi] as i32;
    }
}

/// Encode + prefill a single segment, returning its decode state seeded with the
/// last prefill embedding (the residual-stream seed for the first decode step).
/// Replicates `transcribe::transcribe_segment`'s embed construction + prefill
/// verbatim (past-text is always absent on this path), so the resulting KV cache
/// and first-step input are bit-identical to the sequential decoder.
#[allow(clippy::too_many_arguments)]
fn prepare_segment(
    ctx: &mut QwenCtx,
    cfg: &QwenConfig,
    seg_samples: &[f32],
    prefill_bufs: &mut DecoderBuffers,
    prefill_rope: &mut RopeCache,
) -> Option<SegState> {
    let dim = cfg.dec_hidden;
    let model = ctx.model.clone();

    let (mel, mel_frames) = audio::mel_spectrogram(seg_samples)?;
    let (enc_output, enc_seq_len) =
        model.encoder.forward(cfg, &mel, mel_frames, Some(&mut ctx.enc_bufs))?;

    let tok_emb = model.decoder.tok_embeddings_bf16;
    let n_prompt_tokens = ctx.prompt_tokens.as_ref().map_or(0, |t| t.len());
    let n_force_prompt_tokens = ctx.force_prompt_tokens.as_ref().map_or(0, |t| t.len());

    let prefix_len = PREFIX_HEAD.len() + n_prompt_tokens + PREFIX_TAIL.len();
    let suffix_len = SUFFIX_BASE.len() + n_force_prompt_tokens;
    let total_seq = prefix_len + enc_seq_len + suffix_len; // no past-text on this path

    let mut input_embeds = vec![0.0f32; total_seq * dim];

    let mut off = 0;
    for &tok in PREFIX_HEAD {
        unsafe { tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim) };
        off += 1;
    }
    if let Some(ref ptoks) = ctx.prompt_tokens {
        for &tok in ptoks {
            unsafe { tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim) };
            off += 1;
        }
    }
    for &tok in PREFIX_TAIL {
        unsafe { tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim) };
        off += 1;
    }
    for i in 0..enc_seq_len {
        input_embeds[(prefix_len + i) * dim..(prefix_len + i + 1) * dim]
            .copy_from_slice(&enc_output[i * dim..(i + 1) * dim]);
    }
    let suffix_off = prefix_len + enc_seq_len;
    for (i, &tok) in SUFFIX_BASE.iter().enumerate() {
        unsafe {
            tok_embed_bf16_to_f32(
                &mut input_embeds[(suffix_off + i) * dim..(suffix_off + i + 1) * dim], tok_emb, tok, dim)
        };
    }
    if let Some(ref ftoks) = ctx.force_prompt_tokens {
        for (i, &tok) in ftoks.iter().enumerate() {
            unsafe {
                tok_embed_bf16_to_f32(
                    &mut input_embeds[(suffix_off + SUFFIX_BASE.len() + i) * dim
                        ..(suffix_off + SUFFIX_BASE.len() + i + 1) * dim],
                    tok_emb, tok, dim)
            };
        }
    }

    // KV cache sized like a fresh ctx (2048), grown on demand during decode.
    let mut kv = KvCache::new(cfg.dec_layers, 2048, cfg.dec_kv_heads, cfg.dec_head_dim);
    kv.len = 0;
    let prefill_len = total_seq - 1;
    decoder::decoder_prefill(
        &model.decoder, cfg, &mut kv, prefill_rope, prefill_bufs, &input_embeds, prefill_len);

    let last_embed = input_embeds[prefill_len * dim..(prefill_len + 1) * dim].to_vec();
    let past_asr_text = n_force_prompt_tokens > 0;

    Some(SegState {
        kv,
        x: last_embed,
        token: 0,
        n_generated: 0,
        past_asr_text,
        text_bytes: Vec::new(),
        text_tokens: Vec::new(),
        aborted: false,
        done: false,
    })
}

/// Decode a group of ≤`max_batch` already-prepared segments to completion,
/// returning each segment's trimmed text in input order. The per-segment token
/// handling mirrors `transcribe::transcribe_segment`'s autoregressive loop.
fn decode_group(
    decoder: &decoder::Decoder,
    cfg: &QwenConfig,
    rope: &mut RopeCache,
    tokenizer: &crate::tokenizer::QwenTokenizer,
    mut segs: Vec<SegState>,
    // When true (segmented/clips path, loop_detect on), abort a lane the moment its
    // partial output is a repetition loop and hand it to sequential recovery instead
    // of decoding it to the 2048-token cap in the batch. Must be FALSE for the plain
    // transcribe_audio path, which has no recovery and uses the batched text directly.
    loop_detect: bool,
) -> Vec<SegOutcome> {
    let dim = cfg.dec_hidden;
    let b = segs.len();
    let mut sc = BatchScratch::new(b, cfg);

    // First step: every segment consumes its last prefill embed → first token.
    let active: Vec<usize> = (0..b).collect();
    batched_decode_step(decoder, cfg, rope, &mut segs, &active, &mut sc);

    loop {
        let mut next_active: Vec<usize> = Vec::with_capacity(b);
        for si in 0..b {
            if segs[si].done {
                continue;
            }
            segs[si].n_generated += 1;
            let token = segs[si].token;
            if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END {
                segs[si].done = true;
                continue;
            }
            if token == TOKEN_ASR_TEXT {
                segs[si].past_asr_text = true;
            } else if segs[si].past_asr_text {
                let piece = tokenizer.decode_bytes(token);
                segs[si].text_bytes.extend_from_slice(piece);
                segs[si].text_tokens.push(token);
                // Early abort: a window that is ALREADY a repetition loop will be
                // re-decoded by sequential recovery regardless, so stop burning
                // batched steps on it now. Bit-identical: this only ADDS the window
                // to the recovery set (recovery is authoritative and re-decodes from
                // scratch); a window emitted directly always completes cleanly with a
                // healthy final verdict. The check reuses the exact authoritative
                // `segment_is_degenerate` loop logic (maxed=false mid-stream); polled
                // every 16 text tokens to keep its O(n·period) cost negligible.
                let n = segs[si].text_tokens.len();
                if loop_detect
                    && n >= 32
                    && n % 16 == 0
                    && segment_is_degenerate(&segs[si].text_tokens, false, true)
                {
                    segs[si].aborted = true;
                    segs[si].done = true;
                    continue;
                }
            }
            if segs[si].n_generated >= MAX_DECODE_TOKENS {
                // Handled the token-cap token; the sequential path would compute
                // one more (discarded) forward — skipping it changes no output.
                segs[si].done = true;
                continue;
            }
            // Seed next step's input with this token's embedding.
            unsafe {
                tok_embed_bf16_to_f32(&mut segs[si].x, decoder.tok_embeddings_bf16, token, dim)
            };
            next_active.push(si);
        }
        if next_active.is_empty() {
            break;
        }
        batched_decode_step(decoder, cfg, rope, &mut segs, &next_active, &mut sc);
    }

    segs.into_iter()
        .map(|s| SegOutcome {
            text: String::from_utf8_lossy(&s.text_bytes).trim().to_string(),
            maxed: s.n_generated >= MAX_DECODE_TOKENS,
            aborted: s.aborted,
            text_tokens: s.text_tokens,
        })
        .collect()
}

/// Batched analogue of [`crate::transcribe::transcribe_audio`]. Bit-identical
/// output; decodes up to `max_batch` ~30 s segments at once. Falls back to the
/// sequential path for any configuration it does not batch.
pub fn transcribe_audio_batched(ctx: &mut QwenCtx, samples: &[f32], max_batch: usize) -> Option<String> {
    let batchable = cfg!(target_arch = "aarch64")
        && max_batch >= 1
        && !ctx.past_text_conditioning
        && !ctx.model.config.is_aligner()
        && ctx.token_cb.is_none()
        && ctx.segment_sec > 0.0;
    if !batchable {
        return crate::transcribe::transcribe_audio(ctx, samples);
    }

    ctx.reset_perf();
    ctx.perf_audio_ms = 1000.0 * samples.len() as f64 / SAMPLE_RATE as f64;

    let audio_samples = if ctx.skip_silence {
        if ctx.segment_sec > 0.0 {
            audio::compact_silence_fast(samples)
        } else {
            audio::compact_silence(samples)
        }
    } else {
        samples.to_vec()
    };

    let model = ctx.model.clone();
    let tokenizer = &model.tokenizer;
    if !ctx.prepare_prompt_tokens(tokenizer) {
        return None;
    }

    let target_samples = (ctx.segment_sec * SAMPLE_RATE as f32) as usize;
    let search = ctx.search_sec.min(ctx.segment_sec / 2.0);
    let margin_samples = (search * SAMPLE_RATE as f32) as usize;

    // Single segment (or no splitting) → sequential path is already optimal & identical.
    if audio_samples.len() <= target_samples + margin_samples {
        return crate::transcribe::transcribe_audio(ctx, samples);
    }

    // Split points — identical to transcribe_audio.
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

    let cfg = {
        let mut c = model.config.clone();
        if let Some(w) = ctx.enc_n_window_infer_override {
            c.enc_n_window_infer = w;
        }
        c
    };
    let min_samples = SAMPLE_RATE as usize / 2;

    // Build the per-segment audio slices (with the model-floor padding the
    // sequential path applies before transcribe_segment).
    let mut seg_audio: Vec<Vec<f32>> = Vec::with_capacity(n_splits);
    for s in 0..n_splits {
        let seg_start = splits[s];
        let seg_end = if s + 1 < n_splits { splits[s + 1] } else { audio_samples.len() };
        let seg_len = seg_end - seg_start;
        if seg_len < min_samples {
            let mut buf = vec![0.0f32; min_samples];
            buf[..seg_len].copy_from_slice(&audio_samples[seg_start..seg_end]);
            seg_audio.push(buf);
        } else {
            seg_audio.push(audio_samples[seg_start..seg_end].to_vec());
        }
    }

    let mut prefill_bufs = DecoderBuffers::new(&cfg);
    let mut prefill_rope = RopeCache::new();
    let mut decode_rope = RopeCache::new();

    // Decode segments in groups of ≤max_batch, preserving global order.
    let mut seg_texts: Vec<String> = Vec::with_capacity(n_splits);
    let group_cap = max_batch.max(1);
    let mut idx = 0;
    while idx < n_splits {
        let end = (idx + group_cap).min(n_splits);
        let mut group: Vec<SegState> = Vec::with_capacity(end - idx);
        for slice in seg_audio.iter().take(end).skip(idx) {
            match prepare_segment(ctx, &cfg, slice, &mut prefill_bufs, &mut prefill_rope) {
                Some(st) => group.push(st),
                None => return None,
            }
        }
        // Plain path has no recovery: decode every window to completion (no abort).
        let outcomes = decode_group(&model.decoder, &cfg, &mut decode_rope, tokenizer, group, false);
        seg_texts.extend(outcomes.into_iter().map(|o| o.text));
        idx = end;
    }

    // Assemble exactly like transcribe_audio (boundary-space rules, skip empties).
    let mut result = String::new();
    for seg_text in &seg_texts {
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
        }
        result.push_str(seg_text);
    }

    Some(result)
}

// ========================================================================
// Batched segmented / clips paths (the `--json` and `--serve` outputs)
//
// These mirror `transcribe::{transcribe_segmented, transcribe_clips}` →
// `segment_slice`, which split a slice into coarse ~30 s windows and run each
// through `transcribe_with_recovery`. Here the HEALTHY windows are decoded in
// batches of ≤max_batch; a window flagged DEGENERATE (loop) is handed to the
// unchanged sequential `transcribe_with_recovery`, which performs the
// bit-identical 32→16→8 halving on that span. Because the batched decode emits
// identical tokens, a window is flagged degenerate for exactly the same spans as
// the sequential path, so the recovery splits — and every emitted segment's
// text + timestamps — are byte-for-byte identical.
// ========================================================================

/// Map an original-timeline `(start_ms, end_ms)` region to clamped in-buffer
/// sample offsets — replica of `transcribe::region_to_samples`.
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

/// Batched analogue of `transcribe::segment_slice`: split `samples` into coarse
/// windows, batch-decode the healthy ones, defer degenerate ones to sequential
/// recovery. Appends `TranscriptSegment`s (rebased by `base_ms`) in time order.
fn segment_slice_batched(
    ctx: &mut QwenCtx,
    samples: &[f32],
    base_ms: u64,
    out: &mut Vec<TranscriptSegment>,
    max_batch: usize,
) -> Option<()> {
    let model = ctx.model.clone();
    let tokenizer = &model.tokenizer;
    let segment_sec = if ctx.segment_sec > 0.0 { ctx.segment_sec } else { 30.0 };
    let search_sec = ctx.search_sec.min(segment_sec / 2.0);
    let target_samples = (segment_sec * SAMPLE_RATE as f32) as usize;
    let margin_samples = (search_sec * SAMPLE_RATE as f32) as usize;

    // Whole slice fits one coarse window → sequential recovery (it may still halve
    // a degenerate short span); identical to segment_slice's fast path.
    if samples.len() <= target_samples + margin_samples {
        transcribe_with_recovery(ctx, samples, tokenizer, base_ms, out);
        return Some(());
    }

    // Coarse ~30s split points over this slice (identical to segment_slice).
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

    let cfg = {
        let mut c = model.config.clone();
        if let Some(w) = ctx.enc_n_window_infer_override {
            c.enc_n_window_infer = w;
        }
        c
    };
    let min_samples = SAMPLE_RATE as usize / 2;

    // Coarse windows. `padded` (model-floor padded) is what the model decodes; the
    // UNPADDED [start,end) drives both timestamps and the recovery fallback span —
    // exactly as transcribe_with_recovery does internally.
    struct Coarse {
        start: usize,
        end: usize,
        base_ms: u64,
        padded: Vec<f32>,
    }
    let mut coarse: Vec<Coarse> = Vec::with_capacity(n_splits);
    for s in 0..n_splits {
        let seg_start = splits[s];
        let seg_end = if s + 1 < n_splits { splits[s + 1] } else { samples.len() };
        let seg_len = seg_end - seg_start;
        let seg_base_ms = base_ms + (seg_start as u64 * 1000) / SAMPLE_RATE as u64;
        let padded = if seg_len < min_samples {
            let mut b = vec![0.0f32; min_samples];
            b[..seg_len].copy_from_slice(&samples[seg_start..seg_end]);
            b
        } else {
            samples[seg_start..seg_end].to_vec()
        };
        coarse.push(Coarse { start: seg_start, end: seg_end, base_ms: seg_base_ms, padded });
    }

    let mut prefill_bufs = DecoderBuffers::new(&cfg);
    let mut prefill_rope = RopeCache::new();
    let mut decode_rope = RopeCache::new();
    let loop_detect = ctx.loop_detect;
    let group_cap = max_batch.max(1);

    let mut idx = 0;
    while idx < n_splits {
        let end = (idx + group_cap).min(n_splits);
        let mut group: Vec<SegState> = Vec::with_capacity(end - idx);
        for c in coarse.iter().take(end).skip(idx) {
            match prepare_segment(ctx, &cfg, &c.padded, &mut prefill_bufs, &mut prefill_rope) {
                Some(st) => group.push(st),
                None => return None,
            }
        }
        let outcomes =
            decode_group(&model.decoder, &cfg, &mut decode_rope, tokenizer, group, loop_detect);
        for (gi, oc) in outcomes.into_iter().enumerate() {
            let (cs, ce, cbase) = {
                let c = &coarse[idx + gi];
                (c.start, c.end, c.base_ms)
            };
            if oc.aborted || segment_is_degenerate(&oc.text_tokens, oc.maxed, loop_detect) {
                // Degenerate window → unchanged sequential recovery (32→16→8 halving
                // at word boundaries) on the UNPADDED span; bit-identical.
                transcribe_with_recovery(ctx, &samples[cs..ce], tokenizer, cbase, out);
            } else if !oc.text.is_empty() {
                let seg_len = ce - cs;
                out.push(TranscriptSegment {
                    start_ms: cbase,
                    end_ms: cbase + (seg_len as u64 * 1000) / SAMPLE_RATE as u64,
                    text: oc.text,
                });
            }
        }
        idx = end;
    }
    Some(())
}

/// Batched analogue of [`crate::transcribe::transcribe_segmented`]. Bit-identical
/// per-segment text + timestamps; falls back to the sequential path when batching
/// can't apply (non-aarch64 / aligner / batch<1).
pub fn transcribe_segmented_batched(
    ctx: &mut QwenCtx,
    samples: &[f32],
    max_batch: usize,
) -> Option<Vec<TranscriptSegment>> {
    let batchable =
        cfg!(target_arch = "aarch64") && max_batch >= 1 && !ctx.model.config.is_aligner();
    if !batchable {
        return crate::transcribe::transcribe_segmented(ctx, samples);
    }
    ctx.reset_perf();
    let model = ctx.model.clone();
    if !ctx.prepare_prompt_tokens(&model.tokenizer) {
        return None;
    }
    let mut segments: Vec<TranscriptSegment> = Vec::new();
    segment_slice_batched(ctx, samples, 0, &mut segments, max_batch)?;
    Some(segments)
}

/// Batched analogue of [`crate::transcribe::transcribe_clips`]. Same fallback
/// rules as [`transcribe_segmented_batched`].
pub fn transcribe_clips_batched(
    ctx: &mut QwenCtx,
    samples: &[f32],
    regions: &[(u64, u64)],
    max_batch: usize,
) -> Option<Vec<TranscriptSegment>> {
    let batchable =
        cfg!(target_arch = "aarch64") && max_batch >= 1 && !ctx.model.config.is_aligner();
    if !batchable {
        return crate::transcribe::transcribe_clips(ctx, samples, regions);
    }
    ctx.reset_perf();
    let model = ctx.model.clone();
    if !ctx.prepare_prompt_tokens(&model.tokenizer) {
        return None;
    }
    let n = samples.len();
    let mut segments: Vec<TranscriptSegment> = Vec::new();
    for &(region_start_ms, region_end_ms) in regions {
        let (start_sample, end_sample) = match region_to_samples(region_start_ms, region_end_ms, n) {
            Some(r) => r,
            None => continue,
        };
        segment_slice_batched(
            ctx,
            &samples[start_sample..end_sample],
            region_start_ms,
            &mut segments,
            max_batch,
        )?;
    }
    Some(segments)
}
