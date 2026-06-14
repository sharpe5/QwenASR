//! Top-level engine state: a shared read-only [`QwenModel`] (weights, tokenizer)
//! plus a per-decode [`QwenCtx`] (KV cache, scratch). One model backs N contexts.

use std::sync::Arc;

use crate::config::*;
use crate::decoder::*;
use crate::encoder::*;
use crate::encoder::EncoderBuffers;
use crate::kernels;
use crate::safetensors::MultiSafetensors;
use crate::tokenizer::QwenTokenizer;

pub type TokenCallback = Box<dyn Fn(&str) + Send>;

/// Read-only model state — weights, tokenizer, config — loaded ONCE and shared
/// across N [`QwenCtx`] via `Arc`. In `--weights bf16` mode the big projection
/// weights stay raw mmap pointers into `_safetensors`, so N contexts back onto a
/// single physical weight copy; the INT8/fused derived weights inside
/// `encoder`/`decoder` (the per-load heap cost) now exist once instead of N times.
///
/// `Encoder`/`Decoder`/`MultiSafetensors` carry their own `unsafe impl Send + Sync`
/// (the raw `*const u16` weight pointers); `QwenConfig`/`QwenTokenizer` are plain
/// `Vec`/`HashMap` data. So `QwenModel` is `Send + Sync` by composition and
/// `Arc<QwenModel>` is safe to share across the serve worker threads.
pub struct QwenModel {
    pub config: QwenConfig,
    pub encoder: Encoder,
    pub decoder: Decoder,
    /// Loaded once here instead of per request (was re-read from disk on every call).
    pub tokenizer: QwenTokenizer,
    pub model_dir: String,
    _safetensors: MultiSafetensors, // kept alive for mmap'd BF16 pointers
}

impl QwenModel {
    /// Load the read-only model from `model_dir` (see [`QwenCtx::load_opts`] for the
    /// `weights_bf16` meaning). Returns an `Arc` so it can be cloned cheaply into
    /// many contexts.
    pub fn load_opts(model_dir: &str, weights_bf16: bool) -> Option<Arc<QwenModel>> {
        if kernels::verbose() >= 1 {
            eprintln!("Loading model from {}", model_dir);
        }

        let ms = MultiSafetensors::open(model_dir)?;

        // Detect model variant from tensor shapes
        let info = crate::config::DetectInfo {
            has_enc_layer_18: ms.has_tensor("thinker.audio_tower.layers.18.self_attn.q_proj.weight"),
            lm_head_shape: ms.find("thinker.lm_head.weight").map(|(_, t)| t.shape.as_slice()),
            embed_tokens_shape: ms.find("thinker.model.embed_tokens.weight").map(|(_, t)| t.shape.as_slice()),
            gate_proj_shape: ms.find("thinker.model.layers.0.mlp.gate_proj.weight").map(|(_, t)| t.shape.as_slice()),
        };
        let cfg = QwenConfig::detect(&info);

        if kernels::verbose() >= 1 {
            let variant = if cfg.dec_hidden >= 2048 { "1.7B" } else { "0.6B" };
            let model_type = if cfg.is_aligner() { "ForcedAligner" } else { "ASR" };
            eprintln!("Detected: Qwen3-{}-{}", model_type, variant);
            if cfg.is_aligner() {
                eprintln!("  classify_num={}, timestamp_segment_time={:.0}ms",
                          cfg.classify_num, cfg.timestamp_segment_time);
                eprintln!("  encoder: {}d {}L, decoder: {}d {}L",
                          cfg.enc_d_model, cfg.enc_layers, cfg.dec_hidden, cfg.dec_layers);
            }
        }

        // Load encoder
        if kernels::verbose() >= 1 {
            eprintln!("Loading encoder weights...");
        }
        let encoder = Encoder::load(&ms, &cfg, weights_bf16)?;

        // Load decoder
        if kernels::verbose() >= 1 {
            eprintln!("Loading decoder weights...");
        }
        let decoder = Decoder::load(&ms, &cfg, weights_bf16)?;

        // Tokenizer: load once and keep resident (callers borrow `&model.tokenizer`).
        let tokenizer = QwenTokenizer::load(&format!("{}/vocab.json", model_dir))?;

        if kernels::verbose() >= 1 {
            eprintln!("Model loaded.");
        }

        Some(Arc::new(QwenModel {
            config: cfg,
            encoder,
            decoder,
            tokenizer,
            model_dir: model_dir.to_string(),
            _safetensors: ms,
        }))
    }
}

/// Per-decode engine state: a shared [`QwenModel`] handle plus the mutable KV
/// cache and scratch buffers a single decode needs. Cheap to build from a model
/// (`from_model` only allocates scratch). One context is single-owner while
/// in use, so its `&mut` scratch never aliases across threads.
///
/// Create with [`QwenCtx::load`], then pass to functions in the [`crate::transcribe`] module.
///
/// `--past-text` tri-state. `Auto` resolves to ON for streaming and OFF otherwise —
/// that resolution depends on HOW the audio is fed (a run-mode decision, not a fixed
/// default), so it happens in [`QwenCtx::apply_settings`], not here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PastText {
    Auto,
    On,
    Off,
}

/// The SINGLE source of truth for every decode default shared by the CLI and `--serve`.
///
/// The CLI defaults are the gold standard, and `DecodeSettings::default()` IS that
/// table — written once, consumed everywhere:
///   * [`QwenCtx::from_model`] builds its decode fields from it (so a bare ctx already
///     carries the gold-standard defaults),
///   * the CLI starts from it and overrides per command-line flag,
///   * `--serve` applies it verbatim to every resident ctx (so serve can NEVER drift
///     from the CLI defaults — the whole point), and
///   * the CLI `--help` default strings are formatted from it (so the docs can't drift
///     either; the old hand-written "default: …" literals had already gone stale).
/// Change a default HERE and it changes in all four places at once.
///
/// `--enc-window-sec`'s default is intentionally NOT here: the encoder attention window
/// is a model-architecture knob whose default lives in the model's own
/// `config.enc_n_window_infer`, and the flag is a pure override of that
/// (`enc_n_window_infer_override`). Folding it in would duplicate a model constant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeSettings {
    /// `-S`: segment target seconds for long audio (0 = full-audio, no splitting).
    /// ~30 keeps the model from looping/repeating on long spans.
    pub segment_sec: f32,
    /// `-W`: ± seconds silence-search window when cutting segments.
    pub search_sec: f32,
    /// `--past-text`: reuse previously decoded text as decode context.
    pub past_text: PastText,
    /// `--skip-silence`: drop long silent spans before inference.
    pub skip_silence: bool,
    /// `--stream-chunk-sec`: chunk size in streaming mode.
    pub stream_chunk_sec: f32,
    /// `--stream-max-new-tokens`: max generated tokens per streaming step.
    pub stream_max_new_tokens: i32,
    pub stream_rollback: i32,
    pub stream_unfixed_chunks: i32,
    // Loop / repetition handling. The detection THRESHOLDS (period / min-reps / token-run) are
    // not runtime-tunable — they're constants in `transcribe.rs` (the stream path proves hard-
    // coded values are fine, and nobody needs to tune a degeneracy detector per run). The only
    // loop knobs here are the on/off switch and the segment-recovery bounds.
    /// `--loop-detect`: master switch — detect & recover from decoder repetition loops (both the
    /// --stream and segment/--serve paths). When false, neither path detects or recovers.
    pub loop_detect: bool,
    /// `--loop-min-split-sec`: segment-recovery size floor — only split spans >= 2× this.
    pub loop_min_split_sec: f32,
    /// `--loop-max-depth`: segment-recovery max halving depth (e.g. 32→16→8 at the 8s floor).
    pub loop_max_depth: i32,
}

impl Default for DecodeSettings {
    fn default() -> Self {
        // ─── THE gold-standard decode defaults (one number each, the only copy) ───
        DecodeSettings {
            segment_sec: 30.0,
            search_sec: 3.0,
            past_text: PastText::Auto,
            skip_silence: false,
            // 8.0 is the runtime gold standard; the CLI help string had drifted to "2.0".
            stream_chunk_sec: 8.0,
            stream_max_new_tokens: 32,
            stream_rollback: 5,
            stream_unfixed_chunks: 99,
            loop_detect: true,
            // Recovery floor 8s: the shortest segment recovery emits is 8s — spans < 16s
            // (= 2× the floor) aren't split — so a typical <=33s coarse segment goes 32 -> 16 ->
            // 8 and stops. loop_max_depth=3 caps total halvings for any larger span; for the
            // common case the 8s floor is the binding stop.
            loop_min_split_sec: 8.0,
            loop_max_depth: 3,
        }
    }
}

/// # Configurable fields
///
/// | Field | Default | Description |
/// |-------|---------|-------------|
/// | `segment_sec` | 30.0 | Segment duration for long audio (0 = no splitting) |
/// | `skip_silence` | false | Drop silent spans before transcription |
/// | `token_cb` | None | Streaming callback invoked for each decoded token |
/// | `prompt` | None | Optional text prompt (set via [`QwenCtx::set_prompt`]) |
/// | `force_language` | None | Force a language (set via [`QwenCtx::set_force_language`]) |
pub struct QwenCtx {
    /// Shared read-only weights/tokenizer/config (one copy backs N contexts).
    pub model: Arc<QwenModel>,

    // KV cache
    pub kv_cache: KvCache,

    // Decoder buffers
    pub dec_bufs: DecoderBuffers,

    // Encoder scratch buffers (reusable across calls)
    pub enc_bufs: EncoderBuffers,

    // RoPE cache
    pub rope_cache: RopeCache,

    // Token streaming callback
    pub token_cb: Option<TokenCallback>,

    // Segmentation settings
    pub segment_sec: f32,
    pub search_sec: f32,

    // Streaming settings
    pub stream_chunk_sec: f32,
    pub stream_rollback: i32,
    pub stream_unfixed_chunks: i32,
    pub stream_max_new_tokens: i32,
    pub past_text_conditioning: bool,
    pub skip_silence: bool,

    // Loop / repetition-degeneracy handling (see DecodeSettings + docs/loop-detection.md). The
    // detection thresholds are constants in transcribe.rs; only the on/off switch and the
    // segment-recovery bounds are per-ctx settings.
    /// Master switch (`--loop-detect`). When false, no detection/recovery runs in either
    /// path — decode is byte-for-byte the legacy behavior. Default true.
    pub loop_detect: bool,
    /// Segment-recovery size floor in seconds (`--loop-min-split-sec`, default 8.0): a clip is
    /// only halved when it is at least 2× this (>= 16s), so the shortest segment recovery emits
    /// is 8s and short segments are never over-split. The floor gates WHETHER to split, not the
    /// exact half sizes — the cut follows the word gap (lowest-energy point near the midpoint),
    /// so halves can be uneven and one may end up below this; that half just isn't split again.
    pub loop_min_split_sec: f32,
    /// Segment-recovery depth cap (`--loop-max-depth`, default 3): at most this many halvings,
    /// so a 32s degenerate clip goes 32→16→8 (stopping at the 8s floor) and larger spans split
    /// no more than 3 times.
    pub loop_max_depth: i32,

    /// Per-ctx override for `config.enc_n_window_infer` (CLI `--enc-window-sec`).
    /// `config` is now shared/read-only, so the override lives here and is folded
    /// into the per-call config clone in `crate::transcribe`. `None` = model default.
    pub enc_n_window_infer_override: Option<usize>,

    // Optional prompt/language
    pub prompt: Option<String>,
    pub force_language: Option<String>,
    pub prompt_tokens: Option<Vec<i32>>,
    pub force_prompt_tokens: Option<Vec<i32>>,
    pub prompt_tokens_ready: bool,

    // Perf stats
    pub perf_total_ms: f64,
    pub perf_text_tokens: i32,
    pub perf_audio_ms: f64,
    /// Mel spectrogram + encoder forward pass time combined.
    pub perf_encode_ms: f64,
    pub perf_decode_ms: f64,
    /// Observe-only decode-health counters (read by the serve logger; no behavior
    /// change). `perf_segments` = segments decoded this request; `perf_maxed_segments`
    /// = those that hit the autoregressive `max_tokens` cap WITHOUT emitting EOS — the
    /// degeneracy signature behind multi-hour "stuck" decodes. A healthy segment stops
    /// at EOS in ~50-200 tokens; a maxed one burns the full 2048, so a request where
    /// every segment maxes out runs ~10-40x longer than a normal one.
    pub perf_segments: i32,
    pub perf_maxed_segments: i32,
}

impl QwenCtx {
    /// Load a Qwen3-ASR model from `model_dir`.
    ///
    /// The directory must contain `model*.safetensors` and `vocab.json`.
    /// Returns `None` if any required file is missing or malformed.
    ///
    /// ```rust,no_run
    /// use qwen_asr::context::QwenCtx;
    /// let ctx = QwenCtx::load("qwen3-asr-0.6b").expect("failed to load");
    /// ```
    pub fn load(model_dir: &str) -> Option<Self> {
        Self::load_opts(model_dir, false)
    }

    /// Like [`load`], but `weights_bf16` keeps the big projection weights BF16-resident
    /// (raw mmap pointers, widened to f32 per matmul) instead of dequantizing them to
    /// f32 Vecs at load. Roughly halves weight RAM; math is identical. See `--weights`.
    ///
    /// [`load`]: QwenCtx::load
    pub fn load_opts(model_dir: &str, weights_bf16: bool) -> Option<Self> {
        Some(Self::from_model(QwenModel::load_opts(model_dir, weights_bf16)?))
    }

    /// Build a context around an already-loaded shared model. Cheap: bumps the
    /// `Arc` and allocates only this context's KV cache and scratch buffers — no
    /// weight load. The serve loop calls this N times against one `Arc<QwenModel>`.
    pub fn from_model(model: Arc<QwenModel>) -> Self {
        let cfg = &model.config;
        let kv_cache = KvCache::new(cfg.dec_layers, 2048, cfg.dec_kv_heads, cfg.dec_head_dim);
        let dec_bufs = DecoderBuffers::new(cfg);

        // Decode fields come from the SINGLE source of truth (DecodeSettings), so a
        // bare ctx — including every `--serve` resident ctx — already carries the
        // gold-standard CLI defaults. `Auto` past-text resolves to OFF for a
        // freshly-built (non-streaming) ctx, matching the historical `false`.
        let d = DecodeSettings::default();
        QwenCtx {
            model,
            kv_cache,
            dec_bufs,
            enc_bufs: EncoderBuffers::new(),
            rope_cache: RopeCache::new(),
            token_cb: None,
            segment_sec: d.segment_sec,
            search_sec: d.search_sec,
            stream_chunk_sec: d.stream_chunk_sec,
            stream_rollback: d.stream_rollback,
            stream_unfixed_chunks: d.stream_unfixed_chunks,
            stream_max_new_tokens: d.stream_max_new_tokens,
            past_text_conditioning: matches!(d.past_text, PastText::On),
            skip_silence: d.skip_silence,
            loop_detect: d.loop_detect,
            loop_min_split_sec: d.loop_min_split_sec,
            loop_max_depth: d.loop_max_depth,
            enc_n_window_infer_override: None,
            prompt: None,
            force_language: None,
            prompt_tokens: None,
            force_prompt_tokens: None,
            prompt_tokens_ready: false,
            perf_total_ms: 0.0,
            perf_text_tokens: 0,
            perf_audio_ms: 0.0,
            perf_encode_ms: 0.0,
            perf_decode_ms: 0.0,
            perf_segments: 0,
            perf_maxed_segments: 0,
        }
    }

    /// Apply a [`DecodeSettings`] onto this ctx's decode fields — the ONE place
    /// settings become ctx state, shared by `from_model` (gold-standard defaults), the
    /// CLI (defaults + flag overrides) and `--serve` (defaults). `stream_mode` resolves
    /// `PastText::Auto` (ON for streaming, OFF otherwise), since that choice depends on
    /// how the audio is fed, not on a fixed default. Does NOT touch `prompt` /
    /// `force_language` / `enc_n_window_infer_override` — those are per-invocation
    /// overrides, not part of the shared default table.
    pub fn apply_settings(&mut self, s: &DecodeSettings, stream_mode: bool) {
        self.segment_sec = s.segment_sec;
        self.search_sec = s.search_sec;
        self.stream_chunk_sec = s.stream_chunk_sec;
        self.stream_rollback = s.stream_rollback;
        self.stream_unfixed_chunks = s.stream_unfixed_chunks;
        self.stream_max_new_tokens = s.stream_max_new_tokens;
        self.skip_silence = s.skip_silence;
        self.loop_detect = s.loop_detect;
        self.loop_min_split_sec = s.loop_min_split_sec;
        self.loop_max_depth = s.loop_max_depth;
        self.past_text_conditioning = match s.past_text {
            PastText::On => true,
            PastText::Off => false,
            PastText::Auto => stream_mode,
        };
    }

    /// Set an optional text prompt to guide transcription. Pass an empty string to clear.
    #[allow(clippy::result_unit_err)]
    pub fn set_prompt(&mut self, prompt: &str) -> Result<(), ()> {
        if prompt.is_empty() {
            self.prompt = None;
        } else {
            self.prompt = Some(prompt.to_string());
        }
        self.prompt_tokens_ready = false;
        Ok(())
    }

    /// Force a specific language (e.g. `"English"`, `"Chinese"`). Pass an empty
    /// string for auto-detection. Returns `Err(())` if the language is not recognized.
    #[allow(clippy::result_unit_err)]
    pub fn set_force_language(&mut self, language: &str) -> Result<(), ()> {
        if language.is_empty() {
            self.force_language = None;
            self.prompt_tokens_ready = false;
            return Ok(());
        }

        match normalize_language(language) {
            Some(normalized) => {
                self.force_language = Some(normalized);
                self.prompt_tokens_ready = false;
                Ok(())
            }
            None => Err(()),
        }
    }

    pub fn prepare_prompt_tokens(&mut self, tokenizer: &QwenTokenizer) -> bool {
        if self.prompt_tokens_ready {
            return true;
        }

        self.prompt_tokens = None;
        self.force_prompt_tokens = None;

        if let Some(ref prompt) = self.prompt {
            match tokenizer.encode(prompt) {
                Some(tokens) => self.prompt_tokens = Some(tokens),
                None => {
                    eprintln!("qwen: failed to encode --prompt text");
                    return false;
                }
            }
        }

        if let Some(ref lang) = self.force_language {
            let force_text = format!("language {}", lang);
            match tokenizer.encode(&force_text) {
                Some(mut lang_tokens) => {
                    lang_tokens.push(TOKEN_ASR_TEXT);
                    self.force_prompt_tokens = Some(lang_tokens);
                }
                None => {
                    eprintln!("qwen: failed to encode --language text");
                    return false;
                }
            }
        } else {
            self.force_prompt_tokens = Some(vec![11528, 6364, TOKEN_ASR_TEXT]);
        }

        self.prompt_tokens_ready = true;
        true
    }

    pub fn reset_perf(&mut self) {
        self.perf_total_ms = 0.0;
        self.perf_text_tokens = 0;
        self.perf_audio_ms = 0.0;
        self.perf_encode_ms = 0.0;
        self.perf_decode_ms = 0.0;
        self.perf_segments = 0;
        self.perf_maxed_segments = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Locks the gold-standard loop defaults. This is the serve/CLI **parity guard**: both the
    /// one-shot CLI and `--serve` build their settings from this single `DecodeSettings::default()`
    /// (`main` builds it once and hands the SAME value to `serve::run` and `apply_settings`), and
    /// `apply_settings` copies the fields verbatim — so the only place a default could diverge is
    /// this table, and any drift fails here.
    ///
    /// (We do NOT assert `apply_settings`'s ctx copy directly: a `QwenCtx` needs a loaded model,
    /// so it belongs in an integration test, not a unit test. The straight-line field copy in
    /// `apply_settings` is compile-checked; this test guards the values it copies.)
    #[test]
    fn loop_defaults_are_gold_standard() {
        let d = DecodeSettings::default();
        assert!(d.loop_detect);
        assert_eq!(d.loop_min_split_sec, 8.0);
        assert_eq!(d.loop_max_depth, 3);
    }
}
