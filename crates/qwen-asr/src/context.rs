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

        QwenCtx {
            model,
            kv_cache,
            dec_bufs,
            enc_bufs: EncoderBuffers::new(),
            rope_cache: RopeCache::new(),
            token_cb: None,
            segment_sec: 30.0,
            search_sec: 3.0,
            stream_chunk_sec: 8.0,
            stream_rollback: 5,
            stream_unfixed_chunks: 99,
            stream_max_new_tokens: 32,
            past_text_conditioning: false,
            skip_silence: false,
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
        }
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
    }
}
