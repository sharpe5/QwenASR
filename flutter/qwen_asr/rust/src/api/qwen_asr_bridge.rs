use std::sync::Mutex;
use flutter_rust_bridge::frb;
use qwen_asr::context::QwenCtx;
use qwen_asr::{audio, kernels, transcribe};

#[frb(opaque)]
pub struct QwenAsrEngine {
    pub(crate) inner: Mutex<QwenCtx>,
}

/// Model information returned by [QwenAsrEngine.model_info].
#[frb]
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub variant: String,      // "0.6B" or "1.7B"
    pub model_type: String,   // "ASR" or "ForcedAligner"
    pub enc_hidden: i32,
    pub enc_layers: i32,
    pub dec_hidden: i32,
    pub dec_layers: i32,
}

impl QwenAsrEngine {
    /// Load model from a directory path.
    /// 
    /// Returns an error if:
    /// - The directory doesn't exist
    /// - Required model files are missing
    /// - Thread count is invalid
    pub fn load(model_dir: String, n_threads: i32, verbosity: i32) -> Result<QwenAsrEngine, String> {
        kernels::set_verbose(verbosity);
        
        // Validate thread count
        if n_threads < 0 {
            return Err(format!("Invalid thread count: {} (must be >= 0)", n_threads));
        }
        
        // Check directory exists
        let path = std::path::Path::new(&model_dir);
        if !path.exists() {
            return Err(format!("Model directory not found: {}", model_dir));
        }
        if !path.is_dir() {
            return Err(format!("Path is not a directory: {}", model_dir));
        }
        
        let threads = if n_threads == 0 {
            kernels::get_num_cpus()
        } else {
            n_threads as usize
        };
        kernels::set_threads(threads);
        
        match QwenCtx::load(&model_dir) {
            Some(ctx) => Ok(QwenAsrEngine {
                inner: Mutex::new(ctx),
            }),
            None => Err(format!(
                "Failed to load model from: {}. Ensure the directory contains model*.safetensors and vocab.json files.",
                model_dir
            )),
        }
    }

    /// Transcribe a WAV file at the given path.
    /// 
    /// Returns an error if the file doesn't exist or transcription fails.
    pub fn transcribe_file(&self, wav_path: String) -> Result<String, String> {
        if !std::path::Path::new(&wav_path).exists() {
            return Err(format!("Audio file not found: {}", wav_path));
        }
        
        let mut ctx = self.inner.lock().unwrap();
        transcribe::transcribe(&mut ctx, &wav_path)
            .ok_or_else(|| "Transcription failed. Ensure the file is a valid WAV file.".into())
    }

    /// Transcribe raw PCM f32 samples (16kHz mono, range [-1.0, 1.0]).
    /// 
    /// Returns an error if samples are empty or too short.
    pub fn transcribe_pcm(&self, samples: Vec<f32>) -> Result<String, String> {
        if samples.is_empty() {
            return Err("Empty audio samples".into());
        }
        if samples.len() < 1600 { // 100ms minimum at 16kHz
            return Err(format!(
                "Audio too short: {} samples (minimum 100ms = 1600 samples at 16kHz)",
                samples.len()
            ));
        }
        
        let mut ctx = self.inner.lock().unwrap();
        transcribe::transcribe_audio(&mut ctx, &samples)
            .ok_or_else(|| "Transcription failed".into())
    }

    /// Transcribe from a WAV file buffer (bytes).
    /// 
    /// Returns an error if the buffer is not a valid WAV file.
    pub fn transcribe_wav_buffer(&self, wav_data: Vec<u8>) -> Result<String, String> {
        if wav_data.is_empty() {
            return Err("Empty WAV buffer".into());
        }
        
        let samples = audio::parse_wav_buffer(&wav_data)
            .ok_or("Failed to parse WAV buffer. Ensure valid WAV format (16kHz, 16-bit PCM or float).")?;
        
        let mut ctx = self.inner.lock().unwrap();
        transcribe::transcribe_audio(&mut ctx, &samples)
            .ok_or_else(|| "Transcription failed".into())
    }

    /// Set the segment duration in seconds (0 = no segmentation).
    #[frb(sync)]
    pub fn set_segment_sec(&self, sec: f32) {
        let mut ctx = self.inner.lock().unwrap();
        ctx.segment_sec = sec.max(0.0);
    }

    /// Set the forced language. Returns false if the language is invalid.
    #[frb(sync)]
    pub fn set_language(&self, language: String) -> bool {
        let mut ctx = self.inner.lock().unwrap();
        ctx.set_force_language(&language).is_ok()
    }

    /// Get last transcription performance stats as a formatted string.
    #[frb(sync)]
    pub fn perf_stats(&self) -> String {
        let ctx = self.inner.lock().unwrap();
        format!(
            "audio={:.1}ms encode={:.1}ms decode={:.1}ms total={:.1}ms tokens={}",
            ctx.perf_audio_ms, ctx.perf_encode_ms, ctx.perf_decode_ms,
            ctx.perf_total_ms, ctx.perf_text_tokens
        )
    }

    // ==================== Configuration APIs ====================

    /// Set an optional text prompt to guide transcription.
    /// Pass empty string or null to clear.
    #[frb(sync)]
    pub fn set_prompt(&self, prompt: Option<String>) {
        let mut ctx = self.inner.lock().unwrap();
        let prompt_str = prompt.unwrap_or_default();
        let _ = ctx.set_prompt(&prompt_str);
    }

    /// Enable/disable silence skipping for long recordings.
    #[frb(sync)]
    pub fn set_skip_silence(&self, skip: bool) {
        let mut ctx = self.inner.lock().unwrap();
        ctx.skip_silence = skip;
    }

    /// Configure streaming chunk size in seconds (default 2.0).
    /// Valid range: 0.5 to 10.0 seconds.
    #[frb(sync)]
    pub fn set_stream_chunk_sec(&self, sec: f32) {
        let mut ctx = self.inner.lock().unwrap();
        ctx.stream_chunk_sec = sec.clamp(0.5, 10.0);
    }

    /// Configure token rollback window for streaming (default 5).
    #[frb(sync)]
    pub fn set_stream_rollback(&self, tokens: i32) {
        let mut ctx = self.inner.lock().unwrap();
        ctx.stream_rollback = tokens.max(0);
    }

    /// Configure unfixed chunks count before emitting (default 2).
    #[frb(sync)]
    pub fn set_stream_unfixed_chunks(&self, chunks: i32) {
        let mut ctx = self.inner.lock().unwrap();
        ctx.stream_unfixed_chunks = chunks.max(0);
    }

    /// Configure max new tokens per streaming chunk (default 32).
    #[frb(sync)]
    pub fn set_stream_max_new_tokens(&self, tokens: i32) {
        let mut ctx = self.inner.lock().unwrap();
        ctx.stream_max_new_tokens = tokens.max(1);
    }

    /// Enable past text conditioning for better context in streaming.
    #[frb(sync)]
    pub fn set_past_text_conditioning(&self, enable: bool) {
        let mut ctx = self.inner.lock().unwrap();
        ctx.past_text_conditioning = enable;
    }

    /// Get model information.
    #[frb(sync)]
    pub fn model_info(&self) -> ModelInfo {
        let ctx = self.inner.lock().unwrap();
        ModelInfo {
            variant: if ctx.config.dec_hidden >= 2048 { "1.7B".into() } else { "0.6B".into() },
            model_type: if ctx.config.is_aligner() { "ForcedAligner".into() } else { "ASR".into() },
            enc_hidden: ctx.config.enc_d_model as i32,
            enc_layers: ctx.config.enc_layers as i32,
            dec_hidden: ctx.config.dec_hidden as i32,
            dec_layers: ctx.config.dec_layers as i32,
        }
    }
}

#[frb(init)]
pub fn init_app() {
    flutter_rust_bridge::setup_default_user_utils();
}
