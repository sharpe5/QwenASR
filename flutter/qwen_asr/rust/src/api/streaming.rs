//! Streaming transcription API for real-time audio processing.

use flutter_rust_bridge::frb;
use qwen_asr::transcribe::StreamState;
use crate::api::qwen_asr_bridge::QwenAsrEngine;
use std::sync::Mutex;

/// Opaque handle to a streaming transcription session.
/// 
/// Create with [QwenAsrStream::new], then repeatedly call [push_audio]
/// as audio chunks arrive, and [finalize] when done.
#[frb(opaque)]
pub struct QwenAsrStream {
    state: Mutex<StreamState>,
    audio_buffer: Mutex<Vec<f32>>,
}

#[frb]
impl QwenAsrStream {
    /// Create a new streaming transcription session.
    /// 
    /// Pre-allocates buffer for 60 seconds of audio at 16kHz.
    #[frb(sync)]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(StreamState::new()),
            audio_buffer: Mutex::new(Vec::with_capacity(16000 * 60)),
        }
    }

    /// Reset the stream for a new utterance.
    /// 
    /// Clears audio buffer and resets state, but preserves allocations.
    #[frb(sync)]
    pub fn reset(&self) {
        let mut state = self.state.lock().unwrap();
        let mut buffer = self.audio_buffer.lock().unwrap();
        state.reset();
        buffer.clear();
    }

    /// Push audio samples and process incrementally.
    /// 
    /// [samples] must be f32 PCM at 16kHz, mono.
    /// Returns newly emitted text delta, if any.
    /// 
    /// This method accumulates all audio internally - you can call it
    /// repeatedly with new chunks as they arrive from the microphone.
    pub fn push_audio(
        &self,
        engine: &QwenAsrEngine,
        samples: Vec<f32>,
    ) -> Option<String> {
        // Append new samples to accumulated buffer
        {
            let mut buffer = self.audio_buffer.lock().unwrap();
            buffer.extend_from_slice(&samples);
        }

        // Process with the engine
        let mut ctx = engine.inner.lock().unwrap();
        let mut state = self.state.lock().unwrap();
        let buffer = self.audio_buffer.lock().unwrap();

        qwen_asr::transcribe::stream_push_audio(
            &mut ctx,
            &buffer,
            &mut state,
            false, // not finalizing
        )
    }

    /// Finalize the stream and emit remaining tokens.
    /// 
    /// Call this when the audio stream has ended. Returns any final
    /// text delta that wasn't yet emitted.
    pub fn finalize(
        &self,
        engine: &QwenAsrEngine,
    ) -> Option<String> {
        let buffer = self.audio_buffer.lock().unwrap();
        let mut ctx = engine.inner.lock().unwrap();
        let mut state = self.state.lock().unwrap();

        qwen_asr::transcribe::stream_push_audio(
            &mut ctx,
            &buffer,
            &mut state,
            true, // finalize
        )
    }

    /// Get the full accumulated text so far.
    /// 
    /// This includes both stable text and any pending tokens.
    #[frb(sync)]
    pub fn text(&self) -> String {
        let state = self.state.lock().unwrap();
        state.text()
    }

    /// Get the number of samples that have been processed.
    /// 
    /// Divide by 16000 to get duration in seconds.
    #[frb(sync)]
    pub fn audio_cursor_samples(&self) -> usize {
        let state = self.state.lock().unwrap();
        state.audio_cursor()
    }

    /// Get processed audio duration in seconds.
    #[frb(sync)]
    pub fn processed_seconds(&self) -> f32 {
        self.audio_cursor_samples() as f32 / 16000.0
    }
}

/// Streaming configuration options.
#[frb]
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Audio chunk size in seconds (default: 2.0)
    pub chunk_sec: f32,
    /// Token rollback window (default: 5)
    pub rollback: i32,
    /// Unfixed chunks before emitting (default: 2)
    pub unfixed_chunks: i32,
    /// Max new tokens per chunk (default: 32)
    pub max_new_tokens: i32,
    /// Use past text conditioning (default: false)
    pub past_text_conditioning: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_sec: 2.0,
            rollback: 5,
            unfixed_chunks: 2,
            max_new_tokens: 32,
            past_text_conditioning: false,
        }
    }
}

/// Utility to apply streaming configuration to an engine.
pub fn apply_stream_config(engine: &QwenAsrEngine, config: &StreamConfig) {
    let mut ctx = engine.inner.lock().unwrap();
    ctx.stream_chunk_sec = config.chunk_sec.clamp(0.5, 10.0);
    ctx.stream_rollback = config.rollback.max(0);
    ctx.stream_unfixed_chunks = config.unfixed_chunks.max(0);
    ctx.stream_max_new_tokens = config.max_new_tokens.max(1);
    ctx.past_text_conditioning = config.past_text_conditioning;
}
