//! Forced alignment API for word-level timestamps.

use flutter_rust_bridge::frb;
use qwen_asr::align::{AlignResult, forced_align};
use crate::api::qwen_asr_bridge::QwenAsrEngine;

/// A single word (or character for CJK) with its time span.
#[frb]
#[derive(Debug, Clone)]
pub struct WordTimestamp {
    /// The word or character text.
    pub text: String,
    /// Start time in milliseconds.
    pub start_ms: f32,
    /// End time in milliseconds.
    pub end_ms: f32,
}

impl From<AlignResult> for WordTimestamp {
    fn from(r: AlignResult) -> Self {
        Self {
            text: r.text,
            start_ms: r.start_ms,
            end_ms: r.end_ms,
        }
    }
}

/// Result of a forced alignment operation.
#[frb]
#[derive(Debug, Clone)]
pub struct AlignmentResult {
    /// The language used for alignment.
    pub language: String,
    /// Word-level timestamps.
    pub words: Vec<WordTimestamp>,
}

impl QwenAsrEngine {
    /// Perform forced alignment on audio with a known transcript.
    ///
    /// Returns word-level timestamps showing when each word was spoken.
    /// Requires an aligner model (qwen3-aligner-0.6b).
    ///
    /// [samples] must be f32 PCM at 16kHz, mono.
    /// [text] is the known transcript to align.
    /// [language] controls word splitting - use "Chinese", "Japanese", 
    /// "Korean", "Cantonese" for character-level, anything else for word-level.
    pub fn align_words(
        &self,
        samples: Vec<f32>,
        text: String,
        language: String,
    ) -> Result<AlignmentResult, String> {
        let mut ctx = self.inner.lock().unwrap();

        // Check if this is an aligner model
        if !ctx.config.is_aligner() {
            return Err(
                "Model is not a forced aligner. Use qwen3-aligner-0.6b model.".into(),
            );
        }

        if samples.is_empty() {
            return Err("Empty audio samples".into());
        }

        if text.is_empty() {
            return Err("Empty transcript".into());
        }

        match forced_align(&mut ctx, &samples, &text, &language) {
            Some(results) => Ok(AlignmentResult {
                language: language.clone(),
                words: results.into_iter().map(WordTimestamp::from).collect(),
            }),
            None => Err("Alignment failed. Check audio and transcript match.".into()),
        }
    }

    /// Check if the loaded model is a forced aligner.
    #[frb(sync)]
    pub fn is_aligner(&self) -> bool {
        let ctx = self.inner.lock().unwrap();
        ctx.config.is_aligner()
    }
}

/// Export alignment results to SRT subtitle format.
///
/// Creates subtitle entries with one phrase per entry (merged words).
#[frb]
pub fn export_srt(result: &AlignmentResult, max_words_per_line: i32) -> String {
    let max_words = max_words_per_line.max(1) as usize;
    let mut srt = String::new();
    let mut entry_num = 1;

    for chunk in result.words.chunks(max_words) {
        if chunk.is_empty() {
            continue;
        }

        let start_ms = chunk.first().unwrap().start_ms;
        let end_ms = chunk.last().unwrap().end_ms;
        let text: String = chunk.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");

        srt.push_str(&format!("{}\n", entry_num));
        srt.push_str(&format!("{} --> {}\n", _ms_to_srt(start_ms), _ms_to_srt(end_ms)));
        srt.push_str(&format!("{}\n\n", text.trim()));

        entry_num += 1;
    }

    srt
}

/// Export alignment results to WebVTT subtitle format.
#[frb]
pub fn export_webvtt(result: &AlignmentResult, max_words_per_line: i32) -> String {
    let max_words = max_words_per_line.max(1) as usize;
    let mut vtt = String::from("WEBVTT\n\n");

    for chunk in result.words.chunks(max_words) {
        if chunk.is_empty() {
            continue;
        }

        let start_ms = chunk.first().unwrap().start_ms;
        let end_ms = chunk.last().unwrap().end_ms;
        let text: String = chunk.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");

        vtt.push_str(&format!("{} --> {}\n", _ms_to_vtt(start_ms), _ms_to_vtt(end_ms)));
        vtt.push_str(&format!("{}\n\n", text.trim()));
    }

    vtt
}

fn _ms_to_srt(ms: f32) -> String {
    let total_secs = (ms / 1000.0) as i32;
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    let millis = (ms as i32) % 1000;
    format!("{:02}:{:02}:{:02},{:03}", hours, mins, secs, millis)
}

fn _ms_to_vtt(ms: f32) -> String {
    let total_secs = (ms / 1000.0) as i32;
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    let millis = (ms as i32) % 1000;
    format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, millis)
}
