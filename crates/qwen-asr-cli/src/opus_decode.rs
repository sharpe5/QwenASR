//! Native Ogg/Opus decoding — no ffmpeg subprocess.
//!
//! Decodes an `.opus`/`.ogg` (Opus-in-Ogg) file straight to 16 kHz mono f32 PCM, the
//! exact shape `qwen-asr` wants (`SAMPLE_RATE = 16000`). The libopus decoder is created
//! at 16 kHz mono, so libopus resamples (48k→16k) and downmixes internally — no separate
//! resampler. Versus shelling out to `ffmpeg -ar 16000 -ac 1 -f s16le`, this drops the
//! child process and its decode buffers AND the intermediate s16le byte buffer (~0.7 GB
//! for a 6 h block), decoding directly into the f32 Vec the pipeline already holds.
//!
//! This is the same decoder used by the sibling `silero-vad-rs` VAD lane (`ogg` for the
//! pure-Rust container demux, `opus`/libopus for the codec), kept in the CLI crate so the
//! dependency-free, mobile/JNI `qwen-asr` core library is untouched.

use std::fs::File;
use std::io::{BufReader, Read, Seek};

use qwen_asr::config::SAMPLE_RATE;

/// Decode an Ogg/Opus file to 16 kHz mono f32 PCM. Returns `None` (after logging) on any
/// read/demux/decode error so it slots into the CLI's `Option<Vec<f32>>` load path.
pub fn load_opus(path: &str) -> Option<Vec<f32>> {
    let file = File::open(path)
        .map_err(|e| eprintln!("Error: failed to open {}: {}", path, e))
        .ok()?;
    decode_opus(BufReader::new(file))
        .map_err(|e| eprintln!("Error: failed to decode Opus {}: {}", path, e))
        .ok()
}

/// Demux Ogg pages + decode every Opus audio packet into one 16 kHz mono f32 buffer.
fn decode_opus(input: impl Read + Seek) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = ogg::reading::PacketReader::new(input);
    let mut decoder = opus::Decoder::new(SAMPLE_RATE as u32, opus::Channels::Mono)?;
    let mut scratch = vec![0f32; SAMPLE_RATE as usize]; // 1 s ≫ any opus frame (≤120 ms)
    let mut pcm: Vec<f32> = Vec::new();

    while let Some(packet) = reader.read_packet()? {
        // Skip the two Ogg/Opus header packets; the rest are audio.
        if packet.data.starts_with(b"OpusHead") || packet.data.starts_with(b"OpusTags") {
            continue;
        }
        let n = decoder.decode_float(&packet.data, &mut scratch, false)?;
        pcm.extend_from_slice(&scratch[..n]);
    }
    Ok(pcm)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A real ~20 s mono Ogg/Opus clip (RFI radio). Decoded natively it must come out as
    // 16 kHz mono f32 — the same shape ffmpeg -ar 16000 -ac 1 produced before. ffmpeg
    // yields 320216 samples; libopus keeps the Opus pre-skip padding ffmpeg trims, so we
    // get ~320320 (~6 ms more) — within the documented ~0.1 s decoder tolerance.
    const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/sample.opus");

    #[test]
    fn load_opus_decodes_to_16k_mono_f32() {
        let pcm = load_opus(FIXTURE).expect("fixture should decode");
        let dur = pcm.len() as f32 / SAMPLE_RATE as f32;
        // ~20.0 s clip; allow the pre-skip slack vs ffmpeg's 320216 samples.
        assert!(
            (19.9..20.2).contains(&dur),
            "expected ~20 s, got {dur:.3} s ({} samples)",
            pcm.len()
        );
        // Decoded real audio: not silent, and in a sane range. (libopus float output
        // can slightly overshoot ±1.0 — unlike the i16/ffmpeg path — which is fine: the
        // downstream mel stage normalises by global max. We only guard against garbage.)
        let peak = pcm.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(peak > 0.01, "audio should not be silent (peak {peak})");
        assert!(peak.is_finite() && peak < 4.0, "samples out of sane range (peak {peak})");
    }

    #[test]
    fn load_opus_missing_file_returns_none() {
        assert!(load_opus("/no/such/file.opus").is_none());
    }

    #[test]
    fn load_opus_rejects_non_opus_bytes() {
        // A WAV (or any non-Ogg) handed to the opus loader must fail cleanly (None),
        // not panic — the CLI only routes real .opus/.ogg here, but be defensive.
        let tmp = std::env::temp_dir().join("qwen_asr_not_an_opus.opus");
        std::fs::write(&tmp, b"RIFF....WAVEfmt ").unwrap();
        let got = load_opus(tmp.to_str().unwrap());
        let _ = std::fs::remove_file(&tmp);
        assert!(got.is_none());
    }
}
