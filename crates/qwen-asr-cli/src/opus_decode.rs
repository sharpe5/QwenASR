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

use std::convert::TryInto;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use qwen_asr::config::SAMPLE_RATE;

/// Why a native Opus load did not yield PCM — lets the caller distinguish "this Ogg
/// isn't Opus, try ffmpeg" from "this is a broken/short file, fail loudly".
#[derive(Debug)]
pub enum OpusError {
    /// The container is Ogg but carries no Opus stream (e.g. Vorbis/FLAC-in-Ogg).
    /// The caller can fall back to a general decoder rather than treating it as fatal.
    NotOpus,
    /// A valid Opus stream that decoded far short of its declared length — a
    /// truncated/corrupt file. Surfaced loudly instead of silently returning a
    /// partial buffer (the loud failure ffmpeg's non-zero exit used to give us).
    Truncated { got_s: f32, expected_s: f32 },
    /// An I/O, demux, or codec error.
    Decode(String),
}

impl fmt::Display for OpusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpusError::NotOpus => write!(f, "not an Opus-in-Ogg stream"),
            OpusError::Truncated { got_s, expected_s } => write!(
                f,
                "truncated Opus: decoded {got_s:.1}s of a declared {expected_s:.1}s"
            ),
            OpusError::Decode(e) => write!(f, "{e}"),
        }
    }
}

/// Decode an Ogg/Opus file to 16 kHz mono f32 PCM.
pub fn load_opus(path: &str) -> Result<Vec<f32>, OpusError> {
    let mut file = File::open(path).map_err(|e| OpusError::Decode(format!("open {path}: {e}")))?;
    // The final Ogg page's granule position is the stream's total 48 kHz sample count.
    // Use it to pre-size the PCM buffer (one alloc instead of ~log2(n) doublings of a
    // multi-hundred-MB Vec) and to detect a truncated file after decode.
    let granule = final_granule(&mut file);
    file.seek(SeekFrom::Start(0))
        .map_err(|e| OpusError::Decode(e.to_string()))?;
    let expected = granule.map(|g| (g as f64 * SAMPLE_RATE as f64 / 48_000.0) as usize);
    decode_opus(BufReader::new(file), expected)
}

/// Scan backward from EOF for the last `OggS` page header and read its 64-bit LE
/// granule position (RFC 7845 §4: samples at 48 kHz). `None` if no page is found.
fn final_granule(file: &mut File) -> Option<u64> {
    let len = file.seek(SeekFrom::End(0)).ok()?;
    if len == 0 {
        return None;
    }
    let window = len.min(65_536);
    file.seek(SeekFrom::End(-(window as i64))).ok()?;
    let mut buf = vec![0u8; window as usize];
    file.read_exact(&mut buf).ok()?;
    // granule_position lives 6 bytes into each "OggS" page header; keep the last one.
    let mut granule = None;
    let mut i = 0usize;
    while i + 14 <= buf.len() {
        if &buf[i..i + 4] == b"OggS" {
            granule = Some(u64::from_le_bytes(buf[i + 6..i + 14].try_into().unwrap()));
        }
        i += 1;
    }
    granule
}

/// Demux Ogg pages + decode every Opus audio packet into one 16 kHz mono f32 buffer.
/// `expected_len` (16 kHz samples, if known) pre-sizes the buffer and bounds a
/// truncation check.
fn decode_opus(
    input: impl Read + Seek,
    expected_len: Option<usize>,
) -> Result<Vec<f32>, OpusError> {
    let mut reader = ogg::reading::PacketReader::new(input);
    let mut decoder = opus::Decoder::new(SAMPLE_RATE as u32, opus::Channels::Mono)
        .map_err(|e| OpusError::Decode(e.to_string()))?;
    let mut scratch = vec![0f32; SAMPLE_RATE as usize]; // 1 s ≫ any opus frame (≤120 ms)
    let mut pcm: Vec<f32> = Vec::with_capacity(expected_len.unwrap_or(0));
    let mut seen_head = false;

    loop {
        let packet = match reader.read_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(e) => return Err(OpusError::Decode(e.to_string())),
        };
        // A second OpusHead marks a new logical stream in a chained Ogg file: reset the
        // decoder so the previous stream's history doesn't bleed across the boundary.
        if packet.data.starts_with(b"OpusHead") {
            if seen_head {
                decoder
                    .reset_state()
                    .map_err(|e| OpusError::Decode(e.to_string()))?;
            }
            seen_head = true;
            continue;
        }
        if packet.data.starts_with(b"OpusTags") {
            continue;
        }
        if !seen_head {
            // Audio packets before any OpusHead → this Ogg isn't Opus (Vorbis/FLAC/…).
            return Err(OpusError::NotOpus);
        }
        let n = decoder
            .decode_float(&packet.data, &mut scratch, false)
            .map_err(|e| OpusError::Decode(e.to_string()))?;
        pcm.extend_from_slice(&scratch[..n]);
    }

    if !seen_head {
        return Err(OpusError::NotOpus);
    }
    // Truncation guard: if the granule promised N samples but we decoded far fewer, the
    // file was cut short. (Chained Ogg under-counts here — the last stream's granule
    // only — so pcm.len() ≥ expected and this stays quiet, which is the safe direction.)
    if let Some(expected) = expected_len {
        if expected > 0 && pcm.len() < expected * 9 / 10 {
            return Err(OpusError::Truncated {
                got_s: pcm.len() as f32 / SAMPLE_RATE as f32,
                expected_s: expected as f32 / SAMPLE_RATE as f32,
            });
        }
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

    // (A chained-Ogg decode test would need a synthetic multi-serial fixture with
    // recomputed page CRCs to be valid; the reset_state() boundary handling in
    // decode_opus is exercised in the field, not by a hand-rolled fixture here.)

    #[test]
    fn load_opus_missing_file_errs() {
        assert!(load_opus("/no/such/file.opus").is_err());
    }

    #[test]
    fn load_opus_rejects_non_opus_bytes() {
        // A WAV (or any non-Ogg) handed to the opus loader must fail cleanly (Err),
        // not panic — the CLI only routes real .opus/.ogg here, but be defensive.
        let tmp = std::env::temp_dir().join("qwen_asr_not_an_opus.opus");
        std::fs::write(&tmp, b"RIFF....WAVEfmt ").unwrap();
        let got = load_opus(tmp.to_str().unwrap());
        let _ = std::fs::remove_file(&tmp);
        assert!(got.is_err());
    }
}
