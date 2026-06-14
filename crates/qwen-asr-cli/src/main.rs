mod download;
mod opus_decode;
#[cfg(target_os = "macos")]
mod live_capture;
#[cfg(unix)]
mod serve;

use qwen_asr::{audio, config, context, kernels, transcribe, align};
use config::*;
use context::{DecodeSettings, PastText, QwenCtx};

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mkv", "mov", "avi", "webm", "m4v", "flv", "ts", "mpg", "mpeg", "wmv",
];

/// Ogg/Opus container extensions decoded natively (libopus), bypassing ffmpeg.
const OPUS_EXTENSIONS: &[&str] = &["opus", "ogg"];

fn has_extension(path: &str, exts: &[&str]) -> bool {
    std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| exts.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn is_video_file(path: &str) -> bool {
    has_extension(path, VIDEO_EXTENSIONS)
}

fn is_opus_file(path: &str) -> bool {
    has_extension(path, OPUS_EXTENSIONS)
}

/// Resolve a model-directory argument to an existing path.
///
/// Used verbatim when it is absolute, or relative but already resolvable
/// against the current working directory. Otherwise — typically a bare model
/// name such as `qwen3-asr-0.6b` passed while the CLI is on `$PATH` and invoked
/// from some unrelated directory — search relative to the BINARY's own
/// location: its own directory first, then up to four parent directories
/// (parent, parent², parent³, parent⁴). The first existing match wins; if none
/// match, the original string is returned so the caller's download path can
/// handle it (downloading into ./<name>/ as before).
fn resolve_model_dir(given: String) -> String {
    let p = std::path::Path::new(&given);
    if p.is_absolute() || p.exists() {
        return given;
    }
    if let Ok(exe) = std::env::current_exe() {
        // Anchor at the binary's directory, then walk up: dir, parent, parent²,
        // parent³, parent⁴ (five directories total).
        let mut base = exe.parent().map(|d| d.to_path_buf());
        for _ in 0..5 {
            let dir = match base {
                Some(d) => d,
                None => break,
            };
            let candidate = dir.join(&given);
            if candidate.exists() {
                return candidate.to_string_lossy().into_owned();
            }
            base = dir.parent().map(|d| d.to_path_buf());
        }
    }
    given
}

/// Extract audio from a video file using ffmpeg, returning 16 kHz mono f32 samples.
fn extract_audio_from_video(path: &str) -> Option<Vec<f32>> {
    let output = std::process::Command::new("ffmpeg")
        .args(["-loglevel", "error", "-i", path, "-ar", "16000", "-ac", "1", "-f", "s16le", "pipe:1"])
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                eprintln!("Error: ffmpeg not found — install it to process video files");
                eprintln!("  macOS:  brew install ffmpeg");
                eprintln!("  Linux:  sudo apt install ffmpeg");
            } else {
                eprintln!("Error: failed to run ffmpeg: {}", e);
            }
        })
        .ok()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("Error: ffmpeg failed:\n{}", stderr);
        return None;
    }

    let raw = &output.stdout;
    if raw.len() % 2 != 0 {
        eprintln!("Error: ffmpeg returned odd number of bytes");
        return None;
    }

    let samples: Vec<f32> = raw
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect();
    Some(samples)
}

/// Load audio from an Ogg/Opus file (native libopus), a video (via ffmpeg), or a WAV file.
pub(crate) fn load_audio(path: &str) -> Option<Vec<f32>> {
    if is_opus_file(path) {
        match opus_decode::load_opus(path) {
            Ok(pcm) => Some(pcm),
            // A .ogg can instead carry Vorbis/FLAC, which libopus can't decode — hand
            // those to ffmpeg rather than failing outright.
            Err(opus_decode::OpusError::NotOpus) => {
                eprintln!("Note: {path} is Ogg but not Opus; falling back to ffmpeg.");
                extract_audio_from_video(path)
            }
            Err(e) => {
                eprintln!("Error: failed to decode Opus {path}: {e}");
                None
            }
        }
    } else if is_video_file(path) {
        extract_audio_from_video(path)
    } else {
        audio::load_wav(path)
    }
}

fn ms_to_srt_time(ms: u64) -> String {
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1_000;
    let millis = ms % 1_000;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, millis)
}

fn format_srt(segments: &[transcribe::TranscriptSegment]) -> String {
    let mut out = String::new();
    let mut idx = 1u32;
    for seg in segments {
        if seg.text.trim().is_empty() {
            continue;
        }
        out.push_str(&idx.to_string());
        out.push('\n');
        out.push_str(&ms_to_srt_time(seg.start_ms));
        out.push_str(" --> ");
        out.push_str(&ms_to_srt_time(seg.end_ms));
        out.push('\n');
        out.push_str(seg.text.trim());
        out.push_str("\n\n");
        idx += 1;
    }
    out
}

/// One speech region from a `--clip-timestamps` JSON file, in milliseconds on
/// the original file timeline.
#[derive(serde::Deserialize)]
struct ClipRegion {
    start_ms: f64,
    end_ms: f64,
}

/// The `{"speech":[...]}` envelope produced by a VAD/segmenter pre-pass.
#[derive(serde::Deserialize)]
struct ClipFile {
    speech: Vec<ClipRegion>,
}

/// Parse a `--clip-timestamps` file into `(start_ms, end_ms)` speech regions on
/// the ORIGINAL file timeline. Three shapes are accepted:
///   1. JSON envelope: `{"speech":[{"start_ms":1500,"end_ms":42300}, ...]}`
///   2. JSON array of [start_s, end_s] pairs in SECONDS — the UNEDITED output of
///      inaSpeechSegmenter / the mrecord audio2vad pre-pass: `[[1.2,5.8],[7.1,12.3]]`.
///   3. Whisper-style inline seconds list: `"s,e,s,e,…"` (comma-separated).
/// Regions are returned in file order; degenerate (`end <= start`) spans are
/// dropped.
fn parse_clip_timestamps(path: &str) -> Result<Vec<(u64, u64)>, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("failed to read {}: {}", path, e))?;
    parse_clip_content(&content)
}

/// Parse the textual contents of a clip-timestamps file. Split out from
/// [`parse_clip_timestamps`] so the parsing logic is unit-testable without
/// touching the filesystem.
fn parse_clip_content(content: &str) -> Result<Vec<(u64, u64)>, String> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Err("file is empty".to_string());
    }

    if trimmed.starts_with('{') {
        // JSON envelope {"speech":[{"start_ms":..,"end_ms":..}, ...]}.
        let parsed: ClipFile = serde_json::from_str(trimmed)
            .map_err(|e| format!("invalid JSON ({})", e))?;
        let mut regions = Vec::with_capacity(parsed.speech.len());
        for r in parsed.speech {
            let s_ms = r.start_ms.max(0.0).round() as u64;
            let e_ms = r.end_ms.max(0.0).round() as u64;
            if e_ms > s_ms {
                regions.push((s_ms, e_ms));
            }
        }
        Ok(regions)
    } else if trimmed.starts_with('[') {
        // JSON array of [start_s, end_s] pairs in SECONDS — inaSpeechSegmenter's
        // unedited output (mrecord audio2vad pre-pass): [[1.2,5.8],[7.1,12.3]].
        let pairs: Vec<Vec<f64>> = serde_json::from_str(trimmed)
            .map_err(|e| format!("invalid JSON pairs array ({})", e))?;
        let mut regions = Vec::with_capacity(pairs.len());
        for p in pairs {
            if p.len() != 2 {
                return Err(format!(
                    "expected [start_s, end_s] pairs, got a sub-array of length {}",
                    p.len()
                ));
            }
            let s_ms = (p[0].max(0.0) * 1000.0).round() as u64;
            let e_ms = (p[1].max(0.0) * 1000.0).round() as u64;
            if e_ms > s_ms {
                regions.push((s_ms, e_ms));
            }
        }
        Ok(regions)
    } else {
        // Whisper-style inline list of seconds: s,e,s,e,…
        let nums: Vec<f64> = trimmed
            .split(',')
            .filter_map(|t| {
                let t = t.trim();
                if t.is_empty() { None } else { t.parse::<f64>().ok() }
            })
            .collect();
        if nums.is_empty() || nums.len() % 2 != 0 {
            return Err(
                "expected an even-length comma-separated seconds list \"s,e,s,e,…\"".to_string(),
            );
        }
        let mut regions = Vec::with_capacity(nums.len() / 2);
        for pair in nums.chunks_exact(2) {
            let s_ms = (pair[0].max(0.0) * 1000.0).round() as u64;
            let e_ms = (pair[1].max(0.0) * 1000.0).round() as u64;
            if e_ms > s_ms {
                regions.push((s_ms, e_ms));
            }
        }
        Ok(regions)
    }
}

/// Escape a string for inclusion in a JSON string literal.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Build a Parakeet-compatible JSON transcript:
/// `{"text":..,"segments":[{"start":S,"end":E,"text":".."}]}`.
/// `start`/`end` are per-segment timestamps in seconds (3 decimal places).
pub(crate) fn format_json(segments: &[transcribe::TranscriptSegment]) -> String {
    let mut full = String::new();
    let mut segs = String::new();
    for seg in segments {
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        if !full.is_empty() {
            full.push(' ');
        }
        full.push_str(text);
        if !segs.is_empty() {
            segs.push(',');
        }
        segs.push_str(&format!(
            "{{\"start\":{:.3},\"end\":{:.3},\"text\":\"{}\"}}",
            seg.start_ms as f64 / 1000.0,
            seg.end_ms as f64 / 1000.0,
            json_escape(text),
        ));
    }
    format!("{{\"text\":\"{}\",\"segments\":[{}]}}", json_escape(&full), segs)
}

fn stream_token(piece: &str) {
    use std::io::Write;
    print!("{}", piece);
    std::io::stdout().flush().ok();
}

fn usage(prog: &str) {
    // All option descriptions align to one column: 2-space indent + label padded
    // to the longest label width (`--stream-max-new-tokens <n>` = 27) + 2 spaces.
    fn opt(label: &str, desc: &str) {
        eprintln!("  {:<27}  {}", label, desc);
    }
    // Default strings come from the ONE source (DecodeSettings) so help can't drift
    // from runtime — the old hand-written literals had (e.g. stream-chunk "2.0" vs 8.0).
    let d = DecodeSettings::default();

    eprintln!("qwen-asr — Qwen3-ASR speech-to-text (pure Rust)\n");
    eprintln!("Usage: {} -d <model_dir> (-i <input> | --stdin | --live) [options]\n", prog);
    eprintln!("Required:");
    opt("-d <dir>", "Model directory (with *.safetensors, vocab.json)");
    opt("-i <file>", "Input file: WAV (16-bit PCM), Opus (.opus/.ogg, decoded natively), or video (mp4/mkv/mov/…, requires ffmpeg)");
    opt("--stdin", "Read audio from stdin (auto-detect WAV or raw s16le 16kHz mono)");
    opt("--language <lang>", "Output language (REQUIRED). See note and language list below.");
    eprintln!("\nLive capture (macOS only):");
    opt("--live", "Capture from audio input device in real time (default: off)");
    opt("--device <name>", "Input device name (default: system default)");
    opt("--list-devices", "List available audio input devices and exit");
    opt("--vad", "Live VAD mode: detect speech segments, transcribe each (default: off)");
    eprintln!("\nOptions:");
    opt("-t <n>", "Number of threads (default: all CPUs, capped at 10)");
    opt("-S <secs>", &format!("Segment target seconds (default: {}; 0 = full-audio decode). Audio is split into ~30s segments because decoding longer spans makes the model loop and repeat itself; keep this near 30 unless you know what you're doing.", d.segment_sec));
    opt("--weights <f32|bf16>", "Weight residency (default: f32). 'bf16' keeps projection weights BF16-resident (widened per matmul) instead of f32 copies — roughly halves weight RAM, identical math, slightly slower per process.");
    opt("-W <secs>", &format!("Segment-cutting silence search window ± seconds (default: {})", d.search_sec));
    opt("--stream", "Streaming mode: process in chunks with prefix rollback (default: off)");
    opt("--stream-max-new-tokens <n>", &format!("Max generated tokens per stream step (default: {})", d.stream_max_new_tokens));
    opt("--stream-chunk-sec <secs>", &format!("Chunk size for streaming (default: {}, min ~1.0)", d.stream_chunk_sec));
    opt("--enc-window-sec <secs>", "Encoder attention window in seconds (1..8, default: 8)");
    opt("--past-text <yes|no|auto>", "Reuse previously decoded text as context (default: auto — yes for --stream, else no)");
    // Loop / repetition handling. Detection thresholds are fixed constants (not tunable);
    // --loop-detect toggles detection+recovery in both --stream and the segment/--serve path,
    // and the two recovery bounds apply only to the segment path's halve-&-re-decode.
    opt("--loop-detect <yes|no>", &format!("Detect & recover from decoder repetition loops, --stream + segment (default: {})", if d.loop_detect { "yes" } else { "no" }));
    opt("--loop-min-split-sec <secs>", &format!("Segment recovery: don't re-split a degenerate clip below this (default: {})", d.loop_min_split_sec));
    opt("--loop-max-depth <n>", &format!("Segment recovery: max halvings, e.g. 32->16->8 (0 disables) (default: {})", d.loop_max_depth));
    opt("--skip-silence", "Drop long silent spans before inference (default: off)");
    opt("--prompt <text>", "System prompt for biasing (default: none)");
    opt("--language <lang>", "Force output language (REQUIRED). The audio is decoded in ~30s segments to avoid repetition loops, and the model re-detects the language on each segment independently — so without a fixed language it guesses wrong on some blocks and quality suffers. Set this to lock the language for the whole file.");
    opt("", &format!("Valid languages: {}", SUPPORTED_LANGUAGES.join(", ")));
    eprintln!("\nAlignment mode (requires ForcedAligner model):");
    opt("--align <text>", "Align transcript to audio (word-level timestamps); supply <text> to activate (default: off)");
    opt("--align-language <lang>", "Language for word splitting (default: English)");
    eprintln!("\nSubtitle output:");
    opt("--srt [path]", "Write SRT subtitle file (default path: <input>.srt); requires -i (default: off)");
    opt("--json", "Emit JSON {\"text\":..,\"segments\":[{start,end,text}]} with per-segment timestamps in seconds (Parakeet-compatible; suppresses token streaming) (default: off)");
    opt("--clip-timestamps <path>", "VAD-driven transcription: transcribe ONLY the speech regions in <path>, skipping silence/music, in one model load. Feed it the output of a VAD/segmenter pre-pass (e.g. inaSpeechSegmenter) as JSON {\"speech\":[{\"start_ms\":..,\"end_ms\":..}]}, the unedited [[start_s,end_s],…] seconds-pairs array, or a \"s,e,s,e,…\" seconds list. Segment times are re-based to the original file timeline. Affects --json/--srt (default: off)");
    opt("--profile", "Print per-operation timing breakdown (default: off)");
    opt("--debug", "Debug output (per-layer details) (default: off)");
    opt("--silent", "No status output (only transcription on stdout) (default: off)");
    eprintln!("\nServer mode (resident, load-once):");
    opt("--serve <sock>", "Load the model once and answer many transcription requests over an AF_UNIX socket at <sock> (4-byte big-endian length prefix + JSON; request {\"audio\":<path>,\"regions\":[[s,e],…],\"language\":<lang>}, reply {\"text\":..,\"segments\":[…]}). Forces single-threaded matmul; concurrency comes from --workers. Ignores -i/--language/--clip-timestamps (those travel per request).");
    opt("--workers <n>", "Concurrent decoders in --serve mode (default: 1). Each is one QwenCtx + connection decoding single-threaded; all share the mmap'd weights (one physical copy). Open <n> client connections to drive them in parallel.");
    eprintln!("\nModel management:");
    eprintln!("  {} download [--list] [<model>] [--output <dir>]", prog);
    opt("-h", "Show this help");
    opt("-v, --version", "Show version and exit");
}

/// Shared yes/no flag-value vocabulary (`--loop-detect`, and the yes/no half of `--past-text`),
/// so the accepted spellings live in ONE place and can't drift between flags.
fn parse_bool_flag(s: &str) -> Option<bool> {
    match s.to_lowercase().as_str() {
        "yes" | "true" | "on" | "1" => Some(true),
        "no" | "false" | "off" | "0" => Some(false),
        _ => None,
    }
}

fn parse_past_text_mode(s: &str) -> Option<i32> {
    if s.eq_ignore_ascii_case("auto") {
        return Some(-1);
    }
    parse_bool_flag(s).map(|b| if b { 1 } else { 0 })
}

/// Parse a required integer flag value with a minimum, exiting on missing/unparseable/
/// below-min (strict, like `--loop-detect`) so a typo or out-of-range value is never silently
/// swallowed into the default. `min` lets a knob admit 0 as "disable" (token-run, depth) while
/// others require >= 1.
fn parse_min_i32(args: &[String], i: usize, flag: &str, min: i32) -> i32 {
    match args.get(i).and_then(|s| s.parse::<i32>().ok()) {
        Some(v) if v >= min => v,
        _ => {
            eprintln!("Error: {flag} requires an integer >= {min}");
            std::process::exit(1);
        }
    }
}

/// Float analog of [`parse_min_i32`].
fn parse_min_f32(args: &[String], i: usize, flag: &str, min: f32) -> f32 {
    match args.get(i).and_then(|s| s.parse::<f32>().ok()) {
        Some(v) if v >= min => v,
        _ => {
            eprintln!("Error: {flag} requires a number >= {min}");
            std::process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Handle --version / -v (no model needed). `--json` emits the machine-readable form
    // the version-lockstep check reads: {"engine","version"} — the same engine/version
    // the serve READY frame advertises, so the check can confirm the BUILT binary matches
    // the protocol version without loading a model.
    if args.iter().any(|a| a == "--version" || a == "-v") {
        if args.iter().any(|a| a == "--json") {
            println!("{{\"engine\":\"qwen\",\"version\":\"{}\"}}", env!("CARGO_PKG_VERSION"));
        } else {
            println!("qwen-asr {}", env!("CARGO_PKG_VERSION"));
        }
        return;
    }

    // Handle `download` subcommand: qwen-asr download [args...]
    if args.len() >= 2 && args[1] == "download" {
        download::handle_download_command(&args[2..]);
        return;
    }

    // Handle --list-devices (no model needed)
    if args.iter().any(|a| a == "--list-devices") {
        #[cfg(target_os = "macos")]
        {
            live_capture::print_devices();
        }
        #[cfg(not(target_os = "macos"))]
        {
            eprintln!("--list-devices is only supported on macOS.");
            eprintln!("On Linux, use: arecord -l");
        }
        return;
    }

    let mut model_dir: Option<String> = None;
    let mut input_wav: Option<String> = None;
    let mut verbosity = 1i32;
    let mut use_stdin = false;
    let mut live_mode = false;
    let mut device_name: Option<String> = None;
    let mut n_threads = 0i32;
    let mut segment_sec: f32 = -1.0;
    let mut search_sec: f32 = -1.0;
    let mut stream_mode = false;
    let mut vad_mode = false;
    let mut stream_max_new_tokens: i32 = -1;
    let mut stream_chunk_sec: f32 = -1.0;
    let mut enc_window_sec: f32 = -1.0;
    let mut prompt_text: Option<String> = None;
    let mut force_language: Option<String> = None;
    let mut past_text_mode: i32 = -1; // -1 auto, 0 off, 1 on
    let mut skip_silence = false;
    // Loop/repetition handling overrides (-1 / -1.0 = unset → keep DecodeSettings default).
    let mut loop_detect: i32 = -1; // -1 unset, 0 off, 1 on
    let mut loop_min_split_sec: f32 = -1.0;
    let mut loop_max_depth: i32 = -1;
    let mut profile = false;
    let mut align_text: Option<String> = None;
    let mut align_language: Option<String> = None;
    // None = no SRT, Some(path) = write SRT to path
    let mut srt_path: Option<String> = None;
    let mut srt_requested = false;
    let mut json_output = false;
    let mut clip_timestamps: Option<String> = None;
    let mut weights_bf16 = false;
    // Resident `--serve <sock>` mode: load once, answer many requests over an
    // AF_UNIX socket (see serve.rs). `--workers N` = concurrent decoders (one
    // QwenCtx + connection each, all sharing the mmap'd weights).
    let mut serve_sock: Option<String> = None;
    let mut serve_workers: usize = 1;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-d" => {
                i += 1;
                model_dir = args.get(i).cloned();
            }
            "-i" => {
                i += 1;
                input_wav = args.get(i).cloned();
            }
            "-t" => {
                i += 1;
                n_threads = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "-S" => {
                i += 1;
                segment_sec = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1.0);
            }
            "-W" => {
                i += 1;
                search_sec = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1.0);
            }
            "--weights" => {
                i += 1;
                weights_bf16 = matches!(args.get(i).map(|s| s.as_str()), Some("bf16"));
            }
            "--stream" => {
                stream_mode = true;
            }
            "--vad" => {
                vad_mode = true;
            }
            "--stream-max-new-tokens" => {
                i += 1;
                stream_max_new_tokens = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1);
            }
            "--stream-chunk-sec" => {
                i += 1;
                stream_chunk_sec = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1.0);
            }
            "--enc-window-sec" => {
                i += 1;
                enc_window_sec = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1.0);
            }
            "--loop-detect" => {
                i += 1;
                loop_detect = match args.get(i).and_then(|s| parse_bool_flag(s)) {
                    Some(true) => 1,
                    Some(false) => 0,
                    None => {
                        eprintln!("Error: --loop-detect must be yes|no, got '{}'",
                                  args.get(i).map(|s| s.as_str()).unwrap_or(""));
                        std::process::exit(1);
                    }
                };
            }
            "--loop-min-split-sec" => {
                i += 1;
                loop_min_split_sec = parse_min_f32(&args, i, "--loop-min-split-sec", 0.0);
            }
            "--loop-max-depth" => {
                // min 0: 0 disables recovery halving entirely.
                i += 1;
                loop_max_depth = parse_min_i32(&args, i, "--loop-max-depth", 0);
            }
            "--past-text" => {
                i += 1;
                if let Some(s) = args.get(i) {
                    match parse_past_text_mode(s) {
                        Some(m) => past_text_mode = m,
                        None => {
                            eprintln!("Error: --past-text must be one of yes|no|auto, got '{}'", s);
                            std::process::exit(1);
                        }
                    }
                }
            }
            "--skip-silence" => {
                skip_silence = true;
            }
            "--prompt" => {
                i += 1;
                prompt_text = args.get(i).cloned();
            }
            "--language" => {
                i += 1;
                force_language = args.get(i).cloned();
            }
            "--align" => {
                i += 1;
                align_text = args.get(i).cloned();
            }
            "--align-language" => {
                i += 1;
                align_language = args.get(i).cloned();
            }
            "--srt" => {
                srt_requested = true;
                // Optional next arg: path (if present and doesn't start with '-')
                if let Some(next) = args.get(i + 1) {
                    if !next.starts_with('-') {
                        srt_path = Some(next.clone());
                        i += 1;
                    }
                }
            }
            "--json" => {
                json_output = true;
            }
            "--clip-timestamps" => {
                i += 1;
                clip_timestamps = args.get(i).cloned();
            }
            "--serve" => {
                i += 1;
                serve_sock = args.get(i).cloned();
            }
            "--workers" => {
                i += 1;
                serve_workers = args.get(i).and_then(|s| s.parse().ok()).filter(|&n| n >= 1).unwrap_or(1);
            }
            "--stdin" => {
                use_stdin = true;
            }
            "--live" => {
                live_mode = true;
            }
            "--device" => {
                i += 1;
                device_name = args.get(i).cloned();
            }
            "--list-devices" => {
                // Already handled above, but don't error
            }
            "--profile" => {
                profile = true;
            }
            "--debug" => {
                verbosity = 2;
            }
            "--silent" => {
                verbosity = 0;
            }
            "-h" | "--help" => {
                usage(&args[0]);
                return;
            }
            other => {
                eprintln!("Unknown option: {}", other);
                usage(&args[0]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let model_dir = match model_dir {
        Some(d) => d,
        None => {
            usage(&args[0]);
            std::process::exit(1);
        }
    };

    // Resolve a relative model dir (e.g. a bare `qwen3-asr-0.6b`) against the
    // binary's own location before deciding whether to download — so the CLI
    // finds a model that sits alongside the install/source tree even when it is
    // on $PATH and invoked from an unrelated working directory.
    let model_dir = resolve_model_dir(model_dir);

    // Auto-prompt to download if model directory doesn't exist
    if !std::path::Path::new(&model_dir).exists() {
        if let Some(model) = download::find_model(&model_dir) {
            if download::prompt_download(&model_dir) {
                if let Err(e) = download::download_model(model, &model_dir) {
                    eprintln!("Download failed: {}", e);
                    std::process::exit(1);
                }
                eprintln!(); // blank line before model loading
            } else {
                eprintln!("Aborted.");
                std::process::exit(1);
            }
        } else {
            eprintln!("Error: Model directory '{}' not found.", model_dir);
            eprintln!();
            download::list_models();
            std::process::exit(1);
        }
    }

    // Build decode settings from the gold-standard defaults, then override per flag — the
    // sentinels (<0 / <=0 / mode<0) mean "flag not given → keep the default". Built ONCE here,
    // BEFORE the serve dispatch, and passed verbatim to BOTH `--serve` and the one-shot CLI
    // path, so the two can never decode differently (incl. every --loop-* knob and -S/-W).
    let mut settings = DecodeSettings::default();
    if segment_sec >= 0.0 {
        settings.segment_sec = segment_sec;
    }
    if search_sec >= 0.0 {
        settings.search_sec = search_sec;
    }
    if stream_max_new_tokens > 0 {
        settings.stream_max_new_tokens = stream_max_new_tokens;
    }
    if stream_chunk_sec > 0.0 {
        settings.stream_chunk_sec = stream_chunk_sec;
    }
    if past_text_mode >= 0 {
        settings.past_text = if past_text_mode == 1 { PastText::On } else { PastText::Off };
    }
    if skip_silence {
        settings.skip_silence = true;
    }
    // Loop/repetition overrides. The locals are the -1/-1.0 "unset" sentinel (flag absent) or a
    // value already range-validated at parse time, so each knob applies whenever it's >= its
    // minimum. token-run and depth admit 0 (disable); min-repeats/max-period require >= 1.
    if loop_detect >= 0 {
        settings.loop_detect = loop_detect == 1;
    }
    if loop_min_split_sec >= 0.0 {
        settings.loop_min_split_sec = loop_min_split_sec;
    }
    if loop_max_depth >= 0 {
        settings.loop_max_depth = loop_max_depth;
    }

    // Resident serve mode: load the model (×`--workers`) once and answer many
    // requests over the AF_UNIX socket. Language/input/clip flags don't apply —
    // they travel per-request — so dispatch here, before the one-shot validation.
    if let Some(sock) = serve_sock {
        serve::run(&sock, &model_dir, weights_bf16, serve_workers, &settings);
        return; // serve::run loops until the socket closes
    }

    if input_wav.is_none() && !use_stdin && !live_mode {
        usage(&args[0]);
        std::process::exit(1);
    }

    // Check mutual exclusivity of input modes
    let input_count = [input_wav.is_some(), use_stdin, live_mode].iter().filter(|&&x| x).count();
    if input_count > 1 {
        eprintln!("Error: -i, --stdin, and --live are mutually exclusive");
        std::process::exit(1);
    }

    // Parse --clip-timestamps up front (fail fast on a bad file). Regions are
    // (start_ms, end_ms) on the original timeline; the audio is still loaded and
    // decoded once, then only these regions are transcribed (re-based to the
    // original timeline). Only the --json / --srt timestamped paths consume it.
    let clip_regions: Option<Vec<(u64, u64)>> = match clip_timestamps {
        Some(ref path) => {
            if live_mode {
                eprintln!("Error: --clip-timestamps cannot be used with --live");
                std::process::exit(1);
            }
            if !json_output && !srt_requested {
                eprintln!(
                    "Error: --clip-timestamps requires a timestamped output mode (--json or --srt)"
                );
                std::process::exit(1);
            }
            match parse_clip_timestamps(path) {
                Ok(r) => Some(r),
                Err(e) => {
                    eprintln!("Error: --clip-timestamps {}: {}", path, e);
                    std::process::exit(1);
                }
            }
        }
        None => None,
    };

    // Require an explicit --language for transcription. The audio is decoded in
    // ~30s segments to avoid repetition loops, and the model re-detects the
    // language on each segment independently — so without a fixed language it
    // guesses wrong on some blocks and quality suffers. Alignment mode has its
    // own --align-language and does not transcribe, so it is exempt.
    if align_text.is_none() && force_language.is_none() {
        eprintln!("Error: --language <lang> is required.");
        eprintln!(
            "  Audio is decoded in ~30s segments to avoid repetition loops, and the model"
        );
        eprintln!(
            "  re-detects the language on each segment — without a fixed language it guesses"
        );
        eprintln!(
            "  wrong on some segments and quality suffers. Pass --language to lock it."
        );
        eprintln!("  Supported languages: {}", SUPPORTED_LANGUAGES.join(","));
        std::process::exit(1);
    }

    // Resolve SRT output path (--srt requires -i)
    if srt_requested {
        if input_wav.is_none() {
            eprintln!("Error: --srt requires -i <file>");
            std::process::exit(1);
        }
        if srt_path.is_none() {
            // Default: same stem as input, .srt extension
            let input = input_wav.as_ref().unwrap();
            let stem = std::path::Path::new(input)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(input.as_str());
            let dir = std::path::Path::new(input)
                .parent()
                .and_then(|p| p.to_str())
                .unwrap_or(".");
            srt_path = Some(format!("{}/{}.srt", dir, stem));
        }
    }

    kernels::set_verbose(verbosity);
    if profile {
        kernels::set_profile(true);
        kernels::profile_reset();
    }
    // --json prints one clean object on stdout: no live token streaming.
    let emit_tokens = verbosity > 0 && !json_output;

    // Initialize thread pool. Default: all CPUs, capped at 10 (more threads give
    // diminishing returns for the 0.6B model and add scheduling overhead). An
    // explicit -t value is honored as-is.
    if n_threads <= 0 {
        n_threads = (kernels::get_num_cpus() as i32).min(10);
    }
    kernels::set_threads(n_threads as usize);

    // Print optimization info
    if verbosity >= 1 {
        let opts = qwen_asr::optimization_flags();
        eprintln!(
            "Optimizations: {} | {} threads | {}",
            opts.join(", "),
            n_threads,
            std::env::consts::ARCH,
        );
    }

    // Load model
    let mut ctx = match QwenCtx::load_opts(&model_dir, weights_bf16) {
        Some(c) => c,
        None => {
            eprintln!("Failed to load model from {}", model_dir);
            std::process::exit(1);
        }
    };

    // `settings` was built once above (shared verbatim with the `--serve` path, so CLI and
    // serve can never decode differently). `apply_settings` resolves PastText::Auto from
    // `stream_mode` (ON for streaming, OFF otherwise), matching the prior behaviour.
    ctx.apply_settings(&settings, stream_mode);

    // --enc-window-sec stays a PURE override of the model's config (its default is a
    // model-architecture constant, not part of DecodeSettings — see its docs).
    if enc_window_sec >= 0.0 {
        let window_frames = (enc_window_sec * 100.0 + 0.5) as usize;
        ctx.enc_n_window_infer_override = Some(window_frames.clamp(100, 800));
    }
    if let Some(ref prompt) = prompt_text {
        if ctx.set_prompt(prompt).is_err() {
            eprintln!("Failed to set --prompt text");
            std::process::exit(1);
        }
    }
    if let Some(ref lang) = force_language {
        if ctx.set_force_language(lang).is_err() {
            eprintln!("Unsupported language for --language: {}", lang);
            eprintln!(
                "Supported languages: {}",
                SUPPORTED_LANGUAGES.join(",")
            );
            std::process::exit(1);
        }
    }

    // Alignment mode
    if let Some(ref atext) = align_text {
        let lang = align_language.as_deref().unwrap_or("English");
        let lang_normalized = match normalize_language(lang) {
            Some(l) => l,
            None => {
                eprintln!("Unsupported --align-language: {}", lang);
                eprintln!("Supported languages: {}", SUPPORTED_LANGUAGES.join(","));
                std::process::exit(1);
            }
        };

        let samples = if use_stdin {
            audio::read_pcm_stdin()
        } else {
            load_audio(input_wav.as_ref().unwrap())
        };
        let samples = match samples {
            Some(s) => s,
            None => {
                eprintln!("Failed to load audio");
                std::process::exit(1);
            }
        };

        match align::forced_align(&mut ctx, &samples, atext, &lang_normalized) {
            Some(results) => {
                // Output JSON array
                println!("[");
                for (i, r) in results.iter().enumerate() {
                    let comma = if i + 1 < results.len() { "," } else { "" };
                    // Escape the text for JSON
                    let escaped = r.text.replace('\\', "\\\\").replace('"', "\\\"");
                    println!(
                        "  {{\"text\": \"{}\", \"start\": {:.0}, \"end\": {:.0}}}{}",
                        escaped, r.start_ms, r.end_ms, comma
                    );
                }
                println!("]");
            }
            None => {
                eprintln!("Alignment failed");
                std::process::exit(1);
            }
        }

        if verbosity >= 1 {
            eprintln!(
                "Alignment: {:.0} ms (encoding: {:.0}ms, decoding: {:.0}ms)",
                ctx.perf_total_ms, ctx.perf_encode_ms, ctx.perf_decode_ms
            );
        }

        if profile {
            kernels::profile_report();
        }
        return;
    }

    // Set token callback
    if emit_tokens {
        ctx.token_cb = Some(Box::new(stream_token));
    }

    // Live capture mode
    if live_mode {
        #[cfg(not(target_os = "macos"))]
        {
            eprintln!("Error: --live is only supported on macOS.");
            eprintln!("On Linux, pipe audio via: arecord -f S16_LE -r 16000 -c 1 | qwen-asr -d <model> --stdin");
            std::process::exit(1);
        }

        #[cfg(target_os = "macos")]
        {
            run_live_capture(&mut ctx, device_name.as_deref(), stream_mode, vad_mode, verbosity, profile);
            return;
        }
    }

    // JSON mode: transcribe with per-segment timestamps, emit one Parakeet-compatible object
    if json_output {
        let samples = if use_stdin {
            audio::read_pcm_stdin()
        } else {
            let input = input_wav.as_ref().unwrap();
            if verbosity >= 1 && is_video_file(input) {
                eprintln!("Extracting audio from video: {}", input);
            }
            load_audio(input)
        };
        let samples = match samples {
            Some(s) => s,
            None => {
                eprintln!("Failed to load audio");
                std::process::exit(1);
            }
        };
        let segments = match clip_regions {
            Some(ref regions) => transcribe::transcribe_clips(&mut ctx, &samples, regions),
            None => transcribe::transcribe_segmented(&mut ctx, &samples),
        };
        let segments = match segments {
            Some(s) => s,
            None => {
                eprintln!("Transcription failed");
                std::process::exit(1);
            }
        };
        println!("{}", format_json(&segments));

        if verbosity >= 1 {
            let tokens_per_sec = if ctx.perf_total_ms > 0.0 {
                1000.0 * ctx.perf_text_tokens as f64 / ctx.perf_total_ms
            } else {
                0.0
            };
            eprintln!(
                "Inference: {:.0} ms, {} text tokens ({:.2} tok/s, encoding: {:.0}ms, decoding: {:.0}ms)",
                ctx.perf_total_ms, ctx.perf_text_tokens, tokens_per_sec,
                ctx.perf_encode_ms, ctx.perf_decode_ms
            );
        }
        if profile {
            kernels::profile_report();
        }
        return;
    }

    // SRT subtitle mode: load audio (video or WAV), transcribe with timestamps
    if let Some(ref out_path) = srt_path {
        let input = input_wav.as_ref().unwrap();
        if verbosity >= 1 {
            if is_video_file(input) {
                eprintln!("Extracting audio from video: {}", input);
            }
        }
        let samples = match load_audio(input) {
            Some(s) => s,
            None => {
                eprintln!("Failed to load audio from {}", input);
                std::process::exit(1);
            }
        };
        let segments = match clip_regions {
            Some(ref regions) => transcribe::transcribe_clips(&mut ctx, &samples, regions),
            None => transcribe::transcribe_segmented(&mut ctx, &samples),
        };
        let segments = match segments {
            Some(s) => s,
            None => {
                eprintln!("Transcription failed");
                std::process::exit(1);
            }
        };
        if emit_tokens {
            println!();
        }
        let srt = format_srt(&segments);
        match std::fs::write(out_path, &srt) {
            Ok(()) => {
                if verbosity >= 1 {
                    eprintln!("SRT written to {}", out_path);
                }
            }
            Err(e) => {
                eprintln!("Error: failed to write {}: {}", out_path, e);
                std::process::exit(1);
            }
        }

        if verbosity >= 1 {
            let tokens_per_sec = if ctx.perf_total_ms > 0.0 {
                1000.0 * ctx.perf_text_tokens as f64 / ctx.perf_total_ms
            } else {
                0.0
            };
            eprintln!(
                "Inference: {:.0} ms, {} text tokens ({:.2} tok/s, encoding: {:.0}ms, decoding: {:.0}ms)",
                ctx.perf_total_ms, ctx.perf_text_tokens, tokens_per_sec,
                ctx.perf_encode_ms, ctx.perf_decode_ms
            );
        }
        if profile {
            kernels::profile_report();
        }
        return;
    }

    // Transcribe
    let text = if stream_mode {
        let samples = if use_stdin {
            audio::read_pcm_stdin()
        } else {
            load_audio(input_wav.as_ref().unwrap())
        };
        match samples {
            Some(s) => transcribe::transcribe_stream(&mut ctx, &s),
            None => None,
        }
    } else if use_stdin {
        transcribe::transcribe_stdin(&mut ctx)
    } else {
        // load_audio routes opus/video/wav internally (same as the align branch above),
        // so there's one load path for every input — no per-format special-casing here.
        match load_audio(input_wav.as_ref().unwrap()) {
            Some(s) => transcribe::transcribe_audio(&mut ctx, &s),
            None => None,
        }
    };

    match text {
        Some(t) => {
            if emit_tokens {
                println!();
            } else {
                println!("{}", t);
            }
        }
        None => {
            eprintln!("Transcription failed");
            std::process::exit(1);
        }
    }

    if verbosity >= 1 {
        let tokens_per_sec = if ctx.perf_total_ms > 0.0 {
            1000.0 * ctx.perf_text_tokens as f64 / ctx.perf_total_ms
        } else {
            0.0
        };
        eprintln!(
            "Inference: {:.0} ms, {} text tokens ({:.2} tok/s, encoding: {:.0}ms, decoding: {:.0}ms)",
            ctx.perf_total_ms, ctx.perf_text_tokens, tokens_per_sec,
            ctx.perf_encode_ms, ctx.perf_decode_ms
        );
        if ctx.perf_audio_ms > 0.0 && ctx.perf_total_ms > 0.0 {
            let audio_s = ctx.perf_audio_ms / 1000.0;
            let infer_s = ctx.perf_total_ms / 1000.0;
            eprintln!(
                "Audio: {:.1} s processed in {:.1} s ({:.2}x realtime)",
                audio_s, infer_s, audio_s / infer_s
            );
        }
    }

    if profile {
        kernels::profile_report();
    }
}

// ========================================================================
// Live Capture Loop (macOS only)
// ========================================================================

#[cfg(target_os = "macos")]
fn run_live_capture(
    ctx: &mut QwenCtx,
    device_name: Option<&str>,
    stream_mode: bool,
    vad_mode: bool,
    verbosity: i32,
    profile: bool,
) {
    use std::time::Duration;

    // Resolve device
    let device_id = if let Some(name) = device_name {
        match live_capture::find_device_by_name(name) {
            Some(dev) => {
                if verbosity >= 1 {
                    eprintln!("Using input device: {} ({} ch)", dev.name, dev.input_channels);
                }
                dev.id
            }
            None => {
                eprintln!("Error: No input device matching '{}'", name);
                if name.to_lowercase().contains("blackhole") {
                    eprintln!();
                    eprintln!("BlackHole does not appear to be installed.");
                    eprintln!("Install it with: brew install blackhole-2ch");
                    eprintln!("Then set it up as a Multi-Output Device in Audio MIDI Setup.");
                    eprintln!("See: https://github.com/ExistentialAudio/BlackHole");
                }
                eprintln!();
                live_capture::print_devices();
                std::process::exit(1);
            }
        }
    } else {
        match live_capture::default_input_device() {
            Some(id) => {
                if verbosity >= 1 {
                    let devices = live_capture::list_input_devices();
                    if let Some(dev) = devices.iter().find(|d| d.id == id) {
                        eprintln!("Using default input device: {}", dev.name);
                    }
                }
                id
            }
            None => {
                eprintln!("Error: No default input device found");
                std::process::exit(1);
            }
        }
    };

    // Start capture
    let (rx, _handle, device_rate) = match live_capture::start_capture(device_id) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: Failed to start audio capture: {}", e);
            std::process::exit(1);
        }
    };

    let mode_label = if stream_mode { "streaming" } else if vad_mode { "VAD" } else { "segmented" };
    if verbosity >= 1 {
        if vad_mode {
            eprintln!("Listening (VAD segmented)... press Ctrl+C to stop\n");
        } else {
            eprintln!(
                "Listening ({}, {:.1}s chunks)... press Ctrl+C to stop\n",
                mode_label,
                if stream_mode { ctx.stream_chunk_sec } else { ctx.segment_sec }
            );
        }
    }

    // Set up Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl+C handler");

    // Configure context
    ctx.past_text_conditioning = true;
    ctx.reset_perf();

    // Audio accumulation
    let target_rate = 16000;
    let mut raw_buf: Vec<f32> = Vec::new();
    let mut resampled_buf: Vec<f32> = Vec::new();
    let needs_resample = (device_rate - target_rate as f64).abs() > 1.0;
    let wall_start = std::time::Instant::now();

    if stream_mode {
        // ---- Streaming mode: incremental stream_push_audio ----
        //
        // We accumulate audio and call stream_push_audio() which only
        // processes NEW audio incrementally (persistent encoder cache,
        // LCP-reused decoder prefill, monotonic token commit).
        //
        // Buffer reset after ~120s to bound memory.
        let max_window_samples: usize = 120 * target_rate as usize;
        let mut stream_state = transcribe::StreamState::new();

        // Set token callback for direct printing
        ctx.token_cb = None; // stream_push_audio returns delta text, we print it

        // Text-emission timeout: flush rollback tokens after no new text for 5s
        let mut last_text_time: Option<std::time::Instant> = None;
        let text_flush_secs = 5.0_f32;
        let mut flushed = false;

        while running.load(Ordering::SeqCst) {
            // Receive audio
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(chunk) => raw_buf.extend_from_slice(&chunk),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
            while let Ok(chunk) = rx.try_recv() {
                raw_buf.extend_from_slice(&chunk);
            }

            // Resample
            if needs_resample {
                if !raw_buf.is_empty() {
                    let resampled = qwen_asr::audio::resample(
                        &raw_buf, device_rate as i32, target_rate,
                    );
                    resampled_buf.extend_from_slice(&resampled);
                    raw_buf.clear();
                }
            } else {
                resampled_buf.append(&mut raw_buf);
            }

            // Reset window if buffer exceeds max
            if resampled_buf.len() > max_window_samples {
                // Flush rollback tokens before reset
                if let Some(delta) = transcribe::stream_push_audio(
                    ctx, &resampled_buf, &mut stream_state, true
                ) {
                    if !delta.is_empty() {
                        print!("{}", delta);
                    }
                }
                println!();
                resampled_buf.clear();
                stream_state.reset();
                last_text_time = None;
                flushed = false;
                continue;
            }

            // Determine if we should finalize: flush rollback tokens
            // when no new text has been emitted for 5 seconds
            let finalize = !flushed
                && last_text_time.is_some_and(|t| t.elapsed().as_secs_f32() >= text_flush_secs);

            // Process all available full chunks
            if resampled_buf.len() > stream_state.audio_cursor() {
                if let Some(delta) = transcribe::stream_push_audio(
                    ctx, &resampled_buf, &mut stream_state, finalize
                ) {
                    if !delta.is_empty() {
                        print!("{}", delta);
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                        last_text_time = Some(std::time::Instant::now());
                        flushed = false;
                    } else if finalize {
                        flushed = true; // Don't keep calling finalize
                    }
                }
            }
        }

        // Final flush
        if !raw_buf.is_empty() && needs_resample {
            let resampled = qwen_asr::audio::resample(
                &raw_buf, device_rate as i32, target_rate,
            );
            resampled_buf.extend_from_slice(&resampled);
        } else {
            resampled_buf.append(&mut raw_buf);
        }

        if resampled_buf.len() > stream_state.audio_cursor() {
            if let Some(delta) = transcribe::stream_push_audio(
                ctx, &resampled_buf, &mut stream_state, true // finalize: flush rollback
            ) {
                if !delta.is_empty() {
                    print!("{}", delta);
                }
            }
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
        println!();
    } else if vad_mode {
        // ---- VAD mode: energy-based speech detection + segment transcription ----
        //
        // Detect speech using RMS energy. When speech ends (silence > 1.5s),
        // transcribe the accumulated speech segment using transcribe_audio().
        // This gives better accuracy than streaming (full segment context)
        // with automatic speech boundary detection.
        let speech_threshold: f32 = 0.001;
        let silence_hangover_secs = 1.5_f32;
        let min_segment_secs = 0.5_f32;
        let max_segment_secs = 30.0_f32;
        let min_segment_samples = (min_segment_secs * target_rate as f32) as usize;
        let max_segment_samples = (max_segment_secs * target_rate as f32) as usize;
        let check_samples = (target_rate as usize) * 30 / 1000; // 30ms window for RMS

        let mut speech_active = false;
        let mut silence_start: Option<std::time::Instant> = None;
        let mut speech_start_idx: usize = 0;

        // Keep a small pre-speech buffer to avoid clipping word beginnings
        let pre_speech_samples = (target_rate as usize) / 4; // 250ms lookback

        // Disable token callback — we print the full result after each segment
        ctx.token_cb = None;

        // Cross-segment context: accumulate text to use as prompt for next segment
        let mut accumulated_text = String::new();

        while running.load(Ordering::SeqCst) {
            // Receive audio
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(chunk) => raw_buf.extend_from_slice(&chunk),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
            while let Ok(chunk) = rx.try_recv() {
                raw_buf.extend_from_slice(&chunk);
            }

            // Resample
            if needs_resample {
                if !raw_buf.is_empty() {
                    let resampled = qwen_asr::audio::resample(
                        &raw_buf, device_rate as i32, target_rate,
                    );
                    resampled_buf.extend_from_slice(&resampled);
                    raw_buf.clear();
                }
            } else {
                resampled_buf.append(&mut raw_buf);
            }

            // Compute RMS energy of latest 30ms
            let buf_len = resampled_buf.len();
            let rms = if buf_len >= check_samples {
                let tail = &resampled_buf[buf_len - check_samples..];
                let sum_sq: f32 = tail.iter().map(|&s| s * s).sum();
                (sum_sq / check_samples as f32).sqrt()
            } else {
                0.0
            };
            let is_speech = rms >= speech_threshold;

            // Periodic RMS debug output
            if verbosity >= 2 && buf_len % (target_rate as usize * 2) < check_samples {
                eprintln!("  [VAD] rms={:.6} threshold={:.4} speech={}",
                    rms, speech_threshold, if speech_active { "active" } else { "inactive" });
            }

            if !speech_active {
                if is_speech {
                    // Speech started — mark the start with lookback
                    speech_active = true;
                    silence_start = None;
                    speech_start_idx = buf_len.saturating_sub(pre_speech_samples);
                    if verbosity >= 2 {
                        eprintln!("  [VAD] speech start at {:.1}s",
                            buf_len as f32 / target_rate as f32);
                    }
                } else {
                    // No speech — bound buffer to avoid unlimited growth
                    // Keep only last 0.5s for lookback context
                    let keep = (target_rate as usize) / 2;
                    if resampled_buf.len() > keep * 4 {
                        let drain = resampled_buf.len() - keep;
                        resampled_buf.drain(..drain);
                    }
                }
            } else {
                // Speech is active
                let segment_len = buf_len - speech_start_idx;

                if is_speech {
                    // Still speaking — reset silence timer
                    silence_start = None;

                    // Force-flush if segment exceeds max duration
                    if segment_len >= max_segment_samples {
                        if verbosity >= 2 {
                            eprintln!("  [VAD] max segment reached ({:.1}s), flushing",
                                segment_len as f32 / target_rate as f32);
                        }
                        let segment = &resampled_buf[speech_start_idx..];
                        // Set previous text as context
                        if !accumulated_text.is_empty() {
                            ctx.prompt = Some(accumulated_text.clone());
                            ctx.prompt_tokens_ready = false;
                        }
                        ctx.reset_perf();
                        if let Some(text) = transcribe::transcribe_audio(ctx, segment) {
                            if !text.is_empty() {
                                println!("{}", text);
                                accumulated_text.push_str(&text);
                            }
                        }
                        resampled_buf.clear();
                        speech_active = false;
                        silence_start = None;
                    }
                } else {
                    // Silence during speech — track duration
                    if silence_start.is_none() {
                        silence_start = Some(std::time::Instant::now());
                    }
                    if let Some(start) = silence_start {
                        if start.elapsed().as_secs_f32() >= silence_hangover_secs {
                            // End of utterance — transcribe the segment
                            if segment_len >= min_segment_samples {
                                // Trim trailing silence (keep only 200ms of it)
                                let trail_keep = (target_rate as usize) / 5;
                                let seg_end = (buf_len - check_samples + trail_keep).min(buf_len);
                                let segment = &resampled_buf[speech_start_idx..seg_end];

                                if verbosity >= 2 {
                                    eprintln!("  [VAD] speech end, segment {:.1}s",
                                        segment.len() as f32 / target_rate as f32);
                                }

                                ctx.reset_perf();
                                // Set previous text as context
                                if !accumulated_text.is_empty() {
                                    ctx.prompt = Some(accumulated_text.clone());
                                    ctx.prompt_tokens_ready = false;
                                }
                                let t0 = std::time::Instant::now();
                                if let Some(text) = transcribe::transcribe_audio(ctx, segment) {
                                    if !text.is_empty() {
                                        println!("{}", text);
                                        accumulated_text.push_str(&text);
                                        if verbosity >= 1 {
                                            let audio_secs = segment.len() as f32 / target_rate as f32;
                                            let compute_secs = t0.elapsed().as_secs_f32();
                                            eprintln!(
                                                "  ({:.1}s audio in {:.1}s, {:.1}x realtime)",
                                                audio_secs, compute_secs,
                                                audio_secs / compute_secs.max(0.001)
                                            );
                                        }
                                    }
                                }
                            } else if verbosity >= 2 {
                                eprintln!("  [VAD] segment too short ({:.2}s), discarding",
                                    segment_len as f32 / target_rate as f32);
                            }

                            resampled_buf.clear();
                            speech_active = false;
                            silence_start = None;
                        }
                    }
                }
            }
        }

        // Flush remaining speech on Ctrl+C
        if speech_active && resampled_buf.len() > speech_start_idx + min_segment_samples {
            let segment = &resampled_buf[speech_start_idx..];
            ctx.reset_perf();
            if let Some(text) = transcribe::transcribe_audio(ctx, segment) {
                if !text.is_empty() {
                    println!("{}", text);
                }
            }
        }
    } else {
        // ---- Segmented mode: independent segments ----
        if ctx.segment_sec <= 0.0 {
            ctx.segment_sec = 5.0;
        }
        let segment_samples_16k = (ctx.segment_sec * target_rate as f32) as usize;

        while running.load(Ordering::SeqCst) {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(chunk) => raw_buf.extend_from_slice(&chunk),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
            while let Ok(chunk) = rx.try_recv() {
                raw_buf.extend_from_slice(&chunk);
            }

            if needs_resample {
                if !raw_buf.is_empty() {
                    let resampled = qwen_asr::audio::resample(
                        &raw_buf, device_rate as i32, target_rate,
                    );
                    resampled_buf.extend_from_slice(&resampled);
                    raw_buf.clear();
                }
            } else {
                resampled_buf.append(&mut raw_buf);
            }

            if resampled_buf.len() >= segment_samples_16k {
                ctx.reset_perf();
                let _text = transcribe::transcribe_audio(ctx, &resampled_buf);
                resampled_buf.clear();
                if verbosity > 0 {
                    println!();
                }
            }
        }

        // Flush remaining
        if !raw_buf.is_empty() && needs_resample {
            let resampled = qwen_asr::audio::resample(
                &raw_buf, device_rate as i32, target_rate,
            );
            resampled_buf.extend_from_slice(&resampled);
        } else {
            resampled_buf.append(&mut raw_buf);
        }
        if !resampled_buf.is_empty() {
            ctx.reset_perf();
            let _text = transcribe::transcribe_audio(ctx, &resampled_buf);
            if verbosity > 0 {
                println!();
            }
        }
    }

    // ---- Benchmark summary ----
    let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    let audio_s = resampled_buf.len() as f64 / target_rate as f64;

    if verbosity >= 1 {
        eprintln!("\nStopped.");
        let tokens_per_sec = if ctx.perf_total_ms > 0.0 {
            1000.0 * ctx.perf_text_tokens as f64 / ctx.perf_total_ms
        } else {
            0.0
        };
        eprintln!(
            "Inference: {:.0} ms, {} text tokens ({:.2} tok/s, encoding: {:.0}ms, decoding: {:.0}ms)",
            ctx.perf_total_ms, ctx.perf_text_tokens, tokens_per_sec,
            ctx.perf_encode_ms, ctx.perf_decode_ms
        );
        if audio_s > 0.0 && ctx.perf_total_ms > 0.0 {
            let infer_s = ctx.perf_total_ms / 1000.0;
            eprintln!(
                "Audio: {:.1} s processed in {:.1} s compute ({:.2}x realtime), {:.1} s wall clock",
                audio_s, infer_s, audio_s / infer_s, wall_ms / 1000.0
            );
        }
    }

    if profile {
        kernels::profile_report();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opus_extensions_route_to_native_loader() {
        // .opus/.ogg (any case) must take the native libopus path, NOT be mistaken
        // for a video (ffmpeg) or fall through to the WAV parser.
        for p in ["clip.opus", "CLIP.OPUS", "a/b/c.ogg", "x.Opus"] {
            assert!(is_opus_file(p), "{p} should be opus");
            assert!(!is_video_file(p), "{p} must not be treated as video");
        }
        for p in ["clip.wav", "clip.mp4", "clip.mkv", "noext"] {
            assert!(!is_opus_file(p), "{p} should not be opus");
        }
    }

    #[test]
    fn parse_json_tolerates_whitespace() {
        let s = "{ \"speech\" : [ { \"start_ms\" : 1500 , \"end_ms\" : 42300 } ] }";
        assert_eq!(parse_clip_content(s).unwrap(), vec![(1500, 42300)]);
    }

    #[test]
    fn parse_json_invalid_errors() {
        assert!(parse_clip_content(r#"{"speech":[{"start_ms":1500"#).is_err());
    }

    #[test]
    fn parse_json_multiple_regions_in_order() {
        let s = r#"{"speech":[{"start_ms":1500,"end_ms":42300},
                              {"start_ms":58000,"end_ms":131200}]}"#;
        let r = parse_clip_content(s).unwrap();
        assert_eq!(r, vec![(1500, 42300), (58000, 131200)]);
    }

    #[test]
    fn parse_json_rounds_and_drops_degenerate() {
        // 1000.6 -> 1001, 2000.4 -> 2000; second region end == start is dropped.
        let s = r#"{"speech":[{"start_ms":1000.6,"end_ms":2000.4},
                              {"start_ms":3000,"end_ms":3000}]}"#;
        let r = parse_clip_content(s).unwrap();
        assert_eq!(r, vec![(1001, 2000)]);
    }

    #[test]
    fn parse_json_missing_field_errors() {
        // A region missing end_ms fails deserialization (both fields required).
        let s = r#"{"speech":[{"start_ms":1000,"end_ms":2000},{"start_ms":3000}]}"#;
        assert!(parse_clip_content(s).is_err());
    }

    #[test]
    fn parse_pairs_array_seconds() {
        // inaSpeechSegmenter's unedited output: [[start_s, end_s], ...] in SECONDS.
        let s = "[[1.2, 5.8], [7.1, 12.3]]";
        assert_eq!(parse_clip_content(s).unwrap(), vec![(1200, 5800), (7100, 12300)]);
    }

    #[test]
    fn parse_pairs_array_drops_degenerate_and_errors_on_bad_arity() {
        assert_eq!(parse_clip_content("[[1.0,1.0],[2.0,3.0]]").unwrap(), vec![(2000, 3000)]);
        assert!(parse_clip_content("[[1.0,2.0,3.0]]").is_err());
    }

    #[test]
    fn parse_json_empty_speech_is_ok() {
        // A valid envelope with no speech regions -> nothing to transcribe.
        assert_eq!(parse_clip_content(r#"{"speech":[]}"#).unwrap(), vec![]);
    }

    #[test]
    fn parse_empty_errors() {
        assert!(parse_clip_content("   \n  ").is_err());
    }

    #[test]
    fn parse_seconds_list() {
        // 1.5s,42.3s -> ms; 58s,131.2s -> ms.
        let r = parse_clip_content("1.5,42.3,58,131.2").unwrap();
        assert_eq!(r, vec![(1500, 42300), (58000, 131200)]);
    }

    #[test]
    fn parse_seconds_list_with_spaces() {
        let r = parse_clip_content(" 1.0, 2.0 , 3.0, 4.0 ").unwrap();
        assert_eq!(r, vec![(1000, 2000), (3000, 4000)]);
    }

    #[test]
    fn parse_seconds_list_odd_length_errors() {
        assert!(parse_clip_content("1.0,2.0,3.0").is_err());
    }
}
