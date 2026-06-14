//! Resident `--serve` mode: load the model once, answer many transcription
//! requests over an AF_UNIX socket. Mirrors the `fluidaudio-serve --serve`
//! protocol so a single resident-worker client (mrecord's coproc asr lane) can
//! drive Parakeet and qwen identically — load-once, no per-block spawn/free.
//!
//! Concurrency model. `kernels::parallel_for` runs matmul INLINE on the calling
//! thread when the global thread count is 1, leaving the process-global worker
//! pool untouched. That global pool has a single dispatch slot, so it is NOT safe
//! for concurrent callers — but with threads pinned to 1 there are no callers into
//! it at all. So we force `set_threads(1)` and get parallelism from `--workers` N
//! connection threads, each owning its own `QwenCtx` and decoding fully
//! single-threaded. Every ctx mmaps the SAME safetensors (in `--weights bf16`
//! mode the weights stay raw mmap pointers), so the OS page cache backs all N with
//! ONE physical copy of the weights regardless of N. Per-ctx anonymous RAM is just
//! the decode scratch / KV — small next to the weights.
//!
//! Wire protocol — a COMMAND/RESPONSE protocol, identical across both engines
//! (qwen-asr and tools/fluidaudio-serve; see mrecord util/coproc_qwen_worker.py).
//! 4-byte big-endian length prefix + JSON body, both directions. Audio by PATH.
//!   server → {"ready":true,"engine":"qwen","model":"<dir>","version":"<v>"}  per connection
//!   client → {"command":"<cmd>", …}                       (command is REQUIRED — no legacy fallback)
//! Commands (the metadata ones are engine-agnostic and answered identically by
//! fluidaudio-serve; only the response CONTENT differs):
//!   {"command":"ping"}        → {"pong":true}
//!   {"command":"version"}     → {"engine":"qwen","model":"<dir>","version":"<v>"}
//!   {"command":"languages"}   → {"languages":[…]}
//!   {"command":"transcribe","audio":"<path>","regions":[[s,e],…],"language":"<lang>"}
//!                             → {"text":"…","segments":[{"start","end","text"}]}   (format_json)
//! Any reply may instead be {"error":"…"} (transcribe adds "unsupported_language" when
//! the model can't do the requested language). fluidaudio-serve's `transcribe` reply is
//! a SUPERSET — it adds a `wordTimings` field — but is otherwise the same interface.
//! stdout/stderr carry status noise and are NOT part of the protocol (the client
//! routes them to a log file).

use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Arc, Condvar, Mutex};

use serde::Deserialize;

use qwen_asr::config::SUPPORTED_LANGUAGES;
use qwen_asr::context::{DecodeSettings, QwenCtx, QwenModel};
use qwen_asr::{kernels, transcribe};

use crate::{format_json, load_audio};

/// This server's engine family, advertised in the READY/`info` metadata. The
/// concrete model (e.g. `qwen3-asr-0.6b`) is reported separately; fluidaudio-serve
/// advertises `"fluidaudio"` here. Same wire interface for both (see module docs).
const ENGINE: &str = "qwen";

/// A request envelope. `command` selects the operation — there is NO default, so a
/// client must speak the command/response protocol (the legacy bare-`{audio}` form
/// is intentionally rejected). `transcribe` reads `audio`/`regions`/`language`; the
/// metadata commands (`ping`/`version`/`languages`) ignore them.
#[derive(Deserialize)]
struct Request {
    command: String,
    #[serde(default)]
    audio: String,
    #[serde(default)]
    regions: Vec<(f64, f64)>,
    #[serde(default)]
    language: String,
}

/// Server self-description, built ONCE at startup and shared (read-only) across all
/// connection threads. Sent unsolicited in the READY frame and returned by the
/// `version` command, so a client can confirm engine/model/version before driving it.
struct ServeMeta {
    ready: String,      // {"ready":true,"engine":..,"model":..,"version":..}
    info: String,       // {"engine":..,"model":..,"version":..}   (the `version` reply)
    languages: String,  // {"languages":[..]}
}

impl ServeMeta {
    fn new(model: &str) -> Self {
        let version = env!("CARGO_PKG_VERSION");
        ServeMeta {
            ready: serde_json::json!({
                "ready": true, "engine": ENGINE, "model": model, "version": version,
            })
            .to_string(),
            info: serde_json::json!({
                "engine": ENGINE, "model": model, "version": version,
            })
            .to_string(),
            languages: serde_json::json!({ "languages": SUPPORTED_LANGUAGES }).to_string(),
        }
    }
}

/// A stack of resident contexts handed out one-per-connection. `take` blocks on
/// the condvar if every ctx is busy (more connections than `--workers`); `give`
/// returns one and wakes a waiter. Each ctx is single-owner while checked out, so
/// the `&mut QwenCtx` the transcribe API wants never aliases across threads.
struct CtxPool {
    free: Mutex<Vec<QwenCtx>>,
    cv: Condvar,
}

impl CtxPool {
    fn take(&self) -> QwenCtx {
        let mut free = self.free.lock().unwrap_or_else(|p| p.into_inner());
        while free.is_empty() {
            free = self.cv.wait(free).unwrap_or_else(|p| p.into_inner());
        }
        free.pop().unwrap()
    }

    fn give(&self, ctx: QwenCtx) {
        self.free.lock().unwrap_or_else(|p| p.into_inner()).push(ctx);
        self.cv.notify_one();
    }
}

/// Load `workers` resident contexts, bind the socket, and serve forever. Returns
/// only on a fatal bind/load error (via `process::exit`); otherwise loops.
pub fn run(sock: &str, model_dir: &str, weights_bf16: bool, workers: usize,
           settings: &DecodeSettings) {
    // Single-threaded matmul → inline on each connection thread, shared pool
    // untouched (see module docs). Parallelism is across connections, not within.
    kernels::set_threads(1);

    let n = workers.max(1);
    // Load the weights ONCE and share them across all N contexts. Each context
    // is just an Arc bump + its own KV/scratch, so RAM no longer scales with N
    // for the (identical, read-only) weights — the whole point of serve mode.
    let model = match QwenModel::load_opts(model_dir, weights_bf16) {
        Some(m) => m,
        None => {
            eprintln!("serve: failed to load model from {}", model_dir);
            std::process::exit(1);
        }
    };
    // Apply the SAME DecodeSettings the CLI built from argv (defaults + any -S/-W/--loop-*
    // overrides), passed in by main — the VISIBLE contract that `--serve` decodes identically
    // to the one-shot CLI for the same flags, and can't silently drift.
    // stream_mode=false: serve always runs the segmented/clips file path, never --stream.
    let mut ctxs = Vec::with_capacity(n);
    for _ in 0..n {
        let mut ctx = QwenCtx::from_model(model.clone());
        ctx.apply_settings(settings, false);
        ctxs.push(ctx);
    }
    let pool = Arc::new(CtxPool { free: Mutex::new(ctxs), cv: Condvar::new() });

    // The model name advertised to clients: the model_dir's final component
    // (e.g. "qwen3-asr-0.6b"), falling back to the whole path if it has none.
    let model_name = std::path::Path::new(model_dir)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(model_dir);
    let meta = Arc::new(ServeMeta::new(model_name));

    // Bind only AFTER the models load, so the client's "connect succeeds ⇒ ready"
    // assumption holds. Clear any stale socket from a previous run first.
    let _ = std::fs::remove_file(sock);
    let listener = match UnixListener::bind(sock) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("serve: bind {} failed: {}", sock, e);
            std::process::exit(1);
        }
    };
    eprintln!("serve: ready on {} (engine={} model={} v{}, {} worker(s), threads=1, bf16={})",
              sock, ENGINE, model_name, env!("CARGO_PKG_VERSION"), n, weights_bf16);

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let pool = pool.clone();
                let meta = meta.clone();
                std::thread::spawn(move || handle_conn(stream, &pool, &meta));
            }
            Err(e) => eprintln!("serve: accept error: {}", e),
        }
    }
}

/// RAII: return the borrowed ctx to the pool on EVERY exit — normal, early-return,
/// or a panic unwinding out of `handle_request`/transcribe. Without this a panicking
/// decode would drop the ctx instead of returning it, permanently shrinking the pool
/// until `take()` blocks forever and the (still-alive) server stops answering.
struct CtxGuard<'a> {
    pool: &'a CtxPool,
    ctx: Option<QwenCtx>,
}

impl Drop for CtxGuard<'_> {
    fn drop(&mut self) {
        if let Some(ctx) = self.ctx.take() {
            self.pool.give(ctx);
        }
    }
}

/// Own one ctx for the connection's lifetime (via a guard so a panic can't leak it),
/// send the READY frame, then loop request→reply until the peer closes.
fn handle_conn(mut stream: UnixStream, pool: &CtxPool, meta: &ServeMeta) {
    let mut guard = CtxGuard { pool, ctx: Some(pool.take()) };
    if send_frame(&mut stream, meta.ready.as_bytes()).is_err() {
        return;  // guard returns the ctx on drop
    }
    loop {
        match recv_frame(&mut stream) {
            Ok(Some(body)) => {
                let ctx = guard.ctx.as_mut().expect("ctx checked out for this connection");
                let reply = handle_request(ctx, &body, meta);
                if send_frame(&mut stream, reply.as_bytes()).is_err() {
                    break;
                }
            }
            Ok(None) => break,  // peer closed cleanly
            Err(_) => break,    // framing/IO error → drop the connection
        }
    }
    // guard returns the ctx on drop (loop exit or unwind)
}

/// Decode one request body into a reply body (the JSON the client reads). Dispatches
/// on `command`; metadata commands answer from `meta`, `transcribe` runs the model.
/// Errors become `{"error": ...}` frames rather than dropping the connection.
fn handle_request(ctx: &mut QwenCtx, body: &[u8], meta: &ServeMeta) -> String {
    let req: Request = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => return error_json(&format!("bad request: {}", e)),
    };
    // Command/response dispatch. The metadata commands are engine-agnostic and shared
    // verbatim with fluidaudio-serve; only `transcribe` touches the model.
    match req.command.as_str() {
        "ping" => return r#"{"pong":true}"#.to_string(),
        "version" => return meta.info.clone(),
        "languages" => return meta.languages.clone(),
        "transcribe" => {}  // fall through to the decode path below
        "" => return error_json("missing 'command' field"),
        other => return error_json(&format!("unknown command: {}", other)),
    }
    // Language is per-request (the lane routes zh/ar/tr/… to one resident model),
    // so re-arm the prompt for this request's language before decoding. A language
    // this model can't do is flagged structurally (`unsupported_language`) so the lane
    // can SKIP the station cleanly instead of treating it as a transient failure.
    if ctx.set_force_language(&req.language).is_err() {
        return serde_json::json!({
            "error": format!("unsupported language: {}", req.language),
            "unsupported_language": req.language,
        })
        .to_string();
    }
    // Per-request observability (routed to qwen-serve.log by the client). The START
    // line is emitted BEFORE the decode so an in-flight wedge is visible as a START with
    // no matching DONE — the only way to catch a "stuck" decode while it is still stuck,
    // since stderr is unbuffered and the request otherwise produces no output until it
    // returns. The DONE line carries the wall time and the degeneracy counters that
    // explain a slow one. See module docs: stdout/stderr is status noise, not protocol.
    let n_regions = req.regions.len();
    eprintln!(
        "[qwen-serve] t={} START transcribe lang={} regions={} audio={}",
        now_ms(), req.language, n_regions, req.audio
    );
    let wall = std::time::Instant::now();
    let samples = match load_audio(&req.audio) {
        Some(s) => s,
        None => {
            eprintln!(
                "[qwen-serve] t={} FAIL load_audio wall_ms={} audio={}",
                now_ms(), wall.elapsed().as_millis(), req.audio
            );
            return error_json(&format!("could not load audio: {}", req.audio));
        }
    };
    // Input is always 16 kHz mono (the lane decodes opus to that before sending); used
    // only to report ×realtime in the log, so a hardcoded rate is fine here.
    let clip_ms = samples.len() as f64 * 1000.0 / 16_000.0;
    // Seconds-pairs → (start_ms, end_ms) on the original timeline; transcribe_clips
    // re-bases segment times back to it. No regions → whole-clip segmented decode.
    let regions: Vec<(u64, u64)> = req
        .regions
        .iter()
        .map(|&(s, e)| (ms(s), ms(e)))
        .collect();
    let segments = if regions.is_empty() {
        transcribe::transcribe_segmented(ctx, &samples)
    } else {
        transcribe::transcribe_clips(ctx, &samples, &regions)
    };
    let wall_ms = wall.elapsed().as_millis();
    match segments {
        Some(segs) => {
            // ×realtime = audio seconds ÷ compute seconds; a maxed>0 request is the
            // degenerate, multi-hour-per-block case (see transcribe_segment).
            let xrt = if wall_ms > 0 { clip_ms / wall_ms as f64 } else { 0.0 };
            eprintln!(
                "[qwen-serve] t={} DONE transcribe lang={} wall_ms={} clip_ms={:.0} xrt={:.1} \
                 segments={} maxed={} text_tokens={}{} audio={}",
                now_ms(), req.language, wall_ms, clip_ms, xrt,
                ctx.perf_segments, ctx.perf_maxed_segments, ctx.perf_text_tokens,
                if ctx.perf_maxed_segments > 0 { " DEGENERATE" } else { "" },
                req.audio,
            );
            format_json(&segs)
        }
        None => {
            eprintln!(
                "[qwen-serve] t={} FAIL transcribe lang={} wall_ms={} audio={}",
                now_ms(), req.language, wall_ms, req.audio
            );
            error_json("transcription failed")
        }
    }
}

/// Wall-clock epoch milliseconds for log line prefixes, so a serve log line can be
/// correlated against the coproc job ledger's timestamps. Monotonic time is used for
/// durations (`Instant`); this is only for the absolute `t=` stamp.
fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn ms(seconds: f64) -> u64 {
    (seconds * 1000.0).round().max(0.0) as u64
}

fn error_json(msg: &str) -> String {
    serde_json::json!({ "error": msg }).to_string()
}

fn send_frame(stream: &mut UnixStream, body: &[u8]) -> std::io::Result<()> {
    stream.write_all(&(body.len() as u32).to_be_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

/// Read one length-prefixed frame. `Ok(None)` means the peer closed at a frame
/// boundary (clean EOF); a partial frame is an error.
fn recv_frame(stream: &mut UnixStream) -> std::io::Result<Option<Vec<u8>>> {
    let mut len = [0u8; 4];
    match stream.read_exact(&mut len) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let n = u32::from_be_bytes(len) as usize;
    let mut buf = vec![0u8; n];
    stream.read_exact(&mut buf)?;
    Ok(Some(buf))
}
