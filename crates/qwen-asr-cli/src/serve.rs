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
//! Wire protocol (identical framing to tools/fluidaudio-serve; see
//! mrecord util/coproc_fa_worker.py). Audio by PATH, params by value:
//!   4-byte big-endian length prefix + JSON body, both directions.
//!   server → {"ready": true}                                   once per connection
//!   client → {"audio": "<path>", "regions": [[s,e],…], "language": "<lang>"}
//!   server → {"text": "...", "segments": [{"start","end","text"}]}   (format_json)
//!          | {"error": "..."}
//! stdout/stderr carry status noise and are NOT part of the protocol (the client
//! routes them to a log file).

use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Arc, Condvar, Mutex};

use serde::Deserialize;

use qwen_asr::context::QwenCtx;
use qwen_asr::{kernels, transcribe};

use crate::{format_json, load_audio};

#[derive(Deserialize)]
struct Request {
    audio: String,
    #[serde(default)]
    regions: Vec<(f64, f64)>,
    #[serde(default)]
    language: String,
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
pub fn run(sock: &str, model_dir: &str, weights_bf16: bool, workers: usize) {
    // Single-threaded matmul → inline on each connection thread, shared pool
    // untouched (see module docs). Parallelism is across connections, not within.
    kernels::set_threads(1);

    let n = workers.max(1);
    let mut ctxs = Vec::with_capacity(n);
    for k in 0..n {
        match QwenCtx::load_opts(model_dir, weights_bf16) {
            Some(c) => ctxs.push(c),
            None => {
                eprintln!("serve: failed to load model from {} (worker {})", model_dir, k);
                std::process::exit(1);
            }
        }
    }
    let pool = Arc::new(CtxPool { free: Mutex::new(ctxs), cv: Condvar::new() });

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
    eprintln!("serve: ready on {} ({} worker(s), threads=1, bf16={})", sock, n, weights_bf16);

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let pool = pool.clone();
                std::thread::spawn(move || handle_conn(stream, &pool));
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
fn handle_conn(mut stream: UnixStream, pool: &CtxPool) {
    let mut guard = CtxGuard { pool, ctx: Some(pool.take()) };
    if send_frame(&mut stream, br#"{"ready":true}"#).is_err() {
        return;  // guard returns the ctx on drop
    }
    loop {
        match recv_frame(&mut stream) {
            Ok(Some(body)) => {
                let ctx = guard.ctx.as_mut().expect("ctx checked out for this connection");
                let reply = handle_request(ctx, &body);
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

/// Decode one request body into a reply body (the JSON the client reads). Errors
/// become `{"error": ...}` frames rather than dropping the connection.
fn handle_request(ctx: &mut QwenCtx, body: &[u8]) -> String {
    let req: Request = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => return error_json(&format!("bad request: {}", e)),
    };
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
    let samples = match load_audio(&req.audio) {
        Some(s) => s,
        None => return error_json(&format!("could not load audio: {}", req.audio)),
    };
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
    match segments {
        Some(segs) => format_json(&segs),
        None => error_json("transcription failed"),
    }
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
