//! Parity guarantee: `--serve` must produce byte-identical output to the one-shot CLI
//! on the SAME clip. Both paths now build their decode config from the single
//! `DecodeSettings` source (qwen-asr `context.rs`), so this test is what makes that a
//! guarantee instead of a coincidence — any future change that lets the two paths drift
//! (a default applied in one but not the other, a new flag serve forgets to mirror)
//! fails right here.
//!
//! SKIPPED (not failed) when the model dir or sample clip is absent, so `cargo test`
//! stays green on a checkout without the ~600 MB model; it runs the real end-to-end
//! comparison wherever the model is present (the dev box, the coproc node).

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn send_frame(stream: &mut UnixStream, body: &[u8]) -> std::io::Result<()> {
    stream.write_all(&(body.len() as u32).to_be_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn recv_frame(stream: &mut UnixStream) -> std::io::Result<Vec<u8>> {
    let mut len = [0u8; 4];
    stream.read_exact(&mut len)?;
    let n = u32::from_be_bytes(len) as usize;
    let mut buf = vec![0u8; n];
    stream.read_exact(&mut buf)?;
    Ok(buf)
}

#[test]
fn serve_output_equals_cli_output() {
    let root = repo_root();
    let model = root.join("qwen3-asr-0.6b");
    let clip = root.join("audio.wav");
    if !model.is_dir() || !clip.is_file() {
        eprintln!(
            "SKIP serve_cli_parity: model ({}) or clip ({}) absent",
            model.display(),
            clip.display()
        );
        return;
    }
    let bin = env!("CARGO_BIN_EXE_qwen-asr");
    let model = model.to_str().unwrap();
    let clip = clip.to_str().unwrap();
    let lang = "English";

    // ── CLI: the gold standard ──
    let cli = Command::new(bin)
        .args(["-d", model, "-i", clip, "--language", lang, "--json", "--silent"])
        .output()
        .expect("run CLI");
    assert!(
        cli.status.success(),
        "CLI run failed: {}",
        String::from_utf8_lossy(&cli.stderr)
    );
    let cli_json = String::from_utf8(cli.stdout).unwrap().trim().to_string();
    assert!(cli_json.starts_with('{'), "CLI did not emit JSON: {cli_json:?}");

    // ── serve: load once, transcribe the same clip over the socket ──
    let sock = std::env::temp_dir().join(format!("qwen-parity-{}.sock", std::process::id()));
    let _ = std::fs::remove_file(&sock);
    let mut serve = Command::new(bin)
        .args(["--serve", sock.to_str().unwrap(), "-d", model])
        .spawn()
        .expect("spawn serve");

    // serve binds the socket only AFTER the model loads, so connect-success ⇒ ready.
    let mut stream = {
        let deadline = Instant::now() + Duration::from_secs(180);
        loop {
            if let Ok(s) = UnixStream::connect(&sock) {
                break s;
            }
            assert!(Instant::now() < deadline, "serve did not come up within 180s");
            std::thread::sleep(Duration::from_millis(100));
        }
    };
    let _ready = recv_frame(&mut stream).expect("READY frame");
    // {:?} on a &str yields a valid JSON-escaped string literal — keeps this test
    // dependency-free (no serde_json) for a simple {audio, language} request.
    let req = format!(
        r#"{{"command":"transcribe","audio":{clip:?},"language":{lang:?}}}"#
    );
    send_frame(&mut stream, req.as_bytes()).unwrap();
    let serve_json = String::from_utf8(recv_frame(&mut stream).unwrap())
        .unwrap()
        .trim()
        .to_string();

    let _ = serve.kill();
    let _ = serve.wait();
    let _ = std::fs::remove_file(&sock);

    assert_eq!(
        serve_json, cli_json,
        "serve output diverged from the CLI on the same clip — a decode default differs \
         between the two paths. They must BOTH derive from DecodeSettings (context.rs)."
    );
}
