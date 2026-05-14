#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_START = "89c7283a5b64ce7d790ed06d0c2ad0ab3d996200"
DEFAULT_END = "80947fb53ae309e2afa180646d0c855af237e9f9"


def run(cmd: list[str], cwd: Path, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(cmd)}", flush=True)
    if capture:
        return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return subprocess.run(cmd, cwd=cwd, text=True, check=True)


def output(cmd: list[str], cwd: Path) -> str:
    return run(cmd, cwd, capture=True).stdout.strip()


def commit_range(repo: Path, start: str, end: str) -> list[str]:
    spec = f"{start}^..{end}"
    try:
        raw = output(["git", "rev-list", "--reverse", "--ancestry-path", spec], repo)
    except subprocess.CalledProcessError:
        raw = output(["git", "rev-list", "--reverse", spec], repo)
    commits = [line.strip() for line in raw.splitlines() if line.strip()]
    if not commits:
        raise SystemExit(f"No commits found for range {spec}")
    return commits


def short_commit(repo: Path, commit: str) -> str:
    return output(["git", "rev-parse", "--short", commit], repo)


def subject(repo: Path, commit: str) -> str:
    return output(["git", "show", "-s", "--format=%s", commit], repo)


def ensure_worktree(repo: Path, worktree: Path, commit: str) -> None:
    if worktree.exists():
        current = output(["git", "rev-parse", "HEAD"], worktree)
        target = output(["git", "rev-parse", commit], repo)
        if current == target:
            return
        run(["git", "worktree", "remove", "--force", str(worktree)], repo)
    run(["git", "worktree", "add", "--force", "--detach", str(worktree), commit], repo)


def build_binary(worktree: Path) -> Path:
    try:
        run(["cargo", "build", "--release"], worktree)
    except subprocess.CalledProcessError:
        if not patch_cli_dependency(worktree):
            raise
        run(["cargo", "build", "--release"], worktree)
    binary = worktree / "target" / "release" / "qwen-asr"
    if not binary.exists():
        raise SystemExit(f"Built binary not found: {binary}")
    return binary


def patch_cli_dependency(worktree: Path) -> bool:
    cargo_toml = worktree / "crates" / "qwen-asr-cli" / "Cargo.toml"
    if not cargo_toml.exists():
        return False

    lines = cargo_toml.read_text(encoding="utf-8").splitlines()
    patched: list[str] = []
    changed = False
    for line in lines:
        if line.strip().startswith("qwen-asr = "):
            indent = line[: len(line) - len(line.lstrip())]
            patched.append(f'{indent}qwen-asr = {{ path = "../qwen-asr" }}')
            changed = True
        else:
            patched.append(line)

    if changed:
        cargo_toml.write_text("\n".join(patched) + "\n", encoding="utf-8")
        print(f"Patched local CLI dependency in {cargo_toml}", flush=True)
    return changed


def run_wer(args: argparse.Namespace, binary: Path, label: str) -> Path:
    summary = args.output_dir / label / "summary.json"
    if summary.exists() and not args.force:
        print(f"Reusing existing summary: {summary}", flush=True)
        return summary

    cmd = [
        sys.executable,
        str(args.script_dir / "librispeech_wer.py"),
        "--dataset",
        str(args.dataset),
        "--binary",
        str(binary),
        "--model-dir",
        str(args.model_dir),
        "--output-dir",
        str(args.output_dir),
        "--label",
        label,
        "--limit",
        str(args.limit),
    ]
    if args.download_dataset:
        cmd.append("--download-dataset")
        cmd.extend(["--dataset-url", args.dataset_url])
        cmd.extend(["--download-cache", str(args.download_cache)])
    run(cmd, args.repo)
    if not summary.exists():
        raise SystemExit(f"WER summary not found after run: {summary}")
    return summary


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_reports(report_dir: Path, rows: list[dict]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    csv_path = report_dir / "summary.csv"
    fields = [
        "commit",
        "short",
        "subject",
        "items",
        "ok",
        "failed",
        "corpus_wer",
        "macro_wer",
        "word_edits",
        "ref_words",
        "corpus_cer",
        "macro_cer",
        "summary_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_path = report_dir / "report.md"
    lines = [
        "# 100-File LibriSpeech WER Range Report",
        "",
        "| Commit | Subject | OK | Corpus WER | Macro WER | Word edits / words | Corpus CER |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| `{short}` | {subject} | {ok}/{items} | `{corpus_wer:.4f}` | `{macro_wer:.4f}` | "
            "`{word_edits} / {ref_words}` | `{corpus_cer:.4f}` |".format(**row)
        )
    lines.extend(
        [
            "",
            f"Best WER: `{min(rows, key=lambda r: r['corpus_wer'])['short']}`",
            f"Worst WER: `{max(rows, key=lambda r: r['corpus_wer'])['short']}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("")
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo = script_dir.parent.resolve()

    parser = argparse.ArgumentParser(description="Run 100-file LibriSpeech WER over a git commit range.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--dataset", type=Path, default=script_dir / "dev-clean-2")
    parser.add_argument("--model-dir", type=Path, default=repo / "qwen3-asr-0.6b")
    parser.add_argument("--worktree-root", type=Path, default=repo / "tmp" / "wer-range")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "range-results")
    parser.add_argument("--report-dir", type=Path, default=script_dir / "range-report")
    parser.add_argument("--download-dataset", action="store_true", help="Pass through to librispeech_wer.py")
    parser.add_argument(
        "--dataset-url",
        default="https://www.openslr.org/resources/12/dev-clean.tar.gz",
        help="Dataset .tar.gz URL used with --download-dataset",
    )
    parser.add_argument(
        "--download-cache",
        type=Path,
        default=script_dir / "download-cache",
        help="Where dataset archives are cached",
    )
    parser.add_argument("--force", action="store_true", help="Rerun WER even if a summary already exists")
    args = parser.parse_args()
    args.script_dir = script_dir
    args.repo = args.repo.resolve()
    args.dataset = args.dataset.resolve()
    args.model_dir = args.model_dir.resolve()
    args.worktree_root = args.worktree_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.report_dir = args.report_dir.resolve()
    args.download_cache = args.download_cache.resolve()

    commits = commit_range(args.repo, args.start, args.end)
    rows: list[dict] = []

    print(f"Commit count: {len(commits)}")
    for commit in commits:
        short = short_commit(args.repo, commit)
        label = short
        wt = args.worktree_root / short
        print("")
        print(f"=== {short} {subject(args.repo, commit)} ===", flush=True)
        summary_path = args.output_dir / label / "summary.json"
        if summary_path.exists() and not args.force:
            print(f"Reusing existing summary: {summary_path}", flush=True)
        else:
            ensure_worktree(args.repo, wt, commit)
            binary = build_binary(wt)
            summary_path = run_wer(args, binary, label)
        summary = load_summary(summary_path)
        rows.append(
            {
                "commit": commit,
                "short": short,
                "subject": subject(args.repo, commit),
                "items": summary["items"],
                "ok": summary["ok"],
                "failed": summary["failed"],
                "corpus_wer": summary["corpus_wer"],
                "macro_wer": summary["macro_wer"],
                "word_edits": summary["word_edits"],
                "ref_words": summary["ref_words"],
                "corpus_cer": summary["corpus_cer"],
                "macro_cer": summary["macro_cer"],
                "summary_path": str(summary_path),
            }
        )

    write_reports(args.report_dir, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
