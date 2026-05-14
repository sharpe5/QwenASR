# LibriSpeech WER Benchmark

This directory contains Git-tracked WER benchmark scripts only. Downloaded
datasets, result files, reports, archives, and temporary worktrees are ignored by
Git.

Run 100-file WER with an existing dataset:

```bash
python3 librispeech-wer-bench/librispeech_wer.py \
  --dataset librispeech-wer-bench/dev-clean-2 \
  --binary target/release/qwen-asr \
  --model-dir qwen3-asr-0.6b \
  --output-dir librispeech-wer-bench/results \
  --label current-offline-100 \
  --limit 100 \
  --mode offline
```

Download LibriSpeech `dev-clean` automatically if the dataset directory is
missing:

```bash
python3 librispeech-wer-bench/librispeech_wer.py \
  --download-dataset \
  --dataset librispeech-wer-bench/dev-clean-2 \
  --binary target/release/qwen-asr \
  --model-dir qwen3-asr-0.6b \
  --output-dir librispeech-wer-bench/results \
  --label current-streaming-100 \
  --limit 100 \
  --mode streaming
```

The default download URL is:

```text
https://www.openslr.org/resources/12/dev-clean.tar.gz
```

