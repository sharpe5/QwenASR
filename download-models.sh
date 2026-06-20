#!/usr/bin/env bash
# process_asr/qwen-asr/download-models.sh — fetch this engine's model.
#
# Downloads Qwen3-ASR-0.6B via the engine's own `make download` (idempotent — skips if the
# model dir already exists). Public model, no token. Called by `make build` after the cargo
# build. Run standalone any time:
#   process_asr/qwen-asr/download-models.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "qwen-asr: downloading qwen3-asr-0.6b…"
make -C "$DIR" download MODEL=qwen3-asr-0.6b
echo "qwen-asr: model ready."
