#!/usr/bin/env bash
# process_asr/qwen-asr/download-models.sh — fetch this engine's model.
#
# Downloads Qwen3-ASR-0.6B (public) via the engine's own `make download`. SKIPS IMMEDIATELY if
# the model is already present. Called by `make build`. Run standalone:
#   process_asr/qwen-asr/download-models.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$DIR/qwen3-asr-0.6b/model.safetensors" ]; then
  echo "qwen-asr: model already present — skipping."
  exit 0
fi

echo "qwen-asr: downloading qwen3-asr-0.6b …"
if ! make -C "$DIR" download MODEL=qwen3-asr-0.6b; then
  cat >&2 <<EOF
ERROR: failed to download Qwen3-ASR-0.6B (public model on Hugging Face).
Check your network, then re-run:  process_asr/qwen-asr/download-models.sh
(manual: cd process_asr/qwen-asr && make download MODEL=qwen3-asr-0.6b)
EOF
  exit 1
fi
echo "qwen-asr: model ready."
