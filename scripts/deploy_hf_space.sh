#!/usr/bin/env bash
set -euo pipefail

REPO="tweet-suspect-detection"
REMOTE_NAME="hf-space"

echo "=== Deploiement Hugging Face Spaces ==="

if ! command -v hf &>/dev/null; then
    echo "[1/4] Installation huggingface-hub..."
    uv add huggingface-hub
fi

echo "[2/4] Login"
if [ -n "${HF_TOKEN:-}" ]; then
    hf auth login --token "$HF_TOKEN" --force
else
    hf auth login
fi

USER=$(hf auth whoami 2>/dev/null | awk -F': ' '/user:/ {print $2}')
echo "  -> $USER"

echo "[3/4] Creation du Space..."
hf repos create "$REPO" --type space --space-sdk docker --exist-ok

echo "[4/4] Push du code avec hf upload..."
hf upload "$REPO" . --type space \
  --include "src/" \
  --include "data/processed/" \
  --include "models/*.pkl" \
  --include "models/*.json" \
  --include "models/*.csv" \
  --include "models/*.png" \
  --include "models/*.npy" \
  --include "models/*.npz" \
  --include "models/bert_model/" \
  --exclude "models/bert_checkpoints/*" \
  --include "pyproject.toml" \
  --include "uv.lock" \
  --include "Dockerfile"
echo ""
echo "=== Fini ! ==="
echo "Lien : https://huggingface.co/spaces/$USER/$REPO"
echo "Build : ~5 min"
