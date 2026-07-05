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
    hf auth login --token "$HF_TOKEN"
else
    hf auth login
fi

USER=$(hf auth whoami 2>/dev/null | tail -1)
echo "  -> $USER"

echo "[3/4] Creation du Space..."
hf repos create "$REPO" --type space --space-sdk docker --exist-ok

echo "[4/4] Push du code..."
git remote remove "$REMOTE_NAME" 2>/dev/null || true
if [ -n "${HF_TOKEN:-}" ]; then
    git remote add "$REMOTE_NAME" "https://$USER:$HF_TOKEN@huggingface.co/spaces/$USER/$REPO"
else
    git remote add "$REMOTE_NAME" "https://huggingface.co/spaces/$USER/$REPO"
fi
git push "$REMOTE_NAME" HEAD:main --force

echo ""
echo "=== Fini ! ==="
echo "Lien : https://huggingface.co/spaces/$USER/$REPO"
echo "Build : ~5 min"
