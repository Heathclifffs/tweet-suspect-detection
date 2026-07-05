#!/usr/bin/env bash
set -euo pipefail

REPO="tweet-suspect-detection"
REMOTE_NAME="hf-space"

echo "=== Deploiement Hugging Face Spaces ==="

if ! command -v huggingface-cli &>/dev/null; then
    echo "[1/4] Installation huggingface-hub..."
    uv add huggingface-hub
fi

echo "[2/4] Login (si token dans HF_TOKEN, skip interactif)"
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
    huggingface-cli login
fi

echo "[3/4] Creation du Space..."
if huggingface-cli repo list --type space 2>/dev/null | grep -q "$REPO"; then
    echo "  -> Space deja existant"
else
    huggingface-cli repo create "$REPO" --type space --space-sdk streamlit
fi

echo "[4/4] Push du code existant vers le Space..."
if git remote get-url "$REMOTE_NAME" &>/dev/null; then
    echo "  -> Remote $REMOTE_NAME deja configure"
else
    USER=$(huggingface-cli whoami 2>/dev/null | head -1)
    git remote add "$REMOTE_NAME" "https://huggingface.co/spaces/$USER/$REPO"
fi
git push "$REMOTE_NAME" HEAD:main --force

echo ""
echo "=== Fini ! ==="
echo "Lien : https://huggingface.co/spaces/$(huggingface-cli whoami 2>/dev/null | head -1)/$REPO"
echo "Le premier build prend ~5 min."
