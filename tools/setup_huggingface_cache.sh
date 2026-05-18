#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CACHE_DIR="${VERSA_HF_CACHE_DIR:-$REPO_ROOT/versa_cache/huggingface}"
DISCRETE_CACHE_DIR="${VERSA_DISCRETE_SPEECH_CACHE_DIR:-$REPO_ROOT/versa_cache/discrete_speech_metrics}"
SOURCE_HF_CACHE="${SOURCE_HF_CACHE:-}"
LOCAL_ONLY="${VERSA_HF_LOCAL_ONLY:-0}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable '$PYTHON_BIN' not found. Activate your environment or set PYTHON=/path/to/python."
    exit 1
fi

mkdir -p "$CACHE_DIR"
mkdir -p "$DISCRETE_CACHE_DIR/km"

if [ -n "$SOURCE_HF_CACHE" ]; then
    echo "Syncing existing Hugging Face cache from $SOURCE_HF_CACHE to $CACHE_DIR"
    for model_dir in \
        models--microsoft--wavlm-large \
        models--facebook--hubert-base-ls960 \
        models--audeering--wav2vec2-large-robust-12-ft-emotion-msp-dim
    do
        if [ -d "$SOURCE_HF_CACHE/$model_dir" ]; then
            cp -a "$SOURCE_HF_CACHE/$model_dir" "$CACHE_DIR/"
        fi
    done
fi

export VERSA_HF_CACHE_DIR="$CACHE_DIR"
export VERSA_DISCRETE_SPEECH_CACHE_DIR="$DISCRETE_CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"
export HF_HOME="$(dirname "$CACHE_DIR")"
export TRANSFORMERS_CACHE="$CACHE_DIR"

"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

cache_dir = Path(os.environ["VERSA_HF_CACHE_DIR"]).resolve()
local_only = os.environ.get("VERSA_HF_LOCAL_ONLY") == "1"

models = [
    (
        "microsoft/wavlm-large",
        ["config.json", "pytorch_model.bin"],
    ),
    (
        "facebook/hubert-base-ls960",
        ["config.json", "pytorch_model.bin"],
    ),
    (
        "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        ["config.json", "model.safetensors", "preprocessor_config.json", "vocab.json"],
    ),
]

for repo_id, patterns in models:
    print(f"Preparing {repo_id} in {cache_dir}")
    snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        allow_patterns=patterns,
        local_files_only=local_only,
    )

print(f"Hugging Face metric cache is ready at {cache_dir}")
PY

KM_FILE="$DISCRETE_CACHE_DIR/km/km200.bin"
if [ ! -s "$KM_FILE" ]; then
    if [ "$LOCAL_ONLY" = "1" ]; then
        echo "ERROR: Missing $KM_FILE and VERSA_HF_LOCAL_ONLY=1 was set."
        echo "       Re-run without VERSA_HF_LOCAL_ONLY=1 to download the real k-means asset."
        exit 1
    fi
    curl -L -o "$KM_FILE" \
        http://sarulab.sakura.ne.jp/saeki/discrete_speech_metrics/km/km200.bin
fi
echo "Discrete speech metric cache is ready at $DISCRETE_CACHE_DIR"
echo
echo "Use these variables when running real model-backed tests:"
echo "  export VERSA_HF_CACHE_DIR=\"$CACHE_DIR\""
echo "  export VERSA_DISCRETE_SPEECH_CACHE_DIR=\"$DISCRETE_CACHE_DIR\""
echo "  export VERSA_RUN_REAL_MODEL_TESTS=1"
