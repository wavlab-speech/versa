#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -n "${PYTHON:-}" ]; then
    PYTHON_BIN="$PYTHON"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python3"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found. Activate your environment or set PYTHON=/path/to/python."
    exit 1
fi

cd "$REPO_ROOT"

"$SCRIPT_DIR/install_fairseq.sh" || {
    echo "fairseq installation exit"
    exit 1
}

# # NOTE(jiatong): a versa-specialized implementation for scoreq
if [ -d "scoreq/.git" ]; then
    git -C scoreq fetch origin
    git -C scoreq checkout main
    git -C scoreq pull --ff-only origin main
elif [ -d "scoreq" ]; then
    echo "ERROR: scoreq exists but is not a git checkout. Move it aside and retry."
    exit 1
else
    git clone https://github.com/ftshijt/scoreq.git
fi
cd scoreq
"$PYTHON_BIN" -m pip install -e . --no-deps
cd ..
