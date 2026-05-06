#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable '$PYTHON_BIN' not found. Activate your environment or set PYTHON=/path/to/python."
    exit 1
fi

"$PYTHON_BIN" -m pip install torch-log-wmse
