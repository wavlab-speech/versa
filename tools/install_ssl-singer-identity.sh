#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable '$PYTHON_BIN' not found. Activate your environment or set PYTHON=/path/to/python."
    exit 1
fi


# # NOTE(jiatong): a versa-specialized implementation for singer identity
if [ -d "ssl-singer-identity/.git" ]; then
    git -C ssl-singer-identity fetch origin
    git -C ssl-singer-identity checkout main
    git -C ssl-singer-identity pull --ff-only origin main
elif [ -d "ssl-singer-identity" ]; then
    echo "ERROR: ssl-singer-identity exists but is not a git checkout. Move it aside and retry."
    exit 1
else
    git clone https://github.com/ftshijt/ssl-singer-identity.git
fi
cd ssl-singer-identity
perl -0pi -e 's/use_auth_token=use_auth_token/token=use_auth_token or None/g' singer_identity/utils/fetch_pretrained.py
perl -0pi -e 's/except ValueError:\\n        if pymodule_file == "custom\\.py":/except Exception:\\n        if pymodule_file == "custom.py":/g' singer_identity/utils/fetch_pretrained.py
"$PYTHON_BIN" -m pip install -e .
"$PYTHON_BIN" -m pip install nnAudio torchvision
cd ..
