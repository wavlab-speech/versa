#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable '$PYTHON_BIN' not found. Activate your environment or set PYTHON=/path/to/python."
    exit 1
fi


cd "$(dirname "$0")"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

# NOTE(jiatong): a versa-specialized implementation for singer identity.
git clone --depth 1 https://github.com/ftshijt/ssl-singer-identity.git "$tmpdir/ssl-singer-identity"
cd "$tmpdir/ssl-singer-identity"
"$PYTHON_BIN" -c "from pathlib import Path; p=Path('singer_identity/utils/fetch_pretrained.py'); s=p.read_text(); old='''repo_id=source,\n                filename=filename,\n                use_auth_token=use_auth_token,'''; new='''repo_id=source,\n                filename=filename,\n                token=use_auth_token or None,'''; p.write_text(s.replace(old, new))"
"$PYTHON_BIN" -c "from pathlib import Path; p=Path('singer_identity/utils/fetch_pretrained.py'); s=p.read_text(); old='''except ValueError:\n        if pymodule_file == \"custom.py\":'''; new='''except Exception:\n        if pymodule_file == \"custom.py\":'''; p.write_text(s.replace(old, new))"
"$PYTHON_BIN" -m pip install .
"$PYTHON_BIN" -m pip install nnAudio torchvision
