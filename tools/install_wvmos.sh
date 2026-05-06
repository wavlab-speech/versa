#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON:-python}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

git clone --depth 1 https://github.com/AndreevP/wvmos.git "$tmpdir/wvmos"
cd "$tmpdir/wvmos"
"$PYTHON_BIN" -m pip install .

"$PYTHON_BIN" - <<'PY'
import wvmos
from transformers import Wav2Vec2Model, Wav2Vec2Processor

wvmos.get_wvmos(cuda=False)
Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
PY
