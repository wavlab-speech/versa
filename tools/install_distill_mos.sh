#!/bin/bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python}"
# Includes variant-7 weights; pin code and assets together.
"$PYTHON_BIN" -m pip install 'distillmos @ git+https://github.com/microsoft/Distill-MOS.git@f676ccec11ab6205073b2689448a32f8d76a8083'
"$PYTHON_BIN" - <<'PY'
from distillmos import ConvTransformerSQAModel
ConvTransformerSQAModel().eval()
PY
