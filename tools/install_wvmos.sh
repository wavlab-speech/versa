#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON:-python}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

git clone --depth 1 https://github.com/AndreevP/wvmos.git "$tmpdir/wvmos"
cd "$tmpdir/wvmos"
"$PYTHON_BIN" -m pip install .
