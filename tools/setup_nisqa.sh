#/bin/bash

set -e

cd "$(dirname "$0")"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

# NOTE(jiatong): only for pre-trained model weights.
git clone --depth 1 https://github.com/gabrielmittag/NISQA.git "$tmpdir/NISQA"
mkdir -p ../versa_cache/nisqa
cp "$tmpdir/NISQA"/weights/*.tar ../versa_cache/nisqa/
cp "$tmpdir/NISQA"/weights/LICENSE_model_weights ../versa_cache/nisqa/
