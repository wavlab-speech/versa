#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VQSCORE_DIR="$REPO_ROOT/versa/utterance_metrics/VQscore"

cd "$REPO_ROOT"

if [ ! -f ".gitmodules" ]; then
    echo "ERROR: VQScore is configured as a git submodule, but .gitmodules was not found."
    exit 1
fi

git submodule update --init --recursive versa/utterance_metrics/VQscore

if [ ! -f "$VQSCORE_DIR/models/VQVAE_models.py" ]; then
    echo "ERROR: VQScore submodule initialized, but models/VQVAE_models.py is missing."
    echo "Check the submodule checkout at $VQSCORE_DIR."
    exit 1
fi

if [ ! -f "$VQSCORE_DIR/config/QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean_github.yaml" ]; then
    echo "ERROR: VQScore config is missing from $VQSCORE_DIR/config."
    exit 1
fi

if [ ! -f "$VQSCORE_DIR/exp/QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean_github/checkpoint-dnsmos_ovr_CC=0.835.pkl" ]; then
    echo "ERROR: VQScore checkpoint is missing from $VQSCORE_DIR/exp."
    echo "The upstream submodule must include or download checkpoint-dnsmos_ovr_CC=0.835.pkl before running vqscore."
    exit 1
fi

echo "VQScore submodule and checkpoint are ready."
