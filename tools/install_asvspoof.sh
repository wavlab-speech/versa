#!/bin/bash

set -e

cd "$(dirname "$0")"

## cloning the AASIST repo into the checkpoint folder
mkdir -p checkpoints
git clone https://github.com/clovaai/aasist.git checkpoints/aasist
