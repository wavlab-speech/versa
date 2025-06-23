#!/bin/bash
set -e

pip install faster-whisper

if ! command -v nvcc &>/dev/null; then
  echo "Error: nvcc not found. Please install the CUDA Toolkit first." >&2
  exit 1
fi

cuda_ver=$(nvcc --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p')
cuda_major=${cuda_ver%%.*}
echo "Detected CUDA version:$cuda_ver"

if [ "$cuda_major" -ge 12 ]; then
  conda install -c conda-forge "cudnn=9.*" "numpy<2.3"
elif [ "$cuda_major" -eq 11 ]; then
  conda install -c conda-forge "cudnn=8.*" "numpy<2.3"
  pip install --force-reinstall 'ctranslate2==3.24.0' 'numpy<2.2'
else
  echo "Error: Unsupported CUDA major version $cuda_major" >&2
  exit 1
fi