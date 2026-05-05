#/bin/bash

set -e

cd "$(dirname "$0")"

mkdir -p ../versa_cache/noresqa_model
if [ ! -f ../versa_cache/noresqa_model/model_noresqa_mos.pth ]; then
    wget -O ../versa_cache/noresqa_model/model_noresqa_mos.pth https://github.com/facebookresearch/Noresqa/raw/refs/heads/main/models/model_noresqa_mos.pth
fi
if [ ! -f ../versa_cache/noresqa_model/model_noresqa.pth ]; then
    wget -O ../versa_cache/noresqa_model/model_noresqa.pth https://github.com/facebookresearch/Noresqa/raw/refs/heads/main/models/model_noresqa.pth
fi
if [ ! -f ../versa_cache/noresqa_model/wav2vec_small.pt ]; then
    wget -O ../versa_cache/noresqa_model/wav2vec_small.pt https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt
fi
