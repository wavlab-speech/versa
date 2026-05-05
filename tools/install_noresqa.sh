#/bin/bash

set -e

cd "$(dirname "$0")"

rm -rf Noresqa

# # NOTE(hyejin): a versa-specialized implementation for Noresqa
git clone https://github.com/ftshijt/Noresqa.git

wget https://github.com/facebookresearch/Noresqa/raw/refs/heads/main/models/model_noresqa_mos.pth
wget https://github.com/facebookresearch/Noresqa/raw/refs/heads/main/models/model_noresqa.pth
mv model_noresqa_mos.pth Noresqa/models/model_noresqa_mos.pth
mv model_noresqa.pth Noresqa/models/model_noresqa.pth

mkdir -p ../versa_cache/noresqa_model
if [ ! -f ../versa_cache/noresqa_model/wav2vec_small.pt ]; then
    wget -O ../versa_cache/noresqa_model/wav2vec_small.pt https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt
fi
