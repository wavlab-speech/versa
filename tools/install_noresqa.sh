#/bin/bash


rm -rf Noresqa

# # NOTE(hyejin): a versa-specialized implementation for Noresqa
git clone https://github.com/ftshijt/Noresqa.git

wget https://github.com/facebookresearch/Noresqa/raw/refs/heads/main/models/model_noresqa_mos.pth
wget wget https://github.com/facebookresearch/Noresqa/raw/refs/heads/main/models/model_noresqa.pth
mv model_noresqa_mos.pth Noresqa/models/model_noresqa_mos.pth
mv model_noresqa.pth Noresqa/models/model_noresqa.pth
