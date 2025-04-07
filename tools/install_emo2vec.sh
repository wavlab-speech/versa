#/bin/bash

if [ -d "emotion2vec" ]; then
    rm -rf emotion2vec
fi

. ./install_fairseq.sh

# # NOTE(jiatong): a versa-specialized implementation for emo2vec
git clone https://github.com/ftshijt/emotion2vec.git

# Note(jiatong): default downloading the base model
cd emotion2vec/emo2vec_versa
wget https://huggingface.co/emotion2vec/emotion2vec_base/resolve/main/emotion2vec_base.pt
cd ../..
