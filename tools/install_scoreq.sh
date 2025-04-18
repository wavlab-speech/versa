#/bin/bash

if [ -d "scoreq" ]; then
    rm -rf scoreq
fi

bash ./install_fairseq.sh

# # NOTE(jiatong): a versa-specialized implementation for scoreq
git clone https://github.com/ftshijt/scoreq.git
cd scoreq
pip install -e .
cd ..
