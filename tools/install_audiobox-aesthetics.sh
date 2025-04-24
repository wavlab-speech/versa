#/bin/bash

if [ -d "audiobox-aesthetics" ]; then
    rm -rf audiobox-aesthetics
fi

git clone https://github.com/ftshijt/audiobox-aesthetics.git
cd audiobox-aesthetics

pip install huggingface_hub
pip install rich
pip install -e .
cd ..
