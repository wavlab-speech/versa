#/bin/bash

if [ -d "wvmos" ]; then
    rm -rf wvmos
fi

# # Clone and install wvmos
git clone https://github.com/AndreevP/wvmos.git
cd wvmos
pip install -e .
cd ..

