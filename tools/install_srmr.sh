#/bin/bash


rm -rf srmr


# # NOTE(hyejin): a versa-specialized implementation for pysepm
git clone https://github.com/shimhz/SRMRpy.git
cd SRMRpy
pip install -e .
cd ..
