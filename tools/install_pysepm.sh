#/bin/bash


rm -rf pysepm


# # NOTE(hyejin): a versa-specialized implementation for pysepm
git clone https://github.com/ftshijt/pysepm.git
cd pysepm
pip install -e .
cd ..
