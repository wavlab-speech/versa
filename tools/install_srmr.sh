#/bin/bash

set -e

cd "$(dirname "$0")"

rm -rf SRMRpy


# # NOTE(hyejin): a versa-specialized implementation for pysepm
git clone https://github.com/shimhz/SRMRpy.git
cd SRMRpy
pip install -e .
