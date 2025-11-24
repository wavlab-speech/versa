#!/bin/bash

## cloning the MultiGauss repo into the checkpoint folder
tools_dir=$(dirname $(realpath $0))
git clone https://github.com/fcumlin/MultiGauss.git $tools_dir/checkpoints/multigauss
pip install gin-config
