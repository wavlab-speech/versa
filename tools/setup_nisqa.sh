#/bin/bash

set -e

cd "$(dirname "$0")"

if [ -d "NISQA" ]; then
    rm -rf NISQA
fi

# # NOTE(jiatong): only for pre-trained model
git clone https://github.com/gabrielmittag/NISQA.git
