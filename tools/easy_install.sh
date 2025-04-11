#!/bin/bash

set -eou pipefail


# An easy install scripts that go through all installation scripts.
# . ./install_fadtk.sh
. ./install_utmosv2.sh || echo "error in utmosv2"
# . ./install_warpq.sh
. ./install_scoreq.sh || echo "error in scoreq"
. ./install_nomad.sh || echo "error in nomad"
. ./install_asvspoof.sh || echo "error in asvspoof"
. ./install_pysepm.sh || echo "error in pysepm"
. ./install_srmr.sh || echo "error in srmr"
. ./install_noresqa.sh || echo "error in noresqa"
. ./setup_nisqa.sh  || echo "error in setup nisqa"
. ./install_audiobox-aesthetics.sh || echo "error in audiobox-aesthetics"
. ./install_emo2vec.sh || echo "error in emo2vec"
