#!/usr/bin/env python3
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

logger = logging.getLogger(__name__)

import numpy as np

try:
    from srmrpy import srmr  # Import the srmr package for speech quality metrics
except ImportError:
    logger.info("srmr is not installed. Please use `tools/install_srmr.sh` to install")
    srmr = None


def srmr_metric(
    pred_x,
    fs,
    n_cochlear_filters=23,
    low_freq=125,
    min_cf=4,
    max_cf=128,
    fast=True,
    norm=False,
):
    if srmr is None:
        raise ImportError(
            # Error message if SRMRpy is not installed
        )
    srmr_score = srmr(
        pred_x,
        fs,
        n_cochlear_filters=n_cochlear_filters,
        low_freq=low_freq,
        min_cf=min_cf,
        max_cf=max_cf,
        fast=fast,
        norm=norm,
    )

    return {
        "srmr": srmr_score,
    }


if __name__ == "__main__":

    a = np.random.random(16000)
    score = srmr_metric(a, 16000)
    print(score)
