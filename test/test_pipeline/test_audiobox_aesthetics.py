import logging
import math
import os

import yaml

from versa.scorer_shared import VersaScorer, compute_summary
from versa.utils_shared import find_files
from versa.definition import MetricRegistry
from versa.utterance_metrics.audiobox_aesthetics_score import (
    register_audiobox_aesthetics_metric,
)

TEST_INFO = {
    "audiobox_aesthetics_CE": 2.986576557159424,
    "audiobox_aesthetics_CU": 5.90676736831665,
    "audiobox_aesthetics_PC": 1.940537929534912,
    "audiobox_aesthetics_PQ": 5.961776256561279,
}


def info_update():

    # find files
    if os.path.isdir("test/test_samples/test2"):
        gen_files = find_files("test/test_samples/test2")

    logging.info("The number of utterances = %d" % len(gen_files))

    with open(
        "egs/separate_metrics/audiobox_aesthetics.yaml", "r", encoding="utf-8"
    ) as f:
        score_config = yaml.full_load(f)

    # Create registry and register AudioBox Aesthetics metric
    registry = MetricRegistry()
    register_audiobox_aesthetics_metric(registry)

    # Initialize VersaScorer with the populated registry
    scorer = VersaScorer(registry)

    # Load metrics using the new API
    metric_suite = scorer.load_metrics(
        score_config,
        use_gt=False,
        use_gpu=False,
    )

    assert len(score_config) > 0, "no scoring function is provided"

    # Score utterances using the new API
    score_info = scorer.score_utterances(
        gen_files, metric_suite, gt_files=None, output_file=None, io="soundfile"
    )

    summary = compute_summary(score_info)
    print("Summary: {}".format(summary), flush=True)

    for key in summary:
        # the plc mos is undeterministic
        if abs(TEST_INFO[key] - summary[key]) > 1e-4 and key != "plcmos":
            raise ValueError(
                "Value issue in the test case, might be some issue in scorer {}".format(
                    key
                )
            )
    print("check successful", flush=True)


if __name__ == "__main__":
    info_update()
