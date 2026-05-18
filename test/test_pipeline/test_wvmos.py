import logging
import os

import yaml

from versa.definition import MetricRegistry
from versa.scorer_shared import VersaScorer, compute_summary
from versa.utils_shared import find_files
from versa.utterance_metrics.wvmos import register_wvmos_metric

TEST_INFO = {"wvmos": 0.621284008026123}


def info_update():
    # find files
    if os.path.isdir("test/test_samples/test2"):
        gen_files = find_files("test/test_samples/test2")

    logging.info("The number of utterances = %d" % len(gen_files))

    with open("egs/separate_metrics/wvmos.yaml", "r", encoding="utf-8") as f:
        score_config = yaml.safe_load(f)

    registry = MetricRegistry()
    register_wvmos_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(score_config, use_gt=False, use_gpu=False)

    assert len(score_config) > 0, "no scoring function is provided"

    score_info = scorer.score_utterances(
        gen_files, metric_suite, gt_files=None, output_file=None, io="soundfile"
    )
    summary = compute_summary(score_info)
    print("Summary: {}".format(summary), flush=True)

    for key in TEST_INFO:
        if abs(TEST_INFO[key] - summary[key]) > 1e-4 and key != "plcmos":
            raise ValueError(
                "Value issue in the test case, might be some issue in scorer {}".format(
                    key
                )
            )
    print("check successful", flush=True)


if __name__ == "__main__":
    info_update()
