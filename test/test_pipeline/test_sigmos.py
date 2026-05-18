import logging
import os

import yaml

from versa.definition import MetricRegistry
from versa.scorer_shared import VersaScorer, compute_summary
from versa.utils_shared import find_files
from versa.utterance_metrics.sigmos import register_sigmos_metric

TEST_INFO = {
    "SIGMOS_COL": 1.3242647647857666,
    "SIGMOS_DISC": 1.0382881164550781,
    "SIGMOS_LOUD": 1.0047355890274048,
    "SIGMOS_REVERB": 1.0245660543441772,
    "SIGMOS_SIG": 1.0186278820037842,
    "SIGMOS_OVRL": 1.0545676946640015,
}


def info_update():
    # find files
    if os.path.isdir("test/test_samples/test2"):
        gen_files = find_files("test/test_samples/test2")

    logging.info("The number of utterances = %d" % len(gen_files))

    with open("egs/separate_metrics/sigmos.yaml", "r", encoding="utf-8") as f:
        score_config = yaml.safe_load(f)

    registry = MetricRegistry()
    register_sigmos_metric(registry)
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
