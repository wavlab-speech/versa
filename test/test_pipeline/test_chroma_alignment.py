import logging
import math
import os

import yaml

from versa.scorer_shared import VersaScorer, compute_summary
from versa.utils_shared import find_files
from versa.definition import MetricRegistry
from versa.utterance_metrics.chroma_alignment import register_chroma_alignment_metric

TEST_INFO = {
    "chroma_stft_cosine_dtw": 0.8895886718828439,
    "chroma_stft_euclidean_dtw": 45.091055545199,
    "chroma_cqt_cosine_dtw": 1.1888872845493323,
    "chroma_cqt_euclidean_dtw": 56.16051355647546,
    "chroma_cens_cosine_dtw": 0.6962623125421354,
    "chroma_cens_euclidean_dtw": 38.38994047138499,
    "chroma_stft_cosine_dtw_raw": 8.895886718828438,
    "chroma_stft_cosine_dtw_log": 20.508107511971517,
}


def info_update():

    # find files
    if os.path.isdir("test/test_samples/test2"):
        gen_files = find_files("test/test_samples/test2")

    # find reference file
    gt_files = None
    if os.path.isdir("test/test_samples/test1"):
        gt_files = find_files("test/test_samples/test1")

    logging.info("The number of utterances = %d" % len(gen_files))

    with open("egs/separate_metrics/chroma_alignment.yaml", "r", encoding="utf-8") as f:
        score_config = yaml.full_load(f)

    # Create registry and register Chroma Alignment metric
    registry = MetricRegistry()
    register_chroma_alignment_metric(registry)
    
    # Initialize VersaScorer with the populated registry
    scorer = VersaScorer(registry)
    
    # Load metrics using the new API
    metric_suite = scorer.load_metrics(
        score_config,
        use_gt=(True if gt_files is not None else False),
        use_gpu=False,
    )

    assert len(score_config) > 0, "no scoring function is provided"

    # Score utterances using the new API
    score_info = scorer.score_utterances(
        gen_files, metric_suite, gt_files, 
        output_file=None, io="soundfile"
    )
    
    summary = compute_summary(score_info)
    print("Summary: {}".format(summary), flush=True)

    for key in summary:
        if math.isinf(TEST_INFO[key]) and math.isinf(summary[key]):
            # for sir"
            continue
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