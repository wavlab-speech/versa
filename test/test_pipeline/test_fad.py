import logging
import math
import os

import yaml

from versa.scorer_shared import (
    find_files,
    corpus_scoring,
    load_corpus_modules,
    load_summary,
)

TEST_INFO = {
    "fad_overall": 0.00753398077542222,
    "fad_r2": float("-inf"),
}


def info_update():

    with open("egs/separate_metrics/fad.yaml", "r", encoding="utf-8") as f:
        score_config = yaml.full_load(f)

    score_modules = load_corpus_modules(
        score_config,
        use_gpu=False,
        cache_folder="versa_cache",
        io="dir",
    )

    assert len(score_config) > 0, "no scoring function is provided"

    score_info = corpus_scoring(
        "test/test_samples/test2",
        score_modules,
        "test/test_samples/test1",
        output_file=None,
    )
    print("Summary: {}".format(score_info), flush=True)

    for key in score_info:
        if math.isinf(TEST_INFO[key]) and math.isinf(score_info[key]):
            # for sir"
            continue
        # the plc mos is undeterministic
        if abs(TEST_INFO[key] - score_info[key]) > 1e-4 and key != "plcmos":
            raise ValueError(
                "Value issue in the test case, might be some issue in scorer {}".format(
                    key
                )
            )
    print("check dir IO successful", flush=True)

    score_modules = load_corpus_modules(
        score_config,
        use_gpu=False,
        cache_folder="versa_cache",
        io="kaldi",
    )
    score_info = corpus_scoring(
        "test/test_samples/test2.scp",
        score_modules,
        "test/test_samples/test1.scp",
        output_file=None,
    )
    print("Summary: {}".format(score_info), flush=True)

    for key in score_info:
        if math.isinf(TEST_INFO[key]) and math.isinf(score_info[key]):
            # for sir"
            continue
        # the plc mos is undeterministic
        if abs(TEST_INFO[key] - score_info[key]) > 1e-4 and key != "plcmos":
            raise ValueError(
                "Value issue in the test case, might be some issue in scorer {}".format(
                    key
                )
            )
    print("check kaldi IO successful", flush=True)

    score_modules = load_corpus_modules(
        score_config,
        use_gpu=False,
        cache_folder="versa_cache",
        io="soundfile",
    )
    score_info = corpus_scoring(
        "test/test_samples/test2.scp",
        score_modules,
        "test/test_samples/test1.scp",
        output_file=None,
    )
    print("Summary: {}".format(score_info), flush=True)

    for key in score_info:
        if math.isinf(TEST_INFO[key]) and math.isinf(score_info[key]):
            # for sir"
            continue
        # the plc mos is undeterministic
        if abs(TEST_INFO[key] - score_info[key]) > 1e-4 and key != "plcmos":
            raise ValueError(
                "Value issue in the test case, might be some issue in scorer {}".format(
                    key
                )
            )
    print("check Soundfile IO successful", flush=True)


if __name__ == "__main__":
    info_update()
