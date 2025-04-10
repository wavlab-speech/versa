import logging
import math
import os

import yaml

from versa.scorer_shared import (
    find_files,
    list_scoring,
    load_score_modules,
    load_summary,
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

    score_modules = load_score_modules(
        score_config,
        use_gt=False,
        use_gpu=False,
    )

    assert len(score_config) > 0, "no scoring function is provided"

    score_info = list_scoring(
        gen_files, score_modules, gt_files=None, output_file=None, io="soundfile"
    )
    summary = load_summary(score_info)
    print("Summary: {}".format(load_summary(score_info)), flush=True)

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
