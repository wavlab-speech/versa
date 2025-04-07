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


def info_update():

    # find files
    if os.path.isdir("test/test_samples/test2"):
        gen_files = find_files("test/test_samples/test2")

    # find reference file
    if os.path.isdir("test/test_samples/test1"):
        gt_files = find_files("test/test_samples/test1")

    logging.info("The number of utterances = %d" % len(gen_files))

    with open("egs/separate_metrics/lid.yaml", "r", encoding="utf-8") as f:
        score_config = yaml.full_load(f)

    score_modules = load_score_modules(
        score_config,
        use_gt=(True if gt_files is not None else False),
        use_gpu=False,
    )

    assert len(score_config) > 0, "no scoring function is provided"

    score_info = list_scoring(
        gen_files, score_modules, gt_files, output_file=None, io="soundfile"
    )
    print("Summary: {}".format((score_info), flush=True))

    if abs(score_info[0]["language"][0][1] - 0.8865218162536621) > 1e-4:
        raise ValueError(
            "Value issue in the test case, might be some issue in scorer lanugage"
        )

    print("check successful", flush=True)


if __name__ == "__main__":
    info_update()
