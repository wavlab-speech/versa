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

TEST_INFO = {"vad": 0.01386283989995718}


def info_update():

    # find files
    if os.path.isdir("test/test_samples/test2"):
        gen_files = find_files("test/test_samples/test2")

    logging.info("The number of utterances = %d" % len(gen_files))

    with open("egs/separate_metrics/vad.yaml", "r", encoding="utf-8") as f:
        score_config = yaml.full_load(f)

    score_modules = load_score_modules(
        score_config,
        use_gt=False,
        use_gpu=False,
    )

    assert len(score_config) > 0, "no scoring function is provided"

    score_info = list_scoring(
        gen_files, score_modules, output_file=None, io="soundfile"
    )
    print(score_info)
    if len(score_info[0]["vad_info"]) == 0:
        print("check successful", flush=True)


if __name__ == "__main__":
    info_update()
