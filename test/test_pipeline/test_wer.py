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
    "espnet_wer_equal": 1,
    "owsm_wer_equal": 1,
    "whisper_wer_equal": 1,
    "faster_whisper_wer_equal": 1,
    "nemo_wer_equal": 1,
    "hubert_wer_equal": 1,
}


def info_update():

    # find files
    if os.path.isdir("test/test_samples/test_wer"):
        gen_files = find_files("test/test_samples/test_wer")

    logging.info("The number of utterances = %d" % len(gen_files))

    with open("egs/separate_metrics/wer.yaml", "r", encoding="utf-8") as f:
        score_config = yaml.full_load(f)

    score_modules = load_score_modules(
        score_config,
        use_gt=False,
        use_gt_text=True,
        use_gpu=False,
    )

    assert len(score_config) > 0, "no scoring function is provided"

    text_info = {}
    with open("test/test_samples/text_wer") as f:
        for line in f.readlines():
            key, value = line.strip().split(maxsplit=1)
            text_info[key] = value

    score_info = list_scoring(
        gen_files, score_modules, text_info=text_info, output_file=None, io="soundfile"
    )
    summary = load_summary(score_info)
    print("Summary: {}".format(load_summary(score_info)), flush=True)

    for key in TEST_INFO:
        if abs(TEST_INFO[key] - summary[key]) > 0 and not key == "espnet_wer_equal":
            raise ValueError(
                "Value issue in the test case, might be some issue in scorer {}".format(
                    key
                )
            )
    print("check successful", flush=True)


if __name__ == "__main__":
    info_update()
