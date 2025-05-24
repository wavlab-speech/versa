import logging
import os

import yaml
import numpy as np

from versa.scorer_shared import (
    find_files,
    list_scoring,
    load_score_modules,
)

TEST_INFO = {
    "openbeats_embedding_extraction": np.array([-0.42187455, -0.6287595, 0.1792216]),
}


def info_update():

    # find files
    if os.path.isdir("test/test_samples/test2"):
        gen_files = find_files("test/test_samples/test2")

    logging.info("The number of utterances = %d" % len(gen_files))

    with open("egs/separate_metrics/openbeats.yaml", "r", encoding="utf-8") as f:
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
    assert score_info[0]["embedding"].shape[:-1] == (
        1,
        48,
    ), f'The frame size is off. Expected (1,48) but got {score_info[0]["embedding"].shape[:-1]}'
    summary = score_info[0]
    print("Summary: {}".format(summary), flush=True)

    summary_value = summary["embedding"][0, :3, 0]
    if np.any(
        np.abs(TEST_INFO["openbeats_embedding_extraction"] - summary_value) > 1e-3
    ):
        raise ValueError(
            "Value issue in the test case, might be some issue in scorer {}".format(
                "openbeats_embedding_extraction"
            )
        )
    print("check successful", flush=True)


if __name__ == "__main__":
    info_update()
