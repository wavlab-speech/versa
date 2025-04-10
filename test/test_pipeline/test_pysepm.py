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
    "pysepm_fwsegsnr": 12.010879979211092,
    "pysepm_llr": 0.10582867330588923,
    "pysepm_wss": 36.349490489992895,
    "pysepm_cd": 1.9559655879411542,
    "pysepm_Csig": 3.60503599984457,
    "pysepm_Cbak": 2.319464152555435,
    "pysepm_Covl": 2.551047048465036,
    "pysepm_csii_high": 0.03851460997893698,
    "pysepm_csii_mid": 0.05246553069271649,
    "pysepm_csii_low": 0,
    "pysepm_ncm": 0.1451686480269653,
}


def info_update():

    # find files
    if os.path.isdir("test/test_samples/test2"):
        gen_files = find_files("test/test_samples/test2")

    # find reference file
    if os.path.isdir("test/test_samples/test1"):
        gt_files = find_files("test/test_samples/test1")

    logging.info("The number of utterances = %d" % len(gen_files))

    with open("egs/separate_metrics/pysepm.yaml", "r", encoding="utf-8") as f:
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
    summary = load_summary(score_info)
    print("Summary: {}".format(load_summary(score_info)), flush=True)

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
