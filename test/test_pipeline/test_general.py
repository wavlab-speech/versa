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
    "mcd": 5.045226506332897,
    "f0rmse": 20.281004489942777,
    "f0corr": -0.07540903652440145,
    "sdr": 4.873952979593643,
    "sir": float("inf"),
    "sar": 4.873952979593643,
    "si_snr": 1.0702790021896362,
    "ci_sdr": 4.873951435089111,
    "pesq": 1.5722705125808716,
    "stoi": 0.007625108859647406,
    "speech_bert": 0.9727544784545898,
    "speech_bleu": 0.6699938983346256,
    "speech_token_distance": 0.850506056080969,
    "utmos": 1.9074374437332153,
    "dns_overall": 1.4526055142443377,
    "dns_p808": 2.09430193901062,
    "plcmos": 3.16458740234375,
    "spk_similarity": 0.895357072353363,
    "singmos": 2.0403144359588623,
    "sheet_ssqa": 1.5056276321411133,
    "se_sdr": -10.220576129334987,
    "se_sar": -10.220576129334987,
    "se_si_snr": -16.837026596069336,
    "se_ci_sdr": -10.220579147338867,
    "torch_squim_mos": 3.948253870010376,
    "torch_squim_stoi": 0.6027805209159851,
    "torch_squim_pesq": 1.1683127880096436,
    "torch_squim_si_sdr": -11.109052658081055,
}


def info_update():

    # find files
    if os.path.isdir("test/test_samples/test2"):
        gen_files = find_files("test/test_samples/test2")

    # find reference file
    if os.path.isdir("test/test_samples/test1"):
        gt_files = find_files("test/test_samples/test1")

    logging.info("The number of utterances = %d" % len(gen_files))

    with open("egs/speech.yaml", "r", encoding="utf-8") as f:
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
