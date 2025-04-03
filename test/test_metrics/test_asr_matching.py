import pytest
import torch
from packaging.version import parse as V

import numpy as np
from versa.utterance_metrics.asr_matching import asr_match_setup, asr_match_metric


@pytest.mark.parametrize(
    "model_tag,beam_size,text_cleaner,cache_text",
    [
        ("tiny", 1, "whisper_basic", None),
        ("tiny", 2, "whisper_en", None),
        ("tiny", 1, "whisper_en", "already_text"),
    ],
)
def test_utterance_asr_matching(model_tag, beam_size, text_cleaner, cache_text):
    audio = np.random.random(16000)
    ground_truth = np.random.random(16000)
    wer_utils = asr_match_setup(model_tag, beam_size, text_cleaner, use_gpu=False)
    asr_match_metric(wer_utils, audio, ground_truth, cache_text, 16000)
