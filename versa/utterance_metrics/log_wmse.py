#!/usr/bin/env python3
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Module for audio quality evaluation using LogWMSE metric."""

import logging

import librosa
import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import torchaudio
    import torchaudio.functional as F
    from torch_log_wmse import LogWMSE

    logger.info("Using the torch-log-wmse package for evaluation")
except ImportError:
    raise ImportError("Please install torch-log-wmse and retry")


def log_wmse(unproc_x, proc_x, gt_x, fs):
    """Calculate LogWMSE metric between audio samples.

    Args:
        unproc_x (torch.Tensor): Unprocessed audio (raw, noisy recording).
            Shape: [batch, audio_channels, sample]
        proc_x (torch.Tensor): Processed audio (denoised recording).
            Shape: [batch, audio_stems, audio_channels, sample]
        gt_x (torch.Tensor): Target audio (clean reference).
            Shape: [batch, audio_stems, audio_channels, sample]
        fs (int): Sampling rate.

    Returns:
        dict: Dictionary containing the LogWMSE score.
    """
    # Instantiate logWMSE
    # Set `return_as_loss=False` to return as a positive metric (Default: True)
    # Set `bypass_filter=True` to bypass frequency weighting (Default: False)
    inst_log_wmse = LogWMSE(
        audio_length=1.0,
        sample_rate=44100,
        return_as_loss=True,  # optional
    )

    log_wmse_score = inst_log_wmse(unproc_x, proc_x, gt_x)

    return {"torch log-wmse": log_wmse_score.detach().numpy()}


if __name__ == "__main__":
    """
    Reference:
    https://github.com/crlandsc/torch-log-wmse

    Unlike many audio quality metrics, logWMSE accepts a triple of audio inputs:
    - unprocessed audio (e.g. a raw, noisy recording) # [batch, audio_channels, sample]
    - processed audio (e.g. a denoised recording) # [batch, audio_stems, audio_channels, sample]
    - target audio (e.g. a clean reference without noise) # [batch, audio_stems, audio_channels, sample]

    *
    audio_length: length of the audio
    sample_rate: 44100 the metric performs an internal resampling to 44.1kHz for consistency
    audio_stems: # of audio stems (e.g. vocals, drums, bass, other)
    audio_channels: mono=1, stereo=2
    batch: batch size
    """
    batch = 1
    audio_channels = 1
    audio_stems = 1

    a = np.random.random(44100)
    a_length = a.shape[0]
    a = 2 * torch.rand(batch, audio_channels, a_length) - 1
    b = a.unsqueeze(1).expand(-1, audio_stems, -1, -1) * 0.1
    c = torch.zeros(batch, audio_stems, audio_channels, a_length)

    score = log_wmse(a, b, c, 44100)
    print(score)
