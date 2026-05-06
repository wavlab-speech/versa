#!/usr/bin/env python3
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Module for audio quality evaluation using LogWMSE metric."""

import logging

import numpy as np
import torch

from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType

logger = logging.getLogger(__name__)

try:
    from torch_log_wmse import LogWMSE

    logger.info("Using the torch-log-wmse package for evaluation")
except ImportError:
    LogWMSE = None


def _ensure_log_wmse_available():
    if LogWMSE is None:
        raise ImportError("Please install torch-log-wmse and retry")


def _as_unprocessed_tensor(audio):
    if isinstance(audio, torch.Tensor):
        tensor = audio.float()
    else:
        tensor = torch.from_numpy(np.asarray(audio)).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def _as_stem_tensor(audio):
    tensor = _as_unprocessed_tensor(audio)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    return tensor


def log_wmse(unproc_x, proc_x, gt_x, fs, model=None):
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
    _ensure_log_wmse_available()

    if model is None:
        # Set `return_as_loss=False` to return as a positive metric.
        # Set `bypass_filter=True` to bypass frequency weighting.
        model = LogWMSE(
            audio_length=1.0,
            sample_rate=44100,
            return_as_loss=True,
        )

    log_wmse_score = model(unproc_x, proc_x, gt_x)
    score = log_wmse_score.detach().cpu().numpy()
    if np.size(score) == 1:
        score = float(np.asarray(score).reshape(-1)[0])
    return {"log_wmse": score}


class LogWmseMetric(BaseMetric):
    """Log-weighted mean square error."""

    def _setup(self):
        _ensure_log_wmse_available()
        self.audio_length = self.config.get("audio_length", 1.0)
        self.sample_rate = self.config.get("sample_rate", 44100)
        self.return_as_loss = self.config.get("return_as_loss", True)
        self.bypass_filter = self.config.get("bypass_filter")

        kwargs = {
            "audio_length": self.audio_length,
            "sample_rate": self.sample_rate,
            "return_as_loss": self.return_as_loss,
        }
        if self.bypass_filter is not None:
            kwargs["bypass_filter"] = self.bypass_filter
        self.model = LogWMSE(**kwargs)

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")
        if references is None:
            raise ValueError("Reference signal must be provided")

        metadata = metadata or {}
        unprocessed = metadata.get("unprocessed")
        if unprocessed is None:
            unprocessed = metadata.get("unproc_x", predictions)

        unproc_x = _as_unprocessed_tensor(unprocessed)
        proc_x = _as_stem_tensor(predictions)
        gt_x = _as_stem_tensor(references)
        fs = metadata.get("sample_rate", 16000)
        return log_wmse(unproc_x, proc_x, gt_x, fs, model=self.model)

    def get_metadata(self):
        return _log_wmse_metadata()


def _log_wmse_metadata():
    return MetricMetadata(
        name="log_wmse",
        category=MetricCategory.DEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=False,
        auto_install=False,
        dependencies=["torch_log_wmse", "torch", "numpy"],
        description="Log-weighted mean square error",
        implementation_source="https://github.com/nomonosound/log-wmse-audio-quality",
    )


def register_log_wmse_metric(registry):
    """Register log-weighted mean square error with the registry."""
    registry.register(
        LogWmseMetric,
        _log_wmse_metadata(),
        aliases=["log-wmse", "torch_log_wmse"],
    )


if __name__ == "__main__":
    """
    Reference:
    https://github.com/crlandsc/torch-log-wmse

    Unlike many audio quality metrics, logWMSE accepts a triple of audio inputs:
    - unprocessed audio (raw/noisy recording), shape [batch, channels, sample]
    - processed audio (denoised recording), shape [batch, stems, channels, sample]
    - target clean reference, shape [batch, stems, channels, sample]

    *
    audio_length: length of the audio
    sample_rate: 44100 for the package's internal resampling
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
