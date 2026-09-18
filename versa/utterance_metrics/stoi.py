#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Standard and extended short-time objective intelligibility scoring."""

import numpy as np

try:
    from pystoi import stoi
except ImportError:
    raise ImportError("Please install pystoi and retry: pip install stoi")

from versa.definition import BaseMetric
from versa.metric_metadata import _stoi_metadata


def stoi_metric(pred_x, gt_x, fs):
    """Return ``stoi`` for mono arrays at fs Hz, trimming both to the shorter length."""
    if pred_x.shape[0] != gt_x.shape[0]:
        min_length = min(pred_x.shape[0], gt_x.shape[0])
        pred_x = pred_x[:min_length]
        gt_x = gt_x[:min_length]
    score = stoi(gt_x, pred_x, fs, extended=False)
    return {"stoi": score}


def estoi_metric(pred_x, gt_x, fs):
    """Return ``estoi`` for mono arrays at fs Hz, trimming both to the shorter length."""
    if pred_x.shape[0] != gt_x.shape[0]:
        min_length = min(pred_x.shape[0], gt_x.shape[0])
        pred_x = pred_x[:min_length]
        gt_x = gt_x[:min_length]
    score = stoi(gt_x, pred_x, fs, extended=True)
    return {"estoi": score}


class StoiMetric(BaseMetric):
    """Short-Time Objective Intelligibility metric."""

    def _setup(self):
        """Select standard or extended STOI and its corresponding result key."""
        self.extended = self.config.get("extended", False)
        self.output_key = "estoi" if self.extended else "stoi"

    def compute(self, predictions, references=None, metadata=None):
        """Return ``stoi`` or ``estoi`` intelligibility for a pair of mono waveforms.

        Both arrays are required and truncated to their shared minimum length.
        ``metadata["sample_rate"]`` supplies Hz, defaulting to 16000. No channel
        mixing is performed. Larger scores indicate greater intelligibility;
        the backend score is returned without clipping. Missing audio raises
        ValueError; other input restrictions are enforced by pystoi."""
        if predictions is None:
            raise ValueError("Predicted signal must be provided")
        if references is None:
            raise ValueError("Reference signal must be provided")

        fs = metadata.get("sample_rate", 16000) if metadata else 16000
        pred_x = np.asarray(predictions)
        gt_x = np.asarray(references)

        if self.extended:
            return estoi_metric(pred_x, gt_x, fs)
        return stoi_metric(pred_x, gt_x, fs)

    def get_metadata(self):
        """Return input requirements and provenance for this metric configuration."""
        return _stoi_metadata(self.output_key, self.extended)


class EstoiMetric(StoiMetric):
    """Extended Short-Time Objective Intelligibility metric."""

    def _setup(self):
        """Default to extended STOI unless the configuration explicitly disables it."""
        self.extended = self.config.get("extended", True)
        self.output_key = "estoi" if self.extended else "stoi"


def register_stoi_metric(registry):
    """Register STOI and ESTOI metrics with the registry."""
    registry.register(
        StoiMetric,
        _stoi_metadata("stoi", extended=False),
        aliases=["STOI", "stoi_metric"],
    )
    registry.register(
        EstoiMetric,
        _stoi_metadata("estoi", extended=True),
        aliases=["ESTOI", "estoi_metric"],
    )


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    metric = StoiMetric()
    scores = metric.compute(a, b, metadata={"sample_rate": 16000})
    print(scores)
