#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np

try:
    from pystoi import stoi
except ImportError:
    raise ImportError("Please install pystoi and retry: pip install stoi")

from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType


def stoi_metric(pred_x, gt_x, fs):
    if pred_x.shape[0] != gt_x.shape[0]:
        min_length = min(pred_x.shape[0], gt_x.shape[0])
        pred_x = pred_x[:min_length]
        gt_x = gt_x[:min_length]
    score = stoi(gt_x, pred_x, fs, extended=False)
    return {"stoi": score}


def estoi_metric(pred_x, gt_x, fs):
    if pred_x.shape[0] != gt_x.shape[0]:
        min_length = min(pred_x.shape[0], gt_x.shape[0])
        pred_x = pred_x[:min_length]
        gt_x = gt_x[:min_length]
    score = stoi(gt_x, pred_x, fs, extended=True)
    return {"estoi": score}


class StoiMetric(BaseMetric):
    """Short-Time Objective Intelligibility metric."""

    def _setup(self):
        self.extended = self.config.get("extended", False)
        self.output_key = "estoi" if self.extended else "stoi"

    def compute(self, predictions, references=None, metadata=None):
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
        return _stoi_metadata(self.output_key, self.extended)


class EstoiMetric(StoiMetric):
    """Extended Short-Time Objective Intelligibility metric."""

    def _setup(self):
        self.extended = self.config.get("extended", True)
        self.output_key = "estoi" if self.extended else "stoi"


def _stoi_metadata(name, extended):
    label = "ESTOI" if extended else "STOI"
    description = (
        "Extended Short-Time Objective Intelligibility"
        if extended
        else "Short-Time Objective Intelligibility"
    )
    return MetricMetadata(
        name=name,
        category=MetricCategory.DEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=False,
        auto_install=False,
        dependencies=["pystoi", "numpy"],
        description=f"{label}: {description}",
        paper_reference="https://doi.org/10.1109/TASL.2010.2045551",
        implementation_source="https://github.com/mpariente/pystoi",
    )


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
