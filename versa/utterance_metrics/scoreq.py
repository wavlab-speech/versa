#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import librosa
import numpy as np

from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType

logger = logging.getLogger(__name__)

try:
    from scoreq_versa import Scoreq
except ImportError:
    logger.info(
        "scoreq is not installed. Please use `tools/install_scoreq.sh` to install"
    )
    Scoreq = None


def scoreq_nr_setup(
    data_domain="synthetic",
    cache_dir="versa_cache/scoreq_pt-models",
    use_gpu=False,
):
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    if Scoreq is None:
        raise ModuleNotFoundError(
            "scoreq is not installed. Please use `tools/install_scoreq.sh` to install"
        )

    return Scoreq(
        data_domain=data_domain, mode="nr", cache_dir=cache_dir, device=device
    )


def scoreq_ref_setup(
    data_domain="synthetic",
    cache_dir="./scoreq_pt-models",
    use_gpu=False,
):
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    if Scoreq is None:
        raise ModuleNotFoundError(
            "scoreq is not installed. Please use `tools/install_scoreq.sh` to install"
        )

    return Scoreq(
        data_domain=data_domain, mode="ref", cache_dir=cache_dir, device=device
    )


def scoreq_nr(model, pred_x, fs):
    # NOTE(jiatong): current model only have 16k options
    if fs != 16000:
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)

    return {"scoreq_nr": model.predict(test_path=pred_x, ref_path=None)}


def scoreq_ref(model, pred_x, gt_x, fs):
    # NOTE(jiatong): current model only have 16k options
    if fs != 16000:
        gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=16000)
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)

    return {"scoreq_ref": model.predict(test_path=pred_x, ref_path=gt_x)}


class ScoreqMetric(BaseMetric):
    """ScoreQ speech quality metric."""

    def _setup(self):
        self.mode = self.config.get("mode", "nr")
        if self.mode not in {"nr", "ref"}:
            raise ValueError(f"Invalid ScoreQ mode: {self.mode}")

        self.data_domain = self.config.get("data_domain", "synthetic")
        self.cache_dir = self.config.get(
            "cache_dir", self.config.get("model_cache", "versa_cache/scoreq_pt-models")
        )
        self.use_gpu = self.config.get("use_gpu", False)

        if self.mode == "ref":
            self.model = scoreq_ref_setup(
                data_domain=self.data_domain,
                cache_dir=self.cache_dir,
                use_gpu=self.use_gpu,
            )
        else:
            self.model = scoreq_nr_setup(
                data_domain=self.data_domain,
                cache_dir=self.cache_dir,
                use_gpu=self.use_gpu,
            )

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")
        if self.mode == "ref" and references is None:
            raise ValueError("Reference signal must be provided for ScoreQ ref mode")

        fs = metadata.get("sample_rate", 16000) if metadata else 16000
        pred_x = np.asarray(predictions)
        if self.mode == "ref":
            return scoreq_ref(self.model, pred_x, np.asarray(references), fs)
        return scoreq_nr(self.model, pred_x, fs)

    def get_metadata(self):
        return _scoreq_metadata(f"scoreq_{self.mode}", self.mode)


class ScoreqNrMetric(ScoreqMetric):
    """Reference-less ScoreQ speech quality metric."""

    def _setup(self):
        self.config = {**self.config, "mode": self.config.get("mode", "nr")}
        super()._setup()


class ScoreqRefMetric(ScoreqMetric):
    """Reference-based ScoreQ speech quality metric."""

    def _setup(self):
        self.config = {**self.config, "mode": self.config.get("mode", "ref")}
        super()._setup()


def _scoreq_metadata(name, mode):
    requires_reference = mode == "ref"
    description = (
        "ScoreQ reference-based speech quality assessment"
        if requires_reference
        else "ScoreQ reference-less speech quality assessment"
    )
    return MetricMetadata(
        name=name,
        category=(
            MetricCategory.DEPENDENT
            if requires_reference
            else MetricCategory.INDEPENDENT
        ),
        metric_type=MetricType.FLOAT,
        requires_reference=requires_reference,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["scoreq_versa", "torch", "librosa", "numpy"],
        description=description,
        paper_reference="https://arxiv.org/pdf/2410.06675",
        implementation_source="https://github.com/ftshijt/scoreq",
    )


def register_scoreq_metric(registry):
    """Register ScoreQ reference-less and reference-based metrics."""
    registry.register(
        ScoreqNrMetric,
        _scoreq_metadata("scoreq_nr", "nr"),
        aliases=["scoreq", "scoreq_metric", "scoreq_no_ref"],
    )
    registry.register(
        ScoreqRefMetric,
        _scoreq_metadata("scoreq_ref", "ref"),
        aliases=["scoreq_reference"],
    )


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    metric_nr = ScoreqNrMetric({"use_gpu": True})
    metric_ref = ScoreqRefMetric({"use_gpu": True})
    print(metric_nr.compute(a, metadata={"sample_rate": 16000}))
    print(metric_ref.compute(a, b, metadata={"sample_rate": 16000}))
