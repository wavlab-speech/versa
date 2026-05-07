#!/usr/bin/env python3

# Copyright 2024 Wen-Chin Huang
# MIT License (https://opensource.org/licenses/MIT)
# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import time
import torch
import urllib.error

from versa.audio_utils import resample_audio
from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType


def sheet_ssqa_setup(
    model_tag="default",
    model_path=None,
    model_config=None,
    cache_dir="versa_cache",
    use_gpu=False,
):
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    if model_path is not None and model_config is not None:
        raise NotImplementedError(
            "Pending implementation for customized setup (Jiatong)"
        )
    else:
        if model_tag == "default":
            model_tag = "unilight/sheet:v0.1.0"
        torch.hub.set_dir(cache_dir)
        for attempt in range(3):
            try:
                model = torch.hub.load(
                    model_tag, "default", trust_repo=True, force_reload=False
                )
                break
            except (urllib.error.HTTPError, urllib.error.URLError):
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))

    model.model.to(device)
    return model


def sheet_ssqa(model, pred_x, fs, use_gpu=False):
    # NOTE(jiatong): current model only work for 16000 Hz
    if fs != 16000:
        pred_x = resample_audio(pred_x, fs, 16000)
    pred_x = torch.tensor(pred_x).float()
    if use_gpu:
        pred_x = pred_x.to("cuda")
    return {"sheet_ssqa": model.predict(wav=pred_x)}


class SheetSsqaMetric(BaseMetric):
    """Sheet SSQA MOS prediction metric."""

    def _setup(self):
        self.model_tag = self.config.get("model_tag", "default")
        self.model_path = self.config.get("model_path")
        self.model_config = self.config.get("model_config")
        self.cache_dir = self.config.get("cache_dir", "versa_cache")
        self.use_gpu = self.config.get("use_gpu", False)
        self.model = sheet_ssqa_setup(
            model_tag=self.model_tag,
            model_path=self.model_path,
            model_config=self.model_config,
            cache_dir=self.cache_dir,
            use_gpu=self.use_gpu,
        )

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")

        fs = metadata.get("sample_rate", 16000) if metadata else 16000
        return sheet_ssqa(self.model, np.asarray(predictions), fs, use_gpu=self.use_gpu)

    def get_metadata(self):
        return _sheet_ssqa_metadata()


def _sheet_ssqa_metadata():
    return MetricMetadata(
        name="sheet_ssqa",
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["torch", "librosa", "numpy"],
        description="Sheet SSQA MOS prediction metric",
        paper_reference="https://arxiv.org/abs/2411.03715",
        implementation_source="https://github.com/unilight/sheet",
    )


def register_sheet_ssqa_metric(registry):
    """Register Sheet SSQA with the registry."""
    registry.register(
        SheetSsqaMetric,
        _sheet_ssqa_metadata(),
        aliases=["sheet", "sheet_ssqa_metric"],
    )


if __name__ == "__main__":
    a = np.random.random(16000)
    metric = SheetSsqaMetric()
    print("metrics: {}".format(metric.compute(a, metadata={"sample_rate": 16000})))
