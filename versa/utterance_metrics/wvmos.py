#!/usr/bin/env python3

#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

logger = logging.getLogger(__name__)

import librosa
import torch
from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType


def wvmos_setup(use_gpu=False):
    try:
        from wvmos import get_wvmos
    except ImportError as e:
        raise ModuleNotFoundError(
            "WVMOS is not installed. Please use `tools/install_wvmos.sh` to install"
        ) from e

    model = get_wvmos(cuda=use_gpu)

    return model


def wvmos_calculate(model, pred_x, gen_sr):
    """
    Reference:
    https://github.com/AndreevP/wvmos/tree/main

    """

    # If gen_sr is not 16000, resample the audio using librosa:
    # This check is also performed in model.processor
    if gen_sr != 16000:
        pred_x = librosa.resample(pred_x, orig_sr=gen_sr, target_sr=16000)

    x = model.processor(
        pred_x, return_tensors="pt", padding=True, sampling_rate=16000
    ).input_values

    with torch.no_grad():
        if model.cuda_flag:
            x = x.cuda()
        res = model.forward(x).mean()
    return {"wvmos": res.cpu().item()}


class WvmosMetric(BaseMetric):
    """WV-MOS metric using a fine-tuned wav2vec2 model."""

    def _setup(self):
        self.use_gpu = self.config.get("use_gpu", False)
        self.model = wvmos_setup(use_gpu=self.use_gpu)

    def compute(self, predictions, references=None, metadata=None):
        metadata = metadata or {}
        sample_rate = metadata.get("sample_rate", 16000)
        return wvmos_calculate(self.model, predictions, sample_rate)

    def get_metadata(self):
        return _wvmos_metadata()


def _wvmos_metadata():
    return MetricMetadata(
        name="wvmos",
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["librosa", "torch", "transformers"],
        description="WV-MOS score prediction using a fine-tuned wav2vec2 model",
        paper_reference="https://arxiv.org/abs/2203.13086",
        implementation_source="https://github.com/AndreevP/wvmos",
    )


def register_wvmos_metric(registry):
    """Register WV-MOS with the metric registry."""
    registry.register(
        WvmosMetric,
        _wvmos_metadata(),
        aliases=["Wvmos", "wvmos", "wv_mos"],
    )
