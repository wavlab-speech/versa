#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import numpy as np

from versa.audio_utils import resample_audio
from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType

logger = logging.getLogger(__name__)

try:
    from WARPQ.WARPQmetric import warpqMetric
except ImportError:
    logger.info(
        "Please install WARP-Q from <versa_root>/tools/install_warpq.sh"
        "and retry after installation"
    )
    warpqMetric = None


def warpq_setup(
    fs=8000,
    n_mfcc=13,
    fmax=4000,
    patch_size=0.5,
    sigma=[[1, 0], [0, 3], [1, 3]],
    apply_vad=False,
):
    args = {
        "sr": fs,
        "n_mfcc": n_mfcc,
        "fmax": fmax,
        "patch_size": patch_size,
        "sigma": sigma,
        "apply_vad": apply_vad,
    }
    if warpqMetric is None:
        raise ImportError(
            "Please install WARP-Q from <versa_root>/tools/install_warpq.sh, "
            "and retry after installation"
        )
    model = warpqMetric(args)
    logger.info("Mapping model is not loaded for current implementation.")
    return model


def warpq(model, pred_x, gt_x, fs=8000):
    """
    Reference:
        W. A. Jassim, J. Skoglund, M. Chinen, and A. Hines,
        “Speech quality assessmentwith WARP‐Q: From similarity to subsequence
        dynamic time warp cost,” IET Signal Processing, 16(9), 1050–1070 (2022)

    """
    target_fs = model.args["sr"]
    if target_fs != fs:
        gt_x = resample_audio(gt_x, fs, target_fs)
        pred_x = resample_audio(pred_x, fs, target_fs)

    score = model.evaluate_versa(gt_x, pred_x)
    return {"warpq": score}


class WarpqMetric(BaseMetric):
    """WARP-Q dynamic time warping cost metric."""

    def _setup(self):
        self.fs = self.config.get("fs", 8000)
        self.n_mfcc = self.config.get("n_mfcc", 13)
        self.fmax = self.config.get("fmax", 4000)
        self.patch_size = self.config.get("patch_size", 0.5)
        self.sigma = self.config.get("sigma", [[1, 0], [0, 3], [1, 3]])
        self.apply_vad = self.config.get("apply_vad", False)
        self.model = warpq_setup(
            fs=self.fs,
            n_mfcc=self.n_mfcc,
            fmax=self.fmax,
            patch_size=self.patch_size,
            sigma=self.sigma,
            apply_vad=self.apply_vad,
        )

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")
        if references is None:
            raise ValueError("Reference signal must be provided")

        fs = metadata.get("sample_rate", 16000) if metadata else 16000
        return warpq(
            self.model,
            np.asarray(predictions),
            np.asarray(references),
            fs=fs,
        )

    def get_metadata(self):
        return _warpq_metadata()


def _warpq_metadata():
    return MetricMetadata(
        name="warpq",
        category=MetricCategory.DEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=False,
        auto_install=False,
        dependencies=["WARPQ", "librosa", "numpy"],
        description="WARP-Q dynamic time warping cost metric",
        paper_reference="https://arxiv.org/abs/2102.10449",
        implementation_source="https://github.com/wjassim/WARP-Q",
    )


def register_warpq_metric(registry):
    """Register WARP-Q with the registry."""
    registry.register(
        WarpqMetric,
        _warpq_metadata(),
        aliases=["warpq_metric", "warp_q"],
    )


if __name__ == "__main__":
    test_audio = np.zeros(16000)
    ref_audio = np.zeros(16000)
    metric = WarpqMetric()
    print(metric.compute(test_audio, ref_audio, metadata={"sample_rate": 8000}))
