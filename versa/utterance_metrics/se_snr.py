#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np

try:
    from espnet2.bin.enh_inference import SeparateSpeech
except ImportError:
    SeparateSpeech = None

from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType
from versa.sequence_metrics.signal_metric import signal_metric


def se_snr_setup(
    model_tag="default",
    model_path=None,
    model_config=None,
    use_gpu=False,
    cache_dir=None,
):
    if SeparateSpeech is None:
        raise ImportError("se_snr requires espnet. Please install espnet and retry")

    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    if model_path is not None and model_config is not None:
        model = SeparateSpeech.from_pretrained(
            model_file=model_path,
            train_config=model_config,
            normalize_output_wav=True,
            device=device,
        )
    else:
        if model_tag == "default":
            model_tag = "wyz/tfgridnet_for_urgent24"
        if cache_dir is None:
            model = SeparateSpeech.from_pretrained(
                model_tag=model_tag, normalize_output_wav=True, device=device
            )
        else:
            try:
                from espnet_model_zoo.downloader import ModelDownloader
            except ImportError:
                raise ImportError(
                    "se_snr requires espnet_model_zoo. Please install it and retry"
                )
            model_kwargs = ModelDownloader(cachedir=cache_dir).download_and_unpack(
                model_tag
            )
            model = SeparateSpeech(
                normalize_output_wav=True, device=device, **model_kwargs
            )
    return model


def se_snr(model, pred_x, fs):
    enhanced_x = model(pred_x[None, :], fs=fs)[0]
    signal_metrics = signal_metric(pred_x, enhanced_x)
    updated_metrics = {f"se_{key}": value for key, value in signal_metrics.items()}
    updated_metrics.pop("se_sir")
    return updated_metrics


class SeSnrMetric(BaseMetric):
    """Speech enhancement-based signal quality metrics."""

    def _setup(self):
        self.model_tag = self.config.get("model_tag", "default")
        self.model_path = self.config.get("model_path")
        self.model_config = self.config.get("model_config")
        self.use_gpu = self.config.get("use_gpu", False)
        self.cache_dir = self.config.get("cache_dir", "versa_cache/espnet_model_zoo")
        self.model = se_snr_setup(
            model_tag=self.model_tag,
            model_path=self.model_path,
            model_config=self.model_config,
            use_gpu=self.use_gpu,
            cache_dir=self.cache_dir,
        )

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")

        fs = metadata.get("sample_rate", 16000) if metadata else 16000
        return se_snr(self.model, np.asarray(predictions), fs)

    def get_metadata(self):
        return _se_snr_metadata()


def _se_snr_metadata():
    return MetricMetadata(
        name="se_snr",
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.DICT,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=[
            "espnet2",
            "ci_sdr",
            "fast_bss_eval",
            "mir_eval",
            "numpy",
            "torch",
        ],
        description="Speech enhancement-based SDR, SAR, SI-SNR, and CI-SDR metrics",
        implementation_source="https://github.com/espnet/espnet",
    )


def register_se_snr_metric(registry):
    """Register speech enhancement-based signal metrics with the registry."""
    registry.register(
        SeSnrMetric,
        _se_snr_metadata(),
        aliases=["se_snr_metric", "speech_enhancement_snr"],
    )


if __name__ == "__main__":
    a = np.random.random(16000)
    metric = SeSnrMetric()
    print("metrics: {}".format(metric.compute(a, metadata={"sample_rate": 16000})))
