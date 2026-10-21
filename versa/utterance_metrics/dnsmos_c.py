"""Optional DNSMOS-C TorchScript Gaussian quality predictor."""

from pathlib import Path

from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType


DATASETS = ("NISQA", "BVCC", "Tencent")


def dnsmos_c_setup(checkpoint, dataset, use_gpu=False):
    """Load an explicitly identified dataset-specific TorchScript checkpoint."""
    if dataset not in DATASETS:
        raise ValueError(f"DNSMOS-C dataset must be one of {DATASETS}")
    if not checkpoint:
        raise ValueError("DNSMOS-C requires checkpoint; see docs/pretrained_mos.md")
    import torch

    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Missing DNSMOS-C checkpoint: {path}")
    device = "cuda" if use_gpu else "cpu"
    return torch.jit.load(str(path), map_location=device).eval()


def dnsmos_c_calculate(model, pred_x, gen_sr, device="cpu"):
    """Repeat/crop to ten seconds and reproduce upstream default log-STFT."""
    import librosa
    import numpy as np
    import torch

    if isinstance(gen_sr, bool) or not isinstance(gen_sr, int) or gen_sr <= 0:
        raise ValueError("DNSMOS-C requires a positive integer sample rate")
    samples = np.asarray(pred_x, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("DNSMOS-C requires a nonempty finite mono waveform")
    if gen_sr != 16000:
        samples = librosa.resample(samples, orig_sr=gen_sr, target_sr=16000)
    samples = np.tile(samples, (160000 + len(samples) - 1) // len(samples))[:160000]
    spec = librosa.stft(samples, n_fft=320, win_length=320, hop_length=160).T
    spec = np.log10(np.clip(np.abs(spec), 1e-7, 1e7))
    features = torch.as_tensor(spec, dtype=torch.float32, device=device)[None, None]
    with torch.inference_mode():
        output = model(features)
        prediction = output[-1] if isinstance(output, tuple) else output
    if prediction.shape != (1, 2) or not torch.isfinite(prediction).all():
        raise ValueError("DNSMOS-C returned invalid mean/variance or unexpected shape")
    mean, variance = prediction[0].tolist()
    if variance < 0:
        raise ValueError("DNSMOS-C returned negative variance")
    return {"dnsmos_c_mean": mean, "dnsmos_c_variance": variance}


class DnsmosCMetric(BaseMetric):
    def _setup(self):
        """Load the configured optional DNSMOS-C backend and select its device."""
        self.device = "cuda" if self.config.get("use_gpu", False) else "cpu"
        self.model = dnsmos_c_setup(
            self.config.get("checkpoint"),
            self.config.get("dataset"),
            self.config.get("use_gpu", False),
        )

    def compute(self, predictions, references=None, metadata=None):
        """Score mono audio using the metadata sample rate, defaulting to 16 kHz."""
        return dnsmos_c_calculate(
            self.model,
            predictions,
            (metadata or {}).get("sample_rate", 16000),
            self.device,
        )

    def get_metadata(self):
        """Return DNSMOS-C input requirements and backend provenance."""
        return _dnsmos_c_metadata()


def _dnsmos_c_metadata():
    """Describe DNSMOS-C for discovery without loading model assets."""
    return MetricMetadata(
        name="dnsmos_c",
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["torch", "librosa", "numpy"],
        description="DNSMOS-C dataset-specific MOS mean and score-squared variance",
        paper_reference="https://arxiv.org/abs/2606.26903",
        implementation_source="https://github.com/Hope-Liang/DNSMOS-C",
    )


def register_dnsmos_c_metric(registry):
    """Register the optional DNSMOS-C metric class and its metadata."""
    registry.register(DnsmosCMetric, _dnsmos_c_metadata())
