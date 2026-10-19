"""Optional Microsoft Distill-MOS pretrained speech quality estimator."""

from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType


def distill_mos_setup(use_gpu=False):
    """Load bundled variant-7 weights without importing the backend at discovery."""
    try:
        from distillmos import ConvTransformerSQAModel
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Distill-MOS is not installed. Run tools/install_distill_mos.sh"
        ) from exc
    device = "cuda" if use_gpu else "cpu"
    return ConvTransformerSQAModel(segmenting_in_forward=True).to(device).eval()


def distill_mos_calculate(model, pred_x, gen_sr):
    """Resample mono audio and delegate segmentation and normalization upstream."""
    import torch
    import torchaudio

    if isinstance(gen_sr, bool) or not isinstance(gen_sr, int) or gen_sr <= 0:
        raise ValueError("Distill-MOS requires a positive integer sample rate")
    waveform = torch.as_tensor(pred_x, dtype=torch.float32)
    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError("Distill-MOS requires a nonempty mono waveform")
    if not torch.isfinite(waveform).all():
        raise ValueError("Distill-MOS requires finite audio samples")
    waveform = waveform.unsqueeze(0)
    if gen_sr != 16000:
        waveform = torchaudio.transforms.Resample(gen_sr, 16000)(waveform)
    waveform = waveform.to(next(model.parameters()).device)
    with torch.inference_mode():
        score = model(waveform).item()
    return {"distill_mos": score}


class DistillMosMetric(BaseMetric):
    """Reference-free MOS using the upstream default overlapping segments."""

    def _setup(self):
        self.model = distill_mos_setup(self.config.get("use_gpu", False))

    def compute(self, predictions, references=None, metadata=None):
        return distill_mos_calculate(
            self.model, predictions, (metadata or {}).get("sample_rate", 16000)
        )

    def get_metadata(self):
        return _distill_mos_metadata()


def _distill_mos_metadata():
    return MetricMetadata(
        name="distill_mos",
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["distillmos", "torch", "torchaudio"],
        description="Microsoft Distill-MOS variant 7 speech MOS (higher is better)",
        paper_reference="https://arxiv.org/abs/2502.05356",
        implementation_source="https://github.com/microsoft/Distill-MOS",
    )


def register_distill_mos_metric(registry):
    registry.register(DistillMosMetric, _distill_mos_metadata(), aliases=["distillmos"])
