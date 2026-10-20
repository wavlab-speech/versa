"""Optional WhiSQA single- and multi-head pretrained quality prediction."""

import importlib
from pathlib import Path
import sys

from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType


def whisqa_setup(repo_path, head="single", use_gpu=False):
    """Load the author architecture and checkpoint from an explicit checkout."""
    if head not in ("single", "multi"):
        raise ValueError("WhiSQA head must be single or multi")
    if not repo_path:
        raise ValueError("WhiSQA requires repo_path; see docs/pretrained_mos.md")
    import torch

    root = Path(repo_path).expanduser().resolve()
    checkpoint = root / "checkpoints" / f"{head}_head_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing WhiSQA checkpoint: {checkpoint}")
    # Upstream uses absolute imports from its models package.
    sys.path.insert(0, str(root))
    try:
        backend = importlib.import_module("models.whisper_ni_predictors")
        if Path(backend.__file__).resolve().parent != root / "models":
            raise ImportError(
                "A different 'models' package shadows WhiSQA; use a fresh process"
            )
    finally:
        sys.path.remove(str(root))
    constructor = (
        backend.whisperMetricPredictorEncoderLayersTransformerSmall
        if head == "single"
        else backend.whisperMetricPredictorEncoderLayersTransformerSmalldim
    )
    device = "cuda" if use_gpu else "cpu"
    model = constructor()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def whisqa_calculate(model, pred_x, gen_sr, head="single"):
    """Apply upstream waveform processing and the CLI's times-five conversion."""
    import torch
    import torchaudio

    if head not in ("single", "multi"):
        raise ValueError("WhiSQA head must be single or multi")
    if isinstance(gen_sr, bool) or not isinstance(gen_sr, int) or gen_sr <= 0:
        raise ValueError("WhiSQA requires a positive integer sample rate")
    waveform = torch.as_tensor(pred_x, dtype=torch.float32)
    if (
        waveform.ndim != 1
        or waveform.numel() == 0
        or not torch.isfinite(waveform).all()
    ):
        raise ValueError("WhiSQA requires a nonempty finite mono waveform")
    if gen_sr != 16000:
        waveform = torchaudio.functional.resample(waveform, gen_sr, 16000)
    waveform = waveform.unsqueeze(0).to(next(model.parameters()).device)
    with torch.inference_mode():
        scores = model(waveform).reshape(-1) * 5
    names = (
        ["mos"]
        if head == "single"
        else ["mos", "noisiness", "coloration", "discontinuity", "loudness"]
    )
    if scores.numel() != len(names) or not torch.isfinite(scores).all():
        raise ValueError("WhiSQA returned invalid scores or an unexpected head shape")
    return {f"whisqa_{name}": value.item() for name, value in zip(names, scores)}


class WhisqaMetric(BaseMetric):
    def _setup(self):
        self.head = self.config.get("head", "single")
        self.model = whisqa_setup(
            self.config.get("repo_path"), self.head, self.config.get("use_gpu", False)
        )

    def compute(self, predictions, references=None, metadata=None):
        return whisqa_calculate(
            self.model,
            predictions,
            (metadata or {}).get("sample_rate", 16000),
            self.head,
        )

    def get_metadata(self):
        return _whisqa_metadata()


def _whisqa_metadata():
    return MetricMetadata(
        name="whisqa",
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["torch", "torchaudio", "transformers", "torchinfo"],
        description="WhiSQA MOS and optional noisiness/coloration/discontinuity/loudness",
        paper_reference="https://arxiv.org/abs/2508.02210",
        implementation_source="https://github.com/leto19/WhiSQA",
    )


def register_whisqa_metric(registry):
    registry.register(WhisqaMetric, _whisqa_metadata())
