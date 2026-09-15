"""Dependency-light metadata shared by runtime registration and discovery.

Keep model imports and initialization in the metric implementation modules.
"""

from versa.definition import MetricCategory, MetricMetadata, MetricType


def _qwen2_audio_metadata(name):
    """Describe a named Qwen2-Audio prompt metric without importing its model."""
    return MetricMetadata(
        name=name,
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.STRING,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["transformers", "librosa", "numpy"],
        description="Speech property extraction with Qwen2-Audio",
        paper_reference="https://arxiv.org/abs/2407.10759",
        implementation_source="https://github.com/QwenLM/Qwen2-Audio",
    )


def _qwen2_audio_aliases(metric_name):
    """Return the legacy Qwen aliases for an unprefixed prompt name."""
    return [
        f"qwen2_{metric_name}_metric",
        f"qwen_{metric_name}",
    ]


def _qwen_omni_metadata(name):
    """Describe a named Qwen2.5-Omni prompt metric without importing its model."""
    return MetricMetadata(
        name=name,
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.STRING,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["transformers", "librosa", "numpy", "torch"],
        description="Speech property extraction with Qwen2.5-Omni",
        paper_reference="https://arxiv.org/abs/2503.20215",
        implementation_source="https://github.com/QwenLM/Qwen2.5-Omni",
    )


def _qwen_omni_aliases(metric_name):
    """Return the legacy Omni alias for an unprefixed prompt name."""
    return [f"qwen_omni_{metric_name}_metric"]


def _squim_metadata(name, mode):
    """Describe SQUIM input requirements; only ``ref`` mode requires a reference."""
    requires_reference = mode == "ref"
    description = (
        "TorchAudio-SQUIM subjective MOS metric"
        if requires_reference
        else "TorchAudio-SQUIM reference-less PESQ, STOI, and SI-SDR metrics"
    )
    return MetricMetadata(
        name=name,
        category=(
            MetricCategory.DEPENDENT
            if requires_reference
            else MetricCategory.INDEPENDENT
        ),
        metric_type=MetricType.DICT,
        requires_reference=requires_reference,
        requires_text=False,
        gpu_compatible=False,
        auto_install=False,
        dependencies=["torch", "torchaudio"],
        description=description,
        paper_reference="https://arxiv.org/abs/2302.01147",
        implementation_source=(
            "https://pytorch.org/audio/main/tutorials/squim_tutorial.html"
        ),
    )


def _scoreq_metadata(name, mode):
    """Describe ScoreQ input requirements and dependencies for the selected mode."""
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


def _stoi_metadata(name, extended):
    """Return registry metadata describing stoi inputs and dependencies."""
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
