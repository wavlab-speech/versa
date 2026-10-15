"""Check the canonical ESPnet AQA loaders without downloading model weights."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from versa.utterance_metrics import arecho, universa


@pytest.fixture
def aqa_backend(monkeypatch):
    """Record AQA constructor and model-zoo calls for both metric adapters."""
    calls = []

    class Inference:
        """Stand in for the canonical AqaInference constructor."""

        def __init__(self, **kwargs):
            """Record the downloaded config, checkpoint, and device."""
            calls.append(("init", kwargs))

        @classmethod
        def from_pretrained(cls, tag):
            """Record an uncached Hugging Face model request."""
            calls.append(("pretrained", tag))
            return cls(model_tag=tag)

    class Downloader:
        """Record cache ownership while returning local artifact paths."""

        def __init__(self, cachedir):
            """Keep the requested model-zoo cache in the call record."""
            calls.append(("cache", cachedir))

        def download_and_unpack(self, tag):
            """Return the keyword arguments accepted by AqaInference."""
            calls.append(("download", tag))
            return {"train_config": "config.yaml", "model_file": "model.pth"}

    monkeypatch.setattr(universa, "AqaInference", Inference)
    monkeypatch.setattr(universa, "_universa_models", {})
    monkeypatch.setitem(
        sys.modules,
        "espnet2.bin.aqa_inference",
        SimpleNamespace(AqaInference=Inference),
    )
    monkeypatch.setitem(
        sys.modules,
        "espnet_model_zoo.downloader",
        SimpleNamespace(ModelDownloader=Downloader),
    )
    return Inference, calls


@pytest.mark.parametrize("mode", ["noref", "audioref", "textref", "fullref"])
@pytest.mark.parametrize("cache", [None, "local-cache"])
def test_universa_aqa_loader(aqa_backend, mode, cache):
    """All published reference modes use AQA and reuse their cached model."""
    inference, calls = aqa_backend
    model = universa.get_universa_model(mode, cache_dir=cache)
    assert isinstance(model, inference)
    assert universa.get_universa_model(mode, cache_dir=cache) is model
    tag = f"espnet/universa-wavlm_base_urgent24_multi-metric_{mode}"
    assert ("pretrained", tag) in calls if cache is None else ("download", tag) in calls
    assert sum(name == "init" for name, _ in calls) == 1


def test_arecho_aqa_loader(aqa_backend):
    """ARECHO imports AqaInference and preserves model-tag/cache/device options."""
    inference, calls = aqa_backend
    model = arecho.arecho_model_setup("base_v0", cache_dir="local-cache")
    assert isinstance(model, inference)
    assert ("download", "espnet/arecho_base_v0") in calls
    assert calls[-1] == (
        "init",
        {"train_config": "config.yaml", "model_file": "model.pth", "device": "cpu"},
    )


@pytest.mark.parametrize("mode", ["noref", "audioref", "textref", "fullref"])
def test_universa_scores_exclude_aqa_features(monkeypatch, mode):
    """Feature vectors and token sequences are diagnostics rather than scores."""
    prediction = {
        "mos": np.array([3.5]),
        "encoded_feat": torch.zeros(1, 8),
        "token_seq": [[2, 4, 6]],
        "use_tokenizer_metrics": True,
        "sequential_metrics": True,
    }
    monkeypatch.setattr(
        universa, "get_universa_model", lambda *a, **kw: lambda *a, **kw: prediction
    )
    audio = np.zeros(256, dtype=np.float32)
    extra = {}
    if mode in {"audioref", "fullref"}:
        extra["ref_audio_data"] = audio
    if mode in {"textref", "fullref"}:
        extra["ref_text"] = "hello"
    score = getattr(universa, f"universa_metric_{mode}")(audio, **extra)
    assert score == {"universa_mos": 3.5}


def test_arecho_scores_exclude_aqa_features():
    """Keep categorical scores while omitting feature and token diagnostics."""

    class Predictor:
        """Expose the model device and the shape of an AQA result."""

        model = torch.nn.Linear(1, 1)

        def __call__(self, *args, **kwargs):
            """Return metric values alongside internal AQA outputs."""
            return {
                "mos": [3.5],
                "language": ["eng"],
                "encoded_feat": torch.zeros(1, 8),
                "token_seq": [[2, 4, 6]],
                "use_tokenizer_metrics": True,
                "sequential_metrics": True,
            }

    assert arecho.arecho_metric(Predictor(), np.zeros(256), 16000) == {
        "arecho_mos": 3.5,
        "arecho_language": "eng",
    }
