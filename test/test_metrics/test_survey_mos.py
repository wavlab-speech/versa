"""Survey MOS adapter contracts; no pretrained downloads."""

import numpy as np
import pytest
import torch

from versa.definition import MetricRegistry
from versa.utterance_metrics import dnsmos_c, whisqa


class WhisperModel(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.values = values

    def forward(self, waveform):
        assert not torch.is_grad_enabled()
        self.waveform = waveform
        return torch.tensor([self.values])


@pytest.mark.parametrize(
    "head,values", [("single", [0.6]), ("multi", [0.6, 0.2, 0.4, 0.8, 1.0])]
)
def test_whisqa_heads(head, values):
    model = WhisperModel(values)
    result = whisqa.whisqa_calculate(model, np.zeros(8000), 8000, head)
    assert result["whisqa_mos"] == pytest.approx(3)
    assert model.waveform.shape == (1, 16000)
    if head == "multi":
        assert result == pytest.approx(
            dict(
                zip(
                    [
                        "whisqa_mos",
                        "whisqa_noisiness",
                        "whisqa_coloration",
                        "whisqa_discontinuity",
                        "whisqa_loudness",
                    ],
                    [3, 1, 2, 4, 5],
                )
            )
        )


@pytest.mark.parametrize("length", [1, 8000, 159999, 160000, 160001, 320000])
@pytest.mark.parametrize("tuple_output", [False, True])
def test_dnsmos_features_match_upstream(length, tuple_output):
    import librosa

    samples = np.random.default_rng(0).normal(size=length).astype(np.float32)
    repeated = samples.copy()
    while len(repeated) < 160000:
        repeated = np.concatenate((repeated, repeated))
    expected = np.log10(
        np.clip(
            np.abs(
                librosa.stft(
                    repeated[:160000], n_fft=320, win_length=320, hop_length=160
                ).T
            ),
            1e-7,
            1e7,
        )
    )

    def model(features):
        assert not torch.is_grad_enabled()
        assert features.shape == (1, 1, 1001, 161)
        np.testing.assert_allclose(features[0, 0].numpy(), expected)
        score = torch.tensor([[3.5, 0.25]])
        return (torch.zeros(1), score) if tuple_output else score

    assert dnsmos_c.dnsmos_c_calculate(model, samples, 16000) == {
        "dnsmos_c_mean": 3.5,
        "dnsmos_c_variance": 0.25,
    }


@pytest.mark.parametrize("audio", [[], [[0.0]], [float("nan")], [float("inf")]])
@pytest.mark.parametrize(
    "backend", [whisqa.whisqa_calculate, dnsmos_c.dnsmos_c_calculate]
)
def test_invalid_audio(backend, audio):
    with pytest.raises(ValueError, match="mono waveform"):
        backend(None, audio, 16000)


@pytest.mark.parametrize("rate", [True, 0, -1, 16000.5])
@pytest.mark.parametrize(
    "backend", [whisqa.whisqa_calculate, dnsmos_c.dnsmos_c_calculate]
)
def test_invalid_rate(backend, rate):
    with pytest.raises(ValueError, match="sample rate"):
        backend(None, [0.0], rate)


def test_bad_output():
    with pytest.raises(ValueError, match="head shape"):
        whisqa.whisqa_calculate(WhisperModel([1, 2]), [0.0], 16000)
    with pytest.raises(ValueError, match="negative variance"):
        dnsmos_c.dnsmos_c_calculate(lambda _: torch.tensor([[3.0, -1.0]]), [0.0], 16000)


def test_setup_requirements():
    with pytest.raises(ValueError, match="head"):
        whisqa.whisqa_setup(None, "bad")
    with pytest.raises(ValueError, match="repo_path"):
        whisqa.whisqa_setup(None)
    with pytest.raises(ValueError, match="dataset"):
        dnsmos_c.dnsmos_c_setup(None, "bad")
    with pytest.raises(ValueError, match="checkpoint"):
        dnsmos_c.dnsmos_c_setup(None, "NISQA")


@pytest.mark.parametrize(
    "module,symbol,name",
    [
        (whisqa, "WhisqaMetric", "whisqa"),
        (dnsmos_c, "DnsmosCMetric", "dnsmos_c"),
    ],
)
def test_discovery(module, symbol, name):
    from versa.metric_registry import load_metric_symbol

    registry = MetricRegistry()
    getattr(module, f"register_{name}_metric")(registry)
    assert registry.get_metric(name) is getattr(module, symbol)
    assert load_metric_symbol(symbol) is getattr(module, symbol)
    assert registry.get_metadata(name).requires_reference is False


def test_dnsmos_setup(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    model = WhisperModel([0.6])

    def load(path, map_location):
        assert path == str(checkpoint)
        assert map_location == "cpu"
        return model

    monkeypatch.setattr(torch.jit, "load", load)
    metric = dnsmos_c.DnsmosCMetric({"dataset": "BVCC", "checkpoint": str(checkpoint)})
    assert metric.model is model
    assert not model.training


def test_whisqa_setup(tmp_path, monkeypatch):
    from types import SimpleNamespace

    (tmp_path / "checkpoints").mkdir()
    checkpoint = tmp_path / "checkpoints" / "single_head_model.pt"
    model = WhisperModel([0.6])
    torch.save(model.state_dict(), checkpoint)
    backend = SimpleNamespace(
        __file__=str(tmp_path / "models" / "whisper_ni_predictors.py"),
        whisperMetricPredictorEncoderLayersTransformerSmall=lambda: model,
    )
    monkeypatch.setattr(whisqa.importlib, "import_module", lambda _: backend)
    metric = whisqa.WhisqaMetric({"repo_path": str(tmp_path)})
    assert metric.model is model
    assert not model.training
    assert metric.compute([0.0])["whisqa_mos"] == pytest.approx(3)


def test_dnsmos_resampling(monkeypatch):
    import librosa

    original = librosa.resample
    calls = []

    def resample(samples, **kwargs):
        calls.append(kwargs)
        return original(samples, **kwargs)

    monkeypatch.setattr(librosa, "resample", resample)
    dnsmos_c.dnsmos_c_calculate(
        lambda _: torch.tensor([[3.0, 0.1]]), np.ones(8000), 8000
    )
    assert calls == [{"orig_sr": 8000, "target_sr": 16000}]
