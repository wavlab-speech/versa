"""Survey MOS adapter contracts; no pretrained downloads."""

import numpy as np
import pytest
import torch

from versa.definition import MetricRegistry
from versa.utterance_metrics import dnsmos_c, whisqa


class WhisperModel(torch.nn.Module):
    def __init__(self, values):
        """Store fixed predictions and a parameter for device selection."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.values = values

    def forward(self, waveform):
        """Record the waveform and emit fixed scores with gradients disabled."""
        assert not torch.is_grad_enabled()
        self.waveform = waveform
        return torch.tensor([self.values])


@pytest.mark.parametrize(
    "head,values", [("single", [0.6]), ("multi", [0.6, 0.2, 0.4, 0.8, 1.0])]
)
def test_whisqa_heads(head, values):
    """Check resampling, times-five scaling, and named head ordering."""
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
    """Compare repeat/crop and log-STFT features with the upstream recipe."""
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
        """Validate inference features before returning synthetic Gaussian outputs."""
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
    """Reject empty, multichannel, and nonfinite audio before inference."""
    with pytest.raises(ValueError, match="mono waveform"):
        backend(None, audio, 16000)


@pytest.mark.parametrize("rate", [True, 0, -1, 16000.5])
@pytest.mark.parametrize(
    "backend", [whisqa.whisqa_calculate, dnsmos_c.dnsmos_c_calculate]
)
def test_invalid_rate(backend, rate):
    """Reject invalid sample rates before preprocessing."""
    with pytest.raises(ValueError, match="sample rate"):
        backend(None, [0.0], rate)


def test_bad_output():
    """Reject inconsistent head dimensions and negative uncertainty."""
    with pytest.raises(ValueError, match="head shape"):
        whisqa.whisqa_calculate(WhisperModel([1, 2]), [0.0], 16000)
    with pytest.raises(ValueError, match="negative variance"):
        dnsmos_c.dnsmos_c_calculate(lambda _: torch.tensor([[3.0, -1.0]]), [0.0], 16000)


def test_setup_requirements():
    """Require explicit backend assets and supported variants."""
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
    """Resolve metric classes through lightweight registry discovery."""
    from versa.metric_registry import load_metric_symbol

    registry = MetricRegistry()
    getattr(module, f"register_{name}_metric")(registry)
    assert registry.get_metric(name) is getattr(module, symbol)
    assert load_metric_symbol(symbol) is getattr(module, symbol)
    assert registry.get_metadata(name).requires_reference is False


def test_dnsmos_setup(tmp_path, monkeypatch):
    """Load the configured TorchScript checkpoint on CPU in evaluation mode."""
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    model = WhisperModel([0.6])

    def load(path, map_location):
        """Check the checkpoint path and device passed to the TorchScript loader."""
        assert path == str(checkpoint)
        assert map_location == "cpu"
        return model

    monkeypatch.setattr(torch.jit, "load", load)
    metric = dnsmos_c.DnsmosCMetric({"dataset": "BVCC", "checkpoint": str(checkpoint)})
    assert metric.model is model
    assert not model.training


def test_whisqa_setup(tmp_path, monkeypatch):
    """Load local WhiSQA weights without interfering with dependency imports."""
    from types import SimpleNamespace

    (tmp_path / "checkpoints").mkdir()
    checkpoint = tmp_path / "checkpoints" / "single_head_model.pt"
    model = WhisperModel([0.6])
    torch.save(model.state_dict(), checkpoint)
    backend = SimpleNamespace(
        __file__=str(tmp_path / "models" / "whisper_ni_predictors.py"),
        whisperMetricPredictorEncoderLayersTransformerSmall=lambda: model,
    )
    original_import = whisqa.importlib.import_module

    def import_module(name, *args, **kwargs):
        """Stub the author backend while preserving unrelated lazy imports."""
        if name == "models.whisper_ni_predictors":
            return backend
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(whisqa.importlib, "import_module", import_module)
    metric = whisqa.WhisqaMetric({"repo_path": str(tmp_path)})
    assert metric.model is model
    assert not model.training
    assert metric.compute([0.0])["whisqa_mos"] == pytest.approx(3)


def test_numpy_integer_rate():
    """Accept NumPy integer sample rates such as those read via NumPy."""
    audio = np.zeros(8000, dtype=np.float32)
    whisqa.whisqa_calculate(WhisperModel([0.6]), audio, np.int64(8000))
    dnsmos_c.dnsmos_c_calculate(
        lambda _: torch.tensor([[3.0, 0.1]]), audio, np.int64(8000)
    )


def test_reporting_semantics():
    """Group new keys with audio quality and leave variance undirected."""
    from versa.reporting import metric_category, metric_direction

    for key in ["whisqa_mos", "whisqa_noisiness", "dnsmos_c_mean"]:
        assert metric_category(key) == "audio_quality"
        assert metric_direction(key) is True
    assert metric_category("dnsmos_c_variance") == "audio_quality"
    assert metric_direction("dnsmos_c_variance") is None


def test_dnsmos_resampling(monkeypatch):
    """Pass the input rate and 16-kHz target to the default resampler."""
    import librosa

    original = librosa.resample
    calls = []

    def resample(samples, **kwargs):
        """Record resampler arguments while preserving real preprocessing."""
        calls.append(kwargs)
        return original(samples, **kwargs)

    monkeypatch.setattr(librosa, "resample", resample)
    dnsmos_c.dnsmos_c_calculate(
        lambda _: torch.tensor([[3.0, 0.1]]), np.ones(8000), 8000
    )
    assert calls == [{"orig_sr": 8000, "target_sr": 16000}]


@pytest.mark.parametrize("length", [160000, 320000])
def test_dnsmos_long_audio_does_not_tile(monkeypatch, length):
    """Avoid allocating a duplicate recording when a ten-second crop suffices."""

    def reject_tile(*args, **kwargs):
        """Fail if preprocessing copies the recording through tiling."""
        raise AssertionError("Long inputs must be cropped without tiling")

    monkeypatch.setattr(np, "tile", reject_tile)
    result = dnsmos_c.dnsmos_c_calculate(
        lambda _: torch.tensor([[3.0, 0.25]]), np.ones(length), 16000
    )
    assert result == {"dnsmos_c_mean": 3.0, "dnsmos_c_variance": 0.25}
