"""Exercise VERSA against the released ESPnet AQA API without model downloads."""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("VERSA_TEST_PYPI_ESPNET") != "1",
    reason="requires the released ESPnet AQA runtime",
)


@pytest.mark.parametrize("mode", ["noref", "audioref", "textref", "fullref"])
def test_released_aqa_through_universa(tmp_path, monkeypatch, mode):
    """Run real AQA frontend/predictor inference through every reference mode."""
    import yaml
    from espnet2.bin.aqa_inference import AqaInference
    from espnet2.tasks.aqa import AqaTask
    from versa.utterance_metrics import universa

    args = AqaTask.get_parser().parse_args([])
    args.metric2id = ["mos"]
    args.use_ref_audio = mode in {"audioref", "fullref"}
    args.use_ref_text = mode in {"textref", "fullref"}
    args.token_type = "char"
    args.token_list = ["<blank>", "<unk>", "a", "<sos/eos>"]
    args.frontend_conf = {"n_fft": 64, "hop_length": 32, "n_mels": 16}
    encoder = {
        "num_blocks": 1,
        "attention_heads": 2,
        "linear_units": 32,
        "input_layer": "linear",
    }
    args.universa_conf = {
        "embedding_size": 16,
        "audio_encoder_params": encoder,
        "text_encoder_params": encoder,
    }
    AqaTask.build_model(args)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(vars(args)))
    model = AqaInference(train_config=str(config))
    monkeypatch.setattr(universa, "get_universa_model", lambda *a, **kw: model)
    audio = np.random.default_rng(0).normal(size=1600).astype(np.float32)
    scores = universa.universa_metric(
        audio,
        ref_audio=audio if args.use_ref_audio else None,
        ref_text="aaa" if args.use_ref_text else None,
    )
    assert set(scores) == {"universa_mos"}
    assert np.isfinite(scores["universa_mos"])


def test_arecho_excludes_aqa_internal_outputs():
    """Sequential inference flags and encoded features must not become scores."""
    import torch
    from versa.utterance_metrics.arecho import arecho_metric

    class Model:
        """Supply the output contract of the released sequential AQA predictor."""

        model = torch.nn.Linear(1, 1)

        def __call__(self, *args, **kwargs):
            """Return a score together with upstream diagnostic fields."""
            return {
                "mos": [3.5],
                "encoded_feat": torch.zeros(1, 2, 3),
                "sequential_metrics": True,
                "use_tokenizer_metrics": True,
                "token_seq": [[1, 2]],
            }

    assert arecho_metric(Model(), np.zeros(1600, dtype=np.float32), 16000) == {
        "arecho_mos": 3.5
    }


@pytest.mark.real_model
@pytest.mark.skipif(
    os.environ.get("VERSA_RUN_REAL_MODEL_TESTS") != "1",
    reason="downloads published AQA checkpoints and the WavLM frontend",
)
@pytest.mark.parametrize("backend", ["universa", "arecho"])
def test_published_aqa_checkpoint(backend, tmp_path):
    """Smoke-test the published checkpoint through VERSA's score adapter."""
    cache_dir = os.environ.get("VERSA_AQA_CACHE_DIR", str(tmp_path / "models"))
    audio = np.random.default_rng(0).normal(0, 0.01, 16000).astype(np.float32)
    if backend == "universa":
        from versa.utterance_metrics.universa import universa_metric_noref

        scores = universa_metric_noref(audio, 16000, cache_dir=cache_dir)
    else:
        from versa.utterance_metrics.arecho import arecho_metric, arecho_model_setup

        model = arecho_model_setup(cache_dir=cache_dir)
        scores = arecho_metric(model, audio, 16000)
    assert scores
    assert all(key.startswith(backend + "_") for key in scores)
    numeric = [value for value in scores.values() if not isinstance(value, str)]
    assert numeric and all(np.isfinite(float(value)) for value in numeric)
    assert all(value for value in scores.values() if isinstance(value, str))
