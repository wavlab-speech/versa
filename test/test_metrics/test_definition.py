import importlib.util
from pathlib import Path

import pytest
import numpy as np

from versa.definition import (
    BaseMetric,
    MetricCategory,
    MetricFactory,
    MetricMetadata,
    MetricRegistry,
    MetricSuite,
    MetricType,
)
from versa.scorer_shared import VersaScorer, compute_summary
from versa.utils_shared import find_files


def _load_calculate_average_wer():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "survey" / "get_wer.py"
    )
    spec = importlib.util.spec_from_file_location("versa_test_get_wer", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.calculate_average_wer


calculate_average_wer = _load_calculate_average_wer()


class DummyMetric(BaseMetric):
    def _setup(self):
        pass

    def compute(self, predictions, references=None, metadata=None):
        return {"dummy": 1.0}

    def get_metadata(self):
        return DUMMY_METADATA


DUMMY_METADATA = MetricMetadata(
    name="dummy",
    category=MetricCategory.INDEPENDENT,
    metric_type=MetricType.FLOAT,
    requires_reference=False,
    requires_text=False,
    gpu_compatible=False,
    auto_install=False,
    dependencies=["definitely_missing_dependency_for_versa_test"],
    description="Dummy metric for registry tests.",
)


def test_metric_factory_create_suite_with_missing_dependency_and_default_config():
    registry = MetricRegistry()
    registry.register(DummyMetric, DUMMY_METADATA)

    suite = MetricFactory(registry).create_metric_suite(["dummy"])

    assert suite.compute_all(predictions=None) == {"dummy": {"dummy": 1.0}}


def test_metric_suite_compute_parallel_is_explicitly_unimplemented():
    suite = MetricSuite({})

    with pytest.raises(NotImplementedError, match="compute_parallel"):
        suite.compute_parallel(predictions=None)


def test_compute_summary_infers_numeric_scores_without_metric_registry():
    score_info = [
        {
            "key": "utt1",
            "pesq": 1.0,
            "match_details": {"ok": True},
            "ref_text": "hello",
            "espnet_wer_insert": 1,
        },
        {
            "key": "utt2",
            "pesq": 3.0,
            "match_details": {"ok": False},
            "ref_text": "world",
            "espnet_wer_insert": 2,
        },
    ]

    assert compute_summary(score_info) == {
        "espnet_wer_insert": 3,
        "pesq": 2.0,
    }


def test_default_registry_includes_asvspoof_and_emo_vad():
    scorer = VersaScorer()

    assert scorer.registry.get_metadata("asvspoof_score").name == "asvspoof"
    assert scorer.registry.get_metadata("emo_vad").name == "emo_vad"


def test_find_files_preserves_nested_duplicate_basenames(tmp_path):
    first = tmp_path / "speaker1" / "test.wav"
    second = tmp_path / "speaker2" / "test.wav"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"")
    second.write_bytes(b"")

    files = find_files(str(tmp_path))

    assert files == {
        "speaker1/test.wav": str(first),
        "speaker2/test.wav": str(second),
    }


def test_validate_audio_uses_metric_specific_minimum_length():
    scorer = VersaScorer(MetricRegistry())
    short_audio = np.arange(1600, dtype=np.float64)

    assert not scorer._validate_audio(
        short_audio,
        16000,
        "short",
        "generated",
        ["visqol"],
    )


def test_get_wer_uses_safe_parsing_and_espnet_fallback(tmp_path, capsys):
    input_file = tmp_path / "scores.txt"
    input_file.write_text(
        '{"whisper_wer_equal": 8, "whisper_wer_delete": 1, '
        '"whisper_wer_insert": 0, "whisper_wer_replace": 1}\n'
        "{'espnet_wer_equal': 7, 'espnet_wer_delete': 0, "
        "'espnet_wer_insert': 1, 'espnet_wer_replace': 2}\n",
        encoding="utf-8",
    )

    calculate_average_wer(str(input_file))

    assert capsys.readouterr().out.strip() == "wer: 0.2631578947368421"
