"""Entrypoint contracts exercised through the real scorer with tiny fake metrics."""

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from test.audio_utils import generate_fixed_wav
from versa import scorer_shared
from versa.completion import COMPLETION_FIELD
from versa.bin import scorer, scorer_chunk, scoring
from versa.definition import (
    BaseMetric,
    MetricCategory,
    MetricMetadata,
    MetricRegistry,
    MetricType,
)


class UtteranceMetric(BaseMetric):
    """Record reference/text routing and return a fixed utterance score."""

    def _setup(self):
        """Retain I/O and cache options to test configuration forwarding."""
        self.io = self.config.get("io")
        self.cache_dir = self.config.get("cache_dir")

    def get_metadata(self):
        """Expose a dependency-free utterance metric for real scorer orchestration."""
        return MetricMetadata(
            "test_utterance",
            MetricCategory.INDEPENDENT,
            MetricType.FLOAT,
            False,
            False,
            False,
            False,
            [],
            "test",
        )

    def compute(self, predictions, references=None, metadata=None):
        """Record reference presence and transcript metadata, returning a fixed score."""
        self.calls["utterance"].append((references is not None, metadata["text"]))
        return {"test_score": 0.5}


class CorpusMetric(UtteranceMetric):
    """Record corpus paths, configuration, and metadata without model inference."""

    def get_metadata(self):
        """Declare a distributional metric to exercise corpus dispatch."""
        return MetricMetadata(
            "test_corpus",
            MetricCategory.DISTRIBUTIONAL,
            MetricType.FLOAT,
            False,
            False,
            False,
            False,
            [],
            "test",
        )

    def compute(self, predictions, references=None, metadata=None):
        """Record the corpus invocation and return a fixed corpus score."""
        self.calls["corpus"].append((predictions, references, self.config, metadata))
        return {"corpus_score": 0.25}


@pytest.fixture
def scoring_case(tmp_path, monkeypatch):
    """Build tiny WAV inputs, fake metrics, CLI arguments, and cleanup call tracking.

    Patch environment variables through monkeypatch so cache changes are
    restored after each test; all generated inputs live under tmp_path."""
    for key in (
        "VERSA_CACHE_DIR",
        "VERSA_HF_CACHE_DIR",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "NEMO_CACHE_DIR",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    calls = {"utterance": [], "corpus": [], "closed": [], "released": []}

    UtteranceMetric.calls = calls
    registry = MetricRegistry()
    for cls in (UtteranceMetric, CorpusMetric):
        registry.register(cls, cls().get_metadata())
    real_scorer = scorer_shared.VersaScorer
    monkeypatch.setattr(scorer_shared, "VersaScorer", lambda: real_scorer(registry))
    monkeypatch.setattr(scorer_chunk, "VersaScorer", lambda: real_scorer(registry))
    original_close = scorer_shared.ScoreProcessor.close

    def close(processor):
        """Close the real processor and record whether its output handle was released."""
        original_close(processor)
        calls["closed"].append(
            processor.file_handle is None or processor.file_handle.closed
        )

    monkeypatch.setattr(scorer_shared.ScoreProcessor, "close", close)
    monkeypatch.setattr(
        scorer_shared,
        "_release_metric_resources",
        lambda: calls["released"].append(True),
    )
    for folder in ("pred", "gt"):
        (tmp_path / folder).mkdir()
        generate_fixed_wav(tmp_path / folder / "utt.wav")
    (tmp_path / "text").write_text("utt.wav hello world\n")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump([{"name": "test_utterance"}, {"name": "test_corpus"}])
    )
    argv = [
        "versa-score",
        "--pred",
        str(tmp_path / "pred"),
        "--gt",
        str(tmp_path / "gt"),
        "--text",
        str(tmp_path / "text"),
        "--io",
        "dir",
        "--score_config",
        str(config),
        "--output_file",
        str(tmp_path / "scores.jsonl"),
        "--cache_folder",
        str(tmp_path / "cache"),
    ]
    return tmp_path, argv, calls


@pytest.mark.parametrize("mode", ["ordinary", "metric", "chunk_cli", "chunks"])
@pytest.mark.parametrize("no_match", [False, True])
def test_entrypoint_outputs_inputs_and_resume(
    scoring_case, monkeypatch, mode, no_match
):
    """Verify scoring modes preserve routing, chunk keys, outputs, resume, and cleanup."""
    root, argv, calls = scoring_case
    entrypoint = scorer_chunk if mode in ("chunk_cli", "chunks") else scorer
    if mode == "metric":
        argv += ["--scoring_mode", "metric"]
    if mode == "chunks":
        argv += [
            "--enable_chunking",
            "--chunk_duration",
            "0.5",
            "--hop_duration",
            "0.5",
        ]
    if no_match:
        argv += ["--no_match"]
    monkeypatch.setattr(sys, "argv", argv)
    entrypoint.main()
    output = root / "scores.jsonl"
    before = output.read_text()
    rows = [json.loads(line) for line in before.splitlines()]
    keys = (
        ["utt.wav@0.000-0.500", "utt.wav@0.500-1.000"]
        if mode == "chunks"
        else ["utt.wav"]
    )
    scores = [
        {name: value for name, value in row.items() if name != COMPLETION_FIELD}
        for row in rows
    ]
    assert scores == [{"key": key, "test_score": 0.5} for key in keys]
    for row in rows:
        entry = row[COMPLETION_FIELD]["metrics"]["test_utterance"]
        assert entry["status"] == "success"
        assert entry["fields"] == ["test_score"]
    assert calls["utterance"] == [(not no_match, "hello world")] * len(keys)
    assert yaml.safe_load(Path(str(output) + ".corpus").read_text()) == {
        "corpus_score": 0.25
    }
    pred, gt, config, metadata = calls["corpus"][0]
    if mode == "chunks":
        assert pred == str(output) + ".chunks/pred"
        assert gt == (None if no_match else str(output) + ".chunks/gt")
    elif mode == "chunk_cli":
        assert pred == str(root / "pred")
        assert gt == (None if no_match else str(root / "gt"))
    else:
        assert pred == {"utt.wav": str(root / "pred/utt.wav")}
        assert gt == (None if no_match else {"utt.wav": str(root / "gt/utt.wav")})
    assert metadata["text_info"] == dict.fromkeys(keys, "hello world")
    if entrypoint is scorer_chunk:
        assert config["cache_dir"] == str(root / "cache/test_corpus")
        assert config["io"] == "dir"
    else:
        assert config["cache_dir"] == str(root / "cache/test_corpus")
        assert "io" not in config
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    entrypoint.main()
    assert output.read_text() == before
    # Completed metrics are never recomputed, including in metric mode.
    assert len(calls["utterance"]) == len(keys)
    assert calls["closed"] and all(calls["closed"])
    assert bool(calls["released"]) == (mode == "metric")


@pytest.mark.parametrize("entrypoint", [scorer, scorer_chunk])
def test_explicit_corpus_config_wins(scoring_case, monkeypatch, entrypoint):
    """Ensure explicit corpus I/O and cache settings override entrypoint defaults."""
    root, argv, calls = scoring_case
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            [{"name": "test_corpus", "io": "soundfile", "cache_dir": "explicit"}]
        )
    )
    monkeypatch.setattr(sys, "argv", argv)
    entrypoint.main()
    assert calls["utterance"] == []
    assert calls["corpus"][0][2]["cache_dir"] == "explicit"
    assert calls["corpus"][0][2]["io"] == "soundfile"


@pytest.mark.parametrize(
    "entrypoint,error", [(scorer, SystemExit), (scorer_chunk, SystemExit)]
)
def test_empty_config_contract(scoring_case, monkeypatch, entrypoint, error):
    """Reject an empty score configuration with CLI exit status two."""
    root, argv, _ = scoring_case
    (root / "config.yaml").write_text("[]\n")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(error) as exc:
        entrypoint.main()
    assert exc.value.code == 2


def test_cuda_validation(scoring_case, monkeypatch):
    """Reject GPU execution when CUDA is unavailable."""
    _, argv, _ = scoring_case
    monkeypatch.setattr(sys, "argv", argv + ["--use_gpu"])
    monkeypatch.setattr(scoring.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        scorer.main()


def test_report_from_shared_scoring(scoring_case, monkeypatch):
    """Verify the CLI writes a report containing scores from shared orchestration."""
    root, argv, _ = scoring_case
    report = root / "report.md"
    monkeypatch.setattr(sys, "argv", argv + ["--report", str(report)])
    scorer.main()
    assert "test_score" in report.read_text()


def test_worker_count_reaches_scorer(scoring_case, monkeypatch):
    """Verify the requested CPU worker count reaches utterance scoring."""
    _, argv, _ = scoring_case
    original = scoring.run_scoring
    workers = []

    def run(args, instance, *positional, **keywords):
        """Wrap shared orchestration to observe the worker argument without spawning workers."""
        score_utterances = instance.score_utterances

        def score(*positional, **keywords):
            """Record the requested worker count and score serially for this routing test."""
            workers.append(keywords.pop("num_workers"))
            return score_utterances(*positional, **keywords)

        monkeypatch.setattr(instance, "score_utterances", score)
        return original(args, instance, *positional, **keywords)

    monkeypatch.setattr(scoring, "run_scoring", run)
    monkeypatch.setattr(sys, "argv", argv + ["--num_workers", "2"])
    scorer.main()
    assert workers == [2]


@pytest.mark.parametrize("entrypoint", [scorer, scorer_chunk])
def test_invalid_config_rejected_before_audio_loading(
    scoring_case, monkeypatch, entrypoint
):
    """Reject unknown metrics before attempting to load missing audio files."""
    root, argv, calls = scoring_case
    (root / "config.yaml").write_text("- name: unknown_metric\n")
    (root / "pred/utt.wav").unlink()
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        entrypoint.main()
    assert exc.value.code == 2
    assert not calls["utterance"] and not calls["corpus"]


@pytest.mark.parametrize("entrypoint", [scorer, scorer_chunk])
def test_literal_none_reference_without_text(scoring_case, monkeypatch, entrypoint):
    """Interpret the literal None reference argument as absent in both entrypoints."""
    _, argv, calls = scoring_case
    argv[argv.index("--gt") + 1] = "None"
    text_index = argv.index("--text")
    del argv[text_index : text_index + 2]
    monkeypatch.setattr(sys, "argv", argv)
    entrypoint.main()
    assert calls["utterance"] == [(False, None)]
    assert calls["corpus"][0][1] is None


@pytest.mark.parametrize("reference_keys", [[], ["different.wav"]])
def test_chunking_rejects_missing_reference_keys(scoring_case, reference_keys):
    """Validate pairing before any chunk files are created, including equal counts."""
    root, argv, calls = scoring_case
    args = scorer_chunk.get_parser().parse_args(argv[1:] + ["--enable_chunking"])
    with pytest.raises(ValueError, match="Ground truth is missing.*utt.wav"):
        scorer_chunk._maybe_chunk_filelists(
            args,
            {"utt.wav": str(root / "pred/utt.wav")},
            dict.fromkeys(reference_keys, str(root / "gt/utt.wav")),
            None,
        )
    assert not (root / "scores.jsonl.chunks").exists()
    assert not calls["corpus"] and not calls["utterance"]


def test_chunked_corpus_uses_directories_for_scp_inputs(scoring_case, monkeypatch):
    """Route paired chunk directories to corpus metrics even when input uses SCP."""
    root, argv, calls = scoring_case
    for folder in ("pred", "gt"):
        scp = root / f"{folder}.scp"
        scp.write_text(f"utt.wav {root / folder / 'utt.wav'}\n")
        argv[argv.index(f"--{folder}") + 1] = str(scp)
    argv[argv.index("--io") + 1] = "soundfile"
    monkeypatch.setattr(sys, "argv", argv + ["--enable_chunking"])
    scorer_chunk.main()
    pred, gt, config, _ = calls["corpus"][0]
    assert config["io"] == "dir"
    assert Path(pred).is_dir() and Path(gt).is_dir()
    assert {p.name for p in Path(pred).glob("*.wav")} == {
        p.name for p in Path(gt).glob("*.wav")
    }
    assert all(paired for paired, _ in calls["utterance"])


def _failing_compute(self, predictions, references=None, metadata=None):
    """Raise as a runtime backend failure would during inference."""
    self.calls["utterance"].append(("failed", metadata["text"]))
    raise RuntimeError("backend exploded")


@pytest.mark.parametrize("entrypoint", [scorer, scorer_chunk])
def test_strict_run_fails_on_a_failing_metric_but_keeps_results(
    scoring_case, monkeypatch, entrypoint
):
    """Strict mode exits unsuccessfully while retaining the written results."""
    root, argv, calls = scoring_case
    monkeypatch.setattr(UtteranceMetric, "compute", _failing_compute)
    monkeypatch.setattr(sys, "argv", argv + ["--strict"])

    with pytest.raises(SystemExit) as exc:
        entrypoint.main()

    assert "Strict run did not complete" in str(exc.value)
    rows = [
        json.loads(line) for line in (root / "scores.jsonl").read_text().splitlines()
    ]
    entry = rows[0][COMPLETION_FIELD]["metrics"]["test_utterance"]
    assert entry["status"] == "failed"
    assert entry["error_category"] == "inference"
    assert "backend exploded" in entry["error"]


@pytest.mark.parametrize("entrypoint", [scorer, scorer_chunk])
def test_tolerant_run_reports_a_failing_metric_without_exiting(
    scoring_case, monkeypatch, entrypoint
):
    """The default tolerant mode records the failure and completes."""
    root, argv, calls = scoring_case
    monkeypatch.setattr(UtteranceMetric, "compute", _failing_compute)
    monkeypatch.setattr(sys, "argv", argv)

    entrypoint.main()

    rows = [
        json.loads(line) for line in (root / "scores.jsonl").read_text().splitlines()
    ]
    assert rows[0][COMPLETION_FIELD]["metrics"]["test_utterance"]["status"] == "failed"


@pytest.mark.parametrize("entrypoint", [scorer, scorer_chunk])
def test_strict_run_succeeds_when_every_metric_completes(
    scoring_case, monkeypatch, entrypoint
):
    """A complete strict run exits normally."""
    _, argv, _ = scoring_case
    monkeypatch.setattr(sys, "argv", argv + ["--strict"])

    entrypoint.main()


def test_strict_run_writes_the_requested_report_before_exiting(
    scoring_case, monkeypatch
):
    """Every requested artifact is produced even when a strict run fails."""
    root, argv, _ = scoring_case
    report = root / "report.md"
    monkeypatch.setattr(UtteranceMetric, "compute", _failing_compute)
    monkeypatch.setattr(sys, "argv", argv + ["--strict", "--report", str(report)])

    with pytest.raises(SystemExit):
        scorer.main()

    assert report.exists()


def test_chunking_failure_is_reported_as_a_skipped_input(scoring_case, monkeypatch):
    """An input dropped while chunking cannot pass as a complete strict run."""
    root, argv, _ = scoring_case
    chunk_argv = argv + [
        "--enable_chunking",
        "--chunk_duration",
        "0.5",
        "--hop_duration",
        "0.5",
    ]

    def failing_chunk(*args, **kwargs):
        """Fail the way an unreadable or malformed input would."""
        raise RuntimeError("cannot chunk")

    monkeypatch.setattr(scorer_chunk, "_chunk_pair_to_tmp", failing_chunk)
    monkeypatch.setattr(sys, "argv", chunk_argv)

    # Tolerant mode records the dropped input and finishes.
    scorer_chunk.main()

    monkeypatch.setattr(sys, "argv", chunk_argv + ["--strict"])
    with pytest.raises(SystemExit) as exc:
        scorer_chunk.main()

    assert "skipped_utterances" in str(exc.value)
