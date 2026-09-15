"""Discovery remains model-free and agrees with runtime registration."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from versa.definition import MetricRegistry
from versa.metric_discovery import (
    _discover_module_metadata,
    _RECOMMENDED_CONFIGS,
    recommend_config,
)
from versa import metric_metadata


@pytest.mark.parametrize("family", ["qwen2_audio", "qwen_omni", "squim", "scoreq"])
def test_runtime_registration_matches_source_discovery(family):
    """Compare real registration with AST discovery using model-free placeholder classes."""
    path = (
        Path(__file__).resolve().parents[2]
        / "versa/utterance_metrics"
        / (family + ".py")
    )
    tree = ast.parse(path.read_text())
    register = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == f"register_{family}_metric"
    )
    discovered = _discover_module_metadata(path)
    # Execute actual registration with placeholders, without the model stack.
    namespace = vars(metric_metadata).copy()
    for node in ast.walk(register):
        if isinstance(node, ast.Name) and node.id.endswith("Metric"):
            namespace[node.id] = type(node.id, (), {})
    if family.startswith("qwen"):
        prefix = family + "_"
        namespace[
            (
                "QWEN2_AUDIO_METRIC_CLASSES"
                if family == "qwen2_audio"
                else "QWEN_OMNI_METRIC_CLASSES"
            )
        ] = {
            metadata.name[len(prefix) :]: type("Placeholder", (), {})
            for metadata, _ in discovered
        }
    exec(
        compile(ast.Module(body=[register], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    registry = MetricRegistry()
    namespace[register.name](registry)
    assert registry.list_metrics() == sorted(
        metadata.name for metadata, _ in discovered
    )
    for metadata, aliases in discovered:
        assert registry.get_metadata(metadata.name) == metadata
        assert registry.get_aliases(metadata.name) == sorted(aliases)


def test_discovery_cli_does_not_import_model_stacks():
    """Run discovery in a fresh process that rejects heavy model imports."""
    code = """
import sys
sys.argv = ["versa-score", "--list-metrics"]
class BlockModels:
    def find_spec(self, fullname, path=None, target=None):
        blocked = {"torch", "torchaudio", "transformers", "librosa", "scoreq_versa"}
        if fullname.split(".")[0] in blocked:
            raise AssertionError("Model stack imported: " + fullname)
sys.meta_path.insert(0, BlockModels())
from versa.bin.scorer import main
main()
"""
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=True
    )
    for name in (
        "qwen2_audio_speaker_age",
        "qwen_omni_speaker_age",
        "squim_ref",
        "scoreq_ref",
    ):
        assert name in result.stdout


@pytest.mark.parametrize("task", sorted(_RECOMMENDED_CONFIGS))
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_recommendation_values_and_header(task, device):
    """Check YAML recommendation values and their task/device header against the source."""
    result = recommend_config(task, device)
    assert result.startswith(
        f"# Recommended VERSA score config for task={task}, device={device}\n"
    )
    assert yaml.safe_load(result) == _RECOMMENDED_CONFIGS[task][device]


def test_asr_match_dictionary_type_matches_runtime_and_discovery():
    """Execute ASR metadata/registration definitions without its inference imports."""
    from versa.definition import MetricType

    path = (
        Path(__file__).resolve().parents[2] / "versa/utterance_metrics/asr_matching.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    metric_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ASRMatchMetric"
    )
    method = next(
        node
        for node in metric_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_metadata"
    )
    register = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "register_asr_match_metric"
    )
    namespace = vars(metric_metadata).copy()
    namespace["ASRMatchMetric"] = type("ASRMatchMetric", (), {})
    exec(
        compile(
            ast.Module(body=[method, register], type_ignores=[]), str(path), "exec"
        ),
        namespace,
    )
    namespace["ASRMatchMetric"].get_metadata = namespace["get_metadata"]
    registry = MetricRegistry()
    namespace["register_asr_match_metric"](registry)
    assert registry.list_metrics(metric_type=MetricType.DICT) == ["asr_match"]
    assert registry.list_metrics(metric_type=MetricType.FLOAT) == []
    metric = namespace["ASRMatchMetric"]()
    assert metric.get_metadata() == registry.get_metadata("asr_match")
    discovered = dict((item.name, item) for item, _ in _discover_module_metadata(path))
    assert discovered["asr_match"].metric_type == MetricType.DICT
