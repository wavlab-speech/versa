# Metric Migration Guide

This guide summarizes the preferred process for migrating existing Versa metrics
to the new object-oriented metric interface.

## Migration Goal

Use `versa.definition.BaseMetric` as the source of truth for metric
implementations. Preserve user-facing behavior, but do not preserve legacy
internal helper APIs unless they are still needed by public callers.

Preserve:

- YAML metric names
- CLI/scorer behavior
- output score keys
- documented config defaults
- optional dependency behavior

Clean up:

- old function-style metric internals
- duplicated setup code
- eager optional dependency imports
- tests that only exercise legacy helper functions

## Required Metric Shape

Each migrated metric should provide:

- a `BaseMetric` subclass
- `_setup(self)` for config defaults, dependency checks, and model setup
- `compute(self, predictions, references=None, metadata=None)` for scoring
- `get_metadata(self)` returning `MetricMetadata`
- `register_<metric>_metric(registry)` as the registry integration point

`compute` should:

- validate required inputs
- read sample rate from `metadata.get("sample_rate", 16000)` when needed
- return the same output keys users already receive
- avoid changing user-visible numeric conventions unless the migration requires it

## Metadata Checklist

Every metric registration should define:

- canonical metric name
- `MetricCategory`: `INDEPENDENT`, `DEPENDENT`, `NON_MATCH`, or `DISTRIBUTIONAL`
- `MetricType`: usually `FLOAT` for one score or `DICT` for grouped scores
- `requires_reference`
- `requires_text`
- `gpu_compatible`
- `auto_install`
- dependency import names
- short description
- paper reference and implementation source when known
- useful aliases for existing YAML or common names

## Optional Dependencies

Optional dependencies must not break `import versa`.

Use guarded imports inside metric modules, and raise a clear `ImportError` from
`_setup` when a required optional package is missing. Register optional metrics
from `versa/__init__.py` through `_optional_metric_import(...)`.

## Tests

Prefer tests for the new public path:

- metric class behavior
- registry registration and aliases
- `VersaScorer` pipeline behavior with existing sample audio when lightweight
- missing optional dependency behavior
- unchanged user-facing output keys

Do not add tests solely to preserve old internal helper APIs unless those APIs
remain part of the public interface.

Base-install focused tests currently live in:

- `test/test_metrics/test_base_metrics.py`
- `test/test_pipeline/test_base_metrics_pipeline.py`

## Migration Candidates

The following modules still appear to use the old interface because they do not
define or import `BaseMetric`. This list is based on a repository scan and should
be updated as each metric is migrated.

### Utterance-Level Metrics

Good early candidates:
- `versa/utterance_metrics/visqol_score.py`

Model-backed or broader migrations:

- `versa/utterance_metrics/pseudo_mos.py`
- `versa/utterance_metrics/se_snr.py`
- `versa/utterance_metrics/speaker.py`
- `versa/utterance_metrics/singer.py`
- `versa/utterance_metrics/qwen2_audio.py`
- `versa/utterance_metrics/qwen_omni.py`
- `versa/utterance_metrics/universa.py`
- `versa/utterance_metrics/log_wmse.py`

### Sequence Metrics

- `versa/sequence_metrics/mcd_f0.py`
- `versa/sequence_metrics/warpq.py`

### Corpus and Distributional Metrics

- `versa/corpus_metrics/espnet_wer.py`
- `versa/corpus_metrics/owsm_wer.py`
- `versa/corpus_metrics/whisper_wer.py`
- `versa/corpus_metrics/fad.py`
- `versa/corpus_metrics/individual_fad.py`
- `versa/corpus_metrics/kid.py`
- `versa/corpus_metrics/clap_score.py`

### Already Migrated Examples

Use these as local references when migrating the remaining metrics:

- `versa/utterance_metrics/speaking_rate.py`
- `versa/utterance_metrics/scoreq.py`
- `versa/utterance_metrics/sheet_ssqa.py`
- `versa/utterance_metrics/stoi.py`
- `versa/utterance_metrics/pesq_score.py`
- `versa/utterance_metrics/squim.py`
- `versa/utterance_metrics/vad.py`
- `versa/utterance_metrics/vqscore.py`
- `versa/sequence_metrics/signal_metric.py`

## Verification

Run focused checks before broader validation:

```bash
/opt/homebrew/bin/mamba run -n versa-dev python -m pytest <focused tests> -q
/opt/homebrew/bin/mamba run -n versa-dev python -m black --check <touched files>
/opt/homebrew/bin/mamba run -n versa-dev python -m flake8 <touched files>
```
