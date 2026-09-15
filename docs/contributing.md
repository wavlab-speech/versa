## Contributor Guidelines

This guide describes the current path for adding or updating a VERSA metric.
All maintained metrics should use the object-oriented registry interface
described below.

### Step 1: Choose the Metric Location

Add the implementation to the directory that matches the metric behavior:

- `versa/corpus_metrics`: corpus- or distribution-level metrics such as FAD,
  KID, or WER-style metrics that operate on a collection or transcript set.
- `versa/utterance_metrics`: utterance-level metrics that score one prediction,
  optionally against a reference signal or text.
- `versa/sequence_metrics`: sequence-comparison metrics. This area is kept for
  compatibility and is expected to merge into utterance metrics over time.

Prefer the current object-oriented interface for new metrics. A metric module
should provide:

- a `BaseMetric` subclass from `versa.definition`
- `_setup(self)` for config defaults, dependency checks, and model setup
- `compute(self, predictions, references=None, metadata=None)` for inference
- `get_metadata(self)` returning `MetricMetadata`
- `register_<metric>_metric(registry)` to register the metric and aliases

Use migrated metrics such as `versa/utterance_metrics/stoi.py` and
`versa/utterance_metrics/speaking_rate.py` as references. Legacy helper
functions may remain when they are still useful for compatibility, but new
scoring should go through the metric class and registry.

### Step 2: Define Metadata and Registration

Every metric registration should include a `MetricMetadata` entry with:

- canonical metric name
- `MetricCategory`: `INDEPENDENT`, `DEPENDENT`, `NON_MATCH`, or
  `DISTRIBUTIONAL`
- `MetricType`: commonly `FLOAT` for one score or `DICT` for grouped outputs
- whether the metric requires reference audio or reference text
- whether it requires explicitly ordered multi-source inputs
- whether it is GPU compatible
- whether it is installed automatically by the base package
- dependency import names
- a short description
- paper reference and implementation source when known
- aliases for existing YAML names or common alternate names

Register the metric in `register_<metric>_metric(registry)`, then expose that
registration function and public classes in a `MetricModuleSpec` entry in
`versa/metric_registry.py`. Add an installation hint for optional dependencies.
`versa/__init__.py` exposes those symbols lazily. Discovery extracts metadata and
aliases from source without importing backends; keep metadata constructors and
alias lists statically readable (or extend the source parser for generated names).
The scorer resolves the exact canonical name or alias to its module and calls only
that module’s registration functions. Unknown names and unavailable dependencies
fail for the selected metric without probing unrelated backends.

Run `pytest -q test/test_metric_imports.py` when changing registration or discovery.
Cover canonical names, aliases, and generated prompt names; tests must not download
models. Explicit `create_populated_registry()` remains a bulk-import operation.

If the metric returns a dict, make the output keys clear and stable. Numeric
summary handling is inferred from numeric values in scorer output, while string
or dict values are preserved in per-utterance JSONL results and omitted from
aggregate summaries.

### Step 3: Handle Inputs and Dependencies

`compute(...)` receives:

- `predictions`: the predicted audio, file collection, or other metric input
- `references`: optional reference audio or baseline data
- `metadata`: optional context such as `sample_rate`, `text`, and shared cache
  values

Use `metadata.get("sample_rate", 16000)` when a waveform metric needs a sample
rate. Raise clear `ValueError` messages for missing required predictions,
references, or text.

Optional dependencies must not break `import versa`. Guard optional imports, keep
heavy model loading in `_setup`, and raise a clear `ImportError` when a required
optional package is missing. When an external package needs a custom interface or
strict dependency versions, prefer:

- using the original tool or API when it already fits VERSA's interface
- forking the tool only when needed
- adding a localized installer under `tools`
- documenting that installer in `docs/supported_metrics.md`

### Step 4: Add Docs and Examples

Document each new or changed function, including private helpers, constructors,
and test fixtures. A short behavioral summary is enough for a simple adapter;
avoid docstrings that only repeat the function name. For metric `compute` methods,
describe waveform dimensions and channel handling, sample-rate source and default,
required references/text, output keys and units/direction, and failure behavior.
For `_setup` and persistence helpers, describe downloads, cache changes, file
ownership, and resume behavior. Keep existing docstrings accurate when behavior
changes. See `MapssMetric.compute`, `StoiMetric.compute`, and
`LogWmseMetric.compute` for concrete examples.

Update `docs/supported_metrics.md` with the new metric. Mark the Auto-Install
column according to whether the base installation provides all required
dependencies.

Add a YAML example under `egs/separate_metrics` following existing examples.
Keep the YAML metric name aligned with the registered canonical name or alias.

### Step 5: Add Tests

Add focused tests for new metric behavior:

- `test/test_metrics/test_<metric_name>.py` for metric class behavior,
  registration, aliases, input validation, and optional dependency handling
- `test/test_pipeline/test_<metric_name>.py` when the scorer pipeline should be
  exercised with sample audio or YAML config
- `test/test_general.py` only when the metric is part of the default/core setup

Prefer tests that exercise the current registry and `VersaScorer` path. Do not
add tests solely for old internal helper functions unless those helpers remain a
supported public interface.

For optional real-model tests, use the `real_model` pytest marker and keep them
skipped unless the required environment variable or dependency is available.

### Step 6: Run Local Checks

Before opening a pull request, run focused tests for the touched metric and the
standard style checks:

```bash
python -m pytest test/test_metrics/test_<metric_name>.py -q
python -m pytest test/test_pipeline/test_<metric_name>.py -q
python -m black --check versa test scripts setup.py
python -m flake8 versa scripts test setup.py \
  --count --select=E9,F63,F7,F82 --show-source --statistics
```

The repository also uses pre-commit with Black. If you have development
dependencies installed, you can run:

```bash
pre-commit run --all-files
```

Run the documentation checks from the repository root (no model installation
required):

```bash
python -m pip install -r ci/requirements-docstrings.txt
interrogate --verbose --verbose --fail-under 80 versa
docstr-coverage --include-setter --include-deleter --fail-under 80 versa
python ci/check_function_docstrings.py --fail-under 80 versa
```

All three checks must pass at 80% or higher. Their output identifies missing
definitions. The two package checks include modules, classes, constructors,
private/nested functions, and bundled model code under `versa/`; the last check
counts only actual function definitions. CodeRabbit separately checks functions
touched by a PR, including files outside `versa/`. Package coverage does not
replace that review. See [the coverage audit](docstring_coverage.md) for scope,
baseline results, and checker differences.
