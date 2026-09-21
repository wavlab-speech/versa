# Continuous Integration for VERSA

This document explains the CI/CD setup for the VERSA repository.

## CI Workflow

The CI workflow is defined in `.github/workflows/ci.yml` and consists of several jobs:

1. **Code Quality**: Checks code formatting with Black and linting with Flake8
2. **Installation Tests**: Builds a wheel and tests a clean installation on Python
   3.8–3.12, including the oldest declared supported version
3. **Core Tests**: Runs the explicit base-dependency suite in
   `ci/pytest-core.ini`, including registry/configuration, import isolation,
   aggregation/reporting, mocked MAPSS, scorer entrypoints, and CPU workers
4. **Docstring Coverage**: Runs pinned Interrogate and docstr-coverage package
   checks plus an AST-based function-only check, each with an 80% minimum

The independent `real-model-smoke.yml` workflow runs WavLM inference only after
a manual dispatch with an explicit model commit SHA. It is not a required PR
check. Other model-backed tests still need their own dependencies and assets.

## Running Tests Locally

### Package Validation

The independent `.github/workflows/packaging.yml` workflow builds the sdist and
then a wheel from that sdist on pushes and pull requests. It checks both artifacts
with Twine and `ci/check_package_metadata.py`. The latter rejects direct URL
dependencies in every `Requires-Dist` field, including extras and inactive
environment markers, which `twine check` alone does not catch.

```bash
python -m pip install build twine packaging pytest
python -m pytest -q test/test_package_metadata.py
python -m build
python -m twine check --strict dist/*
python ci/check_package_metadata.py dist/*
```

Use a clean checkout/output directory when preparing a release so `dist/` contains
only the intended version. Publish under the declared distribution name
`versa-speech-audio-toolkit`; the Python import remains `versa`. After the checks
pass, a maintainer with PyPI access can upload those same artifacts using
`python -m twine upload dist/*`. Building and validating does not publish a release.

The discrete-speech fork remains in `tools/requirements-external.txt`, included
in the sdist, and is installed separately alongside the `external` extra.

**Current release blocker:** ESPnet remains a declared dependency in the existing
`external` extra, pinned to commit `00275004934c5c0aeed8e1765ab32fca4a693d34`
of the inference fork. Moving it into a manual installation step would break the
existing setup. Consequently, the direct-reference guard currently fails on the
ESPnet requirement and these artifacts must not be uploaded to PyPI.

Keep this packaging change in draft until the separate ESPnet work lands the
required Uni-VERSA and Arecho changes and a compatible release is available.
Then replace the Git requirement with the verified PyPI version constraint,
validate the affected backends, and rerun the packaging checks. The guard must
continue rejecting all direct references; pinning a Git commit does not make it
acceptable to PyPI.

Before pushing your changes, you can run the same checks locally:

### Code Quality

```bash
# Install development dependencies
pip install -e .[dev]

# Run Black formatting check
black --check versa test scripts *.py

# Apply Black formatting
black versa test scripts *.py

# Run Flake8 linting
flake8 versa test scripts *.py
```

### Running Tests

```bash
# Run dependency-light core tests
pytest -q test/test_metrics/test_definition.py test/test_docstring_check.py \
  test/test_slurm_launcher.py test/test_aggregate_results.py \
  test/test_result_summary.py test/test_reporting.py test/test_completion.py

# Run the resume and run-status contracts
pytest -q test/test_completion.py test/test_pipeline/test_resume_contract.py \
  test/test_pipeline/test_scorer_entrypoints.py \
  test/test_pipeline/test_local_workers.py \
  test/test_pipeline/test_mapss_pipeline.py

# Run specific test modules
pytest test/test_general.py
pytest test/test_metrics/test_stoi.py
pytest test/test_metrics/test_pesq.py
```

The core lane installs only the base package and test extra. Its explicit file
list avoids collecting the optional model suite. It uses real orchestration with
small fake metrics and mocked backends; passing these checks is not numerical
validation of MAPSS or a neural evaluator. The JUnit check rejects empty suites,
skips, failures, and errors. CI retains the report as an artifact.

The resume tests cover the configuration-aware completion contract:
`test/test_completion.py` checks the record and status vocabulary, and
`test/test_pipeline/test_resume_contract.py` exercises resume, strict runs, and
run-status counters through the real scorer. Add prompt-resource checks to
`ci/pytest-core.ini` when the bank exists.

### Installed-wheel checks

CI builds a wheel instead of testing an editable install. Each Python-version job
creates a fresh virtual environment, installs that wheel with its base
dependencies, then runs `ci/check_installed_wheel.py` using `python -I` from outside
the checkout. This verifies installed distribution/version identity, packaged
source discovery, aliases and generated Qwen names, absence of backend imports
in lightweight APIs, all three console scripts, and actual CSV/HTML output.
It also rejects test fixtures accidentally included through repository symlinks.

To reproduce, start from the repository root with a clean `build/` directory:

```bash
python -m pip wheel --no-deps --wheel-dir dist .
python -m venv /tmp/versa-wheel-check
/tmp/versa-wheel-check/bin/python -m pip install dist/*.whl
VERSA_CHECK_SCRIPT="$PWD/ci/check_installed_wheel.py"
cd /tmp
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/tmp/versa-wheel-check/bin/python -I "$VERSA_CHECK_SCRIPT"
```

Run subsequent repository commands from the checkout again. The matrix tests
base-package compatibility; optional model stacks may have narrower Python
requirements. This does not change the package's declared support policy.

### Optional pinned WavLM check

Manually run **Optional WavLM model smoke** in GitHub Actions and supply a full
40-character commit SHA from `microsoft/wavlm-base-sv`. Mutable branch names and
short SHAs are rejected. The workflow:

1. Installs the base/test dependencies and the selected Transformers backend.
2. Downloads the requested snapshot into an explicit runner cache and records
   its model ID, revision, local path, and dependency versions.
3. Passes that local directory through `VERSA_WAVLM_MODEL_PATH`, then runs the
   real speaker pipeline with Hugging Face network access disabled.
4. Requires an executed, successful test with no skips, and retains the model
   manifest, full dependency inventory, and JUnit result including the measured
   speaker similarity.

This check uses the small bundled WAV fixtures on CPU. It establishes model
loading and finite output in the expected range, not benchmark accuracy or human
agreement. It may download a large checkpoint during preparation; default CI
never triggers it. Local real-model tests retain their example-config behavior
unless `VERSA_WAVLM_MODEL_PATH` is explicitly set.

### Docstring Coverage

The independent `docstring-coverage` job installs only
`ci/requirements-docstrings.txt`; it does not import VERSA or download models.
Run the same checks locally:

```bash
python -m pip install -r ci/requirements-docstrings.txt
interrogate --verbose --verbose --fail-under 80 versa
docstr-coverage --include-setter --include-deleter --fail-under 80 versa
python ci/check_function_docstrings.py --fail-under 80 versa
```

All files under `versa/`, including bundled model helpers, remain in scope.
Private methods, constructors, nested and async functions count. There are no
coverage exclusions or inherited-docstring exemptions. The AST check prints
each missing function's path, line, and qualified name and compares the unrounded
percentage to the threshold. Its parser/counting contracts run in core tests.

CodeRabbit's PR check measures changed functions and can report a different
percentage. It remains separate from these package-wide gates and from pytest
execution coverage. See [the audit and checker survey](docstring_coverage.md).

### Real Model Cache Setup

Full model-backed checks require external checkpoint assets. Keep those assets
in a visible workspace cache so local runs and dedicated real-model CI jobs are
reproducible:

```bash
PYTHON=python tools/setup_huggingface_cache.sh
```

The script prepares `versa_cache/huggingface` for Hugging Face models and
`versa_cache/discrete_speech_metrics` for discrete-speech k-means assets. Then
run the real-model suite with explicit cache paths:

```bash
VERSA_RUN_REAL_MODEL_TESTS=1 \
VERSA_HF_CACHE_DIR="$PWD/versa_cache/huggingface" \
VERSA_DISCRETE_SPEECH_CACHE_DIR="$PWD/versa_cache/discrete_speech_metrics" \
python -m pytest --import-mode=importlib test
```

For offline machines that already have the Hugging Face models cached, seed the
workspace cache from the user cache:

```bash
SOURCE_HF_CACHE="$HOME/.cache/huggingface/hub" \
VERSA_HF_LOCAL_ONLY=1 \
PYTHON=python \
tools/setup_huggingface_cache.sh
```

## Adding New Metric Tests

When implementing a new metric, follow these steps:

1. Add the metric implementation in the appropriate directory
2. Create a test file in `test/test_metrics/` following the existing pattern
3. Keep dependency-light tests runnable in default CI
4. Mark optional model-loading tests with `real_model` and gate them on the
   required dependency or environment variable

Example test structure:

```python
class TestNewMetric:
    """Tests for the New Metric implementation"""

    @pytest.fixture
    def reference_signal(self):
        # Generate test signal
        pass

    @pytest.fixture
    def test_signal(self):
        # Generate test signal
        pass

    def test_metric_initialization(self):
        # Test initialization
        pass

    def test_metric_calculate(self):
        # Test calculation
        pass

    # Additional tests...
```

## CI Badges

You can add the following badge to your README.md to show the CI status:

```markdown
[![VERSA CI](https://github.com/wavlab-speech/versa/actions/workflows/ci.yml/badge.svg)](https://github.com/wavlab-speech/versa/actions/workflows/ci.yml)
```

## Development Workflow

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Run the tests locally
5. Push your changes and create a pull request
6. CI will automatically run on your pull request
7. Address any CI failures
8. Request a review

## Best Practices

- Always run tests locally before pushing
- Follow the Black code style
- Add appropriate tests for new features
- Keep test coverage high
