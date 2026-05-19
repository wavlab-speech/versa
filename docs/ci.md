# Continuous Integration for VERSA

This document explains the CI/CD setup for the VERSA repository.

## CI Workflow

The CI workflow is defined in `.github/workflows/ci.yml` and consists of several jobs:

1. **Code Quality**: Checks code formatting with Black and linting with Flake8
2. **Installation Tests**: Tests the lean package install across Python versions
3. **Core Tests**: Runs dependency-light registry and scoring unit tests

Full metric tests are intentionally not part of the default CI path because many
metrics require large models, Git dependencies, or external toolkits. Run those
locally or in a dedicated real-model CI job with the required extras and
environment variables.

## Running Tests Locally

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
pytest test/test_metrics/test_definition.py

# Run specific test modules
pytest test/test_general.py
pytest test/test_metrics/test_stoi.py
pytest test/test_metrics/test_pesq.py
```

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
