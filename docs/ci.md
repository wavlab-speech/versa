# Continuous Integration for VERSA

This document explains the CI/CD setup for the VERSA repository.

## CI Workflow

The CI workflow is defined in `.github/workflows/ci.yml` and consists of several jobs:

1. **Code Quality**: Checks code formatting with Black and linting with Flake8
2. **Installation Tests**: Tests installation across multiple Python and Linux versions
3. **Basic Tests**: Runs the general test suite
4. **Metric Tests**: Runs specific tests for individual metrics

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
# Run all tests
pytest

# Run specific test modules
pytest test/test_general.py
pytest test/test_metrics/test_stoi.py
pytest test/test_metrics/test_pesq.py
```

## Adding New Metric Tests

When implementing a new metric, follow these steps:

1. Add the metric implementation in the appropriate directory
2. Create a test file in `test/test_metrics/` following the existing pattern
3. Add the test to the CI workflow in `.github/workflows/ci.yml`

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

