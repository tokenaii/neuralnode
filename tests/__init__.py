"""Run all tests."""

import pytest
import sys

if __name__ == "__main__":
    # Run all tests with coverage
    sys.exit(pytest.main([
        "-v",
        "--cov=src/neuralnode",
        "--cov-report=term-missing",
        "--cov-report=html",
        "tests/"
    ]))
