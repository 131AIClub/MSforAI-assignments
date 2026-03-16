import os
import sys

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--mode",
        action="store",
        default="startup",
        help="run tests against startup or solution implementation",
    )


def pytest_configure(config):
    mode = config.getoption("--mode")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_dir, mode)
    if not os.path.exists(target_dir):
        pytest.exit(f"Mode directory not found: {target_dir}")
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)


@pytest.fixture
def mode(request):
    return request.config.getoption("--mode")
