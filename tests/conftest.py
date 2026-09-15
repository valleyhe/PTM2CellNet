import os
import sys

import pytest

os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def pytest_addoption(parser):
    """Accept pytest-timeout's CLI flag when the plugin is not installed."""
    try:
        parser.addoption("--timeout", action="store", default=None)
    except ValueError:
        pass


@pytest.fixture(scope="session", autouse=True)
def _seed_random():
    """Set fixed random seeds for reproducible tests."""
    import random
    import torch
    import numpy as np

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
