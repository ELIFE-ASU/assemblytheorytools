"""Shared test isolation, bundled data and opt-in environment checks."""

import os
import random
import shutil
from pathlib import Path

# Configure caches and the backend before importing pyplot or the package.
TEST_CACHE = Path(__file__).resolve().parents[1] / ".pytest_cache" / "runtime"
TEST_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TEST_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TEST_CACHE))
SHOW_PLOTS = os.environ.get("ATT_TEST_SHOW_PLOTS", "") not in ("", "0")

import matplotlib

if not SHOW_PLOTS:
    matplotlib.use("Agg")

import ase.visualize
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

OPT_IN_GROUPS = {
    "integration": "tests requiring live services, external data, or executables",
    "slow": "long-running local calculations",
}


def pytest_addoption(parser):
    for marker, description in OPT_IN_GROUPS.items():
        parser.addoption(
            f"--run-{marker}", action="store_true", help=f"run {description}"
        )


def pytest_collection_modifyitems(config, items):
    for marker in OPT_IN_GROUPS:
        if not config.getoption(f"--run-{marker}"):
            skip = pytest.mark.skip(reason=f"pass --run-{marker} to run {marker} tests")
            for item in items:
                if item.get_closest_marker(marker) is not None:
                    item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated_random_state():
    """Make stochastic tests reproducible without leaking seeds between tests."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    random.seed(0)
    np.random.seed(0)
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)


@pytest.fixture(autouse=True)
def headless_display(monkeypatch):
    """Suppress external viewers and release figures even when a test fails."""
    if not SHOW_PLOTS:
        monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
        monkeypatch.setattr(Image.Image, "show", lambda *args, **kwargs: None)
        monkeypatch.setattr(ase.visualize, "view", lambda *args, **kwargs: None)
    yield
    if not SHOW_PLOTS:
        plt.close("all")


@pytest.fixture(scope="session")
def data_dir():
    """Bundled test data, independent of the working directory."""
    return Path(__file__).resolve().parent / "data"


@pytest.fixture
def serial_data_mp(monkeypatch):
    """Keep data-transform checks local; process pools have their own tests."""
    from assemblytheorytools import tools_data

    monkeypatch.setattr(
        tools_data, "mp_calc", lambda function, values: list(map(function, values))
    )


@pytest.fixture
def orca_path():
    """Return a configured ORCA executable or skip the integration test."""
    configured = os.environ.get("ORCA_PATH") or "orca"
    executable = shutil.which(configured)
    if executable is None:
        pytest.skip("ORCA executable is not configured")
    return executable
