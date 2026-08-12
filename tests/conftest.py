"""Keep the test run headless.

The tests exercise the plotting helpers the same way the examples do, so
they call the display entry points too: ``plt.show()``, PIL's
``Image.show()`` and ASE's ``view()``. None of those affect an assertion,
but on a desktop they open matplotlib windows, spawn external image
viewers and launch ASE GUI subprocesses, and on a headless box they warn
on every call instead. This module stubs the display layer out once,
rather than editing the ~30 call sites that use it.

Set ``ATT_TEST_SHOW_PLOTS=1`` to restore the real viewers when you want to
eyeball a figure while debugging locally.
"""

import os

import matplotlib
import pytest

SHOW_PLOTS = os.environ.get("ATT_TEST_SHOW_PLOTS", "") not in ("", "0")

if not SHOW_PLOTS:
    # Select the non-interactive backend before pyplot is imported below.
    # Without it, plt.show() blocks waiting for a display that never
    # appears in a headless/CI environment, hanging the whole run instead
    # of failing fast.
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

if not SHOW_PLOTS:
    import ase.visualize
    from PIL import Image

    # Agg alone still makes plt.show() warn "FigureCanvasAgg is
    # non-interactive, and thus cannot be shown" once per *open* figure.
    # Because the tests never close their figures, that count climbs as
    # the run goes on and buries the real warnings. Drop the call.
    plt.show = lambda *args, **kwargs: None

    # PIL hands the image to an external viewer and leaves the process
    # running, which pytest reports as a leaked-subprocess ResourceWarning.
    Image.Image.show = lambda self, *args, **kwargs: None

    # ASE's view() launches a GUI subprocess per call. test_tools_cell.py
    # calls it once per CIF file, so an unstubbed run opens a window for
    # every structure in tests/data/cif_files/. Patch the function on the
    # package here, before the test modules do `from ase.visualize import
    # view` and bind their own reference to it.
    ase.visualize.view = lambda *args, **kwargs: None


@pytest.fixture(autouse=True)
def close_figures():
    """Close any figures a test leaves behind.

    pyplot keeps every unclosed figure alive for the whole session, which
    leaks memory across the suite and trips the "More than 20 figures have
    been opened" warning. The tests assert on the figure they were handed
    and never close it, so do it for them.
    """
    yield
    if not SHOW_PLOTS:
        plt.close("all")
