"""Regression coverage for JCAMP formats and spectrum processing contracts."""

import numpy as np
import pytest

from assemblytheorytools import tools_data


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        (
            """99 .9
##xfactor=2 cm-1
##XFACTOR=unknown
##yfactor=.5
##XYPOINTS=(XY..XY)
$$ 999 .9
1,.2;2,.4;999
##malformed header
3e0 .6
##TITLE=ends the data block
4 .8
##END=
##XYDATA=(XY..XY)
5 1
""",
            [[2, 0.9], [4, 0.8], [6, 0.7]],
        ),
        (
            """##XFACTOR=2
##FIRSTX=10
##LASTX=6
##NPOINTS=3 points
##DATA TABLE=(X++(Y..Y))
10 .1 .2
6 .3 .4
""",
            [[20, 0.9], [16, 0.8], [12, 0.7]],
        ),
        (
            """##DELTAX=0
##YFACTOR=0
##DATATABLE=(X++(Y..Y))
7 .1 .2
##XYDATA=(XY..XY)
9 .3
""",
            [[7, 1], [7, 1], [9, 1]],
        ),
        (
            """##POINTS=0
##XYPOINTS=(XY..XY)
1 .2
""",
            [],
        ),
    ],
)
def test_jcamp_formats_scaling_and_data_boundaries(tmp_path, contents, expected):
    path = tmp_path / "spectrum.jdx"
    path.write_text(contents)

    spectrum = tools_data.load_ir_jcamp_data(path)

    np.testing.assert_allclose(spectrum, np.asarray(expected).reshape(-1, 2))
    assert spectrum.dtype == np.dtype(float)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("##TITLE=No spectrum\n1 .2\n", "No XY data block found"),
        ("##XYDATA=(X++(Y..Y))\n1\n", "No XY data block found"),
        (
            "##FIRSTX=1\n##LASTX=2\n##NPOINTS=1\n##XYDATA=(X++(Y..Y))\n1 .2\n",
            "DELTAX is missing",
        ),
    ],
)
def test_jcamp_missing_data_and_step_errors(tmp_path, contents, message):
    path = tmp_path / "spectrum.jdx"
    path.write_text(contents)

    with pytest.raises(ValueError, match=message):
        tools_data.load_ir_jcamp_data(path)


def test_metadata_uses_first_spectrum_attachment():
    entries = [
        {
            "attacments": [
                {"filename": "first.peak.jdx", "identifier": "a/first.peak.jdx"},
                {"filename": "first.jdx", "identifier": "a/b/first.jdx"},
                {"filename": "second.jdx", "identifier": "a/second.jdx"},
            ]
        }
    ]

    assert tools_data._process_meta_data_name(entries) == "first.jdx"


@pytest.mark.parametrize("entry", [None, [], [{}], [{"attacments": None}]])
def test_metadata_missing_structure_returns_none(entry):
    assert tools_data._process_meta_data_name(entry) is None


def test_metadata_malformed_attachment_stops_search():
    entries = [
        {
            "attacments": [
                {"filename": "missing_identifier.jdx"},
                {"filename": "valid.jdx", "identifier": "a/valid.jdx"},
            ]
        }
    ]

    assert tools_data._process_meta_data_name(entries) is None


def test_peak_bounds_are_inclusive_and_applied_after_detection():
    spectrum = np.column_stack((np.arange(7), [0, 1, 0, 2, 0, 1, 0]))
    original = spectrum.copy()

    peaks = tools_data.find_peak_indices_in_range(
        spectrum, min_x=1, max_x=5, prominence=None, distance=None
    )
    np.testing.assert_array_equal(peaks, [1, 3, 5])
    assert (
        tools_data.find_n_peak_indices_in_range(
            spectrum, min_x=1, max_x=5, prominence=None, distance=None
        )
        == 3
    )
    assert tools_data.find_n_peak_indices_in_range(spectrum, min_x=1, max_x=1, distance=3) == 0
    np.testing.assert_array_equal(spectrum, original)


def test_sg_filter_preserves_polynomial_and_input():
    x = np.arange(9)
    spectrum = np.column_stack((x, x**3 - 2 * x**2 + x))
    original = spectrum.copy()

    smoothed = tools_data.apply_sg_filter(spectrum, window_length=5, polyorder=3)

    np.testing.assert_allclose(smoothed, spectrum, atol=1e-12)
    np.testing.assert_array_equal(spectrum, original)
    assert smoothed.dtype == np.dtype(float)
    assert not np.shares_memory(smoothed, spectrum)
