"""JCAMP parsing, IR peak processing, and Chemotion archive caching."""

import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest

from assemblytheorytools import tools_data
from assemblytheorytools.tools_plotting import plot_ir_spectrum

CHEMOTION_IR_TAR = Path("~/Downloads/10.22000-OGoEQGlsZGElrgst.tar").expanduser()


@pytest.fixture
def chemotion_archive(tmp_path, data_dir):
    """Build a miniature Chemotion IR archive around one real JCAMP-DX spectrum.

    Mirrors the published layout: a tar holding ``meta_data.json`` and a nested
    ``IR_data.tar.xz``. The metadata's non-``.peak.jdx`` identifier has to match the
    spectrum's filename, which is what the two halves are merged on.
    """
    staging = tmp_path / "staging"
    staging.mkdir()
    shutil.copy(data_dir / "ir_jcamp", staging / "SPEC1.jdx")

    inner = staging / "IR_data.tar.xz"
    with tarfile.open(inner, "w:xz") as tar:
        tar.add(staging / "SPEC1.jdx", arcname="SPEC1.jdx")

    meta = staging / "meta_data.json"
    meta.write_text(
        json.dumps(
            [
                {
                    "cano_smiles": "CCO",
                    "datasets": [
                        {
                            "attacments": [
                                {
                                    "filename": "SPEC1.peak.jdx",
                                    "identifier": "a/b/SPEC1.peak.jdx",
                                },
                                {
                                    "filename": "SPEC1.jdx",
                                    "identifier": "a/b/SPEC1.jdx",
                                },
                            ]
                        }
                    ],
                }
            ]
        )
    )

    archive = tmp_path / "10.22000-OGoEQGlsZGElrgst.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(meta, arcname="meta_data.json")
        tar.add(inner, arcname="IR_data.tar.xz")

    shutil.rmtree(staging)
    return archive


def _cache_path(archive):
    return archive.parent / "chemotion_ir_data" / "chemotion_ir_data.pkl.gz"


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


def test_load_ir_jcamp_data(data_dir):
    spectrum = tools_data.load_ir_jcamp_data(data_dir / "ir_jcamp")

    assert spectrum.ndim == 2
    assert spectrum.shape[0] > 0
    assert spectrum.shape[1] == 2


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
    assert (
        tools_data.find_n_peak_indices_in_range(spectrum, min_x=1, max_x=1, distance=3)
        == 0
    )
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


def test_find_peak_indices_in_range(data_dir):
    spectrum = tools_data.load_ir_jcamp_data(data_dir / "ir_jcamp")
    spectrum = tools_data.apply_sg_filter(spectrum, window_length=35, polyorder=3)

    peaks = tools_data.find_peak_indices_in_range(
        spectrum, min_x=400, max_x=1500, prominence=0.01, distance=5
    )
    assert len(peaks) == 14
    assert np.all((400 <= spectrum[peaks, 0]) & (spectrum[peaks, 0] <= 1500))


def test_calc_n_peaks_in_range(data_dir):
    spectrum = tools_data.load_ir_jcamp_data(data_dir / "ir_jcamp")

    assert (
        tools_data.find_n_peak_indices_in_range(spectrum, min_x=500, max_x=1500) == 19
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not CHEMOTION_IR_TAR.is_file(),
    reason="Chemotion IR dataset archive is not installed",
)
def test_process_chemotion_ir_archive(tmp_path):
    frame = tools_data.process_chemotion_ir_data(CHEMOTION_IR_TAR)

    assert {"smiles", "spectrum"}.issubset(frame.columns)
    assert not frame.empty

    spectrum = tools_data.apply_sg_filter(frame.iloc[0]["spectrum"])
    peaks = tools_data.find_peak_indices_in_range(spectrum)
    fig, _ = plot_ir_spectrum(spectrum, peaks=peaks)
    output = tmp_path / "ir-spectrum.png"
    fig.savefig(output)

    assert output.stat().st_size > 0


def test_process_chemotion_ir_data_builds_frame(chemotion_archive, serial_data_mp):
    frame = tools_data.process_chemotion_ir_data(chemotion_archive)

    assert list(frame.columns) == ["smiles", "name", "spectrum"]
    assert frame["smiles"].tolist() == ["CCO"]
    spectrum = frame["spectrum"].iloc[0]
    assert isinstance(spectrum, np.ndarray)
    assert spectrum.ndim == 2 and spectrum.shape[1] == 2


def test_process_chemotion_ir_data_does_not_cache_by_default(
    chemotion_archive, serial_data_mp, tmp_path, monkeypatch
):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    tools_data.process_chemotion_ir_data(chemotion_archive)

    assert not _cache_path(chemotion_archive).exists()
    assert list(cwd.iterdir()) == []


def test_chemotion_cache_follows_archive_and_round_trips_arrays(
    chemotion_archive, serial_data_mp, tmp_path, monkeypatch
):
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)

    first = tools_data.process_chemotion_ir_data(chemotion_archive, save=True)

    assert _cache_path(chemotion_archive).is_file()
    assert list(working_dir.iterdir()) == []
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(
        tools_data,
        "_process_chemotion_meta_section",
        lambda _: pytest.fail("A valid cache should bypass archive processing"),
    )

    second = tools_data.process_chemotion_ir_data(chemotion_archive)

    assert second["smiles"].tolist() == first["smiles"].tolist()
    assert second["name"].tolist() == first["name"].tolist()
    spectrum = second["spectrum"].iloc[0]
    assert isinstance(spectrum, np.ndarray)
    np.testing.assert_array_equal(spectrum, first["spectrum"].iloc[0])
    assert np.all(np.isfinite(spectrum.T[1]))
    assert list(elsewhere.iterdir()) == []


def test_process_chemotion_ir_data_ignores_unreadable_cache(
    chemotion_archive, serial_data_mp
):
    """A corrupt cache is reprocessed instead of raising."""
    tools_data.process_chemotion_ir_data(chemotion_archive, save=True)
    _cache_path(chemotion_archive).write_bytes(b"not a pickle")

    frame = tools_data.process_chemotion_ir_data(chemotion_archive)

    assert isinstance(frame["spectrum"].iloc[0], np.ndarray)
