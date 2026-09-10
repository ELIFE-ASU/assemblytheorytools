import json
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import assemblytheorytools as att
from assemblytheorytools.tools_ms_json import _link_msn, _scan_to_df, _try_parse


def _scan(mz_intensity, scan, retention_time, **extra):
    """Build a raw scan with numeric mass keys and scan metadata."""
    scan_dict = dict(mz_intensity)
    scan_dict["scan"] = scan
    scan_dict["retention_time"] = retention_time
    scan_dict.update(extra)
    return scan_dict


def test_process_mzml_json_converts_scan_metadata_and_peak_values():
    data = {
        "ms1": {
            "spectrum_1": _scan({"100.5": 1000.0}, scan="1", retention_time="5.5"),
        }
    }

    result = att.process_mzml_json(data)

    assert set(result) == {1}
    assert_frame_equal(
        result[1],
        pd.DataFrame(
            {"intensity": [1000.0], "scan": [1], "retention_time": [5.5]},
            index=pd.MultiIndex.from_tuples([(1, 100.5)], names=["spectrum_id", "mz"]),
        ),
    )


@pytest.mark.parametrize(
    "metadata_position", [0, 1, 2], ids=["before", "between", "after"]
)
def test_process_mzml_json_ignores_metadata_without_misaligning_levels(
    metadata_position,
):
    # Regresses pairing filtered scan results with unfiltered top-level keys.
    levels = [
        (
            "ms1",
            {"spectrum_1": _scan({"100.5": 1000.0}, scan="1", retention_time="5.5")},
        ),
        (
            "ms2",
            {"spectrum_1": _scan({"200.25": 500.0}, scan="2", retention_time="6.5")},
        ),
    ]
    levels.insert(metadata_position, ("meta", {"description": "metadata"}))

    result = att.process_mzml_json(dict(levels))

    assert set(result) == {1, 2}
    assert result[1]["scan"].tolist() == [1]
    assert result[2]["scan"].tolist() == [2]
    assert result[1].index.tolist() == [(1, 100.5)]
    assert result[2].index.tolist() == [(1, 200.25)]


def test_process_mzml_json_empty_level_omitted():
    """Empty levels are omitted without dropping populated levels."""
    data = {
        "ms1": {},
        "ms2": {
            "spectrum_1": _scan({"200.25": 500.0}, scan="2", retention_time="6.5"),
        },
    }
    result = att.process_mzml_json(data)
    assert set(result.keys()) == {2}


def test_process_mzml_json_preserves_peak_order_and_optional_column_types():
    data = {
        "ms2": {
            "spectrum_12": _scan(
                {"101": 8, "100.0": 5, "100": 7, "-1": 9, ".2": 6},
                scan="2",
                retention_time="3",
                parent="201",
                parent_scan="1",
                hcd="not available",
            ),
            "spectrum_3": _scan({"80": 2.5}, scan="3", retention_time="4"),
        }
    }
    expected = pd.DataFrame(
        {
            "intensity": [8.0, 7.0, 2.5],
            "scan": [2, 2, 3],
            "retention_time": [3.0, 3.0, 4.0],
            "parent": [201.0, 201.0, float("nan")],
            "parent_scan": [1.0, 1.0, float("nan")],
            "hcd": [0.0, 0.0, float("nan")],
        },
        index=pd.MultiIndex.from_tuples(
            [(12, 101.0), (12, 100.0), (3, 80.0)],
            names=["spectrum_id", "mz"],
        ),
    )

    assert_frame_equal(att.process_mzml_json(data)[2], expected)


@pytest.mark.parametrize("path_type", [str, Path])
def test_process_mzml_json_reads_files(tmp_path, path_type):
    data = {"ms1": {"spectrum_1": _scan({"100": 2}, scan="1", retention_time="2")}}
    path = tmp_path / "spectra.json"
    path.write_text(json.dumps(data))

    assert_frame_equal(
        att.process_mzml_json(path_type(path))[1],
        att.process_mzml_json(data)[1],
    )


def test_scan_without_peaks_retains_metadata_schema():
    expected = pd.DataFrame(
        {
            "intensity": pd.Series(dtype=object),
            "scan": pd.Series(dtype="int64"),
            "retention_time": pd.Series(dtype="float64"),
            "hcd": pd.Series(dtype="float64"),
        },
    )

    assert_frame_equal(
        _scan_to_df(_scan({}, scan="1", retention_time="2", hcd="3")), expected
    )


def test_try_parse_falls_back_only_for_value_errors():
    parse = _try_parse(float, 0.0)

    assert parse("2.5") == 2.5
    assert parse("unknown") == 0.0
    with pytest.raises(TypeError):
        parse(None)


def test_link_msn_preserves_duplicate_matches_and_links_next_level_indices():
    ms1 = pd.DataFrame(
        {"scan": [1, 1, 2], "mz": [100.0, 100.0, 200.0]}, index=[7, 9, 13]
    )
    ms2 = pd.DataFrame(
        {
            "scan": [3, 4, 5],
            "mz": [10.0, 11.0, 12.0],
            "parent_scan": [1, 1, 99],
            "parent": [100.0, 101.0, 5.0],
        }
    )
    ms3 = pd.DataFrame({"scan": [6], "mz": [1.0], "parent_scan": [3], "parent": [10.0]})

    result = _link_msn({3: ms3, 1: ms1, 2: ms2})

    assert list(result) == [1, 2, 3]
    assert result[1] is ms1
    assert_frame_equal(
        result[2],
        pd.DataFrame(
            {
                "parent_id": [7, 9],
                "scan": [3, 3],
                "mz": [10.0, 10.0],
                "parent_scan": [1, 1],
                "parent": [100.0, 100.0],
            }
        ),
    )
    assert_frame_equal(
        result[3],
        pd.DataFrame(
            {
                "parent_id": [0, 1],
                "scan": [6, 6],
                "mz": [1.0, 1.0],
                "parent_scan": [3, 3],
                "parent": [10.0, 10.0],
            }
        ),
    )
