import assemblytheorytools as att


def _scan(mz_intensity, scan, retention_time, **extra):
    """
    Build a raw scan dict in the shape expected by process_mzml_json.

    Parameters
    ----------
    mz_intensity : dict
        Mapping of m/z (as a numeric-looking string key) to intensity.
    scan : str
        Scan number.
    retention_time : str
        Retention time.
    **extra
        Additional optional scan fields (e.g. parent, parent_scan, hcd).

    Returns
    -------
    dict
        A scan dict with mass entries plus the required metadata fields.
    """
    scan_dict = dict(mz_intensity)
    scan_dict["scan"] = scan
    scan_dict["retention_time"] = retention_time
    scan_dict.update(extra)
    return scan_dict


def test_process_mzml_json_basic():
    """
    Test that a simple single-level mzML JSON object is processed correctly.

    Asserts:
        - The MS1 level is present in the output, keyed by its integer level.
        - The scan and intensity data survive the round trip.
    """
    data = {
        "ms1": {
            "spectrum_1": _scan({"100.5": 1000.0}, scan="1", retention_time="5.5"),
        }
    }
    result = att.process_mzml_json(data)
    assert set(result.keys()) == {1}
    assert result[1]["scan"].tolist() == [1]
    assert result[1]["intensity"].tolist() == [1000]


def test_process_mzml_json_ignores_non_ms_keys_interleaved():
    """
    Regression test for a key/value misalignment bug.

    process_mzml_json used to build its per-level results by zipping the
    *unfiltered* top-level keys of `data` against a list of parsed levels
    that had already been filtered to only "ms"-prefixed keys. Any non-"ms"
    key positioned before or between "ms" keys shifted that zip out of
    alignment, silently dropping later MS levels or pairing a key with the
    wrong level's data.

    This test mixes a non-"ms" "meta" key between "ms1" and "ms2" and
    asserts both levels are still present and correctly attributed.

    Asserts:
        - Both MS1 and MS2 levels are present in the output.
        - Each level's scan data matches the level it was defined under, not
          a neighbouring level's data.
    """
    data = {
        "ms1": {
            "spectrum_1": _scan({"100.5": 1000.0}, scan="1", retention_time="5.5"),
        },
        "meta": {"some": "metadata, not an MS level"},
        "ms2": {
            "spectrum_1": _scan({"200.25": 500.0}, scan="2", retention_time="6.5"),
        },
    }
    result = att.process_mzml_json(data)

    assert set(result.keys()) == {1, 2}
    assert result[1]["scan"].tolist() == [1]
    assert result[2]["scan"].tolist() == [2]


def test_process_mzml_json_empty_level_omitted():
    """
    Test that an MS level with no scans is omitted from the result.

    Asserts:
        - A level with an empty scan dict does not appear in the output.
        - A populated level elsewhere in the input is still returned.
    """
    data = {
        "ms1": {},
        "ms2": {
            "spectrum_1": _scan({"200.25": 500.0}, scan="2", retention_time="6.5"),
        },
    }
    result = att.process_mzml_json(data)
    assert set(result.keys()) == {2}
