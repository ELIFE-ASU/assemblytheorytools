"""
Processing of mass spectrometry data in JSON form.

This module reads the JSON representation produced from mzML files and extracts
the spectra and peak lists used for downstream assembly analysis.
"""

import json

import pandas as pd

# Characters that can lead an m/z key in a scan dict, as opposed to metadata keys.
_DECIMAL_DIGITS = set("0123456789")


def _link_msn(data):
    """
    Link MSn data levels by merging parent and child scans.

    Parameters
    ----------
    data : dict
        Dictionary where keys are MSn levels (integers) and values are pandas DataFrames
        containing scan data for each level.

    Returns
    -------
    dict
        Dictionary with the same structure as input, but with child levels linked to their parent scans.
    """
    first_level = min(data)
    new_dataset = {first_level: data[first_level]}
    for level in sorted(data)[:-1]:
        new_dataset[level + 1] = (
            pd.merge(
                new_dataset[level][["scan", "mz"]].reset_index(),
                data[level + 1],
                how="inner",
                left_on=["scan", "mz"],
                right_on=["parent_scan", "parent"],
                suffixes=("_x", ""),
            )
            .rename(columns={"index": "parent_id"})
            .drop(columns=["scan_x", "mz_x"])
        )
    return new_dataset


def _try_parse(parser, default):
    """
    Create a function that attempts to parse a value, returning a default on failure.

    Parameters
    ----------
    parser : callable
        Function to parse the value (e.g., int, float).
    default : any
        Default value to return if parsing fails.

    Returns
    -------
    callable
        Function that takes a value and returns the parsed value or the default.
    """

    def inner(value):
        """
        Parse a value, falling back to the default on failure.

        Parameters
        ----------
        value : Any
            The value to pass to the wrapped parser.

        Returns
        -------
        Any
            The parsed value, or the captured default if parsing raised
            ``ValueError``.
        """
        try:
            return parser(value)
        except ValueError:
            return default

    return inner


def _scan_to_df(scan_dict: dict):
    """
    Convert a scan dictionary to a pandas DataFrame.

    Parameters
    ----------
    scan_dict : dict
        Dictionary containing scan information and mass/intensity pairs.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns for intensity, scan, retention_time, and optional parent information.
    """
    required_keys = {
        "scan": int,
        "retention_time": float,
    }

    optional_keys = {
        "parent": float,
        "parent_scan": int,
        "hcd": _try_parse(float, 0.0),
    }
    mass_dict = {float(k): v for k, v in scan_dict.items() if k[0] in _DECIMAL_DIGITS}
    df = pd.DataFrame.from_dict(mass_dict, orient="index", columns=["intensity"])
    df = df.assign(scan=int(scan_dict["scan"]), retention_time=float(scan_dict["retention_time"]))
    for key, process_fn in required_keys.items():
        df[key] = process_fn(scan_dict[key])
    for key, process_fn in optional_keys.items():
        if key in scan_dict:
            df[key] = process_fn(scan_dict[key])
    return df


def _read_level(level_data: dict):
    """
    Convert a dictionary of scans for a given MS level to a concatenated DataFrame.

    Parameters
    ----------
    level_data : dict
        Dictionary where each value is a scan dictionary for a given spectrum.

    Returns
    -------
    pandas.DataFrame or None
        Concatenated DataFrame of all scans in the level, or None if input is empty.
    """
    if not level_data:
        return None
    return pd.concat(
        [_scan_to_df(s) for s in level_data.values()],
        keys=[int(k.split("_")[1]) for k in level_data],
        names=["spectrum_id", "mz"],
    )


def process_mzml_json(data):
    """
    Process an mzML JSON object or file into a dictionary of MSn level DataFrames.

    Parameters
    ----------
    data : dict or str
        JSON object or path to a JSON file containing MSn data.

    Returns
    -------
    dict
        Dictionary mapping MSn levels (int) to pandas DataFrames of scan data.
    """
    if not isinstance(data, dict):
        with open(data) as f:
            data = json.load(f)
    return {
        int(k[2:]): level
        for k, v in data.items()
        if k.startswith("ms") and (level := _read_level(v)) is not None
    }
