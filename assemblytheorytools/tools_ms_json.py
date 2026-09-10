"""
Processing of mass spectrometry data in JSON form.

This module reads the JSON representation produced from mzML files and extracts
the spectra and peak lists used for downstream assembly analysis.
"""

import json
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd

# Characters that can lead an m/z key in a scan dict, as opposed to metadata keys.
_DECIMAL_DIGITS = set("0123456789")


def _link_msn(data: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
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
    linked = {first_level: data[first_level]}
    for level in sorted(data)[:-1]:
        parent_peaks = linked[level][["scan", "mz"]].reset_index()
        linked[level + 1] = (
            pd.merge(
                parent_peaks,
                data[level + 1],
                how="inner",
                left_on=["scan", "mz"],
                right_on=["parent_scan", "parent"],
                suffixes=("_x", ""),
            )
            .rename(columns={"index": "parent_id"})
            .drop(columns=["scan_x", "mz_x"])
        )
    return linked


def _try_parse(parser: Callable[[Any], Any], default: Any) -> Callable[[Any], Any]:
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

    def inner(value: Any) -> Any:
        """Parse a value, falling back only when the parser raises ValueError."""
        try:
            return parser(value)
        except ValueError:
            return default

    return inner


def _scan_to_df(scan_dict: dict) -> pd.DataFrame:
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
    optional_parsers = {
        "parent": float,
        "parent_scan": int,
        "hcd": _try_parse(float, 0.0),
    }
    peaks = {
        float(mass): intensity
        for mass, intensity in scan_dict.items()
        if mass[0] in _DECIMAL_DIGITS
    }
    df = pd.DataFrame.from_dict(peaks, orient="index", columns=["intensity"]).assign(
        scan=int(scan_dict["scan"]),
        retention_time=float(scan_dict["retention_time"]),
    )
    for key, parser in optional_parsers.items():
        if key in scan_dict:
            df[key] = parser(scan_dict[key])
    return df


def _read_level(level_data: dict) -> Optional[pd.DataFrame]:
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
        [_scan_to_df(scan) for scan in level_data.values()],
        keys=[int(name.split("_")[1]) for name in level_data],
        names=["spectrum_id", "mz"],
    )


def process_mzml_json(data: Union[Dict[str, Any], str]) -> Dict[int, pd.DataFrame]:
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
        with open(data) as source:
            data = json.load(source)
    return {
        int(name[2:]): level
        for name, scans in data.items()
        if name.startswith("ms") and (level := _read_level(scans)) is not None
    }
