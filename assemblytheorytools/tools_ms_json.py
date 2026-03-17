import json

import pandas as pd


def link_msn(data):
    """
    Link MSn peaks to their parent MSn-1 peaks.
    Parameters
    dataset: a dictionary of dataframes, one for each level: {1: ms1_df, 2: ms2_df, ...}
    At a minimum, ms1_df must have columns `scan`, `mz`; ms2_df, etc. have columns `scan`, `mz`, `parent`, `parent_scan`.
    """
    new_dataset = {}
    new_dataset[min(data)] = data[min(data)]
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


DIGITS = set(str(x) for x in range(10))


def try_parse(parser, default):
    def inner(value):
        try:
            return parser(value)
        except ValueError:
            return default

    return inner


REQUIRED_KEYS = {
    "scan": int,
    "retention_time": float,
}

OPTIONAL_KEYS = {
    "parent": float,
    "parent_scan": int,
    "hcd": try_parse(float, 0.0),
}


def scan_to_df(scan_dict: dict):
    mass_dict = {float(k): v for k, v in scan_dict.items() if k[0] in DIGITS}
    df = pd.DataFrame.from_dict(mass_dict, orient="index", columns=["intensity"])
    df = df.assign(scan=int(scan_dict["scan"]), retention_time=float(scan_dict["retention_time"]))
    for key, process_fn in REQUIRED_KEYS.items():
        df[key] = process_fn(scan_dict[key])

    for key, process_fn in OPTIONAL_KEYS.items():
        if key in scan_dict:
            df[key] = process_fn(scan_dict[key])
    return df


def read_level(level_data: dict):
    """
    concatenate all spectra into one dataframe, with a column for the spectrum id extracted from ms1's keys (e.g. spectrum_1 -> 1)
    and a column for the mass extracted from the index
    """
    if not level_data:
        return None
    return pd.concat(
        [scan_to_df(s) for s in level_data.values()],
        keys=[int(k.split("_")[1]) for k in level_data],
        names=["spectrum_id", "mz"],
    )


def process_mzml_json(data):
    """
    Process a mzml json file into a dictionary of dataframes, one for each level.

    Parameters
    data: either a parsed json object or a path to a json file.

    Returns
    a dictionary of dataframes, one for each level. Each dataframe has a multiindex with the first level being `spectra_id`
    and the second level being `mass` (unrounded). The dataframe has a column `intensity` and for MS2 and above, columns
    `parent` and `parent_scan`.
    """
    if not isinstance(data, dict):
        with open(data) as f:
            data = json.load(f)
    level_data = [read_level(v) for k, v in data.items() if k.startswith("ms")]
    return {
        int(k[2:]): v
        for k, v in zip(data, level_data)
        if k.startswith("ms") and v is not None
    }
