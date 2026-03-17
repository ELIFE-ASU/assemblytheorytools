import json

import pandas as pd


def _link_msn(data):
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


def _try_parse(parser, default):
    def inner(value):
        try:
            return parser(value)
        except ValueError:
            return default

    return inner


def _scan_to_df(scan_dict: dict):
    required_keys = {
        "scan": int,
        "retention_time": float,
    }

    optional_keys = {
        "parent": float,
        "parent_scan": int,
        "hcd": _try_parse(float, 0.0),
    }
    digits = set(str(x) for x in range(10))

    mass_dict = {float(k): v for k, v in scan_dict.items() if k[0] in digits}
    df = pd.DataFrame.from_dict(mass_dict, orient="index", columns=["intensity"])
    df = df.assign(scan=int(scan_dict["scan"]), retention_time=float(scan_dict["retention_time"]))
    for key, process_fn in required_keys.items():
        df[key] = process_fn(scan_dict[key])

    for key, process_fn in optional_keys.items():
        if key in scan_dict:
            df[key] = process_fn(scan_dict[key])
    return df


def _read_level(level_data: dict):
    if not level_data:
        return None
    return pd.concat(
        [_scan_to_df(s) for s in level_data.values()],
        keys=[int(k.split("_")[1]) for k in level_data],
        names=["spectrum_id", "mz"],
    )


def process_mzml_json(data):
    if not isinstance(data, dict):
        with open(data) as f:
            data = json.load(f)
    level_data = [_read_level(v) for k, v in data.items() if k.startswith("ms")]
    return {
        int(k[2:]): v
        for k, v in zip(data, level_data)
        if k.startswith("ms") and v is not None
    }
