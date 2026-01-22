import json
import os
import re
import tarfile
from typing import List, Tuple, Optional, Sequence

import pandas as pd

import assemblytheorytools as att


def load_jcamp_xy_lists(path: str) -> Tuple[List[float], List[float]]:
    # Metadata defaults
    xfactor = 1.0
    yfactor = 1.0
    firstx: Optional[float] = None
    lastx: Optional[float] = None
    deltax: Optional[float] = None
    npoints: Optional[int] = None

    # State
    in_data = False
    data_mode: Optional[str] = None  # "xy_pairs" or "xpp_ylist"

    frequencies: List[float] = []
    intensities: List[float] = []

    # Regex helpers
    keyval_re = re.compile(r"^##\s*([^=]+)\s*=\s*(.*)\s*$")
    float_re = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")
    int_re = re.compile(r"[-+]?\d+")

    def extract_numbers(line: str) -> List[float]:
        # JCAMP often uses commas; normalize commas/semicolons to spaces.
        line = line.replace(",", " ").replace(";", " ")
        return [float(x) for x in float_re.findall(line)]

    def parse_float(s: str) -> Optional[float]:
        m = float_re.search(s)
        return float(m.group(0)) if m else None

    def parse_int(s: str) -> Optional[int]:
        m = int_re.search(s)
        return int(m.group(0)) if m else None

    # Keys that may introduce a data block in the wild
    DATA_KEYS = {"XYDATA", "XYPOINTS", "DATA TABLE", "DATATABLE"}

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Ignore JCAMP comment lines
            if line.startswith("$$"):
                continue

            # Any ##END... ends parsing
            if line.upper().startswith("##END"):
                break

            # Header line
            if line.startswith("##"):
                m = keyval_re.match(line)
                if not m:
                    continue

                key = m.group(1).strip().upper()
                val = m.group(2).strip()

                # Start of a data block
                if key in DATA_KEYS:
                    in_data = True
                    uval = val.upper()
                    data_mode = "xpp_ylist" if "X++" in uval else "xy_pairs"
                    continue

                # If we were in a data block and we hit another header, data block is over
                if in_data:
                    in_data = False
                    data_mode = None
                    # continue on to parse metadata from this header too

                # Metadata parsing (same as your original)
                if key == "XFACTOR":
                    v = parse_float(val)
                    if v is not None:
                        xfactor = v
                elif key == "YFACTOR":
                    v = parse_float(val)
                    if v is not None:
                        yfactor = v
                elif key == "FIRSTX":
                    firstx = parse_float(val)
                elif key == "LASTX":
                    lastx = parse_float(val)
                elif key == "DELTAX":
                    deltax = parse_float(val)
                elif key in ("NPOINTS", "POINTS"):
                    npoints = parse_int(val)

                continue

            # Non-header lines: only parse if inside a data block
            if not in_data or data_mode is None:
                continue

            nums = extract_numbers(line)
            if not nums:
                continue

            if data_mode == "xy_pairs":
                # x y x y ... (your file is 1 pair per line, comma separated)
                if len(nums) % 2 == 1:
                    nums = nums[:-1]

                for i in range(0, len(nums), 2):
                    frequencies.append(nums[i] * xfactor)
                    intensities.append(nums[i + 1] * yfactor)

            elif data_mode == "xpp_ylist":
                # x0 y1 y2 y3 ...
                x0 = nums[0]
                yvals = nums[1:]
                if not yvals:
                    continue

                dx = deltax
                if dx is None and firstx is not None and lastx is not None and npoints and npoints > 1:
                    dx = (lastx - firstx) / (npoints - 1)

                if dx is None:
                    raise ValueError("X++(Y..Y) data encountered but DELTAX is missing.")

                for j, y in enumerate(yvals):
                    frequencies.append((x0 + j * dx) * xfactor)
                    intensities.append(y * yfactor)

    if not frequencies:
        raise ValueError("No XY data block found (expected ##XYDATA= or ##XYPOINTS=).")

    if npoints is not None and len(frequencies) > npoints:
        frequencies = frequencies[:npoints]
        intensities = intensities[:npoints]

    return frequencies, intensities


def peak_indices_in_range(
        freqs: Sequence[float],
        intens: Sequence[float],
        f_min: float,
        f_max: float,
        *,
        prominence: Optional[float] = None,
        min_distance: Optional[float] = None,
) -> List[int]:
    if len(freqs) != len(intens):
        raise ValueError("freqs and intens must be the same length.")
    n = len(freqs)
    if n < 3:
        return []

    lo, hi = (f_min, f_max) if f_min <= f_max else (f_max, f_min)

    # Candidate peaks
    candidates: List[int] = []
    for i in range(1, n - 1):
        f = freqs[i]
        if f < lo or f > hi:
            continue

        y0, y1, y2 = intens[i - 1], intens[i], intens[i + 1]
        if not (y1 > y0 and y1 >= y2):
            continue

        if prominence is not None and (y1 - max(y0, y2) < prominence):
            continue

        candidates.append(i)

    if not candidates:
        return []

    # Optional: minimum distance filtering
    if min_distance is not None and min_distance > 0:
        # Sort by height (desc), keep far-enough peaks, then return sorted by index
        by_height = sorted(candidates, key=lambda i: intens[i], reverse=True)
        kept: List[int] = []
        for i in by_height:
            fi = freqs[i]
            if all(abs(fi - freqs[j]) >= min_distance for j in kept):
                kept.append(i)
        return sorted(kept)

    return candidates


def process_meta_data_name(entry):
    entries = entry[0]['attacments']
    for e in entries:
        if not e['filename'].endswith('.peak.jdx'):
            return e['identifier'][2:]
    return None


def process_meta_data(extract_dir):
    dir_files = att.file_list_all(extract_dir)

    # find the metadata file 'meta_data.json'
    meta_file = [f for f in dir_files if f.endswith("meta_data.json")][0]
    if not meta_file:
        raise FileNotFoundError("No meta_data.json file found in extracted data.")

    with open(meta_file, "r") as f:
        meta_data = json.load(f)

    # convert to pandas dataframe
    df = pd.DataFrame(meta_data)
    df_selected = df[['cano_smiles', 'datasets']]
    df_selected['name'] = att.mp_calc(process_meta_data_name, df_selected['datasets'])
    # drop entries with name as None
    df_selected = df_selected[df_selected['name'].notna()]
    df_selected = df_selected.drop(columns=['datasets'])
    return df_selected


def process_ir_data(extract_dir):
    dir_files = att.file_list_all(extract_dir)
    ir_file = [f for f in dir_files if f.endswith("IR_data.tar.xz")][0]
    if not ir_file:
        raise FileNotFoundError("No IR_data.tar.xz file found in extracted data.")

    ir_extract_dir = os.path.join(os.path.dirname(ir_file), "IR_data")
    if not os.path.exists(ir_extract_dir):
        with tarfile.open(ir_file, "r:xz") as tar:
            tar.extractall(path=ir_extract_dir)

    ir_files = att.file_list_all(ir_extract_dir)
    # only select files that match with meta_data names
    ir_files = [f for f in ir_files if any(name in f for name in meta_data['name'].tolist())]
    filenames = [os.path.basename(f) for f in ir_files]
    ir_data = pd.DataFrame(filenames, columns=['name'])
    ir_xy = att.mp_calc(load_jcamp_xy_lists, ir_files)
    ir_data['freq'] = [xy[0] for xy in ir_xy]
    ir_data['intensity'] = [xy[1] for xy in ir_xy]
    return ir_data


if __name__ == "__main__":
    # x, y = load_jcamp_xy_lists("651c9779-e75d-4c7c-ad41-1ce312a9e281")
    #
    # # sort the data by x values
    # xy_sorted = sorted(zip(x, y), key=lambda pair: pair[0])
    # x, y = zip(*xy_sorted)
    #
    # plt.plot(x, y)
    # # limit x axis to 0-4000
    # plt.xlim(500, 1500)
    #
    # plt.xlabel("Freq. 1/cm")
    # plt.ylabel("Intensity")
    #
    #
    # locs = peak_indices_in_range(x, y, f_min=500, f_max=1500, prominence=None, min_distance=None)
    # print(f"Number of peaks between 500 and 1500 cm^-1: {len(locs)}", flush=True)
    # print("Peak locations (1/cm):", flush=True)
    # for loc in locs:
    #     print(f"{x[loc]:.2f}", flush=True)
    #
    # plt.scatter([x[i] for i in locs], [y[i] for i in locs], color='red')
    # plt.show()

    # Download the file
    # https://radar4chem.radar-service.eu/radar/en/dataset/OGoEQGlsZGElrgst#
    target_file = "/home/louie/Downloads/10.22000-OGoEQGlsZGElrgst.tar"
    extract_dir = os.path.join(os.path.dirname(target_file), "radar_data")

    # check if the extract_dir exists, if not create it
    if not os.path.exists(extract_dir):
        with tarfile.open(target_file, "r") as tar:
            tar.extractall(path=extract_dir)
        print(f"Extracted data to {extract_dir}", flush=True)

    meta_data = process_meta_data(extract_dir)

    ir_data = process_ir_data(extract_dir)

    # merge meta_data and ir_data on name
    merged_data = pd.merge(meta_data, ir_data, on='name')
    # drop rows with any NaN values
    merged_data = merged_data.dropna()
    print(len(merged_data))
    print(merged_data.head(), flush=True)
    # save to csv
    merged_data.to_csv("radar_ir_data.csv", index=False)
