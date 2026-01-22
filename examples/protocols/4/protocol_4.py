import json
import os
import re
import tarfile
from typing import List, Tuple, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import assemblytheorytools as att


def load_ir_jcamp_data(path: str) -> Tuple[List[float], List[float]]:
    xfactor = 1.0
    yfactor = 1.0
    firstx: Optional[float] = None
    lastx: Optional[float] = None
    deltax: Optional[float] = None
    npoints: Optional[int] = None
    in_data = False
    data_mode: Optional[str] = None
    frequencies: List[float] = []
    intensities: List[float] = []

    keyval_re = re.compile(r"^##\s*([^=]+)\s*=\s*(.*)\s*$")
    float_re = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")
    int_re = re.compile(r"[-+]?\d+")

    def _extract_numbers(line: str) -> List[float]:
        line = line.replace(",", " ").replace(";", " ")
        return [float(x) for x in float_re.findall(line)]

    def _parse_float(s: str) -> Optional[float]:
        m = float_re.search(s)
        return float(m.group(0)) if m else None

    def _parse_int(s: str) -> Optional[int]:
        m = int_re.search(s)
        return int(m.group(0)) if m else None

    data_keys = {"XYDATA", "XYPOINTS", "DATA TABLE", "DATATABLE"}

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("$$"):
                continue

            if line.upper().startswith("##END"):
                break

            if line.startswith("##"):
                m = keyval_re.match(line)
                if not m:
                    continue

                key = m.group(1).strip().upper()
                val = m.group(2).strip()

                if key in data_keys:
                    in_data = True
                    uval = val.upper()
                    data_mode = "xpp_ylist" if "X++" in uval else "xy_pairs"
                    continue

                if in_data:
                    in_data = False
                    data_mode = None

                if key == "XFACTOR":
                    v = _parse_float(val)
                    if v is not None:
                        xfactor = v
                elif key == "YFACTOR":
                    v = _parse_float(val)
                    if v is not None:
                        yfactor = v
                elif key == "FIRSTX":
                    firstx = _parse_float(val)
                elif key == "LASTX":
                    lastx = _parse_float(val)
                elif key == "DELTAX":
                    deltax = _parse_float(val)
                elif key in ("NPOINTS", "POINTS"):
                    npoints = _parse_int(val)
                continue

            if not in_data or data_mode is None:
                continue

            nums = _extract_numbers(line)
            if not nums:
                continue

            if data_mode == "xy_pairs":
                if len(nums) % 2 == 1:
                    nums = nums[:-1]
                for i in range(0, len(nums), 2):
                    frequencies.append(nums[i] * xfactor)
                    intensities.append(nums[i + 1] * yfactor)

            elif data_mode == "xpp_ylist":
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


def find_peak_indices_in_range(
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


def _calculate_peaks(row):
    freqs = row['freq']
    intensities = row['intensity']
    locs = find_peak_indices_in_range(freqs,
                                      intensities,
                                      f_min=500,
                                      f_max=1500,
                                      prominence=None,
                                      min_distance=None)
    return len(locs)


def _calc_ai(smi: str) -> int:
    try:
        ai = att.calculate_assembly_index(att.smi_to_nx(smi),
                                          strip_hydrogen=True,
                                          timeout=300.0)[0]
        return ai
    except Exception:
        return -1


def _process_meta_data_name(entry):
    entries = entry[0]['attacments']
    for e in entries:
        if not e['filename'].endswith('.peak.jdx'):
            return e['identifier'].split('/')[-1]
    return None


def _valid_smi(smi: str) -> bool:
    return bool(smi) and all(x not in smi for x in [".", "*", "->", "$"])


def _process_chemotion_meta_section(extract_dir):
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
    # rename 'cano_smiles' to 'smiles'
    df_selected = df_selected.rename(columns={'cano_smiles': 'smiles'})
    # drop entries with invalid smiles
    df_selected = df_selected[att.mp_calc(_valid_smi, df_selected['smiles'])]
    # process 'datasets' to get 'name'
    df_selected['name'] = att.mp_calc(_process_meta_data_name, df_selected['datasets'])
    # drop entries with name as None
    df_selected = df_selected[df_selected['name'].notna()]
    df_selected = df_selected.drop(columns=['datasets'])
    return df_selected


def _process_chemotion_ir_section(extract_dir, meta_data):
    dir_files = att.file_list_all(extract_dir)
    ir_file = [f for f in dir_files if f.endswith("IR_data.tar.xz")][0]
    if not ir_file:
        raise FileNotFoundError("No IR_data.tar.xz file found in extracted data.")

    ir_extract_dir = os.path.join(os.path.dirname(ir_file), "IR_data")
    if not os.path.exists(ir_extract_dir):
        with tarfile.open(ir_file, "r:xz") as tar:
            tar.extractall(path=ir_extract_dir)

    ir_files = att.file_list_all(ir_extract_dir)
    ir_files = [f for f in ir_files if any(name in f for name in meta_data['name'].tolist())]
    filenames = [os.path.basename(f) for f in ir_files]
    ir_data = pd.DataFrame(filenames, columns=['name'])
    ir_xy = att.mp_calc(load_ir_jcamp_data, ir_files)
    ir_data['freq'] = [xy[0] for xy in ir_xy]
    ir_data['intensity'] = [xy[1] for xy in ir_xy]
    return ir_data


def process_chemotion_ir_data(target_file):
    extract_dir = os.path.join(os.path.dirname(target_file), "chemotion_ir_data")
    out_file = "chemotion_ir_data.csv.gz"

    if os.path.exists(out_file):
        print(f"{out_file} already exists. Skipping processing.", flush=True)
        return pd.read_csv(out_file)
    else:

        # Check if the extract_dir exists, if not create it
        if not os.path.exists(extract_dir):
            with tarfile.open(target_file, "r") as tar:
                tar.extractall(path=extract_dir)
            print(f"Extracted data to {extract_dir}", flush=True)

        meta_data = _process_chemotion_meta_section(extract_dir)
        ir_data = _process_chemotion_ir_section(extract_dir, meta_data)

        # Merge meta_data and ir_data on name
        merged_data = pd.merge(meta_data, ir_data, on='name')
        # Drop rows with any NaN values
        merged_data = merged_data.dropna()
        # Save to csv
        merged_data.to_csv(out_file, index=False)
        return merged_data


if __name__ == "__main__":
    # Download the file
    # https://radar4chem.radar-service.eu/radar/en/dataset/OGoEQGlsZGElrgst#
    target_file = "/home/louie/Downloads/10.22000-OGoEQGlsZGElrgst.tar"
    df = process_chemotion_ir_data(target_file)

    max_bonds = 100
    df = att.filter_by_nh_bonds(df, max_bonds=max_bonds)


    # sample molecules for testing
    #df = df.sample(n=300, random_state=42).reset_index(drop=True)

    # calculate number of peaks
    df['n_peaks'] = att.mp_calc(_calculate_peaks, [row for _, row in df.iterrows()])

    # only keep rows with n_peaks > 0
    df = df[df['n_peaks'] > 0].reset_index(drop=True)
    # only keep rows with n_peaks <= 20
    df = df[df['n_peaks'] <= 60].reset_index(drop=True)

    # calculate assembly index
    df['ai'] = att.mp_calc(_calc_ai, df['smiles'].tolist())

    n_x_bins = len(set(df['ai']))
    n_y_bins = len(set(df['n_peaks']))

    fig, ax = att.plot_heatmap(np.array(df['ai']),
                               np.array(df['n_peaks']),
                               "Assembly Index",
                               "Number of Peaks",
                               nbins=(n_x_bins, n_y_bins),
                               )
    plt.show()
