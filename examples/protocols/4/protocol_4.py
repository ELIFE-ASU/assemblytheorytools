import json
import os
import tarfile
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import assemblytheorytools as att


def _calc_ai(smi: str, settings: Optional[dict] = None) -> int:
    if settings is None:
        settings = {}
    try:
        ai = att.calculate_assembly_index(att.smi_to_nx(smi), **settings)[0]
        return int(ai)
    except Exception:
        return -1


def _process_meta_data_name(entry: List[dict]) -> Optional[str]:
    try:
        entries = entry[0]['attacments']
        for e in entries:
            if not e['filename'].endswith('.peak.jdx'):
                return e['identifier'].split('/')[-1]
    except (IndexError, KeyError, TypeError):
        pass
    return None


def _valid_smi(smi: str) -> bool:
    return bool(smi) and all(x not in smi for x in [".", "*", "->", "$"])


def _process_chemotion_meta_section(extract_dir: str) -> pd.DataFrame:
    # Locate the metadata file
    meta_file = next((f for f in att.file_list_all(extract_dir) if f.endswith("meta_data.json")), None)
    if not meta_file:
        raise FileNotFoundError("No meta_data.json file found in extracted data.")

    with open(meta_file, "r") as f:
        meta_data = json.load(f)

    # Convert to pandas dataframe and select columns
    df = pd.DataFrame(meta_data)[['cano_smiles', 'datasets']]
    df = df.rename(columns={'cano_smiles': 'smiles'})

    # Drop entries with invalid smiles
    df = df[att.mp_calc(_valid_smi, df['smiles'])]

    # Process 'datasets' to get 'name' and clean up
    df['name'] = att.mp_calc(_process_meta_data_name, df['datasets'])
    df = df.dropna(subset=['name']).drop(columns=['datasets'])

    return df


def _process_chemotion_ir_section(extract_dir: str, meta_data: pd.DataFrame) -> pd.DataFrame:
    # Locate the IR data archive
    ir_file = next((f for f in att.file_list_all(extract_dir) if f.endswith("IR_data.tar.xz")), None)
    if not ir_file:
        raise FileNotFoundError("No IR_data.tar.xz file found in extracted data.")

    # Extract the IR data if not already extracted
    ir_extract_dir = os.path.join(os.path.dirname(ir_file), "IR_data")
    if not os.path.exists(ir_extract_dir):
        with tarfile.open(ir_file, "r:xz") as tar:
            tar.extractall(path=ir_extract_dir)

    # Filter IR files based on metadata names
    target_names = meta_data['name'].tolist()
    ir_files = [
        f for f in att.file_list_all(ir_extract_dir)
        if any(name in f for name in target_names)
    ]
    filenames = [os.path.basename(f) for f in ir_files]

    # Create a DataFrame with filenames and their corresponding spectra
    ir_data = pd.DataFrame({'name': filenames})
    ir_data['spectrum'] = att.mp_calc(att.load_ir_jcamp_data, ir_files)
    return ir_data


def process_chemotion_ir_data(target_file: str) -> pd.DataFrame:
    extract_dir = os.path.join(os.path.dirname(target_file), "chemotion_ir_data")
    out_file = "chemotion_ir_data.csv.gz"

    if os.path.exists(out_file):
        print(f"{out_file} already exists. Skipping processing.", flush=True)
        return pd.read_csv(out_file)

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
    os.remove('chemotion_ir_data.csv.gz') if os.path.exists('chemotion_ir_data.csv.gz') else None
    df = process_chemotion_ir_data(target_file)

    max_bonds = 20
    df = att.filter_by_nh_bonds(df, max_bonds=max_bonds)

    # calculate number of peaks
    df['n_peaks'] = att.mp_calc(att.calc_n_peaks_in_range, df['spectrum'])

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
