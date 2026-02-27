import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

import assemblytheorytools as att


def test_pubchem():
    """
    Test PubChem data retrieval functions.

    This function tests various functions for retrieving data from PubChem, including:
    - `pubchem_name_to_smi`
    - `pubchem_name_to_mol`
    - `pubchem_name_to_nx`
    - `pubchem_id_to_smi`
    - `pubchem_id_to_mol`
    - `pubchem_id_to_nx`
    - `sample_random_pubchem`
    - `sample_first_pubchem`

    It asserts that the retrieved data is correct and that the sampling functions
    return the expected number of samples.
    """
    id_str = 'Aspirin'
    id = 2244
    n_sample = 3

    print(flush=True)
    smi = att.pubchem_name_to_smi(id_str)
    print(smi, flush=True)
    assert smi == 'CC(=O)OC1=CC=CC=C1C(=O)O'
    mol = att.pubchem_name_to_mol(id_str, add_hydrogens=True)
    smi_out = Chem.MolToSmiles(mol)
    print(smi_out, flush=True)
    assert smi_out == '[H]OC(=O)c1c([H])c([H])c([H])c([H])c1OC(=O)C([H])([H])[H]'
    graph = att.pubchem_name_to_nx(id_str, add_hydrogens=True)
    smi_out = att.nx_to_smi(graph)
    print(smi_out, flush=True)
    assert smi_out == '[H]OC(=O)C1=C([H])C([H])=C([H])C([H])=C1OC(=O)C([H])([H])[H]'

    smi = att.pubchem_id_to_smi(id)
    print(smi, flush=True)
    assert smi == 'CC(=O)OC1=CC=CC=C1C(=O)O'
    mol = att.pubchem_id_to_mol(id, add_hydrogens=True)
    smi_out = Chem.MolToSmiles(mol)
    print(smi_out, flush=True)
    assert smi_out == '[H]OC(=O)c1c([H])c([H])c([H])c([H])c1OC(=O)C([H])([H])[H]'
    graph = att.pubchem_id_to_nx(id, add_hydrogens=True)
    smi_out = att.nx_to_smi(graph)
    print(smi_out, flush=True)
    assert smi_out == '[H]OC(=O)C1=C([H])C([H])=C([H])C([H])=C1OC(=O)C([H])([H])[H]'

    _, smi_out = att.sample_random_pubchem(n_sample)
    print(smi_out, flush=True)
    assert len(smi_out) == n_sample

    _, smi_out = att.sample_first_pubchem(n_sample)
    print(smi_out, flush=True)
    assert len(smi_out) == n_sample

    # att.download_pubchem_cid_smiles_gz()
    # assert os.path.exists('CID-SMILES.gz')
    # id_out, smi_out = att.sample_pubchem_cid_smiles_gz(n_sample)
    # print(id_out, smi_out, flush=True)
    # assert len(smi_out) == n_sample


def test_pubchem_smi_to_name():
    """
    Test the conversion of a SMILES string to a PubChem name.

    This function converts a SMILES string for lidocaine to its name using
    `pubchem_smi_to_name` and asserts that the result is 'lidocaine'.
    """
    print(flush=True)
    smi = 'CCN(CC)CC(=O)NC1=C(C=CC=C1C)C'
    name = att.pubchem_smi_to_name(smi)
    assert name == 'lidocaine'


def test_filter_by_n_bonds():
    """
    Test filtering a DataFrame of SMILES strings by the number of bonds.

    This function creates a DataFrame of SMILES strings and filters it based on the
    total number of bonds, asserting that the correct number of molecules remain.
    """
    print(flush=True)
    smis = ['[Fe]',
            'CC(N(C(=O)Nc1cc2ccc1CCc1ccc(CC2)cc1)C(C)C)C',
            'O=C1OC(N=C1Cc1c[nH]c2c1cccc2)C(F)(F)F',
            'Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1',
            'Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1']
    df = pd.DataFrame({'smiles': smis})
    filtered_smis = att.filter_by_bonds(df, min_bonds=1, max_bonds=50)
    assert len(filtered_smis) == 2


def test_filter_by_nh_bonds():
    """
    Test filtering a DataFrame of SMILES strings by the number of non-hydrogen bonds.

    This function creates a DataFrame of SMILES strings and filters it based on the
    number of non-hydrogen bonds, asserting that the correct number of molecules remain.
    """
    print(flush=True)
    smis = ['[Fe]',
            'CC(N(C(=O)Nc1cc2ccc1CCc1ccc(CC2)cc1)C(C)C)C',
            'O=C1OC(N=C1Cc1c[nH]c2c1cccc2)C(F)(F)F',
            'Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1',
            'Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1']
    df = pd.DataFrame({'smiles': smis})
    filtered_smis = att.filter_by_nh_bonds(df, min_bonds=1, max_bonds=30)

    assert len(filtered_smis) == 3


def test_filter_by_mw():
    """
    Test filtering a DataFrame of SMILES strings by molecular weight.

    This function creates a DataFrame of SMILES strings and filters it based on
    molecular weight, asserting that the correct number of molecules remain.
    """
    print(flush=True)
    smis = ['[Fe]',
            'CC(N(C(=O)Nc1cc2ccc1CCc1ccc(CC2)cc1)C(C)C)C',
            'O=C1OC(N=C1Cc1c[nH]c2c1cccc2)C(F)(F)F',
            'Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1',
            'Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1']
    df = pd.DataFrame({'smiles': smis})
    filtered_smis = att.filter_by_mw(df, min_mw=100, max_mw=300)
    assert len(filtered_smis) == 2


def test_load_ir_jcamp_data():
    """
    Test loading of IR data from a JCAMP-DX file.

    This function loads an IR spectrum from a JCAMP-DX file and asserts that the
    resulting spectrum has the correct shape and that the wavenumbers and intensities
    arrays have the same length.
    """
    print(flush=True)
    ir_file = 'tests/data/ir_jcamp'
    spectrum = att.load_ir_jcamp_data(ir_file)
    assert len(np.shape(spectrum)) == 2  # Expecting two arrays: wavenumbers and intensities
    assert len(spectrum[0]) == len(spectrum[1])  # Wavenumbers and intensities should have the same length


def test_find_peak_indices_in_range():
    """
    Test finding peak indices in a given range of an IR spectrum.

    This function loads an IR spectrum, applies a Savitzky-Golay filter, finds peaks
    within a specified range, and asserts that the correct number of peaks are found.
    It also visualizes the spectrum with the detected peaks.
    """
    print(flush=True)
    ir_file = 'tests/data/ir_jcamp'
    spectrum = att.load_ir_jcamp_data(ir_file)
    spectrum = att.apply_sg_filter(spectrum, window_length=35, polyorder=3)
    peaks = att.find_peak_indices_in_range(spectrum, min_x=400, max_x=1500, prominence=0.01, distance=5)

    att.plot_ir_spectrum(spectrum, peaks=peaks)
    plt.show()

    assert len(peaks) == 14


def test_calc_n_peaks_in_range():
    """
    Test calculating the number of peaks in a given range of an IR spectrum.

    This function loads an IR spectrum and calculates the number of peaks within a
    specified range, asserting that the result is correct.
    """
    print(flush=True)
    ir_file = 'tests/data/ir_jcamp'
    spectrum = att.load_ir_jcamp_data(ir_file)
    n_peaks = att.find_n_peak_indices_in_range(spectrum, min_x=500, max_x=1500)
    assert n_peaks == 32


def test_get_github_file():
    """
    Test downloading a file from a GitHub repository.

    This function downloads a file from a specified GitHub repository and asserts that
    the file path is not None. It then removes the downloaded file.
    """
    print(flush=True)
    repo_url = "https://raw.githubusercontent.com/ELIFE-ASU/CBRdb/refs/heads/main"
    path = att.get_github_file("CBRdb_C.csv.zip", repo_url)

    assert path is not None
    os.remove(path)


def test_sample_cbrdb():
    """
    Test sampling from the CBRdb database.

    This function samples a specified number of entries from the CBRdb database and
    asserts that the resulting DataFrame is not None and has the correct length.
    """
    print(flush=True)
    n_sample = 100
    df = att.sample_cbrdb(n_sample)
    assert df is not None
    assert len(df) == n_sample


def test_enumerate_stereoisomers_shortest():
    """
    Test the enumeration of stereoisomers to find the one with the shortest name.

    This function enumerates the stereoisomers of a given molecule and finds the one
    with the shortest name, asserting that the name is 'Codeine'.
    """
    print(flush=True)
    smi = 'COC1=C2OC3C(O)C=CC4C5CC(=C2C43CCN5C)C=C1'
    mol = Chem.MolFromSmiles(smi)
    smi_out = att.enumerate_stereoisomers_shortest(mol)
    name_out = att.pubchem_smi_to_name(smi_out, prefer="synonym")
    assert name_out == 'Codeine'


def test_show_ir_data():
    """
    Test function to process and visualize IR data from Chemotion.

    This function performs the following steps:
    1. Processes a Chemotion IR data archive.
    2. Filters the data by the number of non-hydrogen bonds and removes certain elements.
    3. Samples 100 entries from the filtered data.
    4. Calculates the assembly index for each molecule.
    5. Sorts the data by assembly index.
    6. Selects two molecules (one with low AI, one with high AI) for visualization.
    7. Applies a Savitzky-Golay filter to the IR spectra.
    8. For each selected molecule, it finds peaks in the IR spectrum, plots the spectrum,
       and plots the 3D structure of the molecule.
    """
    import time
    df = att.process_chemotion_ir_data('/home/louie/Downloads/10.22000-OGoEQGlsZGElrgst.tar')
    df = att.filter_by_nh_bonds(df, max_bonds=30)

    # print the set of symbols in the smiles column
    all_symbols = set()
    for smi in df['smiles']:
        for char in smi:
            all_symbols.add(char)
    print(f"All symbols in SMILES: {all_symbols}", flush=True)
    df = df[~df['smiles'].str.contains('B|I|F|P|K|S')].reset_index(drop=True)

    # Sample 50 entries from the dataframe
    df = df.sample(n=100, random_state=42).reset_index(drop=True)
    graphs = att.mp_calc(att.smi_to_nx, df['smiles'].tolist())
    df['ai'] = att.calculate_assembly_index_parallel(graphs, settings={'strip_hydrogen': True})[0]

    # sort by ai descending
    df = df.sort_values(by='ai', ascending=False).reset_index(drop=True)

    # # loop over the entire dataframe and print ai and smiles
    # for i, row in df.iterrows():
    #     time.sleep(0.5)
    #     # print(f"{i}, AI: {row['ai']}, SMILES: {row['smiles']}, Name: {row['name']}", flush=True)
    #     print(f"{i}, AI: {row['ai']}, SMILES: {row['smiles']}, Name: {att.pubchem_smi_to_name(row['smiles'])}",
    #           flush=True)

    # 99, AI: 3, SMILES: CCCCCN, Name: Amylamine
    smi_1 = 'CCCCCN'
    # 9, AI: 15, SMILES: COC(=O)[C@H](Cc1c[nH]c2c1cccc2)NC(=O)OC(C)(C)C, Name: Boc-Trp-Ome
    smi_2 = 'COC(=O)[C@H](Cc1c[nH]c2c1cccc2)NC(=O)OC(C)(C)C'

    # find indices of these two smiles in the dataframe
    idx_1 = df.index[df['smiles'] == smi_1].tolist()[0]
    idx_2 = df.index[df['smiles'] == smi_2].tolist()[0]

    # Apply Savitzky-Golay filter to smooth the IR spectra in parallel
    df['spectrum'] = att.mp_calc(att.apply_sg_filter, df['spectrum'])

    view_idx = idx_1
    # print the
    peaks = att.find_peak_indices_in_range(df['spectrum'].iloc[view_idx])
    att.plot_ir_spectrum(df['spectrum'].iloc[view_idx], peaks=peaks)
    plt.savefig(f"{idx_1}_ir_spectrum.svg")
    plt.savefig(f"{idx_1}_ir_spectrum.png", dpi=300)
    plt.show()
    atoms = att.smiles_to_atoms(df['smiles'].iloc[view_idx])
    att.plot_ase_atoms(atoms, f"{idx_1}_atoms.png", rotation='30x,0y,0z')
    plt.show()

    view_idx = idx_2
    peaks = att.find_peak_indices_in_range(df['spectrum'].iloc[view_idx])
    att.plot_ir_spectrum(df['spectrum'].iloc[view_idx], peaks=peaks)
    plt.savefig(f"{idx_2}_ir_spectrum.svg")
    plt.savefig(f"{idx_2}_ir_spectrum.png", dpi=300)
    plt.show()
    atoms = att.smiles_to_atoms(df['smiles'].iloc[view_idx])
    att.plot_ase_atoms(atoms, f"{idx_2}_atoms.png", rotation='0x,120y,90z')
    plt.show()
