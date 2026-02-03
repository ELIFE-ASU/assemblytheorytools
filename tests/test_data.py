import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import assemblytheorytools as att


def test_pubchem_smi_to_name():
    print(flush=True)
    smi = 'CCN(CC)CC(=O)NC1=C(C=CC=C1C)C'
    name = att.pubchem_smi_to_name(smi)
    assert name == 'lidocaine'


def test_filter_by_n_bonds():
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
    print(flush=True)
    ir_file = 'tests/data/ir_jcamp'
    spectrum = att.load_ir_jcamp_data(ir_file)
    assert len(np.shape(spectrum)) == 2  # Expecting two arrays: wavenumbers and intensities
    assert len(spectrum[0]) == len(spectrum[1])  # Wavenumbers and intensities should have the same length


def test_find_peak_indices_in_range():
    print(flush=True)
    ir_file = 'tests/data/ir_jcamp'
    spectrum = att.load_ir_jcamp_data(ir_file)
    spectrum = att.apply_sg_filter(spectrum, window_length=35, polyorder=3)
    peaks = att.find_peak_indices_in_range(spectrum, min_x=500, max_x=1500, prominence=0.01, distance=5)

    att.plot_ir_spectrum(spectrum, peaks=peaks)
    plt.show()

    assert len(peaks) == 12


def test_calc_n_peaks_in_range():
    print(flush=True)
    ir_file = 'tests/data/ir_jcamp'
    spectrum = att.load_ir_jcamp_data(ir_file)
    n_peaks = att.find_n_peak_indices_in_range(spectrum, 500, 1500)
    assert n_peaks == 32


def test_get_github_file():
    print(flush=True)
    repo_url = "https://raw.githubusercontent.com/ELIFE-ASU/CBRdb/refs/heads/main"
    path = att.get_github_file("CBRdb_C.csv.zip", repo_url)

    assert path is not None
    os.remove(path)


def test_sample_cbrdb():
    print(flush=True)
    n_sample = 100
    df = att.sample_cbrdb(n_sample)
    assert df is not None
    assert len(df) == n_sample
