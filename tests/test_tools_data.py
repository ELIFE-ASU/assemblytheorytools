import pandas as pd

import assemblytheorytools as att
import numpy as np
import matplotlib.pyplot as plt


def test_filter_by_n_bonds():
    smis = ['[Fe]',
            'CC(N(C(=O)Nc1cc2ccc1CCc1ccc(CC2)cc1)C(C)C)C',
            'O=C1OC(N=C1Cc1c[nH]c2c1cccc2)C(F)(F)F',
            'Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1',
            'Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1']
    df = pd.DataFrame({'smiles': smis})
    filtered_smis = att.filter_by_bonds(df, min_bonds=1, max_bonds=50)
    assert len(filtered_smis) == 2


def test_filter_by_nh_bonds():
    smis = ['[Fe]',
            'CC(N(C(=O)Nc1cc2ccc1CCc1ccc(CC2)cc1)C(C)C)C',
            'O=C1OC(N=C1Cc1c[nH]c2c1cccc2)C(F)(F)F',
            'Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1',
            'Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1']
    df = pd.DataFrame({'smiles': smis})
    filtered_smis = att.filter_by_nh_bonds(df, min_bonds=1, max_bonds=30)

    assert len(filtered_smis) == 3


def test_filter_by_mw():
    smis = ['[Fe]',
            'CC(N(C(=O)Nc1cc2ccc1CCc1ccc(CC2)cc1)C(C)C)C',
            'O=C1OC(N=C1Cc1c[nH]c2c1cccc2)C(F)(F)F',
            'Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1',
            'Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1Nc1ccc(cc1)c1cc2ccc1CCc1ccc(CC2)cc1']
    df = pd.DataFrame({'smiles': smis})
    filtered_smis = att.filter_by_mw(df, min_mw=100, max_mw=300)
    assert len(filtered_smis) == 2


def test_load_ir_jcamp_data():
    ir_file = 'tests/data/ir_jcamp'
    spectrum = att.load_ir_jcamp_data(ir_file)
    assert len(np.shape(spectrum)) == 2  # Expecting two arrays: wavenumbers and intensities
    assert len(spectrum[0]) == len(spectrum[1])  # Wavenumbers and intensities should have the same length


def test_find_peak_indices_in_range():
    ir_file = 'tests/data/ir_jcamp'
    spectrum = att.load_ir_jcamp_data(ir_file)
    peaks = att.find_peak_indices_in_range(spectrum, 500, 1500)

    plt.plot(spectrum.T[0], spectrum.T[1])
    plt.scatter(spectrum.T[0][peaks], spectrum.T[1][peaks], color='red')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.show()

    assert len(peaks)


def test_calc_n_peaks_in_range():
    ir_file = 'tests/data/ir_jcamp'
    spectrum = att.load_ir_jcamp_data(ir_file)
    n_peaks = att.calc_n_peaks_in_range(spectrum, 500, 1500)
    print(n_peaks)
    assert n_peaks > 32
