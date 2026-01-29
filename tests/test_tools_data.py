import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import assemblytheorytools as att


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

    freq = spectrum.T[0]
    intensity = spectrum.T[1]

    plt.plot(freq, intensity)
    plt.scatter(freq[peaks], intensity[peaks], color='red')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.show()

    assert len(peaks) == 12


def test_calc_n_peaks_in_range():
    print(flush=True)
    ir_file = 'tests/data/ir_jcamp'
    spectrum = att.load_ir_jcamp_data(ir_file)
    n_peaks = att.find_n_peak_indices_in_range(spectrum, 500, 1500)
    assert n_peaks == 32


from pathlib import Path
from urllib.request import urlopen, Request
import shutil


def get_github_file(
        filename: str,
        repo_url: str = "https://raw.githubusercontent.com/ELIFE-ASU/CBRdb/refs/heads/main",
        dest_dir: str | Path = Path.cwd(),
        overwrite: bool = False,
        timeout: int = 30,
) -> Path:
    dest_dir = Path(dest_dir).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    out_path = dest_dir / filename
    url = f"{repo_url.rstrip('/')}/{filename}"

    if out_path.exists() and not overwrite:
        print(f"File already exists, skipping download: {out_path}")
        return out_path

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    # Some servers like having a User-Agent set
    req = Request(url, headers={"User-Agent": "python-download/1.0"})

    print(f"Downloading {url} -> {out_path}", flush=True)
    with urlopen(req, timeout=timeout) as r, open(tmp_path, "wb") as f:
        shutil.copyfileobj(r, f)

    tmp_path.replace(out_path)
    return out_path


def test_get_github_file():
    path = get_github_file("CBRdb_C.csv.zip", dest_dir=Path.cwd())
