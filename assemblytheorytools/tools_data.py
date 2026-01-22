import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union
from urllib.request import Request, urlopen

import networkx as nx
import numpy as np
import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from scipy.stats import gaussian_kde

from .complexity_scores import count_bonds, count_non_h_bonds, molecular_weight
from .tools_graph import smi_to_nx
from .tools_mol import smi_to_mol, standardize_mol
from .tools_mp import mp_calc


def random_argmin(arr: np.ndarray) -> int:
    """
    Find the index of the minimum value in the array.

    If there are multiple minimum values, return a random index among them.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array to search for the minimum value.

    Returns
    -------
    int
        Index of one of the minimum values, chosen randomly if there are multiple.
    """
    min_value = np.min(arr)
    min_indices = np.where(arr == min_value)[0]
    return int(np.random.choice(min_indices))


def get_close_random_index(data: np.ndarray, point: Union[float, np.ndarray]) -> int:
    """
    Find the index of the data point closest to the given point.
    
    If there are multiple closest points, return a random index among them.

    Parameters
    ----------
    data : numpy.ndarray
        Array of data points.
    point : float or numpy.ndarray
        The point to which the closest data point is to be found.

    Returns
    -------
    int
        Index of one of the closest data points, chosen randomly if there are multiple.
    """
    return random_argmin(np.abs(np.subtract(data, point)))


def sample_boostrapping(data: np.ndarray, n_sample: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform random sampling with replacement (bootstrapping) on the given dataset.

    If you assume your dataset is a fair representation of the underlying distribution,
    then random sampling with replacement (bootstrapping) will retain its PDF.

    Parameters
    ----------
    data : numpy.ndarray
        The input dataset from which to sample.
    n_sample : int
        The number of samples to draw.

    Returns
    -------
    sample : numpy.ndarray
        The sampled values.
    sample_indices : numpy.ndarray
        The indices of the sampled values in the original dataset.
    """
    sample_indices: np.ndarray = np.random.choice(len(data), size=n_sample, replace=True)
    # Extract the selected values
    sample: np.ndarray = data[sample_indices]
    return sample, sample_indices


def sample_kde_resampling(data: np.ndarray, n_sample: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform KDE-based resampling on the given dataset.
    
    If the dataset is 1D, use 1D KDE resampling. If the dataset is 2D, use 2D KDE resampling.

    Parameters
    ----------
    data : numpy.ndarray
        The input dataset from which to sample.
    n_sample : int
        The number of samples to draw.

    Returns
    -------
    sample : numpy.ndarray
        The sampled values.
    sample_indices : numpy.ndarray
        The indices of the sampled values in the original dataset.
        
    Raises
    ------
    ValueError
        If data is not 1D or 2D.
    """
    if data.ndim == 1:
        # 1D data: Use KDE-based resampling
        kde = gaussian_kde(data)
        sample: np.ndarray = kde.resample(n_sample).flatten()
        sample_indices = np.array([get_close_random_index(data, point) for point in sample], dtype=int)
    elif data.ndim == 2:
        # 2D data: Use KDE-based resampling
        kde = gaussian_kde(data.T)
        sample: np.ndarray = kde.resample(n_sample).T
        sample_indices = np.array([get_close_random_index(data, point) for point in sample], dtype=int)
    else:
        raise ValueError("Data must be either 1D or 2D.")

    return sample, sample_indices


def sample_importance_sampling(data: np.ndarray, n_sample: int, n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform importance sampling on the given dataset.

    Parameters
    ----------
    data : numpy.ndarray
        The input dataset from which to sample.
    n_sample : int
        The number of samples to draw.
    n_bins : int, optional
        The number of bins to use for importance sampling. Default is 50.

    Returns
    -------
    sample : numpy.ndarray
        The sampled values.
    sample_indices : numpy.ndarray
        The indices of the sampled values in the original dataset.
        
    Raises
    ------
    ValueError
        If data is not 1D or 2D.
    """
    if data.ndim == 1:
        # 1D data: Use importance sampling
        hist_values: np.ndarray
        bin_edges: np.ndarray
        hist_values, bin_edges = np.histogram(data, bins=n_bins, density=True)
        bin_centers: np.ndarray = (bin_edges[:-1] + bin_edges[1:]) / 2
        probabilities: np.ndarray = hist_values / hist_values.sum()
        selected_bin_indices: np.ndarray = np.random.choice(len(probabilities), size=n_sample, p=probabilities)
        sample_indices = np.array([get_close_random_index(data, bin_centers[idx]) for idx in selected_bin_indices],
                                  dtype=int)
        sample: np.ndarray = data[sample_indices]
    elif data.ndim == 2:
        # 2D data: Use importance sampling
        hist: np.ndarray
        x_edges: np.ndarray
        y_edges: np.ndarray
        hist, x_edges, y_edges = np.histogram2d(data[:, 0], data[:, 1], bins=n_bins, density=True)
        x_centers: np.ndarray = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers: np.ndarray = (y_edges[:-1] + y_edges[1:]) / 2
        xx, yy = np.meshgrid(x_centers, y_centers)
        probabilities: np.ndarray = hist.flatten()
        probabilities /= probabilities.sum()
        selected_indices: np.ndarray = np.random.choice(len(probabilities), size=n_sample, p=probabilities)
        sample_indices = np.array([
            np.argmin(np.linalg.norm(data - np.array([xx.flatten()[idx], yy.flatten()[idx]]), axis=1))
            for idx in selected_indices
        ], dtype=int)
        sample: np.ndarray = data[sample_indices]
    else:
        raise ValueError("Data must be either 1D or 2D.")

    return sample, sample_indices


def filter_by_n_bonds(df: pd.DataFrame,
                      *,
                      min_bonds: int = 0,
                      max_bonds: int = 100,
                      c_smiles: str = 'smiles',
                      c_bonds: str = 'n_bonds') -> pd.DataFrame:
    """
    Filter a DataFrame of molecules based on the number of bonds.

    This function calculates the number of bonds for each molecule in the DataFrame
    and filters the rows to include only those within the specified range.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing molecular data.
    min_bonds : int, optional
        The minimum number of bonds to include. Default is 0.
    max_bonds : int, optional
        The maximum number of bonds to include. Default is 100.
    c_smiles : str, optional
        The column name in the DataFrame containing SMILES strings. Default is 'smiles'.
    c_bonds : str, optional
        The column name to store the calculated number of bonds. Default is 'n_bonds'.

    Returns
    -------
    pandas.DataFrame
        A filtered DataFrame containing only rows with the number of bonds
        within the specified range.
    """
    df[c_bonds] = mp_calc(count_bonds, mp_calc(smi_to_mol, df[c_smiles]))
    return df[(df[c_bonds] >= min_bonds) & (df[c_bonds] <= max_bonds)].reset_index(drop=True)


def filter_by_nh_bonds(df: pd.DataFrame,
                       *,
                       min_bonds: int = 0,
                       max_bonds: int = 100,
                       c_smiles: str = 'smiles',
                       c_bonds: str = 'n_bonds') -> pd.DataFrame:
    """
    Filter a DataFrame of molecules based on the number of non-hydrogen bonds.

    This function calculates the number of non-hydrogen bonds for each molecule
    in the DataFrame and filters the rows to include only those within the specified range.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing molecular data.
    min_bonds : int, optional
        The minimum number of non-hydrogen bonds to include. Default is 0.
    max_bonds : int, optional
        The maximum number of non-hydrogen bonds to include. Default is 100.
    c_smiles : str, optional
        The column name in the DataFrame containing SMILES strings. Default is 'smiles'.
    c_bonds : str, optional
        The column name to store the calculated number of non-hydrogen bonds. Default is 'n_bonds'.

    Returns
    -------
    pandas.DataFrame
        A filtered DataFrame containing only rows with the number of non-hydrogen bonds
        within the specified range.
    """
    df[c_bonds] = mp_calc(count_non_h_bonds, mp_calc(smi_to_mol, df[c_smiles]))
    return df[(df[c_bonds] >= min_bonds) & (df[c_bonds] <= max_bonds)].reset_index(drop=True)


def filter_by_mw(df: pd.DataFrame,
                 *,
                 min_mw: float = 0.0,
                 max_mw: float = 1000.0,
                 c_smiles: str = 'smiles',
                 c_mw: str = 'mw') -> pd.DataFrame:
    """
    Filter a DataFrame of molecules based on their molecular weight.

    This function calculates the molecular weight for each molecule in the DataFrame
    and filters the rows to include only those within the specified range.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing molecular data.
    min_mw : float, optional
        The minimum molecular weight to include. Default is 0.0.
    max_mw : float, optional
        The maximum molecular weight to include. Default is 1000.0.
    c_smiles : str, optional
        The column name in the DataFrame containing SMILES strings. Default is 'smiles'.
    c_mw : str, optional
        The column name to store the calculated molecular weight. Default is 'mw'.

    Returns
    -------
    pandas.DataFrame
        A filtered DataFrame containing only rows with molecular weights
        within the specified range.
    """
    df[c_mw] = mp_calc(molecular_weight, mp_calc(smi_to_mol, df[c_smiles]))
    return df[(df[c_mw] >= min_mw) & (df[c_mw] <= max_mw)].reset_index(drop=True)


def pubchem_name_to_smi(name: str) -> str:
    """
    Retrieve the SMILES string of a compound from PubChem by its name.

    Parameters
    ----------
    name : str
        The name of the compound to search for in PubChem.

    Returns
    -------
    str
        The canonical SMILES string of the compound.

    Raises
    ------
    pubchempy.PubChemHTTPError
        If there is an error communicating with PubChem.
    IndexError
        If no compounds match the given name.
    """
    return pcp.get_compounds(name, "name")[0].smiles


def pubchem_name_to_mol(name: str, add_hydrogens: bool = True, sanitize: bool = True) -> Chem.Mol:
    """
    Convert a compound name to an RDKit Mol object by querying PubChem.

    Parameters
    ----------
    name : str
        The name of the compound to search for in PubChem.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens to the molecule. Default is True.
    sanitize : bool, optional
        Whether to sanitize the molecule. Default is True.

    Returns
    -------
    rdkit.Chem.Mol
        The RDKit Mol object representing the compound.

    Raises
    ------
    pubchempy.PubChemHTTPError
        If the compound with the given name is not found.
    IndexError
        If no compounds match the given name.
    """
    smiles = pubchem_name_to_smi(name)
    return smi_to_mol(smiles, add_hydrogens=add_hydrogens, sanitize=sanitize)


def pubchem_name_to_nx(name: str, add_hydrogens: bool = True, sanitize: bool = True) -> nx.Graph:
    """
    Convert a compound name to a NetworkX Graph by querying PubChem.

    Parameters
    ----------
    name : str
        The name of the compound to search for in PubChem.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens to the molecule. Default is True.
    sanitize : bool, optional
        Whether to sanitize the molecule. Default is True.

    Returns
    -------
    networkx.Graph
        A NetworkX graph representation of the compound's molecular structure.

    Raises
    ------
    pubchempy.PubChemHTTPError
        If the compound with the given name is not found.
    IndexError
        If no compounds match the given name.
    """
    smiles = pubchem_name_to_smi(name)
    return smi_to_nx(smiles, add_hydrogens=add_hydrogens, sanitize=sanitize)


def pubchem_id_to_smi(id: int) -> str:
    """
    Retrieve the SMILES string of a compound from PubChem by its CID.

    Parameters
    ----------
    id : int
        The PubChem Compound ID (CID) of the compound.

    Returns
    -------
    str
        The canonical SMILES string of the compound.

    Raises
    ------
    pubchempy.PubChemHTTPError
        If the compound with the given CID is not found.
    """
    return pcp.Compound.from_cid(id).smiles


def pubchem_id_to_mol(id: int, add_hydrogens: bool = True, sanitize: bool = True) -> Chem.Mol:
    """
    Retrieve an RDKit Mol object of a compound from PubChem by its CID.

    Parameters
    ----------
    id : int
        The PubChem Compound ID (CID) of the compound.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens to the molecule. Default is True.
    sanitize : bool, optional
        Whether to sanitize the molecule. Default is True.

    Returns
    -------
    rdkit.Chem.Mol
        The RDKit Mol object representing the compound.

    Raises
    ------
    pubchempy.PubChemHTTPError
        If the compound with the given CID is not found.
    """
    smiles = pubchem_id_to_smi(id)
    return smi_to_mol(smiles, add_hydrogens=add_hydrogens, sanitize=sanitize)


def pubchem_id_to_nx(id: int, add_hydrogens: bool = True, sanitize: bool = True) -> nx.Graph:
    """
    Retrieve a NetworkX Graph of a compound from PubChem by its CID.

    Parameters
    ----------
    id : int
        The PubChem Compound ID (CID) of the compound.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens to the molecule. Default is True.
    sanitize : bool, optional
        Whether to sanitize the molecule. Default is True.

    Returns
    -------
    networkx.Graph
        A NetworkX graph representation of the compound's molecular structure.

    Raises
    ------
    pubchempy.PubChemHTTPError
        If the compound with the given CID is not found.
    """
    smiles = pubchem_id_to_smi(id)
    return smi_to_nx(smiles, add_hydrogens=add_hydrogens, sanitize=sanitize)


def sample_random_pubchem(
        n: int,
        *,
        seed: Optional[int] = None,
        max_cid: int = 123_431_215,
        delay_s: float = 0.01,
        max_attempts: int = 500_000,
        max_bonds: int = 100,
        batch_size: Optional[int] = None,
) -> Tuple[List[int], List[str]]:
    """
    Sample random valid molecules from PubChem by randomly selecting compound IDs.

    Parameters
    ----------
    n : int
        The number of valid molecules to sample.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    max_cid : int, optional
        Maximum PubChem Compound ID to sample from. Default is 123,431,215.
    delay_s : float, optional
        Delay in seconds between batch requests to avoid rate limiting. Default is 0.01.
    max_attempts : int, optional
        Maximum number of CID sampling attempts before raising an error. Default is 500,000.
    max_bonds : int, optional
        Maximum number of bonds allowed in a valid molecule. Default is 100.
    batch_size : int, optional
        Number of CIDs to query per batch. Defaults to n if not specified.

    Returns
    -------
    ids : List[int]
        List of PubChem CIDs for the sampled molecules.
    smi_list : List[str]
        List of canonical SMILES strings for the sampled molecules.

    Raises
    ------
    ValueError
        If batch_size is <= 0.
    RuntimeError
        If unable to collect n valid molecules within max_attempts.

    Notes
    -----
    - Molecules containing disconnected fragments (indicated by "." in SMILES) are excluded.
    - Each CID is only attempted once (no duplicates).
    """
    if n <= 0:
        return [], []
    if batch_size is None:
        batch_size = n

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    rng = random.Random(seed)

    smi_list: List[str] = []
    ids: List[int] = []
    seen: set[int] = set()

    attempts = 0

    def _is_valid_smiles(smi: str) -> bool:
        if not smi or "." in smi:
            return False
        mol = smi_to_mol(smi, sanitize=True, add_hydrogens=True)
        if mol is None:
            return False
        return mol.GetNumBonds() <= max_bonds

    while len(smi_list) < n:
        remaining = n - len(smi_list)
        target_gen = min(batch_size, max(remaining * 5, remaining))
        cids_batch: List[int] = []
        while len(cids_batch) < target_gen:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Only collected {len(smi_list)} valid molecules after {max_attempts} attempts."
                )

            cid = rng.randint(1, max_cid)
            if cid in seen:
                continue
            seen.add(cid)
            cids_batch.append(cid)

        try:
            compounds = pcp.get_compounds(cids_batch, "cid")
        except Exception:
            compounds = []

        for c in compounds:
            if len(smi_list) >= n:
                break

            cid = getattr(c, "cid", None)
            smi = getattr(c, "smiles", None)
            if cid is None or smi is None:
                continue

            try:
                if _is_valid_smiles(smi):
                    ids.append(int(cid))
                    smi_list.append(smi)
            except Exception:
                continue

        if delay_s:
            time.sleep(delay_s)

    return ids, smi_list


def sample_first_pubchem(
        n: int,
        *,
        start_cid: int = 1,
        max_cid: int = 123_431_215,
        delay_s: float = 0.01,
        max_attempts: int = 500_000,
        max_bonds: int = 100,
        batch_size: Optional[int] = None,
) -> Tuple[List[int], List[str]]:
    """
    Sample the first n valid molecules from PubChem starting from a given CID.

    Parameters
    ----------
    n : int
        The number of valid molecules to sample.
    start_cid : int, optional
        The PubChem Compound ID to start sampling from. Default is 1.
    max_cid : int, optional
        Maximum PubChem Compound ID to sample up to. Default is 123,431,215.
    delay_s : float, optional
        Delay in seconds between batch requests to avoid rate limiting. Default is 0.01.
    max_attempts : int, optional
        Maximum number of CID attempts before raising an error. Default is 500,000.
    max_bonds : int, optional
        Maximum number of bonds allowed in a valid molecule. Default is 100.
    batch_size : int, optional
        Number of CIDs to query per batch. Defaults to n if not specified.

    Returns
    -------
    ids : List[int]
        List of PubChem CIDs for the sampled molecules.
    smi_list : List[str]
        List of canonical SMILES strings for the sampled molecules.

    Raises
    ------
    ValueError
        If batch_size is <= 0 or start_cid is outside [1, max_cid].
    RuntimeError
        If unable to collect n valid molecules within max_attempts or max_cid is reached.

    Notes
    -----
    - Molecules containing disconnected fragments (indicated by "." in SMILES) are excluded.
    - CIDs are queried sequentially starting from start_cid.
    """
    if n <= 0:
        return [], []

    if batch_size is None:
        batch_size = n
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    if start_cid < 1 or start_cid > max_cid:
        raise ValueError("start_cid must be in [1, max_cid]")

    smi_list: List[str] = []
    ids: List[int] = []

    attempts = 0
    next_cid = start_cid

    def _is_valid_smiles(smi: str) -> bool:
        if not smi or "." in smi:
            return False
        mol = smi_to_mol(smi, sanitize=True, add_hydrogens=True)
        if mol is None:
            return False
        return mol.GetNumBonds() <= max_bonds

    while len(smi_list) < n:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Only collected {len(smi_list)} valid molecules after {max_attempts} attempts."
            )
        if next_cid > max_cid:
            raise RuntimeError(
                f"Reached max_cid={max_cid} after {attempts} attempts; collected {len(smi_list)} valid molecules."
            )

        remaining = n - len(smi_list)
        # Query enough sequential CIDs to have a decent chance of finding `remaining` valid ones.
        target = min(batch_size, max_cid - next_cid + 1)
        # Optional: you can be more aggressive like in the random sampler:
        # target = min(batch_size, max(remaining * 5, remaining), max_cid - next_cid + 1)

        cids_batch = list(range(next_cid, next_cid + target))
        next_cid += target
        attempts += len(cids_batch)

        try:
            compounds = pcp.get_compounds(cids_batch, "cid")
        except Exception:
            compounds = []

        for c in compounds:
            if len(smi_list) >= n:
                break

            cid = getattr(c, "cid", None)
            smi = getattr(c, "smiles", None)
            if cid is None or smi is None:
                continue

            try:
                if _is_valid_smiles(smi):
                    ids.append(int(cid))
                    smi_list.append(smi)
            except Exception:
                continue

        if delay_s:
            time.sleep(delay_s)

    return ids, smi_list


def download_pubchem_cid_smiles_gz(
        target_dir: str | os.PathLike = ".",
        url: str | None = None,
        filename: str = "CID-SMILES.gz",
        chunk_size: int = 1024 * 1024,
        overwrite: bool = False,
) -> Path:
    """
    Download the PubChem CID-SMILES mapping file.

    Parameters
    ----------
    target_dir : str or os.PathLike, optional
        Directory to save the downloaded file. Default is current directory.
    url : str or None, optional
        URL to download from. Default is the official PubChem FTP URL.
    filename : str, optional
        Name of the output file. Default is "CID-SMILES.gz".
    chunk_size : int, optional
        Size of chunks to read during download in bytes. Default is 1MB.
    overwrite : bool, optional
        Whether to overwrite existing file. Default is False.

    Returns
    -------
    Path
        Path to the downloaded file.

    Notes
    -----
    The default URL points to the PubChem FTP server containing all
    compound IDs mapped to their canonical SMILES strings.
    """
    if url is None:
        url = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / filename

    if out_path.exists() and not overwrite:
        print(f"File already exists: {out_path}", flush=True)
        return out_path

    req = Request(url, headers={"User-Agent": "python-download/1.0"})

    with urlopen(req) as resp, open(out_path, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)

    return out_path


def sample_pubchem_cid_smiles_gz(
        n: int,
        *,
        gz_path: str = 'CID-SMILES.gz',
        seed: int = 0,
        sep: str = "\t",
        max_bonds: int = 100,
) -> Tuple[List[int], List[str]]:
    """
    Sample random CID-SMILES pairs from a downloaded PubChem gzip file.

    Parameters
    ----------
    n : int
        Number of samples to draw from the file.
    gz_path : str, optional
        Path to the gzipped CID-SMILES file. Default is "CID-SMILES.gz".
    seed : int, optional
        Random seed for reproducibility. Default is 0.
    sep : str, optional
        Column separator in the file. Default is tab character.
    max_bonds : int, optional
        Maximum number of bonds allowed in a valid molecule. Default is 100.

    Returns
    -------
    cids : List[int]
        List of PubChem Compound IDs.
    smiles : List[str]
        List of corresponding SMILES strings.

    Notes
    -----
    If n exceeds the number of valid rows in the file, all available valid rows are returned.
    Bad lines in the file are skipped during parsing.
    Molecules containing disconnected fragments (indicated by "." in SMILES) are excluded.
    """
    df = pd.read_csv(
        gz_path,
        compression="gzip",
        sep=sep,
        header=None,
        names=["cid", "smiles"],
        dtype={"cid": "int64", "smiles": "string"},
        on_bad_lines="skip",
    )

    # Filter out SMILES with disconnected fragments
    df = df[~df['smiles'].str.contains(r'\.', na=True, regex=True)]

    # Sample more than needed to account for invalid molecules
    sample_size = min(n * 3, len(df))
    sampled = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    cids: List[int] = []
    smiles_list: List[str] = []

    for _, row in sampled.iterrows():
        if len(cids) >= n:
            break
        smi = row['smiles']
        try:
            mol = smi_to_mol(smi)
            mol = standardize_mol(mol)
            if mol is not None and mol.GetNumBonds() <= max_bonds:
                cids.append(int(row['cid']))
                smiles_list.append(smi)
        except Exception:
            continue

    return cids, smiles_list


def _valid(s):
    try:
        mol = smi_to_mol(s)
        standardize_mol(mol)
    except Exception:
        return False
    return True


def _valid_smi(smi: str) -> bool:
    return bool(smi) and all(x not in smi for x in [".", "*", "->", "$"])


def _valid_mol(smi: str) -> float:
    try:
        mol = smi_to_mol(smi)
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol)
        return Chem.Descriptors.MolWt(mol)
    except Exception:
        return 0.0


def sample_pubchem_cid_smiles_gz_mw(
        n: int,
        *,
        gz_path: str = 'CID-SMILES.gz',
        out_file: str = 'sampled_cid_smiles_mw.csv.gz',
        seed: int = 0,
        sep: str = "\t",
        max_mw: float = 600.0,
        max_bonds: int = 100,
) -> pd.DataFrame:
    if os.path.exists(out_file):
        print(f"Loading existing sampled file: {out_file}", flush=True)
        return pd.read_csv(out_file, compression="gzip")
    else:
        print(f"Creating sampled file: {out_file}", flush=True)
        df = pd.read_csv(
            gz_path,
            compression="gzip",
            sep=sep,
            header=None,
            names=["cid", "smiles"],
            dtype={"cid": "int64", "smiles": "string"},
            on_bad_lines="skip",
        )
        print(f"Total number of molecules in PubChem: {len(df)}", flush=True)
        df = df.sample(n=min(n * 3, len(df)), random_state=seed).reset_index(drop=True)
        # Remove rows with invalid SMILES
        print("Filtering invalid SMILES...", flush=True)
        df = df[mp_calc(_valid_smi, df['smiles'])]
        # Calculate molecular weights
        print('Filtering mols', flush=True)
        df['mw'] = mp_calc(_valid_mol, df['smiles'])
        # Filter out the SMILES with mw > max_mw
        df = df[(df['mw'] <= max_mw) & (df['mw'] > 0.0)]
        # Calculate number of bonds
        df['n_bonds'] = mp_calc(count_non_h_bonds, mp_calc(smi_to_mol, df['smiles']))
        # Filter out the SMILES with too many bonds
        df = df[df['n_bonds'] <= max_bonds]
        # Sample n rows
        sampled = df.sample(n=n, random_state=seed).reset_index(drop=True)
        print(f"Total number of molecules in PubChem: {len(sampled)}", flush=True)
        sampled.to_csv(out_file, index=False, compression='gzip')
        return sampled
