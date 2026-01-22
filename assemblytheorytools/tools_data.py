import os
import random
import re
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


def filter_by_bonds(df: pd.DataFrame,
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


def load_ir_jcamp_data(path: str) -> np.ndarray:
    """
    Parse IR JCAMP-DX data from a file and return it as a NumPy array.

    This function reads a JCAMP-DX file containing infrared (IR) spectroscopy data,
    extracts the frequency and intensity values, and returns them as a 2D NumPy array.

    Parameters
    ----------
    path : str
        The file path to the JCAMP-DX file.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (N, 2), where each row contains a frequency and its
        corresponding intensity.

    Raises
    ------
    ValueError
        If no valid XY data block is found or if required metadata (e.g., DELTAX) is missing.

    Notes
    -----
    - The function supports two data modes: "xy_pairs" and "xpp_ylist".
    - Frequencies and intensities are scaled by XFACTOR and YFACTOR, respectively.
    - If the number of points (NPOINTS) is specified, the output is truncated to that length.
    """
    xfactor = 1.0  # Scaling factor for frequencies
    yfactor = 1.0  # Scaling factor for intensities
    firstx: Optional[float] = None  # First frequency value
    lastx: Optional[float] = None  # Last frequency value
    deltax: Optional[float] = None  # Frequency step size
    npoints: Optional[int] = None  # Number of data points
    in_data = False  # Flag to indicate if data parsing is active
    data_mode: Optional[str] = None  # Mode of data representation
    frequencies = []  # List to store frequency values
    intensities = []  # List to store intensity values

    # Regular expressions for parsing metadata and numerical values
    keyval_re = re.compile(r"^##\s*([^=]+)\s*=\s*(.*)\s*$")
    float_re = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")
    int_re = re.compile(r"[-+]?\d+")

    def _extract_numbers(line: str):
        """
        Extract numerical values from a line of text.

        Parameters
        ----------
        line : str
            The input line of text.

        Returns
        -------
        List[float]
            A list of extracted floating-point numbers.
        """
        line = line.replace(",", " ").replace(";", " ")
        return [float(x) for x in float_re.findall(line)]

    def _parse_float(s: str) -> Optional[float]:
        """
        Parse a floating-point number from a string.

        Parameters
        ----------
        s : str
            The input string.

        Returns
        -------
        Optional[float]
            The parsed floating-point number, or None if parsing fails.
        """
        m = float_re.search(s)
        return float(m.group(0)) if m else None

    def _parse_int(s: str) -> Optional[int]:
        """
        Parse an integer from a string.

        Parameters
        ----------
        s : str
            The input string.

        Returns
        -------
        Optional[int]
            The parsed integer, or None if parsing fails.
        """
        m = int_re.search(s)
        return int(m.group(0)) if m else None

    # Keys indicating the start of data blocks
    data_keys = {"XYDATA", "XYPOINTS", "DATA TABLE", "DATATABLE"}

    # Open the JCAMP-DX file and parse its contents
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Skip comment lines
            if line.startswith("$$"):
                continue

            # End of the file
            if line.upper().startswith("##END"):
                break

            # Metadata lines
            if line.startswith("##"):
                m = keyval_re.match(line)
                if not m:
                    continue

                key = m.group(1).strip().upper()
                val = m.group(2).strip()

                # Check if the line starts a data block
                if key in data_keys:
                    in_data = True
                    uval = val.upper()
                    data_mode = "xpp_ylist" if "X++" in uval else "xy_pairs"
                    continue

                # End of a data block
                if in_data:
                    in_data = False
                    data_mode = None

                # Parse metadata values
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

            # Skip lines outside data blocks
            if not in_data or data_mode is None:
                continue

            # Extract numerical data
            nums = _extract_numbers(line)
            if not nums:
                continue

            # Parse data based on the mode
            if data_mode == "xy_pairs":
                # Data is in (x, y) pairs
                if len(nums) % 2 == 1:
                    nums = nums[:-1]
                for i in range(0, len(nums), 2):
                    frequencies.append(nums[i] * xfactor)
                    intensities.append(nums[i + 1] * yfactor)

            elif data_mode == "xpp_ylist":
                # Data is in x++(y..y) format
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

    # Raise an error if no data was found
    if not frequencies:
        raise ValueError("No XY data block found (expected ##XYDATA= or ##XYPOINTS=).")

    # Truncate data to the specified number of points
    if npoints is not None and len(frequencies) > npoints:
        frequencies = frequencies[:npoints]
        intensities = intensities[:npoints]

    # Return Nx2 array: [frequency, intensity]
    return np.column_stack((np.asarray(frequencies, dtype=float),
                            1.0-np.asarray(intensities, dtype=float)))


def find_peak_indices_in_range(
        xy: np.ndarray,
        f_min: float,
        f_max: float,
        *,
        prominence: Optional[float] = None,
        min_distance: Optional[float] = None,
) -> List[int]:
    """
    Identify the indices of peaks within a specified frequency range in a 2D array.

    This function analyzes a 2D array of frequency and intensity data to find peaks
    within a given frequency range. Peaks are defined as local maxima that satisfy
    optional prominence and minimum distance criteria.

    Parameters
    ----------
    xy : np.ndarray
        A 2D NumPy array of shape (N, 2), where each row contains a frequency and its
        corresponding intensity.
    f_min : float
        The lower bound of the frequency range to search for peaks.
    f_max : float
        The upper bound of the frequency range to search for peaks.
    prominence : float, optional
        The minimum required prominence of a peak. Peaks with prominence less than this
        value are ignored. Default is None.
    min_distance : float, optional
        The minimum required distance (in frequency units) between peaks. Peaks closer
        than this distance are filtered, keeping the most prominent ones. Default is None.

    Returns
    -------
    List[int]
        A list of indices corresponding to the peaks that satisfy the criteria.

    Raises
    ------
    ValueError
        If the input array `xy` is not a 2D array of shape (N, 2).

    Notes
    -----
    - A peak is defined as a point where the intensity is greater than the intensity
      of its immediate neighbors.
    - If `prominence` is specified, only peaks with sufficient prominence are included.
    - If `min_distance` is specified, peaks are filtered to ensure a minimum spacing
      between them, with the most prominent peaks retained.
    """
    xy = np.asarray(xy)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be a 2D array of shape (N, 2): [freq, intensity].")

    n = xy.shape[0]
    if n < 3:
        return []

    freqs = xy[:, 0]
    intens = xy[:, 1]

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

    # Optional: minimum distance filtering (by frequency spacing)
    if min_distance is not None and min_distance > 0:
        by_height = sorted(candidates, key=lambda i: intens[i], reverse=True)
        kept: List[int] = []
        for i in by_height:
            fi = freqs[i]
            if all(abs(fi - freqs[j]) >= min_distance for j in kept):
                kept.append(i)
        return sorted(kept)

    return candidates


def calc_n_peaks_in_range(data: np.ndarray,
                          f_min: float = 500,
                          f_max: float = 1500,
                          prominence: Optional[float] = None,
                          min_distance: Optional[float] = None) -> int:
    """
    Calculate the number of peaks within a specified frequency range in a 2D array.

    This function identifies peaks in the given frequency-intensity data within the
    specified frequency range and returns the count of such peaks. Peaks can be filtered
    based on optional prominence and minimum distance criteria.

    Parameters
    ----------
    data : np.ndarray
        A 2D NumPy array of shape (N, 2), where each row contains a frequency and its
        corresponding intensity.
    f_min : float, optional
        The lower bound of the frequency range to search for peaks. Default is 500.
    f_max : float, optional
        The upper bound of the frequency range to search for peaks. Default is 1500.
    prominence : float, optional
        The minimum required prominence of a peak. Peaks with prominence less than this
        value are ignored. Default is None.
    min_distance : float, optional
        The minimum required distance (in frequency units) between peaks. Peaks closer
        than this distance are filtered, keeping the most prominent ones. Default is None.

    Returns
    -------
    int
        The number of peaks that satisfy the specified criteria.
    """
    locs = find_peak_indices_in_range(data,
                                      f_min=f_min,
                                      f_max=f_max,
                                      prominence=prominence,
                                      min_distance=min_distance)
    return len(locs)
