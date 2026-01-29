import json
import os
import random
import re
import shutil
import tarfile
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union
from urllib.request import Request, urlopen

import networkx as nx
import numpy as np
import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from scipy.optimize import minimize
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import gaussian_kde

from .complexity_scores import count_bonds, count_non_h_bonds, molecular_weight
from .tools_file import file_list_all
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


def pubchem_smi_to_name(
            smiles: str,
            prefer: Tuple[str, ...] = ("synonym", "iupac_name", "title"),
            timeout: int = 20,
    ) -> Optional[str]:
        """
        Retrieve the name of a compound from PubChem using its SMILES string.

        This function queries the PubChem database with a given SMILES string and attempts
        to retrieve a human-readable name for the compound. The name is selected based on
        a preference order of fields such as synonyms, IUPAC name, or title.

        Parameters
        ----------
        smiles : str
            The SMILES string of the compound to search for.
        prefer : Tuple[str, ...], optional
            A tuple specifying the preferred fields to retrieve the name from, in order of priority.
            Default is ("synonym", "iupac_name", "title").
        timeout : int, optional
            The timeout in seconds for the PubChem query. Default is 20.

        Returns
        -------
        Optional[str]
            The name of the compound if found, otherwise None.

        Notes
        -----
        - The function uses the `pubchempy` library to query the PubChem database.
        - If the SMILES string is invalid or no matching compound is found, the function returns None.
        - The function includes a scoring mechanism to select the most appropriate synonym
          if multiple options are available.

        Raises
        ------
        Exception
            If there is an error during the PubChem query, the function handles it and returns None.
        """
        smiles = (smiles or "").strip()
        if not smiles:
            return None
        try:
            comps = pcp.get_compounds(smiles, namespace="smiles", timeout=timeout)
        except Exception:
            return None

        if not comps:
            return None

        c = comps[0]

        # Helper to decide if something looks like a "nice" synonym
        def _synonym_score(name: str) -> int:
            """
            Calculate a score for a synonym to determine its suitability.

            Parameters
            ----------
            name : str
                The synonym to evaluate.

            Returns
            -------
            int
                A score indicating the quality of the synonym. Higher scores are better.
            """
            n = name.strip()
            if not n:
                return -10

            # Penalize very long names or names that look like systematic strings
            score = 0
            if len(n) <= 30:
                score += 5
            elif len(n) <= 60:
                score += 1
            else:
                score -= 5

            # Prefer names with letters and spaces; penalize lots of punctuation/digits
            if re.search(r"[A-Za-z]", n):
                score += 2
            if re.search(r"\d", n):
                score -= 1
            if re.search(r"[{}[\]=#@]", n):  # SMILES-ish / formula-ish characters
                score -= 4
            if "," in n or ";" in n:
                score -= 2

            # Penalize names that look like full IUPAC (lots of hyphens/parentheses)
            if n.count("-") >= 3 or n.count("(") >= 2:
                score -= 2

            return score

        # Try preferred fields
        for field in prefer:
            if field == "iupac_name":
                v = getattr(c, "iupac_name", None)
                if v:
                    return v.strip()

            elif field == "title":
                v = getattr(c, "title", None)
                if v:
                    return v.strip()

            elif field == "synonym":
                syns = getattr(c, "synonyms", None) or []
                if syns:
                    # pick the best-looking synonym
                    best = max(syns, key=_synonym_score)
                    if _synonym_score(best) > 0:
                        return best.strip()

        return None


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


def _valid_mol(s):
    """
    Validate a SMILES string by attempting to convert and standardize it.

    This function checks if a given SMILES string can be successfully converted
    into an RDKit Mol object and standardized. If any step fails, the SMILES
    string is considered invalid.

    Parameters
    ----------
    s : str
        The SMILES string to validate.

    Returns
    -------
    bool
        True if the SMILES string is valid and can be standardized, otherwise False.
    """
    try:
        mol = smi_to_mol(s)
        standardize_mol(mol)
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol)
    except Exception:
        return False
    return True


def _valid_smi(smi: str) -> bool:
    """
    Validate a SMILES string.

    This function checks if the given SMILES string is non-empty and does not contain
    any invalid characters such as ".", "*", "->", or "$".

    Parameters
    ----------
    smi : str
        The SMILES string to validate.

    Returns
    -------
    bool
        True if the SMILES string is valid, otherwise False.
    """
    return bool(smi) and all(x not in smi for x in [".", "*", "->", "$"])


def _valid_mol_mw(smi: str) -> float:
    """
    Validate a SMILES string and calculate its molecular weight.

    This function converts a SMILES string into an RDKit Mol object, sanitizes it,
    kekulizes it, and calculates its molecular weight. If any step fails, the function
    returns 0.0.

    Parameters
    ----------
    smi : str
        The SMILES string representing the molecule.

    Returns
    -------
    float
        The molecular weight of the molecule if valid, otherwise 0.0.
    """
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
    """
    Sample random CID-SMILES pairs from a PubChem gzip file and filter them by molecular weight and bond count.

    This function reads a gzipped file containing CID-SMILES mappings, filters the molecules based on their
    molecular weight and number of bonds, and saves the sampled data to a compressed CSV file.

    Parameters
    ----------
    n : int
        Number of samples to draw from the file.
    gz_path : str, optional
        Path to the gzipped CID-SMILES file. Default is 'CID-SMILES.gz'.
    out_file : str, optional
        Path to save the sampled data as a compressed CSV file. Default is 'sampled_cid_smiles_mw.csv.gz'.
    seed : int, optional
        Random seed for reproducibility. Default is 0.
    sep : str, optional
        Column separator in the file. Default is tab character.
    max_mw : float, optional
        Maximum molecular weight allowed for a molecule. Default is 600.0.
    max_bonds : int, optional
        Maximum number of bonds allowed for a molecule. Default is 100.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the sampled and filtered CID-SMILES pairs.

    Notes
    -----
    - If the output file already exists, the function loads and returns its contents.
    - Invalid SMILES strings and molecules exceeding the specified molecular weight or bond count are excluded.
    - The function samples more rows than required to account for filtering, ensuring the desired number of valid samples.
    """
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
        df['molecular_weight'] = mp_calc(_valid_mol_mw, df['smiles'])
        # Filter out the SMILES with mw > max_mw
        df = df[(df['molecular_weight'] <= max_mw) & (df['molecular_weight'] > 0.0)]
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
                            1.0 - np.asarray(intensities, dtype=float)))


def _process_meta_data_name(entry: List[dict]) -> Optional[str]:
    """
    Extract the name identifier from the metadata entry.

    This function processes a list of metadata dictionaries to locate the first
    entry that does not correspond to a '.peak.jdx' file and extracts the identifier
    from its path.

    Parameters
    ----------
    entry : List[dict]
        A list of dictionaries representing metadata entries. Each dictionary is
        expected to contain an 'attacments' key, which is a list of dictionaries
        with 'filename' and 'identifier' keys.

    Returns
    -------
    Optional[str]
        The extracted name identifier if found, otherwise None.

    Raises
    ------
    IndexError, KeyError, TypeError
        If the structure of the input data does not match the expected format.
    """
    try:
        entries = entry[0]['attacments']
        for e in entries:
            if not e['filename'].endswith('.peak.jdx'):
                return e['identifier'].split('/')[-1]
    except (IndexError, KeyError, TypeError):
        pass
    return None


def _process_chemotion_meta_section(extract_dir: str) -> pd.DataFrame:
    """
    Process the metadata section from the extracted Chemotion data.

    This function locates the metadata file in the specified extraction directory,
    reads its contents, and processes the data to extract relevant information.
    The resulting DataFrame contains SMILES strings and corresponding names.

    Parameters
    ----------
    extract_dir : str
        The directory where the Chemotion data has been extracted.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the following columns:
        - 'smiles': The canonical SMILES strings of the molecules.
        - 'name': The processed names extracted from the metadata.

    Raises
    ------
    FileNotFoundError
        If the metadata file ('meta_data.json') is not found in the extraction directory.
    """
    # Locate the metadata file
    meta_file = next((f for f in file_list_all(extract_dir) if f.endswith("meta_data.json")), None)
    if not meta_file:
        raise FileNotFoundError("No meta_data.json file found in extracted data.")

    # Read the metadata file
    with open(meta_file, "r") as f:
        meta_data = json.load(f)

    # Convert to pandas DataFrame and select relevant columns
    df = pd.DataFrame(meta_data)[['cano_smiles', 'datasets']]
    df = df.rename(columns={'cano_smiles': 'smiles'})

    # Drop entries with invalid SMILES strings
    df = df[mp_calc(_valid_smi, df['smiles'])]

    # Process 'datasets' to extract 'name' and clean up the DataFrame
    df['name'] = mp_calc(_process_meta_data_name, df['datasets'])
    df = df.dropna(subset=['name']).drop(columns=['datasets'])

    return df


def _process_chemotion_ir_section(extract_dir: str, meta_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process the IR data section from the extracted Chemotion data.

    This function locates the IR data archive within the specified extraction directory,
    extracts the IR data if not already extracted, filters the IR files based on the
    metadata names, and creates a DataFrame containing the filenames and their corresponding
    spectra.

    Parameters
    ----------
    extract_dir : str
        The directory where the Chemotion data has been extracted.
    meta_data : pd.DataFrame
        A pandas DataFrame containing metadata, including a 'name' column used to filter
        the IR files.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with two columns:
        - 'name': The filenames of the IR data files.
        - 'spectrum': The corresponding spectra loaded from the IR files.

    Raises
    ------
    FileNotFoundError
        If the IR data archive ('IR_data.tar.xz') is not found in the extraction directory.
    """
    # Locate the IR data archive
    ir_file = next((f for f in file_list_all(extract_dir) if f.endswith("IR_data.tar.xz")), None)
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
        f for f in file_list_all(ir_extract_dir)
        if any(name in f for name in target_names)
    ]
    filenames = [os.path.basename(f) for f in ir_files]

    # Create a DataFrame with filenames and their corresponding spectra
    ir_data = pd.DataFrame({'name': filenames})
    ir_data['spectrum'] = mp_calc(load_ir_jcamp_data, ir_files)
    return ir_data


def process_chemotion_ir_data(target_file: str) -> pd.DataFrame:
    """
    Process Chemotion IR data from a tar file and return it as a pandas DataFrame.

    This function extracts metadata and IR spectra from a Chemotion tar file, processes
    the data, and merges the metadata with the corresponding IR spectra. The resulting
    DataFrame is saved to a compressed CSV file for future use.

    Parameters
    ----------
    target_file : str
        Path to the Chemotion tar file containing the IR data and metadata.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the merged metadata and IR spectra.

    Notes
    -----
    - If the processed data file (`chemotion_ir_data.csv.gz`) already exists, the function
      skips processing and loads the data directly from the file.
    - The function extracts the tar file to a directory named `chemotion_ir_data` in the
      same location as the `target_file`.
    - Rows with any missing values are dropped from the final DataFrame.
    """
    extract_dir = os.path.join(os.path.dirname(target_file), "chemotion_ir_data")
    out_file = "chemotion_ir_data.csv.gz"

    # Check if the processed data file already exists
    if os.path.exists(out_file):
        print(f"{out_file} already exists. Skipping processing.", flush=True)
        return pd.read_csv(out_file)

    # Extract the tar file if the extraction directory does not exist
    if not os.path.exists(extract_dir):
        with tarfile.open(target_file, "r") as tar:
            tar.extractall(path=extract_dir)
        print(f"Extracted data to {extract_dir}", flush=True)

    # Process metadata and IR data sections
    meta_data = _process_chemotion_meta_section(extract_dir)
    ir_data = _process_chemotion_ir_section(extract_dir, meta_data)

    # Merge metadata and IR data on the 'name' column
    merged_data = pd.merge(meta_data, ir_data, on='name')
    # Drop rows with any NaN values
    merged_data = merged_data.dropna()
    # Save the merged data to a compressed CSV file
    merged_data.to_csv(out_file, index=False)
    return merged_data


def find_peak_indices_in_range(
        xy: np.ndarray,
        min_x: float,
        max_x: float,
        *,
        prominence: Optional[float] = None,
        distance: Optional[float] = None,
) -> np.ndarray:
    """
    Find the indices of peaks within a specified x-range in a 2D spectrum.

    This function identifies peaks in the intensity values of a 2D array
    representing a spectrum and filters them to include only those within
    the specified x-range.

    Parameters
    ----------
    xy : np.ndarray
        A 2D NumPy array of shape (N, 2), where the first column represents
        the x-values (e.g., frequencies) and the second column represents
        the y-values (e.g., intensities).
    min_x : float
        The minimum x-value of the range to include peaks.
    max_x : float
        The maximum x-value of the range to include peaks.
    prominence : float, optional
        The required prominence of peaks. This parameter is passed to
        `scipy.signal.find_peaks`. Default is None.
    distance : float, optional
        The required minimum horizontal distance (in number of samples)
        between neighboring peaks. This parameter is passed to
        `scipy.signal.find_peaks`. Default is None.

    Returns
    -------
    np.ndarray
        A 1D NumPy array containing the indices of the peaks that fall
        within the specified x-range.

    Raises
    ------
    ValueError
        If the input `xy` is not a 2D array of shape (N, 2).

    Notes
    -----
    - The function uses `scipy.signal.find_peaks` to detect peaks in the
      intensity values (second column of `xy`).
    - The x-values of the detected peaks are filtered to include only those
      within the range [min_x, max_x].
    """
    xy = np.asarray(xy)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be a 2D array of shape (N, 2): [freq, intensity].")

    peaks = find_peaks(xy.T[1], prominence=prominence, distance=distance)[0]
    # Filter peaks to be within the specified x-range using boolean indexing
    return peaks[(xy[peaks, 0] >= min_x) & (xy[peaks, 0] <= max_x)]


def find_n_peak_indices_in_range(
        xy: np.ndarray,
        min_x: float,
        max_x: float,
        *,
        prominence: Optional[float] = None,
        distance: Optional[float] = None,
) -> int:
    """
    Count the number of peaks within a specified x-range in a 2D spectrum.

    This function identifies peaks in the intensity values of a 2D array
    representing a spectrum, filters them to include only those within
    the specified x-range, and returns the count of such peaks.

    Parameters
    ----------
    xy : np.ndarray
        A 2D NumPy array of shape (N, 2), where the first column represents
        the x-values (e.g., frequencies) and the second column represents
        the y-values (e.g., intensities).
    min_x : float
        The minimum x-value of the range to include peaks.
    max_x : float
        The maximum x-value of the range to include peaks.
    prominence : float, optional
        The required prominence of peaks. This parameter is passed to
        `scipy.signal.find_peaks`. Default is None.
    distance : float, optional
        The required minimum horizontal distance (in number of samples)
        between neighboring peaks. This parameter is passed to
        `scipy.signal.find_peaks`. Default is None.

    Returns
    -------
    int
        The number of peaks that fall within the specified x-range.

    Notes
    -----
    - This function uses `find_peak_indices_in_range` to identify the indices
      of peaks within the specified range and then calculates their count.
    """
    peak_indices = find_peak_indices_in_range(
        xy,
        min_x,
        max_x,
        prominence=prominence,
        distance=distance,
    )
    return len(peak_indices)


def apply_sg_filter(spectrum, window_length=11, polyorder=3):
    """
    Apply a Savitzky-Golay filter to smooth the intensity values of a spectrum.

    This function takes a 2D array representing a spectrum, applies a Savitzky-Golay
    filter to the intensity values, and returns the smoothed spectrum.

    Parameters
    ----------
    spectrum : array-like
        A 2D array where the first column represents the x-values (e.g., frequencies)
        and the second column represents the y-values (e.g., intensities).
    window_length : int, optional
        The length of the filter window (number of coefficients). Must be a positive odd integer.
        Default is 11.
    polyorder : int, optional
        The order of the polynomial used to fit the samples. Must be less than `window_length`.
        Default is 3.

    Returns
    -------
    np.ndarray
        A 2D NumPy array with the same shape as the input, where the first column contains
        the original x-values and the second column contains the smoothed intensity values.
    """
    intensity = savgol_filter(spectrum.T[1], window_length=window_length, polyorder=polyorder)
    return np.column_stack((np.asarray(spectrum.T[0], dtype=float), np.asarray(intensity, dtype=float)))


def linear_func(x, m, b):
    """
    Linear function to model a straight line.

    Parameters:
    -----------
    x : float or array-like
        The independent variable.
    m : float
        The slope of the line.
    b : float
        The y-intercept of the line.

    Returns:
    --------
    float or array-like
        The dependent variable calculated as m * x + b.
    """
    return m * x + b


def quadratic_func(x, a, b, c):
    """
    Quadratic function to model a parabola.

    Parameters:
    -----------
    x : float or array-like
        The independent variable.
    a : float
        The coefficient for the quadratic term (x^2).
    b : float
        The coefficient for the linear term (x).
    c : float
        The constant term.

    Returns:
    --------
    float or array-like
        The dependent variable calculated as a * x^2 + b * x + c.
    """
    return a * x ** 2 + b * x + c


def cubic_func(x, a, b, c, d):
    """
    Cubic function to model a polynomial of degree 3.

    Parameters:
    -----------
    x : float or array-like
        The independent variable.
    a : float
        The coefficient for the cubic term (x^3).
    b : float
        The coefficient for the quadratic term (x^2).
    c : float
        The coefficient for the linear term (x).
    d : float
        The constant term.

    Returns:
    --------
    float or array-like
        The dependent variable calculated as a * x^3 + b * x^2 + c * x + d.
    """
    return a * x ** 3 + b * x ** 2 + c * x + d


def quartic_func(x, a, b, c, d, e):
    """
    Quartic function to model a polynomial of degree 4.

    Parameters:
    -----------
    x : float or array-like
        The independent variable.
    a : float
        The coefficient for the quartic term (x^4).
    b : float
        The coefficient for the cubic term (x^3).
    c : float
        The coefficient for the quadratic term (x^2).
    d : float
        The coefficient for the linear term (x).
    e : float
        The constant term.

    Returns:
    --------
    float or array-like
        The dependent variable calculated as a * x^4 + b * x^3 + c * x^2 + d * x + e.
    """
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e


def quintic_func(x, a, b, c, d, e):
    """
    Quintic function to model a polynomial of degree 5.

    Parameters:
    -----------
    x : float or array-like
        The independent variable.
    a : float
        The coefficient for the quintic term (x^5).
    b : float
        The coefficient for the quartic term (x^4).
    c : float
        The coefficient for the cubic term (x^3).
    d : float
        The coefficient for the quadratic term (x^2).
    e : float
        The coefficient for the linear term (x).

    Returns:
    --------
    float or array-like
        The dependent variable calculated as a * x^5 + b * x^4 + c * x^3 + d * x^2 + e * x.
    """
    return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x


def get_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Pearson correlation coefficient (r) between two arrays.

    This function computes the Pearson correlation coefficient, which measures
    the linear relationship between two datasets. The result is a value between
    -1 and 1, where 1 indicates a perfect positive linear relationship, -1 indicates
    a perfect negative linear relationship, and 0 indicates no linear relationship.

    Parameters
    ----------
    y_true : np.ndarray
        The ground truth (true) values as a NumPy array.
    y_pred : np.ndarray
        The predicted values as a NumPy array.

    Returns
    -------
    float
        The Pearson correlation coefficient. If the denominator is zero (e.g., when
        the variance of one or both arrays is zero), the function returns NaN.

    Notes
    -----
    - The input arrays are flattened to 1D before computation.
    - The function handles cases where the denominator is close to zero to avoid
      division by zero errors.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    yt = y_true - np.mean(y_true)
    yp = y_pred - np.mean(y_pred)
    denom = np.sqrt(np.sum(yt ** 2) * np.sum(yp ** 2))
    if np.isclose(denom, 0.0):
        return np.nan
    return float(np.sum(yt * yp) / denom)


def get_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the coefficient of determination (R^2) between two arrays.

    This function computes the R^2 score, which measures how well the predicted
    values approximate the true values. The R^2 score is defined as:
    R^2 = 1 - (SS_res / SS_tot), where:
    - SS_res is the sum of squared residuals.
    - SS_tot is the total sum of squares.

    Parameters
    ----------
    y_true : np.ndarray
        The ground truth (true) values as a NumPy array.
    y_pred : np.ndarray
        The predicted values as a NumPy array.

    Returns
    -------
    float
        The R^2 score. A value of 1.0 indicates a perfect fit, 0.0 indicates
        that the model does not explain any of the variance, and negative values
        indicate that the model performs worse than a horizontal line.

    Notes
    -----
    - The input arrays are flattened to 1D before computation.
    - Handles degenerate cases where the total variance (SS_tot) is zero.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # Handle the degenerate case
    if np.isclose(ss_tot, 0.0):
        return 1.0 if np.isclose(ss_res, 0.0) else 0.0
    return 1.0 - (ss_res / ss_tot)


def get_rmsd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Root Mean Square Deviation (RMSD) between two arrays.

    This function computes the RMSD, which is a measure of the differences
    between values predicted by a model and the values actually observed.
    It is commonly used to evaluate the accuracy of predictions.

    Parameters
    ----------
    y_true : np.ndarray
        The ground truth (true) values as a NumPy array.
    y_pred : np.ndarray
        The predicted values as a NumPy array.

    Returns
    -------
    float
        The RMSD value, which is a non-negative float. A lower value indicates
        better agreement between the true and predicted values.

    Notes
    -----
    - The input arrays are flattened to 1D before computation.
    - The RMSD is calculated as the square root of the mean of the squared differences.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _peaks_to_ai(n_peaks, model, params):
    """
    Convert the number of peaks to an assembly index (AI) using a given model.

    This function applies a mathematical model to the number of peaks and
    model parameters to calculate the corresponding assembly index.

    Parameters
    ----------
    n_peaks : int
        The number of peaks in the spectrum.
    model : callable
        A mathematical model function that takes the number of peaks and
        model parameters as input and returns a predicted value.
    params : list or tuple
        The parameters to be passed to the model function.

    Returns
    -------
    int
        The calculated assembly index (AI) as an integer.
    """
    return int(model(n_peaks, *params))


def _func_min_helper(x, *args):
    """
    Helper function for minimizing the root mean square deviation (RMSD).

    This function calculates the RMSD between observed assembly indices and
    predicted assembly indices based on a given model and parameters. It is
    used as the objective function in optimization routines.

    Parameters
    ----------
    x : array-like
        The parameters to be optimized for the model.
    *args : tuple
        Additional arguments passed to the function:
        - n_peaks : array-like
            The number of peaks in the spectrum.
        - obs : array-like
            The observed assembly indices.
        - model_fit : callable
            The model function used to predict assembly indices.

    Returns
    -------
    float
        The RMSD between the observed and predicted assembly indices.
    """
    n_peaks, obs, model_fit = args
    pred = np.array([_peaks_to_ai(n, model_fit, x) for n in n_peaks], dtype=int)
    return get_rmsd(obs, pred)


def estimate_ai_from_ir_peaks(peaks_data,
                              ai_obs,
                              model,
                              params_0):
    """
    Estimate assembly indices (AI) from IR peaks using a given model.

    This function optimizes the parameters of a model to minimize the root mean square
    deviation (RMSD) between observed assembly indices and predicted assembly indices.
    The optimized parameters and the predicted assembly indices are returned.

    Parameters
    ----------
    peaks_data : array-like
        The input data representing the number of peaks in the spectrum.
    ai_obs : array-like
        The observed assembly indices corresponding to the input data.
    model : callable
        A mathematical model function that predicts assembly indices based on the
        number of peaks and model parameters.
    params_0 : array-like
        The initial guess for the model parameters.

    Returns
    -------
    tuple
        A tuple containing:
        - res.x : numpy.ndarray
            The optimized model parameters.
        - data_pred : numpy.ndarray
            The predicted assembly indices as integers.

    Notes
    -----
    - The optimization is performed using the Nelder-Mead method.
    - The tolerance for the optimization is set to 1e-6.
    """
    res = minimize(_func_min_helper,
                   np.array(params_0),
                   args=(peaks_data,
                         ai_obs,
                         model),
                   method='Nelder-Mead',
                   tol=1e-6)
    data_pred = [_peaks_to_ai(x_i, model, res.x) for x_i in peaks_data]
    return res.x, np.array(data_pred, dtype=int)


def get_github_file(
        filename: str,
        repo_url: str,
        dest_dir: str | Path = Path.cwd(),
        overwrite: bool = False,
        timeout: int = 30,
) -> Path:
    """
    Download a file from a GitHub repository and save it to a specified directory.

    This function downloads a file from a given GitHub repository URL and saves it
    to the specified destination directory. If the file already exists, it can be
    skipped unless the `overwrite` parameter is set to True.

    Parameters
    ----------
    filename : str
        The name of the file to download.
    repo_url : str
        The URL of the GitHub repository (e.g., "https://raw.githubusercontent.com/user/repo/branch").
    dest_dir : str | Path, optional
        The directory where the file will be saved. Defaults to the current working directory.
    overwrite : bool, optional
        Whether to overwrite the file if it already exists. Defaults to False.
    timeout : int, optional
        The timeout in seconds for the download request. Defaults to 30.

    Returns
    -------
    Path
        The path to the downloaded file.

    Notes
    -----
    - If the file already exists and `overwrite` is False, the function skips the download.
    - The function creates the destination directory if it does not exist.
    - A temporary file with a ".part" suffix is used during the download process to ensure
      atomicity.

    Raises
    ------
    urllib.error.URLError
        If there is an issue with the download request.
    """
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


def sample_cbrdb(n_samples: int,
                 max_mw: float = 550.0,
                 max_bonds: float = 50,
                 c_select: List[str] | None = None) -> pd.DataFrame:
    """
    Sample a subset of compounds from the CBRdb dataset.

    This function downloads the CBRdb dataset, filters the compounds based on
    molecular weight, bond count, and other criteria, and returns a random sample
    of the filtered compounds.

    Parameters
    ----------
    n_samples : int
        The number of samples to return.
    max_mw : float, optional
        The maximum molecular weight allowed for the compounds. Default is 550.0.
    max_bonds : float, optional
        The maximum number of bonds allowed for the compounds. Default is 50.
    c_select : List[str] or None, optional
        The columns to select from the dataset. If None, defaults to
        ['compound_id', 'nickname', 'smiles', 'molecular_weight', 'n_heavy_atoms'].

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the sampled compounds.

    Notes
    -----
    - The function downloads the dataset from the CBRdb GitHub repository.
    - Invalid SMILES strings and molecules exceeding the specified molecular weight
      or bond count are excluded.
    - If the requested number of samples exceeds the available filtered compounds,
      all available compounds are returned.
    """
    repo_url = "https://raw.githubusercontent.com/ELIFE-ASU/CBRdb/refs/heads/main"
    target_file = "CBRdb_C.csv.zip"
    # Set default columns to select if not provided
    if c_select is None:
        c_select = ['compound_id', 'nickname', 'smiles', 'molecular_weight', 'n_heavy_atoms']
    # Download the dataset file
    path = get_github_file(target_file, repo_url)
    # Load the dataset into a DataFrame
    df = pd.read_csv(path, low_memory=False)
    # Remove the downloaded file after loading
    os.remove(path)
    # Select the specified columns
    df = df[c_select]
    # Drop rows with missing SMILES strings
    df = df.dropna(subset=['smiles'])
    # Filter out invalid SMILES strings
    df = df[mp_calc(_valid_smi, df['smiles'])]
    # Filter out invalid molecules
    df = df[mp_calc(_valid_mol, df['smiles'])]
    # Filter by maximum molecular weight
    df = df[df['molecular_weight'] <= max_mw]
    # Calculate the number of bonds for each molecule
    df['n_bonds'] = mp_calc(count_non_h_bonds, mp_calc(smi_to_mol, df['smiles']))
    # Filter out molecules with too many bonds
    df = df[df['n_bonds'] <= max_bonds]
    # Randomly sample the requested number of entries
    if n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42)
    else:
        print(f"Requested {n_samples} samples, but only {len(df)} available after filtering.")
    # Return the sampled DataFrame
    return df.reset_index(drop=True)
