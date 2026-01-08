import random
import time
from typing import Union, List, Optional, Tuple

import networkx as nx
import numpy as np
import pubchempy as pcp
from rdkit import Chem
from scipy.stats import gaussian_kde

from .tools_mol import smi_to_mol
from .tools_graph import smi_to_nx

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


def pubchem_name_to_smi(name: str) -> str:
    return pcp.get_compounds(name, "name")[0].smiles


def pubchem_name_to_mol(name: str, add_hydrogens: bool = True, sanitize: bool = True) -> Chem.Mol:
    smiles = pubchem_name_to_smi(name)
    return smi_to_mol(smiles, add_hydrogens=add_hydrogens, sanitize=sanitize)


def pubchem_name_to_nx(name: str, add_hydrogens: bool = True, sanitize: bool = True) -> nx.Graph:
    smiles = pubchem_name_to_smi(name)
    return smi_to_nx(smiles, add_hydrogens=add_hydrogens, sanitize=sanitize)


def pubchem_id_to_smi(id: int) -> str:
    return pcp.Compound.from_cid(id).smiles


def pubchem_id_to_mol(id: int, add_hydrogens: bool = True, sanitize: bool = True) -> Chem.Mol:
    smiles = pubchem_id_to_smi(id)
    return smi_to_mol(smiles, add_hydrogens=add_hydrogens, sanitize=sanitize)


def pubchem_id_to_nx(id: int, add_hydrogens: bool = True, sanitize: bool = True) -> nx.Graph:
    smiles = pubchem_id_to_smi(id)
    return smi_to_nx(smiles, add_hydrogens=add_hydrogens, sanitize=sanitize)


def sample_random_pubchem(n: int,
                          *,
                          seed: Optional[int] = None,
                          max_cid: int = 123_431_215,
                          delay_s: float = 0.2,
                          max_attempts: int = 50_000,
                          ) -> Tuple[List[int], List[str]]:
    if n <= 0:
        raise ValueError("n must be a positive integer")

    rng = random.Random(seed)
    mols: List[str] = []
    seen: set[int] = set()
    ids: List[int] = []

    attempts = 0
    while len(mols) < n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Only collected {len(mols)} valid molecules after {max_attempts} attempts.")

        cid = rng.randint(1, max_cid)
        if cid in seen:
            continue
        seen.add(cid)

        try:
            c = pcp.Compound.from_cid(cid)
            mol = getattr(c, "smiles", None)
            if mol is None:
                continue
            mols.append(mol)
            ids.append(cid)

        except Exception:
            continue
        finally:
            time.sleep(delay_s)

    return ids, mols
