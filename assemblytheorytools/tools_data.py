from typing import Union, Tuple

import numpy as np
from scipy.stats import gaussian_kde


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
