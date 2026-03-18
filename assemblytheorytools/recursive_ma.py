import functools
import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats.distributions import skewnorm

ISOTOPES = {
    "Antimony": 120.903824,
    # "Argon": 39.962383,
    "Arsenic": 74.921596,
    "Barium": 137.905236,
    "Bismuth": 208.980388,
    "Bromine": 78.918336,
    "Cadmium": 113.903361,
    "Calcium": 39.962591,
    "Cerium": 139.905442,
    "Cesium": 132.905433,
    "Chlorine": 34.968853,
    "Chromium": 51.94051,
    "Cobalt": 58.933198,
    "Copper": 62.929599,
    "Dysprosium": 163.929183,
    "Erbium": 165.930305,
    "Europium": 152.921243,
    "Gadolinium": 157.924111,
    "Gallium": 68.925581,
    "Germanium": 73.921179,
    "Gold": 196.96656,
    "Hafnium": 179.946561,
    "Holmium": 164.930332,
    "Indium": 114.903875,
    "Iodine": 126.904477,
    "Iridium": 192.962942,
    "Iron": 55.934939,
    "Krypton": 83.911506,
    "Lanthanum": 138.906355,
    "Lead": 207.976641,
    "Lutetium": 174.940785,
    "Manganese": 54.938046,
    "Mercury": 201.970632,
    # "Molybdenum": 97.905405,
    "Neodymium": 141.907731,
    "Nickel": 57.935347,
    "Niobium": 92.906378,
    "Osmium": 191.961487,
    "Palladium": 105.903475,
    "Platinum": 194.964785,
    "Potassium": 38.963708,
    "Praseodymium": 140.907657,
    "Rhenium": 186.955765,
    "Rhodium": 102.905503,
    "Rubidium": 84.9118,
    "Ruthenium": 101.904348,
    "Samarium": 151.919741,
    # "Scandium": 44.955914,
    "Selenium": 79.916521,
    "Silver": 106.905095,
    # "Strontium": 87.905625,
    "Sulfur": 33.967868,
    "Tantalum": 180.948014,
    "Tellurium": 129.906229,
    "Terbium": 158.92535,
    "Thallium": 204.97441,
    "Thorium": 232.038054,
    "Thulium": 168.934225,
    "Tin": 119.902199,
    "Titanium": 47.947947,
    "Tungsten": 183.950953,
    "Uranium": 238.050786,
    "Vanadium": 50.943963,
    "Xenon": 131.904148,
    "Ytterbium": 173.938873,
    "Yttrium": 88.905856,
    "Zinc": 63.929145,
    "Zirconium": 89.904708,
}
# Source: https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses
# Monoisotopic masses of common adduct ions
COMMON_PRECURSORS = [
    0.0,  # Nothing
    1.007825,  # H+
]

# Minimum MW of what can be considered a fragment
MIN_CHUNK = 20.0


def ma_distribution_params(mw):
    """
    Calculate the parameters for the molecular assembly (MA) distribution based on molecular weight.

    This function computes the shape (alpha), location (loc), and scale (scale) parameters
    for a skew-normal distribution that models the distribution of assembly numbers (MA)
    for a given molecular weight (mw). These parameters are used to generate random samples
    or to describe the expected distribution of assembly numbers for fragments of a given size.

    Parameters
    ----------
    mw : float
        The molecular weight for which to calculate the distribution parameters.

    Returns
    -------
    tuple
        A tuple (alpha, loc, scale) containing the skewness, location, and scale parameters
        for the skew-normal distribution.
    """
    alpha = -0.0044321370413747405 * mw + -1.1014882364398888
    loc = 0.075 * mw - 1.3
    scale = 0.008058454819492319 * mw + 0.546185725719078
    return alpha, loc, scale


def ma_samples(mw, n_samples):
    """
    Generate random samples from the molecular assembly (MA) distribution for a given molecular weight.

    This function uses the skew-normal distribution parameters (alpha, loc, scale) calculated
    for the specified molecular weight (mw) to generate random samples representing possible
    assembly numbers (MA). Negative values are replaced with zero to ensure all samples are non-negative.

    Parameters
    ----------
    mw : float
        The molecular weight for which to generate the MA distribution samples.
    n_samples : int
        The number of random samples to generate.

    Returns
    -------
    np.ndarray
        An array of non-negative random samples from the MA distribution for the given molecular weight.
    """
    alpha, loc, scale = ma_distribution_params(mw)
    return np.maximum(skewnorm(alpha, loc, scale).rvs(n_samples), 0.)


def rma_unify_trees(trees: list[dict]):
    """
    Recursively merge multiple fragmentation trees into a single unified tree.

    This function takes a list of fragmentation trees (each represented as a nested dictionary)
    and merges them into a single tree. If the list is empty, it returns an empty dictionary.
    If there is only one tree, it returns that tree. If there are multiple trees, it merges
    them by:
      - Taking all unique keys from each tree and including their subtrees as-is.
      - For keys present in more than one tree, recursively merging their subtrees.

    This is useful for combining results from multiple samples or experiments into a single
    hierarchical structure for further analysis.

    Parameters
    ----------
    trees : list of dict
        A list of fragmentation trees, where each tree is a nested dictionary.

    Returns
    -------
    dict
        A single unified fragmentation tree as a nested dictionary.
    """
    if not trees:
        return {}
    elif len(trees) == 1:
        return trees[0]
    else:
        child1, child2, *rest = trees
        child1_keys = set(child1 or {})
        child2_keys = set(child2 or {})
        common_keys = child1_keys.intersection(child2_keys)
        return {
            **{k: child1[k] for k in child1_keys - common_keys},
            **{k: child2[k] for k in child2_keys - common_keys},
            **{k: rma_unify_trees([child1[k], child2[k]]) for k in common_keys},
        }


class MAEstimator:
    """
    A class for estimating molecular assembly (MA) numbers in fragmentation trees.

    The MAEstimator provides methods to estimate the assembly number (MA) for a given molecular
    weight (MW) or fragmentation tree, identify common precursors, and recursively analyze
    fragmentation patterns. It supports both same-level and cross-level precursor analysis,
    configurable mass tolerance, and customizable adduct masses.

    Parameters
    ----------
    same_level : bool, optional
        If True, only consider same-level precursors when searching for fragmentation relationships.
        Defaults to True.
    tol : float, optional
        Mass tolerance for matching m/z values. Defaults to 0.01.
    adduct_masses : list of float, optional
        List of adduct ion masses to consider when searching for possible precursor ions.
        Defaults to COMMON_PRECURSORS.
    n_samples : int, optional
        Number of random samples to use for MA estimation. Defaults to 20.

    Attributes
    ----------
    same_level : bool
        Whether to restrict precursor search to the same fragmentation level.
    tol : float
        Mass tolerance for m/z matching.
    adduct_masses : list of float
        Adduct ion masses considered in precursor search.
    n_samples : int
        Number of samples for MA estimation.
    zero : np.ndarray
        Array of zeros used when a fragment matches an isotope (MA = 0).
    """

    def __init__(self, same_level=True, tol=0.01, adduct_masses=COMMON_PRECURSORS, n_samples=500):
        """
        Initialize the MAEstimator with configuration options.

        Parameters
        ----------
        same_level : bool, optional
            Restrict precursor search to the same fragmentation level. Defaults to True.
        tol : float, optional
            Mass tolerance for m/z matching. Defaults to 0.01.
        adduct_masses : list of float, optional
            List of adduct ion masses to consider. Defaults to COMMON_PRECURSORS.
        n_samples : int, optional
            Number of random samples for MA estimation. Defaults to 500.
        """
        self.same_level = same_level
        self.tol = tol
        self.adduct_masses = adduct_masses
        self.n_samples = n_samples
        self.zero = np.zeros(n_samples)

    @functools.cache
    def estimate_by_MW(self, mw, has_children):
        """
        Estimate the MA distribution for a given molecular weight.

        If the fragment matches a known isotope within the specified tolerance and has no children,
        returns an array of zeros (MA = 0). Otherwise, generates random MA samples for the given MW.

        Parameters
        ----------
        mw : float
            The molecular weight for which to estimate the MA distribution.
        has_children : bool
            Indicates whether the fragment has child fragments.

        Returns
        -------
        np.ndarray
            An array of estimated MA values for the given molecular weight.
        """
        lower, upper = mw - self.tol, mw + self.tol
        if not has_children:
            for isotope, weight in ISOTOPES.items():
                if lower < weight < upper:
                    # MW matches an isotope; MA = 0
                    print(f"HIT: {mw} ~ {isotope} ({weight})")
                    return self.zero
        return ma_samples(mw, self.n_samples)

    def estimate_MA(self, tree: dict[float, dict], mw: float, progress_levels=0, joint=False):
        """
        Recursively estimate the mean assembly number (MA) for a given molecular weight in a fragmentation tree.

        This method traverses the fragmentation tree, considering all possible fragmentations
        and their complements, and recursively estimates the MA for each. It supports both
        simple and joint estimation strategies, and can consider common precursors for more
        complex fragmentation patterns.

        Parameters
        ----------
        tree : dict[float, dict]
            The fragmentation tree as a nested dictionary.
        mw : float
            The molecular weight for which to estimate the MA.
        progress_levels : int, optional
            Number of recursive levels to consider in the estimation. Defaults to 0.
        joint : bool, optional
            If True, use a joint estimation strategy. Defaults to False.

        Returns
        -------
        np.ndarray
            An array of estimated MA values for the given molecular weight.
        """
        children = rma_unify_trees([tree.get(mw, None) or self.precursors(tree, mw)])
        if joint:
            return sum(self.estimate_MA(children, child, progress_levels - 1) for child in children)
        child_estimates = {mw: self.estimate_by_MW(mw, bool(children))}

        for child in children:
            complement = mw - child
            if complement < MIN_CHUNK or child < MIN_CHUNK:
                continue

            common = [
                p
                for p in self.common_precursors(children, child, complement)
                if p > MIN_CHUNK and max(child - p, complement - p) > MIN_CHUNK
            ]

            if common and progress_levels > 0:
                print(f"Common precursors of {mw} = {child} + {complement}: {common}")

            # Simple child + complement with no common precursors
            ma_candidates = [
                self.estimate_MA(children, child, progress_levels - 1)
                + self.estimate_MA(children, complement, progress_levels - 1)
                + 1.0
            ]

            for precursor in common:
                chunks = [child - precursor, complement - precursor, precursor]
                if min(chunks) < MIN_CHUNK:
                    continue
                chunk_mas = sum(
                    self.estimate_MA(
                        children,
                        chunk,
                        progress_levels - 1,
                    )
                    for chunk in chunks
                )
                ma_candidates.append(chunk_mas + 3)

            child_estimates[child] = min(ma_candidates, key=np.mean)
            if progress_levels > 0:
                print(f"MA({mw} = {child} + {complement}) = {child_estimates[child].mean()}")

        # estimate = np.concatenate(list(child_estimates.values()))
        estimate = min(child_estimates.values(), key=np.mean)
        return estimate

    def common_precursors(self, data, parent1, parent2):
        """
        Find common precursor ions between two parent ions in a fragmentation tree.

        Parameters
        ----------
        data : dict
            The fragmentation tree or data structure.
        parent1 : float
            The m/z value of the first parent ion.
        parent2 : float
            The m/z value of the second parent ion.

        Returns
        -------
        set
            Set of m/z values representing common precursor ions.
        """
        precursors1 = self.precursors(data, parent1)
        precursors2 = self.precursors(data, parent2)
        return set(precursors1).intersection(precursors2)

    def precursors(self, data, parent):
        """
        Identify possible precursor ions for a given parent ion in the fragmentation tree.

        Considers adduct masses and mass tolerance to find candidate precursor ions.
        If no children are found and same_level is enabled, attempts same-level precursor search.

        Parameters
        ----------
        data : dict
            The fragmentation tree or data structure.
        parent : float
            The m/z value of the parent ion.

        Returns
        -------
        dict
            Dictionary of precursor ions and their subtrees.
        """
        if parent < MIN_CHUNK:
            return {}

        possible_ions = [parent + adduct for adduct in self.adduct_masses]
        parent_candidates = [
            d
            for d in data
            if any(d - self.tol < p < d + self.tol for p in possible_ions)
        ]
        children = rma_unify_trees([
            {
                **{
                    p - child: self.same_level_precursors(data, p - child)
                    for child in data[p] or {}
                },
                **(data[p] or {}),
            }
            for p in parent_candidates
        ])
        if not children and self.same_level:
            children = rma_unify_trees([
                self.same_level_precursors(data, p)
                for p in parent_candidates or possible_ions
            ])

        # sometimes child peaks are heavier than parent
        return {k: v for k, v in children.items() if 0 < k < parent}

    def same_level_precursors(self, data, parent):
        """
        Find same-level precursor ions for a given parent ion.

        Searches for ions at the same fragmentation level that could serve as precursors,
        considering adduct masses and mass tolerance.

        Parameters
        ----------
        data : dict
            The fragmentation tree or data structure.
        parent : float
            The m/z value of the parent ion.

        Returns
        -------
        dict
            Dictionary of same-level precursor ions and their subtrees.
        """
        result = {}
        adducts, tol = self.adduct_masses, self.tol
        for ion in data:
            target = parent - ion
            if any(
                    d - tol < target + adduct < d + tol for d in data for adduct in adducts
            ):
                result[ion] = data[ion]
        return result


def _build_tree(data, level=1, acc=None, parent=None, max_level=3):
    """
    Recursively build a nested fragmentation tree from a multi-level mass spectrometry dataset.

    This internal helper function constructs a hierarchical tree structure from a dictionary
    of pandas DataFrames, where each DataFrame represents a different MS level (e.g., MS1, MS2, etc.).
    The resulting tree is a nested dictionary, with each node corresponding to a peak (m/z value)
    and its children representing fragment ions at subsequent MS levels. The recursion continues
    up to a specified maximum MS level.

    Parameters
    ----------
    data : dict
        A dictionary mapping MS levels (integers) to pandas DataFrames containing peak data.
    level : int, optional
        The current MS level being processed. Defaults to 1 (root level).
    acc : dict or None, optional
        The accumulator dictionary for building the tree. If None, a new dictionary is created at the root.
    parent : float or None, optional
        The m/z value of the parent peak for the current recursion level. Used to filter child peaks.
    max_level : int, optional
        The maximum MS level to include in the tree. Defaults to 3.

    Returns
    -------
    dict or None
        A nested dictionary representing the fragmentation tree, or None if the recursion exceeds max_level.
    """
    max_level = min(max_level, max(data))
    if level == 1:
        acc = {}
        for peak in data[1]["mz"]:
            peak_float = float(peak)
            acc[peak_float] = {}
            _build_tree(data, level=2, acc=acc[peak_float], parent=peak_float, max_level=max_level)
        return acc
    if level > max_level:
        return
    level_df = data[level]
    parent_df = data[level - 1].drop(columns=["parent_id"], errors="ignore")
    level_df = level_df.join(parent_df, on="parent_id", rsuffix="_parent")
    child_peaks = level_df[level_df["mz_parent"] == parent]["mz"].unique()
    for peak in child_peaks:
        peak_float = float(peak)
        acc[peak_float] = None if level == max_level else {}
        _build_tree(
            data, level=level + 1, acc=acc[peak_float], parent=peak_float, max_level=max_level
        )


def rma_build_tree(data: dict, max_level=3):
    """
    Build a nested fragmentation tree from a multi-level mass spectrometry dataset.

    This function constructs a hierarchical tree structure from a dictionary of pandas DataFrames,
    where each DataFrame represents a different MS level (e.g., MS1, MS2, etc.). The resulting tree
    is a nested dictionary, with each node corresponding to a peak (m/z value) and its children
    representing fragment ions at subsequent MS levels. The tree is built up to a specified maximum
    MS level.

    Parameters
    ----------
    data : dict
        A dictionary mapping MS levels (integers) to pandas DataFrames containing peak data.
    max_level : int, optional
        The maximum MS level to include in the tree. Defaults to 3.

    Returns
    -------
    dict
        A nested dictionary representing the fragmentation tree, where each key is an m/z value
        and each value is either a subtree (dict) or None for terminal nodes.
    """
    return _build_tree(data, max_level=max_level)


def rma_tree_depth(tree: dict):
    """
    Recursively determine the depth of a fragmentation tree.

    This function computes the maximum depth of a nested dictionary structure representing
    a fragmentation tree. The depth is defined as the number of levels from the root node
    to the deepest leaf node. An empty dictionary or non-dictionary input returns a depth of 0.

    Parameters
    ----------
    tree : dict
        The fragmentation tree as a nested dictionary, where each key is an m/z value and
        each value is either a subtree (dict) or a terminal node (e.g., None or empty dict).

    Returns
    -------
    int
        The maximum depth of the tree. Returns 0 for empty or non-dictionary input.
    """
    if isinstance(tree, dict) and len(tree) > 0:
        return 1 + max(rma_tree_depth(v) for v in tree.values())
    else:
        return 0


def _process_df(
        level,
        ms_df: pd.DataFrame,
        max_num_peaks,
        min_abs_intensity,
        min_rel_intensity,
        n_digits,
):
    """
    Filter, bin, and aggregate peaks for a single MS level in a mass spectrometry dataset.

    This function processes a pandas DataFrame containing peak data for a specific MS level.
    It performs the following steps:
      - Ensures a 'parent' column exists (assigns a large default if missing).
      - Filters out peaks where the m/z is not less than the parent m/z minus 1.
      - Bins m/z and parent m/z values to integer bins with a specified number of decimal digits.
      - Aggregates peaks by (mz_bin, parent_bin), summing intensities and taking the median m/z and parent.
      - Applies intensity-based filtering: keeps only peaks above a minimum absolute or relative intensity threshold.
      - Limits the number of peaks per parent to a maximum.
      - Returns the processed DataFrame.

    Parameters
    ----------
    level : int
        The MS level being processed (e.g., 1 for MS1, 2 for MS2, etc.).
    ms_df : pd.DataFrame
        DataFrame containing columns 'mz', 'intensity', and optionally 'parent'.
    max_num_peaks : int
        Maximum number of peaks to retain per parent_bin.
    min_abs_intensity : dict
        Dictionary mapping MS levels to minimum absolute intensity thresholds.
    min_rel_intensity : float
        Minimum relative intensity threshold (as a fraction of the maximum intensity in a group).
    n_digits : int
        Number of decimal digits to use for binning m/z values.

    Returns
    -------
    pd.DataFrame
        The filtered, binned, and aggregated DataFrame for this MS level.
    """
    original_len = len(ms_df)

    if "parent" not in ms_df:
        ms_df["parent"] = 10 ** 6
    ms_df = ms_df[ms_df.mz < ms_df.parent - 1]

    ms_df = ms_df.assign(
        mz_bin=(ms_df.mz.round(n_digits) * 10 ** n_digits).astype(int),
        parent_bin=(ms_df.parent.round(n_digits) * 10 ** n_digits).astype(int),
    )

    min_intensity_fn = lambda df: max(
        min_abs_intensity[level], df["intensity"].max() * min_rel_intensity
    )
    filter_fn = (
        lambda g: g[g["intensity"] > min_intensity_fn(g)]
        .sort_values("intensity")
        .tail(max_num_peaks)
    )
    ms_df = (
        ms_df.groupby(["mz_bin", "parent_bin"])
        .agg({"intensity": "sum", "mz": "median", "parent": "median"})
        .reset_index()
        .groupby("parent_bin", group_keys=False)
        .apply(filter_fn)
    )

    result = (
        ms_df.groupby("parent_bin", group_keys=False)
        .apply(lambda g: g.sort_values("intensity").tail(max_num_peaks))
    )

    logging.debug(f"Level {level}: {len(result)} out of {original_len} peaks retained")
    return result


def rma_process(
        sample: dict[int, pd.DataFrame],
        max_num_peaks: int = 200,
        min_abs_intensity: dict[int, float] = defaultdict(lambda: 0.0),
        min_rel_intensity: float = 0.0,
        n_digits: int = 3,
) -> dict[int, pd.DataFrame]:
    """
    Process a multi-level mass spectrometry sample by filtering and binning peaks for each MS level.

    This function processes a dictionary of pandas DataFrames representing different MS levels
    (e.g., MS1, MS2, etc.) in a mass spectrometry experiment. For each level, it applies filtering
    based on intensity thresholds and bins m/z values to reduce noise and redundancy. If the MS1
    level is missing, a placeholder is generated from the most intense parent peak in MS2.

    Parameters
    ----------
    sample : dict[int, pd.DataFrame]
        A dictionary mapping MS levels (integers) to pandas DataFrames containing peak data.
    max_num_peaks : int, optional
        The maximum number of peaks to retain per parent per MS level. Defaults to 200.
    min_abs_intensity : dict[int, float], optional
        Minimum absolute intensity threshold for each MS level. Defaults to 0.0 for all levels.
    min_rel_intensity : float, optional
        Minimum relative intensity threshold (as a fraction of the maximum intensity) for peak filtering.
        Defaults to 0.0.
    n_digits : int, optional
        Number of decimal digits to use for binning m/z values. Defaults to 3.

    Returns
    -------
    dict[int, pd.DataFrame]
        A dictionary with the same structure as the input, where each DataFrame has been filtered
        and binned. If MS1 was missing, it will be added as a placeholder.
    """
    sample = {
        level: _process_df(
            level,
            df.reset_index(),
            max_num_peaks,
            min_abs_intensity,
            min_rel_intensity,
            n_digits,
        )
        for level, df in sample.items()
    }
    if 1 not in sample:
        # Generate placeholder MS1 if only MS2+ present
        parent_peak = (
            sample[2].groupby("parent")["intensity"].sum().sort_values().index[-1]
        )
        sample[1] = pd.DataFrame(
            {"mz": [parent_peak],
             "intensity": 100000.0,
             "mz_bin": [int(parent_peak * 10 ** n_digits)],
             }
        )
    return sample


def rma_identify_parents(dataset, mass_tol: float, ms_n_digits: int=3):
    """
    Assign parent-child relationships between MS levels in a mass spectrometry dataset.

    This function iteratively assigns parent IDs to peaks in higher MS levels (e.g., MS2, MS3)
    by matching their parent_bin values to the mz_bin values of peaks in the previous level,
    within a specified mass tolerance. It uses pandas' merge_asof for efficient nearest-neighbor
    matching, and ensures that each child peak is associated with the closest parent peak.

    Parameters
    ----------
    dataset : dict
        A dictionary mapping MS levels (integers) to pandas DataFrames. Each DataFrame must
        contain at least the columns 'mz_bin' and 'parent_bin'.
    mass_tol : float
        The mass tolerance for matching parent and child peaks, in the same units as m/z.
    ms_n_digits : int
        The number of decimal digits used for binning m/z values. Used to scale the tolerance.

    Returns
    -------
    dict
        A new dictionary with the same structure as the input, but with parent-child relationships
        assigned. Each DataFrame in the output will have a 'parent_id' column indicating the
        matched parent peak index.
    """
    new_dataset = {}
    new_dataset[min(dataset)] = dataset[min(dataset)]
    for level in sorted(dataset)[:-1]:
        new_dataset[level + 1] = (
            pd.merge_asof(
                dataset[level + 1].sort_values("parent_bin"),
                new_dataset[level][["mz_bin"]]
                .sort_values("mz_bin")
                .reset_index(),
                left_on="parent_bin",
                right_on="mz_bin",
                suffixes=("", "_x"),
                tolerance=int(mass_tol * 10 ** ms_n_digits),
                direction="nearest",
            )
            .rename(columns={"index": "parent_id"})
            .dropna(subset=["parent_id"])
            .astype({"parent_id": int})
            .drop(columns=["mz_bin_x"])
        )
    return new_dataset


def _tol_from_decimals(decimals):
    """
    Convert a number of decimal places to a mass tolerance value.

    This utility function is used to determine the mass tolerance (e.g., for m/z matching)
    based on the number of decimal places specified. If `decimals` is None, a default
    tolerance of 0.01 is returned. Otherwise, the tolerance is calculated as 10 to the
    negative power of the number of decimals (e.g., decimals=3 yields 0.001).

    Parameters
    ----------
    decimals : int or None
        The number of decimal places to use for the tolerance. If None, a default value is used.

    Returns
    -------
    float
        The calculated tolerance value.
    """
    if decimals is None:
        return 0.01
    return 10 ** (-decimals)


def rma_estimate_ma(tree, mw, progress_levels=0, joint=False, **kwargs):
    """
    Estimate the mean assembly number (MA) for a given molecular weight (MW) in a fragmentation tree.

    This function uses the MAEstimator class to estimate the assembly number (MA) for a specified
    molecular weight (mw) within a given fragmentation tree. The estimation can be performed
    recursively to a specified depth (progress_levels), and can optionally use a joint estimation
    strategy.

    Parameters
    ----------
    tree : dict
        The fragmentation tree, represented as a nested dictionary structure.
    mw : float
        The molecular weight (MW) for which to estimate the assembly number.
    progress_levels : int, optional
        The number of recursive levels to consider in the estimation. Defaults to 0.
    joint : bool, optional
        If True, use a joint estimation strategy for the assembly number. Defaults to False.
    **kwargs
        Additional keyword arguments to pass to the MAEstimator constructor.

    Returns
    -------
    float
        The mean estimated assembly number (MA) for the given molecular weight.
    """
    estimator = MAEstimator(**kwargs)
    result = estimator.estimate_MA(
        tree=tree,
        mw=mw,
        progress_levels=progress_levels,
        joint=joint,
    )
    return float(result.mean())


def rma_estimate_by_mw(mw, has_children=False, **kwargs):
    """
    Estimate the assembly number (MA) distribution for a given molecular weight (MW).

    This function creates an instance of the MAEstimator class and uses it to estimate
    the assembly number distribution for a specified molecular weight. If the fragment
    is known to have children (i.e., it is not a terminal node in a fragmentation tree),
    this can be indicated with the has_children flag.

    Parameters
    ----------
    mw : float
        The molecular weight (MW) for which to estimate the assembly number distribution.
    has_children : bool, optional
        Indicates whether the fragment has child fragments (default is False).
    **kwargs
        Additional keyword arguments to pass to the MAEstimator constructor.

    Returns
    -------
    np.ndarray
        An array of estimated assembly numbers (MA) for the given molecular weight.
    """
    estimator = MAEstimator(**kwargs)
    return estimator.estimate_by_MW(mw=mw, has_children=has_children)


def find_common_precursors(data, parent1, parent2, same_level=True, decimals=None, **kwargs):
    """
    Find common precursor ions between two parent ions in a fragmentation tree.

    This function identifies precursor ions that are shared between two specified parent ions
    within a given fragmentation tree or dataset. It uses the MAEstimator class to perform the
    search, allowing for control over matching tolerance and whether to consider only same-level
    precursors.

    Parameters
    ----------
    data : dict
        The fragmentation tree or data structure containing parent and child ion relationships.
    parent1 : float
        The m/z value of the first parent ion.
    parent2 : float
        The m/z value of the second parent ion.
    same_level : bool, optional
        If True, only consider precursors at the same fragmentation level. Defaults to True.
    decimals : int or None, optional
        Number of decimal places to use for m/z tolerance. If None, a default tolerance is used.
    **kwargs
        Additional keyword arguments passed to the MAEstimator.

    Returns
    -------
    set
        A set of m/z values representing the common precursor ions between parent1 and parent2.
    """
    estimator = MAEstimator(same_level=same_level, tol=_tol_from_decimals(decimals), **kwargs)
    return estimator.common_precursors(data=data, parent1=parent1, parent2=parent2)


def rma_print_tree(tree: Dict, indent: int = 0, max_depth: int = 10) -> None:
    """
    Pretty print a fragmentation tree.

    Recursively prints the structure of a fragmentation tree in a readable format,
    showing the m/z values at each level with indentation to represent tree depth.

    Parameters
    ----------
    tree : dict
        The fragmentation tree as a nested dictionary, where each key is an m/z value
        and each value is a subtree (or None/empty dict for leaves).
    indent : int, optional
        Current indentation level. Used internally for recursive calls. Defaults to 0.
    max_depth : int, optional
        Maximum depth to print. If the tree is deeper, deeper levels are not shown.
        Defaults to 10.

    Returns
    -------
    None
        This function prints the tree structure to standard output and does not return a value.
    """
    if indent > max_depth or not isinstance(tree, dict):
        return

    for mz, children in sorted(tree.items(), reverse=True):
        print("  " * indent + f"├─ m/z: {mz:.2f}")
        rma_print_tree(children, indent + 1, max_depth)


def rma_meta_tree(samples: List[Dict], meta_parent_mz: float = 1e6) -> Dict[float, Dict]:
    """
    Combine multiple fragmentation trees under a single 'meta' parent precursor.

    This function takes a list of fragmentation trees and merges them into a single
    unified tree structure, encapsulated under a 'meta' parent precursor with a specified
    mass-to-charge ratio (m/z).

    Parameters
    ----------
    samples : list of dict
        A list of fragmentation trees, where each tree is represented as a dictionary.
    meta_parent_mz : float, optional
        The mass-to-charge ratio (m/z) of the 'meta' parent precursor. Defaults to 1e6.

    Returns
    -------
    dict
        A dictionary representing the unified tree structure, with the 'meta' parent precursor
        as the root node.
    """
    return {meta_parent_mz: rma_unify_trees(samples)}
