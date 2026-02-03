import functools
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats.distributions import skewnorm

# Monoisotopic masses for a set of atomic elements.
# Used to detect when a mass corresponds directly to a single element (MA = 0 case).
ISOTOPES = {
    # "Antimony": 120.903824,
    # "Argon": 39.962383,
    "Arsenic": 74.921596,
    # "Barium": 137.905236,
    # "Bismuth": 208.980388,
    "Bromine": 78.918336,
    # "Cadmium": 113.903361,
    "Calcium": 39.962591,
    # "Cerium": 139.905442,
    "Cesium": 132.905433,
    "Chlorine": 34.968853,
    # "Chromium": 51.94051,
    "Cobalt": 58.933198,
    "Copper": 62.929599,
    # "Dysprosium": 163.929183,
    # "Erbium": 165.930305,
    # "Europium": 152.921243,
    # "Gadolinium": 157.924111,
    # "Gallium": 68.925581,
    # "Germanium": 73.921179,
    # "Gold": 196.96656,
    # "Hafnium": 179.946561,
    # "Holmium": 164.930332,
    # "Indium": 114.903875,
    "Iodine": 126.904477,
    # "Iridium": 192.962942,
    "Iron": 55.934939,
    # "Krypton": 83.911506,
    # "Lanthanum": 138.906355,
    # "Lead": 207.976641,
    # "Lutetium": 174.940785,
    "Manganese": 54.938046,
    "Mercury": 201.970632,
    # "Molybdenum": 97.905405,
    # "Neodymium": 141.907731,
    "Nickel": 57.935347,
    # "Niobium": 92.906378,
    # "Osmium": 191.961487,
    # "Palladium": 105.903475,
    # "Platinum": 194.964785,
    "Potassium": 38.963708,
    # "Praseodymium": 140.907657,
    # "Rhenium": 186.955765,
    # "Rhodium": 102.905503,
    # "Rubidium": 84.9118,
    # "Ruthenium": 101.904348,
    # "Samarium": 151.919741,
    # "Scandium": 44.955914,
    "Selenium": 79.916521,
    # "Silver": 106.905095,
    # "Strontium": 87.905625,
    "Sulfur": 33.967868,
    # "Tantalum": 180.948014,
    # "Tellurium": 129.906229,
    # "Terbium": 158.92535,
    # "Thallium": 204.97441,
    # "Thorium": 232.038054,
    # "Thulium": 168.934225,
    # "Tin": 119.902199,
    # "Titanium": 47.947947,
    # "Tungsten": 183.950953,
    # "Uranium": 238.050786,
    # "Vanadium": 50.943963,
    # "Xenon": 131.904148,
    # "Ytterbium": 173.938873,
    # "Yttrium": 88.905856,
    # "Zinc": 63.929145,
    # "Zirconium": 89.904708,
}


def _ma_distribution_params(mw: float) -> Tuple[float, float, float]:
    """
    Convert a molecular weight into skew-normal distribution parameters.

    Jirasek, M., (2024).
    Investigating and quantifying molecular complexity using assembly theory and spectroscopy.
    ACS Central Science, 10(5), 1054-1064.

    Parameters
    ----------
    mw : float
        Molecular weight (mass) for which to estimate distribution parameters.

    Returns
    -------
    (alpha, loc, scale) : tuple of float
        alpha
            Shape (skewness) parameter for :class:`scipy.stats.skewnorm`.
        loc
            Location (central tendency) parameter.
        scale
            Scale (spread) parameter.

    Notes
    -----
    The returned parameters are produced by an empirically fitted linear model:
    - ``alpha = -0.0044321370413747405 * mw - 1.1014882364398888``
    - ``loc   =  0.075 * mw - 1.3``
    - ``scale =  0.008058454819492319 * mw + 0.546185725719078``
    """
    alpha = -0.0044321370413747405 * mw - 1.1014882364398888
    loc = 0.075 * mw - 1.3
    scale = 0.008058454819492319 * mw + 0.546185725719078
    return alpha, loc, scale


def _ma_samples(mw: float, n_samples: int) -> np.ndarray:
    """
    Generate Molecular Assembly (MA) complexity estimates from a skew-normal distribution.

    This function samples values from a skew-normal distribution defined by the molecular weight (mw),
    ensuring that all sampled values are non-negative.

    Parameters
    ----------
    mw : float
        Molecular weight (mass) used to calculate the skew-normal distribution parameters.
    n_samples : int
        Number of samples to generate from the distribution.

    Returns
    -------
    numpy.ndarray
        An array of non-negative MA complexity estimates sampled from the skew-normal distribution.
    """
    alpha, loc, scale = _ma_distribution_params(mw)
    return np.maximum(skewnorm(alpha, loc, scale).rvs(n_samples), 0.)


def unify_trees(trees: List[Optional[Dict]]) -> Dict:
    """
    Merge multiple fragmentation trees into one unified structure.

    Parameters
    ----------
    trees : list of dict
        A list of fragmentation trees, where each tree is represented as a dictionary.

    Returns
    -------
    dict
        A unified tree structure. Keys unique to each tree are preserved as-is,
        while shared keys are recursively unified.

    Notes
    -----
    - If the input list is empty, an empty dictionary is returned.
    - If the input list contains only one tree, that tree is returned as-is.
    - For multiple trees, keys unique to each tree are preserved, and shared keys
      are recursively unified by calling `unify_trees` on their values.
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
            **{k: unify_trees([child1[k], child2[k]]) for k in common_keys},
        }


def meta_tree(samples: List[Dict], meta_parent_mz: float = 1e6) -> Dict[float, Dict]:
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
    tree = unify_trees(samples)
    return {meta_parent_mz: tree}


class MAEstimator:
    """
    Estimate Molecular Assembly (MA) values given MS fragmentation trees.

    This class provides methods to estimate Molecular Assembly (MA) values
    based on fragmentation trees. It supports recursive decomposition of
    molecular weights into fragments and evaluates pathway steps to compute
    MA values.

    Jirasek, M., (2024).
    Investigating and quantifying molecular complexity using assembly theory and spectroscopy.
    ACS Central Science, 10(5), 1054-1064.
    """

    def __init__(self,
                 same_level: bool = True,
                 tol: float = 0.01,
                 adduct_masses: Optional[List[float]] = None,
                 n_samples: int = 20,
                 min_chunk: float = 20.0) -> None:
        """
        Initialize the MAEstimator instance.

        Parameters
        ----------
        same_level : bool, optional
            Whether to allow fallback to sibling peaks at the same fragmentation level. Defaults to True.
        tol : float, optional
            Tolerance for matching mass-to-charge ratios (m/z). Defaults to 0.01.
        adduct_masses : list of float, optional
            List of adduct masses to consider for parent-child relationships. Defaults to [0.0, 1.007825].
        n_samples : int, optional
            Number of samples to generate for MA complexity estimates. Defaults to 20.
        min_chunk : float, optional
            Minimum molecular weight (MW) that can be considered a fragment. Defaults to 20.0.
        """
        # Source: https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses
        if adduct_masses is None:
            adduct_masses = [
                0.0,  # Nothing
                1.007825,  # H+
            ]
        self.same_level = same_level
        self.tol = tol
        self.adduct_masses = adduct_masses
        self.n_samples = n_samples
        self.zero = np.zeros(n_samples)
        self.min_chunk = min_chunk  # Minimum MW of what can be considered a fragment

    @functools.cache
    def estimate_by_mw(self, mw: float, has_children: bool) -> np.ndarray:
        """
        Estimate MA values based on molecular weight.

        If the molecular weight matches an element isotope, the MA value is set to zero.
        Otherwise, values are drawn from an empirical skew-normal distribution.

        Parameters
        ----------
        mw : float
            Molecular weight to estimate MA values for.
        has_children : bool
            Whether the molecular weight has child fragments.

        Returns
        -------
        numpy.ndarray
            An array of MA values, either zeros or sampled from the distribution.
        """
        lower, upper = mw - self.tol, mw + self.tol
        if not has_children:
            for isotope, weight in ISOTOPES.items():
                if lower < weight < upper:
                    return self.zero
        return _ma_samples(mw, self.n_samples)

    def estimate_ma(self,
                    tree: Dict[float, Dict],
                    mw: Optional[float] = None,
                    progress_levels: int = 0,
                    joint: bool = False) -> np.ndarray:
        """
        Estimate MA values recursively for a fragmentation tree.

        This method decomposes a molecular weight into child fragments, evaluates
        pathway steps, and combines MA values from the fragmentation hierarchy.

        Parameters
        ----------
        tree : dict[float, dict]
            A nested dictionary representing the fragmentation tree.
        mw : float, optional
            Molecular weight to start the estimation from. Defaults to the root of the tree.
        progress_levels : int, optional
            Number of levels to print progress for. Defaults to 0.
        joint : bool, optional
            Whether to compute joint MA values for all children. Defaults to False.

        Returns
        -------
        float
            The estimated MA value for the given molecular weight.
        """
        if mw is None:
            mw = tree[0]

        children = unify_trees([tree.get(mw, None) or self.precursors(tree, mw)])
        if joint:
            return sum(self.estimate_ma(children, child, progress_levels - 1) for child in children)
        child_estimates = {mw: self.estimate_by_mw(mw, bool(children))}

        for child in children:
            complement = mw - child
            if complement < self.min_chunk or child < self.min_chunk:
                continue

            # Detect shared precursor masses
            common = [
                p
                for p in self.common_precursors(children, child, complement)
                if p > self.min_chunk and max(child - p, complement - p) > self.min_chunk
            ]

            if common and progress_levels > 0:
                print(f"Common precursors of {mw} = {child} + {complement}: {common}")

            # Simple child + complement with no common precursors
            ma_candidates = [
                self.estimate_ma(children, child, progress_levels - 1)
                + self.estimate_ma(children, complement, progress_levels - 1)
                + 1.0
            ]

            for precursor in common:
                chunks = [child - precursor, complement - precursor, precursor]
                if min(chunks) < self.min_chunk:
                    continue
                chunk_mas = sum(
                    self.estimate_ma(
                        children,
                        chunk,
                        progress_levels - 1,
                    )
                    for chunk in chunks
                )
                ma_candidates.append(chunk_mas + 3)

            # Pick minimum-mean pathway
            child_estimates[child] = min(ma_candidates, key=np.mean)
            if progress_levels > 0:
                print(f"MA({mw} = {child} + {complement}) = {child_estimates[child].mean()}")

        estimate = min(child_estimates.values(), key=np.mean)
        return estimate

    def common_precursors(self, data: Dict, parent1: float, parent2: float) -> set:
        """
        Find shared parent peaks for two fragments.

        Parameters
        ----------
        data : dict
            Fragmentation tree data.
        parent1 : float
            First fragment mass.
        parent2 : float
            Second fragment mass.

        Returns
        -------
        set
            A set of shared parent peaks.
        """
        precursors1 = self.precursors(data, parent1)
        precursors2 = self.precursors(data, parent2)
        return set(precursors1).intersection(precursors2)

    def precursors(self, data: Dict, parent: float) -> Dict:
        """
        Find potential parent peaks for a given fragment.

        Parameters
        ----------
        data : dict
            Fragmentation tree data.
        parent : float
            Fragment mass to find parents for.

        Returns
        -------
        dict
            A dictionary of potential parent peaks and their children.
        """
        if parent < self.min_chunk:
            return {}

        # Check for possible adduct-variant parent relationships
        possible_ions = [parent + adduct for adduct in self.adduct_masses]
        parent_candidates = [
            d
            for d in data
            if any(d - self.tol < p < d + self.tol for p in possible_ions)
        ]

        # Combine known children from matching candidates
        children = unify_trees([
            {
                **{
                    p - child: self.same_level_precursors(data, p - child)
                    for child in data[p] or {}
                },
                **(data[p] or {}),
            }
            for p in parent_candidates
        ])
        # Fallback: siblings at the same depth if tree data missing
        if not children and self.same_level:
            children = unify_trees([
                self.same_level_precursors(data, p)
                for p in parent_candidates or possible_ions
            ])

        # Remove invalid (heavier-than-parent) children
        return {k: v for k, v in children.items() if 0 < k < parent}

    def same_level_precursors(self, data: Dict, parent: float) -> Dict:
        """
        Find valid associations at the same fragmentation level.

        Parameters
        ----------
        data : dict
            Fragmentation tree data.
        parent : float
            Parent mass to find associations for.

        Returns
        -------
        dict
            A dictionary of valid associations at the same level.
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


def _build_tree(data: Dict,
                level: int = 1,
                acc: Optional[Dict] = None,
                parent: Optional[float] = None,
                max_level: int = 3) -> Optional[Dict]:
    """
    Build a nested fragmentation tree from MSn DataFrames.

    This function recursively constructs a tree structure where each level groups
    child peaks by their parent peak reference. The tree is built up to a specified
    maximum level.

    Parameters
    ----------
    data : dict
        A dictionary containing MSn DataFrames, where each key represents a level
        and the value is a DataFrame with mass-to-charge ratio (m/z) and parent information.
    level : int, optional
        The current level of the tree being processed. Defaults to 1.
    acc : dict, optional
        The accumulator dictionary that stores the tree structure. Defaults to None.
    parent : float, optional
        The parent peak reference for the current level. Defaults to None.
    max_level : int, optional
        The maximum depth of the tree to construct. Defaults to 3.

    Returns
    -------
    dict or None
        The constructed tree as a nested dictionary if `level` is 1, otherwise None.

    Notes
    -----
    - At the first level, the function initializes the accumulator dictionary.
    - For levels beyond the maximum level, the function returns None.
    - The function joins the current MS level to its parent level using parent IDs
      and groups child peaks by matching parent m/z values.
    """
    max_level = min(max_level, max(data))
    if level == 1:
        acc = {}
        for peak in data[1]["mz"]:
            acc[peak] = {}
            _build_tree(data, level=2, acc=acc[peak], parent=peak, max_level=max_level)
        return acc
    if level > max_level:
        return None

    # Join current MS level to its parent via parent IDs
    level_df = data[level]
    parent_df = data[level - 1].drop(columns=["parent_id"], errors="ignore")
    level_df = level_df.join(parent_df, on="parent_id", rsuffix="_parent")

    # Group children by matching parent m/z
    child_peaks = level_df[level_df["mz_parent"] == parent]["mz"].unique()
    for peak in child_peaks:
        acc[peak] = None if level == max_level else {}
        _build_tree(
            data, level=level + 1, acc=acc[peak], parent=peak, max_level=max_level
        )
    return None


def build_tree(data: Dict, max_level: int = 3) -> Dict:
    """
    Simple wrapper for initiating tree construction.

    This function serves as a wrapper around `_build_tree` to construct a nested
    fragmentation tree from the given data. It validates the input data and ensures
    that the tree construction process starts with the specified maximum depth.

    Parameters
    ----------
    data : dict
        A dictionary containing MSn DataFrames, where each key represents a level
        and the value is a DataFrame with mass-to-charge ratio (m/z) and parent information.
    max_level : int, optional
        The maximum depth of the tree to construct. Defaults to 3.

    Returns
    -------
    dict
        The constructed tree as a nested dictionary.

    Raises
    ------
    TypeError
        If the input `data` is not a dictionary.
    Exception
        If the input `data` is an empty dictionary.
    """
    if type(data) != dict:
        raise TypeError("Data input is not a dictionary")
    elif data == {}:
        raise Exception("Data input is empty")
    return _build_tree(data, max_level=max_level)


def tree_depth(tree: Dict) -> int:
    """
    Compute the maximum depth of a fragmentation tree.

    This function calculates the maximum depth of a nested dictionary structure,
    where each level represents a deeper level of fragmentation.

    Parameters
    ----------
    tree : dict
        A dictionary representing the fragmentation tree.

    Returns
    -------
    int
        The maximum depth of the tree. Returns 0 if the tree is empty or not a dictionary.
    """
    if isinstance(tree, dict) and len(tree) > 0:
        return 1 + max(tree_depth(v) for v in tree.values())
    else:
        return 0


def print_tree(tree: Dict, indent: int = 0, max_depth: int = 10) -> None:
    """
    Pretty print a fragmentation tree.

    Recursively prints the structure of a fragmentation tree in a readable format.

    Parameters
    ----------
    tree : dict
        The fragmentation tree as a nested dictionary.
    indent : int, optional
        Current indentation level. Defaults to 0.
    max_depth : int, optional
        Maximum depth to print. Defaults to 10.
    """
    if indent > max_depth or not isinstance(tree, dict):
        return

    for mz, children in sorted(tree.items(), reverse=True):
        print("  " * indent + f"├─ m/z: {mz:.2f}")
        print_tree(children, indent + 1, max_depth)
