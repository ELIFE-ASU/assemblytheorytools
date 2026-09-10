"""
Recursive molecular assembly (RMA) estimation.

This module implements a recursive decomposition of a molecule into a tree of
subunits and estimates the molecular assembly index from that tree. It provides
tree construction, depth measurement, unification of equivalent subtrees, parent
identification, and the :class:`MAEstimator` driver class.
"""

import functools
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

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


def ma_distribution_params(mw: float) -> Tuple[float, float, float]:
    """Return skew-normal MA prior parameters for a molecular weight.

    Parameters
    ----------
    mw : float
        Molecular weight in Da.

    Returns
    -------
    tuple of float
        Shape, location, and scale ``(alpha, loc, scale)``.

    Notes
    -----
    The fixed coefficients come from an offline fit of assembly index
    against molecular weight, following Marshall et al.
    [Marshall2021a]_. They are not fitted to the input or checked
    against its chemistry. Equal masses receive the same prior
    regardless of structure; this models typical assembly indices, not a
    particular molecule's index.

    All parameters extrapolate linearly without bounds. Below about 17
    Da, the negative location causes many draws to hit the zero floor in
    `ma_samples`. At several thousand Da, the predicted location becomes
    unrealistically large. Masses outside the ordinary small-molecule
    range are out of scope. Zero and negative masses are not validated.

    References
    ----------
    .. [Marshall2021a] Marshall, S. M. *et al.* (2021). Identifying
       molecules as biosignatures with assembly theory and mass
       spectrometry. Nature Communications, 12, 3033.
       https://doi.org/10.1038/s41467-021-23258-x
    """
    alpha = -0.0044321370413747405 * mw - 1.1014882364398888
    loc = 0.075 * mw - 1.3
    scale = 0.008058454819492319 * mw + 0.546185725719078
    return alpha, loc, scale


def ma_samples(mw: float, n_samples: int) -> np.ndarray:
    """Draw MA prior samples for a molecular weight, clipping negatives.

    Parameters
    ----------
    mw : float
        Molecular weight in Da.
    n_samples : int
        Number of samples to draw.

    Returns
    -------
    np.ndarray
        Non-negative, continuous MA samples.

    Notes
    -----
    The spread represents variation across structures of the same mass,
    not measurement uncertainty for one fragment. Clipping negative
    draws to zero changes the skew-normal distribution and raises its
    mean, particularly at low mass where many samples can become exactly
    zero.

    Sampling uses the global NumPy random state; seed NumPy for
    repeatable results. Samples are continuous although assembly indices
    are integers. Round at the end of an analysis rather than separately
    per fragment.
    """
    alpha, loc, scale = ma_distribution_params(mw)
    return np.maximum(skewnorm(alpha, loc, scale).rvs(n_samples), 0.0)


def rma_unify_trees(trees: list[dict]) -> Dict[float, Any]:
    """Merge the first two fragmentation trees recursively.

    Parameters
    ----------
    trees : list of dict
        Nested fragmentation trees; ``None`` is treated as empty when
        merged.

    Returns
    -------
    dict or None
        The merged tree, or an empty dictionary for an empty list. A
        single input is returned unchanged, including ``None``.

    Notes
    -----
    For compatibility, inputs after the first two are ignored. Shared
    keys are merged recursively; other subtrees are retained unchanged.
    """
    if not trees:
        return {}
    if len(trees) == 1:
        return trees[0]

    first, second = trees[:2]
    first_keys = set(first or {})
    second_keys = set(second or {})
    common_keys = first_keys & second_keys
    # Preserve the merge order: estimator traversal consumes random samples.
    return {
        **{key: first[key] for key in first_keys - common_keys},
        **{key: second[key] for key in second_keys - common_keys},
        **{key: rma_unify_trees([first[key], second[key]]) for key in common_keys},
    }


class MAEstimator:
    """Estimate molecular assembly from fragment masses and observed trees.

    Attributes
    ----------
    same_level : bool
        Enable same-level precursor search when matched parents give no
        children. Complement searches also use same-level peaks.
    tol : float
        Mass tolerance for m/z matching.
    adduct_masses : list of float
        Adduct masses considered in precursor searches.
    n_samples : int
        Number of Monte Carlo samples per mass estimate.
    zero : np.ndarray
        Shared zero array returned for childless isotope matches.

    Notes
    -----
    Results are heuristic estimates, neither exact assembly indices nor
    proven bounds. Only observed peaks inform the search; sparse trees
    rely heavily on the molecular-weight prior in
    `ma_distribution_params`.

    Matching uses mass alone, with no chemical feasibility check.
    Accidental mass coincidences can therefore look like precursor
    relationships. Choose `tol` for the instrument: large tolerances
    admit false matches, while small ones discard real relationships.

    Sampling uses the global NumPy random state. Seed NumPy for
    reproducibility and report the mean with its spread. Decomposition
    searches skip fragments and complements below `MIN_CHUNK` (20 Da),
    so small neutral losses do not contribute to those candidates.
    """

    def __init__(
        self,
        same_level: bool = True,
        tol: float = 0.01,
        adduct_masses: List[float] = COMMON_PRECURSORS,
        n_samples: int = 500,
    ) -> None:
        """Configure precursor matching and the MA sampling budget.

        Parameters
        ----------
        same_level : bool, optional
            Enable same-level fallback searches. Defaults to True.
        tol : float, optional
            Mass tolerance for m/z matching. Defaults to 0.01.
        adduct_masses : list of float, optional
            Adduct masses used in precursor searches. Defaults to
            ``COMMON_PRECURSORS``; the supplied list is retained by
            reference.
        n_samples : int, optional
            Number of MA samples per estimate. Defaults to 500.
        """
        self.same_level = same_level
        self.tol = tol
        self.adduct_masses = adduct_masses
        self.n_samples = n_samples
        self.zero = np.zeros(n_samples)

    @functools.cache
    def estimate_by_MW(self, mw: float, has_children: bool) -> np.ndarray:
        """Return cached MA prior samples, or zeros for a childless isotope.

        Parameters
        ----------
        mw : float
            Molecular weight in Da.
        has_children : bool
            Whether the fragment has children, disabling isotope matching.

        Returns
        -------
        np.ndarray
            MA samples, or the estimator's shared zero array for an isotope
            strictly within `tol` of `mw`.

        Notes
        -----
        Results are cached per estimator and call arguments. Repeated calls
        return the same array; equal-mass fragments can therefore share
        draws, and reuse does not average away sampling error. Create a new
        estimator for independent draws.

        Isotope matching uses mass alone without charge or adduct
        correction. A coincident ion mass is treated as a bare atom;
        elements excluded from `ISOTOPES` cannot match. Fragments with
        children always use the prior, regardless of their mass.
        """
        lower, upper = mw - self.tol, mw + self.tol
        if not has_children:
            for isotope, weight in ISOTOPES.items():
                if lower < weight < upper:
                    # MW matches an isotope; MA = 0
                    print(f"HIT: {mw} ~ {isotope} ({weight})")
                    return self.zero
        return ma_samples(mw, self.n_samples)

    def estimate_MA(
        self,
        tree: dict[float, dict],
        mw: float,
        progress_levels: int = 0,
        joint: bool = False,
    ) -> np.ndarray:
        """Recursively select an MA estimate from observed decompositions.

        Parameters
        ----------
        tree : dict[float, dict]
            Fragmentation tree as nested dictionaries keyed by m/z.
        mw : float
            Molecular weight to estimate.
        progress_levels : int, optional
            Number of recursion levels with diagnostic printing. Defaults to
            0 (silent); this does not limit recursion or change the
            estimate.
        joint : bool, optional
            Sum estimates for all children at this node without a joining
            cost. Descendants use the default decomposition search. If there
            are no children, use the molecular-weight prior. Defaults to
            False.

        Returns
        -------
        np.ndarray
            Estimated MA samples for the molecular weight.

        Notes
        -----
        This greedy heuristic retains the candidate with the lowest sample
        mean at each node. It considers only observed fragments, rather than
        all possible decompositions, and gives neither an exact assembly
        index nor a proven bound. More observed fragments generally lower
        the estimate; under-fragmented spectra therefore tend to read high.

        Joining costs are fixed heuristics: 1 for a child/complement pair
        and 3 for a split using a common precursor. These costs stand in for
        assembly joins and are not derived from the spectrum. Fragments and
        complements below `MIN_CHUNK` (20 Da) are excluded from candidates.

        In the default strategy, the molecular-weight prior remains a
        candidate and is the fallback when no usable children exist.
        Without fragmentation data, it supplies the entire estimate.
        Recursive tree estimates are not memoised across branches, though
        `estimate_by_MW` caches draws from the prior.
        """
        children = tree.get(mw) or self.precursors(tree, mw)
        if not children:
            return self.estimate_by_MW(mw, False)

        next_level = progress_levels - 1
        if joint:
            return sum(
                self.estimate_MA(children, child, next_level) for child in children
            )

        estimates = [self.estimate_by_MW(mw, True)]

        for child in children:
            complement = mw - child
            if complement < MIN_CHUNK or child < MIN_CHUNK:
                continue

            common = [
                precursor
                for precursor in self.common_precursors(children, child, complement)
                if precursor > MIN_CHUNK
                and max(child - precursor, complement - precursor) > MIN_CHUNK
            ]

            if common and progress_levels > 0:
                print(f"Common precursors of {mw} = {child} + {complement}: {common}")

            # Simple child + complement with no common precursors
            ma_candidates = [
                self.estimate_MA(children, child, next_level)
                + self.estimate_MA(children, complement, next_level)
                + 1.0
            ]

            for precursor in common:
                chunks = [child - precursor, complement - precursor, precursor]
                if min(chunks) < MIN_CHUNK:
                    continue
                chunk_mas = sum(
                    self.estimate_MA(children, chunk, next_level) for chunk in chunks
                )
                ma_candidates.append(chunk_mas + 3)

            best = min(ma_candidates, key=np.mean)
            estimates.append(best)
            if progress_levels > 0:
                print(f"MA({mw} = {child} + {complement}) = {best.mean()}")

        return min(estimates, key=np.mean)

    def common_precursors(
        self, data: Dict[float, Any], parent1: float, parent2: float
    ) -> set:
        """Return precursor masses shared by two parent-ion searches.

        Parameters
        ----------
        data : dict
            Fragmentation tree keyed by m/z.
        parent1 : float
            m/z of the first parent ion.
        parent2 : float
            m/z of the second parent ion.

        Returns
        -------
        set
            Exact intersection of masses returned by `precursors` for each
            parent; the intersection itself applies no additional tolerance.
        """
        precursors1 = self.precursors(data, parent1)
        precursors2 = self.precursors(data, parent2)
        return set(precursors1).intersection(precursors2)

    def precursors(self, data: Dict[float, Any], parent: float) -> Dict[float, Any]:
        """Find child and complementary fragments for a parent mass.

        Parameters
        ----------
        data : dict
            Fragmentation tree keyed by m/z.
        parent : float
            Parent-ion m/z before adding candidate adduct masses.

        Returns
        -------
        dict
            Candidate fragment masses and subtrees, restricted to masses
            strictly between zero and `parent`. Empty below `MIN_CHUNK`.

        Notes
        -----
        Match tree keys to `parent` plus each adduct, strictly within `tol`.
        Combine matched parents' children with their mass complements; each
        complement subtree comes from `same_level_precursors`. If the merged
        tree is empty and `same_level` is enabled, search same-level peaks
        using matched parent masses, or adduct-adjusted parent masses when
        none match. Tree combinations follow the legacy behavior of
        `rma_unify_trees`.
        """
        if parent < MIN_CHUNK:
            return {}

        possible_ions = [parent + adduct for adduct in self.adduct_masses]
        parent_candidates = [
            mass
            for mass in data
            if any(mass - self.tol < ion < mass + self.tol for ion in possible_ions)
        ]
        candidate_trees = []
        for mass in parent_candidates:
            fragments = data[mass] or {}
            complements = {
                mass - child: self.same_level_precursors(data, mass - child)
                for child in fragments
            }
            candidate_trees.append({**complements, **fragments})

        children = rma_unify_trees(candidate_trees)
        if not children and self.same_level:
            children = rma_unify_trees(
                [
                    self.same_level_precursors(data, p)
                    for p in parent_candidates or possible_ions
                ]
            )

        # Observed peaks can be heavier than their parent.
        return {
            mass: subtree for mass, subtree in children.items() if 0 < mass < parent
        }

    def same_level_precursors(
        self, data: Dict[float, Any], parent: float
    ) -> Dict[float, Any]:
        """Find same-level ions with an observed mass complement.

        Parameters
        ----------
        data : dict
            Same-level ion masses mapped to their subtrees.
        parent : float
            Parent-ion m/z to split.

        Returns
        -------
        dict
            Ions whose ``parent - ion + adduct`` lies strictly within `tol`
            of any key in `data`, retaining their original subtrees.
        """
        result = {}
        adducts, tol = self.adduct_masses, self.tol
        for ion, subtree in data.items():
            target = parent - ion
            if any(
                mass - tol < target + adduct < mass + tol
                for mass in data
                for adduct in adducts
            ):
                result[ion] = subtree
        return result


def _build_tree(
    data: Dict[int, pd.DataFrame],
    level: int = 1,
    acc: Optional[Dict[float, Any]] = None,
    parent: Optional[float] = None,
    max_level: int = 3,
) -> Optional[Dict[float, Any]]:
    """Build the root tree or populate a subtree from linked MS levels.

    Parameters
    ----------
    data : dict
        MS levels mapped to peak DataFrames with ``mz`` and, above MS1,
        ``parent_id`` referring to the previous level's row index.
    level : int, optional
        Current MS level. Defaults to 1.
    acc : dict or None, optional
        Subtree to populate in place. A new root dictionary is created
        at level 1, regardless of this argument.
    parent : float or None, optional
        Parent m/z used to select children above level 1.
    max_level : int, optional
        Last MS level to include, capped at the highest level in `data`.
        Defaults to 3.

    Returns
    -------
    dict or None
        Root tree at level 1; otherwise ``None`` after updating `acc`,
        or immediately when beyond `max_level`.
    """
    max_level = min(max_level, max(data))
    if level == 1:
        acc = {}
        for peak in map(float, data[1]["mz"]):
            acc[peak] = {}
            _build_tree(data, level=2, acc=acc[peak], parent=peak, max_level=max_level)
        return acc
    if level > max_level:
        return
    level_df = data[level]
    parent_df = data[level - 1].drop(columns=["parent_id"], errors="ignore")
    level_df = level_df.join(parent_df, on="parent_id", rsuffix="_parent")
    child_peaks = level_df[level_df["mz_parent"] == parent]["mz"].unique()
    for peak in map(float, child_peaks):
        acc[peak] = None if level == max_level else {}
        if level < max_level:
            _build_tree(
                data, level=level + 1, acc=acc[peak], parent=peak, max_level=max_level
            )


def rma_build_tree(data: dict, max_level: int = 3) -> Dict[float, Any]:
    """Build a fragmentation tree from linked mass-spectrometry levels.

    Parameters
    ----------
    data : dict
        MS levels mapped to peak DataFrames with ``mz`` and, above MS1,
        ``parent_id`` referring to the previous level's row index.
    max_level : int, optional
        Last MS level to include, capped at the highest available level.
        Defaults to 3.

    Returns
    -------
    dict
        Nested dictionaries keyed by m/z. Terminal nodes at the maximum
        level use ``None``; nodes without children can be empty
        dictionaries.
    """
    return _build_tree(data, max_level=max_level)


def rma_tree_depth(tree: dict) -> int:
    """Return the maximum number of nested levels in a fragmentation tree.

    Parameters
    ----------
    tree : dict
        Tree with m/z keys and subtree values; leaves may be ``None`` or
        empty dictionaries.

    Returns
    -------
    int
        Maximum depth, or 0 for empty or non-dictionary input.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> att.rma_tree_depth({500.0: {400.0: {300.0: {}}}})
    3
    >>> att.rma_tree_depth({})
    0
    """
    if not isinstance(tree, dict) or not tree:
        return 0
    return 1 + max(rma_tree_depth(subtree) for subtree in tree.values())


def _process_df(
    level: int,
    ms_df: pd.DataFrame,
    max_num_peaks: int,
    min_abs_intensity: Dict[int, float],
    min_rel_intensity: float,
    n_digits: int,
) -> pd.DataFrame:
    """Bin, aggregate, and filter peaks at one MS level.

    Parameters
    ----------
    level : int
        MS level used to select the absolute intensity threshold.
    ms_df : pd.DataFrame
        Peak data with ``mz``, ``intensity``, and optionally ``parent``.
        A missing ``parent`` column is added to this frame with value
        1e6.
    max_num_peaks : int
        Maximum retained peaks per parent bin.
    min_abs_intensity : dict
        Absolute intensity thresholds keyed by MS level.
    min_rel_intensity : float
        Intensity threshold as a fraction of each parent's strongest
        peak.
    n_digits : int
        Decimal places used to create integer m/z and parent bins.

    Returns
    -------
    pd.DataFrame
        Aggregated peaks with integer ``mz_bin`` and ``parent_bin``
        columns.

    Notes
    -----
    Only peaks with ``mz < parent - 1`` enter binning. Within each
    ``(mz_bin, parent_bin)`` group, intensities are summed and masses
    use the median. Retain peaks strictly above both intensity
    thresholds, then the strongest `max_num_peaks` per parent bin.
    """
    original_len = len(ms_df)
    bin_scale = 10**n_digits

    if "parent" not in ms_df:
        ms_df["parent"] = 1_000_000
    ms_df = ms_df[ms_df["mz"] < ms_df["parent"] - 1]

    ms_df = ms_df.assign(
        mz_bin=(ms_df["mz"].round(n_digits) * bin_scale).astype(int),
        parent_bin=(ms_df["parent"].round(n_digits) * bin_scale).astype(int),
    )

    grouped = (
        ms_df.groupby(["mz_bin", "parent_bin"])
        .agg({"intensity": "sum", "mz": "median", "parent": "median"})
        .reset_index()
    )

    # Broadcast each parent's intensity floor to its peaks.
    group_max = grouped.groupby("parent_bin")["intensity"].transform("max")
    min_intensity = np.maximum(min_abs_intensity[level], group_max * min_rel_intensity)

    # Keep the strongest peaks per parent without dropping the grouping column.
    result = (
        grouped[grouped["intensity"] > min_intensity]
        .sort_values("intensity")
        .groupby("parent_bin", group_keys=False)
        .tail(max_num_peaks)
    )

    logging.debug(
        "Level %s: %s out of %s peaks retained", level, len(result), original_len
    )
    return result


def rma_process(
    sample: dict[int, pd.DataFrame],
    max_num_peaks: int = 200,
    min_abs_intensity: dict[int, float] | None = None,
    min_rel_intensity: float = 0.0,
    n_digits: int = 3,
) -> dict[int, pd.DataFrame]:
    """Filter and bin each MS level, adding a placeholder MS1 if needed.

    Parameters
    ----------
    sample : dict[int, pd.DataFrame]
        MS levels mapped to DataFrames with ``mz``, ``intensity``, and
        optionally ``parent`` columns.
    max_num_peaks : int, optional
        Maximum retained peaks per parent and MS level. Defaults to 200.
    min_abs_intensity : dict[int, float], optional
        Absolute intensity thresholds by MS level. If omitted, use 0.0
        for every level.
    min_rel_intensity : float, optional
        Threshold as a fraction of each parent's strongest aggregated
        peak intensity. Defaults to 0.0.
    n_digits : int, optional
        Decimal places for m/z binning. Defaults to 3.

    Returns
    -------
    dict[int, pd.DataFrame]
        Processed DataFrames keyed by MS level. If MS1 is absent, create
        it from the parent with the greatest summed intensity in the
        processed MS2 data.
    """
    if min_abs_intensity is None:
        min_abs_intensity = defaultdict(float)
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
        # Use the parent with the largest total retained MS2 intensity.
        parent_peak = (
            sample[2].groupby("parent")["intensity"].sum().sort_values().index[-1]
        )
        sample[1] = pd.DataFrame(
            {
                "mz": [parent_peak],
                "intensity": 100000.0,
                "mz_bin": [int(parent_peak * 10**n_digits)],
            }
        )
    return sample


def rma_identify_parents(
    dataset: Dict[int, pd.DataFrame], mass_tol: float, ms_n_digits: int = 3
) -> Dict[int, pd.DataFrame]:
    """Match peaks to the nearest parent in the preceding MS level.

    Parameters
    ----------
    dataset : dict
        Consecutive MS levels mapped to DataFrames with ``mz_bin`` and,
        above the first level, ``parent_bin`` columns. Parent
        identifiers are the previous level's row indices.
    mass_tol : float
        Matching tolerance in m/z units, scaled to integer bins.
    ms_n_digits : int, optional
        Decimal places used for binning. Defaults to 3.

    Returns
    -------
    dict
        New level mapping with the first DataFrame unchanged. Higher
        levels have integer ``parent_id`` columns and omit unmatched
        peaks.
    """
    first_level = min(dataset)
    new_dataset = {first_level: dataset[first_level]}
    for level in sorted(dataset)[:-1]:
        parents = new_dataset[level][["mz_bin"]].sort_values("mz_bin").reset_index()
        new_dataset[level + 1] = (
            pd.merge_asof(
                dataset[level + 1].sort_values("parent_bin"),
                parents,
                left_on="parent_bin",
                right_on="mz_bin",
                suffixes=("", "_x"),
                tolerance=int(mass_tol * 10**ms_n_digits),
                direction="nearest",
            )
            .rename(columns={"index": "parent_id"})
            .dropna(subset=["parent_id"])
            .astype({"parent_id": int})
            .drop(columns=["mz_bin_x"])
        )
    return new_dataset


def _tol_from_decimals(decimals: Optional[int]) -> float:
    """Convert decimal precision to a mass tolerance.

    Parameters
    ----------
    decimals : int or None
        Decimal places defining the tolerance, or ``None`` for the
        default.

    Returns
    -------
    float
        ``10 ** (-decimals)``, or 0.01 when `decimals` is ``None``.
    """
    return 0.01 if decimals is None else 10 ** (-decimals)


def rma_estimate_ma(
    tree: Dict[float, Any],
    mw: float,
    progress_levels: int = 0,
    joint: bool = False,
    **kwargs: Any,
) -> float:
    """Return the mean recursive MA estimate using a fresh estimator.

    Parameters
    ----------
    tree : dict
        Fragmentation tree as nested dictionaries keyed by m/z.
    mw : float
        Molecular weight to estimate.
    progress_levels : int, optional
        Number of recursion levels with diagnostic printing. Defaults to
        0 (silent); this does not limit recursion or change the
        estimate.
    joint : bool, optional
        Sum child estimates at the requested node without a joining
        cost. Defaults to False. See `MAEstimator.estimate_MA`.
    **kwargs
        Additional arguments for the `MAEstimator` constructor.

    Returns
    -------
    float
        Mean estimated MA for the molecular weight.

    Notes
    -----
    This discards the Monte Carlo sample's spread. Call
    `MAEstimator.estimate_MA` directly to retain samples and report the
    standard deviation with the mean. Results depend on mass tolerance;
    seed the global NumPy random state for reproducibility.

    The result is a heuristic based on observed fragments and mass, not
    the exact assembly index. See `MAEstimator.estimate_MA` for the
    underlying assumptions.
    """
    estimator = MAEstimator(**kwargs)
    result = estimator.estimate_MA(
        tree=tree,
        mw=mw,
        progress_levels=progress_levels,
        joint=joint,
    )
    return float(result.mean())


def rma_estimate_by_mw(
    mw: float, has_children: bool = False, **kwargs: Any
) -> np.ndarray:
    """Return MA prior samples using a fresh estimator.

    Parameters
    ----------
    mw : float
        Molecular weight in Da.
    has_children : bool, optional
        Whether the fragment has children, disabling isotope matching.
        Defaults to False.
    **kwargs
        Additional arguments for the `MAEstimator` constructor.

    Returns
    -------
    np.ndarray
        Estimated MA samples, or zeros for a childless isotope match.

    Notes
    -----
    Without fragmentation data, the prior describes typical assembly
    indices at this mass, not a particular structure's index. Its spread
    represents population variation rather than measurement error.

    When `has_children` is False, a mass strictly within the estimator's
    `tol` of a monoisotopic element returns zeros. A fresh estimator is
    created for each call, so draws are independent rather than reused
    from a previous call's cache. Sampling uses NumPy's global random
    state.
    """
    estimator = MAEstimator(**kwargs)
    return estimator.estimate_by_MW(mw=mw, has_children=has_children)


def find_common_precursors(
    data: Dict[float, Any],
    parent1: float,
    parent2: float,
    same_level: bool = True,
    decimals: Optional[int] = None,
    **kwargs: Any,
) -> set:
    """Find shared precursor masses using a fresh estimator.

    Parameters
    ----------
    data : dict
        Fragmentation tree keyed by m/z.
    parent1 : float
        m/z of the first parent ion.
    parent2 : float
        m/z of the second parent ion.
    same_level : bool, optional
        Enable same-level fallback searches. Defaults to True.
    decimals : int or None, optional
        Decimal places defining tolerance as ``10 ** (-decimals)``.
        Defaults to ``None``, which selects a tolerance of 0.01.
    **kwargs
        Additional arguments for the `MAEstimator` constructor.

    Returns
    -------
    set
        Exact intersection of candidate precursor masses found for each
        parent. See `MAEstimator.precursors` for the search behavior.
    """
    estimator = MAEstimator(
        same_level=same_level, tol=_tol_from_decimals(decimals), **kwargs
    )
    return estimator.common_precursors(data=data, parent1=parent1, parent2=parent2)


def rma_print_tree(tree: Dict, indent: int = 0, max_depth: int = 10) -> None:
    """Print tree masses in descending order, indented by depth.

    Parameters
    ----------
    tree : dict
        Tree with m/z keys and subtrees; leaves may be ``None`` or
        empty.
    indent : int, optional
        Starting indentation level, with two spaces per level. Defaults
        to 0; recursive calls increment it.
    max_depth : int, optional
        Greatest indentation level to print, inclusive. Defaults to 10.

    Returns
    -------
    None
        Prints the tree to standard output with masses rounded to two
        decimal places.
    """
    if indent > max_depth or not isinstance(tree, dict):
        return

    for mz, children in sorted(tree.items(), reverse=True):
        print("  " * indent + f"├─ m/z: {mz:.2f}")
        rma_print_tree(children, indent + 1, max_depth)


def rma_meta_tree(
    samples: List[Dict], meta_parent_mz: float = 1e6
) -> Dict[float, Dict]:
    """Wrap a merged fragmentation tree under a synthetic parent mass.

    Parameters
    ----------
    samples : list of dict
        Fragmentation trees to combine using `rma_unify_trees`.
    meta_parent_mz : float, optional
        Synthetic parent m/z. Defaults to 1e6.

    Returns
    -------
    dict
        One root keyed by `meta_parent_mz`, containing the merged tree.

    Notes
    -----
    The legacy unification behavior combines only the first two samples;
    additional samples are ignored. A single sample is retained
    unchanged.
    """
    return {meta_parent_mz: rma_unify_trees(samples)}
