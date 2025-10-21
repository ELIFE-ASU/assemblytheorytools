import functools
import logging
from collections import defaultdict

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

# Monoisotopic masses of common adduct ions
COMMON_PRECURSORS = [
    # Source: https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses
    0.0,  # Nothing
    1.007825,  # H+
]

# Minimum MW of what can be considered a fragment
MIN_CHUNK = 20.0


def ma_distribution_params(mw):
    alpha = -0.0044321370413747405 * mw + -1.1014882364398888
    loc = 0.075 * mw - 1.3
    scale = 0.008058454819492319 * mw + 0.546185725719078
    return alpha, loc, scale


def ma_samples(mw, n_samples):
    alpha, loc, scale = ma_distribution_params(mw)
    return np.maximum(skewnorm(alpha, loc, scale).rvs(n_samples), 0.)


def unify_trees(trees: list[dict]):
    """
    Recursively merge `trees` into one tree.
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


class MAEstimator:
    def __init__(self, same_level=True, tol=0.01, adduct_masses=COMMON_PRECURSORS, n_samples=20):
        self.same_level = same_level
        self.tol = tol
        self.adduct_masses = adduct_masses
        self.n_samples = n_samples
        self.zero = np.zeros(n_samples)

    @functools.cache
    def estimate_by_MW(self, mw, has_children):
        lower, upper = mw - self.tol, mw + self.tol
        if not has_children:
            for isotope, weight in ISOTOPES.items():
                if lower < weight < upper:
                    # MW matches an isotope; MA = 0
                    print(f"HIT: {mw} ~ {isotope} ({weight})")
                    return self.zero
        return ma_samples(mw, self.n_samples)

    def estimate_MA(self, tree: dict[float, dict], mw: float, progress_levels=0, joint=False):
        children = unify_trees([tree.get(mw, None) or self.precursors(tree, mw)])
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
        precursors1 = self.precursors(data, parent1)
        precursors2 = self.precursors(data, parent2)
        return set(precursors1).intersection(precursors2)

    def precursors(self, data, parent):
        if parent < MIN_CHUNK:
            return {}

        possible_ions = [parent + adduct for adduct in self.adduct_masses]
        parent_candidates = [
            d
            for d in data
            if any(d - self.tol < p < d + self.tol for p in possible_ions)
        ]
        children = unify_trees(
            [
                {
                    **{
                        p - child: self.same_level_precursors(data, p - child)
                        for child in data[p] or {}
                    },
                    **(data[p] or {}),
                }
                for p in parent_candidates
            ]
        )
        if not children and self.same_level:
            children = unify_trees(
                [
                    self.same_level_precursors(data, p)
                    for p in parent_candidates or possible_ions
                ]
            )

        # sometimes child peaks are heavier than parent
        return {k: v for k, v in children.items() if 0 < k < parent}

    def same_level_precursors(self, data, parent):
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
    max_level = min(max_level, max(data))
    if level == 1:
        acc = {}
        for peak in data[1]["mz"]:
            acc[peak] = {}
            _build_tree(data, level=2, acc=acc[peak], parent=peak, max_level=max_level)
        return acc
    if level > max_level:
        return
    level_df = data[level]
    parent_df = data[level - 1].drop(columns=["parent_id"], errors="ignore")
    level_df = level_df.join(parent_df, on="parent_id", rsuffix="_parent")
    child_peaks = level_df[level_df["mz_parent"] == parent]["mz"].unique()
    for peak in child_peaks:
        acc[peak] = None if level == max_level else {}
        _build_tree(
            data, level=level + 1, acc=acc[peak], parent=peak, max_level=max_level
        )


def build_tree(data: dict, max_level=3):
    """
    Build a tree from a dictionary of {level: DataFrame, ...}.

    Args:
        data (dict): {level: DataFrame, ...}.
        max_level (int): maximum MS level to include in the tree, e.g. 3 for MS3.
    """
    return _build_tree(data, max_level=max_level)


def tree_depth(tree: dict):
    """
    Calculate the level of nesting in a tree.

    >>> tree_depth({})
    0
    >>> tree_depth({1: None})
    1
    >>> tree_depth({1: {}})
    1
    >>> tree_depth({1: {2: {3: None}}})
    3
    >>> tree_depth({1: {2: {3: None}, 4: None}})
    3
    >>> tree_depth({1: {2: {3: None}, 4: {5: None}}})
    4
    """
    if isinstance(tree, dict) and len(tree) > 0:
        return 1 + max(tree_depth(v) for v in tree.values())
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
        .groupby("parent_bin")
        .apply(filter_fn)
        .reset_index()
    )

    result = (
        ms_df.groupby("parent_rounded")
        .apply(lambda g: g.sort_values("intensity").tail(max_num_peaks))
        .reset_index(drop=True)
    )

    logging.debug(f"Level {level}: {len(result)} out of {original_len} peaks retained")
    return result


def process(
        sample: dict[int, pd.DataFrame],
        max_num_peaks: int = 200,
        min_abs_intensity: dict[int, float] = defaultdict(lambda: 0.0),
        min_rel_intensity: float = 0.0,
        n_digits: int = 3,
) -> dict[int, pd.DataFrame]:
    """
    Process a sample by binning peak intensities and removing low-intensity peaks.

    Args:
        sample (dict[int, pd.DataFrame]): {level: DataFrame, ...}.
        max_num_peaks (int): maximum number of peaks to keep per parent.
        min_abs_intensity (dict[int, float]): minimum absolute child peak intensity by level.
        min_rel_intensity (float): minimum relative child peak intensity, relative to parent's most intense child peak.
        n_digits (int): Number of digits to round m/z values for intensity binning.
            Full precision retained in final output by taking median m/z.

    Returns:
        dict[int, pd.DataFrame]: {level: DataFrame, ...}.
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
            {
                "mz": [parent_peak],
                "intensity": 100000.0,
                "mz_bin": [int(parent_peak * 10 ** n_digits)],
            }
        )
    return sample


def identify_parents(dataset, mass_tol: float, ms_n_digits: int):
    """
    Link MSn+1 peaks to MSn peaks.

    Args:
        dataset (dict[int, pd.DataFrame]): {level: DataFrame, ...}.
            Each DataFrame must have columns "mz_bin" and "parent_bin", created by the process() function.
            These values are used to link parent and child peaks.
        mass_tol (float): mass tolerance in Da.
        ms_n_digits (int): number of decimal places used in binning m/z values.
            **The same value used in process() must be specified here.**
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
                tolerance=int(MASS_TOL * 10 ** MS_N_DIGITS),
                direction="nearest",
            )
            .rename(columns={"index": "parent_id"})
            .dropna(subset=["parent_id"])
            .astype({"parent_id": int})
            .drop(columns=["mz_bin_x"])
        )
    return new_dataset
