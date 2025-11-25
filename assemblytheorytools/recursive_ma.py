import functools
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
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


def _ma_distribution_params(mw):
    """
    Convert mass (mw) into skew-normal distribution parameters.

    These parameters are based on an empirically fitted model that estimates 
    molecular assembly (MA) as a function of mass.
    """
    alpha = -0.0044321370413747405 * mw + -1.1014882364398888
    loc = 0.075 * mw - 1.3
    scale = 0.008058454819492319 * mw + 0.546185725719078
    return alpha, loc, scale


def _ma_samples(mw, n_samples):
    """
    Sample MA complexity estimates from a skew-normal distribution
    while enforcing non-negative results.
    """
    alpha, loc, scale = _ma_distribution_params(mw)
    return np.maximum(skewnorm(alpha, loc, scale).rvs(n_samples), 0.)


def unify_trees(trees: list[dict]):
    """
    Merge multiple fragmentation trees into one unified structure.

    Rules:
    - Keys unique to each tree are preserved as-is.
    - Shared keys are recursively unified.
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


def meta_tree(samples: list[dict], meta_parent_mz: float = 1e6) -> dict:
    """
    Encapsulate multiple trees under a single 'meta' parent precursor.
    Useful for estimating Joint MA of multi-molecular samples.
    """
    tree = unify_trees(samples)
    return {meta_parent_mz: tree}


class MAEstimator:
    """
    Estimate Molecular Assembly (MA) values given MS fragmentation trees.
    """

    def __init__(self,
                 same_level=True,
                 tol=0.01,
                 adduct_masses=None,
                 n_samples=20,
                 min_chunk=20.0):

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
    def estimate_by_mw(self, mw, has_children):
        """
        If the mass matches an element isotope → MA = zero.
        Otherwise → draw from empirical distribution.
        """
        lower, upper = mw - self.tol, mw + self.tol
        if not has_children:
            for isotope, weight in ISOTOPES.items():
                if lower < weight < upper:
                    # MW matches an isotope; MA = 0
                    # print(f"HIT: {mw} ~ {isotope} ({weight})")
                    return self.zero
        return _ma_samples(mw, self.n_samples)

    def estimate_ma(self, tree: dict[float, dict], mw: float = None, progress_levels=0, joint=False):
        """
        Recursive MA estimation:
        - Decompose mass into children and evaluate pathway steps.
        - Combine MAs from fragmentation hierarchy.
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
        return np.mean(estimate)

    def common_precursors(self, data, parent1, parent2):
        """Find shared parent peaks that could generate both fragments."""
        precursors1 = self.precursors(data, parent1)
        precursors2 = self.precursors(data, parent2)
        return set(precursors1).intersection(precursors2)

    def precursors(self, data, parent):
        """
        Find peaks that could serve as parents for both masses
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

    def same_level_precursors(self, data, parent):
        """Search masses at the same fragmentation level for valid associations."""
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
    Build nested fragmentation tree from MSn DataFrames:
    - Each level groups child peaks by their parent peak reference.
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


def build_tree(data: dict, max_level=3):
    """Simple wrapper for initiating tree construction."""
    if type(data) != dict:
        raise TypeError("Data input is not a dictionary")
    elif data == {}:
        raise Exception("Data input is empty")
    return _build_tree(data, max_level=max_level)


def tree_depth(tree: dict):
    """Compute maximum fragmentation depth."""
    if isinstance(tree, dict) and len(tree) > 0:
        return 1 + max(tree_depth(v) for v in tree.values())
    else:
        return 0
