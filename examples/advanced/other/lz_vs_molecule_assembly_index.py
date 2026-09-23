import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import pearsonr, spearmanr

import assemblytheorytools as att

# Setting plot aesthetics for better visibility
plt.rcParams['axes.linewidth'] = 2.0

N_MOLECULES = 1000
# The sampler counts bonds with hydrogens explicit, so this admits molecules of
# roughly 35 heavy-atom bonds -- small enough for the exact assembly index
MAX_BONDS = 45
SEED = 0
# CIDs per PubChem request. Roughly half of a random batch survives the size
# filter, so this collects a few hundred molecules per call -- large enough to
# keep the request count down, small enough that no single request is huge
BATCH_SIZE = 500
CACHE_FILE = "pubchem_lz_sample.csv"


def sample_pubchem_smiles(n, seed, cache_file):
    """
    Draw random PubChem molecules, caching the result to disk.

    Sampling walks random compound IDs over the live PubChem API, so the
    answer is kept in a CSV: a rerun to adjust the figure should not go back
    to the service for data it already has.

    Args:
        n (int): The number of molecules to sample.
        seed (int): Random seed, for a reproducible set of compound IDs.
        cache_file (str): Path of the CSV holding a previous sample.

    Returns:
        pandas.DataFrame: A frame with 'cid' and 'smiles' columns.
    """
    if os.path.exists(cache_file):
        print(f"Reusing the cached sample in {cache_file}", flush=True)
        return pd.read_csv(cache_file)

    print(f"Sampling {n} random molecules from PubChem", flush=True)
    ids, smiles = att.sample_random_pubchem(n, seed=seed, max_bonds=MAX_BONDS,
                                            batch_size=BATCH_SIZE)
    df = pd.DataFrame({"cid": ids, "smiles": smiles})
    df.to_csv(cache_file, index=False)
    return df


def molecule_measures(smi):
    """
    Calculate the assembly index and three compressed SMILES sizes for one molecule.

    The assembly index is taken on the heavy-atom graph, and the compressors
    are handed the heavy-atom SMILES via `add_hydrogens=False`, so both sides
    describe the same object rather than one of them carrying hydrogens the
    other never sees.

    Defined at module level so that `att.mp_calc` can pickle it by reference.

    Args:
        smi (str): The SMILES string of the molecule.

    Returns:
        tuple or None: (non-H bonds, assembly index, zlib, bz2, lzma) bytes,
        or None if the molecule will not parse or the index fails.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    # -1 is the calculator's own failure sentinel, not a real index
    ai, _, _ = att.calculate_assembly_index(att.smi_to_nx(smi), strip_hydrogen=True,
                                            exact=True, timeout=600.0)
    if ai < 0:
        return None

    return (att.count_non_h_bonds(mol),
            ai,
            att.compression_zlib_smi(mol, add_hydrogens=False),
            att.compression_bz2_smi(mol, add_hydrogens=False),
            att.compression_lzma_smi(mol, add_hydrogens=False))


def centre_within_groups(values, groups):
    """
    Subtract each group's mean from its members.

    Every measure here grows with the molecule, so a raw correlation mostly
    reports that shared trend. Centring within bond count strips it out and
    leaves only the variation between molecules of the same size.

    Args:
        values (array-like): The values to centre.
        groups (array-like): The group label for each value.

    Returns:
        numpy.ndarray: The values with their own group mean removed.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    centred = np.empty_like(values)
    for group in np.unique(groups):
        mask = groups == group
        centred[mask] = values[mask] - values[mask].mean()
    return centred


def within_group_spearman(x, y, groups):
    """
    Average the Spearman correlation of x against y over the groups separately.

    Groups in which either measure is constant carry no rank information and
    are skipped rather than counted as zero.

    Args:
        x (array-like): The first measure.
        y (array-like): The second measure.
        groups (array-like): The group label for each observation.

    Returns:
        float: The mean within-group Spearman rho, or nan if no group qualifies.
    """
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    rhos = [
        spearmanr(x[mask], y[mask])[0]
        for mask in (groups == group for group in np.unique(groups))
        if np.ptp(x[mask]) > 0 and np.ptp(y[mask]) > 0
    ]
    return float(np.mean(rhos)) if rhos else float("nan")


def mean_by_x(x, y):
    """
    Average the y values sharing each distinct x value.

    Both measures are integers, so a raw scatter stacks many molecules on one
    point. The per-x mean draws the trend that the overplotting hides.

    Args:
        x (array-like): Integer x values.
        y (array-like): Matching y values.

    Returns:
        tuple: (sorted distinct x values, mean y at each of them).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x_unique = np.unique(x)
    return x_unique, np.array([y[x == value].mean() for value in x_unique])


def compare_panel(ax, x, y, bonds, xlab, ylab="Assembly index", trend="mean",
                  fontsize=14):
    """
    Draw one compressed-size-against-assembly-index panel coloured by molecular size.

    Colour is the third variable, and reading it is how you tell a real
    agreement from a shared size trend. If the points grade smoothly from dark
    to bright along the trend, both axes are largely tracking molecular size
    rather than each other; if the colour is scattered, the relation between
    the two measures stands on its own.

    Args:
        ax (matplotlib.axes.Axes): The axis to draw on.
        x (array-like): The compressed SMILES size.
        y (array-like): The assembly index.
        bonds (array-like): Non-hydrogen bond counts, used to colour the points.
        xlab (str): Label for the x-axis.
        ylab (str, optional): Label for the y-axis. Defaults to 'Assembly index'.
        trend (str, optional): 'mean' for the per-x mean of integer measures,
            'fit' for a least-squares line through centred values. Defaults to 'mean'.
        fontsize (int, optional): Font size for the axis labels. Defaults to 14.

    Returns:
        matplotlib.collections.PathCollection: The scatter artist, for the colorbar.
    """
    points = ax.scatter(x, y, c=bonds, cmap="viridis", s=24, alpha=0.7,
                        edgecolors="none")
    if trend == "mean":
        ax.plot(*mean_by_x(x, y), "o-", color="black", lw=2, ms=4, label="Mean")
    else:
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.array([np.min(x), np.max(x)])
        ax.plot(line_x, slope * line_x + intercept, "-", color="black", lw=2,
                label=f"Fit (slope {slope:.2f})")
    ax.legend(fontsize=fontsize - 4)
    att.ax_plot(ax.get_figure(), ax, xlab, ylab, xs=fontsize, ys=fontsize)
    return points


if __name__ == "__main__":
    molecules = sample_pubchem_smiles(N_MOLECULES, SEED, CACHE_FILE)

    print(f"Calculating {len(molecules)} molecules", flush=True)
    results = att.mp_calc(molecule_measures, molecules["smiles"])

    # Molecules that would not parse or whose index failed come back as None
    usable = [row for row in results if row is not None]
    print(f"{len(usable)} of {len(results)} molecules scored on every measure", flush=True)
    bonds, ai, zlib_b, bz2_b, lzma_b = (np.array(column) for column in zip(*usable))

    print(f"Non-H bonds vs assembly index:  Pearson r = "
          f"{pearsonr(bonds, ai)[0]:.3f}", flush=True)
    compressors = [("zlib", zlib_b), ("bz2 ", bz2_b), ("lzma", lzma_b)]
    for name, measure in compressors:
        print(f"{name} vs assembly index:  "
              f"Pearson r = {pearsonr(measure, ai)[0]:.3f}, "
              f"Spearman rho = {spearmanr(measure, ai)[0]:.3f}, "
              f"within-size rho = {within_group_spearman(measure, ai, bonds):.3f}",
              flush=True)

    # The figure asks the same question at two levels.
    #
    # Panels 1 to 3 plot each compressed SMILES size against the assembly index
    # as measured. Any agreement there is suspect for the same reason as in the
    # string case: bond count alone already tracks the assembly index closely,
    # and a bigger molecule writes a longer SMILES, so both axes partly report
    # size rather than structure. The colour gradient along each trend is what
    # that looks like.
    #
    # Panel 4 takes the size out. Centring both measures within their own bond
    # count leaves only how a molecule differs from the other molecules with
    # exactly as many bonds, which asks the question worth asking: among
    # molecules of one size, does compression single out the ones the assembly
    # index calls complex?
    ai_centred = centre_within_groups(ai, bonds)
    zlib_centred = centre_within_groups(zlib_b, bonds)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Panel 1: zlib is DEFLATE, the LZ77 family member behind gzip and zip, and
    # the finest-grained of the three here -- it resolves the most distinct
    # sizes over molecules this small.
    compare_panel(axes[0, 0], zlib_b, ai, bonds, "zlib compressed SMILES (bytes)")

    # Panel 2: bz2 is the odd one out, a Burrows-Wheeler compressor rather than
    # an LZ one. It is included because the package ships it alongside the
    # others, and it is the control: whatever the LZ panels show, a different
    # compression family showing the same thing means the result is about
    # compression in general, not about LZ.
    compare_panel(axes[0, 1], bz2_b, ai, bonds, "bz2 compressed SMILES (bytes)")

    # Panel 3: lzma is LZ77 again, with a much larger fixed container. Its grid
    # is visibly coarser -- over strings this short the container dominates, so
    # it separates molecules into only a handful of distinct sizes.
    points = compare_panel(axes[1, 0], lzma_b, ai, bonds, "lzma compressed SMILES (bytes)")

    # Panel 4: panel 1 with molecular size taken out, so both axes now read as
    # "more or less than a typical molecule with this many bonds". The fitted
    # slope replaces the per-x mean because centring turns an integer axis into
    # a continuous one.
    compare_panel(axes[1, 1], zlib_centred, ai_centred, bonds,
                  "zlib compressed SMILES (bond-centred)",
                  ylab="Assembly index (bond-centred)", trend="fit")

    # One bar for the whole grid: attaching it to every axis takes the width
    # from all four rather than squeezing the last one. The shared scale is only
    # correct because all four panels were handed the same `bonds` array --
    # plot a subset in one of them and it would need an explicit vmin and vmax.
    colorbar = fig.colorbar(points, ax=axes.ravel().tolist(), pad=0.015)
    colorbar.set_label("Non-hydrogen bonds", fontsize=14)
    colorbar.ax.tick_params(labelsize=12)

    plt.savefig("lz_vs_molecule_assembly_index.png", dpi=600, bbox_inches="tight")
    plt.savefig("lz_vs_molecule_assembly_index.pdf", bbox_inches="tight")
    plt.show()
