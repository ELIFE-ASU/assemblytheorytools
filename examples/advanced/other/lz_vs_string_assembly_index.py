import random
import zlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

import assemblytheorytools as att

# Setting plot aesthetics for better visibility
plt.rcParams['axes.linewidth'] = 2.0

N_STRINGS = 1000
MIN_LENGTH = 2
MAX_LENGTH = 50
POOL = "atgc"
SEED = 0


def random_string(length, pool):
    """
    Generate a random string of a specified length using characters from the given pool.

    Args:
        length (int): The length of the random string to generate.
        pool (str or list): The pool of characters to choose from.

    Returns:
        str: A random string of the specified length.
    """
    return "".join(random.choice(pool) for _ in range(length))


def lz78_complexity(s):
    """
    Count the phrases in the LZ78 parse of a string.

    LZ78 walks the string once, and whenever the piece it is holding is not yet
    in its dictionary it emits that piece as a new phrase and adds it. The
    phrase count is the classic Lempel-Ziv complexity, and it is the
    compression measure that lines up most directly with the assembly index:
    both count construction steps in which the only free material is what has
    already been built.

    Args:
        s (str): The input string.

    Returns:
        int: The number of phrases in the LZ78 parse.
    """
    dictionary = set()
    phrase = ""
    n_phrases = 0
    for char in s:
        phrase += char
        if phrase not in dictionary:
            dictionary.add(phrase)
            n_phrases += 1
            phrase = ""
    # A trailing repeat of a phrase already seen still has to be emitted
    if phrase:
        n_phrases += 1
    return n_phrases


def zlib_length(s, level=9):
    """
    Measure the zlib (DEFLATE) compressed size of a string in bytes.

    DEFLATE is the LZ77 family member behind gzip and zip. The empty-string
    overhead is subtracted so the number reflects the payload alone, matching
    the convention used by `att.compression_zlib_smi` for molecules.

    Args:
        s (str): The input string.
        level (int, optional): zlib compression level (0-9). Defaults to 9.

    Returns:
        int: The compressed length in bytes, net of container overhead.
    """
    payload = s.encode("utf-8")
    return len(zlib.compress(payload, level)) - len(zlib.compress(b"", level))


def string_measures(s):
    """
    Calculate the assembly index and both LZ compression measures for one string.

    Defined at module level so that `att.mp_calc` can pickle it by reference.

    Args:
        s (str): The input string.

    Returns:
        tuple: (length, assembly index, LZ78 phrase count, zlib bytes).
    """
    ai, _, _ = att.calculate_string_assembly_index(s, timeout=1000.0)
    return len(s), ai, lz78_complexity(s), zlib_length(s)


def centre_within_groups(values, groups):
    """
    Subtract each group's mean from its members.

    Every measure here grows with string length, so a raw correlation mostly
    reports that shared trend. Centring within length strips it out and leaves
    only the variation between strings of the same size.

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

    Both measures are integers, so a raw scatter stacks many strings on one
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


def compare_panel(ax, x, y, lengths, xlab, ylab="Assembly index", trend="mean",
                  fontsize=15):
    """
    Draw one LZ-measure-against-assembly-index panel coloured by string length.

    Colour is the third variable, and reading it is how you tell a real
    agreement from a shared size trend. If the points grade smoothly from dark
    to bright along the trend, both axes are largely tracking length rather
    than each other; if the colour is scattered, the relation between the two
    measures stands on its own.

    Args:
        ax (matplotlib.axes.Axes): The axis to draw on.
        x (array-like): The LZ compression measure.
        y (array-like): The assembly index.
        lengths (array-like): String lengths, used to colour the points.
        xlab (str): Label for the x-axis.
        ylab (str, optional): Label for the y-axis. Defaults to 'Assembly index'.
        trend (str, optional): 'mean' for the per-x mean of integer measures,
            'fit' for a least-squares line through centred values. Defaults to 'mean'.
        fontsize (int, optional): Font size for the axis labels. Defaults to 15.

    Returns:
        matplotlib.collections.PathCollection: The scatter artist, for the colorbar.
    """
    points = ax.scatter(x, y, c=lengths, cmap="viridis", s=24, alpha=0.7,
                        edgecolors="none")
    if trend == "mean":
        ax.plot(*mean_by_x(x, y), "o-", color="black", lw=2, ms=5, label="Mean")
    else:
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.array([np.min(x), np.max(x)])
        ax.plot(line_x, slope * line_x + intercept, "-", color="black", lw=2,
                label=f"Fit (slope {slope:.2f})")
    ax.legend(fontsize=fontsize - 4)
    att.ax_plot(ax.get_figure(), ax, xlab, ylab, xs=fontsize, ys=fontsize)
    return points


if __name__ == "__main__":
    random.seed(SEED)

    # Cycle the lengths so every size from MIN_LENGTH to MAX_LENGTH is sampled
    # about equally rather than leaving the coverage to chance
    n_sizes = MAX_LENGTH - MIN_LENGTH + 1
    strings = [random_string(MIN_LENGTH + i % n_sizes, POOL) for i in range(N_STRINGS)]

    print(f"Calculating {len(strings)} random strings of length "
          f"{MIN_LENGTH}-{MAX_LENGTH} over '{POOL}'", flush=True)
    results = att.mp_calc(string_measures, strings)

    lengths, ai, lz78, zlib_bytes = (np.array(column) for column in zip(*results))

    print(f"String length vs assembly index:  Pearson r = "
          f"{pearsonr(lengths, ai)[0]:.3f}", flush=True)
    for name, measure in (("LZ78 phrases", lz78), ("zlib bytes  ", zlib_bytes)):
        print(f"{name} vs assembly index:  "
              f"Pearson r = {pearsonr(measure, ai)[0]:.3f}, "
              f"Spearman rho = {spearmanr(measure, ai)[0]:.3f}, "
              f"within-length rho = {within_group_spearman(measure, ai, lengths):.3f}",
              flush=True)

    # The figure asks the same question at two levels.
    #
    # Panels 1 and 2 plot each LZ measure against the assembly index as
    # measured. They look like near-perfect agreement, and the printed
    # correlations back that up -- but length alone already tracks the assembly
    # index just as closely, because every measure here grows with the string.
    # So most of that agreement is the two axes reporting size twice rather
    # than concurring about structure. The colour gradient running along the
    # diagonal band is what that looks like.
    #
    # Panel 3 takes the size out. Centring each measure within its own length
    # leaves only how a string differs from the other strings of exactly its
    # size, which asks the question worth asking: among strings of one length,
    # does LZ single out the ones the assembly index calls complex? It largely
    # does not. The band falls apart into a diffuse cloud with a shallow
    # slope, and the colour scatters instead of grading -- the signature of a
    # length dependence that has been removed rather than merely hidden.
    ai_centred = centre_within_groups(ai, lengths)
    lz78_centred = centre_within_groups(lz78, lengths)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel 1 is the fairest comparison in the figure: an LZ78 phrase and an
    # assembly step are the same kind of thing, a single construction move
    # whose only free material is what has already been built.
    compare_panel(axes[0], lz78, ai, lengths, "LZ78 phrase count")

    # Panel 2 repeats it for a compressor people actually use. zlib emits whole
    # bytes, so its axis is coarser and its grid blockier, and over strings this
    # short a fair part of what it reports is its own output quantisation rather
    # than anything about the string.
    points = compare_panel(axes[1], zlib_bytes, ai, lengths,
                           "zlib compressed size (bytes)")

    # Panel 3 is panel 1 with length taken out, so both axes now read as "more
    # or less than a typical string of this size". The fitted slope replaces the
    # per-x mean because centring turns an integer axis into a continuous one.
    compare_panel(axes[2], lz78_centred, ai_centred, lengths,
                  "LZ78 phrase count (length-centred)",
                  ylab="Assembly index (length-centred)", trend="fit")

    # One bar for the whole row: attaching it to every axis takes the width from
    # all three rather than squeezing the last one. The shared scale is only
    # correct because all three panels were handed the same `lengths` array --
    # plot a subset in one of them and it would need an explicit vmin and vmax.
    colorbar = fig.colorbar(points, ax=axes.ravel().tolist(), pad=0.015)
    colorbar.set_label("String length", fontsize=15)
    colorbar.ax.tick_params(labelsize=13)

    plt.savefig("lz_vs_string_assembly_index.png", dpi=600, bbox_inches="tight")
    plt.savefig("lz_vs_string_assembly_index.pdf", bbox_inches="tight")
    plt.show()
