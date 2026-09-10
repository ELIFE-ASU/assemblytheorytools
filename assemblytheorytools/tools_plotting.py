"""
Plotting helpers for graphs, pathways and spectra.

This module provides the matplotlib and pyvis figures used to present assembly
theory results: molecular and assembly graph drawings, interactive and metro-map
pathway layouts, assembly circles, scatter, contour, heatmap, hexbin, histogram
and KDE plots, crossing-minimised multipartite layouts, molecule grids, and
infrared and MS2 spectrum plots.

Importing this module sets a small number of global matplotlib ``rcParams`` to
give the figures a consistent appearance.
"""

import math
import random
from collections import defaultdict
from html import escape
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
from ase import Atoms
from ase.visualize.plot import plot_atoms
from IPython.display import HTML
from matplotlib import colormaps, colors
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import ArrowStyle, Circle, FancyArrowPatch
from PIL import Image
from pyvis.network import Network
from rdkit import Chem
from rdkit.Chem import Draw, rdFMCS
from scipy.stats import gaussian_kde

from .tools_atoms import mol_to_atoms
from .tools_data import enumerate_stereoisomers_shortest, pubchem_smi_to_name
from .tools_graph import nx_to_smi, relabel_digraph, set_graph_layer
from .tools_mol import smi_to_mol, standardize_mol

# set the plot axis
plt.rcParams["axes.linewidth"] = 2.0


def n_plot(xlab: str, ylab: str, xs: int = 14, ys: int = 14) -> None:
    """
    Configure plot aesthetics with axis labels, ticks, and tight layout.

    Sets up matplotlib plot formatting including minor ticks, tick parameters,
    axis labels with custom font sizes, and applies tight layout for optimal
    spacing.

    Parameters
    ----------
    xlab : str
        Label for the x-axis.
    ylab : str
        Label for the y-axis.
    xs : int, optional
        Font size for x-axis label, by default 14.
    ys : int, optional
        Font size for y-axis label, by default 14.

    Returns
    -------
    None
        Modifies the current matplotlib plot in-place.
    """
    ax_plot(plt.gcf(), plt.gca(), xlab, ylab, xs, ys)


def ax_plot(
    fig: plt.Figure, ax: plt.Axes, xlab: str, ylab: str, xs: int = 14, ys: int = 14
) -> None:
    """
    Configure axis aesthetics with labels, ticks, and tight layout.

    Sets up matplotlib axis formatting including minor ticks, tick parameters,
    axis labels with custom font sizes, and applies tight layout for optimal
    spacing. Similar to n_plot but operates on specific figure and axis objects.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib Figure object to apply tight layout.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object to configure.
    xlab : str
        Label for the x-axis.
    ylab : str
        Label for the y-axis.
    xs : int, optional
        Font size for x-axis label, by default 14.
    ys : int, optional
        Font size for y-axis label, by default 14.

    Returns
    -------
    None
        Modifies the figure and axis in-place.
    """
    ax.minorticks_on()
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=ys - 2,
        direction="in",
        width=2,
        top=True,
        right=True,
    )
    ax.tick_params(axis="both", which="major", length=6)
    ax.tick_params(axis="both", which="minor", length=4)
    ax.set_xlabel(xlab, fontsize=xs)
    ax.set_ylabel(ylab, fontsize=ys)
    fig.tight_layout()


def _auto_fig_size(
    n_objects: int,
    base_size: Tuple[float, float],
    reference_n: int = 10,
    min_size: Tuple[float, float] = (4.0, 3.0),
    max_size: Tuple[float, float] = (30.0, 30.0),
) -> Tuple[float, float]:
    """
    Scale a base figure size by a number of plotted objects.

    Width and height each scale with ``sqrt(n_objects / reference_n)``, so the
    figure *area* grows in proportion to the object count while the aspect
    ratio of `base_size` is preserved. Results are clamped to `min_size` and
    `max_size` so very small or very large object counts still produce a
    usable figure.

    Parameters
    ----------
    n_objects : int
        Number of discrete objects being drawn (e.g. graph nodes or molecules).
    base_size : tuple of float
        Figure size, (width, height) in inches, considered right-sized for
        `reference_n` objects.
    reference_n : int, optional
        Object count for which `base_size` applies unscaled, by default 10.
    min_size : tuple of float, optional
        Lower bound on (width, height), by default (4.0, 3.0).
    max_size : tuple of float, optional
        Upper bound on (width, height), by default (30.0, 30.0).

    Returns
    -------
    tuple of float
        Scaled (width, height) in inches.
    """
    n = max(int(n_objects), 1)
    scale = math.sqrt(n / float(reference_n))
    width = min(max(base_size[0] * scale, min_size[0]), max_size[0])
    height = min(max(base_size[1] * scale, min_size[1]), max_size[1])
    return width, height


def _graph_layout(graph: nx.Graph, layout: str, seed: int) -> dict:
    """Select a graph layout, falling back to Kamada–Kawai."""
    if layout == "spring":
        return nx.spring_layout(graph, seed=seed)
    layouts = {
        "circular": nx.circular_layout,
        "shell": nx.shell_layout,
        "spectral": nx.spectral_layout,
        "spiral": nx.spiral_layout,
        "arf": nx.arf_layout,
    }
    return layouts.get(layout, nx.kamada_kawai_layout)(graph)


def plot_graph(
    graph: nx.Graph,
    fig_size: tuple = (12, 7),
    layout: str = "kawai",
    f_labs: bool = False,
    edge_color: str = "grey",
    node_size: int = 300,
    edgecolors: str = "black",
    width: int = 2,
    linewidths: int = 2,
    seed: int = 42,
    auto_fig_size: bool = False,
) -> tuple[Figure, Axes]:
    """
    Visualize a NetworkX graph with customizable layout and styling options.

    Creates a matplotlib visualization of a NetworkX graph using various layout
    algorithms. Supports multiple layout types including force-directed, circular,
    spectral, and topological layouts.

    Parameters
    ----------
    graph : networkx.Graph
        NetworkX graph object to visualize.
    fig_size : tuple of float, optional
        Figure size in inches as (width, height), by default (12, 7).
    layout : str, optional
        Layout algorithm to use. Options: 'kawai', 'spring', 'circular',
        'shell', 'spectral', 'spiral', 'arf', 'topological', by default 'kawai'.
    f_labs : bool, optional
        If True, display node labels, by default False.
    edge_color : str, optional
        Color for edges, by default 'grey'.
    node_size : int, optional
        Size of nodes in points^2, by default 300.
    edgecolors : str, optional
        Color for node borders, by default 'black'.
    width : int, optional
        Line width for edges, by default 2.
    linewidths : int, optional
        Line width for node borders, by default 2.
    seed : int, optional
        Random seed for spring layout reproducibility, by default 42.
    auto_fig_size : bool, optional
        If True, ignore `fig_size` and compute a figure size scaled to the
        number of nodes in `graph` instead, by default False.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        Figure and axis objects containing the graph visualization.
    """
    graph = graph.copy()
    if layout == "topological":
        graph = set_graph_layer(graph)
        pos = nx.multipartite_layout(graph, subset_key="layer")
    else:
        pos = _graph_layout(graph, layout, seed)

    if auto_fig_size:
        fig_size = _auto_fig_size(graph.number_of_nodes(), base_size=fig_size)

    fig, ax = plt.subplots(figsize=fig_size)

    nx.draw_networkx(
        graph,
        ax=ax,
        pos=pos,
        with_labels=f_labs,
        edge_color=edge_color,
        node_size=node_size,
        edgecolors=edgecolors,
        width=width,
        linewidths=linewidths,
    )
    fig.tight_layout()
    ax.axis("off")
    return fig, ax


def plot_mol_graph(
    graph: nx.Graph,
    fig_size: tuple = (12, 7),
    layout: str = "kawai",
    f_labs: bool = False,
    node_size: int = 300,
    width: int = 2,
    linewidths: int = 2,
    seed: int = 42,
    auto_fig_size: bool = False,
) -> tuple[Figure, Axes]:
    """
    Visualize a molecular graph with atom-specific coloring.

    Creates a matplotlib visualization of a molecular NetworkX graph where
    nodes are colored according to their atomic element type. Uses standard
    CPK coloring convention for chemical elements.

    Parameters
    ----------
    graph : networkx.Graph
        NetworkX graph representing a molecular structure with 'element'
        node attributes.
    fig_size : tuple of float, optional
        Figure size in inches as (width, height), by default (12, 7).
    layout : str, optional
        Layout algorithm to use. Options: 'kawai', 'spring', 'circular',
        'shell', 'spectral', 'spiral', 'arf', by default 'kawai'.
    f_labs : bool, optional
        If True, display node labels, by default False.
    node_size : int, optional
        Size of nodes in points^2, by default 300.
    width : int, optional
        Line width for edges, by default 2.
    linewidths : int, optional
        Line width for node borders, by default 2.
    seed : int, optional
        Random seed for spring layout reproducibility, by default 42.
    auto_fig_size : bool, optional
        If True, ignore `fig_size` and compute a figure size scaled to the
        number of nodes in `graph` instead, by default False.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        Figure and axis objects containing the molecular graph visualization.
    """
    graph = graph.copy()
    pos = _graph_layout(graph, layout, seed)

    cols_conv = {
        "H": "white",  # Hydrogen
        "C": "darkgray",  # Carbon
        "O": "red",  # Oxygen
        "N": "blue",  # Nitrogen
        "S": "yellow",  # Sulfur
        "P": "orange",  # Phosphorus
        "Cl": "green",  # Chlorine
        "F": "lightgreen",  # Fluorine
        "Br": "brown",  # Bromine
        "I": "purple",  # Iodine
        "Fe": "darkorange",  # Iron
        "Ca": "gold",  # Calcium
        "Na": "lightblue",  # Sodium
        "K": "violet",  # Potassium
        "Mg": "darkgreen",  # Magnesium
        "Cu": "peru",  # Copper
        "Zn": "gray",  # Zinc
        "Au": "gold",  # Gold
        "Ag": "silver",  # Silver
        "Pt": "lightgray",  # Platinum
    }

    color_dict_edge = {1: "black", 2: "green", 3: "red", 4: "orange"}
    graph_colors = [
        cols_conv.get(data["color"], "black") for _, data in graph.nodes(data=True)
    ]
    edge_colors = [
        color_dict_edge.get(data["color"]) for *_, data in graph.edges(data=True)
    ]

    if auto_fig_size:
        fig_size = _auto_fig_size(graph.number_of_nodes(), base_size=fig_size)

    fig, ax = plt.subplots(figsize=fig_size)

    nx.draw_networkx(
        graph,
        ax=ax,
        pos=pos,
        with_labels=f_labs,
        node_size=node_size,
        edge_color=edge_colors,
        node_color=graph_colors,
        edgecolors="black",
        width=width,
        linewidths=linewidths,
    )
    fig.tight_layout()
    ax.axis("off")
    return fig, ax


def plot_interactive_graph(
    graph: nx.Graph, show: bool = False, filename: str = "interactive_graph.html"
) -> Network:
    """
    Create an interactive HTML visualization of a NetworkX graph using PyVis.

    Generates an interactive graph visualization with node coloring based on
    degree (number of connections). Higher degree nodes appear in darker shades
    of blue. The visualization can be displayed in a Jupyter notebook or saved
    as an HTML file.

    Parameters
    ----------
    graph : networkx.Graph
        NetworkX graph object to visualize interactively.
    show : bool, optional
        If True, displays the graph in a Jupyter notebook using an iframe.
        If False, saves to HTML file, by default False.
    filename : str, optional
        Name of the HTML file to save when show=False,
        by default "interactive_graph.html".

    Returns
    -------
    pyvis.network.Network
        PyVis Network object containing the interactive visualization.
    """
    graph = graph.copy()
    max_nbr = max(map(len, graph.adj.values()))
    blues = colormaps.get_cmap("Blues")
    for n, d in graph.nodes(data=True):
        d["color"] = colors.to_hex(blues(len(graph.adj[n]) / max_nbr))

    # Convert to PyVis network
    width, height = (900, 900)
    net = Network(width=f"{width}px", height=f"{height}px", notebook=True, heading="")
    net.from_nx(graph)
    if show:
        html_doc = net.generate_html(notebook=True)
        iframe = (
            f'<iframe width="{width + 25}px" height="{height + 25}px" frameborder="0" '
            f'srcdoc="{escape(html_doc)}"></iframe>'
        )
        HTML(iframe)
    else:
        net.show(filename)
    return net


def plot_digraph_metro(
    digraph: nx.DiGraph,
    filename: str = "metro",
    steps: bool = False,
    vo_str: bool = True,
    vo_names: str | None = None,
) -> None:
    """
    Render a directed acyclic graph (DAG) in a metro-style layout and save as SVG and PNG.

    This function visualizes a directed acyclic graph (DAG) using the `dagviz` library's
    metro-style layout. The graph can be optionally relabeled with topological steps or
    virtual object (VO) labels. The output is saved as both an SVG and a PNG file.

    Parameters
    ----------
    digraph : networkx.DiGraph
        The directed acyclic graph to be visualized.
    filename : str, optional
        The base name for the output files (without extension). Defaults to 'metro'.
    steps : bool, optional
        If True, relabel the graph nodes with their topological step. Defaults to False.
    vo_str : bool, optional
        If True, convert the 'vo' attribute of nodes to string labels. Defaults to True.
    vo_names : str, optional
        If True, attempt to retrieve human-readable names for virtual objects using
        `pubchem_smi_to_name`. Defaults to None.

    Raises
    ------
    ImportError
        If the required `dagviz` or `cairosvg` libraries are not installed.
    ValueError
        If a node's 'vo' attribute is of an unsupported type.

    Notes
    -----
    - The `dagviz` library is used for rendering the graph in a metro-style layout.
    - The `cairosvg` library is used to convert the SVG output to PNG format.
    - Node labels are determined based on the 'vo' attribute, which can be a string,
      a NetworkX graph, or an RDKit molecule object.

    Returns
    -------
    None
        The function saves the graph visualization to files and does not return any value.
    """
    digraph = digraph.copy()  # Avoid modifying the original graph
    try:
        import cairosvg
        import dagviz
    except ImportError as e:
        raise ImportError(
            "The 'dagviz' and 'cairosvg' packages are required for this function.\n"
            "Please install them via pip:\n"
            "pip install git+https://github.com/ELIFE-ASU/dagviz.git \n"
            "pip install cairosvg \n"
        ) from e

    if steps:
        # Relabel the graph nodes with their topological step if requested
        digraph = relabel_digraph(digraph)

    if vo_str:
        for _, data in digraph.nodes(data=True):
            vo = data["vo"]
            if type(vo) is str:
                lab = vo
            elif type(vo) is nx.Graph:
                lab = nx_to_smi(vo, add_hydrogens=False, sanitize=True)
            elif type(vo) is Chem.Mol:
                lab = Chem.MolToSmiles(vo)
            else:
                raise ValueError(f"Unsupported virtual object type: {type(vo)}")

            if vo_names:
                lab = enumerate_stereoisomers_shortest(
                    Chem.MolFromSmiles(lab), prefer=vo_names
                )
                lab = pubchem_smi_to_name(lab, prefer=vo_names)
                if lab is None:
                    lab = ""
            data["label"] = lab

    backend = dagviz.style.metro.svg_renderer(
        dagviz.style.metro.StyleConfig(node_stroke="black")
    )
    svg = dagviz.render_svg(digraph, style=backend)

    with open(f"{filename}.svg", "w") as file:
        file.write(svg)

    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=f"{filename}.png")


def _draw_edge_arrowhead(
    ax: Axes,
    edge_patch: FancyArrowPatch,
    position: float,
    color: str,
    plt_arrow_style: Union[str, ArrowStyle],
    arrow_size: int,
    width: float = 2.5,
) -> None:
    """
    Draw a single arrowhead part way along an edge that has already been drawn.

    The edge patch is asked for its path in data coordinates, so the head follows
    the curvature of the edge and its node margins rather than the straight line
    between the two nodes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis holding the edge.
    edge_patch : matplotlib.patches.FancyArrowPatch
        Edge as returned by `networkx.draw_networkx_edges`, drawn without a head.
    position : float
        Fraction along the edge at which to place the head, from 0 at the source
        to 1 at the target.
    color : str
        Colour of the arrowhead.
    plt_arrow_style : str or matplotlib.patches.ArrowStyle
        Style of the arrowhead.
    arrow_size : int
        Size of the arrowhead (matplotlib mutation scale).
    width : float, optional
        Line width of the arrowhead, by default 2.5.

    Returns
    -------
    None
        The arrowhead is added to the axis in place.
    """
    # Poly-line approximation of the drawn edge, in data coordinates
    verts = np.asarray(
        [v for v, _ in edge_patch.get_path().iter_segments(curves=False)]
    )
    if len(verts) < 2:
        return

    steps = np.diff(verts, axis=0)
    lengths = np.hypot(steps[:, 0], steps[:, 1])
    # Drop repeated vertices, they carry no direction
    keep = lengths > 0.0
    if not keep.any():
        return
    starts, steps, lengths = verts[:-1][keep], steps[keep], lengths[keep]

    # Walk along the arc length, so the head sits part way along the curve
    # rather than part way between the two nodes
    arc = np.concatenate(([0.0], np.cumsum(lengths)))
    target = float(np.clip(position, 0.0, 1.0)) * arc[-1]
    i = int(np.clip(np.searchsorted(arc, target) - 1, 0, len(lengths) - 1))
    point = starts[i] + steps[i] * (target - arc[i]) / lengths[i]

    # Stub pointing along the edge, short enough to hide under the head itself
    stub = 1e-3 * arc[-1] * steps[i] / lengths[i]
    ax.add_patch(
        FancyArrowPatch(
            point - stub,
            point + stub,
            arrowstyle=plt_arrow_style,
            mutation_scale=arrow_size,
            color=color,
            linewidth=width,
            shrinkA=0,
            shrinkB=0,
            zorder=edge_patch.get_zorder(),
        )
    )


def _figure_image(fig: Figure, dpi: int) -> np.ndarray:
    """Render a temporary figure to an in-memory PNG and close it."""
    try:
        with BytesIO() as buffer:
            fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
            buffer.seek(0)
            return plt.imread(buffer, format="png")
    finally:
        plt.close(fig)


def _pathway_icon(vo: Any, plot_type: str) -> OffsetImage:
    """Render a virtual object at the scale used by pathway nodes."""
    if plot_type == "mol":
        smi = vo.replace("[", "").replace("]", "")
        mol = smi_to_mol(smi, add_hydrogens=False)
        img = Draw.MolToImage(mol, size=(200, 200), kekulize=False, fitImage=True)
        return OffsetImage(img, zoom=0.4)
    if plot_type == "graph":
        fig, _ = plot_mol_graph(
            vo, f_labs=False, fig_size=(4, 4), node_size=1000, width=10
        )
        return OffsetImage(_figure_image(fig, dpi=400), zoom=0.05)

    mol = smi_to_mol(vo, add_hydrogens=False)
    atoms = mol_to_atoms(mol, sanitize=False, add_hydrogens=False)
    fig, ax = plt.subplots()
    plot_atoms(atoms, ax, show_unit_cell=0, scale=2.0)
    fig.tight_layout()
    ax.axis("off")
    return OffsetImage(_figure_image(fig, dpi=500), zoom=0.02)


def plot_pathway(
    graph: nx.DiGraph,
    fig_size: tuple = (12, 7),
    show_icons: bool = True,
    node_color: str = "#264f70",
    plot_type: str = "mol",
    arrow_style: str = "1",
    layout_style: str = "crossmin_long",
    frame_on: bool = True,
    font_size: int = 11,
    arrow_color: str = "#264f70",
    plt_arrow_style: Union[str, ArrowStyle] = "->",
    arrow_pos: float = 1.0,
    arrow_size: int = 20,
    auto_fig_size: bool = False,
) -> tuple[Figure, Axes]:
    """
    Visualize a directed acyclic graph as a pathway with customizable layout.

    Creates a layered pathway visualization with topological ordering. Supports
    molecular structure icons, optimized crossing minimization layouts, and
    customizable arrow styles.

    Parameters
    ----------
    graph : networkx.DiGraph
        Directed acyclic graph representing a pathway or assembly process.
    fig_size : tuple of float, optional
        Figure size in inches as (width, height), by default (12, 7).
    show_icons : bool, optional
        If True, displays molecular structure icons on nodes, by default True.
    node_color : str, optional
        Color for nodes in hex format, by default '#264f70'.
    plot_type : str, optional
        Type of plot visualization ('mol' for molecules), by default 'mol'.
    arrow_style : str, optional
        Arrow rendering style: '1' for white edges, '2' for grey edges,
        by default '1'.
    layout_style : str, optional
        Layout algorithm: 'crossmin', 'crossmin_long', 'sa', or default
        multipartite, by default 'crossmin_long'.
    frame_on : bool, optional
        If True, displays axis frame, by default True.
    font_size : int, optional
        Font size for string assembly paths, by default 11.
    arrow_color : str, optional
        Color for arrows in hex format, by default '#264f70'.
    plt_arrow_style : str or matplotlib.patches.ArrowStyle, optional
        Style of the arrowheads in the plot, by default '->'.
    arrow_pos : float, optional
        Fraction along each edge at which the arrowhead is drawn, from 0 at the
        source to 1 at the target, by default 1.0 (head at the target node).
        Only used with arrow_style '1'; see `plot_pathway_mid_arrow` for heads
        half way along the edges.
    arrow_size : int, optional
        Size of the arrowhead (matplotlib mutation scale) when it is placed part
        way along an edge, by default 20. Unused when arrow_pos is 1.
    auto_fig_size : bool, optional
        If True, ignore `fig_size` and compute a figure size scaled to the
        number of nodes in `graph` instead, by default False.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        Figure and axis objects containing the pathway visualization.

    Raises
    ------
    ValueError
        If arrow_style is not '1' or '2'.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import assemblytheorytools as att
    >>> graph = att.smi_to_nx("NCC(=O)O.CC(N)C(=O)O")
    >>> _, _, pathway = att.calculate_assembly_index(
    ...     graph, strip_hydrogen=True)
    >>> fig, ax = att.plot_pathway(pathway, plot_type="graph")
    >>> fig.savefig("pathway.svg")  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP

    Use ``plot_type="mol"`` to draw molecular structures instead of graph
    diagrams, and ``auto_fig_size=True`` to size the canvas to the pathway
    rather than fixing it in advance.
    """
    graph = graph.copy()
    if plot_type == "mol":
        for _, data in graph.nodes(data=True):
            node_graph = data["vo"]
            if isinstance(node_graph, nx.Graph):
                try:
                    data["vo"] = nx_to_smi(
                        node_graph, add_hydrogens=False, sanitize=False
                    )
                except Exception:
                    plot_type = "graph"
    elif plot_type == "string" and show_icons:
        node_color = "white"

    if auto_fig_size:
        fig_size = _auto_fig_size(graph.number_of_nodes(), base_size=fig_size)

    fig, ax = plt.subplots(figsize=fig_size)
    graph = set_graph_layer(graph)

    layouts = {
        "crossmin": multipartite_layout_crossmin,
        "crossmin_long": multipartite_layout_crossmin_long,
        "sa": multipartite_layout_sa,
    }
    layout = layouts.get(layout_style, nx.multipartite_layout)
    pos = layout(graph, subset_key="layer")

    if arrow_style == "1":
        edge_color1 = "white"
    elif arrow_style == "2":
        edge_color1 = "grey"
    else:
        raise ValueError("Invalid arrow style. Use '1' or '2'.")

    nx.draw_networkx(
        graph,
        pos=pos,
        ax=ax,
        with_labels=False,
        node_size=1000,
        node_color=node_color,
        connectionstyle="arc3,rad=0.1",
        edge_color=edge_color1,
        arrows=True,
        arrowstyle="->",
        width=2.0,
    )

    edge_patches = []
    if arrow_style == "1":
        arrow_margin = 70 if show_icons else 20

        for edge in graph.edges():
            src, dst = edge
            # Bend toward the destination; horizontally aligned edges stay straight.
            if pos[src][1] > pos[dst][1]:
                rad = -0.15
            elif pos[src][1] < pos[dst][1]:
                rad = 0.15
            else:
                rad = 0.0

            # A head part way along the edge is added afterwards, so the edge
            # itself is drawn without one
            edge_patches += nx.draw_networkx_edges(
                graph,
                pos=pos,
                edgelist=[edge],
                ax=ax,
                arrows=True,
                arrowstyle=plt_arrow_style if arrow_pos >= 1.0 else "-",
                width=2.5,
                edge_color=arrow_color,
                connectionstyle=f"arc3,rad={rad}",
                min_target_margin=arrow_margin,
            )

    if show_icons:
        for node, data in graph.nodes(data=True):
            if plot_type in ("mol", "graph", "atoms"):
                icon = _pathway_icon(data["vo"], plot_type)
                ax.add_artist(AnnotationBbox(icon, pos[node], frameon=frame_on))
            elif plot_type == "string":
                ax.text(
                    *pos[node],
                    data["vo"],
                    fontsize=font_size,
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="white",
                        edgecolor="white",
                        linewidth=1,
                    ),
                )

    fig.tight_layout()
    ax.axis("off")
    # scatter the positions to fix the view
    ax.scatter(
        [pos[node][0] for node in graph.nodes()],
        [pos[node][1] for node in graph.nodes()],
        s=0,
        color="red",
    )

    if edge_patches and arrow_pos < 1.0:
        # The edge paths are built in display space, so the view has to be
        # settled before they are read back, and frozen so the heads stay on them
        fig.canvas.draw()
        ax.set_xlim(*ax.get_xlim())
        ax.set_ylim(*ax.get_ylim())
        for edge_patch in edge_patches:
            _draw_edge_arrowhead(
                ax, edge_patch, arrow_pos, arrow_color, plt_arrow_style, arrow_size
            )

    return fig, ax


def plot_pathway_mid_arrow(
    graph: nx.DiGraph,
    fig_size: tuple = (12, 7),
    show_icons: bool = True,
    node_color: str = "#264f70",
    plot_type: str = "mol",
    layout_style: str = "crossmin_long",
    frame_on: bool = True,
    font_size: int = 11,
    arrow_color: str = "#264f70",
    plt_arrow_style: Union[str, ArrowStyle] = "->",
    arrow_pos: float = 0.5,
    arrow_size: int = 20,
    auto_fig_size: bool = False,
) -> tuple[Figure, Axes]:
    """
    Visualize a directed acyclic graph as a pathway with mid-edge arrowheads.

    Same as `plot_pathway` with the white edge style, except that each edge is
    drawn as a plain line and its arrowhead is placed half way along it instead
    of at the target node. Useful when the icons are large enough that heads at
    the target node crowd them.

    Parameters
    ----------
    graph : networkx.DiGraph
        Directed acyclic graph representing a pathway or assembly process.
    fig_size : tuple of float, optional
        Figure size in inches as (width, height), by default (12, 7).
    show_icons : bool, optional
        If True, displays molecular structure icons on nodes, by default True.
    node_color : str, optional
        Color for nodes in hex format, by default '#264f70'.
    plot_type : str, optional
        Type of plot visualization ('mol' for molecules), by default 'mol'.
    layout_style : str, optional
        Layout algorithm: 'crossmin', 'crossmin_long', 'sa', or default
        multipartite, by default 'crossmin_long'.
    frame_on : bool, optional
        If True, displays axis frame, by default True.
    font_size : int, optional
        Font size for string assembly paths, by default 11.
    arrow_color : str, optional
        Color for arrows in hex format, by default '#264f70'.
    plt_arrow_style : str or matplotlib.patches.ArrowStyle, optional
        Style of the arrowheads in the plot, by default '->'.
    arrow_pos : float, optional
        Fraction along each edge at which the arrowhead is drawn, from 0 at the
        source to 1 at the target, by default 0.5 (half way).
    arrow_size : int, optional
        Size of the arrowheads (matplotlib mutation scale), by default 20.
    auto_fig_size : bool, optional
        If True, ignore `fig_size` and compute a figure size scaled to the
        number of nodes in `graph` instead, by default False.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        Figure and axis objects containing the pathway visualization.
    """
    return plot_pathway(
        graph,
        fig_size=fig_size,
        show_icons=show_icons,
        node_color=node_color,
        plot_type=plot_type,
        arrow_style="1",
        layout_style=layout_style,
        frame_on=frame_on,
        font_size=font_size,
        arrow_color=arrow_color,
        plt_arrow_style=plt_arrow_style,
        arrow_pos=arrow_pos,
        arrow_size=arrow_size,
        auto_fig_size=auto_fig_size,
    )


def _average_angles(angles: np.ndarray) -> float:
    """
    Average a set of angles, respecting their circular nature.

    Each angle is converted to its corresponding unit vector (using sine and
    cosine), the components are summed, and the angle of the resultant
    vector is returned.

    Parameters
    ----------
    angles : np.ndarray
        Array of angles (in radians) for which to compute the average.

    Returns
    -------
    float
        The average angle (in radians), in the range (-pi, pi].

    """
    return np.arctan2(np.sin(angles).sum(), np.cos(angles).sum())


def _ring_radius(index, spacing_mode: str, factor: float):
    """Map ring indices to radii for both nodes and their guide circles."""
    if spacing_mode == "hyperbolic":
        return index + float(factor) * np.sinh(index)
    return index


def _save_figure(fig: Figure, filename, dpi: int, save_kwargs) -> None:
    """Save on request, allowing callers to replace the default padding."""
    if filename is not None:
        if save_kwargs is None:
            save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        fig.savefig(filename, dpi=dpi, **save_kwargs)


def _plot_directed_network(
    nodes: List[str],
    adjacency_matrix: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    max_ai: int,
    labels: bool,  # can be bool or List[str]
    node_size: float,
    arrow_size: float,
    node_color: str,
    node_edge_color: str,
    node_linewidth: float,
    edge_color: str,
    arrow_alpha: float,
    fig_size: float,
    filename: Optional[str] = None,
    dpi: int = 300,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    save_kwargs: Optional[Dict[str, Any]] = None,
    spacing_mode: str = "linear",
    spacing_hyperbolic_factor: float = 0.4,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw a directed network on top of concentric circles.

    Creates a visualisation of a directed network from a list of nodes,
    their positions, and an adjacency matrix defining the edges. Curved
    arrows are drawn first, then the concentric circles, then the nodes and
    their labels. The figure is written to disk only when ``filename`` is
    given.

    Parameters
    ----------
    nodes : list of str
        List of node names in the network.
    adjacency_matrix : np.ndarray
        Square adjacency matrix (shape ``[n_nodes, n_nodes]``) representing
        directed edges. If ``adjacency_matrix[i, j] != 0`` there is a
        directed edge from node ``i`` to node ``j``.
    x : np.ndarray
        1D array of x-coordinates for each node (same order as ``nodes``).
    y : np.ndarray
        1D array of y-coordinates for each node (same order as ``nodes``).
    max_ai : int
        Maximum assembly index, which sets the number of concentric circles
        drawn.
    labels : bool or sequence of str
        If a boolean, whether to label each node with its own name. If a
        sequence, the per-node label strings; empty or ``None`` entries
        render no text but the node itself is still drawn.
    node_size : float
        Marker size of the nodes, in points squared.
    arrow_size : float
        Size of the arrowheads for directed edges.
    node_color : str
        Colour of the nodes.
    node_edge_color : str
        Colour of the node borders.
    node_linewidth : float
        Line width of the node borders.
    edge_color : str
        Colour of the edges.
    arrow_alpha : float
        Alpha of the edge arrows, from 0.0 to 1.0.
    fig_size : float
        Size of the figure in inches, used for both width and height.
    filename : str or None, optional
        Path to save the rendered figure to. Default is None, which saves
        nothing.
    dpi : int, optional
        Resolution in dots per inch used when saving. Default is 300.
    fig : matplotlib.figure.Figure or None, optional
        Figure to draw onto. Default is None, which creates a new figure.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw onto. Default is None, which creates new axes.
    save_kwargs : dict or None, optional
        Extra keyword arguments forwarded to ``Figure.savefig`` when
        ``filename`` is given. Default is None, which uses
        ``bbox_inches="tight"`` and ``pad_inches=0.02``.
    spacing_mode : {"linear", "hyperbolic"}, optional
        Radial spacing of the concentric circles. ``"linear"`` places circle
        ``i`` at radius ``i``; ``"hyperbolic"`` places it at ``i +
        spacing_hyperbolic_factor * sinh(i)``. Default is ``"linear"``.
    spacing_hyperbolic_factor : float, optional
        Multiplier for the sinh term under hyperbolic spacing. Default is
        0.4.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object containing the plot.

    Raises
    ------
    ValueError
        If ``nodes``, ``adjacency_matrix``, ``x`` and ``y`` do not all have
        the same length.
    """
    if (
        len(nodes) != len(adjacency_matrix)
        or len(adjacency_matrix) != len(x)
        or len(x) != len(y)
    ):
        raise ValueError("Lengths of nodes, adjacency_matrix, x, and y must be equal.")

    graph = nx.DiGraph()
    positions = {node: (float(xi), float(yi)) for node, xi, yi in zip(nodes, x, y)}
    graph.add_nodes_from(nodes)
    for i, src in enumerate(nodes):
        for j, dst in enumerate(nodes):
            if adjacency_matrix[i, j] != 0:
                graph.add_edge(src, dst, weight=float(adjacency_matrix[i, j]))

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # 1) draw curved directed edges first (below circles)
    for src, dst in graph.edges():
        x_src = positions[src][0]
        x_dst = positions[dst][0]
        if x_src > x_dst:
            rad = -0.25
        elif x_src < x_dst:
            rad = 0.25
        else:
            rad = 0.0
        coll = nx.draw_networkx_edges(
            graph,
            pos=positions,
            edgelist=[(src, dst)],
            ax=ax,
            arrows=True,
            arrowstyle="->",
            width=1.8,
            edge_color=edge_color,
            alpha=arrow_alpha,
            connectionstyle=f"arc3,rad={rad}",
            arrowsize=arrow_size,
            min_target_margin=10,
            # don't pass zorder into nx.draw (it may validate kwargs); set after if possible
        )
        if coll is not None:
            artists = coll if isinstance(coll, (list, tuple)) else [coll]
            for artist in artists:
                artist.set_zorder(1)

    # 2) draw concentric circles above edges
    for idx in range(1, max_ai + 2):
        r = _ring_radius(float(idx), spacing_mode, spacing_hyperbolic_factor)
        circle = Circle((0, 0), r, color="black", alpha=1, fill=False, lw=1.5)
        circle.set_zorder(3)
        ax.add_artist(circle)

    # 3) draw nodes on top of circles using ax.scatter so zorder is applied safely
    node_xs = [positions[nn][0] for nn in nodes]
    node_ys = [positions[nn][1] for nn in nodes]
    # node_size in matplotlib scatter is in points^2; keep API-consistent
    ax.scatter(
        node_xs,
        node_ys,
        s=node_size,
        c=node_color,
        edgecolors=node_edge_color,
        linewidths=node_linewidth,
        zorder=4,
    )
    font_size = max(8, int(node_size / 200))
    if isinstance(labels, (list, tuple, np.ndarray)):
        node_labels = ((node, label) for node, label in zip(nodes, labels) if label)
    else:
        node_labels = ((node, str(node)) for node in nodes) if labels else ()
    for node, label in node_labels:
        ax.text(
            *positions[node],
            str(label),
            ha="center",
            va="center",
            fontsize=font_size,
            zorder=5,
        )

    # set limits based on mapped outermost radius
    max_radius = _ring_radius(
        float(max_ai + 1), spacing_mode, spacing_hyperbolic_factor
    )
    margin = max(1.5, 0.1 * max_radius)
    ax.set_xlim(-max_radius - margin, max_radius + margin)
    ax.set_ylim(-max_radius - margin, max_radius + margin)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    _save_figure(fig, filename, dpi, save_kwargs)

    return fig, ax


def _spread_overlapping_angles(angles: np.ndarray) -> None:
    """Separate coincident node angles while retaining deterministic order."""
    finite = np.flatnonzero(~np.isnan(angles))
    processed = set()
    for index in finite:
        if index in processed:
            continue
        distances = np.abs(
            (angles[finite] - angles[index] + np.pi) % (2 * np.pi) - np.pi
        )
        coincident = finite[distances <= 1e-12]
        if coincident.size > 1:
            center = _average_angles(angles[coincident])
            spread = min(0.08, 0.03 * coincident.size)
            offsets = np.linspace(-spread, spread, coincident.size)
            angles[coincident] = (center + offsets) % (2 * np.pi)
        processed.update(coincident)


def _assembly_circle_angles(
    n_nodes: int, adjacency: np.ndarray, assembly_indices: Sequence[int]
) -> np.ndarray:
    """Propagate building-block angles through parents, then neighbors."""
    angles = np.full(n_nodes, np.nan)
    min_ai = int(min(assembly_indices))
    building_blocks = [i for i, ai in enumerate(assembly_indices) if ai == min_ai]
    for k, index in enumerate(building_blocks):
        angles[index] = 2 * np.pi * (k / len(building_blocks))

    while np.isnan(angles).any():
        changed = False
        for include_children in (False, True):
            for index in range(n_nodes):
                if not np.isnan(angles[index]):
                    continue
                neighbors = adjacency[:, index] != 0
                if include_children:
                    neighbors |= adjacency[index, :] != 0
                neighbor_angles = angles[np.flatnonzero(neighbors)]
                known = neighbor_angles[~np.isnan(neighbor_angles)]
                if known.size:
                    angles[index] = _average_angles(known)
                    changed = True
            if changed:
                break

        if not changed:
            remaining = np.flatnonzero(np.isnan(angles))
            for k, index in enumerate(remaining):
                angles[index] = 2 * np.pi * (k / len(remaining))
            break
        _spread_overlapping_angles(angles)

    return angles


def plot_assembly_circle(
    nodes: Sequence[Any],
    adj_matrix: np.ndarray,
    assembly_indices: Sequence[int],
    labels: Optional[Union[bool, Sequence[str]]] = None,
    node_size: float = 1000,
    arrow_size: float = 80,
    node_color: Union[str, Sequence[str]] = "#264f70",
    node_edge_color: str = "black",
    node_linewidth: float = 2.5,
    edge_color: Union[str, Sequence[str]] = "Grey",
    arrow_alpha: float = 1.0,
    fig_size: Union[float, Tuple[float, float]] = 10,
    filename: Optional[str] = None,
    dpi: int = 300,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    cmap: Optional[Any] = None,
    norm: Optional[Any] = None,
    colorbar_label: Optional[str] = None,
    save_kwargs: Optional[Dict[str, Any]] = None,
    spacing_mode: str = "linear",
    spacing_hyperbolic_factor: float = 0.4,
    auto_fig_size: bool = False,
) -> Tuple[Figure, Axes]:
    """
    Plot an assembly pathway as a set of concentric rings.

    Nodes are placed on concentric rings whose radius is proportional to
    their assembly index. Edges between nodes are rendered as curved
    directed arrows. Optional per-node labels, icons (molecule / atoms /
    graph), colormap/norm support and file saving are provided. Both the
    adjacency matrix and one assembly index per node are required.

    Parameters
    ----------
    nodes : sequence
        Sequence of node identifiers (hashable). Order is used when
        `adj_matrix` or `assembly_indices` correspond by index.
    adj_matrix : array-like
        Square adjacency matrix (shape ``[n_nodes, n_nodes]``) indicating
        directed edges. Non-zero entries denote an edge from row index to
        column index.
    assembly_indices : array-like of int
        Assembly index for each node (lower values are closer to the
        center).
    labels : bool or sequence, optional
        If a boolean, ``True`` displays node identifiers as labels,
        ``False`` hides labels. If a sequence, per-node label strings to
        render; empty or ``None`` entries suppress text for that node.
    node_size : float, optional
        Marker size for nodes. Default is ``1000``.
    arrow_size : float, optional
        Arrowhead size for directed edges. Default is ``80``.
    node_color : str or sequence, optional
        Color for nodes; may be a single color string or a sequence of
        colors (one per node). Default is ``'#264f70'``.
    node_edge_color : str, optional
        Color for node borders. Default is ``'black'``.
    node_linewidth : float, optional
        Line width for node borders. Default is ``2.5``.
    edge_color : str or sequence, optional
        Color for edges. Default is ``'Grey'``.
    arrow_alpha : float, optional
        Alpha/transparency for arrows (0.0 - 1.0). Default is ``1.0``.
    fig_size : float or tuple, optional
        Size of the figure in inches. If a single float is provided it is
        used for both width and height. Default is ``10``.
    filename : str or None, optional
        If provided, save the rendered figure to this path; the extension
        selects the output format. Default is ``None`` (no file saved).
    dpi : int, optional
        Resolution in dots-per-inch when saving. Default is ``300``.
    fig : matplotlib.figure.Figure or None, optional
        Figure to draw onto. If ``None`` a new figure is created.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw onto. If ``None`` a new axes is created.
    cmap : matplotlib.colors.Colormap or str or None, optional
        Colormap to map per-node values if node colors are provided as
        numeric values. If provided together with ``norm``, a colorbar may
        be added.
    norm : matplotlib.colors.Normalize or None, optional
        Normalization instance used with ``cmap`` for color scaling.
    colorbar_label : str or None, optional
        Label for the colorbar when a colormap and norm are supplied.
    save_kwargs : dict or None, optional
        Extra keyword arguments forwarded to ``Figure.savefig`` when
        ``filename`` is provided (e.g. ``bbox_inches``).
    spacing_mode : {"linear", "hyperbolic"}, optional
        Radial spacing of the rings. ``"linear"`` uses a radius of ``ai +
        1``; ``"hyperbolic"`` uses ``(ai + 1) + spacing_hyperbolic_factor *
        sinh(ai + 1)``. Default is ``"linear"``.
    spacing_hyperbolic_factor : float, optional
        Multiplier for the sinh term under hyperbolic spacing. Default is
        0.4.
    auto_fig_size : bool, optional
        If True, ignore `fig_size` and compute a size scaled to the number
        of `nodes` instead, by default False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib Figure that contains the rendered assembly circle plot.
    ax : matplotlib.axes.Axes
        Matplotlib Axes used for the drawing.

    Raises
    ------
    ValueError
        If provided arrays (e.g. ``nodes``, ``adj_matrix``,
        ``assembly_indices``) have inconsistent lengths or if ``adj_matrix``
        is not square when supplied.
    TypeError
        If inputs cannot be interpreted as the expected types (e.g. numeric
        assembly indices or a 2D adjacency array).

    Notes
    -----
    - Node placement: nodes with the same assembly index share the same radius and
      are arranged by angle. Building-block nodes (minimum assembly index)
      are evenly spaced around the innermost circle; other nodes inherit an
      average angle from their parents via adjacency propagation.
    - Edges are rendered as curved arcs; curvature is chosen heuristically from
      relative node positions to improve readability.
    - When ``cmap`` and ``norm`` are provided the function adds a colorbar using
      the provided label and places it on the figure prior to optional
      saving.
    - The function may call internal helpers (e.g. a directed-network plotting
      routine) to perform low-level drawing; modify the returned ``ax``
      after calling if custom axis limits or annotations are required.
    """

    n_nodes = len(nodes)
    max_ai = int(max(assembly_indices))
    adj = np.array(adj_matrix)
    angles = _assembly_circle_angles(n_nodes, adj, assembly_indices)
    radii = _ring_radius(
        np.array(assembly_indices, dtype=float) + 1.0,
        spacing_mode,
        spacing_hyperbolic_factor,
    )

    x_positions = radii * np.cos(angles)
    y_positions = radii * np.sin(angles)

    if auto_fig_size:
        base = (
            (fig_size, fig_size)
            if isinstance(fig_size, (int, float))
            else tuple(fig_size)
        )
        fig_size = _auto_fig_size(n_nodes, base_size=base)[0]

    # If fig/ax not provided, create them (no fallbacks beyond this)
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # call the underlying plotting routine drawing onto our fig/ax
    fig, ax = _plot_directed_network(
        nodes=nodes,
        adjacency_matrix=adj,
        x=x_positions,
        y=y_positions,
        max_ai=max_ai,
        labels=labels,
        node_size=node_size,
        arrow_size=arrow_size,
        node_color=node_color,
        node_edge_color=node_edge_color,
        node_linewidth=node_linewidth,
        edge_color=edge_color,
        arrow_alpha=arrow_alpha,
        fig_size=fig_size,
        filename=None,
        fig=fig,
        ax=ax,
        dpi=dpi,
        save_kwargs=save_kwargs,
        spacing_mode=spacing_mode,
        spacing_hyperbolic_factor=spacing_hyperbolic_factor,
    )

    # Add colorbar inline if user passed cmap and norm
    if cmap is not None and norm is not None:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(
            np.linspace(getattr(norm, "vmin", 0), getattr(norm, "vmax", 1), 256)
        )
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.02)

        vmin = float(getattr(norm, "vmin", 0.0))
        vmax = float(getattr(norm, "vmax", 1.0))

        # Decide whether ticks should be integer-only: both endpoints near integers and range reasonable
        endpoints_integer_like = math.isclose(
            vmin, round(vmin), abs_tol=1e-8
        ) and math.isclose(vmax, round(vmax), abs_tol=1e-8)
        reasonable_range_for_integers = (vmax - vmin) <= 100
        integer_ticks = endpoints_integer_like and reasonable_range_for_integers

        # Use MaxNLocator to adaptively choose up to 10 ticks
        locator = mticker.MaxNLocator(nbins=10, integer=integer_ticks)
        cbar.locator = locator
        cbar.update_ticks()

        # Format ticks: use integer formatter when integer ticks requested
        if integer_ticks:
            cbar.formatter = mticker.FormatStrFormatter("%d")
            cbar.update_ticks()

        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=12)

    _save_figure(fig, filename, dpi, save_kwargs)

    return fig, ax


def scatter_plot(
    x: Union[np.ndarray, List],
    y: Union[np.ndarray, List],
    xlab: str = "x",
    ylab: str = "y",
    figsize: Tuple[float, float] = (8, 5),
    fontsize: int = 16,
    alpha: float = 0.5,
) -> Tuple[Figure, Axes]:
    """
    Create a simple scatter plot with customizable styling.

    Generates a basic 2D scatter plot with black markers and configurable
    transparency, labels, and sizing.

    Parameters
    ----------
    x : array-like or list
        X-coordinates of the points.
    y : array-like or list
        Y-coordinates of the points.
    xlab : str, optional
        Label for the x-axis, by default 'x'.
    ylab : str, optional
        Label for the y-axis, by default 'y'.
    figsize : tuple of float, optional
        Figure size in inches as (width, height), by default (8, 5).
    fontsize : int, optional
        Font size for axis labels, by default 16.
    alpha : float, optional
        Transparency of markers (0=transparent, 1=opaque), by default 0.5.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        Figure and axis objects containing the scatter plot.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, color="black", alpha=alpha, s=50)
    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    return fig, ax


def scatter_plot_with_colorbar(
    x: Union[np.ndarray, List],
    y: Union[np.ndarray, List],
    xlab: str = "x",
    ylab: str = "y",
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (8, 5),
    fontsize: int = 16,
) -> Tuple[Figure, Axes]:
    """
    Create a density-colored scatter plot using kernel density estimation.

    Generates a 2D scatter plot where points are colored based on their local
    density calculated via Gaussian kernel density estimation. High-density
    regions appear in warmer colors.

    Parameters
    ----------
    x : array-like or list
        X-coordinates of the points.
    y : array-like or list
        Y-coordinates of the points.
    xlab : str, optional
        Label for the x-axis, by default 'x'.
    ylab : str, optional
        Label for the y-axis, by default 'y'.
    cmap : str, optional
        Matplotlib colormap name for density coloring, by default 'viridis'.
    figsize : tuple of float, optional
        Figure size in inches as (width, height), by default (8, 5).
    fontsize : int, optional
        Font size for axis labels, by default 16.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        Figure and axis objects containing the density-colored scatter plot.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=figsize)

    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)

    # Draw the densest points last so they remain visible.
    order = density.argsort()
    ax.scatter(x[order], y[order], c=density[order], cmap=cmap, s=50, alpha=0.8)

    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    return fig, ax


def plot_contourf_full(
    x: Union[np.ndarray, List],
    y: Union[np.ndarray, List],
    xlab: str,
    ylab: str,
    c_map: str = "Purples",
    figsize: Tuple[float, float] = (8, 5),
    fontsize: int = 16,
) -> Tuple[Figure, Axes]:
    """
    Create a filled contour plot of the joint density estimated from paired data.

    Compute a Gaussian kernel density estimate (KDE) over a square grid spanning the
    range of the provided `x` values and render the result with Matplotlib's
    ``contourf``. Axis limits are set to the same range to preserve aspect and the
    function applies the package's standard axis styling helper.

    Parameters
    ----------
    x : array-like or list
        One-dimensional numeric values for the first coordinate.
    y : array-like or list
        One-dimensional numeric values for the second coordinate. Must be the same
        length as ``x``.
    xlab : str
        Label for the x-axis.
    ylab : str
        Label for the y-axis.
    c_map : str or matplotlib.colors.Colormap, optional
        Colormap used for the filled contours. Default is ``"Purples"``.
    figsize : tuple of float, optional
        Figure size in inches as ``(width, height)``. Default is ``(8, 5)``.
    fontsize : int, optional
        Base font size for axis labels and ticks. Default is ``16``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib Figure object containing the contour plot.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object containing the contour plot.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    fig, ax = plt.subplots(figsize=figsize)
    lims = [min(x), max(x)]

    kde = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[
        lims[0] : lims[1] : x.size**0.6 * 1j, lims[0] : lims[1] : y.size**0.6 * 1j
    ]
    zi = kde(np.vstack([xi.ravel(), yi.ravel()]))
    ax.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.9, cmap=c_map)

    ax.set(xlim=lims, ylim=lims)

    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    return fig, ax


def plot_heatmap(
    x: Union[np.ndarray, List],
    y: Union[np.ndarray, List],
    xlab: str,
    ylab: str,
    c_map: str = "viridis",
    nbins: int | Tuple[int, int] | Tuple[np.ndarray, np.ndarray] = 50,
    figsize: Tuple[float, float] = (8, 5),
    fontsize: int = 16,
) -> Tuple[Figure, Axes]:
    """
    Plot a 2D heatmap (binned density) of paired x, y data using a histogram and imshow.

    Creates a 2D histogram of the input coordinates with configurable binning and
    renders the result with Matplotlib's ``imshow``. A colorbar indicating point
    density is attached and axis labels and styling are applied via the package's
    plotting helpers.

    Parameters
    ----------
    x : array-like or list
        X-coordinates of the points. Converted to a NumPy array internally.
    y : array-like or list
        Y-coordinates of the points. Must be the same length as ``x``.
    xlab : str
        Label for the x-axis.
    ylab : str
        Label for the y-axis.
    c_map : str or matplotlib.colors.Colormap, optional
        Colormap used for the heatmap. Default is ``'viridis'``.
    nbins : int, tuple of int, or tuple of array-like, optional
        Binning for the 2D histogram, passed to ``numpy.histogram2d``. If an
        int, the same number of bins is applied to both axes. If a tuple
        ``(nx, ny)`` of ints, uses ``nx`` and ``ny`` bins for x and y
        respectively. If a tuple ``(x_edges, y_edges)`` of monotonically
        increasing arrays, they are used as the explicit bin edges, which fixes
        the bin widths and the axis extent so that several heatmaps can share
        identical axes. Default is ``50``.
    figsize : tuple of float, optional
        Figure size in inches as ``(width, height)``. Default is ``(8, 5)``.
    fontsize : int, optional
        Base font size for axis labels and colorbar label. Default is ``16``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib Figure containing the heatmap.
    ax : matplotlib.axes.Axes
        The Matplotlib Axes containing the heatmap.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    fig, ax = plt.subplots(figsize=figsize)
    heatmap_data, xedges, yedges = np.histogram2d(x, y, bins=nbins)
    im = ax.imshow(
        heatmap_data.T,
        origin="lower",
        cmap=c_map,
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.locator = mticker.MaxNLocator(integer=True)
    cbar.update_ticks()
    cbar.set_label("Count", fontsize=fontsize)
    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    return fig, ax


def scatter_plot_3d_with_colorbar(
    x: Union[np.ndarray, List],
    y: Union[np.ndarray, List],
    z: Union[np.ndarray, List],
    c: Optional[Union[np.ndarray, List]] = None,
    xlab: str = "x",
    ylab: str = "y",
    zlab: str = "z",
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (10, 8),
    fontsize: int = 20,
    alpha: float = 0.8,
    s: Union[float, np.ndarray] = 50,
    labelpad: float = 20,
) -> Tuple[Figure, Axes]:
    """
    Create a 3D scatter plot with an optional colorbar driven by provided values or KDE-based density.

    Generates a 3D scatter plot on a Matplotlib Axes with points colored by the given `c` values.
    If `c` is None, local point density is estimated with a Gaussian KDE and used for coloring.
    A colorbar is attached to the figure and labeled appropriately. The function returns the
    Matplotlib Figure and 3D Axes objects for further customization or saving.

    Parameters
    ----------
    x : array-like or list
        X-coordinates of the points.
    y : array-like or list
        Y-coordinates of the points.
    z : array-like or list
        Z-coordinates of the points.
    c : array-like, list, or None, optional
        Scalar values used to determine point colors. If ``None`` (default), a Gaussian KDE
        is computed on the stacked (x, y, z) coordinates to estimate local point density.
    xlab : str, optional
        Label for the x-axis, by default ``'x'``.
    ylab : str, optional
        Label for the y-axis, by default ``'y'``.
    zlab : str, optional
        Label for the z-axis, by default ``'z'``.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to map ``c`` values to colors, by default ``'viridis'``.
    figsize : tuple of float, optional
        Figure size in inches as ``(width, height)``, by default ``(10, 8)``.
    fontsize : int, optional
        Base font size for axis labels and colorbar, by default ``20``.
    alpha : float, optional
        Marker transparency in the range [0, 1], by default ``0.8``.
    s : float or array-like, optional
        Marker size for the scatter points, by default ``50``.
    labelpad : float, optional
        Padding for axis labels (useful for 3D labels), by default ``20``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib Figure containing the 3D scatter and colorbar.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        Matplotlib 3D Axes containing the scatter plot.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    if c is None:
        xyz = np.vstack([x, y, z])
        c = gaussian_kde(xyz)(xyz)

        # Draw the densest points last so they remain visible.
        order = c.argsort()
        x, y, z, c = x[order], y[order], z[order], c[order]
    else:
        c = np.asarray(c)

    scatter = ax.scatter(x, y, z, c=c, cmap=cmap, s=s, alpha=alpha)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Point Density", fontsize=fontsize - 4)

    ax.set_xlabel(xlab, fontsize=fontsize, labelpad=labelpad)
    ax.set_ylabel(ylab, fontsize=fontsize, labelpad=labelpad)
    ax.set_zlabel(zlab, fontsize=fontsize, labelpad=labelpad)

    for axis_name in ("x", "y", "z"):
        ax.tick_params(axis=axis_name, labelsize=fontsize - 4)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.line.set_linewidth(2.0)
        except Exception:
            pass
    fig.tight_layout()
    return fig, ax


def plot_hexbin_scatter(
    x: Union[np.ndarray, List],
    y: Union[np.ndarray, List],
    xlab: str = "x",
    ylab: str = "y",
    guide_line: bool = True,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (8, 5),
    fontsize: int = 16,
    bins_scale: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """
    Create a hexbin scatter plot with optional y=x guideline and colorbar.

    Generates a hexagonal-binned 2D density plot using Matplotlib's
    ``hexbin`` to visualize the joint distribution of ``x`` and ``y``.
    Provides configurable colormap, bin scaling (e.g. ``'log'``), figure
    sizing and font sizing, and an optional red dashed y=x guideline.

    Parameters
    ----------
    x : array-like or list
        Data for the x-axis. Converted to a NumPy array internally.
    y : array-like or list
        Data for the y-axis. Must be the same length as ``x``.
    xlab : str, optional
        Label for the x-axis. Default is ``'x'``.
    ylab : str, optional
        Label for the y-axis. Default is ``'y'``.
    guide_line : bool, optional
        If ``True``, draw a reference line for ``y = x`` (red dashed).
        Default is ``True``.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap used for the hexbin plot. Default is ``'viridis'``.
    figsize : tuple of float, optional
        Figure size in inches as ``(width, height)``. Default is ``(8, 5)``.
    fontsize : int, optional
        Font size used for axis labels and colorbar label. Default is
        ``16``.
    bins_scale : {str, None}, optional
        Bin scaling mode passed to Matplotlib's ``hexbin`` ``bins``
        parameter. Common value: ``'log'`` for logarithmic binning; if
        ``None`` (default) uses linear counts.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib Figure object containing the hexbin plot.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object containing the hexbin plot.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different lengths or are empty.
    TypeError
        If inputs cannot be converted to numeric arrays.

    Notes
    -----
    - The function wraps Matplotlib's ``ax.hexbin`` and adds a colorbar labeled
      ``'counts'`` by default; when ``bins_scale == 'log'``, zero-count
      hexagons are not shown on a log scale.
    - The hexagon resolution is fixed at a ``gridsize`` of 30 in x.
    - The optional guideline is drawn across the data range and helps to visually
      assess deviations from the identity relationship.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=figsize)

    xlim = x.min(), x.max()
    ylim = y.min(), y.max()

    hb = ax.hexbin(x, y, gridsize=30, cmap=cmap, bins=bins_scale)
    ax.set(xlim=xlim, ylim=ylim)

    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("counts", fontsize=fontsize)

    if guide_line:
        x_line = np.linspace(*xlim, 2)
        ax.plot(x_line, x_line, color="red", linestyle="--", linewidth=2)

    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    return fig, ax


def plot_histogram(
    data: Union[np.ndarray, List],
    bins: Union[int, Sequence[float]] = 30,
    xlab: str = "Values",
    ylab: str = "Frequency",
    figsize: Tuple[float, float] = (8, 5),
    fontsize: int = 16,
) -> Tuple[Figure, Axes]:
    """
    Plot a histogram for a one-dimensional dataset with configurable styling.

    Produces a Matplotlib histogram for numeric data and applies the package's
    standard axis styling. The function is intended for quick exploratory plots
    or for consistent figure generation in scripts and notebooks.

    Parameters
    ----------
    data : array-like or list
        One-dimensional numeric data to plot. Converted to a NumPy array internally.
    bins : int or sequence, optional
        Number of histogram bins (int) or explicit bin edges (sequence). Default is 30.
    xlab : str, optional
        Label for the x-axis. Default is ``'Values'``.
    ylab : str, optional
        Label for the y-axis. Default is ``'Frequency'``.
    figsize : tuple of float, optional
        Figure size in inches as (width, height). Default is ``(8, 5)``.
    fontsize : int, optional
        Font size used for axis labels and ticks. Default is 16.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib Figure object containing the histogram.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object containing the histogram.

    Raises
    ------
    ValueError
        If ``data`` is empty or cannot be interpreted as one-dimensional numeric data,
        or if ``bins`` is a non-positive integer.
    TypeError
        If input types prevent numeric conversion (for example non-iterable ``data``).

    Notes
    -----
    - The function delegates styling (ticks, labels, layout) to the module's
      helper routines and calls ``ax.hist`` for the rendering.
    - When ``bins`` is provided as an integer, Matplotlib's default binning rules
      are used; pass explicit bin edges to control bin placement precisely.
    - For publication-quality figures, modify ``figsize`` and ``fontsize`` and
      save the returned ``fig`` with appropriate ``dpi`` and ``bbox_inches`` settings.
    """
    data = np.asarray(data)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(data, bins=bins, color="blue", edgecolor="black", alpha=0.8)
    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    return fig, ax


def plot_histogram_all_x(
    data: Union[np.ndarray, List],
    xlab: str = "Number of Bonds",
    ylab: str = "Frequency",
    figsize: Tuple[float, float] = (8, 5),
    fontsize: int = 16,
) -> Tuple[Figure, Axes]:
    """
    Plot a histogram using integer bins spanning the full range of the input data.

    Creates a histogram whose bin edges are chosen to cover every integer value
    present in `data` (from floor(min) to ceil(max)). This is useful for discrete
    integer-valued data (e.g. counts or number of bonds) where each integer value
    should map to its own bin.

    Parameters
    ----------
    data : array-like or list
        One-dimensional numeric data to plot. Converted to a NumPy array internally.
    xlab : str, optional
        Label for the x-axis. Default is ``'Number of Bonds'``.
    ylab : str, optional
        Label for the y-axis. Default is ``'Frequency'``.
    figsize : tuple of float, optional
        Figure size in inches as (width, height). Default is ``(8, 5)``.
    fontsize : int, optional
        Font size used for axis labels and ticks. Default is 16.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib Figure object containing the histogram.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object containing the histogram.

    Raises
    ------
    ValueError
        If ``data`` is empty or cannot be interpreted as numeric data.
    TypeError
        If input types prevent numeric conversion.

    Notes
    -----
    - Bins are constructed as ``range(floor(min(data)), ceil(max(data)) + 2)`` so
      that each integer gets its own bin.
    - Intended for discrete integer data; for continuous data use a fixed bin count
      or another binning strategy via ``plot_histogram``.
    - The function delegates plotting to Matplotlib and applies shared styling
      from ``ax_plot``.
    """
    data = np.asarray(data)
    bins = range(int(data.min()), int(data.max()) + 2)
    return plot_histogram(
        data, bins=bins, xlab=xlab, ylab=ylab, figsize=figsize, fontsize=fontsize
    )


def plot_histogram_compare(
    data1: Union[np.ndarray, List],
    data2: Union[np.ndarray, List],
    labels: Sequence[str],
    bins: Union[int, Sequence[float]] = 30,
    xlab: str = "Values",
    ylab: str = "Frequency",
    y_scale: Optional[str] = "log",
    figsize: Tuple[float, float] = (8, 5),
    fontsize: int = 16,
) -> Tuple[Figure, Axes]:
    """
    Plot comparative histograms for two datasets with optional logarithmic y-scale.

    Creates side-by-side histogram overlays for two datasets to facilitate visual
    comparison. Supports configurable binning, axis labels, figure sizing, font
    sizes and optional logarithmic scaling on the y-axis.

    Parameters
    ----------
    data1 : array-like or list
        First dataset for comparison. Converted to a NumPy array internally.
    data2 : array-like or list
        Second dataset for comparison. Converted to a NumPy array internally.
    labels : sequence of str
        Legend labels for ``data1`` and ``data2``, respectively.
    bins : int or sequence, optional
        Number of histogram bins (int) or explicit bin edges (sequence). Default is 30.
    xlab : str, optional
        Label for the x-axis. Default is ``'Values'``.
    ylab : str, optional
        Label for the y-axis. Default is ``'Frequency'``.
    y_scale : {str, None}, optional
        Y-axis scale; use ``'log'`` for logarithmic scale. Default is ``'log'``.
    figsize : tuple of float, optional
        Figure size in inches as (width, height). Default is ``(8, 5)``.
    fontsize : int, optional
        Font size for axis labels and legend. Default is 16.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib Figure object containing the histograms.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object containing the histograms.

    Raises
    ------
    ValueError
        If ``data1`` or ``data2`` are empty, or if ``labels`` does not contain two strings.
    TypeError
        If inputs cannot be converted to numeric arrays.

    Notes
    -----
    - When ``y_scale == 'log'``, care should be taken with zero or negative bin
      counts, which cannot be displayed on a logarithmic axis.
    - Both datasets are plotted on the same axes and share the same binning to
      ensure a direct comparison.
    - For reproducible styling, pass fully defined parameters (bins, figsize, fontsize).
    """
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    fig, ax = plt.subplots(figsize=figsize)
    for i, data in enumerate((data1, data2)):
        ax.hist(data, bins=bins, alpha=0.8, label=labels[i])
    ax.legend()

    if y_scale is not None:
        ax.set_yscale(y_scale)

    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    return fig, ax


def plot_kde(
    data: Union[np.ndarray, List],
    bandwidth: Optional[float] = None,
    grid_size: int = 1000,
    y_scale: Optional[str] = "log",
    xlab: str = "Value",
    ylab: str = "Frequency",
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    fig_size: Tuple[float, float] = (8, 5),
    fontsize: int = 16,
) -> Tuple[Figure, Axes]:
    """
    Plot a Kernel Density Estimate (KDE) and convert density to expected counts.

    Computes a KDE for one-dimensional data using SciPy's `gaussian_kde`, evaluates
    it on a regular grid, converts the density to expected counts (so the area under
    the curve equals the number of samples), and plots the resulting curve on the
    provided Matplotlib axes or on newly created figure/axes.

    Parameters
    ----------
    data : array-like or list
        One-dimensional numeric data to plot. Converted to a NumPy array internally.
    bandwidth : float or None, optional
        The bandwidth for the KDE. If None, SciPy's default is used.
    grid_size : int, optional
        Number of points in the grid for evaluating the KDE. Default is 1000.
    y_scale : {str, None}, optional
        Y-axis scale; use 'log' for logarithmic scale. Default is 'log'.
    xlab : str, optional
        Label for the x-axis. Default is "Value".
    ylab : str, optional
        Label for the y-axis. Default is "Frequency".
    fig : matplotlib.figure.Figure or None, optional
        Existing Figure to plot on. If None, a new one is created.
    ax : matplotlib.axes.Axes or None, optional
        Existing Axes to plot on. If None, a new one is created.
    fig_size : tuple of float, optional
        Size of the figure to create if `fig` is None. Default is (8, 5).
    fontsize : int, optional
        Font size for axis labels. Default is 16.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib Figure object.
    ax : matplotlib.axes.Axes
        The Matplotlib Axes object.

    Raises
    ------
    ValueError
        If `data` is empty.
    TypeError
        If `data` cannot be converted to a numeric array.

    Notes
    -----
    - The function converts the KDE output (a probability density) to expected
      counts for plotting, making the y-axis more interpretable.
    - Bandwidth behaviour follows SciPy's ``gaussian_kde`` semantics. Passing a
      value overrides the default estimator (e.g., 'scott' or 'silverman').
    - When ``y_scale`` is set to ``'log'``, zero or negative plotted values may
      not be visible.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    data = np.asarray(data)
    kde = gaussian_kde(data, bw_method=bandwidth)

    x_min, x_max = data.min(), data.max()
    xs = np.linspace(x_min, x_max, grid_size)

    # Scale the density by sample count and grid spacing to get expected counts.
    dx = xs[1] - xs[0]
    counts = kde(xs) * len(data) * dx

    ax.set_xlim(x_min, x_max)
    if y_scale is not None:
        ax.set_yscale(y_scale)

    ax.plot(xs, counts, color="red", lw=2)
    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    return fig, ax


def _layout_layers(G, subset_key, *, allow_mixed=False):
    """Return sorted layer keys, stable node orders, and node-to-layer indices."""
    groups = defaultdict(list)
    for node, data in G.nodes(data=True):
        groups[data.get(subset_key, 0)].append(node)

    keys = set(groups) if allow_mixed else groups
    try:
        layer_keys = sorted(keys)
    except TypeError:
        if not allow_mixed:
            raise
        layer_keys = sorted(keys, key=str)

    layers = [
        sorted(groups[key], key=lambda node: (G.degree(node), str(node)))
        for key in layer_keys
    ]
    node_layer = {node: i for i, layer in enumerate(layers) for node in layer}
    return layer_keys, layers, node_layer


def _order_layout_layer(target_nodes, neighbor_nodes, neighbor_weights, method):
    """Order a layer by weighted barycenter or median, retaining stable ties."""
    neighbor_index = {node: i for i, node in enumerate(neighbor_nodes)}
    current_index = {node: i for i, node in enumerate(target_nodes)}
    scores = []
    for node in target_nodes:
        score = current_index[node]
        if method == "barycenter":
            total = weight_sum = 0.0
            for neighbor, edge_weight in neighbor_weights(node):
                if neighbor in neighbor_index:
                    total += edge_weight * neighbor_index[neighbor]
                    weight_sum += edge_weight
            if weight_sum > 0:
                score = total / weight_sum
        else:
            positions = []
            for neighbor, edge_weight in neighbor_weights(node):
                if neighbor in neighbor_index:
                    repeats = int(round(edge_weight)) if edge_weight != 1.0 else 1
                    positions.extend([neighbor_index[neighbor]] * max(1, repeats))
            if positions:
                positions.sort()
                middle = len(positions) // 2
                score = (
                    positions[middle]
                    if len(positions) % 2
                    else 0.5 * (positions[middle - 1] + positions[middle])
                )
        scores.append((score, current_index[node], node))

    scores.sort()
    return [node for _, _, node in scores]


def _sweep_layout_layers(layers, neighbor_weights, method, iterations):
    """Refine layer orders with alternating forward and backward sweeps."""
    for _ in range(max(1, int(iterations))):
        for indices, offset in (
            (range(1, len(layers)), -1),
            (range(len(layers) - 2, -1, -1), 1),
        ):
            for i in indices:
                layers[i] = _order_layout_layer(
                    layers[i], layers[i + offset], neighbor_weights, method
                )


def _layout_positions(layers, align, layer_spacing, node_spacing, scale):
    """Place each layer on its fixed axis and center its nodes on the free axis."""
    positions = {}
    for i, nodes in enumerate(layers):
        start = -0.5 * (len(nodes) - 1) * node_spacing
        fixed = i * layer_spacing
        for j, node in enumerate(nodes):
            free = start + j * node_spacing
            positions[node] = (fixed, free) if align == "vertical" else (free, fixed)
    if scale != 1.0:
        positions = {node: (scale * x, scale * y) for node, (x, y) in positions.items()}
    return positions


def _layout_result(
    positions,
    layer_keys,
    layers,
    routes,
    dummy_prefix,
    return_order,
    return_dummies,
    return_routes,
):
    """Filter dummy positions and append the requested orders and polylines."""
    pos = (
        dict(positions)
        if return_dummies
        else {
            node: xy
            for node, xy in positions.items()
            if not (isinstance(node, str) and node.startswith(dummy_prefix))
        }
    )
    result = [pos]
    if return_order:
        result.append({key: list(layer) for key, layer in zip(layer_keys, layers)})
    if return_routes:
        result.append(
            [
                {
                    "endpoints": route["endpoints"],
                    "nodes": list(route["nodes"]),
                    "points": [
                        positions[node] for node in route["nodes"] if node in positions
                    ],
                }
                for route in routes
            ]
        )
    return tuple(result) if len(result) > 1 else pos


def multipartite_layout_crossmin(
    G: nx.Graph,
    subset_key: str = "subset",
    align: str = "vertical",
    method: str = "barycenter",
    iterations: int = 100,
    layer_spacing: float = 1.0,
    node_spacing: float = 1.0,
    scale: float = 1.0,
    seed: Optional[int] = None,
    weight: Optional[str] = None,
    return_order: bool = False,
) -> Union[
    Dict[Any, Tuple[float, float]],
    Tuple[Dict[Any, Tuple[float, float]], Dict[Any, List[Any]]],
]:
    """Minimize crossings between adjacent layers using stable ordering sweeps.

    Nodes start in degree/name order within each layer. Alternating forward and
    backward sweeps order each layer by its neighbors in the adjacent layer;
    ties retain the current order. The graph is not modified.

    Parameters
    ----------
    G : networkx.Graph
        Graph whose nodes provide the layer attribute; missing values use 0.
    subset_key : str, optional
        Layer attribute name. Layer values must be mutually sortable.
    align : {'vertical', 'horizontal'}, optional
        Arrange layers in columns or rows, respectively.
    method : {'barycenter', 'median'}, optional
        Order nodes by their weighted mean or unweighted median neighbor index.
    iterations : int, optional
        Number of forward/backward sweep pairs; at least one pair is performed.
    layer_spacing, node_spacing : float, optional
        Spacing between layers and between nodes within each centered layer.
    scale : float, optional
        Multiply final coordinates by this factor.
    seed : int or None, optional
        Reset Python's random seed when supplied. Ordering itself uses stable ties.
    weight : str or None, optional
        Edge attribute for barycenter weights; missing weights default to 1.
        Median ordering ignores edge weights.
    return_order : bool, optional
        Also return the final node ordering for each layer.

    Returns
    -------
    pos : dict
        Node-to-(x, y) coordinates, with layers on the fixed axis and centered
        node positions on the free axis.
    orders : dict, optional
        Layer-to-node-list mapping, returned as ``(pos, orders)`` when requested.

    Notes
    -----
    The heuristic does not guarantee a global minimum of edge crossings.
    """

    if seed is not None:
        random.seed(seed)
    layer_keys, layers, _ = _layout_layers(G, subset_key)

    def neighbor_weights(node):
        for neighbor in G[node]:
            edge_weight = (
                G[node][neighbor].get(weight, 1.0)
                if method == "barycenter" and weight is not None
                else 1.0
            )
            yield neighbor, edge_weight

    _sweep_layout_layers(layers, neighbor_weights, method, iterations)
    pos = _layout_positions(layers, align, layer_spacing, node_spacing, scale)
    if return_order:
        return pos, {key: list(layer) for key, layer in zip(layer_keys, layers)}
    return pos


def multipartite_layout_crossmin_long(
    G: nx.Graph,
    subset_key: str = "subset",
    align: str = "vertical",
    method: str = "barycenter",
    iterations: int = 100,
    layer_spacing: float = 1.0,
    node_spacing: float = 1.0,
    scale: float = 1.0,
    seed: Optional[int] = None,
    weight: Optional[str] = None,
    insert_dummies: bool = True,
    dummy_prefix: str = "__dummy__",
    return_order: bool = False,
    return_dummies: bool = False,
    return_routes: bool = False,
) -> Union[Dict[Any, Tuple[float, float]], Tuple[Any, ...]]:
    """Minimize layer crossings, optionally routing long edges through dummy nodes.

    Stable barycenter/median sweeps use an undirected adjacency map, with
    parallel-edge weights accumulated. Dummy nodes occupy intermediate layers
    along long edges. The input graph is not modified.

    Parameters
    ----------
    G : networkx.Graph
        Graph whose nodes provide the layer attribute; missing values use 0.
    subset_key : str, optional
        Layer attribute name. Mixed layer types are sorted by string if needed.
    align : {'vertical', 'horizontal'}, optional
        Arrange layers in columns or rows, respectively.
    method : {'barycenter', 'median'}, optional
        Order nodes by weighted mean or median neighbor index. Median ordering
        repeats each neighbor index by its rounded weight, with a minimum of one.
    iterations : int, optional
        Number of forward/backward sweep pairs; at least one pair is performed.
    layer_spacing, node_spacing : float, optional
        Spacing between layers and between nodes within each centered layer.
    scale : float, optional
        Multiply final coordinates by this factor.
    seed : int or None, optional
        Reset Python's random seed when supplied. Ordering itself uses stable ties.
    weight : str or None, optional
        Edge weight attribute; missing weights default to 1.
    insert_dummies : bool, optional
        Split long edges into adjacent-layer hops. Disabling insertion uses the
        multigraph edge iterator with ``keys=False``.
    dummy_prefix : str, optional
        Prefix for sequential dummy node names and for filtering dummy positions.
    return_order : bool, optional
        Append the final layer-to-node-list mapping, including dummy nodes.
    return_dummies : bool, optional
        Include dummy positions in ``pos``; does not add a separate return value.
    return_routes : bool, optional
        Append routes with ``endpoints``, ``nodes``, and coordinate ``points``.

    Returns
    -------
    pos : dict
        Node-to-(x, y) coordinates, omitting dummy-prefixed nodes unless requested.
    tuple, optional
        ``(pos, orders)``, ``(pos, routes)``, or ``(pos, orders, routes)`` when
        the corresponding flags are enabled. Orders and routes include dummies
        even when their positions are omitted from ``pos``.
    """

    if seed is not None:
        random.seed(seed)
    layer_keys, layers, node_layer = _layout_layers(G, subset_key, allow_mixed=True)
    neighbors = defaultdict(lambda: defaultdict(float))
    routes = []
    dummy_count = 0

    def edge_weight(u, v, data):
        if weight is None:
            return 1.0
        if weight in data:
            return data[weight]
        try:
            return G[u][v].get(weight, 1.0)
        except Exception:
            return 1.0

    def add_edge(u, v, value):
        neighbors[u][v] += value
        neighbors[v][u] += value

    if insert_dummies:
        for u, v, data in G.edges(data=True):
            value = edge_weight(u, v, data)
            left, right = node_layer[u], node_layer[v]
            endpoints = (u, v)
            if left == right:
                add_edge(u, v, value)
                routes.append({"endpoints": endpoints, "nodes": [u, v]})
                continue

            reverse = left > right
            if reverse:
                u, v = v, u
                left, right = right, left
            if right - left == 1:
                add_edge(u, v, value)
                routes.append({"endpoints": endpoints, "nodes": [u, v]})
                continue

            chain = [u]
            for i in range(left + 1, right):
                dummy_count += 1
                dummy = f"{dummy_prefix}{dummy_count}"
                node_layer[dummy] = i
                layers[i].append(dummy)
                add_edge(chain[-1], dummy, value)
                chain.append(dummy)
            add_edge(chain[-1], v, value)
            chain.append(v)
            if reverse:
                chain.reverse()
            routes.append({"endpoints": (chain[0], chain[-1]), "nodes": chain})
    else:
        for u, v, data in G.edges(data=True, keys=False):
            add_edge(u, v, edge_weight(u, v, data))
            routes.append({"endpoints": (u, v), "nodes": [u, v]})

    _sweep_layout_layers(
        layers, lambda node: neighbors[node].items(), method, iterations
    )
    positions = _layout_positions(layers, align, layer_spacing, node_spacing, scale)
    return _layout_result(
        positions,
        layer_keys,
        layers,
        routes,
        dummy_prefix,
        return_order,
        return_dummies,
        return_routes,
    )


class _BIT:
    """Fenwick tree of float weights, with logarithmic updates and prefix sums."""

    def __init__(self, n: int) -> None:
        """Create an empty tree with ``n`` elements."""
        self.n = n
        self.t = [0.0] * (n + 1)

    def add(self, i: int, delta: float) -> None:
        """Add ``delta`` to the element at zero-based index ``i``."""
        i += 1
        while i <= self.n:
            self.t[i] += delta
            i += i & -i

    def sum_prefix(self, i: int) -> float:
        """Sum elements through zero-based index ``i`` (zero for negative indices)."""
        if i < 0:
            return 0.0
        total = 0.0
        i += 1
        while i > 0:
            total += self.t[i]
            i -= i & -i
        return total


def _pair_crossings_weighted(
    order_left: List[Any], order_right: List[Any], edges: List[Tuple[Any, Any, float]]
) -> float:
    """Count weighted crossings between two ordered, adjacent layers.

    Each crossing contributes the product of its two edge weights. Edges with
    a shared endpoint do not cross; edges whose endpoints are missing from the
    corresponding order are ignored. A Fenwick tree counts weighted inversions
    after sorting edges by left position, then right position.

    Parameters
    ----------
    order_left, order_right : list
        Ordered, unique, hashable nodes in the two layers.
    edges : list of tuple
        ``(left_node, right_node, weight)`` triples with numeric weights.

    Returns
    -------
    float
        Sum of the products of weights for crossing edge pairs.
    """

    left_index = {node: i for i, node in enumerate(order_left)}
    right_index = {node: i for i, node in enumerate(order_right)}
    triples = [
        (left_index[u], right_index[v], float(value))
        for u, v, value in edges
        if u in left_index and v in right_index
    ]
    if not triples:
        return 0.0
    triples.sort(key=lambda item: (item[0], item[1]))

    # Sorting ties by right position prevents edges with a shared endpoint
    # from contributing crossings to the weighted inversion count.
    tree = _BIT(len(order_right))
    crossings = total = 0.0
    for _, right, value in triples:
        crossings += value * (total - tree.sum_prefix(right))
        tree.add(right, value)
        total += value
    return crossings


def multipartite_layout_sa(
    G: nx.Graph,
    subset_key: str = "subset",
    align: str = "vertical",
    insert_dummies: bool = True,
    dummy_prefix: str = "__dummy__",
    node_spacing: float = 1.0,
    layer_spacing: float = 1.5,
    scale: float = 1.0,
    weight: Optional[str] = None,
    max_proposals: int = 8000,
    cooling_rate: float = 0.95,
    cooling_interval: int = 200,
    adjacent_swap_prob: float = 0.7,
    stop_after_no_improve: int = 2000,
    T0: Optional[float] = None,
    seed: Optional[int] = None,
    return_order: bool = False,
    return_dummies: bool = False,
    return_routes: bool = False,
) -> Union[Dict[Any, Tuple[float, float]], Tuple[Any, ...]]:
    """Minimize weighted inter-layer crossings using simulated annealing.

    Nodes start in degree/name order. Annealing explores adjacent and arbitrary
    within-layer swaps, accepting some uphill moves as the temperature falls,
    and returns the best ordering found. The input graph is not modified.

    Parameters
    ----------
    G : networkx.Graph
        Graph whose nodes provide the layer attribute; missing values use 0.
    subset_key : str, optional
        Layer attribute name. Mixed layer types are sorted by string if needed.
    align : {'vertical', 'horizontal'}, optional
        Arrange layers in columns or rows, respectively.
    insert_dummies : bool, optional
        Split long edges into adjacent-layer hops. Disabling insertion uses the
        multigraph edge iterator with ``keys=False``; only adjacent-layer edges
        then contribute to crossing costs.
    dummy_prefix : str, optional
        Prefix for sequential dummy node names and for filtering dummy positions.
    node_spacing, layer_spacing : float, optional
        Spacing between nodes within a centered layer and between layers.
    scale : float, optional
        Multiply final coordinates by this factor.
    weight : str or None, optional
        Edge weight attribute; missing weights default to 1.
    max_proposals : int, optional
        Maximum number of swaps considered during annealing.
    cooling_rate : float, optional
        Temperature multiplier applied at each cooling interval.
    cooling_interval : int, optional
        Number of proposals between temperature reductions, with a minimum of 1.
    adjacent_swap_prob : float, optional
        Probability of choosing an adjacent swap instead of an arbitrary swap.
    stop_after_no_improve : int, optional
        Stop after this many proposals without improving the best crossing cost.
    T0 : float or None, optional
        Initial temperature; estimated from trial swaps when omitted.
    seed : int or None, optional
        Python random seed for reproducible swaps and acceptance decisions.
    return_order : bool, optional
        Append the final layer-to-node-list mapping, including dummy nodes.
    return_dummies : bool, optional
        Include dummy positions in ``pos``; does not add a separate return value.
    return_routes : bool, optional
        Append routes with ``endpoints``, ``nodes``, and coordinate ``points``.

    Returns
    -------
    pos : dict
        Node-to-(x, y) coordinates, omitting dummy-prefixed nodes unless requested.
    tuple, optional
        ``(pos, orders)``, ``(pos, routes)``, or ``(pos, orders, routes)`` when
        the corresponding flags are enabled. Orders and routes include dummies
        even when their positions are omitted from ``pos``.
    """

    if seed is not None:
        random.seed(seed)
    layer_keys, layers, node_layer = _layout_layers(G, subset_key, allow_mixed=True)
    layer_count = len(layers)
    edges_by_pair = [[] for _ in range(max(0, layer_count - 1))]
    routes = []
    dummy_count = 0

    def edge_weight(u, v, data):
        if weight is None:
            return 1.0
        if weight in data:
            return float(data[weight])
        try:
            return float(G[u][v].get(weight, 1.0))
        except Exception:
            return 1.0

    if insert_dummies:
        for u, v, data in G.edges(data=True):
            value = edge_weight(u, v, data)
            left, right = node_layer[u], node_layer[v]
            if left == right:
                routes.append({"endpoints": (u, v), "nodes": [u, v]})
                continue

            reverse = left > right
            if reverse:
                u, v = v, u
                left, right = right, left
            chain = [u]
            for i in range(left + 1, right):
                dummy_count += 1
                dummy = f"{dummy_prefix}{dummy_count}"
                node_layer[dummy] = i
                layers[i].append(dummy)
                edges_by_pair[i - 1].append((chain[-1], dummy, value))
                chain.append(dummy)
            edges_by_pair[right - 1].append((chain[-1], v, value))
            chain.append(v)
            if reverse:
                chain.reverse()
            routes.append({"endpoints": (chain[0], chain[-1]), "nodes": chain})
    else:
        for u, v, data in G.edges(data=True, keys=False):
            value = edge_weight(u, v, data)
            left, right = node_layer[u], node_layer[v]
            if abs(left - right) == 1:
                if left > right:
                    u, v = v, u
                edges_by_pair[min(left, right)].append((u, v, value))
            routes.append({"endpoints": (u, v), "nodes": [u, v]})

    def pair_cross(i):
        """Return the crossing cost for one adjacent pair, or zero at the ends."""
        if i < 0 or i >= layer_count - 1:
            return 0.0
        if not layers[i] or not layers[i + 1] or not edges_by_pair[i]:
            return 0.0
        return _pair_crossings_weighted(layers[i], layers[i + 1], edges_by_pair[i])

    # Swaps preserve layer sizes, so the eligible layers never change.
    candidates = [i for i, layer in enumerate(layers) if len(layer) >= 2]

    def estimate_temperature(samples=64):
        """Estimate an initial temperature from reversible trial swaps."""
        deltas = []
        for _ in range(samples):
            if not candidates:
                break
            i = random.choice(candidates)
            before = pair_cross(i - 1) + pair_cross(i)
            n = len(layers[i])
            a, b = random.randrange(n), random.randrange(n)
            if a == b:
                continue
            layers[i][a], layers[i][b] = layers[i][b], layers[i][a]
            delta = pair_cross(i - 1) + pair_cross(i) - before
            layers[i][a], layers[i][b] = layers[i][b], layers[i][a]
            if delta > 0:
                deltas.append(delta)
        return max(1e-6, sum(deltas) / len(deltas)) if deltas else 1.0

    temperature = estimate_temperature() if T0 is None else float(T0)
    current_total = sum(pair_cross(i) for i in range(layer_count - 1))
    best_total = current_total
    best_layers = [list(layer) for layer in layers]
    last_improve_at = 0

    for step in range(int(max_proposals)):
        if not candidates:
            break
        i = random.choice(candidates)
        n = len(layers[i])
        if random.random() < adjacent_swap_prob:
            a = random.randrange(n - 1)
            b = a + 1
        else:
            a, b = random.randrange(n), random.randrange(n)
            while b == a:
                b = random.randrange(n)

        before = pair_cross(i - 1) + pair_cross(i)
        layers[i][a], layers[i][b] = layers[i][b], layers[i][a]
        delta = pair_cross(i - 1) + pair_cross(i) - before
        accept = delta <= 0
        if not accept:
            probability = math.exp(-delta / max(temperature, 1e-12))
            accept = random.random() < probability

        if accept:
            current_total += delta
            if current_total + 1e-12 < best_total:
                best_total = current_total
                best_layers = [list(layer) for layer in layers]
                last_improve_at = step
        else:
            layers[i][a], layers[i][b] = layers[i][b], layers[i][a]

        if (step + 1) % int(max(1, cooling_interval)) == 0:
            temperature *= float(cooling_rate)
        if step - last_improve_at >= int(stop_after_no_improve):
            break

    positions = _layout_positions(
        best_layers, align, layer_spacing, node_spacing, scale
    )
    return _layout_result(
        positions,
        layer_keys,
        best_layers,
        routes,
        dummy_prefix,
        return_order,
        return_dummies,
        return_routes,
    )


def show_common_bonds(
    smiles_a: str,
    smiles_b: str,
    legends: List[str] | None = None,
    common_bond_color: Tuple[float, float, float] = (0.1, 0.8, 0.1),
    common_atom_color: Tuple[float, float, float] = (0.1, 0.8, 0.1),
    size: Tuple[int, int] = (700, 350),
    timeout_s: int = 5,
    ring_matches_ring_only: bool = True,
    complete_rings_only: bool = True,
) -> Image.Image:
    """
    Visualize the maximum common substructure (MCS) between two molecules.

    This function takes two SMILES strings, computes their MCS, and highlights
    the common atoms and bonds in the resulting visualization. The output is
    an image showing the two molecules side by side with the MCS highlighted.

    Parameters
    ----------
    smiles_a : str
        SMILES string of the first molecule.
    smiles_b : str
        SMILES string of the second molecule.
    legends : List[str] or None, optional
        Legends for the two molecules. Defaults to ["A", "B"] if None.
    common_bond_color : Tuple[float, float, float], optional
        RGB color for highlighting common bonds. Defaults to (0.1, 0.8, 0.1).
    common_atom_color : Tuple[float, float, float], optional
        RGB color for highlighting common atoms. Defaults to (0.1, 0.8, 0.1).
    size : Tuple[int, int], optional
        Size of the output image in pixels (width, height). Defaults to (700, 350).
    timeout_s : int, optional
        Timeout in seconds for the MCS computation. Defaults to 5.
    ring_matches_ring_only : bool, optional
        If True, only matches rings to rings. Defaults to True.
    complete_rings_only : bool, optional
        If True, only matches complete rings. Defaults to True.

    Returns
    -------
    PIL.Image.Image
        An image showing the two molecules with the MCS highlighted. If no MCS
        is found, the molecules are displayed without highlights.

    Raises
    ------
    ValueError
        If one or both SMILES strings cannot be parsed by RDKit.

    Notes
    -----
    - The function uses RDKit to compute the MCS and visualize the molecules.
    - If no MCS is found, the molecules are displayed without any highlights.
    - The function supports customization of colors, image size, and MCS parameters.
    """
    molecules = [Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)]
    if any(mol is None for mol in molecules):
        raise ValueError("One or both SMILES strings could not be parsed by RDKit.")
    molecules = [standardize_mol(mol, add_hydrogens=False) for mol in molecules]
    drawing_options = {
        "molsPerRow": 2,
        "subImgSize": (size[0] // 2, size[1]),
        "legends": ["A", "B"] if legends is None else legends,
    }

    # Keyword arguments remain compatible across RDKit MCSParameters versions.
    mcs = rdFMCS.FindMCS(
        molecules,
        timeout=int(timeout_s),
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        ringMatchesRingOnly=bool(ring_matches_ring_only),
        completeRingsOnly=bool(complete_rings_only),
    )
    query = Chem.MolFromSmarts(mcs.smartsString) if mcs.smartsString else None
    matches = (
        [mol.GetSubstructMatch(query) for mol in molecules] if query is not None else []
    )

    def _mcs_bond_indices(mol: Chem.Mol, match: Tuple[int, ...]) -> List[int]:
        """Map query bonds to their indices in a matched molecule."""
        indices = []
        for bond in query.GetBonds():
            parent_bond = mol.GetBondBetweenAtoms(
                match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()]
            )
            if parent_bond is not None:
                indices.append(parent_bond.GetIdx())
        return indices

    # An empty or unmatchable MCS is rendered without highlights.
    if matches and all(matches):
        atom_lists = [list(match) for match in matches]
        bond_lists = [
            _mcs_bond_indices(mol, match) for mol, match in zip(molecules, matches)
        ]
        drawing_options.update(
            highlightBondLists=bond_lists,
            highlightBondColors=[
                dict.fromkeys(bonds, common_bond_color) for bonds in bond_lists
            ],
            highlightAtomLists=atom_lists,
            highlightAtomColors=[
                dict.fromkeys(atoms, common_atom_color) for atoms in atom_lists
            ],
            useSVG=False,
        )
    return Draw.MolsToGridImage(molecules, **drawing_options)


def _prepare_mol_grid(
    mols: Sequence[Union[Chem.Mol, str]],
    legends: Optional[Sequence[str]],
    max_mols: Optional[int],
) -> Tuple[List[Chem.Mol], List[str]]:
    """Normalize molecules and validate legends after applying the display limit."""
    rdkit_mols = []
    for index, mol in enumerate(mols):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        elif not isinstance(mol, Chem.Mol):
            raise TypeError(
                f"Item {index} is neither an RDKit Mol nor a SMILES string: {type(mol)}"
            )
        rdkit_mols.append(Chem.MolFromSmiles("") if mol is None else mol)

    if max_mols is not None:
        rdkit_mols = rdkit_mols[: int(max_mols)]
    if legends is None:
        return rdkit_mols, [""] * len(rdkit_mols)
    if len(legends) != len(rdkit_mols):
        raise ValueError("legends must be the same length as mols")
    return rdkit_mols, list(legends)


def draw_mol_grid(
    mols: Sequence[Union[Chem.Mol, str]],
    legends: Optional[Sequence[str]] = None,
    n_cols: int = 4,
    sub_img_size: tuple = (200, 200),
    max_mols: Optional[int] = None,
    use_svg: bool = False,
) -> Union[Image.Image, str]:
    """
    Generate a grid image of molecular structures.

    This function takes a sequence of RDKit `Mol` objects or SMILES strings,
    converts them to RDKit `Mol` objects if necessary, and arranges them in a
    grid layout. Optionally, legends can be added below each molecule, and the
    output can be rendered as an SVG image.

    Parameters
    ----------
    mols : Sequence[Union[Chem.Mol, str]]
        A sequence of RDKit `Mol` objects or SMILES strings representing the molecules to be drawn.
    legends : Optional[Sequence[str]], optional
        A sequence of legend strings to display below each molecule. If `None`, no legends are added.
    n_cols : int, optional
        The number of columns in the grid. Must be a positive integer. Defaults to 4.
    sub_img_size : tuple, optional
        The size of each sub-image in the grid, specified as (width, height). Defaults to (200, 200).
    max_mols : Optional[int], optional
        The maximum number of molecules to include in the grid. If `None`, all molecules are included. Defaults to `None`.
    use_svg : bool, optional
        If `True`, the output is rendered as an SVG image. Otherwise, a raster image is generated. Defaults to `False`.

    Returns
    -------
    PIL.Image.Image or str
        The generated grid image. If `use_svg` is `True`, an SVG string is returned. Otherwise, a PIL image is returned.

    Raises
    ------
    ValueError
        If `n_cols` is not a positive integer or if the length of `legends` does not match the number of molecules.
    TypeError
        If an item in `mols` is neither an RDKit `Mol` object nor a SMILES string.

    Notes
    -----
    - If a SMILES string cannot be converted to an RDKit `Mol` object, an empty molecule is used as a placeholder.
    - The function uses RDKit's `MolsToGridImage` for rendering the grid.
    """
    if n_cols <= 0:
        raise ValueError("n_cols must be a positive integer")

    rdkit_mols, legends_list = _prepare_mol_grid(mols, legends, max_mols)
    return Draw.MolsToGridImage(
        mols=rdkit_mols,
        molsPerRow=n_cols,
        subImgSize=sub_img_size,
        legends=legends_list,
        useSVG=use_svg,
    )


def draw_mol_grid_box(
    mols: Sequence[Union[Chem.Mol, str]],
    legends: Optional[Sequence[str]] = None,
    sort_by: Optional[Sequence] = None,
    n_cols: int = 4,
    sub_img_size: Tuple[int, int] = (200, 200),
    max_mols: Optional[int] = None,
    box_bg: str = "#E6E6E6",
    gap: int = 12,
    outer_margin: int = 12,
    inner_pad: int = 10,
) -> Image.Image:
    """
    Draw a grid of molecules, each on its own shaded tile.

    Every molecule is rendered into a padded coloured box, and the boxes are
    tiled onto a white canvas with a configurable gap between them.

    Parameters
    ----------
    mols : sequence of Chem.Mol or str
        The molecules to draw, given as RDKit molecules or SMILES strings.
    legends : sequence of str, optional
        Labels drawn beneath each molecule. Must be the same length as ``mols``.
        Default is None, which draws no labels.
    sort_by : sequence, optional
        Values used to sort the molecules in ascending order before drawing.
        ``legends`` is reordered to match. Default is None, which preserves the
        input order.
    n_cols : int, optional
        Number of columns in the grid. Default is 4.
    sub_img_size : tuple of (int, int), optional
        Width and height in pixels of each tile. Default is ``(200, 200)``.
    max_mols : int, optional
        Maximum number of molecules to draw. Default is None, meaning no limit.
    box_bg : str, optional
        Background colour of each tile. Default is ``"#E6E6E6"``.
    gap : int, optional
        Gap in pixels between adjacent tiles. Default is 12.
    outer_margin : int, optional
        Margin in pixels around the outside of the grid. Default is 12.
    inner_pad : int, optional
        Padding in pixels between a tile edge and its molecule drawing. Default
        is 10.

    Returns
    -------
    PIL.Image.Image
        The assembled grid image. A blank image is returned when ``mols`` is
        empty.

    Raises
    ------
    ValueError
        If ``n_cols`` is not positive, if any of ``gap``, ``outer_margin`` or
        ``inner_pad`` is negative, or if ``legends`` has a different length from
        ``mols``.
    TypeError
        If an item of ``mols`` is neither an RDKit molecule nor a SMILES string.

    See Also
    --------
    draw_mol_grid : Grid drawing without the shaded tile background.
    """
    if n_cols <= 0:
        raise ValueError("n_cols must be a positive integer")
    if gap < 0 or outer_margin < 0 or inner_pad < 0:
        raise ValueError("gap/outer_margin/inner_pad must be >= 0")

    if sort_by is not None:
        order = sorted(range(len(mols)), key=lambda i: sort_by[i])
        mols = [mols[i] for i in order]
        if legends is not None:
            legends = [legends[i] for i in order]

    rdkit_mols, legends_list = _prepare_mol_grid(mols, legends, max_mols)
    if not rdkit_mols:
        return Image.new("RGB", (outer_margin * 2 + 1, outer_margin * 2 + 1), "white")

    n_rows = math.ceil(len(rdkit_mols) / n_cols)
    box_w, box_h = sub_img_size
    inner_size = (max(1, box_w - 2 * inner_pad), max(1, box_h - 2 * inner_pad))
    canvas_size = (
        outer_margin * 2 + n_cols * box_w + (n_cols - 1) * gap,
        outer_margin * 2 + n_rows * box_h + (n_rows - 1) * gap,
    )
    canvas = Image.new("RGB", canvas_size, "white")

    for index, (mol, legend) in enumerate(zip(rdkit_mols, legends_list)):
        row, column = divmod(index, n_cols)
        mol_image = Draw.MolToImage(mol, size=inner_size, legend=legend)
        tile = Image.new("RGB", sub_img_size, box_bg)
        tile.paste(mol_image, (inner_pad, inner_pad))
        position = (
            outer_margin + column * (box_w + gap),
            outer_margin + row * (box_h + gap),
        )
        canvas.paste(tile, position)
    return canvas


def plot_ir_spectrum(
    spectrum: np.ndarray,
    peaks: np.ndarray | None = None,
    highlight_range: Optional[Tuple[float, float]] = (400.0, 1500.0),
    xlab: str = "Wavenumber (cm⁻¹)",
    ylab: str = "Intensity",
    flip_x: bool = True,
    figsize: Tuple[float, float] = (8, 5),
    fontsize: int = 16,
) -> Tuple[Figure, Axes]:
    """
    Plot an infrared spectrum, optionally marking detected peaks.

    Parameters
    ----------
    spectrum : np.ndarray
        Two-column array whose first column holds wavenumbers and whose second
        column holds intensities.
    peaks : np.ndarray, optional
        Indices into ``spectrum`` marking the peaks to highlight in red.
        Default is None, which marks no peaks.
    highlight_range : tuple of (float, float), optional
        Wavenumber range shaded in the background, typically the fingerprint
        region. Default is ``(400.0, 1500.0)``. Pass None to disable shading.
    xlab : str, optional
        Label for the x-axis. Default is ``'Wavenumber (cm⁻¹)'``.
    ylab : str, optional
        Label for the y-axis. Default is ``'Intensity'``.
    flip_x : bool, optional
        If True, invert the x-axis, which is the standard convention for
        infrared spectra. Default is True.
    figsize : tuple of (float, float), optional
        Figure size in inches. Default is ``(8, 5)``.
    fontsize : int, optional
        Font size used for the axis labels and ticks. Default is 16.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The axes holding the spectrum.

    See Also
    --------
    assemblytheorytools.tools_data.load_ir_jcamp_data : Load spectra from JCAMP files.
    assemblytheorytools.tools_data.find_peak_indices_in_range : Locate peak indices.
    """
    freq, intensity = spectrum.T[:2]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(freq, intensity, linewidth=2, color="black")
    if peaks is not None:
        ax.scatter(freq[peaks], intensity[peaks], color="red")

    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    if highlight_range:
        ax.axvspan(*highlight_range, color="lightgrey", alpha=0.5, zorder=0)
    if flip_x:
        ax.invert_xaxis()
    return fig, ax


def plot_ase_atoms(
    atoms: Atoms,
    outfile: Optional[str] = None,
    *,
    rotation: str = "0x,0y,0z",
    show_unit_cell: int = 0,
    fig_size: Tuple[float, float] = (6, 6),
    dpi: int = 300,
    transparent: bool = False,
) -> Tuple[Figure, Axes]:
    """
    Visualize an ASE Atoms object using Matplotlib.

    This function generates a 2D visualization of an ASE Atoms object and optionally
    saves the plot to a file. The visualization can be customized with rotation,
    unit cell display, figure size, and transparency settings.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object to visualize.
    outfile : str or None, optional
        The file path to save the plot. If None, the plot is not saved. Defaults to None.
    rotation : str, optional
        The rotation to apply to the Atoms object, specified as a string (e.g., "90x,0y,0z").
        Defaults to "0x,0y,0z".
    show_unit_cell : int, optional
        Whether to display the unit cell. Set to 1 to show the unit cell, 0 to hide it.
        Defaults to 0.
    fig_size : tuple of float, optional
        The size of the figure in inches (width, height). Defaults to (6, 6).
    dpi : int, optional
        The resolution of the figure in dots per inch. Defaults to 300.
    transparent : bool, optional
        Whether to make the background of the saved figure transparent. Defaults to False.

    Returns
    -------
    tuple
        A tuple containing the Matplotlib Figure and Axes objects.

    Notes
    -----
    - The axis is turned off for a cleaner visualization.
    - If `outfile` is provided, the figure is saved with tight bounding and no padding.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    plot_atoms(atoms, ax=ax, rotation=rotation, show_unit_cell=show_unit_cell)
    ax.axis("off")
    if outfile:
        fig.savefig(
            outfile, dpi=dpi, transparent=transparent, bbox_inches="tight", pad_inches=0
        )
    return fig, ax


def plot_ms2_spectrum(
    ms2_processed: pd.DataFrame,
    parent_mz: float,
    tree: Dict[float, Any],
    figsize: Tuple[float, float] = (8, 5),
) -> None:
    """
    Plot the MS2 spectrum for a given parent m/z value, using processed MS2 data or a fragmentation tree.

    This function visualizes the MS2 fragment ions associated with a specified parent m/z.
    If processed MS2 data is available for the parent, it plots the fragment m/z values
    and their intensities as vertical lines. If no processed data is found, it falls back
    to plotting the fragment m/z values from the provided fragmentation tree, with unit intensity.

    Parameters
    ----------
    ms2_processed : pandas.DataFrame
        DataFrame containing processed MS2 fragment data. Must include columns 'parent', 'mz', and 'intensity'.
    parent_mz : float
        The m/z value of the parent ion for which to plot the MS2 spectrum.
    tree : dict
        Fragmentation tree structure, where keys are parent m/z values and values are dicts of fragment m/z values.
    figsize : tuple of float, optional
        Size of the figure in inches as (width, height). Defaults to (8, 5).

    Returns
    -------
    None
        The function creates and displays a matplotlib plot of the MS2 spectrum.

    Notes
    -----
    - If processed MS2 data for the parent m/z is found, fragment intensities are plotted.
    - If no processed data is found, fragment m/z values from the tree are plotted with intensity 1.
    - The function applies standard axis styling and layout for publication-quality figures.
    """
    parent_data = ms2_processed[
        ms2_processed["parent"].between(parent_mz - 0.01, parent_mz + 0.01)
    ]
    fig, ax = plt.subplots(figsize=figsize)
    plot_df = None
    if len(parent_data) > 0:
        plot_df = parent_data.sort_values("mz")
        ax.vlines(plot_df["mz"], 0, plot_df["intensity"], color="black", linewidth=1.5)
        print(f"Plotting all {len(plot_df)} processed MS2 fragments", flush=True)
    else:
        fragments = sorted(tree[parent_mz])
        ax.vlines(fragments, 0, 1, color="black", linewidth=1.5)
        print(f"Plotting {len(fragments)} MS2 fragments from tree", flush=True)
    ax.axhline(y=0, color="gray", linewidth=1)
    ax.set_title(
        f"Processed Fragments (parent m/z {parent_mz:.2f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(0, max(plot_df["mz"]) + 20 if len(plot_df) > 0 else 300)
    ax.grid(alpha=0.3)
    ax_plot(fig, ax, "MS2 m/z", "Intensity")
    fig.tight_layout()
