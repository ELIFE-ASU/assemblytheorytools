import math
import random
import tempfile
from collections import defaultdict
from html import escape
from typing import List, Optional, Dict, Tuple, Any, Union, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
from IPython.display import HTML
from PIL import Image
from ase import Atoms
from ase.visualize.plot import plot_atoms
from matplotlib import colormaps, colors
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Circle
from pyvis.network import Network
from rdkit import Chem
from rdkit.Chem import Draw, rdFMCS
from scipy.stats import gaussian_kde

from .tools_atoms import mol_to_atoms
from .tools_data import pubchem_smi_to_name, enumerate_stereoisomers_shortest
from .tools_graph import relabel_digraph, nx_to_smi, set_graph_layer
from .tools_mol import smi_to_mol, standardize_mol

# set the plot axis
plt.rcParams['axes.linewidth'] = 2.0


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
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    plt.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    plt.tick_params(axis='both', which='both', top=True, right=True)
    plt.xlabel(xlab, fontsize=xs)
    plt.ylabel(ylab, fontsize=ys)
    plt.tight_layout()
    return None


def ax_plot(fig: plt.Figure, ax: plt.Axes, xlab: str, ylab: str, xs: int = 14, ys: int = 14) -> None:
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
    ax.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    ax.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    ax.tick_params(axis='both', which='both', top=True, right=True)
    ax.set_xlabel(xlab, fontsize=xs)
    ax.set_ylabel(ylab, fontsize=ys)
    fig.tight_layout()
    return None


def plot_graph(graph: nx.Graph,
               fig_size: tuple = (12, 7),
               layout: str = 'kawai',
               f_labs: bool = False,
               edge_color: str = 'grey',
               node_size: int = 300,
               edgecolors: str = "black",
               width: int = 2,
               linewidths: int = 2,
               seed: int = 42) -> tuple[Figure, Axes]:
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
    
    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        Figure and axis objects containing the graph visualization.
    """
    graph = graph.copy()  # Avoid modifying the original graph
    # Get the position of the nodes based on the specified layout
    if layout == 'kawai':
        pos = nx.kamada_kawai_layout(graph)
    elif layout == 'spring':
        pos = nx.spring_layout(graph, seed=seed)
    elif layout == 'circular':
        pos = nx.circular_layout(graph)
    elif layout == 'shell':
        pos = nx.shell_layout(graph)
    elif layout == 'spectral':
        pos = nx.spectral_layout(graph)
    elif layout == 'spiral':
        pos = nx.spiral_layout(graph)
    elif layout == 'arf':
        pos = nx.arf_layout(graph)
    elif layout == 'topological':
        graph = set_graph_layer(graph)
        pos = nx.multipartite_layout(graph, subset_key="layer")
    else:
        pos = nx.kamada_kawai_layout(graph)

    fig, ax = plt.subplots(figsize=fig_size)

    # Draw the graph with the specified parameters
    nx.draw_networkx(graph,
                     ax=ax,
                     pos=pos,
                     with_labels=f_labs,
                     edge_color=edge_color,
                     node_size=node_size,
                     edgecolors=edgecolors,
                     width=width,
                     linewidths=linewidths)
    fig.tight_layout()
    ax.axis('off')
    return fig, ax


def plot_mol_graph(graph: nx.Graph,
                   fig_size: tuple = (12, 7),
                   layout: str = 'kawai',
                   f_labs: bool = False,
                   node_size: int = 300,
                   width: int = 2,
                   linewidths: int = 2,
                   seed: int = 42) -> tuple[Figure, Axes]:
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
    
    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        Figure and axis objects containing the molecular graph visualization.
    """
    graph = graph.copy()  # Avoid modifying the original graph
    # Get the position of the nodes based on the specified layout
    if layout == 'kawai':
        pos = nx.kamada_kawai_layout(graph)
    elif layout == 'spring':
        pos = nx.spring_layout(graph, seed=seed)
    elif layout == 'circular':
        pos = nx.circular_layout(graph)
    elif layout == 'shell':
        pos = nx.shell_layout(graph)
    elif layout == 'spectral':
        pos = nx.spectral_layout(graph)
    elif layout == 'spiral':
        pos = nx.spiral_layout(graph)
    elif layout == 'arf':
        pos = nx.arf_layout(graph)
    else:
        pos = nx.kamada_kawai_layout(graph)

    cols_conv = {
        'H': 'white',  # Hydrogen
        'C': 'darkgray',  # Carbon
        'O': 'red',  # Oxygen
        'N': 'blue',  # Nitrogen
        'S': 'yellow',  # Sulfur
        'P': 'orange',  # Phosphorus
        'Cl': 'green',  # Chlorine
        'F': 'lightgreen',  # Fluorine
        'Br': 'brown',  # Bromine
        'I': 'purple',  # Iodine
        'Fe': 'darkorange',  # Iron
        'Ca': 'gold',  # Calcium
        'Na': 'lightblue',  # Sodium
        'K': 'violet',  # Potassium
        'Mg': 'darkgreen',  # Magnesium
        'Cu': 'peru',  # Copper
        'Zn': 'gray',  # Zinc
        'Au': 'gold',  # Gold
        'Ag': 'silver',  # Silver
        'Pt': 'lightgray'  # Platinum
    }

    color_dict_edge = {1: "black",
                       2: "green",
                       3: "red",
                       4: "orange"}
    # Get the colors for the nodes
    graph_colors = [cols_conv.get(graph.nodes[idx]['color'], 'black') for idx in graph.nodes()]
    # Get the colors for the edges
    edge_colors = [color_dict_edge.get(graph.edges[idx]['color']) for idx in graph.edges()]

    fig, ax = plt.subplots(figsize=fig_size)

    # Draw the graph
    nx.draw_networkx(graph,
                     ax=ax,
                     pos=pos,
                     with_labels=f_labs,
                     node_size=node_size,
                     edge_color=edge_colors,
                     node_color=graph_colors,
                     edgecolors="black",
                     width=width,
                     linewidths=linewidths)
    fig.tight_layout()
    ax.axis('off')
    return fig, ax


def plot_interactive_graph(graph: nx.Graph,
                           show: bool = False,
                           filename: str = "interactive_graph.html") -> Network:
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
    graph = graph.copy()  # Avoid modifying the original graph
    # Color each node based on its degree
    max_nbr = len(max(graph.adj.values(), key=lambda x: len(x)))
    blues = colormaps.get_cmap("Blues")
    for n, d in graph.nodes(data=True):
        n_neighbors = len(graph.adj[n])
        # Show the smaller domain in red and the larger one in blue
        palette = blues
        d["color"] = colors.to_hex(palette(n_neighbors / max_nbr))

    # Convert to PyVis network
    width, height = (900, 900)
    net = Network(width=f"{width}px", height=f"{height}px", notebook=True, heading="")
    net.from_nx(graph)
    if show:
        html_doc = net.generate_html(notebook=True)
        iframe = (
            f'<iframe width="{width + 25}px" height="{height + 25}px" frameborder="0" '
            'srcdoc="{html_doc}"></iframe>'
        )
        HTML(iframe.format(html_doc=escape(html_doc)))
    else:
        # Save the graph
        net.show(filename)
    return net


def plot_digraph_metro(digraph: nx.DiGraph,
                       filename: str = 'metro',
                       steps: bool = False,
                       vo_str: bool = True,
                       vo_names: str | None = None) -> None:
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
        raise ImportError("The 'dagviz' and 'cairosvg' packages are required for this function.\n"
                          "Please install them via pip:\n"
                          "pip install git+https://github.com/ELIFE-ASU/dagviz.git \n"
                          "pip install cairosvg \n") from e

    if steps:
        # Relabel the graph nodes with their topological step if requested
        digraph = relabel_digraph(digraph)

    if vo_str:
        for node in digraph.nodes:
            d_type = type(digraph.nodes[node]['vo'])
            if d_type == str:
                lab = digraph.nodes[node]['vo']
            elif d_type == nx.Graph:
                lab = nx_to_smi(digraph.nodes[node]['vo'],
                                add_hydrogens=False,
                                sanitize=True)
            elif d_type == Chem.Mol:
                lab = Chem.MolToSmiles(digraph.nodes[node]['vo'])
            else:
                raise ValueError(f"Unsupported virtual object type: {d_type}")

            if vo_names:
                lab = enumerate_stereoisomers_shortest(Chem.MolFromSmiles(lab), prefer=vo_names)
                lab = pubchem_smi_to_name(lab, prefer=vo_names)
                if lab is None:
                    lab = ""
            digraph.nodes[node]['label'] = lab

    # Configure the metro-style rendering backend
    backend = dagviz.style.metro.svg_renderer(dagviz.style.metro.StyleConfig(node_stroke="black"))
    # Render the graph as an SVG string
    r = dagviz.render_svg(digraph, style=backend)

    # Save the SVG file
    with open(f'{filename}.svg', 'w') as file:
        file.write(r)

    # Convert the SVG to a PNG file
    cairosvg.svg2png(bytestring=r.encode('utf-8'), write_to=f"{filename}.png")
    return None


def plot_pathway(graph: nx.DiGraph,
                 fig_size: tuple = (12, 7),
                 show_icons: bool = True,
                 node_color: str = '#264f70',
                 plot_type: str = 'mol',
                 arrow_style: str = '1',
                 layout_style: str = 'crossmin_long',
                 frame_on: bool = True,
                 font_size: int = 11,
                 arrow_color: str = '#264f70',
                 plt_arrow_style='->') -> tuple[Figure, Axes]:
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
    plt_arrow_style : str or ArrowStyle object from matplotlib.patches, optional
        Style of the arrowheads in the plot, by default '->'.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        Figure and axis objects containing the pathway visualization.
    
    Raises
    ------
    ValueError
        If arrow_style is not '1' or '2'.
    """
    graph = graph.copy()  # Avoid modifying the original graph
    # If the input is a graph check if it contains molecule graphs and convert to smiles
    if plot_type == 'mol':
        for node in graph.nodes:
            node_graph = graph.nodes[node]['vo']
            if isinstance(node_graph, nx.Graph):
                try:
                    smi = nx_to_smi(node_graph, add_hydrogens=False, sanitize=False)
                    graph.nodes[node]['vo'] = smi
                except:
                    plot_type = 'graph'
    elif plot_type == "string":
        if show_icons:
            node_color = 'white'

    fig, ax = plt.subplots(figsize=fig_size)
    graph = set_graph_layer(graph)

    if layout_style == 'crossmin':
        pos = multipartite_layout_crossmin(graph, subset_key="layer")
    elif layout_style == 'crossmin_long':
        pos = multipartite_layout_crossmin_long(graph, subset_key="layer")
    elif layout_style == 'sa':
        pos = multipartite_layout_sa(graph, subset_key="layer")
    else:
        pos = nx.multipartite_layout(graph, subset_key="layer")

    if arrow_style == '1':
        edge_color1 = 'white'
        edge_color = arrow_color
    elif arrow_style == '2':
        edge_color1 = 'grey'
        edge_color = 'grey'  # Unused
    else:
        raise ValueError("Invalid arrow style. Use '1' or '2'.")

    nx.draw_networkx(graph,
                     pos=pos,
                     ax=ax,
                     with_labels=False,
                     node_size=1000,
                     node_color=node_color,
                     connectionstyle="arc3,rad=0.1",
                     edge_color=edge_color1,
                     arrows=True,
                     arrowstyle="->",
                     width=2.0)

    if arrow_style == '1':
        if show_icons:
            arrow_margin = 70
        else:
            arrow_margin = 20

        for edge in graph.edges():
            src, dst = edge
            # If the source node is above the destination node, curve the arrow downward (negative rad)
            if pos[src][1] > pos[dst][1]:
                rad = -0.15
            # If the source node is below the destination node, curve the arrow upward (positive rad)
            elif pos[src][1] < pos[dst][1]:
                rad = 0.15
            # If the source and destination nodes are horizontally aligned
            else:
                layer_diff = graph.nodes[dst]["layer"] - graph.nodes[src]["layer"]
                if layer_diff > 1:
                    # flip a coin to decide the direction of the curve
                    if random.random() > 0.5:
                        rad = 0.  # -0.10 * layer_diff # HARDCODED, don't push this block!
                    else:
                        rad = 0.  # 0.10 * layer_diff
                else:
                    rad = 0.0

            nx.draw_networkx_edges(
                graph,
                pos=pos,
                edgelist=[edge],
                ax=ax,
                arrows=True,
                arrowstyle=plt_arrow_style,
                width=2.5,
                edge_color=edge_color,
                connectionstyle=f"arc3,rad={rad}",
                min_target_margin=arrow_margin,
            )

    if show_icons:
        if plot_type == 'mol':
            for i, node in enumerate(graph.nodes):
                smi = graph.nodes[node]["vo"]
                smi = smi.replace('[', '').replace(']', '')
                mol = smi_to_mol(smi, add_hydrogens=False)
                img = Draw.MolToImage(mol,
                                      size=(200, 200),
                                      kekulize=False,
                                      fitImage=True)
                imagebox = OffsetImage(img, zoom=0.4)
                ab = AnnotationBbox(imagebox,
                                    xy=(pos[node][0], pos[node][1]),
                                    frameon=frame_on)
                ax.add_artist(ab)
        elif plot_type == 'graph':
            for i, node in enumerate(graph.nodes):
                atom_graph = graph.nodes[node]["vo"]
                with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmpfile:
                    _fig, _ax = plot_mol_graph(atom_graph,
                                               f_labs=False,
                                               fig_size=(4, 4),
                                               node_size=1000,
                                               width=10)
                    # save the figure to a temporary file
                    _fig.savefig(tmpfile.name, dpi=400, bbox_inches='tight')
                    # close the figure
                    plt.close(_fig)

                    img = plt.imread(tmpfile.name)

                imagebox = OffsetImage(img, zoom=0.05)
                ab = AnnotationBbox(imagebox, (pos[node][0], pos[node][1]), frameon=frame_on)
                ax.add_artist(ab)
        elif plot_type == 'atoms':
            for i, node in enumerate(graph.nodes):
                smi = graph.nodes[node]["vo"]
                mol = smi_to_mol(smi, add_hydrogens=False)
                atoms = mol_to_atoms(mol, sanitize=False, add_hydrogens=False)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmpfile:
                    _fig, _ax = plt.subplots()
                    plot_atoms(atoms, _ax, show_unit_cell=0, scale=2.0)
                    _fig.tight_layout()
                    _ax.axis('off')
                    _fig.savefig(tmpfile.name, dpi=500, bbox_inches='tight')

                    # close the figure
                    plt.close(_fig)
                    img = plt.imread(tmpfile.name)

                imagebox = OffsetImage(img, zoom=0.02)
                ab = AnnotationBbox(imagebox, (pos[node][0], pos[node][1]), frameon=frame_on)
                ax.add_artist(ab)
        elif plot_type == "string":
            for node in graph.nodes:
                s = graph.nodes[node]["vo"]
                ax.text(pos[node][0],
                        pos[node][1],
                        s,
                        fontsize=font_size,
                        ha='center',
                        va='center',
                        bbox=dict(boxstyle='round,pad=0.5',
                                  facecolor='white',
                                  edgecolor='white',
                                  linewidth=1))

    fig.tight_layout()
    ax.axis('off')
    # scatter the positions to fix the view
    ax.scatter([pos[node][0] for node in graph.nodes()],
               [pos[node][1] for node in graph.nodes()],
               s=0, color='red')

    return fig, ax


def _average_angles(angles: np.ndarray) -> float:
    """

    This function calculates the sum of a set of angles, taking into account their circular nature.
    The calculation is performed by converting each angle to its corresponding unit vector (using sine and cosine),
    summing the components, and then computing the angle of the resultant vector.

    Parameters
    ----------
    angles : np.ndarray
        Array of angles (in radians) for which to compute the average.

    Returns
    -------
    float
        The average angle (in radians), in the range (-pi, pi].

    """
    # Convert angles to unit vectors
    x_components = np.cos(angles)
    y_components = np.sin(angles)

    # Sum the components to get the resultant vector
    resultant_x = np.sum(x_components)
    resultant_y = np.sum(y_components)

    # Calculate the angle of the resultant vector
    resultant_angle = np.arctan2(resultant_y, resultant_x)

    return resultant_angle


def _plot_directed_network(nodes: List[str],
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
                           spacing_hyperbolic_factor: float = 0.4
                           ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate and save a circle network plot as a PNG file.

    This function creates a visualization of a directed network using a list of nodes,
    their positions, and an adjacency matrix defining the edges. The network is drawn
    on top of concentric circles, and the plot is saved to a file.

    Parameters
    ----------
    nodes : List[str]
        List of node names in the network.

    adjacency_matrix : np.ndarray
        Square adjacency matrix (shape: [n_nodes, n_nodes]) representing directed edges.
        If adjacency_matrix[i, j] != 0, there is a directed edge from node i to node j.

    x : np.ndarray
        1D array of x-coordinates for each node (same order as `nodes`).

    y : np.ndarray
        1D array of y-coordinates for each node (same order as `nodes`).

    max_ai : int
        Maximum assembly index (defines the number of concentric circles to draw).

    'labels' can be a boolean (draw/no draw) or a list of per-node labels (strings).
    If a list is provided, empty strings or None entries will not render text but
    nodes remain present.

    node_size : float
        Size of the nodes in the plot.

    arrow_size : float
        Size of the arrowheads for directed edges.

    node_color : str
        Color of the nodes.

    edge_color : str
        Color of the edges.

    fig_size : float
        Size of the figure (width and height in inches).

    filename : str
        Name of the output PNG file.

    `spacing_mode` ("linear" or "hyperbolic")
    `spacing_hyperbolic_factor` controls the strength of the sinh term when

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    ax : matplotlib.axes.Axes
        The matplotlib Axes object containing the plot.


    """
    if len(nodes) != len(adjacency_matrix) or len(adjacency_matrix) != len(x) or len(x) != len(y):
        raise ValueError("Lengths of nodes, adjacency_matrix, x, and y must be equal.")

    # build graph and positions
    graph = nx.DiGraph()
    positions = {nodes[i]: (float(x[i]), float(y[i])) for i in range(len(nodes))}
    graph.add_nodes_from(nodes)

    n = len(nodes)
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] != 0:
                graph.add_edge(nodes[i], nodes[j], weight=float(adjacency_matrix[i, j]))

    # create fig/ax if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # helper radius mapping consistent with plot_assembly_circle
    def _radius(idx: int) -> float:
        base = float(idx)
        if spacing_mode == "hyperbolic":
            mapped = float(base + spacing_hyperbolic_factor * np.sinh(base))
        else:
            mapped = float(base)
        return mapped

    # 1) draw curved directed edges first (below circles)
    for src, dst, data in graph.edges(data=True):
        x_src, y_src = positions[src]
        x_dst, y_dst = positions[dst]
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
        # try to set zorder on returned artist(s)
        try:
            if coll is None:
                continue
            # coll can be a LineCollection or list; handle common cases
            if hasattr(coll, "set_zorder"):
                coll.set_zorder(1)
            elif isinstance(coll, (list, tuple)):
                for c in coll:
                    try:
                        c.set_zorder(1)
                    except Exception:
                        pass
        except Exception:
            pass

    # 2) draw concentric circles above edges
    for idx in range(1, max_ai + 2):
        r = _radius(idx)
        circle = Circle((0, 0), r, color="black", alpha=1, fill=False, lw=1.5)
        circle.set_zorder(3)
        ax.add_artist(circle)

    # 3) draw nodes on top of circles using ax.scatter so zorder is applied safely
    node_xs = [positions[nn][0] for nn in nodes]
    node_ys = [positions[nn][1] for nn in nodes]
    # node_size in matplotlib scatter is in points^2; keep API-consistent
    scat = ax.scatter(node_xs, node_ys,
                      s=node_size,
                      c=node_color,
                      edgecolors=node_edge_color,
                      linewidths=node_linewidth,
                      zorder=4)
    # optional: draw node outlines if edgecolors requested and backend requires it
    # draw labels using ax.text for explicit zorder control
    font_size = max(8, int(node_size / 200))
    if isinstance(labels, (list, tuple, np.ndarray)):
        for i, node in enumerate(nodes):
            lab = labels[i] if i < len(labels) else ""
            if lab:
                ax.text(positions[node][0], positions[node][1], str(lab),
                        ha='center', va='center', fontsize=font_size, zorder=5)
    elif bool(labels):
        for node in nodes:
            ax.text(positions[node][0], positions[node][1], str(node),
                    ha='center', va='center', fontsize=font_size, zorder=5)

    # set limits based on mapped outermost radius
    max_radius = _radius(max_ai + 1)
    margin = max(1.5, 0.1 * max_radius)
    ax.set_xlim(-max_radius - margin, max_radius + margin)
    ax.set_ylim(-max_radius - margin, max_radius + margin)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    # Save if requested
    if filename is not None:
        if save_kwargs is None:
            save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        fig.savefig(filename, dpi=dpi, **save_kwargs)

    return fig, ax


def plot_assembly_circle(nodes: Sequence[Any],
                         adj_matrix: np.ndarray,
                         assembly_indices: Sequence[int],
                         labels: Optional[Union[bool, Sequence[str]]] = None,
                         node_size: float = 1000,
                         arrow_size: float = 80,
                         node_color: Union[str, Sequence[str]] = '#264f70',
                         node_edge_color: str = "black",
                         node_linewidth: float = 2.5,
                         edge_color: Union[str, Sequence[str]] = 'Grey',
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
                         spacing_hyperbolic_factor: float = 0.4
                         ):
    """
    Nodes are placed on concentric rings whose radius is proportional to their
    assembly index. Edges between nodes are rendered as curved directed arrows.
    Optional per-node labels, icons (molecule / atoms / graph), colormap/norm
    support and file saving are provided. The function delegates low-level
    drawing to an internal routine and may compute missing inputs (assembly
    indices or adjacency) if they are not supplied.

    Parameters
    ----------
    nodes : sequence
        Sequence of node identifiers (hashable). Order is used when `adj_matrix`
        or `assembly_indices` correspond by index.
    adj_matrix : array-like
        Square adjacency matrix (shape ``[n_nodes, n_nodes]``) indicating directed
        edges. Non-zero entries denote an edge from row index to column index.
    assembly_indices : array-like of int
        Assembly index for each node (lower values are closer to the center).
    labels : bool or sequence, optional
        If a boolean, ``True`` displays node identifiers as labels, ``False``
        hides labels. If a sequence, per-node label strings to render; empty or
        ``None`` entries suppress text for that node.
    node_size : float, optional
        Marker size for nodes. Default is ``1000``.
    arrow_size : float, optional
        Arrowhead size for directed edges. Default is ``80``.
    node_color : str or sequence, optional
        Color for nodes; may be a single color string or a sequence of colors
        (one per node). Default is ``'#264f70'``.
    node_edge_color : str, optional
        Color for node borders. Default is ``'black'``.
    node_linewidth : float, optional
        Line width for node borders. Default is ``2.5``.
    edge_color : str or sequence, optional
        Color for edges. Default is ``'Grey'``.
    arrow_alpha : float, optional
        Alpha/transparency for arrows (0.0 - 1.0). Default is ``1.0``.
    fig_size : float or tuple, optional
        Size of the figure in inches. If a single float is provided it is used
        for both width and height. Default is ``10``.
    filename : str or None, optional
        If provided, save the rendered figure to this path (PNG). Default is
        ``None`` (no file saved).
    dpi : int, optional
        Resolution in dots-per-inch when saving. Default is ``300``.
    fig : matplotlib.figure.Figure or None, optional
        Figure to draw onto. If ``None`` a new figure is created.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw onto. If ``None`` a new axes is created.
    cmap : matplotlib.colors.Colormap or str or None, optional
        Colormap to map per-node values if node colors are provided as numeric
        values. If provided together with ``norm``, a colorbar may be added.
    norm : matplotlib.colors.Normalize or None, optional
        Normalization instance used with ``cmap`` for color scaling.
    colorbar_label : str or None, optional
        Label for the colorbar when a colormap and norm are supplied.
    save_kwargs : dict or None, optional
        Extra keyword arguments forwarded to ``Figure.savefig`` when ``filename``
        is provided (e.g. ``bbox_inches``).
    spacing_mode: "linear" (radii = ai+1) or "hyperbolic" (radii = (ai+1) + spacing_hyperbolic_factor * sinh(ai+1))
    spacing_hyperbolic_factor : float. Multiplier for the sinh term in hyperbolic spacing (default 0.4).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib Figure that contains the rendered assembly circle plot.
    ax : matplotlib.axes.Axes
        Matplotlib Axes used for the drawing.

    Raises
    ------
    ValueError
        If provided arrays (e.g. ``nodes``, ``adj_matrix``, ``assembly_indices``)
        have inconsistent lengths or if ``adj_matrix`` is not square when supplied.
    TypeError
        If inputs cannot be interpreted as the expected types (e.g. numeric
        assembly indices or a 2D adjacency array).

    Notes
    -----
    - Node placement: nodes with the same assembly index share the same radius and
      are arranged by angle. Building-block nodes (minimum assembly index) are
      evenly spaced around the innermost circle; other nodes inherit an average
      angle from their parents via adjacency propagation.
    - Edges are rendered as curved arcs; curvature is chosen heuristically from
      relative node positions to improve readability.
    - When ``cmap`` and ``norm`` are provided the function adds a colorbar using
      the provided label and places it on the figure prior to optional saving.
    - The function may call internal helpers (e.g. a directed-network plotting
      routine) to perform low-level drawing; modify the returned ``ax`` after
      calling if custom axis limits or annotations are required.
    """

    n_nodes = len(nodes)

    angles = np.full(n_nodes, np.nan)
    max_ai = int(max(assembly_indices))
    min_ai = int(min(assembly_indices))

    # compute building blocks (minimum ai)
    n_building_blocks = sum(1 for ai in assembly_indices if ai == min_ai)
    # assign equispaced angles to building blocks
    idxs_bb = [i for i, ai in enumerate(assembly_indices) if ai == min_ai]
    for k, i in enumerate(idxs_bb):
        angles[i] = 2 * np.pi * (k / max(1, n_building_blocks))

    # assign angles to others by averaging parents' angles via adjacency (simple propagation)
    # iterative fill: for remaining nodes, average angles of neighbors that have angles
    adj = np.array(adj_matrix)
    # adjacency_matrix[i, j] != 0 means edge i -> j (i points to j)
    # parents of node j are nodes p with adj[p, j] != 0
    remaining_attempts = 0
    while np.any(np.isnan(angles)):
        changed = False
        for i in range(n_nodes):
            if np.isnan(angles[i]):
                parent_idxs = np.where(adj[:, i] != 0)[0]
                if parent_idxs.size > 0:
                    parent_angles = angles[parent_idxs]
                    known = parent_angles[~np.isnan(parent_angles)]
                    if known.size > 0:
                        # circular average
                        angles[i] = _average_angles(known)
                        changed = True
        if not changed:
            # Try averaging any known neighbor angles (parents OR children)
            for i in range(n_nodes):
                if np.isnan(angles[i]):
                    neighbor_idxs = np.where((adj[:, i] != 0) | (adj[i, :] != 0))[0]
                    if neighbor_idxs.size > 0:
                        neighbor_angles = angles[neighbor_idxs]
                        known = neighbor_angles[~np.isnan(neighbor_angles)]
                        if known.size > 0:
                            angles[i] = _average_angles(known)
                            changed = True
        if not changed:
            # final fallback: assign evenly spaced angles to remaining nodes
            remaining = np.where(np.isnan(angles))[0]
            nrem = len(remaining)
            if nrem == 0:
                break
            for k, idx in enumerate(remaining):
                angles[idx] = 2 * np.pi * (k / max(1, nrem))
            break
        remaining_attempts += 1
        if remaining_attempts > n_nodes + 5:
            # safety net to avoid infinite loops; fill any remaining randomly
            remaining = np.where(np.isnan(angles))[0]
            for idx in remaining:
                angles[idx] = 2 * np.pi * random.random()
            break

        # Resolve exact-angle overlaps: if multiple nodes share the same angle (within tol),
        # spread them slightly around that central angle so they don't plot exactly on top.
        tol = 1e-12
        finite_idxs = np.where(~np.isnan(angles))[0]
        processed = np.zeros(n_nodes, dtype=bool)
        for idx in finite_idxs:
            if processed[idx]:
                continue
            # find indices with angles close on the circle
            diffs = np.abs((angles[finite_idxs] - angles[idx] + np.pi) % (2 * np.pi) - np.pi)
            same_mask = diffs <= tol
            same_idxs = finite_idxs[same_mask]
            if same_idxs.size > 1:
                # compute circular center
                center = _average_angles(angles[same_idxs])
                # small angular spread proportional to count
                spread = min(0.08, 0.03 * same_idxs.size)
                offsets = np.linspace(-spread, spread, same_idxs.size)
                # sort by current radius (index) for deterministic ordering
                for k, j in enumerate(sorted(same_idxs)):
                    angles[j] = (center + offsets[k]) % (2 * np.pi)
                    processed[j] = True
            else:
                processed[idx] = True

    # compute radii according to spacing_mode
    ai_plus = np.array(assembly_indices, dtype=float) + 1.0
    if spacing_mode == "hyperbolic":
        # combined linear + mild sinh term so inner rings remain reasonable while outer rings expand
        radii = ai_plus + float(spacing_hyperbolic_factor) * np.sinh(ai_plus)
    else:
        radii = ai_plus

    x_positions = radii * np.cos(angles)
    y_positions = radii * np.sin(angles)

    # If fig/ax not provided, create them (no fallbacks beyond this)
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # call the underlying plotting routine drawing onto our fig/ax
    fig, ax = _plot_directed_network(
        nodes=nodes,
        adjacency_matrix=np.array(adj_matrix),
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
        sm.set_array(np.linspace(getattr(norm, "vmin", 0), getattr(norm, "vmax", 1), 256))
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.02)

        vmin = float(getattr(norm, "vmin", 0.0))
        vmax = float(getattr(norm, "vmax", 1.0))

        # Decide whether ticks should be integer-only: both endpoints near integers and range reasonable
        endpoints_integer_like = math.isclose(vmin, round(vmin), abs_tol=1e-8) and math.isclose(vmax, round(vmax),
                                                                                                abs_tol=1e-8)
        reasonable_range_for_integers = (vmax - vmin) <= 100
        integer_ticks = endpoints_integer_like and reasonable_range_for_integers

        # Use MaxNLocator to adaptively choose up to 10 ticks
        locator = mticker.MaxNLocator(nbins=10, integer=integer_ticks)
        cbar.locator = locator
        cbar.update_ticks()

        # Format ticks: use integer formatter when integer ticks requested
        if integer_ticks:
            cbar.formatter = mticker.FormatStrFormatter('%d')
            cbar.update_ticks()

        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=12)

    # finally save if requested (after colorbar added)
    if filename is not None:
        if save_kwargs is None:
            save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        fig.savefig(filename, dpi=dpi, **save_kwargs)

    return fig, ax


def scatter_plot(x: Union[np.ndarray, List],
                 y: Union[np.ndarray, List],
                 xlab: str = 'x',
                 ylab: str = 'y',
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
    # Convert to numpy arrays if they aren't already
    x = np.asarray(x)
    y = np.asarray(y)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, color='black', alpha=alpha, s=50)
    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    return fig, ax


def scatter_plot_with_colorbar(x: Union[np.ndarray, List],
                               y: Union[np.ndarray, List],
                               xlab: str = 'x',
                               ylab: str = 'y',
                               cmap: str = 'viridis',
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
    # Convert to numpy arrays if they aren't already
    x = np.asarray(x)
    y = np.asarray(y)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Stack the data and calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density so that high-density points are plotted last
    idx = z.argsort()
    x_sorted, y_sorted, z_sorted = x[idx], y[idx], z[idx]

    # Create the scatter plot with colour determined by point density
    scatter = ax.scatter(x_sorted,
                         y_sorted,
                         c=z_sorted,
                         cmap=cmap,
                         s=50,
                         alpha=0.8)

    # Configure the plot
    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    return fig, ax


def plot_contourf_full(x: Union[np.ndarray, List],
                       y: Union[np.ndarray, List],
                       xlab: str,
                       ylab: str,
                       c_map: str = "Purples",
                       figsize: Tuple[float, float] = (8, 5),
                       fontsize: int = 16) -> Tuple[Figure, Axes]:
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
    fig : matplotlib.figure.FigureMatplotlib Figure object containing the contour plot.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object containing the contour plot.
    """
    # Convert to numpy arrays if they aren't already
    x = np.asarray(x)
    y = np.asarray(y)

    fig, ax = plt.subplots(figsize=figsize)
    lims = [min(x), max(x)]

    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[lims[0]:lims[1]:x.size ** 0.6 * 1j, lims[0]:lims[1]:y.size ** 0.6 * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    ax.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.9, cmap=c_map)

    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    return fig, ax


def plot_heatmap(x: Union[np.ndarray, List],
                 y: Union[np.ndarray, List],
                 xlab: str,
                 ylab: str,
                 c_map: str = 'viridis',
                 nbins: int | Tuple[int, int] = 50,
                 figsize: Tuple[float, float] = (8, 5),
                 fontsize: int = 16) -> Tuple[Figure, Axes]:
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
    nbins : int or tuple, optional
        Number of bins to use for the 2D histogram. If an int, the same number of
        bins is applied to both axes. If a tuple ``(nx, ny)``, uses ``nx`` and
        ``ny`` bins for x and y respectively. Default is ``50``.
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
    # Convert to numpy arrays if they aren't already
    x = np.asarray(x)
    y = np.asarray(y)

    fig, ax = plt.subplots(figsize=figsize)
    # Create a 2D histogram of the data
    heatmap_data, xedges, yedges = np.histogram2d(x, y, bins=nbins)
    im = ax.imshow(heatmap_data.T,
                   origin='lower',
                   cmap=c_map,
                   aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    # Add colour bar
    cbar = plt.colorbar(im, ax=ax)
    cbar.locator = mticker.MaxNLocator(integer=True)
    cbar.update_ticks()
    cbar.set_label('Count', fontsize=fontsize)
    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    return fig, ax


def scatter_plot_3d_with_colorbar(x: Union[np.ndarray, List],
                                  y: Union[np.ndarray, List],
                                  z: Union[np.ndarray, List],
                                  c: Optional[Union[np.ndarray, List]] = None,
                                  xlab: str = 'x',
                                  ylab: str = 'y',
                                  zlab: str = 'z',
                                  cmap: str = 'viridis',
                                  figsize: Tuple[float, float] = (10, 8),
                                  fontsize: int = 20,
                                  alpha: float = 0.8,
                                  s: Union[float, np.ndarray] = 50,
                                  labelpad: float = 20) -> Tuple[Figure, Axes]:
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
    ax : matplotlib.axes._subplots.Axes3DSubplot
        Matplotlib 3D Axes containing the scatter plot.
    """
    # Create a figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Convert to numpy arrays if they aren't already
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    if c is not None:
        c = np.asarray(c)

    # If no color values provided, calculate point density
    if c is None:
        # Stack the data and calculate the point density
        xyz = np.vstack([x, y, z])
        c = gaussian_kde(xyz)(xyz)

        # Sort the points by density so that high-density points are plotted last
        idx = c.argsort()
        x, y, z, c = x[idx], y[idx], z[idx], c[idx]

    # Create the 3D scatter plot
    scatter = ax.scatter(x, y, z, c=c, cmap=cmap, s=s, alpha=alpha)

    # Add color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Point Density', fontsize=fontsize - 4)

    # Set labels
    ax.set_xlabel(xlab, fontsize=fontsize, labelpad=labelpad)
    ax.set_ylabel(ylab, fontsize=fontsize, labelpad=labelpad)
    ax.set_zlabel(zlab, fontsize=fontsize, labelpad=labelpad)

    # Set tick font sizes
    ax.tick_params(axis='x', labelsize=fontsize - 4)
    ax.tick_params(axis='y', labelsize=fontsize - 4)
    ax.tick_params(axis='z', labelsize=fontsize - 4)

    # Set line width for axes
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        try:
            axis.line.set_linewidth(2.0)
        except:
            pass
    fig.tight_layout()
    return fig, ax


def plot_hexbin_scatter(x: Union[np.ndarray, List],
                        y: Union[np.ndarray, List],
                        xlab: str = 'x',
                        ylab: str = 'y',
                        guide_line: bool = True,
                        cmap: str = 'viridis',
                        figsize: Tuple[float, float] = (8, 5),
                        fontsize: int = 16,
                        bins_scale: Optional[str] = None) -> Tuple[Figure, Axes]:
    """
    Create a hexbin scatter plot with optional y=x guideline and colorbar.

    Generates a hexagonal-binned 2D density plot using Matplotlib's ``hexbin`` to
    visualize the joint distribution of ``x`` and ``y``. Provides configurable
    colormap, bin scaling (e.g. ``'log'``), figure sizing and font sizing, and an
    optional red dashed y=x guideline.

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
        If ``True``, draw a reference line for ``y = x`` (red dashed). Default is ``True``.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap used for the hexbin plot. Default is ``'viridis'``.
    figsize : tuple of float, optional
        Figure size in inches as ``(width, height)``. Default is ``(8, 5)``.
    fontsize : int, optional
        Font size used for axis labels and colorbar label. Default is ``16``.
    bins_scale : {str, None}, optional
        Bin scaling mode passed to Matplotlib's ``hexbin`` ``bins`` parameter.
        Common value: ``'log'`` for logarithmic binning; if ``None`` (default) uses linear counts.
    gridsize : int or tuple, optional
        The number of hexagons in the x-direction (int) or a (nx, ny) tuple.
        Controls hexagon resolution. Default is ``30``.

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
      ``'counts'`` by default; when ``bins_scale == 'log'``, zero-count hexagons are
      not shown on a log scale.
    - Setting ``gridsize`` larger increases spatial resolution but may increase plotting time.
    - The optional guideline is drawn across the data range and helps to visually
      assess deviations from the identity relationship.
    """
    # Convert to numpy arrays if they aren't already
    x = np.asarray(x)
    y = np.asarray(y)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Determine x and y-axis limits
    xlim = x.min(), x.max()
    ylim = y.min(), y.max()

    # Create the hexbin plot
    hb = ax.hexbin(x, y, gridsize=30, cmap=cmap, bins=bins_scale)
    ax.set(xlim=xlim, ylim=ylim)

    # Add a colorbar to the plot
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label('counts', fontsize=fontsize)

    if guide_line:
        # Plot the y = x guide line
        x_line = np.linspace(min(x), max(x), 2)
        ax.plot(x_line, x_line, color='red', linestyle='--', linewidth=2)

    # Configure the plot with custom labels and layout
    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)

    return fig, ax


def plot_histogram(data: Union[np.ndarray, List],
                   bins: Union[int, Sequence[float]] = 30,
                   xlab: str = 'Values',
                   ylab: str = 'Frequency',
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
      helper routines and calls ``plt.hist`` for the rendering.
    - When ``bins`` is provided as an integer, Matplotlib's default binning rules
      are used; pass explicit bin edges to control bin placement precisely.
    - For publication-quality figures, modify ``figsize`` and ``fontsize`` and
      save the returned ``fig`` with appropriate ``dpi`` and ``bbox_inches`` settings.
    """
    # Convert to numpy array if it isn't already
    data = np.asarray(data)

    fig, ax = plt.subplots(figsize=figsize)
    plt.hist(data,
             bins=bins,
             color='blue',
             edgecolor='black',
             alpha=0.8)
    ax_plot(fig, ax,
            xlab=xlab,
            ylab=ylab,
            xs=fontsize,
            ys=fontsize)
    return fig, ax


def plot_histogram_all_x(data: Union[np.ndarray, List],
                         xlab: str = 'Number of Bonds',
                         ylab: str = 'Frequency',
                         figsize: Tuple[float, float] = (8, 5),
                         fontsize: int = 16) -> Tuple[Figure, Axes]:
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
    # Convert to numpy array if it isn't already
    data = np.asarray(data)

    bins = range(int(data.min()), int(data.max()) + 2)
    fig, ax = plot_histogram(data,
                             bins=bins,
                             xlab=xlab,
                             ylab=ylab,
                             figsize=figsize,
                             fontsize=fontsize)
    return fig, ax


def plot_histogram_compare(data1: Union[np.ndarray, List],
                           data2: Union[np.ndarray, List],
                           labels: Sequence[str],
                           bins: Union[int, Sequence[float]] = 30,
                           xlab: str = 'Values',
                           ylab: str = 'Frequency',
                           y_scale: Optional[str] = 'log',
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
    # Convert to numpy arrays if they aren't already
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    fig, ax = plt.subplots(figsize=figsize)
    plt.hist(data1, bins=bins, alpha=0.8, label=labels[0])
    plt.hist(data2, bins=bins, alpha=0.8, label=labels[1])
    plt.legend()

    if y_scale is not None:
        ax.set_yscale(y_scale)

    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    return fig, ax


def plot_kde(data: Union[np.ndarray, List],
             bandwidth: Optional[float] = None,
             grid_size: int = 1000,
             y_scale: Optional[str] = 'log',
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
    n = len(data)

    # Fit KDE
    kde = gaussian_kde(data, bw_method=bandwidth)

    # Evaluate KDE on a grid
    x_min, x_max = data.min(), data.max()
    xs = np.linspace(x_min, x_max, grid_size)
    ys = kde(xs)

    # Convert density to expected counts
    dx = xs[1] - xs[0]
    counts = ys * n * dx

    ax.set_xlim(x_min, x_max)
    if y_scale is not None:
        ax.set_yscale(y_scale)

    # Overlay KDE scaled to counts
    ax.plot(xs, counts, color='red', lw=2)
    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)

    return fig, ax


def multipartite_layout_crossmin(G: nx.Graph,
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
    Dict[Any, Tuple[float, float]], Tuple[Dict[Any, Tuple[float, float]], Dict[Any, List[Any]]]]:
    """
    Compute a multipartite layout for a graph minimizing inter-layer edge crossings.

    This function computes positions for nodes arranged in discrete layers (a
    multipartite layout) and attempts to reduce edge crossings between adjacent
    layers using iterative heuristics (barycenter or median). The algorithm first
    constructs per-layer node lists from a node attribute (``subset_key``), applies
    an initial stable ordering, then performs a number of top-down and bottom-up
    sweeps to refine ordering. The returned positions follow the chosen
    ``align`` convention with configurable spacing and scaling. Optionally the final
    per-layer node order can be returned.

    Parameters
    ----------
    G : networkx.Graph
        Input graph. Nodes must supply the layer membership via the node attribute
        named by ``subset_key`` (an integer or sortable value). If missing, nodes
        default to layer 0.
    subset_key : str, optional
        Node attribute key used to determine layer membership. Default is ``"subset"``.
    align : {'vertical', 'horizontal'}, optional
        Whether layers are arranged vertically (columns) or horizontally (rows).
        Default is ``"vertical"``.
    method : {'barycenter', 'median'}, optional
        Heuristic used to compute the preferred order of nodes in a layer from the
        neighbour positions in adjacent layers. ``'barycenter'`` uses a (weighted)
        average neighbour index, ``'median'`` uses the median neighbour index.
        Default is ``'barycenter'``.
    iterations : int, optional
        Number of top-down and bottom-up sweeps to refine node ordering. Must be
        >= 1. Larger values increase runtime and may reduce crossings. Default is 100.
    layer_spacing : float, optional
        Distance between consecutive layers on the fixed axis. Must be > 0.
        Default is 1.0.
    node_spacing : float, optional
        Spacing between adjacent nodes within a layer on the free axis. Must be > 0.
        Default is 1.0.
    scale : float, optional
        Global scaling factor applied to final coordinates. Default is 1.0.
    seed : int or None, optional
        Random seed for deterministic tie-breaking. If None, behaviour may be
        non-deterministic. Default is None.
    weight : str or None, optional
        Edge attribute name to use as a numeric weight when computing barycenters.
        If None, unit weights are assumed. Default is None.
    return_order : bool, optional
        If True, also return the per-layer node ordering (dictionary mapping layer
        value -> ordered list of nodes). Default is False.

    Returns
    -------
    pos : dict
        Mapping node -> (x, y) float coordinates. Coordinates follow the chosen
        ``align`` convention (layers along the fixed axis).
    orders : dict, optional
        When ``return_order`` is True, a dictionary mapping each layer value to the
        final ordered list of nodes is returned as the second element of a tuple:
        ``(pos, orders)``.

    Raises
    ------
    ValueError
        If parameters are invalid (for example nonpositive spacings, nonpositive
        iterations, or unsupported ``method`` or ``align`` values).
    TypeError
        If node layer attributes are of an unexpected type that cannot be sorted,
        or if specified edge weights are non-numeric.

    Notes
    -----
    - The algorithm is primarily heuristic and aims to reduce pairwise crossings
      between adjacent layers; it does not guarantee a global minimum.
    - Initial layer orders are chosen deterministically (degree/name) to improve
      reproducibility; tie-breaking uses ``seed`` when provided.
    - For performance, barycenter computations may accumulate neighbour indices
      weighted by edge attribute (if ``weight`` provided) and stable tie-breaking
      is recommended for deterministic output.
    - The produced positions use integer layer coordinates multiplied by
      ``layer_spacing`` on the fixed axis and evenly spaced node positions within
      each layer (centered) multiplied by ``node_spacing`` and ``scale``.
    """
    if seed is not None:
        random.seed(seed)

    # Build layers based on the subset_key attribute
    layers = defaultdict(list)
    for n in G.nodes():
        layer = G.nodes[n].get(subset_key, 0)
        layers[layer].append(n)

    layer_keys = sorted(layers.keys())

    # Initial order within each layer (sorted by degree and name for stability)
    for k in layer_keys:
        nodes = layers[k]
        nodes.sort(key=lambda n: (G.degree(n), str(n)))

    # Helper function to create a mapping of node indices within a layer
    def index_map(nodes):
        return {u: i for i, u in enumerate(nodes)}

    # Helper function to calculate edge weights
    def edge_w(u, v):
        if weight is None:
            return 1.0
        return G[u][v].get(weight, 1.0)

    # Order nodes in a layer based on the barycenter heuristic
    def ordered_by_barycenter(target_nodes, neighbor_nodes):
        neigh_idx = index_map(neighbor_nodes)
        current_idx = index_map(target_nodes)

        scores = []
        for u in target_nodes:
            s = 0.0
            wsum = 0.0
            for v in G[u]:
                if v in neigh_idx:
                    w = edge_w(u, v)
                    s += w * neigh_idx[v]
                    wsum += w
            if wsum > 0:
                score = s / wsum
            else:
                score = current_idx[u]  # Keep original position if no neighbors
            scores.append((score, current_idx[u], u))  # Tie-break by old index

        scores.sort()
        return [u for _, __, u in scores]

    # Order nodes in a layer based on the median heuristic
    def ordered_by_median(target_nodes, neighbor_nodes):
        neigh_idx = index_map(neighbor_nodes)
        current_idx = index_map(target_nodes)

        scores = []
        for u in target_nodes:
            seq = [neigh_idx[v] for v in G[u] if v in neigh_idx]
            if seq:
                seq.sort()
                m = seq[len(seq) // 2] if len(seq) % 2 == 1 else 0.5 * (seq[len(seq) // 2 - 1] + seq[len(seq) // 2])
                score = m
            else:
                score = current_idx[u]
            scores.append((score, current_idx[u], u))

        scores.sort()
        return [u for _, __, u in scores]

    # Select the ordering method
    orderer = ordered_by_barycenter if method == "barycenter" else ordered_by_median

    # Perform top-down and bottom-up sweeps to refine node order
    for _ in range(max(1, int(iterations))):
        # Top-down sweep
        for i in range(1, len(layer_keys)):
            k_prev, k_cur = layer_keys[i - 1], layer_keys[i]
            layers[k_cur] = orderer(layers[k_cur], layers[k_prev])

        # Bottom-up sweep
        for i in range(len(layer_keys) - 2, -1, -1):
            k_next, k_cur = layer_keys[i + 1], layer_keys[i]
            layers[k_cur] = orderer(layers[k_cur], layers[k_next])

    # Assign coordinates to nodes
    coords_by_layer = {}
    for idx, k in enumerate(layer_keys):
        nodes = layers[k]
        n = len(nodes)
        if n == 0:
            coords_by_layer[k] = {}
            continue
        # Calculate positions along the free axis
        start = -0.5 * (n - 1) * node_spacing
        free_axis_positions = {nodes[i]: start + i * node_spacing for i in range(n)}

        fixed = idx * layer_spacing  # Fixed axis value for the layer
        if align == "vertical":
            coords_by_layer[k] = {u: (fixed, free_axis_positions[u]) for u in nodes}
        else:
            coords_by_layer[k] = {u: (free_axis_positions[u], fixed) for u in nodes}

    # Combine coordinates from all layers
    pos = {}
    for k in layer_keys:
        pos.update(coords_by_layer[k])

    # Scale the layout if a scaling factor is provided
    if scale != 1.0:
        pos = {u: (scale * x, scale * y) for u, (x, y) in pos.items()}

    if return_order:
        # Return positions and the final node order per layer
        return pos, {k: list(layers[k]) for k in layer_keys}
    return pos


def multipartite_layout_crossmin_long(G: nx.Graph,
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
    """
    Compute a multipartite layout minimizing crossings with optional dummy-node routing for long edges.

    This function computes positions for nodes arranged in discrete layers (multipartite layout)
    and attempts to minimise edge crossings using iterative barycenter/median heuristics.
    Edges that span more than one intermediate layer may be replaced by synthetic dummy nodes
    to allow polyline routing and improved crossing reduction. The function is deterministic
    when a seed is provided and can optionally return auxiliary information such as per-layer
    orders, inserted dummy node identifiers and routing descriptors for long edges.

    Parameters
    ----------
    G : networkx.Graph
        Input graph. Nodes must expose a layer attribute indicated by ``subset_key`` (or the
        attribute is inferred/treated uniformly). Graph may be directed or undirected.
    subset_key : str, optional
        Node attribute key used to determine layer membership. Defaults to ``"subset"``.
    align : {'vertical', 'horizontal'}, optional
        Whether layers are arranged vertically (columns) or horizontally (rows).
        Defaults to ``"vertical"``.
    method : {'barycenter', 'median'}, optional
        Heuristic used to compute orderings within layers. ``'barycenter'`` computes a
        weighted average position of neighbours; ``'median'`` uses the median neighbour
        index. Defaults to ``'barycenter'``.
    iterations : int, optional
        Number of top-down and bottom-up sweeps to refine node ordering. Must be >= 1.
        Larger values increase runtime and may reduce crossings. Defaults to 100.
    layer_spacing : float, optional
        Spacing between consecutive layers on the fixed axis. Must be > 0. Defaults to 1.0.
    node_spacing : float, optional
        Spacing between adjacent nodes within a layer on the free axis. Must be > 0.
        Defaults to 1.0.
    scale : float, optional
        Global scaling factor applied to final coordinates. Defaults to 1.0.
    seed : int or None, optional
        Random seed for reproducible tie-breaking and stochastic choices. Defaults to None.
    weight : str or None, optional
        Edge attribute name used as a numeric weight for ordering and crossing computations.
        If None, unit weights are assumed. Defaults to None.
    insert_dummies : bool, optional
        If True, insert synthetic dummy nodes for edges spanning multiple intermediate layers
        so those edges are represented as sequences of unit hops. Defaults to True.
    dummy_prefix : str, optional
        Prefix used for synthetic dummy node identifiers when ``insert_dummies`` is True.
        Defaults to ``"__dummy__"``.
    return_order : bool, optional
        If True, return a dictionary mapping each layer value to the final ordered list
        of nodes (excluding or including dummies according to ``return_dummies``). Defaults to False.
    return_dummies : bool, optional
        If True, include dummy node identifiers and their positions in the returned layout.
        Defaults to False.
    return_routes : bool, optional
        If True, return routing descriptors for edges that were split into dummy chains.
        Each routing descriptor describes the node chain or intermediate coordinates for
        drawing a polyline representing the original edge. Defaults to False.

    Returns
    -------
    pos : dict
        Mapping node -> (x, y) coordinate positions (float). Coordinates follow the chosen
        ``align`` convention (layers along the fixed axis). By default dummy nodes are not
        included unless ``return_dummies`` is True.
    tuple, optional
        When one or more of ``return_order``, ``return_dummies`` or ``return_routes`` is True,
        a tuple is returned with the primary ``pos`` as the first element followed by the
        requested auxiliary structures in this order: ``orders``, ``dummies``, ``routes``.
        - orders : dict mapping layer value -> list of nodes in final order (may include dummies
          if ``return_dummies`` is True).
        - dummies : set or list of dummy node identifiers and their positions (present when
          ``return_dummies`` is True).
        - routes : list of routing descriptors for edges spanning layers (present when
          ``return_routes`` is True).

    Raises
    ------
    ValueError
        If parameters are invalid (for example nonpositive spacings, nonpositive iterations,
        or inconsistent layer assignments).
    TypeError
        If node layer attributes are of an unexpected type or edge weights (when specified)
        are non-numeric.

    Notes
    -----
    - The algorithm first constructs per-layer node lists from ``subset_key`` and applies
      repeated top-down and bottom-up sweeps using the chosen heuristic to reduce crossings.
    - Dummy nodes are purely layout artefacts used to represent long edges as chains of
      single-layer hops; they can be filtered from outputs unless explicitly requested.
    - For performance, weighted barycenter computations accumulate neighbour indices weighted
      by the configured edge attribute (if ``weight`` is provided) and use stable tie-breaking.
    - When ``insert_dummies`` is True, the function preserves a mapping from original edges
      to their dummy-node chains so that routed polylines can be reconstructed for drawing.
    - The algorithm is stochastic only in tie-breaking and optional proposals; provide
      ``seed`` for deterministic behaviour.
    """
    if seed is not None:
        random.seed(seed)

    # Collect layers and normalize to contiguous indices
    # Accept mixed types by sorting with str if necessary.
    unique_layers = set()
    node_layer_val = {}
    for n in G.nodes():
        layer_val = G.nodes[n].get(subset_key, 0)
        node_layer_val[n] = layer_val
        unique_layers.add(layer_val)
    try:
        layer_keys = sorted(unique_layers)
    except TypeError:
        layer_keys = sorted(unique_layers, key=str)

    L = len(layer_keys)
    layer_to_idx = {lv: i for i, lv in enumerate(layer_keys)}
    node_layer_idx = {n: layer_to_idx[node_layer_val[n]] for n in G.nodes()}

    # Prepare per-layer buckets (will also hold dummies if we insert them)
    layers_list = [[] for _ in range(L)]
    for n in G.nodes():
        layers_list[node_layer_idx[n]].append(n)

    # Stable initial order: degree then name (helps determinism)
    for i in range(L):
        layers_list[i].sort(key=lambda n: (G.degree(n), str(n)))

    # Build an augmented (adjacent-only) structure
    # We'll keep a light-weight adjacency dict + a list of edges for routing.
    neighbors = defaultdict(lambda: defaultdict(float))  # neighbors[u][v] = weight_sum
    # For routing back to original edges
    routes = []
    dummy_id_counter = 0

    def edge_w(u, v, data=None):
        """
        Calculates the weight of an edge.

        Parameters:
        u (node): Source node.
        v (node): Target node.
        data (dict, optional): Edge data dictionary. Defaults to None.

        Returns:
        float: Weight of the edge.
        """
        if weight is None:
            return 1.0
        if data is not None and weight in data:
            return data[weight]
        # For simple Graphs without data passed in:
        try:
            return G[u][v].get(weight, 1.0)
        except Exception:
            return 1.0

    # Helper to add an (undirected) edge into neighbors with weight accumulation
    def add_edge(u, v, w):
        """
        Adds an undirected edge to the neighbors dictionary, accumulating weights.

        Parameters:
        u (node): Source node.
        v (node): Target node.
        w (float): Weight of the edge.
        """
        neighbors[u][v] += w
        neighbors[v][u] += w

    # Ensure all nodes exist in neighbors (even isolates)
    for n in G.nodes():
        _ = neighbors[n]  # touch

    if insert_dummies:
        # Split every edge that spans more than one layer
        for (u, v, *rest) in G.edges(data=True):
            data = rest[0] if rest else {}
            w = edge_w(u, v, data)
            lu, lv = node_layer_idx[u], node_layer_idx[v]
            if lu == lv:
                # Same-layer edge: keep it as-is (won't affect sweeps)
                add_edge(u, v, w)
                routes.append({"endpoints": (u, v), "nodes": [u, v]})
                continue
            # Orient so lu < lv
            rev = False
            if lu > lv:
                u, v = v, u
                lu, lv = lv, lu
                rev = True

            if lv - lu == 1:
                add_edge(u, v, w)
                routes.append({"endpoints": (u, v) if not rev else (v, u), "nodes": [u, v]})
                continue

            # Need to insert dummies across intermediate layers lu+1 .. lv-1
            chain = [u]
            prev = u
            for k in range(lu + 1, lv):
                dummy_id_counter += 1
                d = f"{dummy_prefix}{dummy_id_counter}"
                # remember its layer
                node_layer_idx[d] = k
                # mark it's a dummy (attribute for reference)
                # (we don't modify the original G; this is layout-only)
                layers_list[k].append(d)
                # touch neighbors so it exists
                _ = neighbors[d]
                # connect prev -> d
                add_edge(prev, d, w)
                chain.append(d)
                prev = d
            # connect last dummy -> v
            add_edge(prev, v, w)
            chain.append(v)
            if rev:
                chain = list(reversed(chain))
                endpoints = (chain[0], chain[-1])  # original orientation (u,v) as in G
            else:
                endpoints = (chain[0], chain[-1])
            routes.append({"endpoints": endpoints, "nodes": chain})
    else:
        # No dummies: just accumulate all edges
        for (u, v, *rest) in G.edges(data=True, keys=False):
            data = rest[0] if rest else {}
            w = edge_w(u, v, data)
            add_edge(u, v, w)
            routes.append({"endpoints": (u, v), "nodes": [u, v]})

    # Ordering heuristics on adjacent layers only
    def index_map(nodes):
        """
        Creates a mapping of node indices within a layer.

        Parameters:
        nodes (list): List of nodes in the layer.

        Returns:
        dict: Mapping of nodes to their indices.
        """
        return {u: i for i, u in enumerate(nodes)}

    def ordered_by_barycenter(target_nodes, neighbor_nodes):
        """
        Orders nodes in a layer based on the barycenter heuristic.

        Parameters:
        target_nodes (list): Nodes in the target layer.
        neighbor_nodes (list): Nodes in the neighboring layer.

        Returns:
        list: Ordered list of nodes in the target layer.
        """
        neigh_idx = index_map(neighbor_nodes)
        cur_idx = index_map(target_nodes)
        scores = []
        for u in target_nodes:
            s = 0.0
            wsum = 0.0
            for v, w in neighbors[u].items():
                if v in neigh_idx:
                    s += w * neigh_idx[v]
                    wsum += w
            score = (s / wsum) if wsum > 0 else cur_idx[u]
            scores.append((score, cur_idx[u], u))
        scores.sort()
        return [u for _, __, u in scores]

    def ordered_by_median(target_nodes, neighbor_nodes):
        """
        Orders nodes in a layer based on the median heuristic.

        Parameters:
        target_nodes (list): Nodes in the target layer.
        neighbor_nodes (list): Nodes in the neighboring layer.

        Returns:
        list: Ordered list of nodes in the target layer.
        """
        neigh_idx = index_map(neighbor_nodes)
        cur_idx = index_map(target_nodes)
        scores = []
        for u in target_nodes:
            seq = []
            for v, w in neighbors[u].items():
                if v in neigh_idx:
                    # push neighbor index 'w' times (weighted median)
                    repeats = int(round(w)) if w != 1.0 else 1
                    if repeats <= 1:
                        seq.append(neigh_idx[v])
                    else:
                        seq.extend([neigh_idx[v]] * repeats)
            if seq:
                seq.sort()
                m = seq[len(seq) // 2] if len(seq) % 2 else 0.5 * (seq[len(seq) // 2 - 1] + seq[len(seq) // 2])
                score = m
            else:
                score = cur_idx[u]
            scores.append((score, cur_idx[u], u))
        scores.sort()
        return [u for _, __, u in scores]

    orderer = ordered_by_barycenter if method == "barycenter" else ordered_by_median

    iterations = max(1, int(iterations))
    for _ in range(iterations):
        # top-down (left->right): order each layer by the previous layer
        for i in range(1, L):
            layers_list[i] = orderer(layers_list[i], layers_list[i - 1])
        # bottom-up (right->left): order by the next layer
        for i in range(L - 2, -1, -1):
            layers_list[i] = orderer(layers_list[i], layers_list[i + 1])

    # Coordinates
    # Place nodes in each layer centered around 0 on the free axis
    pos_all = {}
    for i in range(L):
        nodes = layers_list[i]
        n = len(nodes)
        if n == 0:
            continue
        start = -0.5 * (n - 1) * node_spacing
        for j, u in enumerate(nodes):
            free = start + j * node_spacing
            fixed = i * layer_spacing
            if align == "vertical":
                pos_all[u] = (fixed, free)  # columns
            else:
                pos_all[u] = (free, fixed)  # rows

    if scale != 1.0:
        pos_all = {u: (scale * x, scale * y) for u, (x, y) in pos_all.items()}

    # Filter dummy nodes unless requested
    is_dummy = lambda n: isinstance(n, str) and n.startswith(dummy_prefix)
    if return_dummies:
        pos = dict(pos_all)
    else:
        pos = {u: xy for u, xy in pos_all.items() if not is_dummy(u)}

    # Build per-layer orders keyed by original layer values (for reference)
    if return_order:
        order_by_layer = {layer_keys[i]: list(layers_list[i]) for i in range(L)}

    # If routes requested, translate chain nodes to polylines of points
    if return_routes:
        routed = []
        for item in routes:
            chain = item["nodes"]
            pts = [pos_all[n] for n in chain if n in pos_all]
            routed.append({
                "endpoints": item["endpoints"],
                "nodes": list(chain),
                "points": pts,
            })

    # Final return(s)
    if return_order and return_routes:
        return pos, order_by_layer, routed
    if return_order:
        return pos, order_by_layer
    if return_routes:
        return pos, routed
    return pos


class _BIT:
    # Fenwick tree for prefix sums of floats (weights).

    def __init__(self, n: int) -> None:
        """
        Initializes the Fenwick tree.

        Parameters:
        n (int): The size of the tree (number of elements).
        """
        self.n = n
        self.t = [0.0] * (n + 1)

    def add(self, i: int, delta: float) -> None:
        """
        Adds a value to the element at index `i`.

        Parameters:
        i (int): The index (0-based) to which the value will be added.
        delta (float): The value to add.
        """
        i += 1
        while i <= self.n:
            self.t[i] += delta
            i += i & -i

    def sum_prefix(self, i: int) -> float:
        """
        Computes the prefix sum from index 0 to `i` (inclusive).

        Parameters:
        i (int): The index (0-based) up to which the prefix sum is calculated.

        Returns:
        float: The sum of elements from index 0 to `i`. Returns 0.0 if `i` is negative.
        """
        if i < 0:
            return 0.0
        s = 0.0
        i += 1
        while i > 0:
            s += self.t[i]
            i -= i & -i
        return s


def _pair_crossings_weighted(order_left: List[Any],
                             order_right: List[Any],
                             edges: List[Tuple[Any, Any, float]]) -> float:
    """
    Count weighted crossings between two adjacent layers given their node orders.

    Computes the total weighted crossing value for a bipartite set of edges
    connecting nodes in a left layer to nodes in a right layer. Each edge
    contributes proportionally to its weight and crossings are counted as the
    product of weights of two edges that geometrically cross given the layer
    orderings.

    Parameters
    ----------
    order_left : list
        Ordered list of nodes in the left layer. Elements should be hashable and
        unique within this list.
    order_right : list
        Ordered list of nodes in the right layer. Elements should be hashable and
        unique within this list.
    edges : list of tuple
        Iterable of edges connecting the two layers. Each item should be a
        tuple ``(u, v, w)`` where ``u`` is a node in ``order_left``, ``v`` is a
        node in ``order_right`` and ``w`` is a numeric edge weight (float or int).
        Edges referencing nodes not present in the corresponding order are
        ignored or may trigger an error depending on the caller's expectations.

    Returns
    -------
    float
        Total weighted crossing measure. For every pair of edges ``(u1, v1, w1)``
        and ``(u2, v2, w2)`` that cross given the two orders (i.e. left positions
        satisfy pos_left(u1) < pos_left(u2) but pos_right(v1) > pos_right(v2)),
        the contribution ``w1 * w2`` is added. The returned value is the sum
        of these contributions.

    Raises
    ------
    TypeError
        If inputs are of incorrect types (for example ``order_left``/``order_right``
        not indexable or ``edges`` not iterable of triples).
    ValueError
        If an edge references a node not found in the provided orders and the
        implementation chooses to treat that as an error.

    Notes
    -----
    - A common efficient implementation sorts edges by left-layer index and
      counts inversions on the right-layer indices using a Fenwick tree
      (binary indexed tree) or similar prefix-sum structure to achieve
      near-linearithmic runtime.
    - The function returns a floating point sum to accommodate non-integer
      edge weights and large accumulation.
    """
    # Map nodes in the left layer to their positions
    posL = {u: i for i, u in enumerate(order_left)}
    # Map nodes in the right layer to their positions
    posR = {v: i for i, v in enumerate(order_right)}

    # Create a list of triples (position in left, position in right, weight) for valid edges
    triples = [(posL[u], posR[v], float(w)) for (u, v, w) in edges
               if u in posL and v in posR]

    # If there are no valid triples, return 0.0 as there are no crossings
    if not triples:
        return 0.0

    # Sort triples by the left position, then by the right position
    triples.sort(key=lambda t: (t[0], t[1]))

    # Initialize a Fenwick tree for the right layer
    bit = _BIT(len(order_right))
    inv = 0.0  # Total weighted crossings
    total = 0.0  # Total weight processed so far

    # Iterate through the sorted triples
    for _, j, w in triples:
        # Calculate the weighted crossings for edges with positions greater than j
        inv += w * (total - bit.sum_prefix(j))
        # Add the current weight to the Fenwick tree
        bit.add(j, w)
        # Update the total weight
        total += w

    return inv


def multipartite_layout_sa(G: nx.Graph,
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
    """
    Compute a multipartite layout for a graph using simulated annealing to minimize edge crossings.

    This function produces coordinates for nodes arranged in discrete layers (multipartite layout)
    while attempting to minimise edge crossings via a simulated-annealing-based reordering
    procedure. Optionally inserts dummy nodes for edges that span multiple layers, supports
    weighted edges for ordering, and can return auxiliary information such as per-layer
    orders, dummy-node inclusion and routing information for long edges.

    Parameters
    ----------
    G : networkx.Graph
        Input graph. Nodes must have a layer attribute indicated by *subset_key* (or the
        attribute is inferred/treated uniformly). Graph may be directed or undirected.
    subset_key : str, optional
        Node attribute key used to determine layer membership. Defaults to ``"subset"``.
    align : {'vertical', 'horizontal'}, optional
        Whether layers are arranged vertically (columns) or horizontally (rows). Defaults
        to ``"vertical"``.
    insert_dummies : bool, optional
        If True, insert dummy nodes for edges that span more than one layer to allow
        routing as poly-lines and improve crossing minimisation. Defaults to True.
    dummy_prefix : str, optional
        Prefix used for synthetic dummy node names when *insert_dummies* is True.
        Defaults to ``"__dummy__"``.
    node_spacing : float, optional
        Distance between adjacent nodes within a layer on the free axis. Defaults to 1.0.
    layer_spacing : float, optional
        Spacing between consecutive layers on the fixed axis. Defaults to 1.5.
    scale : float, optional
        Scaling factor applied to final coordinates. Defaults to 1.0.
    weight : str or None, optional
        Edge attribute name used as a weight for ordering and crossing computations.
        If None, unit weights are assumed. Defaults to None.
    max_proposals : int, optional
        Maximum number of proposals (moves) evaluated during simulated annealing.
        Larger values increase optimisation time and may yield fewer crossings. Defaults
        to 8000.
    cooling_rate : float, optional
        Multiplicative cooling factor applied to the temperature when scheduled. Must be
        in (0, 1). Defaults to 0.95.
    cooling_interval : int, optional
        Number of proposals between temperature reductions. Defaults to 200.
    adjacent_swap_prob : float, optional
        Probability of proposing an adjacent node swap vs. a broader move when creating
        candidates. Value in [0.0, 1.0]. Defaults to 0.7.
    stop_after_no_improve : int or None, optional
        If provided, stop optimisation after this many proposals without improvement.
        Set to None to disable early stopping. Defaults to 2000.
    T0 : float or None, optional
        Initial temperature for simulated annealing. If None, an automatic initial
        temperature is estimated from initial crossing costs. Defaults to None.
    seed : int or None, optional
        Random seed to ensure reproducible layouts. Defaults to None.
    return_order : bool, optional
        If True return a dictionary mapping layer values to the final ordered node lists.
        Defaults to False.
    return_dummies : bool, optional
        If True include dummy nodes in the returned positioning and ordering results.
        Defaults to False.
    return_routes : bool, optional
        If True return routing information for edges (lists of intermediate coordinates
        or node chains) useful for drawing multi-segment edges. Defaults to False.

    Returns
    -------
    pos : dict
        Mapping of node -> (x, y) coordinate positions (float). Coordinates follow the
        chosen *align* convention (layers along the fixed axis).
    tuple, optional
        When one or more of ``return_order``, ``return_dummies`` or ``return_routes`` is
        True, a tuple containing additional results is returned. The ordering of
        additional return values is:
        (pos, orders, dummies, routes) where any of ``orders``, ``dummies`` or ``routes``
        is included only if requested via the corresponding flag:
        - orders : dict mapping original layer values -> list of nodes in final order.
        - dummies : set or list of dummy node identifiers (present when ``return_dummies``).
        - routes : list of routing descriptors for edges spanning layers (present when ``return_routes``).

    Raises
    ------
    ValueError
        If provided parameters are invalid (for example nonsensical spacings, nonpositive
        *max_proposals*, or cooling parameters outside valid ranges).
    TypeError
        If graph nodes do not expose the expected layer attribute type and cannot be
        interpreted or if edge weights are non-numeric when *weight* is specified.

    Notes
    -----
    - The algorithm is stochastic; supply *seed* to obtain deterministic behaviour.
    - Dummy nodes are purely layout artefacts to represent long edges as chains of unit
      hops; they may be filtered from the returned position map unless explicitly
      requested via ``return_dummies``.
    - Simulated annealing balances local adjacent swaps and larger perturbations controlled
      by *adjacent_swap_prob*; tuning *max_proposals*, *cooling_rate* and *cooling_interval*
      affects runtime and solution quality.
    - The function attempts to minimise weighted pairwise edge crossings using standard
      barycenter/median heuristics supplemented by the annealing phase.
    - When *return_routes* is True, edge routing information describes the polyline
      coordinates or dummy-node chain that should be used to draw multi-layer edges
      without large visual overlaps.
    """
    if seed is not None:
        random.seed(seed)

    # Layers from subset_key (keep original values but index them 0..L-1)
    node_layer_val = {}
    unique_layers = set()
    for n in G.nodes():
        lv = G.nodes[n].get(subset_key, 0)
        node_layer_val[n] = lv
        unique_layers.add(lv)
    try:
        layer_keys = sorted(unique_layers)
    except TypeError:
        layer_keys = sorted(unique_layers, key=str)

    layer_to_idx = {lv: i for i, lv in enumerate(layer_keys)}
    L = len(layer_keys)
    # per-layer node lists (will later include dummies)
    layers = [[] for _ in range(L)]
    for n in G.nodes():
        layers[layer_to_idx[node_layer_val[n]]].append(n)

    # stable initial order: degree then name
    for i in range(L):
        layers[i].sort(key=lambda n: (G.degree(n), str(n)))

    # Build adjacent-only edges (insert dummies if requested)
    def _edge_w(u, v, data=None):
        """
        Calculates the weight of an edge.

        Parameters:
        u (node): Source node.
        v (node): Target node.
        data (dict, optional): Edge data dictionary. Defaults to None.

        Returns:
        float: Weight of the edge.
        """
        if weight is None:
            return 1.0
        if data is not None and weight in data:
            return float(data[weight])
        try:
            return float(G[u][v].get(weight, 1.0))
        except Exception:
            return 1.0

    # Mapping node -> layer index (will grow with dummies)
    node_layer_idx = {n: layer_to_idx[node_layer_val[n]] for n in G.nodes()}
    # edges_by_pair[i] holds edges between layer i and i+1 as list of (u, v, w)
    edges_by_pair = [[] for _ in range(max(0, L - 1))]
    routes = []  # for returning polylines

    dummy_counter = 0

    def _add_edge_pair(i_left, u, v, w):
        """
        Adds an edge between two nodes in adjacent layers.

        Parameters:
        i_left (int): Index of the left layer.
        u (node): Source node in the left layer.
        v (node): Target node in the right layer.
        w (float): Weight of the edge.
        """
        if i_left < 0 or i_left >= L - 1:
            return
        edges_by_pair[i_left].append((u, v, w))

    if insert_dummies:
        # accumulate edges; handle MultiGraph by iterating .edges(data=True, keys=False)
        for (uu, vv, *rest) in G.edges(data=True):
            data = rest[0] if rest else {}
            w = _edge_w(uu, vv, data)
            lu, lv = node_layer_idx[uu], node_layer_idx[vv]
            if lu == lv:
                # keep same-layer as a "route" but it won't affect crossings
                routes.append({"endpoints": (uu, vv), "nodes": [uu, vv]})
                continue
            # orient lu < lv
            rev = False
            if lu > lv:
                uu, vv = vv, uu
                lu, lv = lv, lu
                rev = True

            chain = [uu]
            prev = uu
            for k in range(lu + 1, lv):
                dummy_counter += 1
                d = f"{dummy_prefix}{dummy_counter}"
                node_layer_idx[d] = k
                layers[k].append(d)
                prev, cur = prev, d
                _add_edge_pair(k - 1, prev, cur, w)
                chain.append(cur)
                prev = cur
            # last hop
            _add_edge_pair(lv - 1, prev, vv, w)
            chain.append(vv)
            if rev:
                chain.reverse()
            routes.append({"endpoints": (chain[0], chain[-1]), "nodes": chain})
    else:
        for (uu, vv, *rest) in G.edges(data=True, keys=False):
            data = rest[0] if rest else {}
            w = _edge_w(uu, vv, data)
            lu, lv = node_layer_idx[uu], node_layer_idx[vv]
            if abs(lu - lv) == 1:
                i_left = min(lu, lv)
                # direct adjacent edge; orient left->right
                if lu <= lv:
                    _add_edge_pair(i_left, uu, vv, w)
                    routes.append({"endpoints": (uu, vv), "nodes": [uu, vv]})
                else:
                    _add_edge_pair(i_left, vv, uu, w)
                    routes.append({"endpoints": (vv, uu), "nodes": [vv, uu]})
            else:
                # non-adjacent edges are ignored for crossing count if dummies off
                routes.append({"endpoints": (uu, vv), "nodes": [uu, vv]})

    # Crossing objective: sum over adjacent layer pairs
    def pair_cross(i):
        """
        Calculates the weighted crossings between two adjacent layers.

        Parameters:
        i (int): Index of the layer pair.

        Returns:
        float: Total weighted crossings for the layer pair.
        """
        if i < 0 or i >= L - 1:
            return 0.0
        left = layers[i]
        right = layers[i + 1]
        if not left or not right or not edges_by_pair[i]:
            return 0.0
        return _pair_crossings_weighted(left, right, edges_by_pair[i])

    def total_cross():
        """
        Calculates the total weighted crossings for all layer pairs.

        Returns:
        float: Total weighted crossings.
        """
        return sum(pair_cross(i) for i in range(L - 1))

    # Initial temperature (optional estimation)
    def estimate_T0(samples=64):
        """
        Estimates the initial temperature for simulated annealing.

        Parameters:
        samples (int, optional): Number of random swaps to sample. Defaults to 64.

        Returns:
        float: Estimated initial temperature.
        """
        if L == 0:
            return 1.0
        deltas = []
        for _ in range(samples):
            # pick a layer with >=2 nodes and that participates in crossings
            candidates = [i for i in range(L) if len(layers[i]) >= 2]
            if not candidates:
                break
            i = random.choice(candidates)
            before = pair_cross(i - 1) + pair_cross(i)  # pairs touching layer i
            n = len(layers[i])
            a, b = random.randrange(n), random.randrange(n)
            if a == b:
                continue
            layers[i][a], layers[i][b] = layers[i][b], layers[i][a]
            after = pair_cross(i - 1) + pair_cross(i)
            d = after - before
            # revert
            layers[i][a], layers[i][b] = layers[i][b], layers[i][a]
            if d > 0:
                deltas.append(d)
        if not deltas:
            return 1.0
        return max(1e-6, sum(deltas) / len(deltas))

    if T0 is None:
        T = estimate_T0()
    else:
        T = float(T0)

    # SA loop over within-layer swaps
    current_total = total_cross()
    best_layers = [list(lst) for lst in layers]
    best_total = current_total
    last_improve_at = 0

    def propose_swap(i):
        """
        Proposes a swap of two nodes in a layer.

        Parameters:
        i (int): Index of the layer.

        Returns:
        tuple: A tuple (i, a, b) representing the layer index and the indices of the nodes to swap.
        """
        n = len(layers[i])
        if n < 2:
            return None
        if random.random() < adjacent_swap_prob:
            a = random.randrange(n - 1)
            b = a + 1
        else:
            a, b = random.randrange(n), random.randrange(n)
            while b == a:
                b = random.randrange(n)
        return i, a, b

    for step in range(int(max_proposals)):
        # pick a layer that has at least 2 nodes
        cand_layers = [i for i in range(L) if len(layers[i]) >= 2]
        if not cand_layers:
            break
        i = random.choice(cand_layers)
        swap = propose_swap(i)
        if swap is None:
            continue
        _, a, b = swap

        before = pair_cross(i - 1) + pair_cross(i)
        # apply
        layers[i][a], layers[i][b] = layers[i][b], layers[i][a]
        after = pair_cross(i - 1) + pair_cross(i)
        delta = after - before

        accept = False
        if delta <= 0:
            accept = True
        else:
            # accept uphill with Boltzmann probability
            p = math.exp(-delta / max(T, 1e-12))
            if random.random() < p:
                accept = True

        if accept:
            current_total += delta
            if current_total + 1e-12 < best_total:
                best_total = current_total
                best_layers = [list(lst) for lst in layers]
                last_improve_at = step
        else:
            # revert
            layers[i][a], layers[i][b] = layers[i][b], layers[i][a]

        # Cooling
        if (step + 1) % int(max(1, cooling_interval)) == 0:
            T *= float(cooling_rate)

        # Early stop if stuck
        if step - last_improve_at >= int(stop_after_no_improve):
            break

    # Use best found ordering
    layers = best_layers

    # Coordinates
    pos_all = {}
    for i in range(L):
        n = len(layers[i])
        if n == 0:
            continue
        start = -0.5 * (n - 1) * node_spacing
        for j, u in enumerate(layers[i]):
            free = start + j * node_spacing
            fixed = i * layer_spacing
            if align == "vertical":  # columns
                pos_all[u] = (fixed, free)
            else:  # rows
                pos_all[u] = (free, fixed)

    if scale != 1.0:
        pos_all = {u: (scale * x, scale * y) for u, (x, y) in pos_all.items()}

    # Only original nodes unless requested
    def _is_dummy(n):
        """
        Checks if a node is a dummy node.

        Parameters:
        n (node): The node to check.

        Returns:
        bool: True if the node is a dummy node, False otherwise.
        """
        return isinstance(n, str) and n.startswith(dummy_prefix)

    if return_dummies:
        pos = dict(pos_all)
    else:
        pos = {u: xy for u, xy in pos_all.items() if not _is_dummy(u)}

    # Optional: return final order per original layer key (includes dummies)
    order_by_layer = None
    if return_order:
        order_by_layer = {layer_keys[i]: list(layers[i]) for i in range(L)}

    # Optional: routes with points
    routed = None
    if return_routes:
        routed = []
        for r in routes:
            chain = r["nodes"]
            pts = [pos_all[n] for n in chain if n in pos_all]
            routed.append({"endpoints": r["endpoints"], "nodes": list(chain), "points": pts})

    if return_order and return_routes:
        return pos, order_by_layer, routed
    if return_order:
        return pos, order_by_layer
    if return_routes:
        return pos, routed
    return pos


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
):
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
    if legends is None:
        legends = ["A", "B"]
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        raise ValueError("One or both SMILES strings could not be parsed by RDKit.")

    # Standardize molecule layouts for better visualization
    mol_a = standardize_mol(mol_a, add_hydrogens=False)
    mol_b = standardize_mol(mol_b, add_hydrogens=False)

    # Compute MCS (Maximum Common Substructure)
    mcs_params = rdFMCS.MCSParameters()
    mcs_params.Timeout = int(timeout_s)
    mcs_params.AtomCompare = rdFMCS.AtomCompare.CompareElements
    mcs_params.BondCompare = rdFMCS.BondCompare.CompareOrderExact
    mcs_params.RingMatchesRingOnly = bool(ring_matches_ring_only)
    mcs_params.CompleteRingsOnly = bool(complete_rings_only)

    mcs_res = rdFMCS.FindMCS([mol_a, mol_b], mcs_params)
    if not mcs_res.smartsString:
        # No overlap found; draw without highlights.
        return Draw.MolsToGridImage(
            [mol_a, mol_b],
            molsPerRow=2,
            subImgSize=(size[0] // 2, size[1]),
            legends=legends,
        )

    mcs_mol = Chem.MolFromSmarts(mcs_res.smartsString)
    if mcs_mol is None:
        return Draw.MolsToGridImage(
            [mol_a, mol_b],
            molsPerRow=2,
            subImgSize=(size[0] // 2, size[1]),
            legends=legends,
        )

    match_a = mol_a.GetSubstructMatch(mcs_mol)
    match_b = mol_b.GetSubstructMatch(mcs_mol)

    if not match_a or not match_b:
        # If MCS SMARTS can't be matched back (rare), draw without highlights.
        return Draw.MolsToGridImage(
            [mol_a, mol_b],
            molsPerRow=2,
            subImgSize=(size[0] // 2, size[1]),
            legends=legends,
        )

    # Map MCS bonds to bond indices in each molecule
    def _mcs_bond_indices(parent_mol, match: Tuple[int, ...]) -> List[int]:
        """
        Map the bonds in the MCS to their indices in the parent molecule.

        Parameters
        ----------
        parent_mol : rdkit.Chem.Mol
            The parent molecule.
        match : Tuple[int, ...]
            Atom indices in the parent molecule that match the MCS.

        Returns
        -------
        List[int]
            List of bond indices in the parent molecule that are part of the MCS.
        """
        bond_idxs = []
        for b in mcs_mol.GetBonds():
            a1 = match[b.GetBeginAtomIdx()]
            a2 = match[b.GetEndAtomIdx()]
            pb = parent_mol.GetBondBetweenAtoms(a1, a2)
            if pb is not None:
                bond_idxs.append(pb.GetIdx())
        return bond_idxs

    common_bonds_a = _mcs_bond_indices(mol_a, match_a)
    common_bonds_b = _mcs_bond_indices(mol_b, match_b)

    common_atoms_a = list(match_a)
    common_atoms_b = list(match_b)

    # Color dictionaries for RDKit drawing
    bond_colors_a: Dict[int, Tuple[float, float, float]] = {i: common_bond_color for i in common_bonds_a}
    bond_colors_b: Dict[int, Tuple[float, float, float]] = {i: common_bond_color for i in common_bonds_b}
    atom_colors_a: Dict[int, Tuple[float, float, float]] = {i: common_atom_color for i in common_atoms_a}
    atom_colors_b: Dict[int, Tuple[float, float, float]] = {i: common_atom_color for i in common_atoms_b}

    # Generate the image with highlighted atoms and bonds
    img = Draw.MolsToGridImage(
        [mol_a, mol_b],
        molsPerRow=2,
        subImgSize=(size[0] // 2, size[1]),
        legends=legends,
        highlightBondLists=[common_bonds_a, common_bonds_b],
        highlightBondColors=[bond_colors_a, bond_colors_b],
        highlightAtomLists=[common_atoms_a, common_atoms_b],
        highlightAtomColors=[atom_colors_a, atom_colors_b],
        useSVG=False,  # Set True if you prefer SVG output
    )
    return img


def draw_mol_grid(
        mols: Sequence[Union[Chem.Mol, str]],
        legends: Optional[Sequence[str]] = None,
        n_cols: int = 4,
        sub_img_size: tuple = (200, 200),
        max_mols: Optional[int] = None,
        use_svg: bool = False,
):
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

        # Convert inputs to RDKit Mol objects
    rdkit_mols: List[Chem.Mol] = []
    for i, m in enumerate(mols):
        if isinstance(m, Chem.Mol):
            mol_obj = m
        elif isinstance(m, str):
            mol_obj = Chem.MolFromSmiles(m)
        else:
            raise TypeError(f"Item {i} is neither an RDKit Mol nor a SMILES string: {type(m)}")

        if mol_obj is None:
            mol_obj = Chem.MolFromSmiles("")
        rdkit_mols.append(mol_obj)

    if max_mols is not None:
        rdkit_mols = rdkit_mols[: int(max_mols)]

        # Legends
    if legends is None:
        legends_list = [""] * len(rdkit_mols)
    else:
        if len(legends) != len(rdkit_mols):
            raise ValueError("legends must be the same length as mols")
        legends_list = list(legends)

        # Draw
    img = Draw.MolsToGridImage(
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
):
    if n_cols <= 0:
        raise ValueError("n_cols must be a positive integer")
    if gap < 0 or outer_margin < 0 or inner_pad < 0:
        raise ValueError("gap/outer_margin/inner_pad must be >= 0")

    if sort_by is not None:
        sorted_indices = sorted(range(len(mols)), key=lambda i: sort_by[i], reverse=False)
        mols = [mols[i] for i in sorted_indices]
        if legends is not None:
            legends = [legends[i] for i in sorted_indices]

    # Convert inputs to RDKit Mol objects
    rdkit_mols: List[Chem.Mol] = []
    for i, m in enumerate(mols):
        if isinstance(m, Chem.Mol):
            mol_obj = m
        elif isinstance(m, str):
            mol_obj = Chem.MolFromSmiles(m)
        else:
            raise TypeError(f"Item {i} is neither an RDKit Mol nor a SMILES string: {type(m)}")

        if mol_obj is None:
            mol_obj = Chem.MolFromSmiles("")
        rdkit_mols.append(mol_obj)

    if max_mols is not None:
        rdkit_mols = rdkit_mols[: int(max_mols)]

    # Legends
    if legends is None:
        legends_list = [""] * len(rdkit_mols)
    else:
        if len(legends) != len(rdkit_mols):
            raise ValueError("legends must be the same length as mols")
        legends_list = list(legends)

    n_mols = len(rdkit_mols)
    if n_mols == 0:
        # Return a tiny blank image rather than erroring
        return Image.new("RGB", (outer_margin * 2 + 1, outer_margin * 2 + 1), "white")

    n_rows = math.ceil(n_mols / n_cols)
    box_w, box_h = sub_img_size

    # Size available for the RDKit drawing inside the grey tile
    inner_w = max(1, box_w - 2 * inner_pad)
    inner_h = max(1, box_h - 2 * inner_pad)

    # Final canvas (white background = the "whitespace" between tiles)
    canvas_w = outer_margin * 2 + n_cols * box_w + (n_cols - 1) * gap
    canvas_h = outer_margin * 2 + n_rows * box_h + (n_rows - 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    # Render each molecule into its own grey tile, then paste into canvas
    for idx, (mol, legend) in enumerate(zip(rdkit_mols, legends_list)):
        r = idx // n_cols
        c = idx % n_cols

        # RDKit per-mol image (PIL). Legend is drawn within this image.
        mol_img = Draw.MolToImage(mol, size=(inner_w, inner_h), legend=legend)

        # Make the grey box and paste the mol drawing with padding
        tile = Image.new("RGB", (box_w, box_h), box_bg)
        tile.paste(mol_img, (inner_pad, inner_pad))

        x = outer_margin + c * (box_w + gap)
        y = outer_margin + r * (box_h + gap)
        canvas.paste(tile, (x, y))

    return canvas


def plot_ir_spectrum(spectrum: np.ndarray,
                     peaks: np.ndarray | None = None,
                     highlight_range: Optional[Tuple[float, float]] = (400.0, 1500.0),
                     xlab: str = 'Wavenumber (cm⁻¹)',
                     ylab: str = 'Intensity',
                     flip_x: bool = True,
                     figsize: Tuple[float, float] = (8, 5),
                     fontsize: int = 16) -> Tuple[Figure, Axes]:
    freq = spectrum.T[0]
    intensity = spectrum.T[1]
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(freq, intensity, linewidth=2, color='black')

    if peaks is not None:
        plt.scatter(freq[peaks], intensity[peaks], color='red')

    # Apply standard styling
    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)

    # Add highlighted background region
    if highlight_range:
        ax.axvspan(highlight_range[0], highlight_range[1], color='lightgrey', alpha=0.5, zorder=0)

    # Invert x-axis if requested (standard for IR)
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
):
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
    # Create a Matplotlib figure and axis with the specified size
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot the ASE Atoms object with the specified parameters
    plot_atoms(atoms, ax=ax, rotation=rotation, show_unit_cell=show_unit_cell)

    # Turn off the axis for a cleaner visualization
    ax.axis("off")

    # Save the figure to the specified file if `outfile` is provided
    if outfile:
        fig.savefig(outfile,
                    dpi=dpi,
                    transparent=transparent,
                    bbox_inches="tight",
                    pad_inches=0)

    # Return the Matplotlib figure and axis
    return fig, ax
