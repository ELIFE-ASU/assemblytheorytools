import math
import random
import tempfile
from collections import defaultdict
from html import escape
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from IPython.display import HTML
from ase.visualize.plot import plot_atoms
from matplotlib import colormaps, colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Circle
from pyvis.network import Network
from rdkit.Chem import Draw
from scipy.stats import gaussian_kde

import CFG
from .tools_atoms import mol_to_atoms
from .tools_graph import relabel_digraph, nx_to_smi
from .tools_mol import smi_to_mol

# set the plot axis
plt.rcParams['axes.linewidth'] = 2.0


def n_plot(xlab: str, ylab: str, xs: int = 14, ys: int = 14) -> None:
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    plt.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    plt.tick_params(axis='both', which='both', top=True, right=True)
    plt.xlabel(xlab, fontsize=xs)
    plt.ylabel(ylab, fontsize=ys)
    plt.tight_layout()
    return None


def ax_plot(fig: plt.Figure, ax: plt.Axes, xlab: str, ylab: str, xs: int = 14, ys: int = 14) -> None:
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
        for layer, nodes in enumerate(nx.topological_generations(graph)):
            for node in nodes:
                graph.nodes[node]["layer"] = layer
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
                       vo_smiles: bool = True) -> None:
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

    if vo_smiles:
        try:
            for node in digraph.nodes:
                # set the node label to the smiles
                digraph.nodes[node]['label'] = nx_to_smi(digraph.nodes[node]['vo'],
                                                         add_hydrogens=False,
                                                         sanitize=False)
        except:
            pass

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
                 frame_on: bool = True) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=fig_size)

    for layer, nodes in enumerate(nx.topological_generations(graph)):
        for node in nodes:
            graph.nodes[node]["layer"] = layer

    if layout_style == 'crossmin':
        pos = multipartite_layout_crossmin(graph, subset_key="layer")
    elif layout_style == 'crossmin_long':
        pos = multipartite_layout_crossmin_long(graph, subset_key="layer")
    elif layout_style == 'sa':
        pos = multipartite_layout_sa(graph, subset_key="layer")
    else:
        pos = nx.multipartite_layout(graph, subset_key="layer")

    if arrow_style == '1':
        edge_color = 'white'
    elif arrow_style == '2':
        edge_color = 'grey'
    else:
        raise ValueError("Invalid arrow style. Use '1' or '2'.")

    nx.draw_networkx(graph,
                     pos=pos,
                     ax=ax,
                     with_labels=False,
                     node_size=1000,
                     node_color=node_color,
                     connectionstyle="arc3,rad=0.1",
                     edge_color=edge_color,
                     arrows=True,
                     arrowstyle="->",
                     width=2.0)

    if arrow_style == '1':
        if show_icons:
            arrow_margin = 50
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
                        rad = -0.10 * layer_diff
                    else:
                        rad = 0.10 * layer_diff
                else:
                    rad = 0.0

            nx.draw_networkx_edges(
                graph,
                pos=pos,
                edgelist=[edge],
                ax=ax,
                arrows=True,
                arrowstyle="->",
                width=2.5,
                edge_color="grey",
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
                atoms = mol_to_atoms(mol, sanitize=False, add_hydrogen=False)
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
    fig.tight_layout()
    ax.axis('off')
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
                           labels: bool,
                           node_size: float,
                           arrow_size: float,
                           node_color: str,
                           edge_color: str,
                           fig_size: float,
                           filename: str):
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

    labels : bool
        If True, display node labels on the plot.

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

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.

    ax : matplotlib.axes.Axes
        The matplotlib Axes object containing the plot.


    """
    if len(nodes) != len(adjacency_matrix) or len(adjacency_matrix) != len(x) or len(x) != len(y):
        raise ValueError("Lengths of nodes, adjacency_matrix, x, and y must be equal.")

    # Create a directed graph
    graph = nx.DiGraph()

    # Add nodes and their positions
    positions = {nodes[i]: (x[i], y[i]) for i in range(len(nodes))}
    graph.add_nodes_from(nodes)

    # Add edges based on the adjacency matrix
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            # Non-zero value indicates an edge
            if adjacency_matrix[i][j] != 0:
                graph.add_edge(nodes[i], nodes[j])

    # Create a plot
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Draw concentric circles
    for radius in range(1, max_ai + 2):
        circle = Circle((0, 0), radius, color="black", alpha=1, fill=False, lw=1.5)
        ax.add_artist(circle)

    # Draw the graph
    nx.draw(
        graph,
        pos=positions,
        with_labels=labels,
        node_color=node_color,
        edge_color='white',
        node_size=node_size,
        font_size=node_size / 200,
        font_color="black",
        arrowstyle="->",
        arrowsize=arrow_size,
        connectionstyle="arc3,rad=0.2"  # For curved edges
    )
    for edge in graph.edges():
        src, dst = edge
        # If the source node is above the destination node, curve the arrow downward (negative rad)
        if positions[src][0] > positions[dst][0]:
            rad = 0.25
        # If the source node is below the destination node, curve the arrow upward (positive rad)
        elif positions[src][0] < positions[dst][0]:
            rad = -0.25
        # If the source and destination nodes are horizontally aligned
        else:
            rad = 0.0

        nx.draw_networkx_edges(
            graph,
            pos=positions,
            edgelist=[edge],
            ax=ax,
            arrows=True,
            arrowstyle="->",
            width=2.5,
            edge_color=edge_color,
            connectionstyle=f"arc3,rad={rad}",
            min_target_margin=10,
        )

    # Set limits for the plot to accommodate the circles
    ax.set_xlim(-max_ai - 1.5, max_ai + 1.5)
    ax.set_ylim(-max_ai - 1.5, max_ai + 1.5)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    plt.savefig(f"{filename}", dpi=600)
    return fig, ax


def plot_assembly_circle(nodes,
                         adj_matrix=None,
                         assembly_indices=None,
                         labels=None,
                         node_size=1000,
                         arrow_size=80,
                         node_color='#264f70',
                         edge_color='Grey',
                         fig_size=10,
                         filename='assembly_circles.png'):
    '''
    Here is a function to plot a graph, where objects are displayed in concentric
    circles according to their assembly index.

        Parameters:
        ----------
        nodes : list
            A list of nodes in the network that are to be visualized.

        assembly_indices (OPTIONAL): list or numpy.ndarray
            If not provided, they will be calculated for strings.

        adj_matrix (OPTIONAL, but recommended): numpy.ndarray
            A square adjacency matrix representing the relationships between nodes.
            If adj_matrix[i, j] >= 1, it signifies that node i points to node j.
            If not provided, rules_graph will be used as adjacency matrix.
            IMPORTANT: if provided, must be assembly-consistent, that is, all nodes
            must be pointed to by at least one node with a lower assembly index,
            except for nodes with the minimum assembly index, which will be considered
            the building blocks.

        labels : list
            A list of labels corresponding to the nodes. These labels can be used for debugging or display purposes.

        node_size : float

        arrow_size (OPTIONAL): float

        node_color (OPTIONAL): str or list

        edge_color (OPTIONAL): str or list

        fig_size (OPTIONAL): float

        filename (OPTIONAL): str

    '''
    if adj_matrix is None:  # If adj matrix is not provided, the rules_graph will be the output
        G = CFG.ai_with_pathways(nodes, f_print=False)[2]
        nodes = list(G.nodes())
        adj_matrix = nx.adjacency_matrix(G).toarray()

    n_nodes = len(nodes)

    if assembly_indices is None:
        assembly_indices = np.zeros(n_nodes, dtype=int)

        for i in range(n_nodes):
            assembly_indices[i] = CFG.ai_with_pathways(nodes[i], f_print=False)[0]

    angles = np.full(n_nodes, np.nan)

    max_ai = max(assembly_indices)
    min_ai = min(assembly_indices)

    # Finding the number of building blocks, defined as those with minimum assembly index
    n_building_blocks = 0
    for i in range(n_nodes):
        if assembly_indices[i] == min_ai:
            n_building_blocks += 1

    # Assign equispaced angles to building blocks
    n = 1
    for i in range(n_nodes):
        if assembly_indices[i] == min_ai:
            angles[i] = n * 2 * np.pi / n_building_blocks
            n += 1

    # Assign angles for higher assembly index objects
    while np.any(np.isnan(angles)):  # While there are angles left
        for i in range(n_nodes):
            # If the string has no angle associated
            if np.isnan(angles[i]):
                set_angles = np.array([])
                for j in range(n_nodes):
                    if adj_matrix[j, i] >= 1:
                        set_angles = np.append(set_angles, angles[j])
                if np.all(~np.isnan(set_angles)):
                    angles[i] = _average_angles(set_angles)

    # Transform positions from polar to cartesian
    x_positions = (assembly_indices + 1) * np.cos(angles)
    y_positions = (assembly_indices + 1) * np.sin(angles)

    # Add custom labels if not "None"
    if labels is None:
        display_labels = nodes
    else:
        display_labels = labels

    # Plot the network
    fig, ax = _plot_directed_network(display_labels,
                                     adj_matrix,
                                     x_positions,
                                     y_positions,
                                     max_ai,
                                     labels,  # always show labels if custom provided
                                     node_size,
                                     arrow_size,
                                     node_color,
                                     edge_color,
                                     fig_size,
                                     filename)
    return fig, ax


def scatter_plot(x,
                 y,
                 xlab='x',
                 ylab='y',
                 figsize=(8, 5),
                 fontsize=16,
                 alpha=0.5,
                 ):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, color='black', alpha=alpha, s=50)
    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    return fig, ax


def scatter_plot_with_colorbar(x,
                               y,
                               xlab='x',
                               ylab='y',
                               cmap='viridis',
                               figsize=(8, 5),
                               fontsize=16,
                               ):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to numpy arrays if they aren't already
    x = np.asarray(x)
    y = np.asarray(y)

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

    # # Add colour bar
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.set_label('Point Density', fontsize=fontsize)

    # Configure the plot
    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    return fig, ax


def plot_contourf_full(x,
                       y,
                       xlab,
                       ylab,
                       c_map="Purples",
                       figsize=(8, 5),
                       fontsize=16):
    fig, ax = plt.subplots(figsize=figsize)
    lims = [min(x), max(x)]

    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[lims[0]:lims[1]:x.size ** 0.6 * 1j, lims[0]:lims[1]:y.size ** 0.6 * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    ax.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.9, cmap=c_map)  # , levels=20)

    # set the axis limits
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # add axis labels
    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    return fig, ax


def plot_heatmap(x,
                 y,
                 xlab,
                 ylab,
                 c_map='viridis',
                 nbins=50,
                 figsize=(8, 5),
                 fontsize=16):
    fig, ax = plt.subplots(figsize=figsize)
    # Create a 2D histogram of the data
    heatmap_data, xedges, yedges = np.histogram2d(x, y, bins=(nbins, nbins))
    im = ax.imshow(heatmap_data.T,
                   origin='lower',
                   cmap=c_map,
                   aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    # Add colour bar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Point Density', fontsize=fontsize)
    ax_plot(fig, ax, xlab=xlab, ylab=ylab, xs=fontsize, ys=fontsize)
    return fig, ax


def scatter_plot_3d_with_colorbar(x,
                                  y,
                                  z,
                                  c=None,
                                  xlab='x',
                                  ylab='y',
                                  zlab='z',
                                  cmap='viridis',
                                  figsize=(10, 8),
                                  fontsize=20,
                                  alpha=0.8,
                                  s=50,
                                  labelpad=20):
    # Create a figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Convert to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

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


def multipartite_layout_crossmin(
        G,
        subset_key="subset",
        align="vertical",
        method="barycenter",
        iterations=100,
        layer_spacing=1.0,
        node_spacing=1.0,
        scale=1.0,
        seed=None,
        weight=None,
        return_order=False,
):
    """
    Computes a multipartite layout for a graph, minimizing edge crossings between layers.

    Parameters:
    G (networkx.Graph): The input graph.
    subset_key (str, optional): Node attribute used to determine layer assignment. Defaults to "subset".
    align (str, optional): Alignment of layers. "vertical" for columns, "horizontal" for rows. Defaults to "vertical".
    method (str, optional): Method for ordering nodes within layers. "barycenter" or "median". Defaults to "barycenter".
    iterations (int, optional): Number of top-down and bottom-up sweeps to refine node order. Defaults to 100.
    layer_spacing (float, optional): Spacing between layers. Defaults to 1.0.
    node_spacing (float, optional): Spacing between nodes within a layer. Defaults to 1.0.
    scale (float, optional): Scaling factor for the layout. Defaults to 1.0.
    seed (int, optional): Random seed for reproducibility. Defaults to None.
    weight (str, optional): Edge attribute used as weight for ordering. Defaults to None.
    return_order (bool, optional): If True, returns the final node order per layer. Defaults to False.

    Returns:
    dict: A dictionary mapping nodes to their (x, y) positions.
    tuple (optional): If return_order is True, also returns a dictionary of node orders per layer.
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


def multipartite_layout_crossmin_long(
        G,
        subset_key="subset",
        align="vertical",
        method="barycenter",
        iterations=100,
        layer_spacing=1.0,
        node_spacing=1.0,
        scale=1.0,
        seed=None,
        weight=None,
        insert_dummies=True,
        dummy_prefix="__dummy__",
        return_order=False,
        return_dummies=False,
        return_routes=False,
):
    """
    Computes a multipartite layout for a graph, minimizing edge crossings between layers.
    This version supports dummy node insertion for edges spanning multiple layers.

    Parameters:
    G (networkx.Graph): The input graph.
    subset_key (str, optional): Node attribute used to determine layer assignment. Defaults to "subset".
    align (str, optional): Alignment of layers. "vertical" for columns, "horizontal" for rows. Defaults to "vertical".
    method (str, optional): Method for ordering nodes within layers. "barycenter" or "median". Defaults to "barycenter".
    iterations (int, optional): Number of top-down and bottom-up sweeps to refine node order. Defaults to 100.
    layer_spacing (float, optional): Spacing between layers. Defaults to 1.0.
    node_spacing (float, optional): Spacing between nodes within a layer. Defaults to 1.0.
    scale (float, optional): Scaling factor for the layout. Defaults to 1.0.
    seed (int, optional): Random seed for reproducibility. Defaults to None.
    weight (str, optional): Edge attribute used as weight for ordering. Defaults to None.
    insert_dummies (bool, optional): If True, inserts dummy nodes for edges spanning multiple layers. Defaults to True.
    dummy_prefix (str, optional): Prefix for dummy node names. Defaults to "__dummy__".
    return_order (bool, optional): If True, returns the final node order per layer. Defaults to False.
    return_dummies (bool, optional): If True, includes dummy nodes in the returned layout. Defaults to False.
    return_routes (bool, optional): If True, returns routing information for edges. Defaults to False.

    Returns:
    dict: A dictionary mapping nodes to their (x, y) positions.
    dict (optional): If return_order is True, also returns a dictionary of node orders per layer.
    list (optional): If return_routes is True, also returns a list of edge routing information.
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

    def __init__(self, n):
        """
        Initializes the Fenwick tree.

        Parameters:
        n (int): The size of the tree (number of elements).
        """
        self.n = n
        self.t = [0.0] * (n + 1)

    def add(self, i, delta):
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

    def sum_prefix(self, i):
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


def _pair_crossings_weighted(order_left, order_right, edges):
    """
    Counts the weighted crossings between two adjacent layers given their current order.

    Parameters:
    order_left (list): A list of nodes representing the order of nodes in the left layer.
    order_right (list): A list of nodes representing the order of nodes in the right layer.
    edges (list of tuples): A list of edges connecting nodes in the left layer to nodes in the right layer.
                            Each edge is represented as a tuple (u, v, w), where:
                            - u: Node in the left layer.
                            - v: Node in the right layer.
                            - w: Weight of the edge.

    Returns:
    float: The total weighted crossing value. The crossing weight is the product of edge weights (w1 * w2).
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


def multipartite_layout_sa(
        G,
        subset_key="subset",
        align="vertical",
        insert_dummies=True,
        dummy_prefix="__dummy__",
        node_spacing=1.0,
        layer_spacing=1.5,
        scale=1.0,
        weight=None,
        max_proposals=8000,
        cooling_rate=0.95,
        cooling_interval=200,
        adjacent_swap_prob=0.7,
        stop_after_no_improve=2000,
        T0=None,
        seed=None,
        return_order=False,
        return_dummies=False,
        return_routes=False,
):
    """
    Computes a multipartite layout for a graph using simulated annealing to minimize edge crossings.

    Parameters:
    G (networkx.Graph): The input graph.
    subset_key (str, optional): Node attribute used to determine layer assignment. Defaults to "subset".
    align (str, optional): Alignment of layers. "vertical" for columns, "horizontal" for rows. Defaults to "vertical".
    insert_dummies (bool, optional): If True, inserts dummy nodes for edges spanning multiple layers. Defaults to True.
    dummy_prefix (str, optional): Prefix for dummy node names. Defaults to "__dummy__".
    node_spacing (float, optional): Spacing between nodes within a layer. Defaults to 1.0.
    layer_spacing (float, optional): Spacing between layers. Defaults to 1.5.
    scale (float, optional): Scaling factor for the layout. Defaults to 1.0.
    weight (str, optional): Edge attribute used as weight for ordering. Defaults to None.
    max_proposals (int, optional): Maximum number of proposals for simulated annealing. Defaults to 8000.
    cooling_rate (float, optional): Cooling rate for simulated annealing. Defaults to 0.95.
    cooling_interval (int, optional): Number of steps between temperature reductions. Defaults to 200.
    adjacent_swap_prob (float, optional): Probability of swapping adjacent nodes during proposals. Defaults to 0.7.
    stop_after_no_improve (int, optional): Number of steps without improvement before stopping. Defaults to 2000.
    T0 (float, optional): Initial temperature for simulated annealing. If None, it is estimated. Defaults to None.
    seed (int, optional): Random seed for reproducibility. Defaults to None.
    return_order (bool, optional): If True, returns the final node order per layer. Defaults to False.
    return_dummies (bool, optional): If True, includes dummy nodes in the returned layout. Defaults to False.
    return_routes (bool, optional): If True, returns routing information for edges. Defaults to False.

    Returns:
    dict: A dictionary mapping nodes to their (x, y) positions.
    dict (optional): If return_order is True, also returns a dictionary of node orders per layer.
    list (optional): If return_routes is True, also returns a list of edge routing information.
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
