import random
import tempfile
from html import escape
from typing import List

import cairosvg
import dagviz
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

import CFG
from .tools_atoms import mol_to_atoms
from .tools_graph import relabel_digraph
from .tools_mol import smi_to_mol


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


def plot_digraph_metro(digraph: nx.DiGraph, filename: str = 'metro', steps: bool = False) -> None:
    if steps:
        # Relabel the graph nodes with their topological step if requested
        digraph = relabel_digraph(digraph)

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


def plot_pathway_mol(graph: nx.DiGraph,
                     fig_size: tuple = (12, 7),
                     show_icons: bool = True,
                     layout: str = 'topological',
                     seed: int = 42,
                     arrow_style: str = '1',
                     frame_on: bool = True) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=fig_size)
    cmap = plt.get_cmap("Blues")
    node_colors = [cmap(0.4) for _ in graph.nodes]

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
                     node_color=node_colors,
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
    fig.tight_layout()
    ax.axis('off')
    return fig, ax


def plot_pathway_graph(graph: nx.DiGraph,
                       fig_size: tuple = (12, 7),
                       show_icons: bool = True,
                       layout: str = 'topological',
                       seed: int = 42,
                       arrow_style: str = '1',
                       frame_on: bool = True) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=fig_size)
    cmap = plt.get_cmap("Blues")
    node_colors = [cmap(0.4) for _ in graph.nodes]

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
                     node_color=node_colors,
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
    fig.tight_layout()
    ax.axis('off')
    return fig, ax


def plot_pathway_atoms(graph: nx.DiGraph,
                       fig_size: tuple = (12, 7),
                       show_icons: bool = True,
                       layout: str = 'topological',
                       seed: int = 42,
                       arrow_style: str = '1',
                       frame_on: bool = True) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=fig_size)
    cmap = plt.get_cmap("Blues")
    node_colors = [cmap(0.4) for _ in graph.nodes]

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
                     node_color=node_colors,
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
        for i, node in enumerate(graph.nodes):
            smi = graph.nodes[node]["vo"]
            mol = smi_to_mol(smi, add_hydrogens=False)
            atoms = mol_to_atoms(mol, sanitize=False, add_hydrogen=False)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmpfile:
                _fig, _ax = plt.subplots()
                plot_atoms(atoms, _ax, show_unit_cell=0, scale=2.0)
                _fig.tight_layout()
                _ax.axis('off')
                _fig.savefig(tmpfile.name, dpi=600, bbox_inches='tight')

                # close the figure
                plt.close(_fig)
                img = plt.imread(tmpfile.name)

            imagebox = OffsetImage(img, zoom=0.01)
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
                         node_color='Skyblue',
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
