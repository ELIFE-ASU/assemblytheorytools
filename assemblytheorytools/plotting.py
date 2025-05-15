import os
from html import escape

import cairosvg
import dagviz
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from IPython.display import HTML
from matplotlib import colormaps, colors
from matplotlib.patches import Circle
from pyvis.network import Network
from typing import List
import CFG


def n_plot(xlab: str, ylab: str, xs: int = 14, ys: int = 14) -> None:
    """
    Configure and style the plot with specified labels and tick parameters.

    Use with
    plt.rcParams['axes.linewidth'] = 2.0

    Args:
        xlab (str): The label for the x-axis.
        ylab (str): The label for the y-axis.
        xs (int, optional): Font size for the x-axis label. Default is 14.
        ys (int, optional): Font size for the y-axis label. Default is 14.

    Returns:
        None
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
    Configure and style the plot with specified labels and tick parameters for a given axis.

    Use with
    plt.rcParams['axes.linewidth'] = 2.0

    Args:
        fig (plt.Figure): The figure object containing the plot.
        ax (plt.Axes): The axes object to be styled.
        xlab (str): The label for the x-axis.
        ylab (str): The label for the y-axis.
        xs (int, optional): Font size for the x-axis label. Default is 14.
        ys (int, optional): Font size for the y-axis label. Default is 14.

    Returns:
        None
    """
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    ax.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    ax.tick_params(axis='both', which='both', top=True, right=True)
    ax.set_xlabel(xlab, fontsize=xs)
    ax.set_ylabel(ylab, fontsize=ys)
    fig.tight_layout()
    return None


def os_plot_show() -> None:
    """
    Display or close the plot based on the operating system.

    If the operating system is Windows, the plot will be displayed using plt.show().
    Otherwise, the plot will be closed using plt.close().

    Returns:
        None
    """
    # check if the OS is windows
    if os.name == 'nt':
        plt.show()
    else:
        plt.close()
    return None


def plot_graph(graph: nx.Graph,
               layout: str = 'kawai',
               f_labs: bool = False,
               edge_color: str = 'grey',
               node_size: int = 300,
               edgecolors: str = "black",
               width: int = 2,
               linewidths: int = 2,
               filename: str = "graph",
               seed: int = 42) -> None:
    """
    Plot a graph using NetworkX and Matplotlib with various layout options.

    Args:
        graph (nx.Graph): The graph to be plotted.
        layout (str, optional): The layout algorithm to use for positioning nodes. Default is 'kawai'.
        f_labs (bool, optional): Whether to display labels on the nodes. Default is False.
        edge_color (str, optional): Color of the edges. Default is 'grey'.
        node_size (int, optional): Size of the nodes. Default is 300.
        edgecolors (str, optional): Color of the node borders. Default is 'black'.
        width (int, optional): Width of the edges. Default is 2.
        linewidths (int, optional): Width of the node borders. Default is 2.
        filename (str, optional): Base name of the file where the plot will be saved. Default is 'graph'.
        seed (int, optional): Seed for the layout algorithm (if applicable). Default is 42.

    Returns:
        None
    """
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

    # Draw the graph with the specified parameters
    nx.draw(graph,
            pos=pos,
            with_labels=f_labs,
            edge_color=edge_color,
            node_size=node_size,
            edgecolors=edgecolors,
            width=width,
            linewidths=linewidths)

    # Save the plot as PNG and PDF
    plt.savefig(f"{filename}.png", dpi=600)
    plt.savefig(f"{filename}.pdf")

    # Display or close the plot based on the operating system
    os_plot_show()

    return None


def plot_mol_graph(graph: nx.Graph, f_labs: bool = False, filename: str = "atom_graphs") -> None:
    """
    Plot a molecular graph using NetworkX and Matplotlib.

    Args:
        graph (nx.Graph): The graph to be plotted where nodes represent atoms and edges represent bonds.
        f_labs (bool, optional): Whether to display labels on the nodes. Default is False.
        filename (str, optional): The base name of the file where the plot will be saved. Default is "atom_graphs".

    Returns:
        None
    """
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

    color_dict_edge = {1.0: "black",
                       2.0: "green",
                       3.0: "red",
                       4.0: "orange"}
    # Get the colors for the nodes
    graph_colors = [cols_conv.get(graph.nodes[idx]['color'], 'black') for idx in graph.nodes()]
    # Get the colors for the edges
    edge_colors = [color_dict_edge.get(graph.edges[idx]['color']) for idx in graph.edges()]

    # Get the position of the nodes
    pos = nx.kamada_kawai_layout(graph)

    # Draw the graph
    nx.draw(graph,
            pos=pos,
            with_labels=f_labs,
            node_size=300,
            edge_color=edge_colors,
            node_color=graph_colors,
            edgecolors="black",
            width=2,
            linewidths=2)
    # Save the plot as PNG and PDF
    plt.savefig(f"{filename}.png", dpi=600)
    plt.savefig(f"{filename}.pdf")
    # Display or close the plot based on the operating system
    os_plot_show()
    return None


def plot_interactive_graph(graph: nx.Graph,
                           show: bool = False,
                           filename: str = "interactive_graph.html") -> Network:
    """
    Plot an interactive graph using PyVis and display it in a Jupyter notebook or save it as an HTML file.

    Args:
        graph (nx.Graph): The graph to be plotted.
        show (bool, optional): Whether to display the graph in a Jupyter notebook. Default is False.
        filename (str, optional): The name of the file where the graph will be saved if not displayed.
            Default is "interactive_graph.html".

    Returns:
        Network: The PyVis network object representing the graph.
    """
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


def plot_digraph(digraph: nx.DiGraph,
                 layout: str = 'spring',
                 f_labs: bool = True,
                 edge_color: str = 'grey',
                 node_size: int = 300,
                 edgecolors: str = "black",
                 width: int = 2,
                 linewidths: int = 2,
                 filename: str = "digraph",
                 seed: int = 42) -> None:
    """
    Plot a directed graph using NetworkX and Matplotlib with various layout options.

    Args:
        digraph (nx.DiGraph): The directed graph to be plotted.
        layout (str, optional): The layout algorithm to use for positioning nodes. Default is 'spring'.
        f_labs (bool, optional): Whether to display labels on the nodes. Default is True.
        edge_color (str, optional): Color of the edges. Default is 'grey'.
        node_size (int, optional): Size of the nodes. Default is 300.
        edgecolors (str, optional): Color of the node borders. Default is 'black'.
        width (int, optional): Width of the edges. Default is 2.
        linewidths (int, optional): Width of the node borders. Default is 2.
        filename (str, optional): Base name of the file where the plot will be saved. Default is 'digraph'.
        seed (int, optional): Seed for the layout algorithm (if applicable). Default is 42.

    Returns:
        None
    """
    plot_graph(digraph,
               layout=layout,
               f_labs=f_labs,
               edge_color=edge_color,
               node_size=node_size,
               edgecolors=edgecolors,
               width=width,
               linewidths=linewidths,
               filename=filename,
               seed=seed)
    return None


def plot_digraph_metro(digraph: nx.DiGraph, filename: str = 'metro') -> None:
    """
    Plot a directed graph using the Metro style and save it as SVG and PNG files.

    Args:
        digraph (nx.DiGraph): The directed graph to be plotted.
        filename (str, optional): The base name of the files where the plot will be saved. Default is 'metro'.

    Returns:
        None
    """
    r = dagviz.render_svg(digraph,
                          style=dagviz.style.metro.svg_renderer(dagviz.style.metro.StyleConfig(node_stroke="black")))
    # Save the SVG
    with open(f'{filename}.svg', 'w') as file:
        file.write(r)

    # Save the PNG file
    cairosvg.svg2png(bytestring=r.encode('utf-8'), write_to=f"{filename}.png")
    return None


def plot_digraph_topological(digraph: nx.DiGraph, filename: str = 'topological') -> None:
    """
    Plot a directed graph using a topological layout and save it as a PNG file.

    This function assigns layers to nodes based on their topological generation and uses the
    multipartite layout to position the nodes. The plot is saved as a PNG file.

    https://networkx.org/documentation/stable/auto_examples/graph/plot_dag_layout.html

    Args:
        digraph (nx.DiGraph): The directed graph to be plotted.
        filename (str, optional): The base name of the file where the plot will be saved. Default is 'topological'.

    Returns:
        None
    """
    for layer, nodes in enumerate(nx.topological_generations(digraph)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            digraph.nodes[node]["layer"] = layer

    # Compute the multipartite_layout using the "layer" node attribute
    pos = nx.multipartite_layout(digraph, subset_key="layer")

    # Compute the multipartite_layout using the "layer" node attribute
    fig, ax = plt.subplots()
    nx.draw_networkx(digraph, pos=pos, ax=ax)
    fig.tight_layout()
    plt.savefig(f"{filename}.png", dpi=600)
    plt.show()
    return None


def match_node_to_image(graph: nx.Graph, image_paths: list[str]) -> dict[str, str]:
    """
    Matches nodes in the graph to their respective image paths.

    This function iterates over the provided image paths, extracts the base name of each image file,
    and checks if it matches any node in the graph. If a match is found, it maps the node to the image path.

    Args:
        graph (nx.Graph): A NetworkX graph object containing nodes.
        image_paths (list[str]): A list of image file paths.

    Returns:
        dict[str, str]: A dictionary mapping node labels to their corresponding image paths.
    """
    # Initialise an empty dictionary to store the node-to-image mapping
    node_image_mapping = {}

    # Iterate over each image path
    for path in image_paths:
        # Get the base name (without extension) of the image file
        file_name = os.path.splitext(os.path.basename(path))[0]

        # Check if the base name matches any node in the graph
        if file_name in graph.nodes:
            # Map the node to the image path
            node_image_mapping[file_name] = path

    return node_image_mapping


def plot_digraph_with_images(graph: nx.DiGraph, image_paths: list[str]) -> None:
    """
    Plot a directed graph with images at each node.

    Assumes the old function plot_pathway in the AssemblyConstruction class has been called!
    def plot_pathway(self):
        pic_path = "path_images"
        os.makedirs(pic_path, exist_ok=True)
        for i, vo in enumerate(self.molecules_vo):
            Draw.MolToFile(vo, os.path.join(pic_path, "virtual_object{}.png").format(i))
        for i, step in enumerate(self.molecules_steps):
            Draw.MolToFile(step, os.path.join(pic_path, "step{}.png").format(i + 1))
        return None

    Args:
        graph (nx.DiGraph): The directed graph to be plotted.
        image_paths (list[str]): A list of image file paths to be matched with the nodes.

    Returns:
        None
    """
    # Create a mapping from node to image path
    node_image_mapping = match_node_to_image(graph, image_paths)

    # Add nodes with images
    for n in graph:
        graph.nodes[n]["image"] = mpimg.imread(node_image_mapping[n])

    # Get graph layout
    pos = nx.spring_layout(graph, seed=1734289230)
    # Set up the plot
    fig, ax = plt.subplots()

    # Draw the graph edges
    nx.draw_networkx_edges(graph,
                           pos=pos,
                           ax=ax,
                           min_source_margin=15,
                           min_target_margin=15)

    # Transform from data coordinates (scaled between xlim and ylim) to display coordinates
    tr_figure = ax.transData.transform
    # Transform from display to figure coordinates
    tr_axes = fig.transFigure.inverted().transform
    # Select the size of the image (relative to the X axis)
    icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.06
    icon_center = icon_size / 2.0
    # Add the respective image to each node
    for n in graph.nodes:
        xf, yf = tr_figure(pos[n])
        xa, ya = tr_axes((xf, yf))
        # get overlapped axes and plot icon
        a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
        a.imshow(graph.nodes[n]["image"])
        a.axis("off")

    # Hide axes
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    return None


def _average_angles(angles: np.ndarray) -> float:
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
        edge_color=edge_color,
        node_size=node_size,
        font_size=node_size / 100,
        font_color="black",
        arrowstyle="->",
        arrowsize=arrow_size,
        connectionstyle="arc3,rad=0.2"  # For curved edges
    )

    # Set limits for the plot to accommodate the circles
    ax.set_xlim(-max_ai - 1.5, max_ai + 1.5)
    ax.set_ylim(-max_ai - 1.5, max_ai + 1.5)
    ax.set_aspect("equal", adjustable="datalim")
    plt.savefig(f"{filename}", dpi=600)
    return fig, ax


def plot_assembly_circle(nodes,
                         adj_matrix=None,
                         assembly_indices=None,
                         labels=True,
                         node_size=1000,
                         arrow_size=80,
                         node_color='Skyblue',
                         edge_color='Grey',
                         fig_size=10,
                         filename='assembly_circles.png'):
    # If adj matrix is not provided, the rules_graph will be the output
    if adj_matrix is None:
        graph = CFG.ai_with_pathways(nodes, f_print=False)[2]
        nodes = list(graph.nodes())
        adj_matrix = nx.adjacency_matrix(graph).toarray()

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

    # Plot the network
    fig, ax = _plot_directed_network(nodes,
                                     adj_matrix,
                                     x_positions,
                                     y_positions,
                                     max_ai,
                                     labels,
                                     node_size,
                                     arrow_size,
                                     node_color,
                                     edge_color,
                                     fig_size,
                                     filename)
    return fig, ax
