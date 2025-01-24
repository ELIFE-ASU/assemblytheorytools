import os
from html import escape

import matplotlib.pyplot as plt
import networkx as nx
from IPython.display import HTML
from matplotlib import colormaps, colors
from pyvis.network import Network


def n_plot(xlab, ylab, xs=14, ys=14):
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


def ax_plot(fig, ax, xlab, ylab, xs=14, ys=14):
    """
    Configure and style the plot with specified labels and tick parameters for a given axis.

    Use with
    plt.rcParams['axes.linewidth'] = 2.0

    Args:
        fig (matplotlib.figure.Figure): The figure object containing the plot.
        ax (matplotlib.axes.Axes): The axes object to be styled.
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


def os_plot_show():
    """
    Display or close the plot based on the operating system.

    If the operating system is Windows, the plot will be displayed using plt.show().
    Otherwise, the plot will be closed using plt.close().

    Args:
        None

    Returns:
        None
    """
    # check if the OS is windows
    if os.name == 'nt':
        plt.show()
    else:
        plt.close()
    return None


def plot_graph(graph,
               layout='kawai',
               f_labs=False,
               edge_color='grey',
               node_size=300,
               edgecolors="black",
               width=2,
               linewidths=2,
               filename="graph",
               seed=1734289230):
    """
    Plot a graph using NetworkX and Matplotlib with various layout options.

    Args:
        graph (networkx.Graph): The graph to be plotted.
        layout (str, optional): The layout algorithm to use for positioning nodes. Default is 'kawai'.
        f_labs (bool, optional): Whether to display labels on the nodes. Default is False.
        edge_color (str, optional): Color of the edges. Default is 'grey'.
        node_size (int, optional): Size of the nodes. Default is 300.
        edgecolors (str, optional): Color of the node borders. Default is 'black'.
        width (int, optional): Width of the edges. Default is 2.
        linewidths (int, optional): Width of the node borders. Default is 2.
        filename (str, optional): Base name of the file where the plot will be saved. Default is 'graph'.
        seed (int, optional): Seed for the layout algorithm (if applicable). Default is 1734289230.

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


def plot_mol_graph(graph, f_labs=False, filename="atom_graphs"):
    """
    Plot a molecular graph using NetworkX and Matplotlib.

    Args:
        graph (networkx.Graph): The graph to be plotted where nodes represent atoms and edges represent bonds.
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


def plot_interactive_graph(graph, show=False, filename="interactive_graph.html"):
    """
    Plot an interactive graph using PyVis and display it in a Jupyter notebook or save it as an HTML file.

    Args:
        graph (networkx.Graph): The graph to be plotted.
        show (bool, optional): Whether to display the graph in a Jupyter notebook. Default is False.
        filename (str, optional): The name of the file where the graph will be saved if not displayed.
        Default is "interactive_graph.html".

    Returns:
        pyvis.network.Network: The PyVis network object representing the graph.
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


def plot_digraph(digraph,
                 layout='spring',
                 f_labs=True,
                 edge_color='grey',
                 node_size=300,
                 edgecolors="black",
                 width=2,
                 linewidths=2,
                 filename="digraph",
                 seed=1734289230):
    """
    Plot a directed graph using NetworkX and Matplotlib with various layout options.

    Args:
        digraph (networkx.DiGraph): The directed graph to be plotted.
        layout (str, optional): The layout algorithm to use for positioning nodes. Default is 'spring'.
        f_labs (bool, optional): Whether to display labels on the nodes. Default is True.
        edge_color (str, optional): Color of the edges. Default is 'grey'.
        node_size (int, optional): Size of the nodes. Default is 300.
        edgecolors (str, optional): Color of the node borders. Default is 'black'.
        width (int, optional): Width of the edges. Default is 2.
        linewidths (int, optional): Width of the node borders. Default is 2.
        filename (str, optional): Base name of the file where the plot will be saved. Default is 'digraph'.
        seed (int, optional): Seed for the layout algorithm (if applicable). Default is 1734289230.

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


def plot_digraph_metro():
    return None
