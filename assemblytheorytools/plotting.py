import os
from html import escape

import matplotlib.pyplot as plt
import networkx as nx
from IPython.display import HTML
from matplotlib import colormaps, colors
from pyvis.network import Network


def plot_mol_graph(graph, filename, layout="spring"):
    cols_conv = {"C": "grey", "O": "red", "N": "blue", "S": "green", "H": "white", "Cl": "pink", "P": "purple"}
    color_dict_edge = {1.0: "black", 2.0: "green", 3.0: "red", 4.0: "orange"}
    graph_colors = [cols_conv.get(graph.nodes[idx]['color']) for idx in graph.nodes()]
    edge_colors = [color_dict_edge.get(graph.edges[idx]['color']) for idx in graph.edges()]
    # position the nodes
    if layout == "spring":
        pos = nx.spring_layout(graph)
    else:
        pos = nx.kamada_kawai_layout(graph)

    # Draw the graph
    nx.draw(graph,
            pos=pos,
            with_labels=False,
            node_size=100,
            edge_color=edge_colors,
            node_color=graph_colors,
            width=2)
    plt.savefig(filename, dpi=600)
    plt.close()
    return None


def plot_interactive_graph(graph, show=False, filename="interactive_graph.html"):
    # color each node based on its degree
    max_nbr = len(max(graph.adj.values(), key=lambda x: len(x)))
    blues = colormaps.get_cmap("Blues")
    for n, d in graph.nodes(data=True):
        n_neighbors = len(graph.adj[n])
        # show the smaller domain in red and the larger one in blue
        palette = blues
        d["color"] = colors.to_hex(palette(n_neighbors / max_nbr))

    # convert to pyvis network
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
        # save the graph
        net.show(filename)
    return net


def n_plot(xlab, ylab, xs=14, ys=14):
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    plt.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    plt.tick_params(axis='both', which='both', top=True, right=True)
    plt.xlabel(xlab, fontsize=xs)
    plt.ylabel(ylab, fontsize=ys)
    plt.tight_layout()
    return None


def ax_plot(fig, ax, xlab, ylab, xs=14, ys=14):
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    ax.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    ax.tick_params(axis='both', which='both', top=True, right=True)
    ax.set_xlabel(xlab, fontsize=xs)
    ax.set_ylabel(ylab, fontsize=ys)
    fig.tight_layout()
    return None


def os_plot_show():
    # check if the OS is windows
    if os.name == 'nt':
        plt.show()
    else:
        plt.close()
    return None


def plot_residue_graph(graph, f_labs=False, filename="res_graphs"):
    cols_conv = {
        "ALA": "cyan",
        "ARG": "blue",
        "ASN": "green",
        "ASP": "red",
        "CYS": "yellow",
        "GLN": "purple",
        "GLU": "orange",
        "GLY": "pink",
        "HSD": "brown",
        "HSE": "gray",
        "HIS": "azure",
        "ILE": "magenta",
        "LEU": "lime",
        "LYS": "navy",
        "MET": "olive",
        "PHE": "teal",
        "PRO": "maroon",
        "SER": "gold",
        "THR": "silver",
        "TRP": "coral",
        "TYR": "orchid",
        "VAL": "plum",
        "HID": "salmon",
        "HIE": "linen",
        "HIP": "lavender",
        "HSP": "turquoise"
    }
    # Get the colours
    graph_colors = [cols_conv.get(graph.nodes[idx]['color']) for idx in graph.nodes()]

    # get the position
    pos = nx.kamada_kawai_layout(graph)

    # Draw the graph
    nx.draw(graph,
            pos=pos,
            with_labels=f_labs,
            edge_color='grey',
            node_color=graph_colors,
            node_size=50
            )
    plt.savefig(filename + ".png", dpi=600)
    plt.savefig(filename + ".pdf")
    plt.close()
    return None


def plot_graphs_in_subplots(graph_dict, f_labs=False, filename="fragment_graphs"):
    # Find the maximum number of graphs in the lists (i.e., the maximum number of columns)
    max_columns = max(len(graphs) for graphs in graph_dict.values())
    num_rows = len(graph_dict)  # The number of dictionary keys determines the number of rows

    cols_conv = {
        "ALA": "cyan",
        "ARG": "blue",
        "ASN": "green",
        "ASP": "red",
        "CYS": "yellow",
        "GLN": "purple",
        "GLU": "orange",
        "GLY": "pink",
        "HSD": "brown",
        "HSE": "gray",
        "HIS": "azure",
        "ILE": "magenta",
        "LEU": "lime",
        "LYS": "navy",
        "MET": "olive",
        "PHE": "teal",
        "PRO": "maroon",
        "SER": "gold",
        "THR": "silver",
        "TRP": "coral",
        "TYR": "orchid",
        "VAL": "plum",
        "HID": "salmon",
        "HIE": "linen",
        "HIP": "lavender",
        "HSP": "turquoise"
    }

    # Set up a grid of subplots with num_rows rows and max_columns columns
    fig, axes = plt.subplots(num_rows, max_columns, figsize=(2 * max_columns, 2 * num_rows))

    # If there is only one row or column, we make sure to handle it properly as a list
    if num_rows == 1:
        axes = [axes]  # If there's only one row, wrap axes in a list to treat it uniformly
    if max_columns == 1:
        axes = [[ax] for ax in axes]  # If only one column, we need to make each row a list

    # Loop through each key and its associated list of graphs
    for row_idx, (key, graph_list) in enumerate(graph_dict.items()):
        for col_idx in range(max_columns):
            ax = axes[row_idx][col_idx]
            if col_idx < len(graph_list):
                graph = graph_list[col_idx]
                graph_colors = [cols_conv.get(graph.nodes[idx]['color']) for idx in graph.nodes()]
                if col_idx == 0:
                    ax.set_title(f"{key.replace("_", " ")} {col_idx + 1}")
                else:
                    ax.set_title(f"{col_idx + 1}")
                pos = nx.kamada_kawai_layout(graph)
                nx.draw(graph, pos=pos, ax=ax, node_color=graph_colors, edge_color='gray', node_size=50,
                        with_labels=f_labs)
            else:
                # Hide the axes if there's no graph to display in this column
                ax.axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(filename + ".png", dpi=600)
    plt.savefig(filename + ".pdf")
    plt.close()
    return None
