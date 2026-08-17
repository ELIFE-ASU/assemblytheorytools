import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import platform
import shutil
from matplotlib.patches import FancyArrowPatch

import assemblytheorytools as att


def test_plot_graph():
    """
    Test the plotting of a simple graph.

    This function creates a simple graph from a SMILES string and plots it using
    `att.plot_graph`. It asserts that the figure and axes are created successfully.
    """
    print(flush=True)
    # Create a simple graph
    smi = "C1=CC=CC=C1"
    graph = att.smi_to_nx(smi)
    fig, ax = att.plot_graph(graph)
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."


def test_plot_mol_graph():
    """
    Test the plotting of a molecular graph.

    This function creates a molecular graph from a SMILES string and plots it using
    `att.plot_mol_graph`. It asserts that the figure and axes are created successfully.
    """
    print(flush=True)
    # Create a simple graph
    smi = "C1=CC=CC=C1"
    graph = att.smi_to_nx(smi)
    fig, ax = att.plot_mol_graph(graph)
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."


def test_plot_interactive_graph():
    """
    Test the plotting of an interactive graph.

    This function creates a simple graph from a SMILES string and plots it interactively
    using `att.plot_interactive_graph`. It asserts that the HTML file for the plot
    is generated and then cleans up the created files.
    """
    print(flush=True)
    # Create a simple graph
    smi = "C1=CC=CC=C1"
    graph = att.smi_to_nx(smi)
    att.plot_interactive_graph(graph)
    assert os.path.isfile('interactive_graph.html'), "Failed to generate the file."
    os.remove('interactive_graph.html')
    # remove the folder lib
    if os.path.exists('lib'):
        shutil.rmtree('lib')


def test_plot_digraph():
    """
    Test the plotting of a directed graph.

    This function creates a directed graph with nodes and edges, and then plots it
    using `att.plot_graph`. It asserts that the figure and axes are created successfully.
    """
    print(flush=True)
    # Create a directed graph
    graph = nx.DiGraph()

    # Define nodes and their levels
    nodes = {"CC": 0, "CCC": 1, "CCCCC": 2, "CCCCCCCCC": 3}
    graph.add_nodes_from(nodes)

    # Define edges between nodes
    edges = [("CC", "CCC"), ("CCC", "CCCCC"), ("CCCCC", "CCCCCCCCC")]
    graph.add_edges_from(edges)
    fig, ax = att.plot_graph(graph)
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."


def test_plot_digraph_metro_calc():
    """
    Test the plotting of a directed graph using the metro layout.

    This function calculates the assembly pathway for a molecule (glycine) and plots
    it using the metro layout. It runs only on Linux and asserts that the plot
    files are generated.
    """
    print(flush=True)
    if platform.system().lower() == "linux":
        # Define the SMILES string for glycine
        smi = "C(C(=O)O)N"

        # Convert to Mol object
        mol = att.smi_to_mol(smi)
        # Compute the assembly index and associated data
        _, _, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)
        att.plot_digraph_metro(pathway)
        assert os.path.isfile('metro.png'), "Failed to generate the file."
        assert os.path.isfile('metro.svg'), "Failed to generate the file."
        os.remove('metro.png')
        os.remove('metro.svg')

        # Convert to Graph
        graph = att.smi_to_nx(smi)
        # Compute the assembly index and associated data
        _, _, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)
        att.plot_digraph_metro(pathway)
        assert os.path.isfile('metro.png'), "Failed to generate the file."
        assert os.path.isfile('metro.svg'), "Failed to generate the file."
        os.remove('metro.png')
        os.remove('metro.svg')
    else:
        print("Skipping test_plot_digraph_metro_calc: not running on Linux.", flush=True)


def test_plot_digraph_topological():
    """
    Test the plotting of a directed graph using the topological layout.

    This function calculates the assembly pathway for a molecule (glycine) and plots
    it using the topological layout. It asserts that the figure and axes are created
    successfully.
    """
    print(flush=True)

    # Define the SMILES string for glycine
    smi = "C(C(=O)O)N"

    # Convert the SMILES string to an RDKit Mol object
    mol = att.smi_to_mol(smi)

    # Compute the assembly index and associated data
    _, _, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)

    fig, ax = att.plot_graph(pathway, layout='topological')
    plt.show()

    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."


def test_plot_graph_auto_fig_size():
    """
    Test that `auto_fig_size` scales the figure size with the node count.

    This function plots a small and a much larger graph with `auto_fig_size=True`
    and asserts that the larger graph produces a bigger figure. It also checks
    that `fig_size` is used verbatim when the flag is left at its default.
    """
    print(flush=True)
    small_graph = att.smi_to_nx("CC")
    large_graph = att.smi_to_nx("C" * 30)

    fig_small, _ = att.plot_graph(small_graph, auto_fig_size=True)
    fig_large, _ = att.plot_graph(large_graph, auto_fig_size=True)
    plt.show()

    assert fig_large.get_size_inches()[0] > fig_small.get_size_inches()[0]
    assert fig_large.get_size_inches()[1] > fig_small.get_size_inches()[1]

    # Without the flag, fig_size is used verbatim regardless of node count
    fig_fixed, _ = att.plot_graph(large_graph, fig_size=(9, 4))
    plt.show()
    assert tuple(fig_fixed.get_size_inches()) == (9.0, 4.0)


def test_plot_pathway_mol():
    """
    Test the plotting of an assembly pathway.

    This function calculates the assembly pathway for a molecule (glycine) in different
    representations (mol, graph, atoms) and plots each of them. It asserts that the
    figure and axes are created successfully for each plot type.
    """
    print(flush=True)

    # Define the SMILES string for glycine
    smi = "C(C(=O)O)N"

    # Convert the SMILES string to an RDKit Mol object
    mol = att.smi_to_mol(smi)
    # Compute the assembly index and associated data
    _, _, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)

    fig, ax = att.plot_pathway(pathway, plot_type='mol')
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."

    # Convert the SMILES string to an RDKit Mol object
    graph = att.smi_to_nx(smi)
    # Compute the assembly index and associated data
    _, _, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)

    fig, ax = att.plot_pathway(pathway, plot_type='graph')
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."

    # Compute the assembly index and associated data
    _, _, pathway = att.calculate_assembly_index(mol, strip_hydrogen=False)

    fig, ax = att.plot_pathway(pathway, plot_type='atoms')
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."


def test_plot_pathway_mid_arrow():
    """
    Test the plotting of an assembly pathway with mid-edge arrowheads.

    This function calculates the assembly pathway for a molecule (glycine) and plots
    it with the arrowheads half way along the edges. It asserts that the figure and
    axes are created successfully, and that every edge carries an arrowhead.
    """
    print(flush=True)

    # Define the SMILES string for glycine
    smi = "C(C(=O)O)N"

    # Convert the SMILES string to an RDKit Mol object
    mol = att.smi_to_mol(smi)
    # Compute the assembly index and associated data
    _, _, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)

    fig, ax = att.plot_pathway_mid_arrow(pathway, plot_type='mol')
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."

    # The mid-edge version carries one extra arrow patch per edge, the head that
    # is no longer drawn as part of the edge itself
    _, ax_ref = att.plot_pathway(pathway, plot_type='mol')
    n_arrows = len([p for p in ax.patches if isinstance(p, FancyArrowPatch)])
    n_ref = len([p for p in ax_ref.patches if isinstance(p, FancyArrowPatch)])
    assert n_arrows == n_ref + pathway.number_of_edges(), "Missing mid-edge arrowheads."


def test_plot_pathway_auto_fig_size():
    """
    Test that `auto_fig_size` scales the pathway figure size with the node count.

    This function builds two synthetic chain DAGs of different sizes and asserts
    that the larger one produces a bigger figure when `auto_fig_size=True`.
    """
    print(flush=True)

    def _chain_pathway(n_nodes):
        graph = nx.DiGraph()
        for i in range(n_nodes):
            graph.add_node(i, vo=str(i))
        for i in range(n_nodes - 1):
            graph.add_edge(i, i + 1)
        return graph

    fig_small, _ = att.plot_pathway(_chain_pathway(2), plot_type='string', auto_fig_size=True)
    fig_large, _ = att.plot_pathway(_chain_pathway(40), plot_type='string', auto_fig_size=True)
    plt.show()

    assert fig_large.get_size_inches()[0] > fig_small.get_size_inches()[0]
    assert fig_large.get_size_inches()[1] > fig_small.get_size_inches()[1]


def test_plot_assembly_circle():
    """
    Test the plotting of an assembly pathway in a circular layout.

    This function creates a sample assembly pathway, calculates assembly indices, and
    plots it in a circular layout using `att.plot_assembly_circle`. It asserts that
    the plot file is generated and then cleans it up.
    """
    nodes = ['b', 'a', 'd', 'c', 'ba', 'dc', 'baa', 'bad', 'badc', 'baab', 'baba', 'ddbcd', 'bcdda']
    os.environ["ASS_PATH"] = "/Users/ejanin/Desktop/assemblycpp/assemblyCpp_linux_v5_combined"
    n = len(nodes)
    adj_matrix = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        adj_matrix[i, i + 1] = 1  # Chain
    adj_matrix[0, 4] = 1  # b -> ba (branch)
    adj_matrix[1, 6] = 1  # a -> baa (branch)
    adj_matrix[3, 5] = 1  # c -> dc (branch)

    # Build DiGraph from adjacency and compute assembly indices (depths)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] != 0:
                G.add_edge(nodes[i], nodes[j])

    # Topologically propagate depths: sources -> 0, others -> max(parent_depth)+1
    depth = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        if not preds:
            depth[node] = 0
        else:
            depth[node] = max(depth[p] for p in preds) + 1

    assembly_indices = [depth[node] for node in nodes]

    labels = nodes
    node_size = 1000
    arrow_size = 50
    node_color = 'Skyblue'
    edge_color = 'Grey'
    fig_size = 10
    filename = 'circle_plot.png'

    fig, ax = att.plot_assembly_circle(
        nodes=nodes,
        adj_matrix=adj_matrix,
        assembly_indices=assembly_indices,
        labels=labels,
        node_size=node_size,
        arrow_size=arrow_size,
        node_color=node_color,
        edge_color=edge_color,
        fig_size=fig_size,
        filename=filename
    )

    assert os.path.isfile('circle_plot.png'), "Failed to generate the file."
    os.remove('circle_plot.png')


def test_plot_assembly_circle_auto_fig_size():
    """
    Test that `auto_fig_size` scales the assembly-circle figure with node count.

    This function plots a small and a much larger chain of nodes with
    `auto_fig_size=True` and asserts that the larger one produces a bigger
    figure.
    """
    print(flush=True)

    def _circle_fig(n_nodes):
        nodes = [str(i) for i in range(n_nodes)]
        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
        for i in range(n_nodes - 1):
            adj_matrix[i, i + 1] = 1
        assembly_indices = list(range(n_nodes))
        fig, _ = att.plot_assembly_circle(
            nodes=nodes,
            adj_matrix=adj_matrix,
            assembly_indices=assembly_indices,
            fig_size=10,
            auto_fig_size=True,
        )
        return fig

    fig_small = _circle_fig(4)
    fig_large = _circle_fig(40)

    assert fig_large.get_size_inches()[0] > fig_small.get_size_inches()[0]
    assert fig_large.get_size_inches()[1] > fig_small.get_size_inches()[1]


def test_show_common_bonds():
    """
    Test the visualization of common bonds between two molecules.

    This function visualizes common bonds for two related local structures. PubChem
    lookup behavior is tested separately, so plotting remains deterministic offline.
    """
    print(flush=True)

    labels = ["ethanol", "propanol"]
    img = att.show_common_bonds("CCO", "CCCO", legends=labels)
    assert img is not None, "Failed to generate the image."
    img.show()


def test_draw_mol_grid():
    """
    Test the drawing of a grid of molecules.

    This function takes a list of SMILES strings and draws them in a grid using
    `att.draw_mol_grid`. It asserts that an image is generated.
    """
    print(flush=True)

    mols_str = ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CCI", "CCF", "CC=O"]

    img = att.draw_mol_grid(mols_str, legends=mols_str)
    assert img is not None, "Failed to generate the image."
    img.show()


def test_plot_ase_atoms():
    """
    Test the plotting of an ASE Atoms object.

    This function creates an ASE Atoms object from a SMILES string and plots it using
    `att.plot_ase_atoms`. It asserts that the figure and axes are created successfully.
    """
    print(flush=True)

    smi = "C1=CC=CC=C1"
    atoms = att.smiles_to_atoms(smi)
    fig, ax = att.plot_ase_atoms(atoms)
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."
