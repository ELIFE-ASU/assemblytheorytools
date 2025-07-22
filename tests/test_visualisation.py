import os
import shutil

import matplotlib.pyplot as plt
import networkx as nx

import assemblytheorytools as att


def test_plot_graph():
    print(flush=True)
    # Create a simple graph
    smi = "C1=CC=CC=C1"
    graph = att.smi_to_nx(smi)
    att.plot_graph(graph)
    assert os.path.isfile('graph.png'), "Failed to generate the file."
    assert os.path.isfile('graph.pdf'), "Failed to generate the file."
    os.remove('graph.png')
    os.remove('graph.pdf')


def test_plot_mol_graph():
    print(flush=True)
    # Create a simple graph
    smi = "C1=CC=CC=C1"
    graph = att.smi_to_nx(smi)
    att.plot_mol_graph(graph)
    assert os.path.isfile('atom_graphs.png'), "Failed to generate the file."
    assert os.path.isfile('atom_graphs.pdf'), "Failed to generate the file."
    os.remove('atom_graphs.png')
    os.remove('atom_graphs.pdf')


def test_plot_interactive_graph():
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
    print(flush=True)
    # Create a directed graph
    graph = nx.DiGraph()

    # Define nodes and their levels
    nodes = {"CC": 0, "CCC": 1, "CCCCC": 2, "CCCCCCCCC": 3}
    graph.add_nodes_from(nodes)

    # Define edges between nodes
    edges = [("CC", "CCC"), ("CCC", "CCCCC"), ("CCCCC", "CCCCCCCCC")]
    graph.add_edges_from(edges)
    att.plot_digraph(graph)
    assert os.path.isfile('digraph.png'), "Failed to generate the file."
    assert os.path.isfile('digraph.pdf'), "Failed to generate the file."
    os.remove('digraph.png')
    os.remove('digraph.pdf')


def test_plot_digraph_metro():
    """
    Test that a metro-style plot of an assembly pathway can be
    generated and saved successfully.

    Workflow:
    - Loads a saved pathway graph from disk using `parse_pathway_file()`.
    - Uses `plot_digraph_metro()` to generate a visual diagram.
    - Saves the plot to 'test.png' and 'test.svg'.
    - Verifies plot generation by removing the files afterward.

    Notes:
    - Expects 'data/pathway/tmpPathway' to be a valid pathway file
      (e.g., from a prior call to `att.save_pathway()`).
    - Passes if no exceptions occur during loading, plotting, or file cleanup.
    """
    print(flush=True)
    # Path to saved pathway file (should be a valid .json or .pkl file)
    pathway_str = "data/pathway/tmpPathway"

    # Load the assembly pathway graph
    digraph = att.parse_pathway_file(pathway_str)
    digraph, _ = digraph  # Unpack the graph and VO list (we only need the graph)

    # Plot the pathway as a metro-style graph and save to files
    att.plot_digraph_metro(digraph)
    assert os.path.isfile('metro.png'), "Failed to generate the file."
    assert os.path.isfile('metro.svg'), "Failed to generate the file."
    os.remove('metro.png')
    os.remove('metro.svg')


def test_plot_digraph_metro_calc():
    print(flush=True)

    # Define the SMILES string for glycine
    smi = "C(C(=O)O)N"

    # Convert the SMILES string to an RDKit Mol object
    mol = att.smi_to_mol(smi)

    # Compute the assembly index and associated data
    _, _, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)

    # Unpack pathway information
    pathway, _ = pathway

    att.plot_digraph_metro(pathway)
    assert os.path.isfile('metro.png'), "Failed to generate the file."
    assert os.path.isfile('metro.svg'), "Failed to generate the file."
    os.remove('metro.png')
    os.remove('metro.svg')


def test_plot_digraph_topological():
    print(flush=True)

    # Define the SMILES string for glycine
    smi = "C(C(=O)O)N"

    # Convert the SMILES string to an RDKit Mol object
    mol = att.smi_to_mol(smi)
    mol = att.smi_to_nx(smi)

    # Compute the assembly index and associated data
    _, _, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)

    # Unpack pathway information
    pathway, _ = pathway
    att.plot_digraph_topological(pathway)
    assert os.path.isfile('topological.png'), "Failed to generate the file."
    assert os.path.isfile('topological.pdf'), "Failed to generate the file."
    os.remove('topological.png')
    os.remove('topological.pdf')


def test_plot_digraph_with_images():
    print(flush=True)

    # Define the SMILES string for glycine
    smi = "C(C(=O)O)N"

    # Convert the SMILES string to an RDKit Mol object
    mol = att.smi_to_mol(smi)
    # Compute the assembly index and associated data
    _, _, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)

    # Unpack pathway information
    pathway, _ = pathway
    att.plot_digraph_with_images(pathway)
    plt.show()
    # assert os.path.isfile('topological.png'), "Failed to generate the file."
    # assert os.path.isfile('topological.pdf'), "Failed to generate the file."
    # os.remove('topological.png')
    # os.remove('topological.pdf')


def test_plot_assembly_circle():
    nodes = ['b', 'a', 'd', 'c', 'ba', 'dc', 'baa', 'bad', 'badc', 'baab', 'baba', 'ddbcd', 'bcdda']

    labels = True
    node_size = 1000
    arrow_size = 50
    node_color = 'Skyblue'
    edge_color = 'Grey'
    fig_size = 10
    filename = 'circle_plot.png'
    att.plot_assembly_circle(nodes,
                             labels=labels,
                             node_size=node_size,
                             arrow_size=arrow_size,
                             node_color=node_color,
                             edge_color=edge_color,
                             fig_size=fig_size,
                             filename=filename)

    assert os.path.isfile('circle_plot.png'), "Failed to generate the file."
    os.remove('circle_plot.png')
