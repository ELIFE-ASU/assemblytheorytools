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
    fig, ax = att.plot_graph(graph)
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."


def test_plot_mol_graph():
    print(flush=True)
    # Create a simple graph
    smi = "C1=CC=CC=C1"
    graph = att.smi_to_nx(smi)
    fig, ax = att.plot_mol_graph(graph)
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."


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
    fig, ax = att.plot_graph(graph)
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."


def test_plot_digraph_metro_calc():
    print(flush=True)

    # Define the SMILES string for glycine
    smi = "C(C(=O)O)N"

    # Convert the SMILES string to an RDKit Mol object
    mol = att.smi_to_mol(smi)

    # Compute the assembly index and associated data
    _, _, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)

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

    # Compute the assembly index and associated data
    _, _, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)

    fig, ax = att.plot_graph(pathway, layout='topological')
    plt.show()

    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."


def test_plot_pathway_mol():
    print(flush=True)

    # Define the SMILES string for glycine
    smi = "C(C(=O)O)N"

    # Convert the SMILES string to an RDKit Mol object
    mol = att.smi_to_mol(smi)
    # Compute the assembly index and associated data
    _, _, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)

    fig, ax = att.plot_pathway_mol(pathway)
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."


def test_plot_pathway_graph():
    print(flush=True)

    # Define the SMILES string for glycine
    smi = "C(C(=O)O)N"

    # Convert the SMILES string to an RDKit Mol object
    graph = att.smi_to_nx(smi)
    # Compute the assembly index and associated data
    _, _, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)

    fig, ax = att.plot_pathway_graph(pathway)
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."


def test_plot_pathway_atoms():
    print(flush=True)

    # Define the SMILES string for glycine
    smi = "C(C(=O)O)N"

    # Convert the SMILES string to an RDKit Mol object
    mol = att.smi_to_mol(smi)
    # Compute the assembly index and associated data
    _, _, pathway = att.calculate_assembly_index(mol, strip_hydrogen=False)

    fig, ax = att.plot_pathway_atoms(pathway)
    plt.show()
    assert fig is not None, "Failed to create the figure."
    assert ax is not None, "Failed to create the axes."


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
