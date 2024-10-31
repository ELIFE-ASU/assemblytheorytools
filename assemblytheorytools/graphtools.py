import os
import random

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from rdkit import Chem
from rdkit.Chem import AllChem as Chem

from .moltools import safe_standardize_mol


def nx_to_mol(graph, add_hydrogens=True):
    """
    Convert a NetworkX graph to an RDKit molecule.

    Args:
        graph (nx.Graph): The input NetworkX graph where nodes represent atoms and edges represent bonds.
        add_hydrogens (bool, optional): Whether to add hydrogens to the molecule. Default is True.

    Returns:
        rdkit.Chem.Mol: The resulting RDKit molecule.
    """
    # Create an editable RDKit molecule
    mol = Chem.RWMol()
    # Dictionary to map node identifiers to atom indices in the RDKit molecule
    node_to_idx = {}

    # Add atoms to the molecule
    for node, data in graph.nodes(data=True):
        # Get the atomic symbol from the node's 'color' attribute, default to 'C' if not present
        atom_symbol = data.get('color', 'C')
        atom = Chem.Atom(atom_symbol)
        idx = mol.AddAtom(atom)
        node_to_idx[node] = idx

    # Add bonds to the molecule
    for u, v, data in graph.edges(data=True):
        # Get the bond order from the edge's 'color' attribute, default to 1 if not present
        bond_order = data.get('color', 1)
        # Map the bond order to RDKit's bond types
        bond_type = {
            1: Chem.rdchem.BondType.SINGLE,
            2: Chem.rdchem.BondType.DOUBLE,
            3: Chem.rdchem.BondType.TRIPLE,
            4: Chem.rdchem.BondType.AROMATIC,
        }.get(bond_order, Chem.rdchem.BondType.SINGLE)
        # Add the bond to the molecule
        mol.AddBond(node_to_idx[u], node_to_idx[v], bond_type)

    # Sanitize the molecule to generate implicit hydrogens and conformations
    return safe_standardize_mol(mol, add_hydrogens=add_hydrogens)


def mol_to_nx(mol, add_hydrogens=True):
    graph = nx.Graph()
    converter = {Chem.rdchem.BondType.SINGLE: 1,
                 Chem.rdchem.BondType.DOUBLE: 2,
                 Chem.rdchem.BondType.TRIPLE: 3,
                 Chem.rdchem.BondType.AROMATIC: 4}

    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(),
                       color=atom.GetSymbol())

    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       color=converter[bond.GetBondType()])
    # Remove the nodes with hydrogen as the color
    if add_hydrogens is False:
        graph = remove_hydrogen_from_graph(graph)
    return graph


def remove_hydrogen_from_graph(graph):
    nodes = list(graph.nodes())
    for node in nodes:
        if graph.nodes[node]["color"] == "H":
            graph.remove_node(node)
    return graph


def write_ass_graph_file(graph, file_name="graph_info"):
    """
    Write the graph information to a file.

    Args:
        graph (nx.Graph): The input NetworkX graph where nodes represent atoms and edges represent bonds.
        file_name (str, optional): The name of the file to write the graph information to. Defaults to "graph_info".

    Writes:
        A file containing the graph's name, number of vertices, edges, vertex colors, and edge colors.
    """
    # Get the number of vertices
    num_vertices = graph.number_of_nodes()
    # Get the edges
    edges = list(graph.edges())
    # Get vertex colors
    vertex_colors = nx.get_node_attributes(graph, 'color')
    # Get edge colors
    edge_colors = nx.get_edge_attributes(graph, 'color')
    # Write the information to a file
    with open(file_name, 'w') as f:
        f.write(f"{graph.name}\n")
        f.write(f"{num_vertices}\n")
        f.write(" ".join([f"{e + 1}" for edge in edges for e in edge]) + "\n")
        f.write(" ".join([f"{color}" for node, color in vertex_colors.items()]) + "\n")
        f.write(" ".join([f"{color}" for node, color in edge_colors.items()]))


def is_graph_isomorphic(g1, g2):
    """
    Check if two graphs are isomorphic.

    Args:
        g1 (nx.Graph): The first input graph.
        g2 (nx.Graph): The second input graph.

    Returns:
        bool: True if the graphs are isomorphic, False otherwise.
    """
    return GraphMatcher(g1, g2).is_isomorphic()


def scramble_node_indices(graph, seed=None):
    """
    Returns a new graph with randomly scrambled node labels.

    Parameters:
    - graph (networkx.Graph): The input graph to be scrambled.
    - seed (int, optional): Seed for the random number generator for reproducibility.

    Returns:
    - networkx.Graph: A new graph with scrambled node labels.
    """
    # Set the random seed if provided for reproducibility
    if seed is not None:
        random.seed(seed)

    # Get the list of nodes and create a shuffled copy
    nodes = list(graph.nodes())
    new_labels = nodes.copy()
    random.shuffle(new_labels)

    # Create a mapping from old labels to new labels
    mapping = dict(zip(nodes, new_labels))

    # Relabel the nodes using the mapping
    graph_scrambled = nx.relabel_nodes(graph, mapping, copy=True)

    return graph_scrambled


def get_disconnected_subgraphs(graph):
    """
    Return subgraphs of connected components without copying if not necessary.

    Args:
        graph (networkx.Graph): The input graph.

    Returns:
        list: A list of subgraphs, each representing a connected component.
    """
    return [graph.subgraph(c) for c in nx.connected_components(graph)]


def write_graph(graph, file_name="graph.graphml"):
    """
    Writes a NetworkX graph to a GraphML file.

    Parameters:
    graph (networkx.Graph): The graph to be written to the file.
    file_name (str): The path to the file where the graph will be saved.

    Returns:
    None
    """
    nx.write_graphml_lxml(graph, os.path.abspath(file_name))
    return None


def read_graph(file_name="graph.graphml"):
    """
    Reads a NetworkX graph from a GraphML file.

    Parameters:
    file_name (str): The path to the file from which the graph will be read.

    Returns:
    networkx.Graph: The graph read from the file.
    """
    return nx.read_graphml(os.path.abspath(file_name))
