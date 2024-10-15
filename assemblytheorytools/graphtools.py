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


def mol_to_nx(mol):
    """
    Convert an RDKit molecule to a NetworkX graph.

    Args:
        mol (Chem.Mol): The input RDKit molecule.

    Returns:
        nx.Graph: The resulting NetworkX graph where nodes represent atoms and edges represent bonds.
    """
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


def scramble_node_indices(graph):
    """
    Randomly scramble the node indices of a NetworkX graph.

    Args:
        graph (networkx.Graph): The input graph whose node indices will be scrambled.

    Returns:
        networkx.Graph: A new graph with scrambled node indices.
    """
    # Create a copy of the original graph to avoid modifying it
    scrambled_graph = graph.copy()
    # Get the list of original node indices
    nodes = list(scrambled_graph.nodes())
    # Create a list of new node indices by shuffling the original indices
    random.shuffle(nodes)
    # Create a mapping from original node indices to new node indices
    mapping = dict(zip(scrambled_graph.nodes(), nodes))
    # Relabel the nodes in the graph using the mapping
    return nx.relabel_nodes(scrambled_graph, mapping)
