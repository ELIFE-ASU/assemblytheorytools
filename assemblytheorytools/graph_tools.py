import os
import random

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from rdkit import Chem
from rdkit.Chem import AllChem as Chem

from .mol_tools import safe_standardize_mol, smi_to_mol


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

    # Define the bond converter dictionary
    converter = {
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
        4: Chem.rdchem.BondType.QUADRUPLE,
        5: Chem.rdchem.BondType.QUINTUPLE,
        6: Chem.rdchem.BondType.IONIC,
    }

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
        bond_type = converter.get(bond_order, Chem.rdchem.BondType.SINGLE)
        # Add the bond to the molecule
        mol.AddBond(node_to_idx[u], node_to_idx[v], bond_type)

    # Sanitize the molecule to generate implicit hydrogens and conformations
    return safe_standardize_mol(mol, add_hydrogens=add_hydrogens)


def mol_to_nx(mol, add_hydrogens=True):
    """
    Convert an RDKit molecule to a NetworkX graph.

    Args:
        mol (rdkit.Chem.Mol): The RDKit molecule to convert.
        add_hydrogens (bool, optional): Whether to keep hydrogen atoms in the graph. Default is True.

    Returns:
        networkx.Graph: The resulting NetworkX graph where nodes represent atoms and edges represent bonds.
    """
    graph = nx.Graph()
    converter = {Chem.rdchem.BondType.SINGLE: 1,
                 Chem.rdchem.BondType.DOUBLE: 2,
                 Chem.rdchem.BondType.TRIPLE: 3,
                 Chem.rdchem.BondType.QUADRUPLE: 4,
                 Chem.rdchem.BondType.QUINTUPLE: 5,
                 Chem.rdchem.BondType.IONIC: 6}

    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(),
                       color=atom.GetSymbol())

    for bond in mol.GetBonds():
        # Map the bond order to RDKit's bond types
        bond_type = converter.get(bond.GetBondType(), 1)
        # Add the bond to the graph
        graph.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       color=bond_type)
    # Remove the nodes with hydrogen as the color
    if add_hydrogens is False:
        graph = remove_hydrogen_from_graph(graph)
    return graph


def remove_hydrogen_from_graph(graph):
    """
    Remove all hydrogen atoms from a NetworkX graph.

    Args:
        graph (nx.Graph): The input NetworkX graph where nodes represent atoms.

    Returns:
        nx.Graph: The modified graph with all hydrogen atoms removed.
    """
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


def get_bond_smiles(mol):
    """
    Get the list of bonds of the system in SMILES format.

    Args:
        mol (rdkit.Chem.Mol): The RDKit molecule object.

    Returns:
        set: A set of strings representing the bonds in SMILES format.
    """
    bond_smiles = set()
    for bond in mol.GetBonds():
        atom1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
        atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
        symbol1 = atom1.GetSymbol()
        symbol2 = atom2.GetSymbol()
        bond_type = bond.GetBondType()

        if bond_type == Chem.BondType.SINGLE:
            bond_symbol = '-'
        elif bond_type == Chem.BondType.DOUBLE:
            bond_symbol = '='
        elif bond_type == Chem.BondType.TRIPLE:
            bond_symbol = '#'
        else:
            bond_symbol = '~'  # For other bond types

        # Create bond SMILES in alphabetical order
        if symbol1 <= symbol2:
            bond_smiles.add(f"{symbol1}{bond_symbol}{symbol2}")
        else:
            bond_smiles.add(f"{symbol2}{bond_symbol}{symbol1}")

    return bond_smiles


def graph_to_smiles(graph, add_hydrogens=True):
    """
    Convert a NetworkX graph to a SMILES string.

    Args:
        graph (nx.Graph): The input NetworkX graph where nodes represent atoms and edges represent bonds.
        add_hydrogens (bool, optional): Whether to add hydrogens to the molecule. Default is True.

    Returns:
        str: The SMILES string representation of the molecule.
    """
    mol = nx_to_mol(graph, add_hydrogens=add_hydrogens)
    return Chem.MolToSmiles(mol, allHsExplicit=True, kekuleSmiles=True)


def graph_to_inchi(graph, add_hydrogens=True):
    """
    Convert a NetworkX graph to an InChI string.

    Args:
        graph (nx.Graph): The input NetworkX graph where nodes represent atoms and edges represent bonds.
        add_hydrogens (bool, optional): Whether to add hydrogens to the molecule. Default is True.

    Returns:
        str: The InChI string representation of the molecule.
    """
    mol = nx_to_mol(graph, add_hydrogens=add_hydrogens)
    return Chem.MolToInchi(mol)


def create_ionic_molecule(smiles):
    """
    Create a combined graph for an ionic molecule from dot-separated SMILES.

    Args:
        smiles (str): The SMILES string representing the ionic molecule, with parts separated by dots.

    Returns:
        tuple: A tuple containing the combined NetworkX graph and a list of RDKit molecule objects.
    """
    # Split the SMILES at the dot
    parts = smiles.split('.')

    # Convert each part to a molecule and graph
    mols = [smi_to_mol(part) for part in parts]
    graphs = [mol_to_nx(mol) for mol in mols]

    # Start with the first graph
    combined = graphs[0]
    offset = combined.number_of_nodes()  # Starting node index offset for the next molecule

    # Combine graphs, linking last node of previous graph to first of current
    for i, graph in enumerate(graphs[1:], start=1):
        # Add the graph to the combined graph
        combined = nx.disjoint_union(combined, graph)

        # Add an ionic bond between the last node of the previous graph and the first node of the current graph
        last_node_prev_graph = offset - 1  # Last node index of the previous graph
        first_node_current_graph = offset  # First node index of the current graph

        combined.add_edge(last_node_prev_graph, first_node_current_graph, bond_type='ionic')

        # Update offset for the next graph
        offset += graph.number_of_nodes()

    return combined, mols  # Return combined graph and both molecules
