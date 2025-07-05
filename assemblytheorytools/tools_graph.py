import os
import random
from typing import Set
from typing import Tuple, List

import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem as Chem

from .tools_mol import safe_standardize_mol, smi_to_mol


def bond_order_assout_to_int(edge_color: str | int) -> int:
    """
    Convert an edge colour to an integer bond order from the Assembly CPP output file

    This function maps a string representation of a bond order (e.g. "single", "double")
    to its corresponding integer value. If the input is already an integer, it returns
    the integer directly.

    Args:
        edge_color (str | int): The edge colour representing the bond order. It can be a string
                                ("single", "double", etc.) or an integer.

    Returns:
        int: The integer representation of the bond order. If the input is a string, it returns
             the corresponding integer value. If the input is already an integer, it returns
             the integer directly.
    """
    edge_color_map = {
        "single": 1,
        "double": 2,
        "triple": 3,
        "quadruple": 4,
        "quintuple": 5,
    }

    if edge_color in edge_color_map:
        return edge_color_map[edge_color]
    else:
        return int(edge_color)


def bond_order_int_to_rdkit(bond_order: int) -> Chem.BondType:
    """
    Convert a bond order int to RDKit's BondType.

    Args:
        bond_order (int): The bond order to convert.

    Returns:
        Chem.BondType: The corresponding RDKit BondType.
    """
    converter = {
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
        4: Chem.rdchem.BondType.QUADRUPLE,
        5: Chem.rdchem.BondType.QUINTUPLE,
        6: Chem.rdchem.BondType.IONIC,
    }
    return converter.get(bond_order, Chem.rdchem.BondType.SINGLE)


def bond_order_rdkit_to_int(bond_type: Chem.BondType) -> int:
    """
    Convert RDKit's BondType to a bond order int.

    Args:
        bond_type (Chem.BondType): The RDKit BondType to convert.

    Returns:
        int: The corresponding bond order int.
    """
    converter = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.QUADRUPLE: 4,
        Chem.rdchem.BondType.QUINTUPLE: 5,
        Chem.rdchem.BondType.IONIC: 6
    }
    return converter.get(bond_type, 1)


def nx_to_mol(graph: nx.Graph, add_hydrogens: bool = True) -> Chem.Mol:
    """
    Convert a NetworkX graph to an RDKit molecule.

    Args:
        graph (nx.Graph): The input NetworkX graph where nodes represent atoms and edges represent bonds.
        add_hydrogens (bool, optional): Whether to add hydrogens to the molecule. Default is True.

    Returns:
        Chem.Mol: The resulting RDKit molecule.
    """
    # Create an editable RDKit molecule
    mol = Chem.RWMol()
    # Dictionary to map node identifiers to atom indices in the RDKit molecule
    node_to_idx = {}

    # Add atoms to the molecule
    for node, data in graph.nodes(data=True):
        # Get the atomic symbol from the node's 'colour' attribute, default to 'C' if not present
        atom_symbol = data.get('color', 'C')
        atom = Chem.Atom(atom_symbol.strip())
        idx = mol.AddAtom(atom)
        node_to_idx[node] = idx

    # Add bonds to the molecule
    for u, v, data in graph.edges(data=True):
        # Get the bond order from the edge's 'color' attribute, default to 1 if not present
        bond_order = int(data.get('color', 1))
        # Map the bond order to RDKit's bond types
        bond_type = bond_order_int_to_rdkit(bond_order)
        # Add the bond to the molecule
        mol.AddBond(node_to_idx[u], node_to_idx[v], bond_type)

    # Sanitise the molecule to generate implicit hydrogens and conformations
    return safe_standardize_mol(mol, add_hydrogens=add_hydrogens)


def mol_to_nx(mol: Chem.Mol, add_hydrogens: bool = True) -> nx.Graph:
    """
    Convert an RDKit molecule to a NetworkX graph.

    Args:
        mol (Chem.Mol): The RDKit molecule to convert.
        add_hydrogens (bool, optional): Whether to keep hydrogen atoms in the graph. Default is True.

    Returns:
        nx.Graph: The resulting NetworkX graph where nodes represent atoms and edges represent bonds.
    """

    mol = safe_standardize_mol(mol, add_hydrogens=add_hydrogens)

    graph = nx.Graph()

    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), color=atom.GetSymbol())

    for bond in mol.GetBonds():
        bond_type = bond_order_rdkit_to_int(bond.GetBondType())
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), color=bond_type)

    if not add_hydrogens:
        graph = remove_hydrogen_from_graph(graph)

    return graph


def remove_hydrogen_from_graph(graph: nx.Graph) -> nx.Graph:
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


def write_ass_graph_file(graph: nx.Graph, file_name: str = "graph_info") -> None:
    """
    Write the graph information to a file for the Assembly CPP calculator.

    Args:
        graph (nx.Graph): The input NetworkX graph where nodes represent atoms and edges represent bonds.
        file_name (str, optional): The name of the file to write the graph information to. Defaults to "graph_info".

    Writes:
        A file containing the graph's name, number of vertices, edges, vertex colours, and edge colours.
    """
    # Get the number of vertices
    num_vertices = graph.number_of_nodes()
    # Get the edges
    edges = list(graph.edges())
    # Get vertex colours
    vertex_colors = nx.get_node_attributes(graph, 'color')
    # Get edge colours
    edge_colors = nx.get_edge_attributes(graph, 'color')

    # Assert that all node colours are strings and do not contain spaces
    for node, color in vertex_colors.items():
        assert isinstance(color, str), f"Node color for node {node} is not a string. Not allowed for Assembly CPP."
        assert ' ' not in color, f"Node color for node {node} contains a space. Not allowed for Assembly CPP."

    # Assert that all edge colours are integers
    for edge, color in edge_colors.items():
        assert isinstance(color, int), f"Edge color for edge {edge} is not an integer. Not allowed for Assembly CPP."

    # Write the information to a file
    with open(file_name, 'w') as f:
        f.write(f"{graph.name}\n")
        f.write(f"{num_vertices}\n")
        f.write(" ".join([f"{e + 1}" for edge in edges for e in edge]) + "\n")
        f.write(" ".join([f"{color}" for node, color in vertex_colors.items()]) + "\n")
        f.write(" ".join([f"{color}" for node, color in edge_colors.items()]))
    return None


def is_graph_isomorphic(g1: nx.Graph, g2: nx.Graph) -> bool:
    """
    Check if two graphs are isomorphic.

    Args:
        g1 (nx.Graph): The first input graph.
        g2 (nx.Graph): The second input graph.

    Returns:
        bool: True if the graphs are isomorphic, False otherwise.
    """
    return nx.is_isomorphic(g1, g2)


def scramble_node_indices(graph: nx.Graph, seed: int | None = None) -> nx.Graph:
    """
    Returns a new graph with randomly scrambled node labels.

    Args:
        graph (nx.Graph): The input graph to be scrambled.
        seed (int | None, optional): Seed for the random number generator for reproducibility. Default is None.

    Returns:
        nx.Graph: A new graph with scrambled node labels.
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


def get_disconnected_subgraphs(graph: nx.Graph) -> List[nx.Graph]:
    """
    Return subgraphs of connected components without copying if not necessary.

    Args:
        graph (nx.Graph): The input graph.

    Returns:
        List[nx.Graph]: A list of subgraphs, each representing a connected component.
    """
    return [graph.subgraph(c) for c in nx.connected_components(graph)]


def write_graphml(graph: nx.Graph, file_name: str = "graph.graphml") -> None:
    """
    Writes a NetworkX graph to a GraphML file.

    Args:
        graph (nx.Graph): The graph to be written to the file.
        file_name (str): The path to the file where the graph will be saved.

    Returns:
        None
    """
    nx.write_graphml_lxml(graph, os.path.abspath(file_name))
    return None


def read_graphml(file_name: str = "graph.graphml") -> nx.Graph:
    """
    Reads a NetworkX graph from a GraphML file.

    Args:
        file_name (str): The path to the file from which the graph will be read.

    Returns:
        nx.Graph: The graph read from the file.
    """
    return nx.read_graphml(os.path.abspath(file_name))


def get_bond_smi(mol: Chem.Mol) -> Set[str]:
    """
    Get the list of bonds of the system in SMILES format.

    Args:
        mol (Chem.Mol): The RDKit molecule object.

    Returns:
        Set[str]: A set of strings representing the bonds in SMILES format.
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


def nx_to_smi(graph: nx.Graph, add_hydrogens: bool = True) -> str:
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


def nx_to_inchi(graph: nx.Graph, add_hydrogens: bool = True) -> str:
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


def create_ionic_molecule(smiles: str) -> Tuple[nx.Graph, List[Chem.Mol]]:
    """
    Create a combined graph for an ionic molecule from dot-separated SMILES.

    Args:
        smiles (str): The SMILES string representing the ionic molecule, with parts separated by dots.

    Returns:
        Tuple[nx.Graph, List[Chem.Mol]]: A tuple containing the combined NetworkX graph and a list of RDKit molecule objects.
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

        combined.add_edge(last_node_prev_graph, first_node_current_graph, color=6)

        # Update offset for the next graph
        offset += graph.number_of_nodes()

    return combined, mols  # Return the combined graph and both molecules


def longest_path_length(digraph: nx.DiGraph) -> int:
    """
    Calculate the longest path length in a Directed Acyclic Graph (DAG).

    Args:
        digraph (nx.DiGraph): The input directed acyclic graph.

    Returns:
        int: The length of the longest path in the graph.

    Raises:
        ValueError: If the input graph is not a Directed Acyclic Graph (DAG).
    """
    if not nx.is_directed_acyclic_graph(digraph):
        raise ValueError("Graph must be a Directed Acyclic Graph (DAG)")

    # Get topological order of nodes
    topological_order = list(nx.topological_sort(digraph))

    # Dictionary to store the longest path distance to each node
    longest_dist: dict[int, int] = {node: 0 for node in digraph.nodes()}

    # Process nodes in topological order
    for node in topological_order:
        for successor in digraph.successors(node):
            longest_dist[successor] = max(longest_dist[successor], longest_dist[node] + 1)

    # Return the maximum path length found
    return max(longest_dist.values(), default=0)


def relabel_digraph(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Relabels the nodes of a directed graph with their topological step.

    This function assigns a "label" attribute to each node in the graph,
    where the label indicates the topological step (generation) of the node
    in a topological sort of the graph.

    Args:
        graph (nx.DiGraph): A directed graph to be relabeled.

    Returns:
        nx.DiGraph: The input graph with nodes relabeled by their topological step.

    Note:
        - The graph must be a directed acyclic graph (DAG) for topological sorting to work.
        - The "label" attribute of each node will be overwritten.
    """
    # Iterate through each topological generation of the graph
    for step, nodes in enumerate(nx.topological_generations(graph)):
        # Assign a label to each node based on its topological step
        for node in nodes:
            graph.nodes[node]["label"] = f"Step {step}"
    return graph


def relabel_identifiers(graph: nx.Graph) -> nx.Graph:
    """
    Relabel the nodes of a NetworkX graph using their "label" attribute.

    This function replaces the current node identifiers in the input graph
    with the values of their "label" attribute. It is useful for creating
    a graph with more meaningful or human-readable node identifiers.

    Args:
        graph (nx.Graph): The input NetworkX graph whose nodes will be relabeled.

    Returns:
        nx.Graph: A new NetworkX graph with nodes relabeled based on their "label" attribute.

    Note:
        - The "label" attribute must exist for all nodes in the graph.
        - If the "label" attribute is not unique, the resulting graph may have issues.
    """
    return nx.relabel_nodes(graph, {n: graph.nodes[n]["label"] for n in graph})


def canonicalize_node_labels(graph: nx.Graph) -> nx.Graph:
    """
    Relabel the nodes of a NetworkX graph to a sequence of integers from 0 to n-1.

    This function ensures that the node labels in the input graph are a contiguous
    sequence of integers starting from 0. This is useful for stabilizing the graph
    structure when the original node labels are non-sequential or arbitrary.

    Args:
        graph (nx.Graph): The input NetworkX graph whose nodes need to be relabeled.

    Returns:
        nx.Graph: A new NetworkX graph with nodes relabeled to a sequence of integers
                  from 0 to n-1.
    """
    # Get the current node labels from the graph
    current_labels = list(graph.nodes())
    # Create a mapping from current labels to new sequential labels
    label_mapping = {current_labels[i]: i for i in range(len(current_labels))}
    # Relabel the graph using the mapping
    graph = nx.relabel_nodes(graph, label_mapping)
    return graph
