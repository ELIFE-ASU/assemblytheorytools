import os
import random
from functools import reduce
from typing import Set
from typing import Tuple, List

import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdchem import GetPeriodicTable

from .tools_mol import safe_standardize_mol, reset_mol_charge, smi_to_mol, inchi_to_mol


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

    https://www.rdkit.org/docs/cppapi/classRDKit_1_1Bond.html

    Args:
        bond_order (int): The bond order to convert.

    Returns:
        Chem.BondType: The corresponding RDKit BondType.

    Raises:
        ValueError: If the bond order is not supported.
    """
    converter = {
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
        4: Chem.rdchem.BondType.QUADRUPLE,
        5: Chem.rdchem.BondType.QUINTUPLE,
        6: Chem.rdchem.BondType.HEXTUPLE,
        7: Chem.rdchem.BondType.ONEANDAHALF,
        8: Chem.rdchem.BondType.TWOANDAHALF,
        9: Chem.rdchem.BondType.THREEANDAHALF,
        10: Chem.rdchem.BondType.FOURANDAHALF,
        11: Chem.rdchem.BondType.FIVEANDAHALF,
        12: Chem.rdchem.BondType.AROMATIC,
        13: Chem.rdchem.BondType.IONIC,
        14: Chem.rdchem.BondType.HYDROGEN,
        15: Chem.rdchem.BondType.THREECENTER,
        16: Chem.rdchem.BondType.DATIVEONE,
        17: Chem.rdchem.BondType.DATIVE,
        18: Chem.rdchem.BondType.DATIVEL,
        19: Chem.rdchem.BondType.DATIVER,
        20: Chem.rdchem.BondType.OTHER,
        21: Chem.rdchem.BondType.ZERO,
    }
    if bond_order not in converter:
        raise ValueError(f"Unsupported bond order: {bond_order}")
    return converter[bond_order]


def bond_order_rdkit_to_int(bond_type: Chem.BondType) -> int:
    """
    Convert RDKit's BondType to a bond order int.

    https://www.rdkit.org/docs/cppapi/classRDKit_1_1Bond.html

    Args:
        bond_type (Chem.BondType): The RDKit BondType to convert.

    Returns:
        int: The corresponding bond order int.

    Raises:
        ValueError: If the bond type is not recognized.
    """
    converter = {
        Chem.rdchem.BondType.UNSPECIFIED: 0,
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.QUADRUPLE: 4,
        Chem.rdchem.BondType.QUINTUPLE: 5,
        Chem.rdchem.BondType.HEXTUPLE: 6,
        Chem.rdchem.BondType.ONEANDAHALF: 7,
        Chem.rdchem.BondType.TWOANDAHALF: 8,
        Chem.rdchem.BondType.THREEANDAHALF: 9,
        Chem.rdchem.BondType.FOURANDAHALF: 10,
        Chem.rdchem.BondType.FIVEANDAHALF: 11,
        Chem.rdchem.BondType.AROMATIC: 12,
        Chem.rdchem.BondType.IONIC: 13,
        Chem.rdchem.BondType.HYDROGEN: 14,
        Chem.rdchem.BondType.THREECENTER: 15,
        Chem.rdchem.BondType.DATIVEONE: 16,
        Chem.rdchem.BondType.DATIVE: 17,
        Chem.rdchem.BondType.DATIVEL: 18,
        Chem.rdchem.BondType.DATIVER: 19,
        Chem.rdchem.BondType.OTHER: 20,
        Chem.rdchem.BondType.ZERO: 21
    }
    if bond_type not in converter:
        raise ValueError(f"Unsupported RDKit BondType: {bond_type}")
    return converter[bond_type]


def nx_to_mol(graph: nx.Graph,
              add_hydrogens: bool = True,
              sanitize: bool = True,
              reset_charge: bool = False) -> Chem.Mol:
    """
    Convert a NetworkX graph to an RDKit molecule object.

    This function creates an RDKit molecule (`Chem.Mol`) from a NetworkX graph representation.
    Nodes in the graph represent atoms, and edges represent bonds. The graph must have specific
    attributes for nodes and edges to define atomic symbols and bond orders.

    Parameters:
    -----------
    graph : nx.Graph
        A NetworkX graph where nodes represent atoms and edges represent bonds. Each node must
        have a 'color' attribute indicating the atomic symbol, and each edge must have a 'color'
        attribute indicating the bond order (as an integer).
    add_hydrogens : bool, optional
        If True, adds explicit hydrogens to the molecule during sanitization. Defaults to True.
    sanitize : bool, optional
        If True, sanitizes the molecule after creation. Defaults to True.
    reset_charge : bool, optional
        If True, recalculates the formal charges of the atoms in the molecule. Defaults to False.

    Returns:
    --------
    Chem.Mol
        An RDKit molecule object created from the input graph.

    Raises:
    -------
    KeyError
        If a node is missing the 'color' attribute or an edge is missing the 'color' attribute.

    Notes:
    ------
    - The 'color' attribute of nodes is used to determine the atomic symbol.
    - The 'color' attribute of edges is used to determine the bond order.
    - The molecule can be sanitized and charges recalculated based on the input parameters.
    """
    # Create an editable RDKit molecule
    mol = Chem.RWMol()
    # Dictionary to map node identifiers to atom indices in the RDKit molecule
    node_to_idx = {}

    # Add atoms to the molecule
    for node, data in graph.nodes(data=True):
        # Get the atomic symbol from the node's 'color' attribute, raise error if not present
        if 'color' not in data:
            raise KeyError(f"Node {node} is missing the 'color' attribute.")
        atom_symbol = data['color']
        atom = Chem.Atom(atom_symbol.strip())
        node_to_idx[node] = mol.AddAtom(atom)

    # Add bonds to the molecule
    for u, v, data in graph.edges(data=True):
        # Get the bond order from the edge's 'color' attribute, raise error if not present
        if 'color' not in data:
            raise KeyError(f"Edge ({u}, {v}) is missing the 'color' attribute.")
        bond_order = int(data['color'])
        # Map the bond order to RDKit's bond types
        bond_type = bond_order_int_to_rdkit(bond_order)
        # Add the bond to the molecule
        mol.AddBond(node_to_idx[u], node_to_idx[v], bond_type)

    # Sanitise the molecule
    if sanitize:
        mol = safe_standardize_mol(mol, add_hydrogens=add_hydrogens)
    # Re-calculate the charges if requested
    if reset_charge:
        mol = reset_mol_charge(mol)
    return mol


def mol_to_nx(mol: Chem.Mol,
              add_hydrogens: bool = True,
              sanitize: bool = True) -> nx.Graph:
    """
    Convert an RDKit molecule object to a NetworkX graph.

    This function creates a NetworkX graph representation of a molecule. Nodes in the graph
    represent atoms, and edges represent bonds. The graph includes attributes for nodes and
    edges to define atomic symbols and bond types.

    Parameters:
    -----------
    mol : Chem.Mol
        An RDKit molecule object to be converted into a NetworkX graph.
    add_hydrogens : bool, optional
        If True, adds explicit hydrogens to the molecule during sanitization. Defaults to True.
    sanitize : bool, optional
        If True, sanitizes the molecule before conversion. Defaults to True.

    Returns:
    --------
    nx.Graph
        A NetworkX graph where nodes represent atoms and edges represent bonds. Node attributes
        include 'color' for atomic symbols, and edge attributes include 'color' for bond types.

    Notes:
    ------
    - The 'color' attribute of nodes corresponds to the atomic symbol (e.g., "C" for carbon).
    - The 'color' attribute of edges corresponds to the bond type as an integer.
    - The graph's node labels are canonicalized to ensure sequential integer labels.
    """
    if sanitize:
        mol = safe_standardize_mol(mol, add_hydrogens=add_hydrogens)

    graph = nx.Graph()

    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), color=atom.GetSymbol())

    for bond in mol.GetBonds():
        bond_type = bond_order_rdkit_to_int(bond.GetBondType())
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), color=bond_type)

    return canonicalize_node_labels(graph)


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


def join_graphs(graphs: List[nx.Graph], disjoint: int = True, rename_prefix: str = "G") -> nx.Graph:
    """
    Combine multiple NetworkX graphs into a single graph.

    This function merges a list of NetworkX graphs into one. It supports two modes:
    - Disjoint union: Ensures no node ID clashes by creating separate components for each graph.
    - Composition: Combines graphs while preserving node IDs, unless conflicts are detected.

    Args:
        graphs (List[nx.Graph]): A list of NetworkX graphs to be combined.
        disjoint (int, optional): If True, performs a disjoint union of the graphs.
                                  If False, attempts to compose the graphs. Default is True.
        rename_prefix (str, optional): Prefix used for relabeling nodes in case of conflicts. Default is "G".

    Returns:
        nx.Graph: The combined graph.

    Raises:
        ValueError: If the input list of graphs is empty.
        TypeError: If the graphs are not of the same NetworkX type.

    Note:
        - When `disjoint` is False, node ID clashes are checked. If clashes exist, nodes are relabeled with a prefix.
    """
    graphs = list(graphs)  # Convert the input iterable to a list
    if not graphs:
        raise ValueError("Need at least one graph.")

        # Ensure all inputs have the same concrete class
    first_type = type(graphs[0])
    if any(type(g) is not first_type for g in graphs[1:]):
        raise TypeError("All graphs must be of the same NetworkX type.")

    # Perform a disjoint union of the graphs
    if disjoint:
        return reduce(nx.disjoint_union, graphs)

    # Check for node ID clashes
    node_sets = [set(g) for g in graphs]
    if all(node_sets[i].isdisjoint(node_sets[j])
           for i in range(len(node_sets)) for j in range(i + 1, len(node_sets))):
        # Compose graphs directly if no clashes are detected
        return nx.compose_all(graphs)

    # Relabel nodes with a prefix to avoid clashes
    relabelled = []
    for i, g in enumerate(graphs):
        # Add prefix to node labels
        prefix = f"{rename_prefix}{i}_"
        relabelled.append(nx.relabel_nodes(g, lambda n, p=prefix: f"{p}{n}"))
    # Compose the relabeled graphs
    return nx.compose_all(relabelled)


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


def nx_to_smi(graph: nx.Graph, add_hydrogens: bool = True, sanitize: bool = True) -> str:
    """
    Convert a NetworkX graph to a SMILES string.

    This function generates a SMILES (Simplified Molecular Input Line Entry System) representation
    of a molecule from its NetworkX graph representation. The graph is first converted to an RDKit
    molecule object, and then the SMILES string is generated.

    Parameters:
    -----------
    graph : nx.Graph
        A NetworkX graph where nodes represent atoms and edges represent bonds. Each node must
        have a 'color' attribute indicating the atomic symbol, and each edge must have a 'color'
        attribute indicating the bond order (as an integer).
    add_hydrogens : bool, optional
        If True, adds explicit hydrogens to the molecule during sanitization. Defaults to True.
    sanitize : bool, optional
        If True, sanitizes the molecule before generating the SMILES string. Defaults to True.

    Returns:
    --------
    str
        A SMILES string representing the molecule.

    Notes:
    ------
    - The SMILES string is generated with explicit hydrogens and Kekulé form if specified.
    - The 'color' attribute of nodes and edges in the graph is used to define atomic symbols
      and bond orders, respectively.
    """
    mol = nx_to_mol(graph, add_hydrogens=add_hydrogens, sanitize=sanitize)
    return Chem.MolToSmiles(mol, allHsExplicit=False, kekuleSmiles=True)


def smi_to_nx(smiles: str, add_hydrogens: bool = True, sanitize: bool = True) -> nx.Graph:
    """
    Convert a SMILES string to a NetworkX graph.

    This function takes a SMILES (Simplified Molecular Input Line Entry System) string,
    converts it to an RDKit molecule object, and then transforms it into a NetworkX graph
    representation. The graph includes attributes for nodes and edges to define atomic
    symbols and bond types.

    Parameters:
    -----------
    smiles : str
        A SMILES string representing the molecular structure.
    add_hydrogens : bool, optional
        If True, adds explicit hydrogens to the molecule during sanitization. Defaults to True.
    sanitize : bool, optional
        If True, sanitizes the molecule before conversion. Defaults to True.

    Returns:
    --------
    nx.Graph
        A NetworkX graph where nodes represent atoms and edges represent bonds. Node attributes
        include 'color' for atomic symbols, and edge attributes include 'color' for bond types.

    Raises:
    -------
    ValueError
        If the SMILES string is invalid or the conversion to an RDKit molecule fails.

    Notes:
    ------
    - The 'color' attribute of nodes corresponds to the atomic symbol (e.g., "C" for carbon).
    - The 'color' attribute of edges corresponds to the bond type as an integer.
    """
    mol = smi_to_mol(smiles, add_hydrogens=add_hydrogens, sanitize=sanitize)
    if mol is None:
        raise ValueError("Invalid SMILES string or conversion failed.")
    return mol_to_nx(mol, add_hydrogens=add_hydrogens, sanitize=sanitize)


def nx_to_inchi(graph: nx.Graph, add_hydrogens: bool = True, sanitize: bool = True) -> str:
    """
    Convert a NetworkX graph to an InChI string.

    This function generates an InChI (International Chemical Identifier) representation
    of a molecule from its NetworkX graph representation. The graph is first converted
    to an RDKit molecule object, and then the InChI string is generated.

    Parameters:
    -----------
    graph : nx.Graph
        A NetworkX graph where nodes represent atoms and edges represent bonds. Each node must
        have a 'color' attribute indicating the atomic symbol, and each edge must have a 'color'
        attribute indicating the bond order (as an integer).
    add_hydrogens : bool, optional
        If True, adds explicit hydrogens to the molecule during sanitization. Defaults to True.
    sanitize : bool, optional
        If True, sanitizes the molecule before generating the InChI string. Defaults to True.

    Returns:
    --------
    str
        An InChI string representing the molecule.

    Notes:
    ------
    - The 'color' attribute of nodes and edges in the graph is used to define atomic symbols
      and bond orders, respectively.
    - Sanitization ensures the molecule is chemically valid before conversion.
    """
    mol = nx_to_mol(graph, add_hydrogens=add_hydrogens, sanitize=sanitize)
    return Chem.MolToInchi(mol)


def inchi_to_nx(inchi: str, add_hydrogens: bool = False, sanitize: bool = True) -> nx.Graph:
    """
    Convert an InChI string to a NetworkX graph.

    This function takes an InChI (International Chemical Identifier) string,
    converts it to an RDKit molecule object, and then transforms it into a
    NetworkX graph representation. The graph includes attributes for nodes
    and edges to define atomic symbols and bond types.

    Parameters:
    -----------
    inchi : str
        An InChI string representing the molecular structure.
    add_hydrogens : bool, optional
        If True, adds explicit hydrogens to the molecule during sanitization. Defaults to True.
    sanitize : bool, optional
        If True, sanitizes the molecule before conversion. Defaults to True.

    Returns:
    --------
    nx.Graph
        A NetworkX graph where nodes represent atoms and edges represent bonds. Node attributes
        include 'color' for atomic symbols, and edge attributes include 'color' for bond types.

    Raises:
    -------
    ValueError
        If the InChI string is invalid or the conversion to an RDKit molecule fails.

    Notes:
    ------
    - The 'color' attribute of nodes corresponds to the atomic symbol (e.g., "C" for carbon).
    - The 'color' attribute of edges corresponds to the bond type as an integer.
    """
    mol = inchi_to_mol(inchi)  # Convert the InChI string to an RDKit molecule object
    if mol is None:
        raise ValueError("Invalid InChI string or conversion failed.")
    return mol_to_nx(mol, add_hydrogens=add_hydrogens, sanitize=sanitize)


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


def get_graph_charges(graph: nx.Graph,
                      pt: Chem.rdchem.PeriodicTable = None) -> List[int]:
    """
    Calculate the formal charges of nodes in a NetworkX graph.

    This function computes the formal charge for each node in the graph based on its atomic symbol
    and the sum of the edge colours (bond orders). The periodic table is used to retrieve atomic
    properties such as valence.

    Parameters:
    -----------
    graph : nx.Graph
        A NetworkX graph where nodes represent atoms and edges represent bonds. Each node must have
        a 'color' attribute indicating the atomic symbol, and each edge must have a 'color' attribute
        indicating the bond order (as an integer).
    pt : Chem.rdchem.PeriodicTable, optional
        The RDKit periodic table object. If not provided, it defaults to the global periodic table.

    Returns:
    --------
    List[int]
        A list of integers representing the formal charges of the nodes in the graph.

    Notes:
    ------
    - The 'color' attribute of each node is expected to contain the atomic symbol (e.g., "C" for carbon).
    - The formal charge is calculated as the difference between the atom's valence and the sum of the
      edge colours (bond orders) for that node.
    """
    pt = pt or GetPeriodicTable()
    charges = []
    for node_id, node_data in graph.nodes(data=True):
        symbol = node_data['color']
        atomic_number = pt.GetAtomicNumber(symbol)
        valence = min(pt.GetValenceList(atomic_number))
        # Sum the edge colours (bond orders) for this node
        bond_order_sum = sum(
            graph.edges[node_id, neighbor].get('color', 1)
            for neighbor in graph.neighbors(node_id)
        )
        charge = valence - bond_order_sum
        charges.append(charge)
    return charges
