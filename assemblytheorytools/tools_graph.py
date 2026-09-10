"""
Conversion and manipulation of molecular graphs.

This module converts between NetworkX graphs and RDKit molecules, SMILES and
InChI strings, and the edge-list format consumed by the assembly calculators. It
also provides graph manipulation utilities: hydrogen stripping, subgraph
extraction, joining and composition, node relabelling and canonicalisation,
charge assignment, and GraphML serialisation.
"""

import os
import random
from functools import reduce
from typing import Iterable, List, Set, Tuple, Union

import networkx as nx
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdchem import GetPeriodicTable

from .tools_mol import inchi_to_mol, reset_mol_charge, safe_standardize_mol, smi_to_mol

_EDGE_COLOR_TO_BOND_ORDER = {
    "single": 1,
    "double": 2,
    "triple": 3,
    "quadruple": 4,
    "quintuple": 5,
}

_BOND_ORDER_TO_RDKIT_TYPE = {
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

_RDKIT_TYPE_TO_BOND_ORDER = {
    Chem.rdchem.BondType.UNSPECIFIED: 0,
    **{bond_type: order for order, bond_type in _BOND_ORDER_TO_RDKIT_TYPE.items()},
}

_BOND_TYPE_TO_SMI_SYMBOL = {
    Chem.BondType.SINGLE: "-",
    Chem.BondType.DOUBLE: "=",
    Chem.BondType.TRIPLE: "#",
}


def bond_order_assout_to_int(edge_color: str | int) -> int:
    """
    Convert a parallelassemblycpp edge colour to an integer bond order.

    Parameters
    ----------
    edge_color : str or int
        A bond name ("single" through "quintuple") or an integer-like value.

    Returns
    -------
    int
        The named bond order, or the result of ``int(edge_color)``.

    Raises
    ------
    ValueError
        If the value is neither a supported name nor an integer string.
    """
    if edge_color in _EDGE_COLOR_TO_BOND_ORDER:
        return _EDGE_COLOR_TO_BOND_ORDER[edge_color]
    return int(edge_color)


def bond_order_int_to_rdkit(bond_order: int) -> Chem.BondType:
    """
    Convert a bond order int to RDKit's BondType.

    Parameters
    ----------
    bond_order : int
        The bond order to convert.

    Returns
    -------
    Chem.BondType
        The corresponding RDKit BondType.

    Raises
    ------
    ValueError
        If the bond order is not supported.

    References
    ----------
    RDKit ``Bond`` class documentation:
    https://www.rdkit.org/docs/cppapi/classRDKit_1_1Bond.html
    """
    if bond_order not in _BOND_ORDER_TO_RDKIT_TYPE:
        raise ValueError(f"Unsupported bond order: {bond_order}")
    return _BOND_ORDER_TO_RDKIT_TYPE[bond_order]


def bond_order_rdkit_to_int(bond_type: Chem.BondType) -> int:
    """
    Convert RDKit's BondType to a bond order int.

    Parameters
    ----------
    bond_type : Chem.BondType
        The RDKit BondType to convert.

    Returns
    -------
    int
        The corresponding bond order int.

    Raises
    ------
    ValueError
        If the bond type is not recognized.

    References
    ----------
    RDKit ``Bond`` class documentation:
    https://www.rdkit.org/docs/cppapi/classRDKit_1_1Bond.html
    """
    if bond_type not in _RDKIT_TYPE_TO_BOND_ORDER:
        raise ValueError(f"Unsupported RDKit BondType: {bond_type}")
    return _RDKIT_TYPE_TO_BOND_ORDER[bond_type]


def nx_to_mol(
    graph: nx.Graph,
    add_hydrogens: bool = True,
    sanitize: bool = True,
    reset_charge: bool = False,
) -> Chem.Mol:
    """
    Convert a molecular graph to an RDKit molecule in node iteration order.

    Parameters
    ----------
    graph : nx.Graph
        Nodes require an atomic symbol in ``color``; edges require an
        integer bond order in ``color``. Node identifiers may be arbitrary.
    add_hydrogens : bool, optional
        Add explicit hydrogens during sanitization. Default is True.
    sanitize : bool, optional
        Standardize the molecule after construction. Default is True.
    reset_charge : bool, optional
        Recalculate formal charges after sanitization. Default is False.

    Returns
    -------
    Chem.Mol
        The converted molecule. The input graph is unchanged.

    Raises
    ------
    KeyError
        If a node or edge is missing its ``color`` attribute.
    ValueError
        If an edge colour is not a supported integer bond order.
    """
    mol = Chem.RWMol()
    node_to_idx = {}

    for node, data in graph.nodes(data=True):
        if "color" not in data:
            raise KeyError(f"Node {node} is missing the 'color' attribute.")
        node_to_idx[node] = mol.AddAtom(Chem.Atom(data["color"].strip()))

    for u, v, data in graph.edges(data=True):
        if "color" not in data:
            raise KeyError(f"Edge ({u}, {v}) is missing the 'color' attribute.")
        bond_type = bond_order_int_to_rdkit(int(data["color"]))
        mol.AddBond(node_to_idx[u], node_to_idx[v], bond_type)

    if sanitize:
        mol = safe_standardize_mol(mol, add_hydrogens=add_hydrogens)
    if reset_charge:
        mol = reset_mol_charge(mol)
    return mol


def mol_to_nx(
    mol: Chem.Mol, add_hydrogens: bool = True, sanitize: bool = True
) -> nx.Graph:
    """
    Convert an RDKit molecule to a graph with consecutive atom indices.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to convert. Standardization may modify it in place.
    add_hydrogens : bool, optional
        Add explicit hydrogens during sanitization. Default is True.
    sanitize : bool, optional
        Standardize the molecule before conversion. Default is True.

    Returns
    -------
    nx.Graph
        A new graph with node ``color`` attributes holding atomic symbols
        and edge ``color`` attributes holding integer bond orders. RDKit's
        atom indices provide consecutive node labels starting at 0.
    """
    if sanitize:
        mol = safe_standardize_mol(mol, add_hydrogens=add_hydrogens)

    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), color=atom.GetSymbol())
    for bond in mol.GetBonds():
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            color=bond_order_rdkit_to_int(bond.GetBondType()),
        )
    return graph


def remove_hydrogen_from_graph(graph: nx.Graph) -> nx.Graph:
    """
    Remove all hydrogen atoms from a NetworkX graph.

    Parameters
    ----------
    graph : nx.Graph
        The input NetworkX graph where nodes represent atoms.

    Returns
    -------
    nx.Graph
        A new graph with all hydrogen atoms removed. The input graph is left
        unchanged.

    Notes
    -----
    Surviving nodes keep their original identifiers. Use
    :func:`canonicalize_node_labels` for consecutive labels starting at 0,
    as the assembly calculators require.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> graph = att.smi_to_nx("CCO")
    >>> stripped = att.remove_hydrogen_from_graph(graph)
    >>> stripped.number_of_nodes(), stripped.number_of_edges()
    (3, 2)

    The input is not modified, so it can be reused:

    >>> graph.number_of_nodes()
    9
    """
    stripped = graph.copy()
    stripped.remove_nodes_from(
        node for node, data in graph.nodes(data=True) if data["color"] == "H"
    )
    return stripped


def write_ass_graph_file(graph: nx.Graph, file_name: str = "graph_info") -> None:
    """
    Write a graph in the edge-list format used by parallelassemblycpp.

    Parameters
    ----------
    graph : nx.Graph
        The graph to write. Node identifiers must support adding 1;
        calculators expect consecutive integer labels starting at 0.
    file_name : str, optional
        Destination path. Default is "graph_info".

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If a supplied node colour is not a string without spaces, or a
        supplied edge colour is not an integer.

    Notes
    -----
    The five lines contain the name, node count, one-based edge endpoints,
    node colours and edge colours, in graph iteration order. No extra
    newline is appended after the final line.
    """
    vertex_colors = nx.get_node_attributes(graph, "color")
    edge_colors = nx.get_edge_attributes(graph, "color")

    for node, color in vertex_colors.items():
        assert isinstance(color, str), (
            f"Node color for node {node} is not a string. Not allowed for parallelassemblycpp."
        )
        assert " " not in color, (
            f"Node color for node {node} contains a space. Not allowed for parallelassemblycpp."
        )

    for edge, color in edge_colors.items():
        assert isinstance(color, int), (
            f"Edge color for edge {edge} is not an integer. Not allowed for parallelassemblycpp."
        )

    with open(file_name, "w") as file:
        file.write(f"{graph.name}\n{graph.number_of_nodes()}\n")
        file.write(
            " ".join(f"{node + 1}" for edge in graph.edges() for node in edge) + "\n"
        )
        file.write(" ".join(f"{color}" for color in vertex_colors.values()) + "\n")
        file.write(" ".join(f"{color}" for color in edge_colors.values()))


def is_graph_isomorphic(g1: nx.Graph, g2: nx.Graph) -> bool:
    """
    Check if two graphs are isomorphic.

    Parameters
    ----------
    g1 : nx.Graph
        The first input graph.
    g2 : nx.Graph
        The second input graph.

    Returns
    -------
    bool
        True if the graphs are isomorphic, False otherwise.

    Notes
    -----
    This helper checks topology only. It does not compare node or edge
    attributes such as ``color``; use NetworkX match functions when those
    attributes are significant.

    Examples
    --------

    >>> import assemblytheorytools as att
    >>> graph = att.smi_to_nx("CCO")
    >>> att.is_graph_isomorphic(graph, att.scramble_node_indices(graph))
    True
    """
    return nx.is_isomorphic(g1, g2)


def scramble_node_indices(graph: nx.Graph, seed: int | None = None) -> nx.Graph:
    """
    Return a copy of a graph with its existing node labels shuffled.

    Parameters
    ----------
    graph : nx.Graph
        The graph to relabel.
    seed : int, optional
        Seed Python's global random generator before shuffling. Default
        is None, which uses its current state.

    Returns
    -------
    nx.Graph
        A relabelled copy preserving the original set of node identifiers
        and the graph's attributes.
    """
    if seed is not None:
        random.seed(seed)

    nodes = list(graph)
    new_labels = nodes.copy()
    random.shuffle(new_labels)
    return nx.relabel_nodes(graph, dict(zip(nodes, new_labels)))


def get_disconnected_subgraphs(graph: nx.Graph) -> List[nx.Graph]:
    """
    Return a view of each connected component, in discovery order.

    Parameters
    ----------
    graph : nx.Graph
        An undirected graph.

    Returns
    -------
    List[nx.Graph]
        Subgraph views sharing attributes with the input. Their structure
        is read-only and reflects changes to the original graph.
    """
    return [graph.subgraph(nodes) for nodes in nx.connected_components(graph)]


def join_graphs(
    graphs: List[nx.Graph], disjoint: int = True, rename_prefix: str = "G"
) -> nx.Graph:
    """
    Combine graphs, separating any clashing node identifiers.

    Parameters
    ----------
    graphs : List[nx.Graph]
        Graphs of the same concrete NetworkX class; iterables are accepted.
    disjoint : int, optional
        If True (the default), use a disjoint union with consecutive
        integer labels. If False, preserve labels unless any overlap.
    rename_prefix : str, optional
        Prefix for conflicting labels. Default is "G". When any overlap
        exists, every label becomes "{rename_prefix}{graph_index}_{node}".

    Returns
    -------
    nx.Graph
        The combined graph, with later graphs taking precedence for graph
        attributes. A disjoint join of one graph returns that graph itself,
        preserving its labels; other joins return a new graph.

    Raises
    ------
    ValueError
        If no graphs are supplied.
    TypeError
        If the graphs have different concrete NetworkX classes.
    """
    graphs = list(graphs)
    if not graphs:
        raise ValueError("Need at least one graph.")

    first_type = type(graphs[0])
    if any(type(graph) is not first_type for graph in graphs[1:]):
        raise TypeError("All graphs must be of the same NetworkX type.")

    if disjoint:
        # Preserve the original object and labels for a single graph.
        return graphs[0] if len(graphs) == 1 else nx.disjoint_union_all(graphs)

    if len(set().union(*graphs)) == sum(map(len, graphs)):
        return nx.compose_all(graphs)

    return nx.compose_all(
        nx.relabel_nodes(graph, {node: f"{rename_prefix}{i}_{node}" for node in graph})
        for i, graph in enumerate(graphs)
    )


def write_graphml(graph: nx.Graph, file_name: str = "graph.graphml") -> None:
    """
    Write a NetworkX graph to a GraphML file.

    Parameters
    ----------
    graph : nx.Graph
        The graph to be written to the file.
    file_name : str, optional
        Destination path. Default is "graph.graphml".

    Returns
    -------
    None
    """
    nx.write_graphml_lxml(graph, os.path.abspath(file_name))


def read_graphml(file_name: str = "graph.graphml") -> nx.Graph:
    """
    Read a NetworkX graph from a GraphML file.

    Parameters
    ----------
    file_name : str, optional
        Source path. Default is "graph.graphml".

    Returns
    -------
    nx.Graph
        The graph read from the file.
    """
    return nx.read_graphml(os.path.abspath(file_name))


def get_bond_smi(mol: Chem.Mol) -> Set[str]:
    """
    Return the unique atom-pair bond strings in a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule whose bonds are inspected.

    Returns
    -------
    Set[str]
        Bond strings with alphabetically ordered atomic symbols. Single,
        double and triple bonds use ``-``, ``=`` and ``#`` respectively;
        all other bond types use ``~``.
    """
    bond_smiles = set()
    for bond in mol.GetBonds():
        symbol1, symbol2 = sorted(
            (bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol())
        )
        bond_symbol = _BOND_TYPE_TO_SMI_SYMBOL.get(bond.GetBondType(), "~")
        bond_smiles.add(f"{symbol1}{bond_symbol}{symbol2}")
    return bond_smiles


def nx_to_smi(
    graph: nx.Graph, add_hydrogens: bool = True, sanitize: bool = True
) -> str:
    """
    Convert a molecular graph to a Kekule SMILES string.

    Parameters
    ----------
    graph : nx.Graph
        A molecular graph with atomic symbols in node ``color`` attributes
        and integer bond orders in edge ``color`` attributes.
    add_hydrogens : bool, optional
        Add explicit hydrogens during sanitization. Default is True.
    sanitize : bool, optional
        Standardize the molecule before writing SMILES. Default is True.

    Returns
    -------
    str
        The SMILES representation produced through :func:`nx_to_mol`.

    Examples
    --------
    This is the inverse of :func:`smi_to_nx`, and the usual way to read the
    virtual objects a calculation returns:

    >>> import assemblytheorytools as att
    >>> graph = att.smi_to_nx("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    >>> ai, virt_obj, pathway = att.calculate_assembly_index(
    ...     graph, strip_hydrogen=True)
    >>> smiles = [att.nx_to_smi(g, add_hydrogens=False) for g in virt_obj]
    >>> "C=O" in smiles
    True
    """
    mol = nx_to_mol(graph, add_hydrogens=add_hydrogens, sanitize=sanitize)
    return Chem.MolToSmiles(mol, allHsExplicit=False, kekuleSmiles=True)


def smi_to_nx(
    smiles: str, add_hydrogens: bool = True, sanitize: bool = True
) -> nx.Graph:
    """
    Convert a SMILES string to a molecular graph.

    Parameters
    ----------
    smiles : str
        The molecular structure in SMILES format.
    add_hydrogens : bool, optional
        Add explicit hydrogens during sanitization. Default is True.
    sanitize : bool, optional
        Standardize the molecule during conversion. Default is True.

    Returns
    -------
    nx.Graph
        A graph with consecutive integer node labels. Node ``color`` holds
        atomic symbols; edge ``color`` holds integer bond orders.

    Raises
    ------
    ValueError
        If the SMILES string is invalid or conversion fails.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> graph = att.smi_to_nx("CCO")
    >>> graph.number_of_nodes(), graph.number_of_edges()
    (9, 8)
    >>> graph.nodes[0]["color"]
    'C'

    The result satisfies the calculator's input requirements, so it can be
    passed straight to
    :func:`~assemblytheorytools.assembly.calculate_assembly_index`:

    >>> att.calculate_assembly_index(graph, strip_hydrogen=True)[0]
    1

    Use a ``.`` separator to build a disconnected graph for a joint
    calculation:

    >>> att.smi_to_nx("NCC(=O)O.CC(N)C(=O)O").number_of_nodes()
    23
    """
    mol = smi_to_mol(smiles, add_hydrogens=add_hydrogens, sanitize=sanitize)
    if mol is None:
        raise ValueError("Invalid SMILES string or conversion failed.")
    return mol_to_nx(mol, add_hydrogens=add_hydrogens, sanitize=sanitize)


def nx_to_inchi(
    graph: nx.Graph, add_hydrogens: bool = True, sanitize: bool = True
) -> str:
    """
    Convert a molecular graph to an InChI string.

    Parameters
    ----------
    graph : nx.Graph
        A molecular graph with atomic symbols in node ``color`` attributes
        and integer bond orders in edge ``color`` attributes.
    add_hydrogens : bool, optional
        Add explicit hydrogens during sanitization. Default is True.
    sanitize : bool, optional
        Standardize the molecule before writing InChI. Default is True.

    Returns
    -------
    str
        The InChI representation produced through :func:`nx_to_mol`.
    """
    mol = nx_to_mol(graph, add_hydrogens=add_hydrogens, sanitize=sanitize)
    return Chem.MolToInchi(mol)


def inchi_to_nx(
    inchi: str, add_hydrogens: bool = False, sanitize: bool = True
) -> nx.Graph:
    """
    Convert an InChI string to a molecular graph.

    Parameters
    ----------
    inchi : str
        The molecular structure in InChI format.
    add_hydrogens : bool, optional
        Add hydrogens during graph conversion. Default is False.
    sanitize : bool, optional
        Standardize the molecule during graph conversion. Default is True.

    Returns
    -------
    nx.Graph
        A graph with consecutive integer node labels. Node ``color`` holds
        atomic symbols; edge ``color`` holds integer bond orders.

    Raises
    ------
    ValueError
        If the InChI string is invalid or conversion fails.

    Notes
    -----
    The initial InChI parser always uses its own defaults for sanitization
    and hydrogen addition. These options control the subsequent graph
    conversion; they do not remove hydrogens already added by the parser.
    """
    mol = inchi_to_mol(inchi)
    if mol is None:
        raise ValueError("Invalid InChI string or conversion failed.")
    return mol_to_nx(mol, add_hydrogens=add_hydrogens, sanitize=sanitize)


def create_ionic_molecule(
    smiles: str, add_hydrogens: bool = True, sanitize: bool = True
) -> Tuple[nx.Graph, List[Chem.Mol]]:
    """
    Combine dot-separated SMILES components and link their charged atoms.

    Parameters
    ----------
    smiles : str
        Component SMILES separated by dots.
    add_hydrogens : bool, optional
        Add explicit hydrogens during sanitization. Default is True.
    sanitize : bool, optional
        Standardize each component during conversion. Default is True.

    Returns
    -------
    Tuple[nx.Graph, List[Chem.Mol]]
        The combined graph and the component molecules, in input order.

    Notes
    -----
    When both charge signs occur, connect the last positive atom and last
    negative atom encountered across the components. This helper retains
    its legacy ionic edge colour of 6, which differs from the RDKit ionic
    bond code returned by :func:`bond_order_rdkit_to_int`.
    """
    mols = [
        smi_to_mol(smi, add_hydrogens=add_hydrogens, sanitize=sanitize)
        for smi in smiles.split(".")
    ]
    graphs = [
        mol_to_nx(mol, add_hydrogens=add_hydrogens, sanitize=sanitize) for mol in mols
    ]

    positive_node = negative_node = None
    offset = 0
    for mol, graph in zip(mols, graphs):
        for atom in mol.GetAtoms():
            charge = atom.GetFormalCharge()
            if charge > 0:
                positive_node = offset + atom.GetIdx()
            elif charge < 0:
                negative_node = offset + atom.GetIdx()
        offset += len(graph)

    combined = nx.disjoint_union_all(graphs)
    if positive_node is not None and negative_node is not None:
        # Keep the legacy ionic colour, distinct from RDKit's bond enum.
        combined.add_edge(positive_node, negative_node, color=6)

    return combined, mols


def longest_path_length(digraph: nx.DiGraph) -> int:
    """
    Return the longest path's edge count in a directed acyclic graph.

    Parameters
    ----------
    digraph : nx.DiGraph
        A directed acyclic graph. Edge weights are ignored.

    Returns
    -------
    int
        The maximum number of edges in a path, or 0 for an empty graph.

    Raises
    ------
    ValueError
        If the graph is undirected or contains a cycle.
    """
    if not nx.is_directed_acyclic_graph(digraph):
        raise ValueError("Graph must be a Directed Acyclic Graph (DAG)")

    # Each generation advances one edge along the longest path from a root.
    generations = nx.topological_generations(digraph)
    return max(0, sum(1 for _ in generations) - 1)


def relabel_digraph(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Set each node's "label" to "Step {generation}" in place.

    Parameters
    ----------
    graph : nx.DiGraph
        A directed acyclic graph.

    Returns
    -------
    nx.DiGraph
        The input graph, with existing ``label`` attributes overwritten.

    Raises
    ------
    networkx.NetworkXUnfeasible
        If a cycle prevents topological sorting. Earlier generations may
        already have been labelled.
    """
    for step, nodes in enumerate(nx.topological_generations(graph)):
        for node in nodes:
            graph.nodes[node]["label"] = f"Step {step}"
    return graph


def relabel_identifiers(graph: nx.Graph) -> nx.Graph:
    """
    Return a copy with node identifiers replaced by their "label" values.

    Parameters
    ----------
    graph : nx.Graph
        The graph to relabel.

    Returns
    -------
    nx.Graph
        A relabelled copy. Equal labels merge their corresponding nodes.

    Raises
    ------
    KeyError
        If any node lacks a ``label`` attribute.
    """
    return nx.relabel_nodes(
        graph, {node: data["label"] for node, data in graph.nodes(data=True)}
    )


def canonicalize_node_labels(graph: nx.Graph) -> nx.Graph:
    """
    Relabel nodes with consecutive integers starting at 0.

    Labels follow node iteration order. This prepares calculator input;
    it does not compute a canonical graph-isomorphism labelling.

    Parameters
    ----------
    graph : nx.Graph
        The input NetworkX graph whose nodes need to be relabeled.

    Returns
    -------
    nx.Graph
        A new NetworkX graph with nodes relabeled to a sequence of integers
        from 0 to n-1.

    Examples
    --------
    The resulting node identifiers are always contiguous, which is why
    :func:`~assemblytheorytools.assembly.calculate_assembly_index`
    applies this normalisation by default:

    >>> import assemblytheorytools as att
    >>> graph = att.smi_to_nx("CCO")
    >>> canonical = att.canonicalize_node_labels(graph)
    >>> sorted(canonical.nodes()) == list(range(graph.number_of_nodes()))
    True
    """
    return nx.relabel_nodes(graph, {node: index for index, node in enumerate(graph)})


def get_graph_charges(
    graph: nx.Graph, pt: Chem.rdchem.PeriodicTable = None
) -> List[int]:
    """
    Estimate formal charges from minimum valences and incident bond orders.

    Parameters
    ----------
    graph : nx.Graph
        A molecular graph with atomic symbols in node ``color`` attributes.
        Edge ``color`` attributes give bond orders; missing colours default
        to 1.
    pt : Chem.rdchem.PeriodicTable, optional
        Periodic table supplying atomic numbers and valence lists. Defaults
        to RDKit's periodic table.

    Returns
    -------
    List[int]
        Minimum allowed valence minus the sum of neighbouring bond orders,
        for each atom in node iteration order.
    """
    pt = pt or GetPeriodicTable()
    charges = []
    for node, data in graph.nodes(data=True):
        atomic_number = pt.GetAtomicNumber(data["color"])
        valence = min(pt.GetValenceList(atomic_number))
        bond_order_sum = sum(
            graph.edges[node, neighbor].get("color", 1)
            for neighbor in graph.neighbors(node)
        )
        charges.append(valence - bond_order_sum)
    return charges


def compose_graphs(
    graphs: Iterable[nx.Graph],
) -> Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]:
    """
    Merge graphs sharing node identifiers, with later attributes winning.

    Parameters
    ----------
    graphs : Iterable[nx.Graph]
        NetworkX graphs to compose in iteration order.

    Returns
    -------
    nx.Graph
        The composed graph. A single input graph is returned unchanged;
        multiple inputs produce a new graph using NetworkX composition.

    Raises
    ------
    ValueError
        If no graphs are supplied.
    networkx.NetworkXError
        If directed and undirected graphs, or simple and multigraphs,
        are mixed.
    """
    graphs = list(graphs)
    if not graphs:
        raise ValueError("compose_graphs() requires at least one graph")

    return reduce(nx.compose, graphs)


def set_graph_layer(digraph: nx.DiGraph) -> nx.DiGraph:
    """
    Set each node's integer "layer" to its topological generation in place.

    Parameters
    ----------
    digraph : nx.DiGraph
        A directed acyclic graph.

    Returns
    -------
    nx.DiGraph
        The input graph, with ``layer`` values overwritten starting at 0.

    Raises
    ------
    networkx.NetworkXUnfeasible
        If a cycle prevents topological sorting. Earlier generations may
        already have been labelled.
    """
    for layer, nodes in enumerate(nx.topological_generations(digraph)):
        for node in nodes:
            digraph.nodes[node]["layer"] = layer

    return digraph


def strip_digraph_layer(digraph: nx.DiGraph, layer: int) -> nx.DiGraph:
    """
    Copy a directed acyclic graph and remove one topological generation.

    Parameters
    ----------
    digraph : nx.DiGraph
        The graph to copy. Layers are recomputed on the copy before removal.
    layer : int
        The topological generation to remove, counting from 0.

    Returns
    -------
    nx.DiGraph
        A mutable copy without the selected nodes and their incident edges.
        Surviving nodes retain their computed ``layer`` values.
    """
    modified_graph = set_graph_layer(digraph.copy())
    nodes_to_remove = [
        node for node, data in modified_graph.nodes(data=True) if data["layer"] == layer
    ]
    modified_graph.remove_nodes_from(nodes_to_remove)
    return modified_graph


def top_n_degree_subgraph(
    G: nx.DiGraph, n: int, must_keep: List[nx.Graph]
) -> nx.DiGraph:
    """
    Keep the highest-degree nodes and nodes matching required subgraphs.

    Parameters
    ----------
    G : nx.DiGraph
        A directed graph whose nodes carry molecular graphs in ``vo``.
    n : int
        Number of nodes to select by total in-degree plus out-degree.
        Ties follow node iteration order; selection uses Python slicing.
    must_keep : List[nx.Graph]
        Reference graphs. Nodes whose ``vo`` is topologically isomorphic
        to any reference are retained regardless of degree.

    Returns
    -------
    nx.DiGraph
        A subgraph view of a copy of the input, containing both selections.

    Notes
    -----
    If the references contain hydrogen atoms but none of the input's ``vo``
    graphs do, matching uses hydrogen-free copies of the references. The
    input and reference graphs are unchanged. Matching ignores colours.
    """
    G = G.copy()

    symbols_g = {
        data["color"]
        for _, node_data in G.nodes(data=True)
        for _, data in node_data.get("vo", {}).nodes(data=True)
    }
    symbols_ref = {
        data["color"] for graph in must_keep for _, data in graph.nodes(data=True)
    }

    if "H" in symbols_ref and "H" not in symbols_g:
        must_keep = [remove_hydrogen_from_graph(graph) for graph in must_keep]

    ranked_nodes = sorted(G.degree(), key=lambda item: item[1], reverse=True)
    top_nodes = {node for node, _ in ranked_nodes[:n]}
    keep_nodes = {
        node
        for node, data in G.nodes(data=True)
        if any(nx.is_isomorphic(data.get("vo"), graph) for graph in must_keep)
    }

    return G.subgraph(top_nodes | keep_nodes)


def strip_digraph_zero_indegree(G: nx.DiGraph) -> nx.DiGraph:
    """
    Remove nodes with zero in-degree in a single pass.

    Parameters
    ----------
    G : nx.DiGraph
        The graph to copy and filter.

    Returns
    -------
    nx.DiGraph
        A subgraph view of a copy, retaining nodes with positive in-degree
        in the input. Nodes that become roots after removal are kept.
    """
    G = G.copy()
    return G.subgraph(node for node, degree in G.in_degree() if degree > 0)
