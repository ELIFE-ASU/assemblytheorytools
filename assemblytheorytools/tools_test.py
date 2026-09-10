"""
Shared test molecules and graph fixtures.

This module exposes the reference molecule set loaded from the bundled
``data/test_molecule_data.csv`` as the ``test_mols`` mapping, together with small
hand-built NetworkX graphs (water, phosphine, PH2+ and carbon dioxide) and helpers
for inspecting graph contents in tests.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeVar

import networkx as nx

T = TypeVar("T")


def check_elements(input_list: Sequence[T], reference_list: Sequence[T]) -> bool:
    """
    Check if all elements in an input list are present in a reference list.

    Returns ``False`` for an empty input list.

    Parameters
    ----------
    input_list : Sequence[T]
        The list of elements to check.
    reference_list : Sequence[T]
        The list of elements to check against.

    Returns
    -------
    bool
        ``True`` if all elements in ``input_list`` are in ``reference_list``,
        ``False`` otherwise.
    """
    return bool(input_list) and all(item in reference_list for item in input_list)


def print_graph_details(graph: nx.Graph) -> None:
    """
    Print the details of a graph.

    This function prints the details of a graph, including node indices, node
    colors, edge connections, and edge colors.

    Parameters
    ----------
    graph : nx.Graph
        The graph whose details are to be printed.

    Returns
    -------
    None
        This function prints information to the console and does not return a
        value.
    """
    print("{", flush=True)
    for node, attributes in graph.nodes(data=True):
        color = attributes.get("color", "No color")
        edges = list(graph.edges(node))
        edge_colors = [graph.get_edge_data(*edge)["color"] for edge in edges]
        print(f"({node}, {color}): {edges}, {edge_colors}", flush=True)
    print("}", flush=True)


def _star_graph(
    center: str, neighbor: str, n_neighbors: int, bond_order: int
) -> nx.Graph:
    """Build a fresh fixture with identically colored neighbors and bonds."""
    graph = nx.Graph()
    graph.add_node(0, color=center)
    graph.add_nodes_from(range(1, n_neighbors + 1), color=neighbor)
    graph.add_edges_from(
        ((0, node) for node in range(1, n_neighbors + 1)), color=bond_order
    )
    return graph


def water_graph() -> nx.Graph:
    """
    Construct a graph representation of a water molecule.

    The graph consists of three nodes representing the atoms in a water molecule:
    one oxygen (O) and two hydrogens (H). Edges represent bonds between the
    atoms, with bond types indicated by edge attributes.

    Returns
    -------
    nx.Graph
        A NetworkX graph object representing the water molecule.
    """
    return _star_graph("O", "H", 2, 1)


def phosphine_graph() -> nx.Graph:
    """
    Construct a graph representation of a phosphine molecule.

    The graph consists of four nodes representing the atoms in a phosphine
    molecule: one phosphorus (P) and three hydrogens (H). Edges represent
    bonds between the atoms, with bond types indicated by edge attributes.

    Returns
    -------
    nx.Graph
        A NetworkX graph object representing the phosphine molecule.
    """
    return _star_graph("P", "H", 3, 1)


def ph_2p_graph() -> nx.Graph:
    """
    Construct a graph representation of a simple phosphine-like molecule.

    The graph consists of two nodes representing the atoms: one phosphorus (P)
    and one hydrogen (H), joined by a single bond. Only atom and bond colors
    are stored; the graph has no charge attributes.

    Returns
    -------
    nx.Graph
        A NetworkX graph object representing the phosphine-like molecule.
    """
    return _star_graph("P", "H", 1, 1)


def co2_graph() -> nx.Graph:
    """
    Construct a graph representation of a carbon dioxide (CO2) molecule.

    The graph consists of three nodes representing the atoms in a CO2 molecule:
    one carbon (C) and two oxygens (O). Edges represent bonds between the
    atoms, with bond types indicated by edge attributes.

    Returns
    -------
    nx.Graph
        A NetworkX graph object representing the CO2 molecule.
    """
    return _star_graph("C", "O", 2, 2)


@dataclass(frozen=True)
class Molecule:
    """
    A container for molecule metadata.

    Attributes
    ----------
    name : str
        The molecule name, typically stored in lowercase by the loader.
    category : str
        The category or type of the molecule.
    smiles : str
        The SMILES (Simplified Molecular Input Line Entry System) string.
    inchi : str or None
        The InChI (International Chemical Identifier) string, or ``None`` if
        not provided.
    assembly_index : int or None
        An optional assembly index parsed from the CSV, or ``None`` if absent.
    test_include : bool
        A flag indicating whether this molecule should be included in tests.
    """

    name: str
    category: str
    smiles: str
    inchi: str | None
    assembly_index: int | None
    test_include: bool


def _load_molecules() -> dict[str, Molecule]:
    """
    Load molecule records from a CSV and return a mapping.

    The CSV is expected to contain the following columns (whitespace is
    trimmed):

    - ``name``: The molecule name (used as the dictionary key, lowercased).
    - ``category``: The category or type of the molecule.
    - ``smiles``: The SMILES string.
    - ``inchi``: An optional InChI string.
    - ``assembly_index``: An optional integer.
    - ``test_include``: An optional boolean represented as 'True'/'False'.

    Returns
    -------
    dict[str, Molecule]
        A mapping from lowercase molecule names to their corresponding
        :class:`Molecule` instances.

    Raises
    ------
    ValueError
        If a non-empty ``assembly_index`` field cannot be converted to an
        integer.
    FileNotFoundError
        If the data file cannot be found at the expected path.
    """
    # Bundled package data, not a sibling of the source tree: this module is
    # imported by __init__.py, so the CSV has to be present in an installed
    # wheel as well as in a checkout.
    data_path = Path(__file__).parent / "data" / "test_molecule_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    mols: dict[str, Molecule] = {}
    with data_path.open(newline="") as file:
        for row in csv.DictReader(file):
            name = row["name"].strip().lower()
            category = row["category"].strip()
            smiles = row["smiles"].strip()
            inchi = (row.get("inchi") or "").strip() or None

            ai_str = (row.get("assembly_index") or "").strip()
            assembly_index = int(ai_str) if ai_str else None

            mols[name] = Molecule(
                name=name,
                category=category,
                smiles=smiles,
                inchi=inchi,
                assembly_index=assembly_index,
                test_include=(row.get("test_include") or "").strip().lower() == "true",
            )
    return mols


test_mols: dict[str, Molecule] = _load_molecules()
