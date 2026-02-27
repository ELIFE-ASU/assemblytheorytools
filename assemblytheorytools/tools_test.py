import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeVar

import networkx as nx

T = TypeVar("T")


def check_elements(input_list: Sequence[T], reference_list: Sequence[T]) -> bool:
    """Check if all elements in an input list are present in a reference list.

    This function returns ``False`` for an empty input list, preserving existing
    behavior.

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
    if not input_list:
        return False
    return all(item in reference_list for item in input_list)


def print_graph_details(graph: nx.Graph) -> None:
    """Print the details of a graph.

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
    for node in graph.nodes(data=True):
        node_index = node[0]
        node_color = node[1].get('color', 'No color')
        edge_connections = list(graph.edges(node_index))
        edge_colors = [graph.get_edge_data(*edge)['color'] for edge in edge_connections]
        print(f"({node_index}, {node_color}): {edge_connections}, {edge_colors}", flush=True)
    print("}", flush=True)


def water_graph() -> nx.Graph:
    """Construct a graph representation of a water molecule.

    The graph consists of three nodes representing the atoms in a water molecule:
    one oxygen (O) and two hydrogens (H). Edges represent bonds between the
    atoms, with bond types indicated by edge attributes.

    Returns
    -------
    nx.Graph
        A NetworkX graph object representing the water molecule.
    """
    graph = nx.Graph()
    # Add nodes with atom types as attributes
    graph.add_node(0, color="O")  # Oxygen atom
    graph.add_node(1, color="H")  # Hydrogen atom
    graph.add_node(2, color="H")  # Hydrogen atom

    # Add edges with bond type attributes
    graph.add_edge(0, 1, color=1)  # Bond between oxygen and first hydrogen
    graph.add_edge(0, 2, color=1)  # Bond between oxygen and second hydrogen
    return graph


def phosphine_graph() -> nx.Graph:
    """Construct a graph representation of a phosphine molecule.

    The graph consists of four nodes representing the atoms in a phosphine
    molecule: one phosphorus (P) and three hydrogens (H). Edges represent
    bonds between the atoms, with bond types indicated by edge attributes.

    Returns
    -------
    nx.Graph
        A NetworkX graph object representing the phosphine molecule.
    """
    graph = nx.Graph()
    # Add nodes with atom types as attributes
    graph.add_node(0, color="P")  # Phosphorus atom
    graph.add_node(1, color="H")  # Hydrogen atom
    graph.add_node(2, color="H")  # Hydrogen atom
    graph.add_node(3, color="H")  # Hydrogen atom

    # Add edges with bond type attributes
    graph.add_edge(0, 1, color=1)  # Bond between phosphorus and first hydrogen
    graph.add_edge(0, 2, color=1)  # Bond between phosphorus and second hydrogen
    graph.add_edge(0, 3, color=1)  # Bond between phosphorus and third hydrogen
    return graph


def ph_2p_graph() -> nx.Graph:
    """Construct a graph representation of a simple phosphine-like molecule.

    The system is +2 charged, with two phosphorus atoms and one hydrogen atom.
    The graph consists of two nodes representing the atoms: one phosphorus (P)
    and one hydrogen (H). An edge represents the bond between the phosphorus
    and hydrogen atoms, with the bond type indicated by an edge attribute.

    Returns
    -------
    nx.Graph
        A NetworkX graph object representing the phosphine-like molecule.
    """
    graph = nx.Graph()
    # Add nodes with atom types as attributes
    graph.add_node(0, color="P")  # Phosphorus atom
    graph.add_node(1, color="H")  # Hydrogen atom

    # Add edge with bond type attribute
    graph.add_edge(0, 1, color=1)  # Bond between phosphorus and hydrogen
    return graph


def co2_graph() -> nx.Graph:
    """Construct a graph representation of a carbon dioxide (CO2) molecule.

    The graph consists of three nodes representing the atoms in a CO2 molecule:
    one carbon (C) and two oxygens (O). Edges represent bonds between the
    atoms, with bond types indicated by edge attributes.

    Returns
    -------
    nx.Graph
        A NetworkX graph object representing the CO2 molecule.
    """
    graph = nx.Graph()
    # Add nodes with atom types as attributes
    graph.add_node(0, color="C")  # Carbon atom
    graph.add_node(1, color="O")  # Oxygen atom
    graph.add_node(2, color="O")  # Oxygen atom

    # Add edges with bond type attributes
    graph.add_edge(0, 1, color=2)  # Double bond between carbon and first oxygen
    graph.add_edge(0, 2, color=2)  # Double bond between carbon and second oxygen
    return graph


@dataclass(frozen=True)
class Molecule:
    """A container for molecule metadata.

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
    """Load molecule records from a CSV and return a mapping.

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
    mols: dict[str, Molecule] = {}

    # Ensure the path is correct and the file exists
    data_path = Path(__file__).parent.parent / "tests/data/test_molecule_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with data_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip().lower()
            category = row["category"].strip()
            smiles = row["smiles"].strip()
            inchi = (row.get("inchi") or "").strip() or None

            ai_str = (row.get("assembly_index") or "").strip()
            assembly_index = int(ai_str) if ai_str else None

            test_include_str = (row.get("test_include") or "").strip()
            # handles 'True'/'False' as in your CSV
            test_include = test_include_str.lower() == "true"

            mols[name] = Molecule(
                name=name,
                category=category,
                smiles=smiles,
                inchi=inchi,
                assembly_index=assembly_index,
                test_include=test_include,
            )
    return mols


test_mols: dict[str, Molecule] = _load_molecules()
