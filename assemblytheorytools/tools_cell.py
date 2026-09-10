"""
Handling of crystal structures and periodic cells.

This module reads CIF files into ASE ``Atoms`` objects, identifies bonded
clusters within a periodic cell, tiles cells and shells to build finite
neighbourhoods, converts cells to NetworkX graphs, and guesses bond orders
for the resulting connectivity.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from ase import Atoms
from ase.io import cif
from ase.neighborlist import NeighborList, natural_cutoffs, neighbor_list
from rdkit import Chem
from scipy import sparse


def read_cif_file(cif_file: str) -> Atoms:
    """
    Read the primitive cell from a CIF file into an ASE atoms object.

    Parameters
    ----------
    cif_file : str
        The path to the CIF file.

    Returns
    -------
    ase.Atoms
        The atoms object.
    """
    return cif.read_cif(cif_file, primitive_cell=True, subtrans_included=False)


def atoms_to_mol_file(atoms: Atoms, file_name: str = "mol.mol") -> None:
    """
    Write a molecule to a .mol file from an ASE atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        The input set of atoms.
    file_name : str, optional
        The name of the output .mol file. Default is "mol.mol".

    Returns
    -------
    None
        This function does not return a value.

    Notes
    -----
    Bond detection clears the input atoms' cell and periodic boundaries.
    """
    bond_pairs = get_bonding_config(atoms)
    lines = [
        "\nLouie's generator\n\n",
        f"{len(atoms):>3}{len(bond_pairs):>3}  0  0  0  0  0  0  0  0999 V2000\n",
    ]
    for (x, y, z), symbol in zip(atoms.get_positions(), atoms.get_chemical_symbols()):
        lines.append(
            f"{x:10.4f}{y:10.4f}{z:10.4f} {symbol:<3}"
            " 0  0  0  0  0  0  0  0  0  0  0  0\n"
        )
    lines.extend(f"{i + 1:>3}{j + 1:>3}  1  0  0  0  0\n" for i, j in bond_pairs)
    lines.append("M  END\n")
    with open(file_name, "w") as mol_file:
        mol_file.writelines(lines)


def get_bonding_config(atoms: Atoms) -> List[List[int]]:
    """
    Generate the bonding configuration for a given set of atoms.

    Parameters
    ----------
    atoms : ase.Atoms
        The input set of atoms.

    Returns
    -------
    List[List[int]]
        Bond pairs, each represented as a list of two atom indices.

    Notes
    -----
    Clears the input atoms' cell and periodic boundaries before finding
    bonds using ASE's default neighbor-list settings.
    """
    atoms.set_pbc([False, False, False])
    atoms.cell = [0, 0, 0]
    neighbors = NeighborList(natural_cutoffs(atoms))
    neighbors.update(atoms)

    bond_pairs = []
    for i in range(len(atoms)):
        indices, _ = neighbors.get_neighbors(i)
        bond_pairs.extend([i, j] for j in indices[indices != i])
    return bond_pairs


def find_clusters(atoms: Atoms, cutoff_smear: float = 1.5) -> Optional[List[int]]:
    """
    Find atoms outside the largest bonded cluster in an atomic structure.

    Connectivity uses natural covalent radii scaled by ``cutoff_smear``.

    Parameters
    ----------
    atoms : ase.Atoms
        The molecular or periodic structure.
    cutoff_smear : float, optional
        Multiplier for natural cutoff radii. Default is 1.5.

    Returns
    -------
    list or None
        Indices outside the largest cluster, or ``None`` if all atoms are
        connected. Ties retain the first component in atom order.

    Notes
    -----
    Uses ASE's ``NeighborList`` and SciPy's ``connected_components``.
    Prints the cluster count and removal indices for disconnected inputs.
    """
    neighbors = NeighborList(
        natural_cutoffs(atoms, mult=cutoff_smear),
        self_interaction=False,
        bothways=True,
    )
    neighbors.update(atoms)
    n_components, labels = sparse.csgraph.connected_components(
        neighbors.get_connectivity_matrix()
    )
    if n_components == 1:
        return None

    largest_component = np.argmax(np.bincount(labels))
    atoms_to_remove = np.flatnonzero(labels != largest_component).tolist()
    print("Number of clusters:", n_components)
    print("Atoms to remove:", atoms_to_remove)
    return atoms_to_remove


def _cell_neighborhood(
    atoms: Atoms, reps: tuple[int, int, int], multi: float, eps: float
) -> tuple[Atoms, np.ndarray, np.ndarray, np.ndarray]:
    """Build a supercell, central-region mask, and directed bond pairs."""
    supercell = atoms.repeat(reps)
    scaled_positions = supercell.get_scaled_positions(wrap=False)
    repetitions = np.asarray(reps, dtype=float)
    low = (repetitions - 1) / (2 * repetitions)
    high = (repetitions + 1) / (2 * repetitions)
    central = np.all(
        (scaled_positions >= low - eps) & (scaled_positions < high + eps), axis=1
    )
    sources, targets = neighbor_list(
        "ij", supercell, natural_cutoffs(supercell, mult=multi)
    )
    return supercell, central, sources, targets


def _wrapped_subset(atoms: Atoms, mask: np.ndarray) -> Atoms:
    """Select atoms in their original order, retaining the cell and PBC."""
    subset = atoms[mask]
    subset.wrap()
    return subset


def tile_cell(
    atoms: Atoms,
    reps: tuple[int, int, int] = (3, 3, 3),
    multi: float = 1.2,
    eps: float = 1e-9,
) -> Atoms:
    """
    Create a tiled supercell with central region and bonded atoms.

    Repeat the unit cell and retain the central region and its bonded
    neighbors.

    Parameters
    ----------
    atoms : ase.Atoms
        The input atomic structure.
    reps : tuple[int, int, int], optional
        Repetitions along (x, y, z). Default is (3, 3, 3).
    multi : float, optional
        Multiplier for natural cutoff distances. Default is 1.2.
    eps : float, optional
        Numerical tolerance for the central region bounds. Default is 1e-9.

    Returns
    -------
    ase.Atoms
        Supercell subset with central atoms and their bonded neighbors.
    """
    supercell, central, sources, targets = _cell_neighborhood(atoms, reps, multi, eps)
    keep = central.copy()
    keep[targets[central[sources]]] = True
    return _wrapped_subset(supercell, keep)


def tile_cell_shells(
    atoms: Atoms,
    reps: tuple[int, int, int] = (3, 3, 3),
    multi: float = 1.2,
    eps: float = 1e-9,
) -> tuple[Atoms, Atoms, Atoms]:
    """
    Tile a cell and separate atoms into central and shell regions.

    Identify the central region and two disjoint coordination shells based
    on bonding connectivity.

    Parameters
    ----------
    atoms : ase.Atoms
        The input atomic structure.
    reps : tuple[int, int, int], optional
        Repetitions along (x, y, z). Default is (3, 3, 3).
    multi : float, optional
        Multiplier for natural cutoff distances. Default is 1.2.
    eps : float, optional
        Numerical tolerance for the central region bounds. Default is 1e-9.

    Returns
    -------
    central_atoms : ase.Atoms
        Atoms in the central region of the supercell.
    first_shell_atoms : ase.Atoms
        Atoms in the first coordination shell around the central region.
    second_shell_atoms : ase.Atoms
        Atoms in the second coordination shell around the central region.
    """
    supercell, central, sources, targets = _cell_neighborhood(atoms, reps, multi, eps)

    def neighbors_of(region: np.ndarray) -> np.ndarray:
        """Select atoms bonded to the region in either edge direction."""
        neighbors = np.zeros(len(supercell), dtype=bool)
        neighbors[targets[region[sources]]] = True
        neighbors[sources[region[targets]]] = True
        return neighbors

    first_shell = neighbors_of(central) & ~central
    second_shell = neighbors_of(first_shell) & ~(central | first_shell)
    return (
        _wrapped_subset(supercell, central),
        _wrapped_subset(supercell, first_shell),
        _wrapped_subset(supercell, second_shell),
    )


def cif_to_nx(
    file: str,
    reps: tuple[int, int, int] = (3, 3, 3),
    cutoff_mult: float = 1.2,
    eps: float = 1e-9,
) -> nx.Graph:
    """
    Convert a CIF file to a NetworkX graph representation.

    Read a CIF file, expand the unit cell, and create a graph where nodes
    represent atoms and edges represent bonds.

    Parameters
    ----------
    file : str
        Path to the CIF file.
    reps : tuple[int, int, int], optional
        Number of repetitions in each direction. Default is (3, 3, 3).
    cutoff_mult : float, optional
        Multiplier for natural cutoff distances. Default is 1.2.
    eps : float, optional
        Small epsilon value for numerical tolerance. Default is 1e-9.

    Returns
    -------
    nx.Graph
        Graph with element symbols in node ``color`` attributes and bond
        orders (all 1) in edge ``color`` attributes.

    Warns
    -----
    UserWarning
        Always, because the CIF conversion is experimental.
    """
    warnings.warn("The cif_to_nx function is experimental.", UserWarning)
    atoms = read_cif_file(file)
    expanded = tile_cell(atoms, reps=reps, multi=cutoff_mult, eps=eps)

    graph = nx.Graph()
    graph.add_nodes_from((i, {"color": atom.symbol}) for i, atom in enumerate(expanded))
    graph.add_edges_from(get_bonding_config(expanded), color=1)
    return graph


def guess_bond_orders(
    G: nx.Graph,
    formal_charge_attr: Optional[str] = "formal_charge",
    max_bond_order: int = 4,
) -> Tuple[nx.Graph, bool, Dict]:
    """
    Assign bond orders to a molecular graph by backtracking over valences.

    Target valences come from periodic table data, with an upward bias for
    positively charged atoms.

    Parameters
    ----------
    G : nx.Graph
        Input molecular graph with nodes having 'color' attribute (element
        symbol).
    formal_charge_attr : Optional[str], optional
        Attribute name for formal charge on nodes. Default is
        "formal_charge".
    max_bond_order : int, optional
        Upper bound on bond order. Default is 4; the search considers only
        single, double, and triple bonds.

    Returns
    -------
    G_with_orders : nx.Graph
        Copy of the input graph with bond orders in edge ``color``
        attributes. Unassigned edges retain their original attributes.
    success : bool
        True if all valence constraints were satisfied, False otherwise.
    info : Dict
        Target valences, remaining valences, and search statistics.
        Failed searches report residuals after backtracking, even when
        the returned graph retains a partial assignment.

    Raises
    ------
    ValueError
        If a node carries an element symbol that is not in the periodic
        table.

    Warns
    -----
    UserWarning
        Always, because the bond order search is experimental.
    """

    warnings.warn("The guess_bond_orders function is experimental.", UserWarning)
    periodic_table = Chem.GetPeriodicTable()
    graph = G.copy()

    def choose_target_valence(atomic_number: int, needed_min: int, charge: int) -> int:
        """Choose the smallest positive valence that fits the degree."""
        valences = periodic_table.GetValenceList(atomic_number)
        bias = 1 if charge > 0 else 0
        candidates = [
            int(v) + bias for v in valences if v > 0 and v + bias >= needed_min
        ]
        if candidates:
            return min(candidates)
        default = periodic_table.GetDefaultValence(atomic_number)
        if default >= needed_min:
            return default + bias
        return max(needed_min, int(max(valences)))

    target_valence = {}
    for node, data in graph.nodes(data=True):
        element = data.get("color")
        atomic_number = periodic_table.GetAtomicNumber(element)
        if atomic_number == 0:
            raise ValueError(f"Node {node} has unknown element symbol: {element}")
        charge = int(data.get(formal_charge_attr, 0)) if formal_charge_attr else 0
        target_valence[node] = choose_target_valence(
            atomic_number, int(graph.degree[node]), charge
        )

    residual = target_valence.copy()
    assigned = {}
    tried_edges = 0
    backtracks = 0
    best_partial = {}
    best_score = -1

    def feasible_after(u, v, order: int) -> bool:
        """Check if other incident edges can absorb each residual."""
        for node in (u, v):
            remaining = residual[node] - order
            if remaining < 0:
                return False
            # Keep incident-edge orientation to preserve the search traversal.
            other_nodes = [
                y if x == node else x
                for x, y in graph.edges(node)
                if (x, y) not in assigned and (x, y) not in ((u, v), (v, u))
            ]
            capacity = sum(
                max(0, min(max_bond_order, remaining, residual[other]))
                for other in other_nodes
            )
            if not len(other_nodes) <= remaining <= capacity:
                return False
        return True

    def select_edge():
        """Choose the smallest domain, breaking ties by edge order."""
        best_edge, best_domain = None, None
        for u, v in graph.edges():
            if (u, v) in assigned or (v, u) in assigned:
                continue
            limit = min(residual[u], residual[v], max_bond_order)
            domain = [order for order in (1, 2, 3) if order <= limit]
            if not domain:
                return (u, v), []
            if best_domain is None or len(domain) < len(best_domain):
                best_edge, best_domain = (u, v), domain
        return best_edge, best_domain

    def search() -> bool:
        """Try feasible orders and retain the best terminal assignment."""
        nonlocal tried_edges, backtracks, best_partial, best_score
        edge, domain = select_edge()
        if edge is None or not domain:
            score = sum(value == 0 for value in residual.values())
            if edge is None and score == len(residual):
                return True
            if score > best_score:
                best_score, best_partial = score, assigned.copy()
            return False

        u, v = edge
        # Prefer higher orders when both endpoints have substantial valence left.
        for order in sorted(domain, reverse=residual[u] > 2 and residual[v] > 2):
            if not feasible_after(u, v, order):
                continue
            tried_edges += 1
            assigned[edge] = order
            residual[u] -= order
            residual[v] -= order
            if search():
                return True
            residual[u] += order
            residual[v] += order
            del assigned[edge]

        backtracks += 1
        return False

    success = search()
    final_assignments = assigned if success else best_partial
    for (u, v), order in final_assignments.items():
        graph.edges[u, v]["color"] = int(order)

    info = {
        "target_valence": target_valence,
        "remaining_valence_per_atom": residual.copy(),
        "tried_edges": tried_edges,
        "backtracks": backtracks,
        "success_edges_assigned": len(final_assignments),
        "total_edges": graph.number_of_edges(),
    }
    return graph, success, info
