"""
Handling of crystal structures and periodic cells.

This module reads CIF files into ASE ``Atoms`` objects, detects bonds from
covalent radii, identifies bonded clusters within a periodic cell, tiles
cells and shells to build finite neighbourhoods, converts periodic cells to
NetworkX graphs, and guesses bond orders for the resulting connectivity.
"""

import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from ase import Atoms
from ase.io import cif
from ase.neighborlist import natural_cutoffs, neighbor_list
from ase.spacegroup.spacegroup import SpacegroupNotFoundError
from rdkit import Chem
from scipy import sparse

# Full Hermann-Mauguin symbols of monoclinic groups with unique axis b, such
# as ``C 1 2/m 1``, which ASE's table only stores in the short form ``C 2/m``.
_FULL_MONOCLINIC_SYMBOL = re.compile(r"([ABCPIF])\s+1\s+(\S+)\s+1")


def _shorten_spacegroup_symbol(block: cif.CIFBlock) -> Optional[cif.CIFBlock]:
    """Rebuild a CIF block with full monoclinic symbols shortened, else None."""
    tags = dict(block)
    changed = False
    for key, value in tags.items():
        if "space_group_name_h-m" not in key or not isinstance(value, str):
            continue
        match = _FULL_MONOCLINIC_SYMBOL.fullmatch(value.strip())
        if match is not None:
            tags[key] = f"{match[1]} {match[2]}"
            changed = True
    return cif.CIFBlock(block.name, tags) if changed else None


def _warn_partial_occupancy(atoms: Atoms, source: str) -> None:
    """Warn when any site of a CIF-derived structure is partially occupied."""
    occupancy = atoms.info.get("occupancy") or {}
    partial = [
        kind
        for kind, species in occupancy.items()
        if any(fraction < 1 for fraction in species.values())
    ]
    if not partial:
        return
    kinds = atoms.arrays.get("spacegroup_kinds")
    if kinds is None:
        affected = len(partial)
    else:
        affected = int(np.isin(kinds, [int(kind) for kind in partial]).sum())
    warnings.warn(
        f"{len(partial)} of {len(occupancy)} sites in {source} have fractional or "
        f"mixed occupancy ({affected} of {len(atoms)} atoms). Every site is kept "
        "with its majority species, so split sites over-count neighbours in bond "
        "graphs.",
        UserWarning,
        stacklevel=3,
    )


def read_cif_file(cif_file: str, index: int = -1) -> Atoms:
    """
    Read the primitive cell from a CIF file into an ASE atoms object.

    Parameters
    ----------
    cif_file : str
        The path to the CIF file.
    index : int, optional
        Structure block to read when the file holds several. Default is -1,
        the last block, matching ``ase.io.cif.read_cif``.

    Returns
    -------
    ase.Atoms
        The atoms object with the primitive cell and periodic boundaries.

    Raises
    ------
    ValueError
        If the file holds no structure block or ``index`` is out of range.
    ase.spacegroup.spacegroup.SpacegroupNotFoundError
        If ASE does not know the space group symbol, even after shortening.

    Warns
    -----
    UserWarning
        If any site has fractional or mixed occupancy. ASE keeps every such
        site with its majority species, so split sites overlap and over-count
        neighbours in bond graphs.

    Notes
    -----
    ASE's space-group table stores short Hermann-Mauguin symbols, so full
    monoclinic symbols with unique axis b, such as ``C 1 2/m 1``, are
    shortened to ``C 2/m`` when the direct lookup fails. Occupancies are kept
    in ``atoms.info["occupancy"]`` and site kinds in
    ``atoms.arrays["spacegroup_kinds"]``.
    """
    path = os.fspath(cif_file)
    blocks = [block for block in cif.parse_cif(path) if block.has_structure()]
    if not blocks:
        raise ValueError(f"{path} contains no crystal structure block.")
    try:
        block = blocks[index]
    except IndexError:
        raise ValueError(
            f"{path} has {len(blocks)} structure block(s); index {index} is out "
            "of range."
        ) from None

    options = {"primitive_cell": True, "subtrans_included": False}
    try:
        atoms = block.get_atoms(**options)
    except SpacegroupNotFoundError:
        shortened = _shorten_spacegroup_symbol(block)
        if shortened is None:
            raise
        atoms = shortened.get_atoms(**options)
    _warn_partial_occupancy(atoms, path)
    return atoms


def _bond_pairs(
    atoms: Atoms, mult: float, periodic: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return directed bonded pairs and image shifts from covalent radii."""
    if len(atoms) == 0:
        empty = np.zeros(0, dtype=int)
        return empty, empty.copy(), np.zeros((0, 3), dtype=int)
    if not periodic:
        atoms = atoms.copy()
        atoms.set_pbc(False)
    return neighbor_list("ijS", atoms, natural_cutoffs(atoms, mult=mult))


def atoms_to_mol_file(
    atoms: Atoms, file_name: str = "mol.mol", mult: float = 1.2
) -> None:
    """
    Write a molecule to a .mol file from an ASE atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        The input set of atoms.
    file_name : str, optional
        The name of the output .mol file. Default is "mol.mol".
    mult : float, optional
        Multiplier for the covalent-radius bond cutoffs; see
        :func:`get_bonding_config`. Default is 1.2.

    Returns
    -------
    None
        This function does not return a value.

    Notes
    -----
    Bonds are detected without periodic boundaries and all receive order 1.
    The input atoms are left unchanged.
    """
    bond_pairs = get_bonding_config(atoms, mult=mult)
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


def get_bonding_config(atoms: Atoms, mult: float = 1.2) -> List[List[int]]:
    """
    Generate the bonding configuration for a given set of atoms.

    Parameters
    ----------
    atoms : ase.Atoms
        The input set of atoms.
    mult : float, optional
        Multiplier for the covalent-radius cutoffs. Two atoms are bonded when
        their distance is below ``mult`` times the sum of their covalent
        radii. Default is 1.2.

    Returns
    -------
    List[List[int]]
        Bond pairs, each represented as a list of two atom indices with the
        smaller index first, in lexicographic order.

    Notes
    -----
    The cell and periodic boundaries are ignored, so bonds through a cell
    boundary are not reported; use :func:`cell_to_nx` for periodic
    connectivity. The input atoms are left unchanged. Earlier versions used
    ASE's ``NeighborList`` with its default 0.3 Å skin, which added 0.6 Å to
    every cutoff.
    """
    sources, targets, _ = _bond_pairs(atoms, mult, periodic=False)
    mask = sources < targets
    pairs = np.column_stack((sources[mask], targets[mask]))
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order].tolist()


def find_clusters(atoms: Atoms, cutoff_smear: float = 1.5) -> Optional[List[int]]:
    """
    Find atoms outside the largest bonded cluster in an atomic structure.

    Connectivity uses natural covalent radii scaled by ``cutoff_smear`` and
    honours the periodic boundaries of ``atoms``.

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
        connected or there are no atoms. Ties retain the first component in
        atom order.

    Notes
    -----
    Uses ASE's ``neighbor_list`` and SciPy's ``connected_components``.
    Prints the cluster count and removal indices for disconnected inputs.
    """
    if len(atoms) == 0:
        return None
    sources, targets, _ = _bond_pairs(atoms, cutoff_smear, periodic=True)
    adjacency = sparse.coo_matrix(
        (np.ones(len(sources)), (sources, targets)), shape=(len(atoms), len(atoms))
    )
    n_components, labels = sparse.csgraph.connected_components(
        adjacency, directed=False
    )
    if n_components <= 1:
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
    sources, targets, _ = _bond_pairs(supercell, multi, periodic=True)
    return supercell, central, sources, targets


def _neighbors_of(
    region: np.ndarray, sources: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Select atoms bonded to the region in either edge direction."""
    neighbors = np.zeros(len(region), dtype=bool)
    neighbors[targets[region[sources]]] = True
    neighbors[sources[region[targets]]] = True
    return neighbors


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
    keep = central | _neighbors_of(central, sources, targets)
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
    first_shell = _neighbors_of(central, sources, targets) & ~central
    second_shell = _neighbors_of(first_shell, sources, targets) & ~(
        central | first_shell
    )
    return (
        _wrapped_subset(supercell, central),
        _wrapped_subset(supercell, first_shell),
        _wrapped_subset(supercell, second_shell),
    )


def _auto_reps(atoms: Atoms, mult: float) -> tuple[int, int, int]:
    """
    Choose repetitions whose wrap-around bond graph is guaranteed simple.

    A bond is at most ``c = 2 * max(cutoffs)`` long. An atom bonds to its own
    image only through a lattice vector no longer than ``c``, and a pair
    bonds through two images only when their shifts differ by a lattice
    vector no longer than ``2c``. A lattice vector with a nonzero component
    along direction ``k`` is at least ``reps_k * width_k`` long, where
    ``width_k`` is the perpendicular width of the cell, so requiring
    ``reps_k * width_k > 2c`` rules out both. The bound is sufficient, not
    minimal.
    """
    if len(atoms) == 0:
        return (1, 1, 1)
    longest_bond = 2 * max(natural_cutoffs(atoms, mult=mult))
    cell = atoms.cell.complete()
    volume = abs(np.linalg.det(cell))
    reps = []
    for axis in range(3):
        if not atoms.pbc[axis]:
            reps.append(1)
            continue
        if not np.any(atoms.cell[axis]):
            raise ValueError(f"Periodic direction {axis} has a zero lattice vector.")
        normal = np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3])
        width = volume / np.linalg.norm(normal)
        reps.append(int(np.floor(2 * longest_bond / width)) + 1)
    return tuple(reps)


def _graph_metadata(
    atoms: Atoms, reps: tuple[int, int, int], cutoff_mult: float, periodic: bool
) -> Dict:
    """Return plain-Python graph attributes describing a cell graph."""
    return {
        "reps": tuple(int(r) for r in reps),
        "cutoff_mult": float(cutoff_mult),
        "periodic": periodic,
        "cell": atoms.cell.array.tolist(),
        "pbc": atoms.pbc.tolist(),
    }


def cell_to_nx(
    atoms: Atoms,
    reps: Optional[tuple[int, int, int]] = None,
    cutoff_mult: float = 1.2,
) -> nx.Graph:
    """
    Convert a periodic cell to the bond graph of a wrap-around supercell.

    Nodes are the atoms of ``atoms.repeat(reps)`` and edges are the bonded
    pairs found under the supercell's periodic boundaries, so every atom
    keeps its full coordination and the graph has no surface. Non-periodic
    directions never wrap; repetitions there only stack copies.

    Parameters
    ----------
    atoms : ase.Atoms
        The periodic structure. Its ``pbc`` flags decide which directions
        wrap.
    reps : tuple[int, int, int], optional
        Repetitions along the three cell vectors. Default is ``None``, which
        picks the smallest repetitions that guarantee a simple graph from the
        cell widths and the longest possible bond.
    cutoff_mult : float, optional
        Multiplier for the covalent-radius cutoffs; two atoms are bonded when
        their distance is below ``cutoff_mult`` times the sum of their
        covalent radii. Default is 1.2.

    Returns
    -------
    nx.Graph
        Graph with element symbols in node ``color`` attributes, the source
        atom index in ``cell_index``, the integer cell shift in ``image``,
        and bond order 1 in every edge ``color``. The graph attributes
        ``reps``, ``cutoff_mult``, ``periodic``, ``cell`` and ``pbc`` record
        the model.

    Raises
    ------
    ValueError
        If ``reps`` is not three positive integers, if a periodic direction
        has a zero lattice vector, or if the tiling is too small: an atom
        bonds to its own periodic image or a pair bonds through more than
        one image, either of which would need a self-loop or parallel edge.

    Notes
    -----
    The result is the bond graph of a finite torus and depends on ``reps``
    and ``cutoff_mult``, which should be reported with any result. Covalent
    radii are a crude criterion for ionic and metallic contacts. The
    ``image`` and ``reps`` tuples cannot be written by ``write_graphml``.

    See Also
    --------
    cif_to_nx : Build the graph directly from a CIF file.
    tile_cell : Cut a finite cluster with open boundaries instead.
    """
    if reps is None:
        reps = _auto_reps(atoms, cutoff_mult)
    else:
        try:
            reps = tuple(int(r) for r in reps)
        except TypeError:
            raise ValueError("reps must be three positive integers.") from None
    if len(reps) != 3 or any(r < 1 for r in reps):
        raise ValueError("reps must be three positive integers.")

    supercell = atoms.repeat(reps)
    sources, targets, _ = _bond_pairs(supercell, cutoff_mult, periodic=True)
    self_bonded = np.unique(sources[sources == targets])
    if len(self_bonded):
        raise ValueError(
            f"{len(self_bonded)} atom(s) bond to their own periodic image with "
            f"reps={reps}; increase reps along the periodic directions (the "
            f"automatic choice is {_auto_reps(atoms, cutoff_mult)})."
        )
    mask = sources < targets
    pairs = np.column_stack((sources[mask], targets[mask]))
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    duplicated = int(np.count_nonzero(counts > 1))
    if duplicated:
        raise ValueError(
            f"{duplicated} atom pair(s) bond through more than one periodic image "
            f"with reps={reps}; increase reps along the periodic directions (the "
            f"automatic choice is {_auto_reps(atoms, cutoff_mult)})."
        )

    graph = nx.Graph(**_graph_metadata(atoms, reps, cutoff_mult, periodic=True))
    n_atoms = len(atoms)
    for node, symbol in enumerate(supercell.get_chemical_symbols()):
        block, cell_index = divmod(node, n_atoms)
        image = (
            block // (reps[1] * reps[2]),
            (block // reps[2]) % reps[1],
            block % reps[2],
        )
        graph.add_node(node, color=symbol, cell_index=cell_index, image=image)
    graph.add_edges_from(map(tuple, unique_pairs.tolist()), color=1)
    return graph


def _open_cluster_graph(
    atoms: Atoms, reps: tuple[int, int, int], cutoff_mult: float, eps: float
) -> nx.Graph:
    """Build the graph of the central cell and its first bonded shell."""
    supercell, central, sources, targets = _cell_neighborhood(
        atoms, reps, cutoff_mult, eps
    )
    keep = central | _neighbors_of(central, sources, targets)
    cluster = supercell[keep]
    shells = np.where(central[keep], 0, 1)

    graph = nx.Graph(**_graph_metadata(atoms, reps, cutoff_mult, periodic=False))
    graph.add_nodes_from(
        (i, {"color": atom.symbol, "shell": int(shell)})
        for i, (atom, shell) in enumerate(zip(cluster, shells))
    )
    graph.add_edges_from(get_bonding_config(cluster, mult=cutoff_mult), color=1)
    return graph


def cif_to_nx(
    file: str,
    reps: Optional[tuple[int, int, int]] = None,
    cutoff_mult: float = 1.2,
    eps: float = 1e-9,
    periodic: bool = True,
) -> nx.Graph:
    """
    Convert a CIF file to a NetworkX graph representation.

    Read the primitive cell and build either the wrap-around supercell graph
    of :func:`cell_to_nx` or a finite open cluster made of the central cell
    and its first bonded shell.

    Parameters
    ----------
    file : str
        Path to the CIF file.
    reps : tuple[int, int, int], optional
        Number of repetitions in each direction. Default is ``None``: the
        smallest simple tiling for the periodic graph, or (3, 3, 3) for the
        open cluster.
    cutoff_mult : float, optional
        Multiplier for the covalent-radius bond cutoffs, used both to select
        atoms and to draw edges. Default is 1.2.
    eps : float, optional
        Numerical tolerance for the central-region bounds of the open
        cluster. Ignored when ``periodic`` is True. Default is 1e-9.
    periodic : bool, optional
        Build the wrap-around graph when True, or the open cluster with
        under-coordinated surface atoms when False. Default is True.

    Returns
    -------
    nx.Graph
        Graph with element symbols in node ``color`` attributes and bond
        order 1 in every edge ``color``. Periodic graphs carry
        ``cell_index`` and ``image`` node attributes; open clusters carry
        ``shell`` (0 for the central cell, 1 for the first shell). The graph
        attributes ``reps``, ``cutoff_mult``, ``periodic``, ``cell``,
        ``pbc`` and ``source`` record the model.

    Raises
    ------
    ValueError
        If the tiling is too small for a simple periodic graph; see
        :func:`cell_to_nx`.

    Warns
    -----
    UserWarning
        Always, because the CIF conversion is experimental, and again if the
        file has partially occupied sites; see :func:`read_cif_file`.

    Notes
    -----
    Bond orders are not inferred; every edge has colour 1 and
    :func:`guess_bond_orders` is not called. The graph and its assembly
    index depend on ``reps`` and ``cutoff_mult``.
    """
    warnings.warn("The cif_to_nx function is experimental.", UserWarning)
    atoms = read_cif_file(file)
    if periodic:
        graph = cell_to_nx(atoms, reps=reps, cutoff_mult=cutoff_mult)
    else:
        cluster_reps = (3, 3, 3) if reps is None else tuple(reps)
        graph = _open_cluster_graph(atoms, cluster_reps, cutoff_mult, eps)
    graph.graph["source"] = os.fspath(file)
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
