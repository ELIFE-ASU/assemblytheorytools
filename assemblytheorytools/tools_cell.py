import warnings
from typing import Dict, Tuple, Optional, Iterable
from typing import List

import ase
import networkx as nx
import numpy as np
from ase import Atoms
from ase.io import cif
from ase.neighborlist import NeighborList, neighbor_list, natural_cutoffs
from rdkit import Chem
from scipy import sparse


def read_cif_file(cif_file: str) -> Atoms:
    """
    Read in a CIF file and return the atom object.
    
    Alternative libraries to consider:
    - https://github.com/MaterSim/PyXtal
    - https://github.com/GKieslich/crystIT
    - https://github.com/torbjornbjorkman/cif2cell/tree/master

    Parameters
    ----------
    cif_file : str
        The path to the CIF file.

    Returns
    -------
    ase.Atoms
        The atoms object.
    """
    # Read in the CIF file
    atoms = cif.read_cif(cif_file, primitive_cell=True, subtrans_included=False)
    return atoms


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
    """
    # Get the bonding configuration
    bond_pairs: List[List[int]] = get_bonding_config(atoms)

    # Get the number of atoms
    n_atoms: int = len(atoms)
    # Get the number of bonds
    n_bonds: int = len(bond_pairs)
    # Write the header
    out_str: str = "\nLouie's generator\n\n"
    out_str += str(n_atoms).rjust(3) + str(n_bonds).rjust(3) + "  0  0  0  0  0  0  0  0999 V2000" + "\n"

    # Get the positions and elements
    pos = atoms.get_positions()
    ele = atoms.get_chemical_symbols()

    end_part: str = " 0  0  0  0  0  0  0  0  0  0  0  0\n"
    # Write the atoms block
    for i in range(n_atoms):
        x, y, z = pos[i]
        out_str += f"{x:.4f}".rjust(10) + f"{y:.4f}".rjust(10) + f"{z:.4f}".rjust(10) + " " + ele[i].ljust(3) + end_part

    # Write the bonds block
    for bond in bond_pairs:
        out_str += str(bond[0] + 1).rjust(3) + str(bond[1] + 1).rjust(3) + "  1  0  0  0  0\n"
    out_str += "M  END\n"

    # Write the molecule to a file
    with open(file_name, "w") as f:
        f.write(out_str)
    return None


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
        A list of bond pairs, where each pair is represented as a list of two atom indices.
    """
    atoms.set_pbc([False, False, False])
    atoms.cell = [0, 0, 0]
    neighbor_list = NeighborList(natural_cutoffs(atoms))
    neighbor_list.update(atoms)
    bond_pairs: List[List[int]] = []
    for i in range(len(atoms)):
        indices, _ = neighbor_list.get_neighbors(i)
        for idx in indices[indices != i]:
            bond_pairs.append([i, idx])
    return bond_pairs


def find_clusters(atoms, cutoff_smear=1.5):
    """
    Identify disconnected atom clusters in an atomic structure using a neighbor-based graph.

    This function builds a connectivity graph of atoms based on natural covalent radii (optionally smeared), 
    and identifies whether the atomic system is fully connected. If multiple disconnected clusters exist, 
    it returns the indices of atoms not belonging to the largest cluster.

    Parameters
    ----------
    atoms : ase.Atoms
        An ASE `Atoms` object representing the molecular or periodic structure.
    cutoff_smear : float, optional
        A multiplicative factor applied to the natural cutoff radii to loosen bonding criteria. Default is 1.5.

    Returns
    -------
    list or None
        If only one cluster exists (fully connected), returns `None`.
        If multiple clusters are found, returns a list of atom indices that do not 
        belong to the largest connected cluster.

    Notes
    -----
    Uses ASE's `NeighborList` for graph construction.
    Uses SciPy's `connected_components` for clustering.
    """
    # Get the natural cutoffs
    cutoffs = natural_cutoffs(atoms)
    # Apply the smear to the cutoffs
    cutoffs = [cutoff_smear * cutoff for cutoff in cutoffs]
    # Get the neighbour list
    neighbor_list = NeighborList(cutoffs, self_interaction=False, bothways=True)
    neighbor_list.update(atoms)
    # Get the connectivity matrix
    n_components, component_list = sparse.csgraph.connected_components(neighbor_list.get_connectivity_matrix())
    if n_components == 1:
        return None
    else:
        # Select the atoms in the largest component
        atoms_in_component = [i for i, c in enumerate(component_list) if c == np.argmax(np.bincount(component_list))]
        atoms_to_remove = [i for i in range(len(atoms)) if i not in atoms_in_component]
        print("Number of clusters:", n_components)
        print("Atoms to remove:", atoms_to_remove)
        return atoms_to_remove


def tile_cell(atoms: Atoms,
              reps: tuple[int, int, int] = (3, 3, 3),
              multi: float = 1.2,
              eps: float = 1e-9
              ) -> Atoms:
    """
    Create a tiled supercell with central region and bonded atoms.
    
    This function creates a supercell by repeating the unit cell, identifies
    atoms in the central region, and includes atoms bonded to the central region.

    Parameters
    ----------
    atoms : ase.Atoms
        The input atomic structure.
    reps : tuple[int, int, int], optional
        Number of repetitions in each direction (x, y, z). Default is (3, 3, 3).
    multi : float, optional
        Multiplier for natural cutoff distances in bonding determination. Default is 1.2.
    eps : float, optional
        Small epsilon value for numerical tolerance in region definition. Default is 1e-9.

    Returns
    -------
    ase.Atoms
        Pruned supercell containing central atoms and their bonded neighbors.
    """
    # Create a supercell and get scaled positions
    sup = atoms.repeat(reps)
    scaled_positions = sup.get_scaled_positions(wrap=False)
    reps = np.array(reps, dtype=float)

    # Define central region bounds
    low, high = (reps - 1) / (2 * reps), (reps + 1) / (2 * reps)
    low[reps == 1], high[reps == 1] = 0.0, 1.0

    # Identify atoms in the central region
    in_central = np.all((scaled_positions >= (low - eps)) & (scaled_positions < (high + eps)), axis=1)
    central_idx = np.where(in_central)[0]

    # Find bonded atoms
    cutoffs = natural_cutoffs(sup, mult=multi)
    i, j = neighbor_list('ij', sup, cutoffs)
    bonded_to_central = set(j[np.isin(i, central_idx)])

    # Keep central atoms and bonded atoms
    keep_indices = np.array(sorted(set(central_idx) | bonded_to_central), dtype=int)
    mask = np.zeros(len(sup), dtype=bool)
    mask[keep_indices] = True

    # Prune atoms and retain supercell properties
    pruned = sup[mask]
    pruned.set_cell(sup.cell)
    pruned.set_pbc(sup.pbc)
    pruned.wrap()

    return pruned


def tile_cell_shells(
        atoms: Atoms,
        reps: tuple[int, int, int] = (3, 3, 3),
        multi: float = 1.2,
        eps: float = 1e-9
) -> tuple[Atoms, Atoms, Atoms]:
    """
    Create a tiled supercell and separate atoms into central and shell regions.
    
    This function creates a supercell and identifies atoms in the central region,
    first coordination shell, and second coordination shell based on bonding connectivity.

    Parameters
    ----------
    atoms : ase.Atoms
        The input atomic structure.
    reps : tuple[int, int, int], optional
        Number of repetitions in each direction (x, y, z). Default is (3, 3, 3).
    multi : float, optional
        Multiplier for natural cutoff distances in bonding determination. Default is 1.2.
    eps : float, optional
        Small epsilon value for numerical tolerance in region definition. Default is 1e-9.

    Returns
    -------
    central_atoms : ase.Atoms
        Atoms in the central region of the supercell.
    first_shell_atoms : ase.Atoms
        Atoms in the first coordination shell around the central region.
    second_shell_atoms : ase.Atoms
        Atoms in the second coordination shell around the central region.
    """
    # Build supercell
    sup = atoms.repeat(reps)
    scaled_positions = sup.get_scaled_positions(wrap=False)
    reps_arr = np.array(reps, dtype=float)

    # Central region bounds in scaled coordinates
    low = (reps_arr - 1) / (2 * reps_arr)
    high = (reps_arr + 1) / (2 * reps_arr)
    low[reps_arr == 1.0] = 0.0
    high[reps_arr == 1.0] = 1.0

    in_central = np.all(
        (scaled_positions >= (low - eps)) & (scaled_positions < (high + eps)),
        axis=1
    )
    central_idx = np.where(in_central)[0]
    central_set = set(map(int, central_idx))

    # Neighbor list
    cutoffs = natural_cutoffs(sup, mult=multi)
    i, j = neighbor_list('ij', sup, cutoffs)

    # Neighbor helper that returns a set (symmetric neighbors)
    def neighbors_of_set(index_set: set[int]) -> set[int]:
        if not index_set:
            return set()
        idx_arr = np.fromiter(index_set, dtype=int)
        mask_i = np.isin(i, idx_arr)
        mask_j = np.isin(j, idx_arr)
        nbrs = np.concatenate([j[mask_i], i[mask_j]])
        return set(map(int, np.unique(nbrs)))

    # Shells as sets
    first_shell_set = neighbors_of_set(central_set) - central_set
    second_candidates = neighbors_of_set(first_shell_set)
    second_shell_set = second_candidates - central_set - first_shell_set

    # To arrays
    central_idx = np.array(sorted(central_set), dtype=int)
    first_shell_idx = np.array(sorted(first_shell_set), dtype=int)
    second_shell_idx = np.array(sorted(second_shell_set), dtype=int)

    # Helper to subset while preserving cell/PBC
    def subset_atoms(indices: np.ndarray) -> Atoms:
        mask = np.zeros(len(sup), dtype=bool)
        mask[indices] = True
        sub = sup[mask]
        sub.set_cell(sup.cell)
        sub.set_pbc(sup.pbc)
        sub.wrap()
        return sub

    return (
        subset_atoms(central_idx),
        subset_atoms(first_shell_idx),
        subset_atoms(second_shell_idx),
    )


def cif_to_nx(file,
              reps: tuple[int, int, int] = (3, 3, 3),
              cutoff_mult: float = 1.2,
              eps: float = 1e-9) -> nx.Graph:
    """
    Convert a CIF file to a NetworkX graph representation.
    
    This function reads a CIF file, expands the unit cell, and creates a graph
    where nodes represent atoms and edges represent bonds.

    Parameters
    ----------
    file : str
        Path to the CIF file.
    reps : tuple[int, int, int], optional
        Number of repetitions in each direction for supercell expansion. Default is (3, 3, 3).
    cutoff_mult : float, optional
        Multiplier for natural cutoff distances in bonding determination. Default is 1.2.
    eps : float, optional
        Small epsilon value for numerical tolerance. Default is 1e-9.

    Returns
    -------
    nx.Graph
        NetworkX graph with nodes representing atoms (with 'color' attribute for element symbol)
        and edges representing bonds (with 'color' attribute for bond order).
    """
    # Raise a warning that the code is experimental
    warnings.warn("The cif_to_nx function is experimental.", UserWarning)
    # Load the original cell
    atoms = read_cif_file(file)
    # Expand the cell
    expanded = tile_cell(atoms,
                         reps=reps,
                         multi=cutoff_mult,
                         eps=eps)
    # Make a graph
    graph = nx.Graph()
    # Add nodes
    for i, atom in enumerate(expanded):
        graph.add_node(i, color=atom.symbol)
    # Add edges based on bonding within the expanded cell
    bonding = get_bonding_config(expanded)
    for bond in bonding:
        graph.add_edge(bond[0], bond[1], color=1)

    # Prune to original cluster and its bonded atoms and their neighbors

    # Replace nearest neighbours +1 with H atoms

    #

    return graph


def guess_bond_orders(
        G: nx.Graph,
        formal_charge_attr: Optional[str] = "formal_charge",
        max_bond_order: int = 4,
) -> Tuple[nx.Graph, bool, Dict]:
    """
    Assign bond orders to edges in a molecular graph using constraint satisfaction.
    
    This function uses backtracking search with constraint propagation to assign
    bond orders that satisfy atomic valence requirements based on periodic table data.

    Parameters
    ----------
    G : nx.Graph
        Input molecular graph with nodes having 'color' attribute (element symbol).
    formal_charge_attr : Optional[str], optional
        Attribute name for formal charge on nodes. Default is "formal_charge".
    max_bond_order : int, optional
        Maximum allowed bond order. Default is 4.

    Returns
    -------
    G_with_orders : nx.Graph
        Graph with bond orders assigned to edge 'color' attributes.
    success : bool
        True if all valence constraints were satisfied, False otherwise.
    info : Dict
        Diagnostic information including target valences, remaining valences,
        and search statistics.
    """

    # Raise a warning that the code is experimental
    warnings.warn("The guess_bond_orders function is experimental.", UserWarning)

    pt = Chem.GetPeriodicTable()
    H = G.copy()
    # Normalize node data and prepare per-atom target valences
    atomic_num: Dict = {}
    target_valence: Dict = {}
    charge: Dict = {}

    # Helper to pick a plausible target valence given degree constraints
    def choose_target_valence(Z: int, needed_min: int, q: int) -> int:
        # RDKit's valence list already accounts (approximately) for common valence states.
        vlist: Iterable[int] = pt.GetValenceList(Z)
        vlist = sorted(set(int(v) for v in vlist if v > 0))
        # Heuristic: adjust with charge for main group atoms (very rough but helpful)
        # Positive charge typically increases valence capacity by ~1 (e.g., [NH4]+),
        # negative can decrease required sigma-bonds (e.g., [O-]).
        # We'll bias, but still ensure >= needed_min.
        bias = 1 if q > 0 else 0
        candidates = [v for v in vlist if v + bias >= needed_min]
        if candidates:
            # prefer the smallest that fits (more common)
            return min(candidates) + bias
        # If nothing fits, fall back to default/max
        dv = pt.GetDefaultValence(Z)
        if dv >= needed_min:
            return dv + bias
        mx = max(pt.GetValenceList(Z))
        return max(needed_min, int(mx))

    # Build per-atom records
    for n, data in H.nodes(data=True):
        elem = data.get("color", None)
        Z = pt.GetAtomicNumber(elem)
        if Z == 0:
            raise ValueError(f"Node {n} has unknown element symbol: {elem}")
        q = int(data.get(formal_charge_attr, 0)) if (formal_charge_attr and formal_charge_attr in data) else 0

        deg = H.degree[n]  # number of incident bonds to assign
        tv = choose_target_valence(Z, deg, q)
        atomic_num[n] = Z
        charge[n] = q
        target_valence[n] = tv

    # Track residual valence and edge domains
    residual: Dict = {n: target_valence[n] for n in H.nodes()}
    assigned: Dict = {}  # edge -> order
    tried_edges = 0
    backtracks = 0

    # Initialize each edge's domain: 1..max_bond_order, limited by each endpoint's residual
    def edge_domain(u, v):
        r = min(residual[u], residual[v], max_bond_order)
        return [o for o in (1, 2, 3) if o <= r]

    # Feasibility check after tentative assignment: can each node still be satisfied?
    def feasible_after(u, v, order) -> bool:
        # Tentatively reduce residuals
        ru = residual[u] - order
        rv = residual[v] - order
        if ru < 0 or rv < 0:
            return False

        # For each endpoint, the remaining edges must be able to absorb remaining residual
        for a, ra in ((u, ru), (v, rv)):
            # Unassigned incident edges:
            rem_edges = [e for e in H.edges(a) if e not in assigned and e != (u, v) and e != (v, u)]
            m = len(rem_edges)
            if m == 0:
                # must have no remaining residual
                if ra != 0:
                    return False
                continue
            # Each remaining edge contributes at least 1 and at most max_bond_order,
            # but capped by the other node's residual as well.
            # Lower bound of total we can still add:
            min_sum = 0
            max_sum = 0
            for e in rem_edges:
                x, y = e
                other = y if x == a else x
                max_here = min(max_bond_order, ra if m == 1 else ra, residual[other])  # upper bound
                max_here = max(0, max_here)
                max_sum += max_here
                min_sum += 1  # at least a single bond per remaining edge

            # ra must lie between min_sum and max_sum (inclusive) to keep hope alive
            if ra < min_sum or ra > max_sum:
                return False
        return True

    # Select next edge (MRV: smallest domain)
    def select_edge():
        best = None
        best_domain = None
        for (u, v) in H.edges():
            if (u, v) in assigned or (v, u) in assigned:
                continue
            dom = edge_domain(u, v)
            if not dom:
                return (u, v), []
            if best is None or len(dom) < len(best_domain):
                best = (u, v)
                best_domain = dom
        return best, best_domain if best is not None else (None, None)

    # Backtracking search
    best_partial = {}
    best_score = -1  # number of atoms fully satisfied

    def score_solution() -> int:
        return sum(1 for n in H.nodes() if residual[n] == 0)

    def search() -> bool:
        nonlocal tried_edges, backtracks, best_partial, best_score
        edge, dom = select_edge()
        if edge is None:
            # all edges assigned: feasible if all residuals are zero
            done = all(residual[n] == 0 for n in H.nodes())
            if not done:
                sc = score_solution()
                if sc > best_score:
                    best_score = sc
                    best_partial = assigned.copy()
            return done
        if not dom:
            # dead end early
            sc = score_solution()
            if sc > best_score:
                best_score = sc
                best_partial = assigned.copy()
            return False

        u, v = edge
        # Heuristic: try higher orders first if both have large residuals
        dom_sorted = sorted(dom, reverse=True if residual[u] > 2 and residual[v] > 2 else False)
        for order in dom_sorted:
            if not feasible_after(u, v, order):
                continue
            # assign
            tried_edges += 1
            assigned[(u, v)] = order
            residual[u] -= order
            residual[v] -= order

            if search():
                return True

            # undo
            residual[u] += order
            residual[v] += order
            assigned.pop((u, v), None)

        backtracks += 1
        return False

    success = search()

    # Commit assignments (best available if not perfect)
    final_assignments = assigned if success else best_partial
    for (u, v), order in final_assignments.items():
        H.edges[u, v]["color"] = int(order)

    # Prepare diagnostics
    info = {
        "target_valence": target_valence,
        "remaining_valence_per_atom": {n: residual[n] for n in H.nodes()},
        "tried_edges": tried_edges,
        "backtracks": backtracks,
        "success_edges_assigned": len(final_assignments),
        "total_edges": H.number_of_edges(),
    }

    return H, success, info
