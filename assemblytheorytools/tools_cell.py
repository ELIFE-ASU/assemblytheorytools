from typing import List

import ase
import numpy as np
from ase.atoms import Atoms
from ase.io import cif
from ase.neighborlist import NeighborList
from ase.neighborlist import neighbor_list, natural_cutoffs
from scipy import sparse


def read_cif_file(cif_file: str) -> Atoms:
    """
    look in to using
    https://github.com/MaterSim/PyXtal
    https://github.com/GKieslich/crystIT
    https://github.com/torbjornbjorkman/cif2cell/tree/master
    Read in a CIF file and return the atom object.

    Args:
        cif_file (str): The path to the CIF file.

    Returns:
        ase.Atoms: The atoms object.
    """
    # Read in the CIF file
    atoms = cif.read_cif(cif_file, primitive_cell=True, subtrans_included=False)
    return atoms


def atoms_to_mol_file(atoms: Atoms, file_name: str = "mol.mol") -> None:
    """
    Write a molecule to a .mol file from an ASE atoms object.

    Args:
        atoms (ase.Atoms): The input set of atoms.
        file_name (str, optional): The name of the output .mol file. Defaults to "mol.mol".

    Returns:
        None
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

    Args:
        atoms (ase.Atoms): The input set of atoms.

    Returns:
        List[List[int]]: A list of bond pairs, where each pair is represented as a list of two atom indices.
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

    Parameters:
    -----------
        atoms (ase.Atoms): An ASE `Atoms` object representing the molecular or periodic structure.
        
        cutoff_smear (float, optional): A multiplicative factor applied to the natural cutoff 
            radii to loosen bonding criteria. Default is 1.5.

    Returns:
    --------
        list or None:
            - If only one cluster exists (fully connected), returns `None`.
            - If multiple clusters are found, returns a list of atom indices that do not 
              belong to the largest connected cluster.

    Notes:
    ------
        - Uses ASE's `NeighborList` for graph construction.
        - Uses SciPy's `connected_components` for clustering.
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


def keep_central_cell_and_bonded(atoms, reps=(3, 3, 3), cutoff_mult=1.2, eps=1e-9):
    """
    Create a supercell, keep atoms in the central region and those bonded to them.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure to process.
    reps : tuple of int, optional
        Number of repetitions along each axis for the supercell (default is (3, 3, 3)).
    cutoff_mult : float, optional
        Multiplier for the natural cutoff distances to determine bonding (default is 1.2).
    eps : float, optional
        Small tolerance to handle numerical precision (default is 1e-9).

    Returns
    -------
    pruned : ase.Atoms
        Pruned atomic structure containing only central atoms and their bonded neighbors.
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
    cutoffs = natural_cutoffs(sup, mult=cutoff_mult)
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
