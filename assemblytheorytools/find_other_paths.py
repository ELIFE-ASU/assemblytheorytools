from typing import List

import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdchem import Mol

from .assembly import calculate_assembly_index


def _get_atom_order(mol: Mol) -> List[int]:
    """
    Calculate canonical atom ordering for a molecule.
    
    Computes the canonical ranks of atoms and returns their indices sorted
    by rank. This provides a consistent atom ordering based on molecular
    structure and chirality.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The input RDKit molecule object.
    
    Returns
    -------
    list of int
        List of atom indices ordered by their canonical ranks.
    
    Notes
    -----
    The function uses RDKit's CanonicalRankAtoms which considers:
    1. Atomic connectivity and properties
    2. Chiral centers
    3. Graph symmetry
    
    This ensures consistent atom ordering across isomorphic molecules.
    """
    # Calculate the canonical ranks of the atoms
    ranks = Chem.CanonicalRankAtoms(mol, includeChirality=True)

    # Pair each atom's canonical rank with its index and sort these pairs
    ranked_atoms = sorted(enumerate(ranks), key=lambda x: x[1])

    # Extract and return the atom indices from the sorted list of pairs
    atom_order = [atom_index for atom_index, rank in ranked_atoms]

    return atom_order


def _scramble_list(lst: list) -> list:
    """
    Randomly shuffle list elements.
    
    Creates a copy of the input list and shuffles it in-place using
    numpy's random shuffle algorithm.
    
    Parameters
    ----------
    lst : list
        The input list to shuffle.
    
    Returns
    -------
    list
        A shuffled copy of the input list.
    
    Notes
    -----
    The original list is not modified. Uses numpy.random.shuffle
    which implements the Fisher-Yates shuffle algorithm.
    """
    out = lst.copy()
    np.random.shuffle(out)
    return out


def all_shortest_paths(mol: Mol,
                       f_graph_care: bool = False,
                       max_attempts: int = 3) -> List[str]:
    """
    Generate all unique shortest paths of a molecule by scrambling atom indices.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The input RDKit molecule object.
    f_graph_care : bool, optional
        Whether to kekulize the molecule, by default False.
    max_attempts : int, optional
        Maximum number of consecutive attempts without finding new virtual
        objects (VOs) before terminating the search, by default 3.

    Returns
    -------
    List[str]
        A list of unique virtual object (VO) InChI strings representing the
        shortest paths.

    Raises
    ------
    ValueError
        If the input is not an RDKit molecule object.

    Notes
    -----
    The function uses atom index scrambling to explore different molecular
    representations and identify all unique virtual objects. The total number
    of attempts is calculated as 4 times the number of bonds in the molecule,
    but the search terminates early if no new VOs are found for `max_attempts`
    consecutive iterations.
    """
    if not isinstance(mol, Chem.Mol):
        raise ValueError("Input must be an RDKit molecule object.")

    m_order = _get_atom_order(mol)
    out_list = []
    n_attempts = int(mol.GetNumBonds() * 4)
    no_new_vo_count = 0

    for ii in range(n_attempts):
        if no_new_vo_count >= max_attempts:
            break

        mol_renum = Chem.RenumberAtoms(mol, _scramble_list(m_order))
        if f_graph_care:
            Chem.Kekulize(mol_renum)

        ai, virt_obj, _ = calculate_assembly_index(mol_renum)

        new_inchi_found = False
        for vo in virt_obj:
            if vo not in out_list:
                out_list.append(vo)
                new_inchi_found = True

        if new_inchi_found:
            no_new_vo_count = 0
        else:
            no_new_vo_count += 1

    return list(set(out_list))
