import random
from typing import List, Tuple, Optional

import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import Mol

from .assembly import calculate_assembly_index


def get_atom_order(mol: Mol) -> List[int]:
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


def mol_with_atom_index(mol: Mol) -> Mol:
    """
    Add atom indices as atom map numbers to a molecule.
    
    Sets each atom's map number to match its index in the molecule,
    useful for visualization and tracking atoms through reactions.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule object to modify.
    
    Returns
    -------
    rdkit.Chem.rdchem.Mol
        The same molecule with atom map numbers set to atom indices.
    
    Notes
    -----
    This modifies the molecule in-place and returns the modified object.
    Atom map numbers are commonly used in reaction SMARTS and for
    visualizing atom correspondences.
    """
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def swap_random_elements_list(arr: list[int]) -> list[int]:
    """
    Swap two randomly selected elements in a list.
    
    Selects two distinct random indices and swaps the elements at those
    positions. If the list has fewer than 2 elements, returns unchanged.
    
    Parameters
    ----------
    arr : list of int
        The input list to modify.
    
    Returns
    -------
    list of int
        The list after swapping two random elements.
    
    Notes
    -----
    - Returns the original list if it has fewer than 2 elements
    - Uses numpy.random.choice for index selection
    - Modifies the list in-place and returns it
    """
    n_atoms = len(arr)
    # Check if the array has at least 2 elements
    if n_atoms < 2:
        return arr

    # Select two distinct random indices
    idx1, idx2 = np.random.choice(n_atoms, 2, replace=False)

    # Swap elements at these indices
    arr[idx1], arr[idx2] = arr[idx2], arr[idx1]

    return arr


def scramble_list(lst: list) -> list:
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


def swap_random_elements(lst: list) -> list:
    """
    Swap two randomly selected elements in a list.
    
    Selects two distinct random indices using Python's random.sample
    and swaps the elements at those positions.
    
    Parameters
    ----------
    lst : list
        The input list to modify.
    
    Returns
    -------
    list
        The list after swapping two random elements.
    
    Notes
    -----
    - Returns the original list unchanged if it has fewer than 2 elements
    - Uses random.sample for index selection
    - Modifies the list in-place and returns it
    """

    # Check if the list has at least 2 elements
    if len(lst) < 2:
        return lst

    # Select two distinct random indices
    idx1, idx2 = random.sample(range(len(lst)), 2)

    # Swap elements at these indices
    lst[idx1], lst[idx2] = lst[idx2], lst[idx1]

    return lst


def pad_list(lst: list, length: int) -> list:
    """
    Pad a list with empty strings to a specified length.
    
    Extends the list by appending empty strings until it reaches the
    desired length. If the list is already at or beyond the target length,
    returns it unchanged.
    
    Parameters
    ----------
    lst : list
        The input list to pad.
    length : int
        The desired length of the padded list.
    
    Returns
    -------
    list
        The padded list with empty strings appended if needed.
    
    Notes
    -----
    If len(lst) >= length, the original list is returned unchanged.
    """
    return lst + [''] * (length - len(lst))


def get_max_list_lengths(lists: list[list]) -> int:
    """
    Calculate the maximum length among a collection of lists.
    
    Parameters
    ----------
    lists : list of list
        Collection of lists to analyze.
    
    Returns
    -------
    int
        The maximum length among all input lists.
    
    Examples
    --------
    >>> get_max_list_lengths([[1, 2], [1, 2, 3], [1]])
    3
    """
    return max([len(lst) for lst in lists])


def ensure_equal_length(l1: list,
                        l2: list,
                        l3: list,
                        max_length: None | int = None) -> list[list]:
    """
    Ensure three lists have equal length by padding with empty strings.
    
    Pads all input lists to the same length by appending empty strings.
    The target length is either specified or determined from the longest
    input list.
    
    Parameters
    ----------
    l1 : list
        First input list.
    l2 : list
        Second input list.
    l3 : list
        Third input list.
    max_length : int, optional
        Target length for all lists. If None, uses the maximum length
        among the three input lists, by default None.
    
    Returns
    -------
    list of list
        Three lists [l1_padded, l2_padded, l3_padded] all with equal length.
    
    Examples
    --------
    >>> ensure_equal_length([1, 2], [1, 2, 3], [1])
    [[1, 2, ''], [1, 2, 3], [1, '', '']]
    """
    max_length = max_length or max(map(len, [l1, l2, l3]))
    return [l + [''] * (max_length - len(l)) for l in [l1, l2, l3]]


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

    m_order = get_atom_order(mol)
    out_list = []
    n_attempts = int(mol.GetNumBonds() * 4)
    no_new_vo_count = 0

    for ii in range(n_attempts):
        if no_new_vo_count >= max_attempts:
            break

        mol_renum = Chem.RenumberAtoms(mol, scramble_list(m_order))
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
