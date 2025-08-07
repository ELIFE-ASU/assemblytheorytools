import random
from typing import List, Tuple, Optional

import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import Mol

from .assembly import calculate_assembly_index


def get_atom_order(mol: Mol) -> List[int]:
    """
    This function calculates and returns the order of atoms in a molecule based on their canonical ranks.

    Parameters:
        mol (rdkit.Chem.rdchem.Mol): The input molecule.

    Returns:
        List[int]: A list of atom indices in the order of their canonical ranks.

    The function works as follows:
    1. It calculates the canonical ranks of the atoms in the molecule.
    2. It pairs each atom's canonical rank with its index and sorts these pairs.
    3. It extracts and returns the atom indices from the sorted list of pairs.
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
    Add atom indices as atom map numbers to the molecule.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object.

    Returns:
        rdkit.Chem.rdchem.Mol: The molecule with atom indices set as atom map numbers.
    """
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def swap_random_elements_list(arr: list[int]) -> list[int]:
    """
    This function swaps two random elements in a list.

    Parameters:
        arr (list[int]): The input list.

    Returns:
        list[int]: The list after swapping two random elements.

    The function works as follows:
    1. It checks if the list has at least 2 elements. If not, it returns the original list.
    2. It selects two distinct random indices from the list.
    3. It swaps the elements at these indices in the list.
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


def scramble_list(lst):
    """
    This function shuffles the elements in a list in-place and returns the shuffled list.

    Parameters:
    lst (list): The input list.

    Returns:
    list: The list after shuffling its elements.

    The function works as follows:
    1. It creates a copy of the input list.
    2. It shuffles the elements in the copied list in-place using numpy's random.shuffle function.
    3. It returns the shuffled list.
    """
    out = lst.copy()
    np.random.shuffle(out)
    return out


def swap_random_elements(lst):
    """
    This function swaps two random elements in a list.

    Parameters:
    lst (list): The input list.

    Returns:
    list: The list after swapping two random elements.

    The function works as follows:
    1. It checks if the list has at least 2 elements. If not, it returns the original list.
    2. It selects two distinct random indices from the list.
    3. It swaps the elements at these indices in the list.
    """

    # Check if the list has at least 2 elements
    if len(lst) < 2:
        return lst

    # Select two distinct random indices
    idx1, idx2 = random.sample(range(len(lst)), 2)

    # Swap elements at these indices
    lst[idx1], lst[idx2] = lst[idx2], lst[idx1]

    return lst


def pad_list(lst, length):
    """
    This function pads a list with empty strings to a specified length.

    Parameters:
    lst (list): The input list.
    length (int): The desired length of the list after padding.

    Returns:
    list: The list after padding. If the original list is shorter than the specified length,
    empty strings are appended to the list until it reaches the specified length.
    If the original list is longer or equal to the specified length, it is returned as is.
    """
    return lst + [''] * (length - len(lst))


def get_max_list_lengths(lists):
    """
    This function calculates and returns the maximum length among a list of lists.

    Parameters:
    lists (list of list): The input list of lists.

    Returns:
    int: The maximum length among the lists.

    The function works as follows:
    1. It calculates the length of each list in the input list of lists.
    2. It returns the maximum of these lengths.
    """
    return max([len(lst) for lst in lists])


def ensure_equal_length(l1, l2, l3, max_length=None):
    """
    This function ensures that all input lists have the same length by padding them with empty strings.

    Parameters:
    l1, l2, l3 (list): The input lists.
    max_length (int, optional): The desired length of the lists after padding. If not provided, the maximum length of the input lists is used.

    Returns:
    list of list: The input lists after padding. If an original list is shorter than the specified length,
    empty strings are appended to the list until it reaches the specified length.
    If an original list is longer or equal to the specified length, it is returned as is.
    """
    max_length = max_length or max(map(len, [l1, l2, l3]))
    return [l + [''] * (max_length - len(l)) for l in [l1, l2, l3]]


def plot_vo(dup_vo: List[str], rem_vo: List[str], ree_vo: List[str], outfile: str = "virtual_objects.png",
            image_size: Tuple[int, int] = (600, 600)) -> None:
    """
    Plot virtual objects in a grid image and save to a file.

    Args:
        dup_vo (List[str]): List of duplicate virtual objects in InChI format.
        rem_vo (List[str]): List of remnant virtual objects in InChI format.
        ree_vo (List[str]): List of removed-edges virtual objects in InChI format.
        outfile (str, optional): The name of the output file. Default is "virtual_objects.png".
        image_size (Tuple[int, int], optional): The size of each sub-image in the grid. Default is (600, 600).

    Returns:
        None
    """
    im_mat = ensure_equal_length(dup_vo, rem_vo, ree_vo)
    max_length = get_max_list_lengths(im_mat)
    mols_mat = [[Chem.MolFromInchi(inchi) for inchi in row] for row in im_mat]
    leg_mat = ensure_equal_length(["Duplicates"],
                                  ["Remnants"],
                                  ["Removed-Edges"],
                                  max_length=max_length)
    Draw.MolsMatrixToGridImage(mols_mat, legendsMatrix=leg_mat, subImgSize=image_size).save(outfile)
    return None


def plot_simple_idx_compare(mol_list: List[Mol], labels: Optional[List[str]] = None,
                            outfile: str = "allpath_indexes.png", image_size: Tuple[int, int] = (600, 600)) -> None:
    """
    Plot a grid image of molecules with atom indices and save to a file.

    Args:
        mol_list (List[Mol]): List of RDKit molecule objects.
        labels (Optional[List[str]], optional): List of labels for each molecule. If None, default labels are generated. Default is None.
        outfile (str, optional): The name of the output file. Default is "allpath_indexes.png".
        image_size (Tuple[int, int], optional): The size of each sub-image in the grid. Default is (600, 600).

    Returns:
        None
    """
    if labels is None:
        labels = [f"Path {i + 1}" for i in range(len(mol_list))]
    Draw.MolsToGridImage([mol_with_atom_index(mol) for mol in mol_list], subImgSize=image_size, legends=labels).save(
        outfile)
    return None


def all_shortest_paths(mol: Mol, f_graph_care: bool = False, max_attempts: int = 3) -> List[str]:
    """
    Generate all unique shortest paths of a molecule by scrambling atom indices.

    Args:
        mol (rdkit.Chem.Mol): The input RDKit molecule object.
        f_graph_care (bool, optional): Whether to kekulize the molecule. Default is False.
        max_attempts (int, optional): Maximum number of consecutive attempts without finding new VOs.

    Returns:
        List[str]: A list of unique VOs SMILES strings representing the shortest paths.
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
