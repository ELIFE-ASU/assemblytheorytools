import random

import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw

from .assembly import calculate_assembly_index, get_mol_pathway_to_inchi, convert_pathway_dict_to_list


def get_atom_order(mol):
    """
    This function calculates and returns the order of atoms in a molecule based on their canonical ranks.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The input molecule.

    Returns:
    list: A list of atom indices in the order of their canonical ranks.

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


def mol_with_atom_index(mol):
    """
    Add atom indices as atom map numbers to the molecule.

    Args:
        mol (rdkit.Chem.Mol): The RDKit molecule object.

    Returns:
        rdkit.Chem.Mol: The molecule with atom indices set as atom map numbers.
    """
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def swap_random_elements_list(arr):
    """
    This function swaps two random elements in a list.

    Parameters:
    arr (list): The input list.

    Returns:
    list: The list after swapping two random elements.

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


def plot_fragments(dup_frags, rem_frags, ree_frags, outfile="fragments.png", image_size=(600, 600)):
    """
    Plot fragments in a grid image and save to a file.

    Args:
        dup_frags (list): List of duplicate fragments in InChI format.
        rem_frags (list): List of remnant fragments in InChI format.
        ree_frags (list): List of removed-edges fragments in InChI format.
        outfile (str, optional): The name of the output file. Default is "fragments.png".
        image_size (tuple, optional): The size of each sub-image in the grid. Default is (600, 600).

    Returns:
        None
    """
    im_mat = ensure_equal_length(dup_frags, rem_frags, ree_frags)
    max_length = get_max_list_lengths(im_mat)
    mols_mat = [[Chem.MolFromInchi(inchi) for inchi in row] for row in im_mat]
    leg_mat = ensure_equal_length(["Duplicate Frags"],
                                  ["Remnant Frags"],
                                  ["Removed-Edges Frags"],
                                  max_length=max_length)
    Draw.MolsMatrixToGridImage(mols_mat, legendsMatrix=leg_mat, subImgSize=image_size).save(outfile)


def plot_simple_idx_compare(mol_list, labels=None, outfile="allpath_indexes.png", image_size=(600, 600)):
    """
    Plot a grid image of molecules with atom indices and save to a file.

    Args:
        mol_list (list): List of RDKit molecule objects.
        labels (list, optional): List of labels for each molecule. If None, default labels are generated. Default is None.
        outfile (str, optional): The name of the output file. Default is "allpath_indexes.png".
        image_size (tuple, optional): The size of each sub-image in the grid. Default is (600, 600).

    Returns:
        None
    """
    if labels is None:
        labels = [f"Path {i + 1}" for i in range(len(mol_list))]
    Draw.MolsToGridImage([mol_with_atom_index(mol) for mol in mol_list], subImgSize=image_size).save(outfile)


def all_shortest_paths(mol, f_graph_care=False):
    # Force the input to be an RDKit molecule object
    if not isinstance(mol, Chem.Mol):
        raise ValueError("Input must be an RDKit molecule object.")
    # Get the atom order
    m_order = get_atom_order(mol)
    out_list = []
    # Set the number of attempts to the number of bonds in the molecule
    n_attempts = int(mol.GetNumBonds() * 4)
    for ii in range(n_attempts):
        # Create the new order
        mol_renum = Chem.RenumberAtoms(mol, scramble_list(m_order))
        # Check if the graph is cared for
        if f_graph_care:
            Chem.Kekulize(mol_renum)
        # Calculate the assembly index
        ai, path = calculate_assembly_index(mol_renum)
        # Convert the path to a dict of InChI strings
        path = get_mol_pathway_to_inchi(path)
        # Flatten the dict to a list of InChI strings
        path = convert_pathway_dict_to_list(path)
        # Check if the inchi is not already in the list
        for inchi in path:
            # check if the inchi is not already in the list
            if inchi not in out_list:
                out_list.append(inchi)
    return list(set(out_list))
