import os
import random

import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw

from .pathway import get_pathway_to_inchi
from .assembly import calculate_assembly_index
from .plotting import n_plot

import matplotlib.pyplot as plt

plt.rcParams['axes.linewidth'] = 2.0


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
    im_mat = ensure_equal_length(dup_frags, rem_frags, ree_frags)
    max_length = get_max_list_lengths(im_mat)
    mols_mat = [[Chem.MolFromInchi(inchi) for inchi in row] for row in im_mat]
    leg_mat = ensure_equal_length(["Duplicate Frags"],
                                  ["Remnant Frags"],
                                  ["Removed-Edges Frags"],
                                  max_length=max_length)
    Draw.MolsMatrixToGridImage(mols_mat, legendsMatrix=leg_mat, subImgSize=image_size).save(outfile)


def plot_simple_idx_compare(mol_list, labels=None, outfile="allpath_indexes.png", image_size=(600, 600)):
    if labels is None:
        labels = [f"Path {i + 1}" for i in range(len(mol_list))]
    # , legendsMatrix=labels
    Draw.MolsToGridImage([mol_with_atom_index(mol) for mol in mol_list], subImgSize=image_size).save(outfile)


def all_shortest_paths(mol, f_graph_care=False):
    m_order = get_atom_order(mol)
    inchi_list = []
    # Set the number of attempts to the number of bonds in the molecule
    n_bonds = mol.GetNumBonds()
    print(f"Number of bonds: {n_bonds}", flush=True)
    n_attempts = int(mol.GetNumBonds() * 4)
    print(f"Number of attempts: {n_attempts}", flush=True)
    for ii in range(n_attempts):
        # print(f"Attempt {ii}", flush=True)
        idx_neworder = scramble_list(m_order)
        m_renum = Chem.RenumberAtoms(mol, idx_neworder)
        dir_path = os.path.join(os.getcwd(), f"path_{ii}")
        # Check if the directory exists if not create it
        os.makedirs(dir_path, exist_ok=True)

        # Check if the graph is cared for
        if f_graph_care:
            Chem.Kekulize(m_renum)

        # Write the molecule to a mol file
        Chem.MolToMolFile(m_renum, os.path.join(dir_path, f"path_{ii}.mol"))
        # Calculate the assembly index
        calculate_assembly_index(os.path.join(dir_path, f"path_{ii}.mol"))
        # Get the pathway
        dup_frags, rem_frags, ree_frags = get_pathway_to_inchi(os.path.join(dir_path, f"path_{ii}Pathway"))
        # print("duplicate_fragments", dup_frags, flush=True)
        # print("remnant_fragments", rem_frags, flush=True)
        # print("removed_edges_fragments", ree_frags, flush=True)
        combined = dup_frags + rem_frags + ree_frags
        for inchi in combined:
            # check if the inchi is not already in the list
            if inchi not in inchi_list:
                inchi_list.append(inchi)

    return inchi_list


if __name__ == "__main__":
    print("Program started", flush=True)
    # ala arg asp asn cys gln glu gly his ile leu lys met phe pro ser thr trp tyr val
    smiles = ["N[C@]([H])(C)C(=O)N[C@]([H])(CC(C)C)C(=O)N[C@]([H])(C)C(=O)O",  # ala
              "N[C@]([H])(C)C(=O)N[C@]([H])(CCCNC(=N)N)C(=O)NCC(=O)O",  # arg
              "N[C@]([H])(C)C(=O)N[C@]([H])(CO)C(=O)N1[C@]([H])(CCC1)C(=O)O",  # asp
              "N[C@]([H])(C)C(=O)N[C@]([H])(CO)C(=O)N[C@]([H])(CC(=O)N)C(=O)O"  # asn
              "N[C@]([H])(C)C(=O)N[C@]([H])(CS)C(=O)N[C@]([H])(CC(=O)N)C(=O)O",  # cys
              "NCC(=O)N[C@]([H])(CC(C)C)C(=O)N[C@]([H])(CC(=O)N)C(=O)O",  # gln
              "NCC(=O)N[C@]([H])(CC(C)C)C(=O)O",  # glu
              "NCC(=O)N[C@]([H])(CC(C)C)C(=O)N[C@]([H])(Cc1ccc(O)cc1)C(=O)O",  # gly
              "N[C@]([H])(CC1=CN=C-N1)C(=O)N[C@]([H])([C@@]([H])(CC)C)C(=O)N[C@]([H])(CO)C(=O)O",  # his
              "N[C@]([H])([C@@]([H])(CC)C)C(=O)N[C@]([H])(CC(C)C)C(=O)N[C@]([H])(CCC(=O)O)C(=O)O",  # ile
              "N[C@]([H])(CC(C)C)C(=O)N[C@]([H])(CCC(=O)O)C(=O)O",  # leu
              "N[C@]([H])(CC(C)C)C(=O)N[C@]([H])(Cc1ccc(O)cc1)C(=O)N[C@]([H])(CO)C(=O)O",  # lys
              "N[C@]([H])(CCSC)C(=O)N[C@]([H])(CCC(=O)O)C(=O)N[C@]([H])([C@@]([H])(O)C)C(=O)O",  # met
              "N1[C@]([H])(CCC1)C(=O)N[C@]([H])(CC1=CN=C-N1)C(=O)N[C@]([H])(CCC(=O)O)C(=O)O",  # phe
              "N[C@]([H])(CO)C(=O)N[C@]([H])(CCC(=O)O)C(=O)N[C@]([H])(CCCNC(=N)N)C(=O)O",  # ser
              "N[C@]([H])([C@@]([H])(O)C)C(=O)N[C@]([H])(CC1=CN=C-N1)C(=O)N[C@]([H])(CCCNC(=N)N)C(=O)O",  # thr
              "N[C@]([H])([C@@]([H])(O)C)C(=O)N[C@]([H])(CCCNC(=N)N)C(=O)N1[C@]([H])(CCC1)C(=O)O",  # trp
              "N[C@]([H])([C@@]([H])(O)C)C(=O)N[C@]([H])(Cc1ccc(O)cc1)C(=O)N[C@]([H])(CCCNC(=N)N)C(=O)O",  # tyr
              "N[C@]([H])(C(C)C)C(=O)N[C@]([H])(C)C(=O)N[C@]([H])(CC(C)C)C(=O)O"  # val
              ]

    n_paths = np.zeros(len(smiles))
    n_bonds = np.zeros(len(smiles))
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        n_bonds[i] = m.GetNumBonds()
        n_paths[i] = len(all_shortest_paths(m))
        print("Number of unique paths: ", n_paths[i], flush=True)

    plt.scatter(n_bonds, n_paths)
    n_plot("Number of Bonds", "Number of Unique virtual objects")
    plt.savefig("unique_objects.png", dpi=600)
    plt.savefig("unique_objects.pdf")
    plt.close()

    # m = Chem.MolFromSmiles("c1([C@H](C)CC)cccc2ccccc12")
    # n = all_shortest_paths(m)
    # print("Number of unique paths: ", n, flush=True)

    print("Program ended", flush=True)

    # n_attempts = 10
    # # Create a molecule
    # m = Chem.MolFromSmiles("c1([C@H](C)CC)cccc2ccccc12")
    # m = Chem.MolFromSmiles("C1CC2=C3C(=CC=C2)C(=CN3C1)[C@H]4[C@@H](C(=O)NC4=O)C5=CNC6=CC=CC=C65")
    # # m = Chem.MolFromSmiles("CCCCCCCCCCCCOS(=O)([O-])=O.[Na+]")
    # m_order = get_atom_order(m)
    # mol_list = []
    # print("Original atom order: ", m_order)
    # for i in range(n_attempts):
    #     ii = i + 1
    #     print(f"Attempt {ii}")
    #     idx_neworder = scramble_list(m_order)  # swap_random_elements_list(m_order)
    #     m_renum = Chem.RenumberAtoms(m, idx_neworder)
    #     print("New atom order: ", get_atom_order(m_renum))
    #     dir_path = os.path.join(os.getcwd(), f"path_{ii}")
    #     # Check if the directory exists if not create it
    #     os.makedirs(dir_path, exist_ok=True)
    #     # Chem.Kekulize(m_renum)
    #     # MolStandardize.rdMolStandardize.FragmentParent(m_renum)
    #     # Write the molecule to a mol file
    #     Chem.MolToMolFile(m_renum, os.path.join(dir_path, f"path_{ii}.mol"))
    #
    #     # Load the molecule
    #     mol_list.append(Chem.MolFromMolFile(os.path.join(dir_path, f"path_{ii}.mol")))
    #
    #     # Calculate the assembly index
    #     u.calculate_assembly(os.path.join(dir_path, f"path_{ii}.mol"))
    #     # Get the pathway
    #     inchi_list = pti.pathway_to_inchi_list(os.path.join(dir_path, f"path_{ii}Pathway"))
    #     print(inchi_list)
    #     mol_list = [Chem.MolFromInchi(inchi) for inchi in inchi_list]
    #     Draw.MolsToGridImage(mol_list).save(f"allpath_{ii}.png")
    #     dup_frags, rem_frags, ree_frags = pti.pathway_to_inchi_sublist(os.path.join(dir_path, f"path_{ii}Pathway"))
    #     print("duplicate_fragments", dup_frags)
    #     print("remnant_fragments", rem_frags)
    #     print("removed_edges_fragments", ree_frags)
    #     plot_fragments(dup_frags, rem_frags, ree_frags, outfile=f"fragments_{ii}.png")
    #
    # # Add atom indices to the molecules
    # mol_list = [u.addAtomIndices(mol) for mol in mol_list]
    # Draw.MolsToGridImage((u.addAtomIndices(mol_list[0]), u.addAtomIndices(mol_list[1]))).save(f"allpath_indexes.png")
    # Draw.MolsToGridImage(mol_list).save(f"allpath_indexes.png")
    # plot_simple_idx_compare(mol_list)
