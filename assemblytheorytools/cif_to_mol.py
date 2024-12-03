import os

import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs


def file_list_all(mypath=None):
    """
    This function generates a list of all files in a specified directory and its subdirectories.
    If no directory is specified, it defaults to the current working directory.

    Parameters:
    mypath (str, optional): The path to the directory. Defaults to None, which means the current working directory.

    Returns:
    list: A list of all files in the specified directory and its subdirectories.
    """
    mypath = mypath or os.getcwd()  # If no path is provided, use the current working directory
    files = []
    # os.walk generates the file names in a directory tree by walking the tree either top-down or bottom-up
    for dirpath, dirnames, filenames in os.walk(mypath):
        for filename in filenames:
            # os.path.join joins one or more path components intelligently
            files.append(os.path.join(dirpath, filename))
    return files


def write_mol_file(atoms, fname="mol.mol"):
    # Get the bonding configuration
    bond_pairs = get_bonding_config(atoms)

    # Get the number of atoms
    n_atoms = len(atoms)
    # Get the number of bonds
    n_bonds = len(bond_pairs)
    # Write the header
    out_str = "\nLouie's generator\n\n"
    out_str += str(n_atoms).rjust(3) + str(n_bonds).rjust(3) + "  0  0  0  0  0  0  0  0999 V2000" + "\n"

    # Get the positions and elements
    pos = atoms.get_positions()
    ele = atoms.get_chemical_symbols()

    end_part = " 0  0  0  0  0  0  0  0  0  0  0  0\n"
    # Write the atoms block
    for i in range(n_atoms):
        x, y, z = pos[i]
        out_str += f"{x:.4f}".rjust(10) + f"{y:.4f}".rjust(10) + f"{z:.4f}".rjust(10) + " " + ele[i].ljust(
            3) + end_part

    # Write the bonds block
    for bond in bond_pairs:
        out_str += str(bond[0]).rjust(3) + str(bond[1]).rjust(3) + "  1  0\n"
    out_str += "M  END\n"

    # Write the molecule to a file
    with open(fname, "w") as f:
        f.write(out_str)
    return None


def get_bonding_config(atoms):
    # Get the natural cutoffs and then generate the neighbor list
    nl = NeighborList(natural_cutoffs(atoms))
    # Update the neighbor list
    nl.update(atoms)
    bond_pairs = []
    for i in range(len(atoms)):
        # Get the neighbors of the atom
        indices, offsets = nl.get_neighbors(i)
        # Remove self from list
        indices = indices[indices != i]
        for idx in indices:
            bond_pairs.append([i, idx])
    return bond_pairs


def test_if_same():
    # input mol file
    atoms = read("Diamond_0.mol")
    write_mol_file(atoms, fname=f"Diamond_0_out.mol")
    atoms2 = read("Diamond_0_out.mol")
    # check that the atoms are the same
    assert np.allclose(atoms.get_positions(), atoms2.get_positions())
    assert np.allclose(atoms.get_atomic_numbers(), atoms2.get_atomic_numbers())


# if __name__ == "__main__":
#     # test_if_same()
#     file_path = os.path.join(os.getcwd(), "Data/CIF_Files_curated")
#     output_dir = os.path.join(os.getcwd(), "Data_processed")
#     files = file_list_all(file_path)
#     for i, file in enumerate(files):
#         print(f"Processing {i + 1}/{len(files)}: {file}")
#         try:
#             atoms = read(file)
#             # get just the file name
#             file_name = os.path.basename(file)
#
#             file_out = os.path.join(output_dir, f"{file_name.split(".")[0]}.mol")
#             write_mol_file(atoms, fname=file_out)
#         except Exception as e:
#             print(f"Failed to process file {file}")
