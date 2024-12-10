import networkx as nx
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs


def read_cif_file(cif_file):
    """
    look in to using
    https://github.com/MaterSim/PyXtal
    https://github.com/GKieslich/crystIT
    Read in a CIF file and return the atoms object.

    Args:
        cif_file (str): The path to the CIF file.

    Returns:
        ase.Atoms: The atoms object.
    """
    # Read in the CIF file
    atoms = read(cif_file)
    return atoms


def atoms_to_mol_file(atoms, fname="mol.mol"):
    """
    Write a molecule to a .mol file from an ASE atoms object.

    Args:
        atoms (ase.Atoms): The input set of atoms.
        fname (str, optional): The name of the output .mol file. Defaults to "mol.mol".

    Returns:
        None
    """
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
        out_str += str(bond[0] + 1).rjust(3) + str(bond[1] + 1).rjust(3) + "  1  0  0  0  0\n"
    out_str += "M  END\n"

    # Write the molecule to a file
    with open(fname, "w") as f:
        f.write(out_str)
    return None


def get_bonding_config(atoms):
    """
    Generate the bonding configuration for a given set of atoms.

    Args:
        atoms (ase.Atoms): The input set of atoms.

    Returns:
        list: A list of bond pairs, where each pair is represented as a list of two atom indices.
    """
    nl = NeighborList(natural_cutoffs(atoms))
    nl.update(atoms)
    bond_pairs = []
    for i in range(len(atoms)):
        indices, _ = nl.get_neighbors(i)
        for idx in indices[indices != i]:
            bond_pairs.append([i, idx])
    return bond_pairs


def atoms_to_nx(atoms):
    """
    Convert an ASE atoms object to a NetworkX graph.

    Args:
        atoms (ase.Atoms): The input set of atoms.

    Returns:
        networkx.Graph: A graph where nodes are atoms and edges are bonds.
    """
    # Get the bonding configuration
    bond_pairs = get_bonding_config(atoms)

    # Create a graph
    G = nx.Graph()

    # Add nodes with atom indices and elements as attributes
    for i, atom in enumerate(atoms):
        G.add_node(i, color=atom.symbol)

    # Add edges based on bond pairs
    for bond in bond_pairs:
        G.add_edge(bond[0], bond[1], color='1')

    return G
