from typing import List

import ase
import networkx as nx
from ase.atoms import Atoms
from ase.io import cif
from ase.neighborlist import NeighborList, natural_cutoffs


def read_cif_file(cif_file: str) -> Atoms:
    """
    look in to using
    https://github.com/MaterSim/PyXtal
    https://github.com/GKieslich/crystIT
    https://github.com/torbjornbjorkman/cif2cell/tree/master
    Read in a CIF file and return the atoms object.

    Args:
        cif_file (str): The path to the CIF file.

    Returns:
        ase.Atoms: The atoms object.
    """
    # Read in the CIF file
    atoms = cif.read_cif(cif_file, primitive_cell=True, subtrans_included=False)
    return atoms


def atoms_to_mol_file(atoms: Atoms, fname: str = "mol.mol") -> None:
    """
    Write a molecule to a .mol file from an ASE atoms object.

    Args:
        atoms (ase.Atoms): The input set of atoms.
        fname (str, optional): The name of the output .mol file. Defaults to "mol.mol".

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
    with open(fname, "w") as f:
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


def atoms_to_nx(atoms: Atoms) -> nx.Graph:
    """
    Convert an ASE atoms object to a NetworkX graph.

    Args:
        atoms (ase.Atoms): The input set of atoms.

    Returns:
        networkx.Graph: A graph where nodes are atoms and edges are bonds.
    """
    # Get the bonding configuration
    bond_pairs: list[list[int]] = get_bonding_config(atoms)

    # Create a graph
    graph: nx.Graph = nx.Graph()

    # Add nodes with atom indices and elements as attributes
    for i, atom in enumerate(atoms):
        graph.add_node(i, color=atom.symbol)

    # Add edges based on bond pairs
    for bond in bond_pairs:
        graph.add_edge(bond[0], bond[1], color='1')

    return graph
