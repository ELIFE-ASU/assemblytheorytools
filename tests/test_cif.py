import os
import shutil
import numpy as np
import networkx as nx
from rdkit import Chem
import ase.build
from ase.visualize import view
from ase.io import read
from rdkit.Chem import AllChem as Chem
import numpy as np
from ase.neighborlist import neighbor_list, natural_cutoffs
import assemblytheorytools as att


def test_cif_loading():
    print(flush=True)
    target_dir = "tests/data/cif_files/"
    dirs = att.file_list_all(os.path.expanduser(os.path.abspath(target_dir)))
    dirs.sort()
    print(dirs, flush=True)
    for file in dirs:
        print(file, flush=True)
        if 'Attakolite_0' in file:  # Attakolite_0 invalid spacegroup C 1 2/m 1
            continue
        if 'Wodginite_3' in file:  # Wodginite_3 invalid spacegroup C 1 2/c 1
            continue
        # input mol file
        atoms = att.read_cif_file(file)
        # tmp = ase.geometry.minkowski_reduce(atoms)
        # print(tmp)

        # ase.build.niggli_reduce(atoms)
        ase.build.tools.niggli_reduce(atoms)
        ase.build.tools.reduce_lattice(atoms)

        view(atoms)
        tmp_file = file.split('.')[0] + ".mol"
        att.atoms_to_mol_file(atoms, file_name=tmp_file)
        atoms2 = read(tmp_file)
        view(atoms2)
        # os.remove(tmp_file)
        # check that the atoms are the same
        assert np.allclose(atoms.get_positions(), atoms2.get_positions(), rtol=1e-04, atol=1e-04)
        assert np.allclose(atoms.get_atomic_numbers(), atoms2.get_atomic_numbers())


def test_cif_ai():
    print(flush=True)
    target_dir = "tests/data/cif_files/"
    dirs = att.file_list_all(os.path.expanduser(os.path.abspath(target_dir)))
    file = dirs[0]

    # input mol file
    atoms = att.read_cif_file(file)
    tmp_file = file.split('.')[0] + ".mol"
    att.atoms_to_mol_file(atoms, file_name=tmp_file)
    ai_mol, _, _ = att.calculate_assembly_index(tmp_file, joint_corr=False)

    os.remove(tmp_file)

    graph = att.atoms_to_nx(atoms)
    ai_graph, _, _ = att.calculate_assembly_index(graph, joint_corr=False)

    assert ai_mol == ai_graph == 4


def keep_central_cell_and_bonded(atoms, reps=(3, 3, 3), cutoff_mult=1.1):
    # 1) Build supercell. (Central image is already centered for odd reps.)
    sup = atoms.repeat(reps)

    # 2) Identify atoms that lie in the *central* image of the supercell.
    #    For odd reps r, the central slab in fractional coords is:
    #    [ (r-1)/(2r), (r+1)/(2r) )
    f = sup.get_scaled_positions(wrap=False)
    reps = np.array(reps, dtype=float)
    low = np.where(reps > 1, (reps - 1) / (2 * reps), 0.0)
    high = np.where(reps > 1, (reps + 1) / (2 * reps), 1.0)
    eps = 1e-9
    in_central = np.all((f >= (low - eps)) & (f < (high + eps)), axis=1)
    central_idx = np.where(in_central)[0]

    # 3) Neighbor list (periodic), using ASE’s element-based "natural" cutoffs.
    #    Tune cutoff_mult if you need looser/tighter bonding (e.g. 1.0–1.3).
    cutoffs = natural_cutoffs(sup, mult=cutoff_mult)
    i, j = neighbor_list('ij', sup, cutoffs)

    # 4) Find all neighbors bonded to *central* atoms.
    mask_i_central = np.isin(i, central_idx)
    bonded_to_central = set(j[mask_i_central])

    # 5) Keep central atoms + their bonded neighbors; delete everything else.
    keep = np.array(sorted(set(central_idx) | bonded_to_central), dtype=int)
    mask = np.zeros(len(sup), dtype=bool)
    mask[keep] = True

    pruned = sup.copy()
    del pruned[~mask]  # remove unneeded atoms
    pruned.set_cell(sup.cell)  # keep the supercell cell
    pruned.set_pbc(sup.pbc)
    pruned.wrap()

    return pruned

def test_keep_central_cell_and_bonded():
    print(flush=True)
    target_dir = "tests/data/cif_files/"
    dirs = att.file_list_all(os.path.expanduser(os.path.abspath(target_dir)))
    file = dirs[0]

    # input mol file
    atoms = att.read_cif_file(file)
    view(atoms)
    pruned = keep_central_cell_and_bonded(atoms, reps=(3, 3, 3), cutoff_mult=1.1)
    view(pruned)
