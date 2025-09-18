import os

import ase.build
import numpy as np
from ase.io import read
from ase.visualize import view

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


def test_keep_central_cell_and_bonded():
    print(flush=True)
    dir = os.path.expanduser(os.path.abspath("tests/data/cif_files/"))
    file = os.path.join(dir, "Capgaronnite_0.cif")

    # input mol file
    atoms = att.read_cif_file(file)

    n_atoms = len(atoms)
    print(f"Original number of atoms: {n_atoms}", flush=True)

    expanded = att.keep_central_cell_and_bonded(atoms)
    n_expanded = len(expanded)
    print(f"Expanded number of atoms: {n_expanded}", flush=True)

    assert n_atoms == 16
    assert n_expanded == 34


def test_cif_to_nx():
    print(flush=True)
    dir = os.path.expanduser(os.path.abspath("tests/data/cif_files/"))
    file = os.path.join(dir, "Capgaronnite_0.cif")
    graph = att.cif_to_nx(file)
    n_nodes = graph.number_of_nodes()
    assert n_nodes == 34
