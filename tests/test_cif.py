import os

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
        view(atoms)
        pass


def test_tile_cell():
    print(flush=True)
    dir = os.path.expanduser(os.path.abspath("tests/data/cif_files/"))
    file = os.path.join(dir, "Capgaronnite_0.cif")

    # input mol file
    atoms = att.read_cif_file(file)

    n_atoms = len(atoms)
    print(f"Original number of atoms: {n_atoms}", flush=True)

    expanded = att.tile_cell(atoms)
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


def test_cif_ai():
    print(flush=True)
    dir = os.path.expanduser(os.path.abspath("tests/data/cif_files/"))
    file = os.path.join(dir, "Capgaronnite_0.cif")
    graph = att.cif_to_nx(file)
    ai, _, _ = att.calculate_assembly_index(graph)
    print(ai)
    assert ai > 0
