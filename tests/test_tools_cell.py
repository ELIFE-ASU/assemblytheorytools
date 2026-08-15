import pytest
from ase.visualize import view

import assemblytheorytools as att


def test_cif_loading(data_dir):
    """
    Test loading of CIF files.

    This function iterates through a directory of CIF files, attempts to read each one
    using `att.read_cif_file`, and then visualizes the resulting `atoms` object.
    It skips files known to have invalid spacegroups.
    """
    print(flush=True)
    cif_files = sorted(att.file_list_all(data_dir / "cif_files"))
    loaded = []
    for file in cif_files:
        print(file, flush=True)
        if 'Attakolite_0' in file:  # Attakolite_0 invalid spacegroup C 1 2/m 1
            continue
        if 'Wodginite_3' in file:  # Wodginite_3 invalid spacegroup C 1 2/c 1
            continue
        # input mol file
        atoms = att.read_cif_file(file)
        view(atoms)
        loaded.append(atoms)

    assert len(loaded) == len(cif_files) - 2
    assert all(len(atoms) > 0 for atoms in loaded)


def test_tile_cell(data_dir):
    """
    Test the `tile_cell` and `tile_cell_shells` functions.

    This function loads a CIF file, tiles the unit cell, and asserts the number of atoms
    before and after tiling. It also visualizes the expanded cell and its shells.
    """
    print(flush=True)
    file = str(data_dir / "cif_files" / "Capgaronnite_0.cif")

    # input mol file
    atoms = att.read_cif_file(file)

    n_atoms = len(atoms)
    print(f"Original number of atoms: {n_atoms}", flush=True)

    expanded = att.tile_cell(atoms)
    n_expanded = len(expanded)
    print(f"Expanded number of atoms: {n_expanded}", flush=True)

    assert n_atoms == 16
    assert n_expanded == 34
    # view(expanded)
    expanded, idx_c, idx_1 = att.tile_cell_shells(atoms)

    view(expanded)
    view(idx_c)
    view(idx_1)


def test_cif_to_nx(data_dir):
    """
    Test the conversion of a CIF file to a NetworkX graph.

    This function loads a CIF file, converts it to a NetworkX graph using `att.cif_to_nx`,
    and asserts that the number of nodes in the graph is correct.
    """
    print(flush=True)
    file = str(data_dir / "cif_files" / "Capgaronnite_0.cif")
    with pytest.warns(UserWarning, match="experimental"):
        graph = att.cif_to_nx(file)
    n_nodes = graph.number_of_nodes()
    assert n_nodes == 34


@pytest.mark.slow
def test_cif_ai(data_dir):
    """
    Test the calculation of the assembly index for a CIF file.

    This function loads a CIF file, converts it to a NetworkX graph, calculates the
    assembly index, and asserts that the assembly index is greater than 0.
    """
    print(flush=True)
    file = str(data_dir / "cif_files" / "Capgaronnite_0.cif")
    with pytest.warns(UserWarning, match="experimental"):
        graph = att.cif_to_nx(file)
    ai, _, _ = att.calculate_assembly_index(graph)
    print(ai)
    assert ai > 0


def test_guess_bond_orders(data_dir):
    """
    Test the `guess_bond_orders` function.

    This function tests the bond order guessing for several graphs, including one from a
    CIF file, a water graph, a phosphine graph, a PH2+ graph, and a CO2 graph.
    It asserts the success status and the resulting bond orders for each case.
    """
    print(flush=True)

    def guess(graph):
        with pytest.warns(UserWarning, match="experimental"):
            return att.guess_bond_orders(graph)

    file = str(data_dir / "cif_files" / "Capgaronnite_0.cif")
    with pytest.warns(UserWarning, match="experimental"):
        graph = att.cif_to_nx(file)
    graph_out, ok, info = guess(graph)
    bond_orders = [graph_out.edges[e]["color"] for e in graph_out.edges()]
    print("Success:", ok)
    print("Diagnostics:", info)
    print("Bond orders:", bond_orders)

    graph = att.water_graph()
    graph_out, ok, info = guess(graph)
    bond_orders = [graph_out.edges[e]["color"] for e in graph_out.edges()]
    print("Success:", ok)
    print("Diagnostics:", info)
    print("Bond orders:", bond_orders)
    assert ok
    assert bond_orders == [1, 1]

    graph = att.phosphine_graph()
    graph_out, ok, info = guess(graph)
    bond_orders = [graph_out.edges[e]["color"] for e in graph_out.edges()]
    print("Success:", ok)
    print("Diagnostics:", info)
    print("Bond orders:", bond_orders)
    assert ok
    assert bond_orders == [1, 1, 1]

    graph = att.ph_2p_graph()
    graph_out, ok, info = guess(graph)
    bond_orders = [graph_out.edges[e]["color"] for e in graph_out.edges()]
    print("Success:", ok)
    print("Diagnostics:", info)
    print("Bond orders:", bond_orders)
    assert not ok
    assert bond_orders == [1]

    graph = att.co2_graph()
    graph_out, ok, info = guess(graph)
    bond_orders = [graph_out.edges[e]["color"] for e in graph_out.edges()]
    print("Success:", ok)
    print("Diagnostics:", info)
    print("Bond orders:", bond_orders)
    assert ok
    assert bond_orders == [2, 2]
