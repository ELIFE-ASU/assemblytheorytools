"""Crystal file loading, periodic and finite cell graphs, and bond-order search."""

import warnings
from collections import Counter

import networkx as nx
import numpy as np
import pytest
from ase import Atoms
from ase.spacegroup.spacegroup import SpacegroupNotFoundError

import assemblytheorytools as att
from assemblytheorytools import tools_cell as cell
from assemblytheorytools.tools_graph import write_ass_graph_file

IGNORE_OCCUPANCY = "ignore:.*occupancy:UserWarning"
IGNORE_EXPERIMENTAL = "ignore:The cif_to_nx function is experimental:UserWarning"


@pytest.mark.filterwarnings(IGNORE_OCCUPANCY)
@pytest.mark.parametrize(
    "filename,atom_count",
    [
        ("Arsenstruvite_0.cif", 59),
        ("Attakolite_0.cif", 70),
        ("Capgaronnite_0.cif", 16),
        ("Carlinite_1.cif", 27),
        ("Cristobalite_5.cif", 15),
        ("Lithiophorite_0.cif", 8),
        ("Paravauxite_0.cif", 45),
        ("Pearceite_4.cif", 55),
        ("Tistarite_0.cif", 10),
        ("Wodginite_3.cif", 24),
    ],
)
def test_cif_loading_reads_primitive_cells(data_dir, filename, atom_count):
    atoms = cell.read_cif_file(str(data_dir / "cif_files" / filename))

    assert len(atoms) == atom_count
    assert atoms.pbc.all()
    assert atoms.get_volume() > 0
    assert np.isfinite(atoms.positions).all()


@pytest.mark.filterwarnings(IGNORE_OCCUPANCY)
@pytest.mark.parametrize(
    "filename,atom_count,number,symbol",
    [("Attakolite_0.cif", 70, 12, "C 2/m"), ("Wodginite_3.cif", 24, 15, "C 2/c")],
)
def test_full_monoclinic_symbols_fall_back_to_the_short_form(
    data_dir, filename, atom_count, number, symbol
):
    # ASE's table stores 'C 2/m', not the full 'C 1 2/m 1' these files use.
    atoms = cell.read_cif_file(str(data_dir / "cif_files" / filename))

    assert len(atoms) == atom_count
    assert atoms.info["spacegroup"].no == number
    assert atoms.info["spacegroup"].symbol == symbol


def test_unknown_spacegroup_symbol_still_raises(data_dir, tmp_path):
    text = (data_dir / "cif_files" / "Attakolite_0.cif").read_text()
    assert "'C 1 2/m 1'" in text
    path = tmp_path / "unknown.cif"
    path.write_text(text.replace("'C 1 2/m 1'", "'C 1 2/q 1'"))

    with pytest.raises(SpacegroupNotFoundError):
        cell.read_cif_file(str(path))


def test_read_cif_index_selects_the_structure_block(data_dir, tmp_path):
    cifs = data_dir / "cif_files"
    combined = tmp_path / "two_blocks.cif"
    combined.write_text(
        (cifs / "Carlinite_1.cif").read_text()
        + "\n"
        + (cifs / "Tistarite_0.cif").read_text()
    )

    assert len(cell.read_cif_file(str(combined), index=0)) == 27
    assert len(cell.read_cif_file(str(combined))) == 10
    with pytest.raises(ValueError, match="2 structure block"):
        cell.read_cif_file(str(combined), index=2)

    empty = tmp_path / "empty.cif"
    empty.write_text("data_empty\n_cell_length_a 5.0\n")
    with pytest.raises(ValueError, match="no crystal structure block"):
        cell.read_cif_file(str(empty))


def test_partial_occupancy_warns_with_site_and_atom_counts(data_dir):
    with pytest.warns(UserWarning, match=r"8 of 12 sites .*\(44 of 55 atoms\)"):
        cell.read_cif_file(str(data_dir / "cif_files" / "Pearceite_4.cif"))

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        cell.read_cif_file(str(data_dir / "cif_files" / "Carlinite_1.cif"))


@pytest.fixture
def capgaronnite_path(data_dir):
    return str(data_dir / "cif_files" / "Capgaronnite_0.cif")


@pytest.fixture
def capgaronnite(capgaronnite_path):
    with pytest.warns(UserWarning, match="2 of 5 sites"):
        return cell.read_cif_file(capgaronnite_path)


def _cif_graph(path, **kwargs):
    with pytest.warns(UserWarning) as record:
        graph = cell.cif_to_nx(path, **kwargs)
    messages = [str(warning.message) for warning in record]
    assert any("cif_to_nx function is experimental" in m for m in messages)
    assert any("2 of 5 sites" in m for m in messages)
    return graph


@pytest.fixture
def crystal_graph(capgaronnite_path):
    return _cif_graph(capgaronnite_path)


@pytest.fixture
def crystal_cluster_graph(capgaronnite_path):
    return _cif_graph(capgaronnite_path, periodic=False)


def test_cif_graph_is_a_wrap_around_supercell(
    crystal_graph, capgaronnite, capgaronnite_path
):
    graph = crystal_graph
    assert graph.graph == {
        "reps": (2, 1, 2),
        "cutoff_mult": 1.2,
        "periodic": True,
        "cell": capgaronnite.cell.array.tolist(),
        "pbc": [True, True, True],
        "source": capgaronnite_path,
    }
    assert graph.number_of_nodes() == 64
    assert graph.number_of_edges() == 88
    assert nx.is_connected(graph)
    assert set(nx.get_node_attributes(graph, "color").values()) == {
        "Ag",
        "Cl",
        "Hg",
        "S",
    }
    assert set(nx.get_edge_attributes(graph, "color").values()) == {1}
    assert Counter(dict(graph.degree()).values()) == {1: 16, 2: 16, 4: 32}

    cell_index = nx.get_node_attributes(graph, "cell_index")
    image = nx.get_node_attributes(graph, "image")
    assert sorted(set(cell_index.values())) == list(range(16))
    assert set(image.values()) == {(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)}
    supercell = capgaronnite.repeat((2, 1, 2))
    for node in graph:
        shift = np.array(image[node]) @ capgaronnite.cell.array
        np.testing.assert_allclose(
            supercell.positions[node], capgaronnite.positions[cell_index[node]] + shift
        )


def test_cif_open_cluster_keeps_the_central_cell_and_first_shell(
    crystal_cluster_graph, capgaronnite
):
    graph = crystal_cluster_graph
    assert graph.graph["reps"] == (3, 3, 3)
    assert graph.graph["periodic"] is False
    assert graph.number_of_nodes() == 34
    assert graph.number_of_edges() == 34
    shells = nx.get_node_attributes(graph, "shell")
    assert Counter(shells.values()) == {0: 16, 1: 18}
    assert [node for node in graph if shells[node] == 0] == list(range(9, 25))
    assert set(nx.get_edge_attributes(graph, "color").values()) == {1}

    tiled = cell.tile_cell(capgaronnite)
    assert len(tiled) == 34
    assert {tuple(sorted(edge)) for edge in graph.edges} == {
        tuple(pair) for pair in cell.get_bonding_config(tiled)
    }


@pytest.mark.parametrize(
    "cutoff_mult,node_count,edge_count", [(1.0, 24, 8), (1.5, 66, 182)]
)
def test_open_cluster_cutoff_controls_both_atoms_and_edges(
    capgaronnite_path, cutoff_mult, node_count, edge_count
):
    graph = _cif_graph(capgaronnite_path, periodic=False, cutoff_mult=cutoff_mult)

    assert graph.graph["cutoff_mult"] == cutoff_mult
    assert (graph.number_of_nodes(), graph.number_of_edges()) == (
        node_count,
        edge_count,
    )


@pytest.mark.slow
def test_crystal_assembly_index(crystal_graph):
    assembly_index, _, _ = att.calculate_assembly_index(crystal_graph)

    assert 0 < assembly_index < crystal_graph.number_of_edges()


@pytest.fixture
def water():
    return Atoms(
        "OH2",
        positions=[(0, 0, 0), (0.96, 0, 0), (-0.25, 0.93, 0)],
        cell=[5, 6, 7],
        pbc=True,
    )


def test_bonding_orders_pairs_and_leaves_input_unchanged(water):
    positions = water.positions.copy()
    assert cell.get_bonding_config(water) == [[0, 1], [0, 2]]
    assert cell.get_bonding_config(water, mult=0.9) == []
    assert water.pbc.all()
    np.testing.assert_array_equal(water.cell.array, np.diag([5, 6, 7]))
    np.testing.assert_array_equal(water.positions, positions)


def test_bonding_has_no_skin_beyond_the_multiplier():
    # 2.05 A is inside the old NeighborList cutoff of r_C + r_C + 0.6 A.
    carbons = Atoms("CC", positions=[(0, 0, 0), (2.05, 0, 0)])
    assert cell.get_bonding_config(carbons) == []
    assert cell.get_bonding_config(carbons, mult=1.4) == [[0, 1]]


def test_mol_file_format_and_input_left_unchanged(water, tmp_path):
    path = tmp_path / "water.mol"
    assert cell.atoms_to_mol_file(water, path) is None
    assert path.read_text() == (
        "\nLouie's generator\n\n"
        "  3  2  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    0.9600    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "   -0.2500    0.9300    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "  1  2  1  0  0  0  0\n"
        "  1  3  1  0  0  0  0\n"
        "M  END\n"
    )
    assert water.pbc.all()
    np.testing.assert_array_equal(water.cell.array, np.diag([5, 6, 7]))

    cell.atoms_to_mol_file(water, path, mult=0.9)
    lines = path.read_text().splitlines()
    assert lines[3] == "  3  0  0  0  0  0  0  0  0  0999 V2000"
    assert lines[-1] == "M  END"
    assert len(lines) == 8


def test_cluster_ties_and_diagnostics(capsys):
    atoms = Atoms("H4", positions=[(0, 0, 0), (0.7, 0, 0), (4, 0, 0), (4.7, 0, 0)])
    assert cell.find_clusters(atoms) == [2, 3]
    assert capsys.readouterr().out == (
        "Number of clusters: 2\nAtoms to remove: [2, 3]\n"
    )
    assert cell.find_clusters(atoms[:2]) is None
    assert cell.find_clusters(Atoms()) is None
    assert capsys.readouterr().out == ""


def test_clusters_honour_periodic_boundaries(capsys):
    pair = Atoms("HH", positions=[(0.1, 0, 0), (3.9, 0, 0)], cell=[4, 5, 5], pbc=True)
    assert cell.find_clusters(pair) is None
    pair.set_pbc(False)
    assert cell.find_clusters(pair) == [1]
    assert capsys.readouterr().out == "Number of clusters: 2\nAtoms to remove: [1]\n"


def test_crystal_tiling_separates_central_cell_and_coordination_shells(capgaronnite):
    tiled = cell.tile_cell(capgaronnite)
    central, first, second = cell.tile_cell_shells(capgaronnite)

    assert len(capgaronnite) == len(central) == 16
    assert len(first) == 18
    assert len(second) == 36
    assert len(tiled) == len(central) + len(first) == 34
    tiled_positions = {tuple(position) for position in tiled.positions}
    assert tiled_positions == {
        tuple(position) for region in (central, first) for position in region.positions
    }
    assert tiled_positions.isdisjoint(map(tuple, second.positions))


@pytest.mark.parametrize(
    ("repetitions", "expected_shells", "expected_tile"),
    [
        (1, [[0], [], []], [0]),
        (4, [[1.4], [0.7, 2.1], [0]], [0.7, 1.4, 2.1]),
        (5, [[1.4, 2.1], [0.7, 2.8], [0]], [0.7, 1.4, 2.1, 2.8]),
    ],
)
def test_tiling_shell_boundaries_and_metadata(
    repetitions, expected_shells, expected_tile
):
    atoms = Atoms("H", cell=[0.7, 4, 5], pbc=[True, False, True], info={"name": "H"})
    atoms.set_tags([7])
    original = atoms.copy()
    reps = (repetitions, 1, 1)
    tiled = cell.tile_cell(atoms, reps=reps)
    shells = cell.tile_cell_shells(atoms, reps=reps)
    assert isinstance(shells, tuple)
    assert len(shells) == 3
    for region, expected_x in zip((tiled, *shells), (expected_tile, *expected_shells)):
        np.testing.assert_allclose(region.positions[:, 0], expected_x)
        np.testing.assert_allclose(region.cell, np.diag([0.7 * repetitions, 4, 5]))
        np.testing.assert_array_equal(region.pbc, atoms.pbc)
        np.testing.assert_array_equal(region.get_tags(), [7] * len(region))
        assert region.info == atoms.info
    assert atoms == original


def test_tiling_wraps_positions_without_reordering_atoms():
    atoms = Atoms(
        "HH", positions=[(-0.1, 0, 0), (0.5, 0, 0)], cell=[1.2, 4, 5], pbc=True
    )
    tiled = cell.tile_cell(atoms, reps=(1, 1, 1))
    central, first, second = cell.tile_cell_shells(atoms, reps=(1, 1, 1))
    np.testing.assert_allclose(tiled.positions[:, 0], [1.1, 0.5])
    np.testing.assert_allclose(central.positions[:, 0], [0.5])
    np.testing.assert_allclose(first.positions[:, 0], [1.1])
    assert len(second) == 0


def test_empty_tiling_retains_cell_and_periodicity():
    atoms = Atoms(cell=[1, 2, 3], pbc=[True, False, True])
    for region in (cell.tile_cell(atoms), *cell.tile_cell_shells(atoms)):
        assert len(region) == 0
        np.testing.assert_array_equal(region.cell, atoms.cell * 3)
        np.testing.assert_array_equal(region.pbc, atoms.pbc)


def test_cell_to_nx_builds_the_cube_graph_with_automatic_repetitions():
    atoms = Atoms(
        "C4",
        positions=[(0, 0, 0), (1.5, 0, 0), (0, 0, 1.5), (1.5, 0, 1.5)],
        cell=[3, 10, 10],
        pbc=[True, False, False],
    )
    graph = cell.cell_to_nx(atoms)

    assert graph.graph == {
        "reps": (2, 1, 1),
        "cutoff_mult": 1.2,
        "periodic": True,
        "cell": [[3.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        "pbc": [True, False, False],
    }
    assert list(graph) == list(range(8))
    assert graph.number_of_edges() == 12
    assert nx.is_isomorphic(graph, nx.hypercube_graph(3))
    assert [graph.nodes[node]["cell_index"] for node in graph] == [0, 1, 2, 3] * 2
    assert [graph.nodes[node]["image"] for node in graph] == [(0, 0, 0)] * 4 + [
        (1, 0, 0)
    ] * 4
    assert set(nx.get_node_attributes(graph, "color").values()) == {"C"}
    assert set(nx.get_edge_attributes(graph, "color").values()) == {1}
    assert att.calculate_assembly_index(graph)[0] == 4
    assert att.calculate_assembly_index_rust(graph) == 4


def test_cell_to_nx_wraps_only_the_periodic_directions():
    atoms = Atoms("C", cell=[1.5, 1.5, 1.5], pbc=[True, False, True])
    graph = cell.cell_to_nx(atoms)

    assert graph.graph["reps"] == (3, 1, 3)
    assert graph.number_of_nodes() == 9
    assert graph.number_of_edges() == 18
    assert set(dict(graph.degree()).values()) == {4}
    assert nx.is_isomorphic(graph, nx.grid_2d_graph(3, 3, periodic=True))


@pytest.mark.parametrize(
    "reps,message",
    [
        ((1, 1, 1), r"1 atom\(s\) bond to their own periodic image .* \(3, 3, 3\)"),
        ((2, 2, 2), r"12 atom pair\(s\) bond through more than one periodic image"),
        ((0, 1, 1), "three positive integers"),
        ((2, 2), "three positive integers"),
        (3, "three positive integers"),
    ],
)
def test_cell_to_nx_rejects_tilings_that_need_loops_or_parallel_edges(reps, message):
    atoms = Atoms("C", cell=[1.5, 1.5, 1.5], pbc=True)
    with pytest.raises(ValueError, match=message):
        cell.cell_to_nx(atoms, reps=reps)


def test_cell_to_nx_rejects_a_zero_periodic_lattice_vector():
    atoms = Atoms("C", cell=[[1.5, 0, 0], [0, 0, 0], [0, 0, 1.5]], pbc=True)
    with pytest.raises(ValueError, match="direction 1 has a zero lattice vector"):
        cell.cell_to_nx(atoms)


def test_cell_to_nx_explicit_repetitions_give_an_alternating_ring():
    atoms = Atoms(
        "NaCl",
        positions=[(0, 0, 0), (2.8, 0, 0)],
        cell=[5.6, 10, 10],
        pbc=[True, False, False],
    )
    assert nx.is_isomorphic(cell.cell_to_nx(atoms), nx.cycle_graph(4))

    ring = cell.cell_to_nx(atoms, reps=(4, 1, 1))
    assert ring.graph["reps"] == (4, 1, 1)
    assert nx.is_isomorphic(ring, nx.cycle_graph(8))
    assert all(ring.nodes[a]["color"] != ring.nodes[b]["color"] for a, b in ring.edges)
    assert [ring.nodes[node]["image"] for node in ring] == [
        (i, 0, 0) for i in range(4) for _ in range(2)
    ]
    assert att.calculate_assembly_index(ring)[0] == 3


def test_cell_to_nx_of_an_empty_cell_is_empty():
    graph = cell.cell_to_nx(Atoms(cell=[1, 2, 3], pbc=True))
    assert graph.number_of_nodes() == 0
    assert graph.graph["reps"] == (1, 1, 1)


@pytest.mark.filterwarnings(IGNORE_OCCUPANCY)
@pytest.mark.filterwarnings(IGNORE_EXPERIMENTAL)
@pytest.mark.parametrize(
    "filename,reps,node_count,edge_count",
    [
        ("Arsenstruvite_0.cif", (1, 2, 1), 118, 100),
        ("Attakolite_0.cif", (1, 1, 2), 140, 268),
        ("Capgaronnite_0.cif", (2, 1, 2), 64, 88),
        ("Carlinite_1.cif", (1, 1, 1), 27, 54),
        ("Cristobalite_5.cif", (2, 3, 3), 270, 576),
        ("Lithiophorite_0.cif", (3, 3, 3), 216, 459),
        ("Paravauxite_0.cif", (2, 1, 1), 90, 104),
        ("Pearceite_4.cif", (2, 2, 1), 220, 1712),
        ("Tistarite_0.cif", (2, 2, 2), 80, 400),
        ("Wodginite_3.cif", (2, 2, 2), 192, 640),
    ],
)
def test_every_cif_fixture_gives_a_calculator_ready_graph(
    data_dir, tmp_path, filename, reps, node_count, edge_count
):
    graph = cell.cif_to_nx(str(data_dir / "cif_files" / filename))

    assert graph.graph["reps"] == reps
    assert list(graph) == list(range(node_count))
    assert graph.number_of_edges() == edge_count
    assert not any(u == v for u, v in graph.edges)
    write_ass_graph_file(graph, str(tmp_path / "graph_in"))
    assert (tmp_path / "graph_in").read_text().splitlines()[1] == str(node_count)


@pytest.mark.parametrize(
    "graph_factory,success,bond_orders",
    [
        (att.water_graph, True, [1, 1]),
        (att.phosphine_graph, True, [1, 1, 1]),
        (att.ph_2p_graph, False, [1]),
        (att.co2_graph, True, [2, 2]),
    ],
)
def test_bond_order_search_for_small_molecules(graph_factory, success, bond_orders):
    with pytest.warns(UserWarning, match="guess_bond_orders function is experimental"):
        result, actual_success, _ = cell.guess_bond_orders(graph_factory())

    assert actual_success is success
    assert list(nx.get_edge_attributes(result, "color").values()) == bond_orders


def test_unsolved_crystal_bond_search_retains_original_connectivity(crystal_graph):
    with pytest.warns(UserWarning, match="guess_bond_orders function is experimental"):
        result, success, diagnostics = cell.guess_bond_orders(crystal_graph)

    assert not success
    assert nx.utils.graphs_equal(result, crystal_graph)
    assert diagnostics["success_edges_assigned"] == 0
    assert diagnostics["total_edges"] == crystal_graph.number_of_edges()


@pytest.mark.parametrize(
    (
        "elements",
        "edges",
        "expected_orders",
        "success",
        "targets",
        "tried",
        "backtracks",
    ),
    [
        ("NN", [(0, 1)], [3], True, {0: 3, 1: 3}, 1, 0),
        ("CC", [(0, 1)], [9], False, {0: 4, 1: 4}, 0, 1),
        ("HHC", [(0, 1)], [1], False, {0: 1, 1: 1, 2: 4}, 1, 1),
        ("O", [], [], False, {0: 2}, 0, 0),
        ("", [], [], True, {}, 0, 0),
    ],
)
def test_bond_search_results_and_diagnostics(
    elements, edges, expected_orders, success, targets, tried, backtracks
):
    graph = nx.Graph(name="original")
    graph.add_nodes_from((i, {"color": element}) for i, element in enumerate(elements))
    graph.add_edges_from(edges, color=9, label="preserved")
    with pytest.warns(
        UserWarning, match="The guess_bond_orders function is experimental"
    ):
        result, ok, info = cell.guess_bond_orders(G=graph, max_bond_order=4)

    assert ok is success
    assert result is not graph
    assert result.graph == graph.graph
    assert list(result.nodes(data=True)) == list(graph.nodes(data=True))
    assert [result.edges[edge]["color"] for edge in edges] == expected_orders
    assert all(graph.edges[edge]["color"] == 9 for edge in edges)
    assert all(result.edges[edge]["label"] == "preserved" for edge in edges)
    assert info == {
        "target_valence": targets,
        "remaining_valence_per_atom": dict.fromkeys(targets, 0) if success else targets,
        "tried_edges": tried,
        "backtracks": backtracks,
        "success_edges_assigned": sum(order != 9 for order in expected_orders),
        "total_edges": len(edges),
    }


@pytest.mark.parametrize(
    ("charge", "attribute", "expected"),
    [(1, "charge", 3), (-1, "charge", 2), (1, None, 2)],
)
def test_custom_charge_attribute(charge, attribute, expected):
    graph = nx.Graph()
    graph.add_node("oxygen", color="O", charge=charge)
    with pytest.warns(UserWarning, match="experimental"):
        _, success, info = cell.guess_bond_orders(graph, formal_charge_attr=attribute)
    assert not success
    assert info["target_valence"] == {"oxygen": expected}


@pytest.mark.parametrize("oxygen_first", [True, False])
def test_bond_search_preserves_node_order_behavior(oxygen_first):
    oxygen, hydrogen, other_hydrogen = "oxygen", ("hydrogen", 1), 8
    nodes = (
        [oxygen, hydrogen, other_hydrogen]
        if oxygen_first
        else [hydrogen, oxygen, other_hydrogen]
    )
    graph = nx.Graph()
    graph.add_nodes_from(
        (node, {"color": "O" if node == oxygen else "H"}) for node in nodes
    )
    graph.add_edges_from([(oxygen, hydrogen), (oxygen, other_hydrogen)], color=9)
    with pytest.warns(UserWarning, match="experimental"):
        result, success, info = cell.guess_bond_orders(graph)
    assert success is oxygen_first
    assert list(result) == nodes
    assert list(nx.get_edge_attributes(result, "color").values()) == (
        [1, 1] if oxygen_first else [9, 9]
    )
    assert (info["tried_edges"], info["backtracks"]) == (
        (2, 0) if oxygen_first else (1, 2)
    )
