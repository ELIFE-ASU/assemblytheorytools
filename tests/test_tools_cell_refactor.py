"""Regression coverage for cell geometry, file output, and search behavior."""

import networkx as nx
import numpy as np
import pytest
from ase import Atoms

from assemblytheorytools import tools_cell as cell


@pytest.fixture
def water():
    return Atoms(
        "OH2",
        positions=[(0, 0, 0), (0.96, 0, 0), (-0.25, 0.93, 0)],
        cell=[5, 6, 7],
        pbc=True,
    )


def test_bonding_preserves_pair_order_and_clears_periodicity(water):
    positions = water.positions.copy()
    assert cell.get_bonding_config(water) == [[0, 1], [0, 2]]
    assert not water.pbc.any()
    assert not water.cell.array.any()
    np.testing.assert_array_equal(water.positions, positions)


def test_mol_file_format_and_input_mutation(water, tmp_path):
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
    assert not water.pbc.any()
    assert not water.cell.array.any()


def test_cluster_ties_and_diagnostics(capsys):
    atoms = Atoms("H4", positions=[(0, 0, 0), (0.7, 0, 0), (4, 0, 0), (4.7, 0, 0)])
    assert cell.find_clusters(atoms) == [2, 3]
    assert capsys.readouterr().out == (
        "Number of clusters: 2\nAtoms to remove: [2, 3]\n"
    )
    assert cell.find_clusters(atoms[:2]) is None
    assert capsys.readouterr().out == ""
    with pytest.raises(ValueError, match="argmax"):
        cell.find_clusters(Atoms())


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
