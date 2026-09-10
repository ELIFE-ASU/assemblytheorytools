"""Colored graph neighborhoods and valid assembly joining operations."""

from copy import deepcopy
from unittest.mock import Mock

import networkx as nx
import pytest

import assemblytheorytools as att
import assemblytheorytools.neighborhood_enumeration as neighborhood

NODE_MATCH = nx.algorithms.isomorphism.categorical_node_match("color", None)
EDGE_MATCH = nx.algorithms.isomorphism.categorical_edge_match("color", None)


def _colored_graph(edges, colors="C", bond_order=1):
    graph = nx.Graph(edges)
    nx.set_node_attributes(graph, colors, "color")
    nx.set_edge_attributes(graph, bond_order, "color")
    return graph


def _isomorphic(first, second):
    return nx.is_isomorphic(first, second, node_match=NODE_MATCH, edge_match=EDGE_MATCH)


def _assert_same_graphs(actual, expected):
    remaining = list(actual)
    for graph in expected:
        matches = [
            index
            for index, candidate in enumerate(remaining)
            if _isomorphic(graph, candidate)
        ]
        assert matches, (
            f"Missing graph with nodes {list(graph.nodes(data=True))} and edges {list(graph.edges(data=True))}"
        )
        remaining.pop(matches[0])
    assert not remaining


def _assert_join_operations_conserve_bonds(result):
    inputs, neighbors = result["input_graphs"], result["N_graphs"]
    for first, second, source in result["down_jos"]:
        assert (
            neighbors[first].number_of_edges() + neighbors[second].number_of_edges()
            == inputs[source].number_of_edges()
        )
    for first, second, target in result["up_jos"]:
        assert (
            inputs[first].number_of_edges() + inputs[second].number_of_edges()
            == neighbors[target].number_of_edges()
        )


@pytest.mark.parametrize(
    "edges, expected_count",
    [
        ([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (3, 6)], 21),
        ([(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6)], 6),
    ],
    ids=["cycle-with-tail", "branched-tree"],
)
def test_down_enumeration_partitions_every_edge_once(edges, expected_count):
    graph = nx.Graph(edges)

    partitions = neighborhood.enumerate_down(graph)

    assert len(partitions) == expected_count
    expected_edges = {frozenset(edge) for edge in edges}
    for partition in partitions:
        assert len(partition) == 2
        first, second = [{frozenset(edge) for edge in part} for part in partition]
        assert first and second
        assert first.isdisjoint(second)
        assert first | second == expected_edges


def test_joining_two_paths_produces_all_five_colored_topologies():
    path = _colored_graph([(0, 1), (1, 2)])
    expected = [
        _colored_graph(nx.path_graph(5).edges),
        _colored_graph([(0, 1), (1, 2), (2, 3), (2, 4)]),
        _colored_graph(nx.star_graph(4).edges),
        _colored_graph(nx.cycle_graph(4).edges),
        _colored_graph([(0, 1), (1, 2), (2, 3), (0, 2)]),
    ]

    products = neighborhood.enumerate_up(path, path)

    unique = []
    for product in products:
        if not any(_isomorphic(product, other) for other in unique):
            unique.append(product)
    _assert_same_graphs(unique, expected)


def test_joining_unequal_paths_preserves_all_bonds():
    first = _colored_graph(nx.path_graph(6).edges)
    second = _colored_graph(nx.path_graph(4).edges)

    products = neighborhood.enumerate_up(first, second)

    assert products
    assert all(product.number_of_edges() == 8 for product in products)
    assert all(nx.is_connected(product) for product in products)


@pytest.mark.parametrize(
    "multicolored", [False, True], ids=["carbon", "carbon-and-phosphorus"]
)
def test_neighborhood_is_invariant_under_node_relabeling(multicolored):
    size = 3 if multicolored else 4
    graphs = [_colored_graph(nx.path_graph(size).edges) for _ in range(2)]
    if multicolored:
        graphs[0].nodes[0]["color"] = "P"
        graphs[1].nodes[1]["color"] = "P"
    scrambled = [
        att.scramble_node_indices(graph, seed=seed)
        for graph, seed in zip(graphs, [42, 45])
    ]

    original = neighborhood.enumerate_neighborhood(graphs)
    relabeled = neighborhood.enumerate_neighborhood(scrambled)

    assert original.keys() == relabeled.keys()
    _assert_same_graphs(original["input_graphs"], relabeled["input_graphs"])
    _assert_same_graphs(original["N_graphs"], relabeled["N_graphs"])
    for key in ["down_jos", "up_jos"]:
        assert len(original[key]) == len(relabeled[key])
    _assert_join_operations_conserve_bonds(original)
    _assert_join_operations_conserve_bonds(relabeled)


def test_metabolic_molecule_neighborhood_conserves_bonds_in_each_join():
    # Acetate, pyruvate and fumarate exercise the rTCA example's charged graphs.
    smiles = ["CC(=O)[O-]", "CC(=O)C(=O)[O-]", r"C(=C/C(=O)[O-])\C(=O)[O-]"]
    graphs = [att.smi_to_nx(smile, add_hydrogens=False) for smile in smiles]

    result = neighborhood.enumerate_neighborhood(graphs)

    assert result["N_graphs"]
    assert result["down_jos"]
    assert result["up_jos"]
    _assert_join_operations_conserve_bonds(result)


@pytest.mark.parametrize("element, valence", [("C", None), ("S", {"S": 4})])
@pytest.mark.parametrize("full_neighborhood", [False, True], ids=["up", "neighborhood"])
def test_double_bonds_join_at_the_central_atom(element, valence, full_neighborhood):
    bond = _colored_graph([(0, 1)], {0: element, 1: "O"}, bond_order=2)
    expected = _colored_graph(
        [(0, 1), (1, 2)], {0: "O", 1: element, 2: "O"}, bond_order=2
    )

    if full_neighborhood:
        products = neighborhood.enumerate_neighborhood(
            [bond], custom_valence_table=valence
        )["N_graphs"]
    else:
        products = neighborhood.enumerate_up(bond, bond, custom_valence_table=valence)

    assert any(_isomorphic(product, expected) for product in products)


def test_debug_output_explains_valence_and_mapping_counts_without_changing_products(
    capsys,
):
    bond = _colored_graph([(0, 1)], {0: "S", 1: "O"}, bond_order=2)
    options = {"custom_valence_table": {"S": 4}}
    expected = neighborhood.enumerate_up(bond, bond, **options)
    assert capsys.readouterr().out == ""

    products = neighborhood.enumerate_up(bond, bond, debug=True, **options)

    _assert_same_graphs(products, expected)
    assert len(products) == 1
    lines = set(capsys.readouterr().out.splitlines())
    assert {
        "Checking valence budgets...",
        "Node 0 in graph 1 has color S and valence budget 4.0",
        "Node 1 in graph 1 has color O and valence budget 2.0",
        "Valence budgets for graph1: [2. 0.]",
        "Valence budgets for graph2: [2. 0.]",
        "Number of valid identifications = 0",
        "Number of valid identifications = 1",
        "Number of valid color-specific maps = 2",
        "Number of valid maps = 1",
    } <= lines


def test_saturated_single_bonds_admit_no_joins_or_partitions():
    graphs = [
        att.smi_to_nx(smiles, add_hydrogens=False, sanitize=False)
        for smiles in ["N#N", "C#O", "O=O", "[H][H]"]
    ]

    result = neighborhood.enumerate_neighborhood(graphs, obey_valence=True)

    assert result["up_jos"] == set()
    assert result["down_jos"] == set()


@pytest.mark.parametrize("allow_dots", [True, False])
def test_enumerate_down_preserves_partition_order_and_edge_orientation(allow_dots):
    graph = nx.Graph([(3, 2), (2, 1), (1, 0)])
    graph.add_node(4)
    original = deepcopy(graph)

    assert neighborhood.enumerate_down(graph, allow_dots=allow_dots) == [
        [[(2, 3)], [(2, 1), (1, 0)]],
        [[(1, 2), (2, 3)], [(1, 0)]],
    ]
    assert nx.utils.graphs_equal(graph, original)


@pytest.mark.parametrize("edges", [[], [(0, 1)]])
def test_enumerate_down_requires_two_nonempty_parts(edges):
    graph = nx.Graph(edges)
    graph.add_node(2)
    assert neighborhood.enumerate_down(graph) == []


def test_enumerate_down_disconnected_union_requires_allow_dots():
    graph = nx.Graph([(0, 1), (2, 3)])
    assert neighborhood.enumerate_down(graph) == [[[(0, 1)], [(2, 3)]]]
    assert neighborhood.enumerate_down(graph, allow_dots=False) == []


def test_map_outer_product_returns_and_updates_the_single_color_set():
    maps = {frozenset(), frozenset({(0, 0)})}
    combinations = {"C": maps}

    result = neighborhood.map_outer_product(combinations, nx.Graph(), nx.Graph())

    assert result is maps
    assert maps == {frozenset({(0, 0)})}


@pytest.mark.parametrize(
    "combinations, expected_type",
    [
        ({}, list),
        ({"C": set()}, set),
        ({"C": {frozenset()}}, set),
        ({"C": set(), "O": set()}, list),
    ],
)
def test_map_outer_product_discards_empty_maps(combinations, expected_type):
    result = neighborhood.map_outer_product(combinations, nx.Graph(), nx.Graph())
    assert isinstance(result, expected_type)
    assert not result


def test_map_outer_product_ignores_colors_without_valid_maps():
    maps = {frozenset(), frozenset({(0, 0)})}
    combinations = {"C": maps, "O": set()}
    original = deepcopy(combinations)

    result = neighborhood.map_outer_product(combinations, nx.Graph(), nx.Graph())

    assert result == [{(0, 0)}]
    assert combinations == original


def test_map_outer_product_rejects_cross_color_parallel_edges():
    graph = nx.Graph([(0, 1)])
    nx.set_node_attributes(graph, {0: "C", 1: "O"}, "color")
    combinations = {
        "C": {frozenset(), frozenset({(0, 0)})},
        "O": {frozenset(), frozenset({(1, 1)})},
    }
    original = deepcopy(combinations)

    result = neighborhood.map_outer_product(combinations, graph, graph)

    assert isinstance(result, list)
    assert all(isinstance(mapping, set) for mapping in result)
    assert {frozenset(mapping) for mapping in result} == {
        frozenset({(0, 0)}),
        frozenset({(1, 1)}),
    }
    assert combinations == original


@pytest.mark.parametrize(
    "mapping, valid",
    [
        (set(), True),
        ({(0, 2)}, True),
        ({(0, 2), (3, 3)}, True),
        ({(0, 2), (1, 3)}, False),
        ({(0, 3), (1, 2)}, False),
    ],
)
@pytest.mark.parametrize("container", [list, set])
def test_multi_edge_check_handles_both_mapping_orientations(mapping, valid, container):
    assert (
        neighborhood.conditional_check_multi_edge_generation(
            frozenset(mapping), container([(0, 1)]), container([(2, 3)])
        )
        is valid
    )


def test_map_application_preserves_attributes_and_inputs_with_a_generator():
    graph1 = nx.Graph(source="first", first=True)
    graph1.add_node(0, color="C", label="retained")
    graph1.add_node(1, color="O", label="left", contraction={9: {}})
    graph1.add_edge(0, 1, color=1, label="left bond")
    graph2 = nx.Graph(source="second", second=True)
    graph2.add_node(0, color="C", label="contracted")
    graph2.add_node(1, color="N", label="right")
    graph2.add_edge(0, 1, color=2, label="right bond")
    originals = deepcopy((graph1, graph2))

    joined = neighborhood.map_application(((0, 0) for _ in range(1)), graph1, graph2)

    assert joined.graph == {"source": "second", "first": True, "second": True}
    assert dict(joined.nodes(data=True)) == {
        0: {"color": "C", "label": "retained"},
        1: {"color": "O", "label": "left"},
        2: {"color": "N", "label": "right"},
    }
    assert dict(joined.edges) == {
        (0, 1): {"color": 1, "label": "left bond"},
        (0, 2): {"color": 2, "label": "right bond"},
    }
    assert all(
        nx.utils.graphs_equal(graph, original)
        for graph, original in zip((graph1, graph2), originals)
    )


def test_map_application_rejects_lost_parallel_edges():
    graph = nx.Graph([(0, 1)])
    with pytest.raises(ValueError, match="wrong number of edges"):
        neighborhood.map_application([(0, 0), (1, 1)], graph, graph)


def test_get_valence_custom_zero_takes_precedence_and_missing_symbols_fall_back():
    table = Mock()
    table.GetDefaultValence.return_value = 7

    assert neighborhood.get_valence("C", table, {"C": 0}) == 0
    table.GetDefaultValence.assert_not_called()
    assert neighborhood.get_valence("O", table, {"C": 0}) == 7
    table.GetDefaultValence.assert_called_once_with("O")


@pytest.mark.parametrize("valence, expected_count", [(2, 0), (3, 0), (4, 4)])
def test_enumerate_up_respects_bond_orders_and_rejects_same_color_parallel_edges(
    valence, expected_count
):
    graph = nx.Graph([(0, 1)])
    nx.set_node_attributes(graph, "C", "color")
    nx.set_edge_attributes(graph, 2, "color")

    joined = neighborhood.enumerate_up(
        graph, graph, custom_valence_table={"C": valence}
    )

    assert len(joined) == expected_count
    assert all(
        result.number_of_nodes() == 3 and result.number_of_edges() == 2
        for result in joined
    )


def test_enumerate_up_zero_valence_can_be_disabled():
    graph = nx.Graph()
    graph.add_node(0, color="C")

    assert neighborhood.enumerate_up(graph, graph, custom_valence_table={"C": 0}) == []
    joined = neighborhood.enumerate_up(
        graph, graph, obey_valence=False, custom_valence_table={"C": 0}
    )
    assert len(joined) == 1
    assert list(joined[0].nodes(data=True)) == [(0, {"color": "C"})]


def test_enumerate_up_requires_node_colors_for_valence_checks():
    graph = nx.empty_graph(1)
    with pytest.raises(ValueError, match="color attribute"):
        neighborhood.enumerate_up(graph, graph)


def test_enumerate_up_requires_a_shared_color():
    graph1, graph2 = nx.empty_graph(1), nx.empty_graph(1)
    nx.set_node_attributes(graph1, "C", "color")
    nx.set_node_attributes(graph2, "O", "color")

    assert neighborhood.enumerate_up(graph1, graph2) == []


def test_enumerate_neighborhood_reports_missing_colors_before_deduplication():
    with pytest.raises(ValueError, match="color attribute"):
        neighborhood.enumerate_neighborhood([nx.path_graph(3)])
