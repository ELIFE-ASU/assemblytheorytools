"""Crossing reduction and route integrity for layered pathway layouts."""

import copy
import itertools
import random

import networkx as nx
import pytest

from assemblytheorytools import tools_plotting as plotting


LAYOUTS = [
    plotting.multipartite_layout_crossmin,
    plotting.multipartite_layout_crossmin_long,
    plotting.multipartite_layout_sa,
]
LONG_LAYOUTS = LAYOUTS[1:]


def _crossings(graph, orders):
    """Count proper crossings directly, independently of the layout cost code."""
    indices = {node: i for layer in orders.values() for i, node in enumerate(layer)}
    total = 0
    for (a, b), (c, d) in itertools.combinations(graph.edges(), 2):
        if graph.nodes[a]["subset"] > graph.nodes[b]["subset"]:
            a, b = b, a
        if graph.nodes[c]["subset"] > graph.nodes[d]["subset"]:
            c, d = d, c
        if (
            graph.nodes[a]["subset"] == graph.nodes[c]["subset"]
            and graph.nodes[b]["subset"] == graph.nodes[d]["subset"]
            and (indices[a] - indices[c]) * (indices[b] - indices[d]) < 0
        ):
            total += 1
    return total


def _branching_graph():
    graph = nx.Graph()
    graph.add_nodes_from((i, {"subset": i // 4}) for i in range(16))
    graph.add_edges_from(
        [
            (0, 4),
            (0, 5),
            (0, 6),
            (1, 4),
            (1, 5),
            (1, 7),
            (2, 7),
            (3, 4),
            (3, 7),
            (4, 8),
            (6, 8),
            (6, 9),
            (6, 10),
            (6, 11),
            (7, 9),
            (8, 14),
            (9, 13),
            (9, 14),
            (10, 13),
            (10, 14),
            (11, 13),
        ]
    )
    return graph


@pytest.mark.parametrize("layout", LAYOUTS)
def test_crossing_minimization_retains_good_intermediate_orders(layout):
    # One-sided sweeps oscillate here and finish at nine crossings, despite
    # visiting an order with six. Retaining that order and transposing gives five.
    graph = _branching_graph()
    before = copy.deepcopy(graph)
    options = {"max_proposals": 0} if layout is plotting.multipartite_layout_sa else {}

    _, orders = layout(graph, seed=5, return_order=True, **options)

    assert _crossings(graph, orders) <= 5
    assert nx.utils.graphs_equal(graph, before)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_crossing_minimization_accounts_for_both_edge_directions(layout):
    graph = nx.DiGraph(_branching_graph())
    # Keep only ascending edges, then compare against the reversed pathway.
    graph.remove_edges_from([(u, v) for u, v in graph.edges if u > v])
    options = {"max_proposals": 0} if layout is plotting.multipartite_layout_sa else {}

    forward = layout(graph, seed=7, return_order=True, **options)
    backward = layout(graph.reverse(), seed=7, return_order=True, **options)

    assert forward == backward
    assert _crossings(graph, forward[1]) <= 5


@pytest.mark.parametrize("layout", LONG_LAYOUTS)
@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
@pytest.mark.parametrize("insert_dummies", [False, True])
def test_routes_preserve_edges_and_direction(layout, graph_type, insert_dummies):
    graph = graph_type()
    graph.add_nodes_from((node, {"subset": i}) for i, node in enumerate("abc"))
    graph.add_edges_from([("c", "b"), ("c", "a"), ("a", "b")])
    if graph.is_multigraph():
        graph.add_edge("c", "a")

    positions, routes = layout(
        graph,
        seed=7,
        insert_dummies=insert_dummies,
        return_dummies=True,
        return_routes=True,
    )

    assert [route["endpoints"] for route in routes] == list(graph.edges())
    for route in routes:
        assert (route["nodes"][0], route["nodes"][-1]) == route["endpoints"]
        assert route["points"] == [positions[node] for node in route["nodes"]]
        if not insert_dummies:
            assert len(route["nodes"]) == 2


@pytest.mark.parametrize("layout", LONG_LAYOUTS)
def test_dummy_names_never_replace_or_filter_real_nodes(layout):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [("a", {"subset": 0}), ("__dummy__1", {"subset": 1}), ("c", {"subset": 2})]
    )
    graph.add_edge("a", "c")

    positions, routes = layout(graph, seed=4, return_routes=True)
    all_positions = layout(graph, seed=4, return_dummies=True)

    assert set(positions) == set(graph)
    assert len(all_positions) == 4
    assert routes[0]["nodes"] == ["a", "__dummy__2", "c"]
    assert positions["__dummy__1"] != all_positions["__dummy__2"]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_layout_seed_is_reproducible_without_mutating_global_random_state(layout):
    graph = _branching_graph()
    state = random.getstate()

    first = layout(graph, seed=123)
    second = layout(graph, seed=123)

    assert first == second
    assert random.getstate() == state


@pytest.mark.parametrize("layout", LONG_LAYOUTS)
def test_mixed_layer_keys_with_identical_text_have_stable_order(layout):
    graph = nx.Graph()
    graph.add_nodes_from([("a", {"subset": 1}), ("b", {"subset": "1"})])
    reordered = nx.Graph()
    reordered.add_nodes_from(reversed(list(graph.nodes(data=True))))

    assert layout(graph, seed=42) == layout(reordered, seed=42)


@pytest.mark.parametrize("layout", LONG_LAYOUTS)
def test_anthracene_long_edge_routes_reach_minimum_crossing_count(layout):
    graph = nx.DiGraph(
        [
            (0, 2),
            (1, 2),
            (0, 3),
            (2, 3),
            (1, 4),
            (3, 4),
            (4, 5),
            (3, 5),
            (2, 6),
            (5, 6),
            (6, 7),
            (5, 7),
        ]
    )
    for rank, nodes in enumerate(nx.topological_generations(graph)):
        nx.set_node_attributes(graph, {node: rank for node in nodes}, "subset")

    _, orders, routes = layout(graph, seed=42, return_order=True, return_routes=True)

    expanded = nx.DiGraph()
    for rank, nodes in orders.items():
        expanded.add_nodes_from((node, {"subset": rank}) for node in nodes)
    for route in routes:
        expanded.add_edges_from(zip(route["nodes"], route["nodes"][1:]))
    # All 1,728 possible layer orders were exhaustively checked: one crossing
    # is unavoidable for this pathway's monotone inter-layer routes.
    assert _crossings(expanded, orders) == 1
