"""Reconstruct reversal-aware pathways emitted by the C++ string solver."""

import json

import networkx as nx
import pytest

from assemblytheorytools.construction import parse_string_pathway_file


def write_pathway(tmp_path, string, duplicate_intervals):
    path = tmp_path / "pathway.json"
    path.write_text(json.dumps({
        "file_graph": [{"Fragments": [string]}],
        "duplicates": [
            {"Left": left, "Right": right}
            for left, right in duplicate_intervals
        ],
    }))
    return path


def assert_construction(graph, string, expected_cost):
    """Check actual operations, including free orientation changes and reuse."""
    assert nx.is_directed_acyclic_graph(graph)
    assert sum(data["cost"] for _, data in graph.nodes(data=True)) == expected_cost
    assert sum(data["operation"] == "concatenate"
               for _, data in graph.nodes(data=True)) == expected_cost
    for fragment, data in graph.nodes(data=True):
        parents = list(graph.predecessors(fragment))
        assert nx.has_path(graph, fragment, string)
        if data["operation"] == "primitive":
            assert len(fragment) == 1
            assert parents == []
            assert data["cost"] == 0
        elif data["operation"] == "reverse":
            assert parents == [fragment[::-1]]
            assert graph.edges[parents[0], fragment] == {
                "operation": "reverse", "cost": 0,
            }
            assert data["cost"] == 0
        else:
            assert data == {"operation": "concatenate", "cost": 1}
            # A repeated operand has one predecessor in a DiGraph.
            operands = parents if len(parents) == 2 else parents * 2
            assert len(operands) == 2
            assert fragment in ("".join(operands), "".join(reversed(operands)))
            assert all(graph.edges[parent, fragment]["operation"] == "concatenate"
                       for parent in parents)


@pytest.mark.parametrize(
    "string, duplicates, expected_cost, expected_reversals",
    [
        ("abcxcba", [([0, 3], [4, 3])], 4, {("abc", "cba")}),
        ("abcxcbaq!qabcxcba", [([0, 8], [9, 8]), ([0, 3], [4, 3])], 7,
         {("abc", "cba"), ("abcxcbaq", "qabcxcba")}),
        ("abcxcbaabcxcbaabcxcba",
         [([7, 7], [14, 7]), ([7, 7], [0, 7]), ([7, 3], [11, 3])], 6,
         {("abc", "cba")}),
        ("bbbadcdaabbaad", [([10, 4], [6, 4]), ([10, 2], [2, 2])], 9,
         {("baad", "daab")}),
        ("aba", [], 2, set()),
        ("abaaba", [([0, 3], [3, 3])], 3, set()),
    ],
    ids=["reversal", "nested-reversals", "survivor-after-copies",
         "nested-survivor-after-reversed-copy", "palindrome", "palindrome-reuse"],
)
def test_reversal_pathway_has_the_cpp_join_count(
    tmp_path, string, duplicates, expected_cost, expected_reversals,
):
    # These intervals and costs come from successful C++ reversal-mode searches.
    source = write_pathway(tmp_path, string, duplicates)

    virtual_objects, graph = parse_string_pathway_file(source, accept_palindromes=True)

    assert virtual_objects == list(graph)
    assert_construction(graph, string, expected_cost)
    assert {(left, right) for left, right, data in graph.edges(data=True)
            if data["operation"] == "reverse"} == expected_reversals


@pytest.mark.parametrize("string", ["", "a"])
def test_reversal_mode_handles_zero_joins(tmp_path, string):
    source = write_pathway(tmp_path, string, [])

    virtual_objects, graph = parse_string_pathway_file(source, accept_palindromes=True)

    assert virtual_objects == list(string)
    assert graph.number_of_edges() == 0
    assert all(data == {"operation": "primitive", "cost": 0}
               for _, data in graph.nodes(data=True))


def test_default_pathway_format_is_unchanged(tmp_path):
    source = write_pathway(tmp_path, "xabab", [([1, 2], [3, 2])])

    implicit_objects, implicit = parse_string_pathway_file(source)
    explicit_objects, explicit = parse_string_pathway_file(source, accept_palindromes=False)

    assert implicit_objects == explicit_objects == ["x", "a", "b", "ab", "xab", "xabab"]
    assert list(implicit.edges(data=True)) == list(explicit.edges(data=True))
    assert set(implicit.edges()) == {
        ("a", "ab"), ("b", "ab"), ("x", "xab"), ("ab", "xab"),
        ("xab", "xabab"), ("ab", "xabab"),
    }
    assert all(data == {} for _, data in implicit.nodes(data=True))
    assert all(data == {} for _, _, data in implicit.edges(data=True))
