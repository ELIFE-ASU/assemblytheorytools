"""Regression coverage for pathway construction's public state and helpers."""

import copy
import json

import networkx as nx
import numpy as np
import pytest
from rdkit import Chem

from assemblytheorytools import construction


def pathway_data(edges, vertex_colours, *, duplicates=(), remnant=None):
    """Make calculator output without invoking either assembly backend."""
    return {
        "file_graph": [
            {
                "Vertices": list(range(len(vertex_colours))),
                "Edges": edges,
                "VertexColours": vertex_colours,
                "EdgeColours": [1] * len(edges),
            }
        ],
        "remnant": [{"Edges": edges if remnant is None else remnant}],
        "removed_edges": [],
        "duplicates": list(duplicates),
    }


def test_transform_array_uses_comparison_edges_and_mutates_only_matches():
    target = [[10, 11], [12, 13], [14, 15]]
    comparison = [[8, 4], [5, 8], [8, 6]]
    untouched = target[2]

    result = construction.transform_array(target, comparison, 2, 8, 9, [[2, 4], [5, 2]])

    assert result is target
    assert result == [[9, 4], [5, 9], [14, 15]]
    assert result[2] is untouched
    assert comparison == [[8, 4], [5, 8], [8, 6]]


def test_edge_comparison_and_lookup_keep_their_distinct_orientation_rules():
    edges = [[0, 1], [1, 2]]
    reversed_edges = [[2, 1], [1, 0], [1, 0]]

    assert construction.equal_list(edges, reversed_edges)
    assert construction.check_edge_in_list(edges, [[[8, 9]], reversed_edges])
    assert construction.index_set([reversed_edges, edges], edges) == 2
    assert construction.index_set([reversed_edges], edges) is None
    assert construction.repeated_sizes([(None, []), (None, edges), (None, edges)]) == [
        0,
        2,
    ]


def test_equivalence_copies_pieces_and_uses_first_mapping_once():
    pieces = [[[2, 9], [4, 2]], [[9, 4]]]
    original = copy.deepcopy(pieces)
    mappings = [[9, 2], [7, 2], [5, 9]]

    result = construction.equivalence(pieces, mappings)

    assert result == [[[9, 5], [4, 9]], [[5, 4]]]
    assert pieces == original
    result[0][0][0] = 100
    assert pieces == original
    assert mappings == [[9, 2], [7, 2], [5, 9]]


def test_equivalence_accepts_an_empty_mapping_table_and_returns_a_copy():
    pieces = [[[0, 1]]]

    result = construction.equivalence(pieces, np.empty((0, 2), dtype=int))

    assert result == pieces
    result[0][0][0] = 2
    assert pieces == [[[0, 1]]]


@pytest.mark.parametrize(
    ("mappings", "expected_mappings"),
    [
        ([[1, 1], [2, 8], [3, 8]], [[1, 1], [2, 9], [3, 8]]),
        ([[1, 1], [2, 8], [3, 8], [2, 9]], [[1, 1], [2, 9], [3, 8]]),
    ],
    ids=["new-equivalent-vertex", "existing-equivalent-vertex"],
)
def test_fix_repeated_equiv_updates_edges_and_both_duplicate_fragments(
    mappings, expected_mappings
):
    edges = [[8, 4], [5, 8], [8, 6], [9, 7]]
    repeated = [[[[8, 4], [8, 6]], [[5, 8], [9, 7]]]]
    original_mappings = copy.deepcopy(mappings)

    result_edges, result_repeated, result_mappings = construction.fix_repeated_equiv(
        edges, repeated, mappings, [[2, 4], [5, 2], [3, 6], [2, 7]]
    )

    assert result_edges is edges
    assert result_repeated is repeated
    assert edges == [[9, 4], [5, 9], [8, 6], [9, 7]]
    assert repeated == [[[[9, 4], [8, 6]], [[5, 9], [9, 7]]]]
    assert result_mappings == expected_mappings
    assert mappings == original_mappings


def test_fix_repeated_equiv_deduplicates_and_orders_mappings_without_relabeling():
    edges = [[7, 4]]
    repeated = [[[[7, 4]], [[3, 4]]]]

    result = construction.fix_repeated_equiv(
        edges, repeated, [[3, 7], [1, 1], [3, 7]], [[3, 4]]
    )

    assert result == (edges, repeated, [[1, 1], [3, 7]])
    assert edges == [[7, 4]]
    assert repeated == [[[[7, 4]], [[3, 4]]]]


def test_tables_use_row_order_for_atoms_and_keep_bond_colours():
    tables = ([(9, "C"), (4, "N"), (2, "O")], [(0, 1, 1), (0, 2, 2)])

    molecule = construction.tables_to_mol(tables)
    graph = construction.tables_to_nx(tables)

    assert [atom.GetSymbol() for atom in molecule.GetAtoms()] == ["C", "N", "O"]
    assert [bond.GetBondTypeAsDouble() for bond in molecule.GetBonds()] == [1.0, 2.0]
    expected = nx.Graph()
    expected.add_nodes_from(
        [(0, {"color": "C"}), (1, {"color": "N"}), (2, {"color": "O"})]
    )
    expected.add_edges_from([(0, 1, {"color": 1}), (0, 2, {"color": 2})])
    assert nx.is_isomorphic(
        graph,
        expected,
        node_match=nx.algorithms.isomorphism.categorical_node_match("color", None),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match("color", None),
    )


def test_construction_keeps_first_bond_representative_and_uses_input_colours():
    data = pathway_data([[0, 1], [1, 2], [2, 3], [3, 4]], ["C", "O", "C", "C", "O"])
    data["remnant"][0]["Edges"] = [[0, 1]]
    data["removed_edges"] = [[1, 2]]
    original = copy.deepcopy(data)
    graph = nx.Graph()
    graph.add_edges_from(
        (u, v, {"color": colour})
        for (u, v), colour in zip(data["file_graph"][0]["Edges"], [1, 1, 1, 2])
    )

    obj = construction.AssemblyConstruction(data, input_graph=graph)

    assert obj.e_l == [1, 1, 1, 2]
    assert obj.remnant_e == [[0, 1], [1, 2]]
    assert obj.atoms == [[{"C", "O"}, 1], [{"C"}, 1], [{"C", "O"}, 2]]
    assert obj.atoms_list == [[["C", "O"], 1], [["C", "C"], 1], [["C", "O"], 2]]
    assert obj.atoms_list_index == [[0, 1], [2, 3], [3, 4]]
    assert len(obj.full_atoms_list) == 4
    assert obj._virtual_object_index([1, 2]) == 0
    assert obj._virtual_object_index([3, 4]) == 2
    assert data == original


@pytest.mark.parametrize("vo_type", ["graph", "smiles", "inchi"])
def test_assembly_digraph_preserves_step_order_payloads_and_input(vo_type):
    edges = [[0, 1], [1, 2], [2, 3]]
    data = pathway_data(edges, ["C", "C", "O", "C"])
    original = copy.deepcopy(data)
    obj = construction.AssemblyConstruction(data, vo_type=vo_type)

    graph, unique = obj.get_assembly_digraph()

    assert obj.steps == [edges[:2], edges]
    assert obj.pieces_mod == [edges]
    assert obj.digraph == [
        ["virtual_object_0", "step_1"],
        ["virtual_object_1", "step_1"],
        ["step_1", "step_2"],
        ["virtual_object_1", "step_2"],
    ]
    assert list(graph) == ["virtual_object_0", "step_1", "virtual_object_1", "step_2"]
    assert set(graph.edges) == {tuple(edge) for edge in obj.digraph}
    assert set(unique) == set(obj.molecules_vo + obj.molecules_steps)
    assert obj.steps_indx_s == [
        [[0, 1, 1], [1, 2, 1]],
        [[0, 1, 1], [1, 2, 1], [2, 3, 1]],
    ]
    assert obj.vs_atoms == [["C", "C", "O"], ["C", "C", "O", "C"]]
    for name, attributes in graph.nodes(data=True):
        assert attributes["type"] == (
            "step" if name.startswith("step_") else "virtual_object"
        )
        assert attributes["label"] == (name if vo_type == "graph" else attributes["vo"])
        if vo_type == "graph":
            assert isinstance(attributes["vo"], nx.Graph)
        elif vo_type == "inchi":
            assert attributes["vo"].startswith("InChI=")
        else:
            assert Chem.MolFromSmiles(attributes["vo"]) is not None
    assert data == original


def test_generate_vo_keeps_mol_bonds_and_smiles_steps():
    obj = construction.AssemblyConstruction(
        pathway_data([[0, 1], [1, 2]], ["C", "C", "O"]), vo_type="mol"
    )
    obj.generate_pathway()

    assert obj.generate_vo() is None
    assert all(isinstance(molecule, Chem.Mol) for molecule in obj.molecules_vo)
    assert len(obj.molecules_steps) == 1
    assert isinstance(obj.molecules_steps[0], str)
    assert Chem.MolFromSmiles(obj.molecules_steps[0]) is not None


@pytest.mark.parametrize(("copies", "reuse_copy"), [(1, False), (2, False), (2, True)])
def test_repeated_fragments_reuse_the_original_step(copies, reuse_copy):
    edges = [[i, i + 1] for i in range(2 + 2 * copies)]
    duplicates = [
        {"Right": edges[2 * i : 2 * i + 2], "Left": edges[:2]}
        for i in range(1, copies + 1)
    ]
    if reuse_copy:
        duplicates[1]["Left"] = duplicates[0]["Right"]
    data = pathway_data(
        edges, ["C"] * (len(edges) + 1), duplicates=duplicates, remnant=edges[:2]
    )
    original = copy.deepcopy(data)
    obj = construction.AssemblyConstruction(data)

    assert obj.generate_pathway() is None

    assert len(obj.steps) == copies + 1
    assert obj.steps[0] == edges[:2]
    assert obj.steps[-1] == edges
    assert obj.pieces_mod == [edges]
    assert obj.digraph[:4] == [
        ["virtual_object_0", "step_1"],
        ["virtual_object_0", "step_1"],
        ["step_1", "step_2"],
        ["step_1", "step_2"],
    ]
    if copies == 2:
        assert obj.digraph[4:] == [["step_2", "step_3"], ["step_1", "step_3"]]
    assert data == original


def test_consistent_join_mutates_supplied_lists_and_only_joins_one_pair():
    obj = construction.AssemblyConstruction(
        pathway_data([[0, 1], [1, 2], [2, 3]], ["C"] * 4)
    )
    pieces = [[[0, 1]], [[1, 2]], [[2, 3]]]
    steps, digraph = [], []

    result = obj.consistent_join(pieces, steps, [], 0, digraph, [])

    assert result[0] is pieces and result[1] is steps and result[3] is digraph
    assert result[2] == 1
    assert pieces == [[[0, 1], [1, 2]], [[2, 3]]]
    assert steps == [[[0, 1], [1, 2]]]
    assert digraph == [["virtual_object_0", "step_1"], ["virtual_object_0", "step_1"]]


@pytest.mark.parametrize("if_string", [False, True])
def test_consistent_join_applies_string_edge_ordering(if_string):
    edges = [[2, 3], [0, 2]]
    obj = construction.AssemblyConstruction(
        pathway_data(edges, ["C"] * 4), if_string=if_string
    )

    obj.generate_pathway()

    expected = list(reversed(edges)) if if_string else edges
    assert obj.steps == [expected]
    assert obj.pieces_mod == [expected]


@pytest.mark.parametrize("edges", [[], [[0, 1]], [[0, 1], [2, 3]]])
def test_generate_pathway_stops_when_no_fragments_can_join(edges):
    obj = construction.AssemblyConstruction(pathway_data(edges, ["C"] * 4))

    obj.generate_pathway()

    assert obj.steps == []
    assert obj.digraph == []
    assert obj.pieces_mod == [[edge] for edge in edges]


def test_parse_pathway_file_keeps_log_and_debug_contract(tmp_path, capsys):
    data = pathway_data([[0, 1], [1, 2]], ["C"] * 3)
    path = tmp_path / "pathway.json"
    path.write_text(json.dumps(data))

    graph, vos, log = construction.parse_pathway_file(path, debug=True, log=True)

    assert len(graph) == 2 and len(vos) == 2
    assert log == (
        "#####Graph#####\n"
        "[0, 1, 2]\n"
        "[[0, 1], [1, 2]]\n"
        "['C', 'C', 'C']\n"
        "[1, 1]\n"
        "#####Atoms#####\n"
        "atom0=[['C', 'C'], 1]\n"
        "#####Steps#####\n"
        "step1=[[0, 1], [1, 2]]\n"
        "#####Digraph#####\n"
        "['virtual_object_0', 'step_1']\n"
        "['virtual_object_0', 'step_1']\n"
    )
    assert capsys.readouterr().out.splitlines() == [
        f"Node: {name}, Type: {attributes['type']}, VO: {attributes['vo']}"
        for name, attributes in graph.nodes(data=True)
    ]
    assert len(construction.parse_pathway_file(path)) == 2
