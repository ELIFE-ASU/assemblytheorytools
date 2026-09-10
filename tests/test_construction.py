"""Assembly pathway construction, layer traversal, and DOT interchange."""

import copy
import json

import networkx as nx
import numpy as np
import pytest
from rdkit import Chem

import assemblytheorytools as att
from assemblytheorytools import construction


@pytest.fixture
def anthracene_pathway_data(data_dir):
    molecule = att.molfile_to_mol(
        str(data_dir / "mol_files" / "anthracene.mol"), add_hydrogens=False
    )
    dot = (data_dir / "pathway" / "anthracene_pathway.dot").read_text()
    return molecule, dot


@pytest.mark.parametrize(
    "levels, edges",
    [
        (
            {"CC": 0, "C=C": 0, "CO": 0, "CC=C": 1, "OCC=C": 2},
            [("CC", "CC=C"), ("C=C", "CC=C"), ("CO", "OCC=C"), ("CC=C", "OCC=C")],
        ),
        (
            {"CC": 0, "CCC": 1, "CCCCC": 2, "CCCCCCCCC": 3},
            [("CC", "CCC"), ("CCC", "CCCCC"), ("CCCCC", "CCCCCCCCC")],
        ),
        ({}, []),
    ],
    ids=["branch", "chain", "empty"],
)
def test_assign_levels_uses_deepest_predecessor(levels, edges):
    graph = nx.DiGraph()
    graph.add_nodes_from(levels)
    graph.add_edges_from(edges)

    assert construction.assign_levels(graph) is None
    assert nx.get_node_attributes(graph, "level") == levels


def test_convert_virtual_objects_to_smiles():
    # Use a local structure so conversion does not depend on PubChem.
    graph = att.smi_to_nx("CCOC(=O)C1=CC=CC=C1C(=O)OCC")
    pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)[2]

    converted = construction.convert_digraph_vo_to_target(pathway)

    expected = [
        "CC",
        "CCO",
        "CO",
        "C=O",
        "CC(=O)O",
        "C=CC(=O)O",
        "C=C",
        "CC=CC(=O)O",
        "CC=CC(=O)OCC",
        "CC=C(C)C(=O)OCC",
        "C=CC=C(C)C(=O)OCC",
        "CCOC(=O)C1=CC=CC=C1C(=O)OCC",
    ]
    # Equivalent ring traversals can differ between RDKit versions.
    actual = [
        Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        for _, smiles in converted.nodes(data="vo")
    ]
    expected = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in expected]
    assert sorted(actual) == sorted(expected)


def test_get_virtual_objects_on_layer():
    graphs = [att.smi_to_nx(smiles) for smiles in ["CC(OC)C=C", "CC(OC)C", "CCC"]]
    pathway = att.calculate_assembly_index(
        att.join_graphs(graphs), strip_hydrogen=True
    )[-1]

    expected_layers = [{"CC", "CO", "C=C"}, {"CCC", "COC"}, {"COC(C)C"}, {"C=CC(C)OC"}]
    assert set(construction.get_vos_on_layer(pathway, 0)) == expected_layers[0]
    assert [
        set(layer) for layer in construction.get_vos_on_layer(pathway, [0, 1])
    ] == expected_layers[:2]
    assert [
        set(layer) for layer in construction.get_vos_on_layer(pathway, "all")
    ] == expected_layers


def test_parse_pathway_dot_preserves_fragments_and_bond_bookkeeping(
    anthracene_pathway_data,
):
    molecule, dot = anthracene_pathway_data

    pathway = construction.parse_pathway_dot(dot, mol=molecule)

    assert isinstance(pathway, nx.MultiDiGraph)
    assert set(pathway) == set(range(8))
    assert pathway.number_of_edges() == 12
    assert nx.get_node_attributes(pathway, "type") == {
        node: "virtual_object" for node in range(8)
    }
    # Fragments are kekulised, matching the graph searched by the backend.
    assert [pathway.nodes[node]["vo"] for node in sorted(pathway)] == [
        "CC",
        "C=C",
        "C=CC",
        "CC=CC",
        "C=CC=CC",
        "CC=CC=CC=CC",
        "CC=CC1=CC=CC=C1",
        "C1=CC=C2C=C3C=CC=CC3=CC2=C1",
    ]
    assert pathway.nodes[7]["bonds"] == frozenset(range(molecule.GetNumBonds()))
    assert pathway.nodes[7]["label"] == "{" + ", ".join(map(str, range(16))) + "}"
    assert pathway.nodes[0]["bonds"] == frozenset({14})
    assert pathway[0][2][0]["bonds"] == frozenset({14})
    assert pathway[2][3][0]["bonds"] == frozenset({14, 15})


@pytest.mark.parametrize("vo_type", ["mol", "graph", "smiles", "inchi"])
def test_parse_pathway_dot_virtual_object_representations(
    anthracene_pathway_data, vo_type
):
    molecule, dot = anthracene_pathway_data
    original = Chem.MolToMolBlock(molecule)

    pathway = construction.parse_pathway_dot(dot, mol=molecule, vo_type=vo_type)

    fragment = pathway.nodes[2]["vo"]
    if vo_type == "mol":
        assert isinstance(fragment, Chem.Mol)
        assert Chem.MolToSmiles(fragment) == "C=CC"
    elif vo_type == "graph":
        assert isinstance(fragment, nx.Graph)
        assert (fragment.number_of_nodes(), fragment.number_of_edges()) == (3, 2)
    elif vo_type == "smiles":
        assert fragment == "C=CC"
    else:
        assert pathway.nodes[7]["vo"].startswith("InChI=1S/C14H10")
    assert Chem.MolToMolBlock(molecule) == original


def test_parse_pathway_dot_without_molecule_retains_bond_labels(
    anthracene_pathway_data,
):
    _, dot = anthracene_pathway_data

    pathway = construction.parse_pathway_dot(dot)

    assert pathway.nodes[2]["vo"] == "{14, 15}"
    assert pathway.nodes[2]["bonds"] == frozenset({14, 15})


@pytest.mark.parametrize(
    "dot, message",
    [
        ("hello world", "Could not parse"),
        ('graph { 0 [label="{1}"] }', "must be a DOT 'digraph'"),
        ('digraph { 0 [label="nope"] }', "malformed bond set"),
        ("digraph { 0 }", "has no 'label' attribute"),
        ('digraph { a [label="{1}"] }', "node names must be integers"),
        (
            'digraph { 0 [label="{1}"]; 1 [label="{2, 3}"]; 0 -> 1 [label="{2}"] }',
            "inputs supply",
        ),
        (
            'digraph { 0 [label="{1}"]; 1 [label="{2, 3}"]; 0 -> 1 [label="{2, 3}"] }',
            "source fragment has 1",
        ),
    ],
    ids=[
        "invalid-dot",
        "undirected",
        "malformed-label",
        "missing-label",
        "node-id",
        "missing-bond",
        "edge-size",
    ],
)
def test_parse_pathway_dot_rejects_invalid_structure(dot, message):
    with pytest.raises(ValueError, match=message):
        construction.parse_pathway_dot(dot)


def test_parse_pathway_dot_rejects_out_of_range_bonds(anthracene_pathway_data):
    molecule, _ = anthracene_pathway_data
    with pytest.raises(ValueError, match="bond"):
        construction.parse_pathway_dot('digraph { 0 [label="{99}"] }', mol=molecule)


def test_parse_pathway_dot_rejects_unknown_representation(anthracene_pathway_data):
    _, dot = anthracene_pathway_data
    with pytest.raises(ValueError, match="vo_type"):
        construction.parse_pathway_dot(dot, vo_type="banana")


def test_parse_pathway_dot_can_relax_bond_bookkeeping():
    dot = 'digraph { 0 [label="{1}"]; 1 [label="{2, 3}"]; 0 -> 1 [label="{2}"] }'

    pathway = construction.parse_pathway_dot(dot, strict=False)

    assert set(pathway) == {0, 1}
    assert list(pathway.edges()) == [(0, 1)]
    assert pathway[0][1][0]["bonds"] == frozenset({2})


def test_parsed_pathway_levels_follow_topological_order(anthracene_pathway_data):
    molecule, dot = anthracene_pathway_data
    pathway = construction.parse_pathway_dot(dot, mol=molecule)
    ordered = nx.MultiDiGraph()
    ordered.add_nodes_from(
        (node, pathway.nodes[node]) for node in nx.topological_sort(pathway)
    )
    ordered.add_edges_from(pathway.edges(data=True))

    construction.assign_levels(ordered)

    assert nx.get_node_attributes(ordered, "level") == {
        0: 0,
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        7: 6,
    }


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


@pytest.mark.parametrize("legacy", [False, True])
def test_pathway_reader_preserves_current_and_legacy_json(tmp_path, legacy):
    data = pathway_data([[0, 1], [1, 2]], ["C", "O", "C"])
    raw = json.dumps(data)
    if legacy:
        raw = raw.replace('"EdgeColours": [1, 1]', '"EdgeColours": [single, single]')
    path = tmp_path / "pathway.json"
    path.write_text(raw)

    graph, virtual_objects = construction.parse_pathway_file(path, vo_type="graph")

    assert len(graph) == 2
    assert sorted(vo.number_of_edges() for vo in virtual_objects) == [1, 2]
    assert path.read_text() == raw


def test_pathway_reader_recovers_omitted_legacy_colors_from_input(tmp_path):
    data = pathway_data([[0, 1], [1, 2]], ["C", "C", "C"])
    raw = json.dumps(data).replace('"EdgeColours": [1, 1]', '"EdgeColours": [, ]')
    path = tmp_path / "pathway.json"
    path.write_text(raw)
    original = nx.path_graph(3)
    nx.set_node_attributes(original, "C", "color")
    nx.set_edge_attributes(original, 12, "color")

    _, virtual_objects = construction.parse_pathway_file(path, vo_type="graph", input_graph=original)

    assert all(color == 12 for vo in virtual_objects for *_, color in vo.edges(data="color"))
    assert path.read_text() == raw


def test_pathway_reader_accepts_native_named_and_numeric_colors(tmp_path):
    # Current parallelassemblycpp names orders 1-3 and writes higher orders
    # as numeric strings, including graph colours outside RDKit's bond types.
    colors = ["single", "double", "triple", "4", "12", "32767"]
    edges = [[i, i + 1] for i in range(len(colors))]
    data = pathway_data(edges, ["C"] * (len(colors) + 1))
    data["file_graph"][0]["EdgeColours"] = colors
    path = tmp_path / "pathway.json"
    raw = json.dumps(data)
    path.write_text(raw)

    _, virtual_objects = construction.parse_pathway_file(path, vo_type="graph")

    target = max(virtual_objects, key=lambda graph: graph.number_of_edges())
    assert target.number_of_edges() == len(edges)
    assert sorted(nx.get_edge_attributes(target, "color").values()) == [1, 2, 3, 4, 12, 32767]
    assert path.read_text() == raw


@pytest.mark.parametrize(
    "edges, colors, expected_nodes, expected_degrees",
    [
        ([[0, 1]], ["C", "O"], ["virtual_object_0"], [0]),
        ([[0, 1], [1, 2], [3, 4]], ["C", "C", "C", "C", "O"],
         ["virtual_object_0", "step_1", "virtual_object_1"], [1, 1, 0]),
    ],
    ids=["single-bond", "disconnected-bond"],
)
def test_assembly_pathway_retains_components_that_need_no_joins(
    edges, colors, expected_nodes, expected_degrees
):
    obj = construction.AssemblyConstruction(pathway_data(edges, colors))

    graph, virtual_objects = obj.get_assembly_digraph()

    assert list(graph) == expected_nodes
    assert [graph.degree(node) for node in graph] == expected_degrees
    assert {data["vo"] for _, data in graph.nodes(data=True)} == set(virtual_objects)
    assert all("label" in data and "type" in data for _, data in graph.nodes(data=True))


def test_string_pathway_builds_a_duplicate_starting_after_zero(tmp_path):
    data = {"file_graph": [{"Fragments": ["xabab"]}],
            "duplicates": [{"Left": [1, 2], "Right": [3, 2]}]}
    path = tmp_path / "pathway.json"
    path.write_text(json.dumps(data))

    virtual_objects, graph = construction.parse_string_pathway_file(path)

    assert virtual_objects == ["x", "a", "b", "ab", "xab", "xabab"]
    assert nx.is_directed_acyclic_graph(graph)
    assert set(graph.edges()) == {("a", "ab"), ("b", "ab"), ("x", "xab"),
                                  ("ab", "xab"), ("xab", "xabab"), ("ab", "xabab")}


def test_string_pathway_only_reuses_copies_starting_at_the_cursor():
    data = {"file_graph": [{"Fragments": ["zababab"]}],
            "duplicates": [{"Left": [1, 2], "Right": [3, 2]}]}

    assert construction.immediate_predecessors(data, (2, 5)) == ["b", "ab", "a", "b"]


@pytest.mark.parametrize("string", ["", "a"])
def test_string_pathway_handles_no_joins(tmp_path, string):
    path = tmp_path / "pathway.json"
    path.write_text(json.dumps({"file_graph": [{"Fragments": [string]}], "duplicates": []}))

    virtual_objects, graph = construction.parse_string_pathway_file(path)

    assert virtual_objects == list(string)
    assert graph.number_of_edges() == 0


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
