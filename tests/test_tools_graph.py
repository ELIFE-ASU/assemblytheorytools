"""Molecular graph conversion, composition, traversal, and serialization."""

import random
from copy import deepcopy

import networkx as nx
import pytest
from rdkit import Chem

import assemblytheorytools as att
from assemblytheorytools import tools_graph as tg


@pytest.mark.parametrize(
    "text, to_graph, from_graph",
    [
        ("[H]O[H]", tg.smi_to_nx, tg.nx_to_smi),
        ("InChI=1S/H2O/h1H2", tg.inchi_to_nx, tg.nx_to_inchi),
    ],
    ids=["smiles", "inchi"],
)
def test_molecular_notation_roundtrip(text, to_graph, from_graph):
    assert from_graph(to_graph(text)) == text


def test_graph_charges():
    assert tg.get_graph_charges(att.ph_2p_graph()) == [2, 0]


def test_hydrogen_removal_returns_a_copy_and_preserves_input():
    graph = tg.smi_to_nx("CCO")
    original = deepcopy(graph)

    stripped = tg.remove_hydrogen_from_graph(graph)

    assert stripped is not graph
    assert (stripped.number_of_nodes(), stripped.number_of_edges()) == (3, 2)
    assert list(stripped.nodes(data="color")) == [(0, "C"), (1, "C"), (2, "O")]
    assert nx.utils.graphs_equal(graph, original)


def test_assembly_calculation_can_reuse_the_graph_after_hydrogen_stripping():
    graph = tg.smi_to_nx("CCO")
    original = deepcopy(graph)

    assert att.calculate_assembly_index(graph, strip_hydrogen=True)[0] == 1
    assert nx.utils.graphs_equal(graph, original)
    assert att.calculate_assembly_index(graph)[0] == 6
    assert nx.utils.graphs_equal(graph, original)


def test_joining_and_splitting_preserves_colored_molecular_components():
    graphs = [tg.smi_to_nx(smiles) for smiles in ["[H][O][H]", "[O][O]"]]

    joined = tg.join_graphs(graphs)
    components = tg.get_disconnected_subgraphs(joined)

    assert (joined.number_of_nodes(), joined.number_of_edges()) == (5, 3)
    assert len(components) == 2
    for original, restored in zip(graphs, components):
        assert nx.is_isomorphic(
            original,
            restored,
            node_match=nx.algorithms.isomorphism.categorical_node_match("color", None),
            edge_match=nx.algorithms.isomorphism.categorical_edge_match("color", None),
        )


def test_composition_merges_overlapping_node_labels():
    graphs = [tg.smi_to_nx(smiles) for smiles in ["[H][O][H]", "[O][O]"]]

    composed = tg.compose_graphs(graphs)

    assert (composed.number_of_nodes(), composed.number_of_edges()) == (3, 2)
    assert nx.get_node_attributes(composed, "color") == {0: "O", 1: "O", 2: "H"}


def test_graph_layers_follow_dependencies():
    graph = nx.DiGraph([(0, 1), (1, 2), (0, 2)])

    assert tg.set_graph_layer(graph) is graph
    assert nx.get_node_attributes(graph, "layer") == {0: 0, 1: 1, 2: 2}


def test_top_degree_pathway_retains_requested_molecules():
    graphs = [
        tg.smi_to_nx(smiles) for smiles in ["CC(OC)C=C", "CC(OC)C", "CC(OC)CCC", "CCC"]
    ]
    pathway = att.calculate_assembly_index(tg.join_graphs(graphs), strip_hydrogen=True)[
        -1
    ]

    selected = tg.top_n_degree_subgraph(pathway, n=3, must_keep=graphs)

    assert len(selected) == 5
    assert set(selected) <= set(pathway)
    for requested in graphs:
        assert any(
            nx.is_isomorphic(vo, tg.remove_hydrogen_from_graph(requested))
            for _, vo in selected.nodes(data="vo")
        )


def colored_path(*colors):
    graph = nx.path_graph(len(colors))
    nx.set_node_attributes(graph, dict(enumerate(colors)), "color")
    nx.set_edge_attributes(graph, 1, "color")
    return graph


@pytest.mark.parametrize("order", range(1, 22))
def test_supported_rdkit_bond_order_roundtrip(order):
    assert tg.bond_order_rdkit_to_int(tg.bond_order_int_to_rdkit(order)) == order


def test_unspecified_rdkit_bond_order_is_zero():
    assert tg.bond_order_rdkit_to_int(Chem.BondType.UNSPECIFIED) == 0


def test_unsupported_bond_orders_are_rejected():
    with pytest.raises(ValueError, match="Unsupported bond order: 0"):
        tg.bond_order_int_to_rdkit(0)
    with pytest.raises(ValueError, match="Unsupported RDKit BondType: 100"):
        tg.bond_order_rdkit_to_int(100)


@pytest.mark.parametrize(
    "value, expected", [("single", 1), ("quintuple", 5), ("12", 12), (2, 2)]
)
def test_assembly_bond_orders_accept_names_and_numeric_values(value, expected):
    assert tg.bond_order_assout_to_int(value) == expected


def test_nx_to_mol_preserves_node_order_and_accepts_string_bond_orders():
    graph = nx.Graph()
    graph.add_node("oxygen", color=" O ")
    graph.add_node(42, color="C")
    graph.add_edge("oxygen", 42, color="2")

    mol = tg.nx_to_mol(graph, sanitize=False)

    assert isinstance(mol, Chem.RWMol)
    assert [atom.GetSymbol() for atom in mol.GetAtoms()] == ["O", "C"]
    assert mol.GetBondWithIdx(0).GetBondType() == Chem.BondType.DOUBLE
    assert graph.nodes["oxygen"]["color"] == " O "
    assert graph.edges["oxygen", 42]["color"] == "2"


@pytest.mark.parametrize("missing", ["node", "edge"])
def test_nx_to_mol_reports_missing_colors(missing):
    graph = colored_path("C", "O")
    if missing == "node":
        del graph.nodes[0]["color"]
        message = "Node 0 is missing the 'color' attribute."
    else:
        del graph.edges[0, 1]["color"]
        message = "Edge (0, 1) is missing the 'color' attribute."

    with pytest.raises(KeyError) as error:
        tg.nx_to_mol(graph)
    assert error.value.args == (message,)


def test_unsanitized_conversion_does_not_add_hydrogens():
    mol = Chem.MolFromSmiles("CO")
    graph = tg.mol_to_nx(mol, sanitize=False, add_hydrogens=True)

    assert list(graph.nodes(data="color")) == [(0, "C"), (1, "O")]
    assert tg.nx_to_mol(graph, sanitize=False, add_hydrogens=True).GetNumAtoms() == 2
    assert mol.GetNumAtoms() == 2


def test_inchi_parser_hydrogens_survive_disabled_graph_sanitization():
    graph = tg.inchi_to_nx("InChI=1S/H2O/h1H2", add_hydrogens=False, sanitize=False)
    assert len(graph) == 3
    assert len(tg.smi_to_nx("O")) == 3


@pytest.mark.parametrize(
    "converter, parser, message",
    [
        (tg.smi_to_nx, "smi_to_mol", "Invalid SMILES string or conversion failed."),
        (tg.inchi_to_nx, "inchi_to_mol", "Invalid InChI string or conversion failed."),
    ],
)
def test_string_converters_report_failed_parsing(
    monkeypatch, converter, parser, message
):
    monkeypatch.setattr(tg, parser, lambda *args, **kwargs: None)
    with pytest.raises(ValueError) as error:
        converter("invalid")
    assert str(error.value) == message


def test_disconnected_subgraphs_share_attributes_with_the_original():
    graph = nx.Graph([(2, 3)])
    graph.add_node("isolated")

    connected, isolated = tg.get_disconnected_subgraphs(graph)

    assert set(connected) == {2, 3}
    assert list(isolated) == ["isolated"]
    assert nx.is_frozen(connected)
    connected.nodes[2]["label"] = "shared"
    assert graph.nodes[2]["label"] == "shared"
    graph.remove_node(3)
    assert list(connected) == [2]


@pytest.mark.parametrize("combine", [tg.join_graphs, tg.compose_graphs])
def test_singleton_combination_returns_the_original_graph(combine):
    graph = nx.Graph()
    graph.add_node("original-label")
    assert combine(iter([graph])) is graph


def test_join_composition_preserves_disjoint_labels_and_copies_singletons():
    first = nx.Graph([(2, 3)])
    second = nx.Graph([("x", "y")])

    assert tg.join_graphs([first], disjoint=False) is not first
    joined = tg.join_graphs(iter([first, second]), disjoint=False)
    assert list(joined) == [2, 3, "x", "y"]


def test_join_composition_prefixes_every_graph_when_any_labels_clash():
    graphs = [nx.Graph([(0, 1)]), nx.Graph([(1, 2)]), nx.Graph([("x", "y")])]
    joined = tg.join_graphs(graphs, disjoint=False, rename_prefix="part")

    assert list(joined) == [
        "part0_0",
        "part0_1",
        "part1_1",
        "part1_2",
        "part2_x",
        "part2_y",
    ]
    assert list(graphs[0]) == [0, 1]


def test_combination_empty_inputs_and_join_type_validation():
    with pytest.raises(ValueError, match="Need at least one graph"):
        tg.join_graphs(iter([]))
    with pytest.raises(
        ValueError, match=r"compose_graphs\(\) requires at least one graph"
    ):
        tg.compose_graphs(iter([]))

    class CustomGraph(nx.Graph):
        pass

    with pytest.raises(TypeError, match="All graphs must be of the same NetworkX type"):
        tg.join_graphs([nx.Graph(), CustomGraph()])


def test_composition_later_attributes_override_without_mutating_inputs():
    first = colored_path("C", "O")
    first.graph["source"] = "first"
    first.nodes[0]["retained"] = True
    second = colored_path("N", "H")
    second.graph["source"] = "second"
    second.edges[0, 1]["color"] = 2

    composed = tg.compose_graphs(iter([first, second]))

    assert composed.graph["source"] == "second"
    assert composed.nodes[0] == {"color": "N", "retained": True}
    assert composed.edges[0, 1]["color"] == 2
    assert first.nodes[0]["color"] == "C"
    assert first.edges[0, 1]["color"] == 1


@pytest.mark.parametrize(
    "labeler, attribute, expected",
    [(tg.set_graph_layer, "layer", 0), (tg.relabel_digraph, "label", "Step 0")],
)
def test_layer_labelers_mutate_in_place_even_before_a_cycle_error(
    labeler, attribute, expected
):
    graph = nx.DiGraph([(0, 1)])
    assert labeler(graph) is graph
    graph = nx.DiGraph([(1, 2), (2, 1)])
    graph.add_node(0)

    with pytest.raises(nx.NetworkXUnfeasible):
        labeler(graph)

    assert graph.nodes[0][attribute] == expected
    assert attribute not in graph.nodes[1]


def test_stripping_a_layer_recomputes_generations_on_a_mutable_copy():
    graph = nx.DiGraph([(0, 1), (1, 2)])
    nx.set_node_attributes(graph, 99, "layer")
    nx.set_node_attributes(graph, "Step 0", "label")

    result = tg.strip_digraph_layer(graph, 0)

    assert list(result.nodes(data="layer")) == [(1, 1), (2, 2)]
    assert nx.get_node_attributes(graph, "layer") == {0: 99, 1: 99, 2: 99}
    assert not nx.is_frozen(result)


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiDiGraph])
def test_longest_path_is_unweighted_and_rejects_cycles(graph_type):
    graph = graph_type([(0, 1), (1, 2), (0, 2)])
    nx.set_edge_attributes(graph, -100, "weight")

    assert tg.longest_path_length(graph) == 2
    assert tg.longest_path_length(nx.DiGraph()) == 0
    graph.add_edge(2, 0)
    with pytest.raises(ValueError, match="Graph must be a Directed Acyclic Graph"):
        tg.longest_path_length(graph)


def test_top_degree_ties_follow_insertion_order_and_return_a_view_of_a_copy():
    graph = nx.DiGraph()
    for node in ["z", "a", "b"]:
        graph.add_node(node, vo=colored_path("C"), label="original")

    result = tg.top_n_degree_subgraph(graph, 1, [])

    assert list(result) == ["z"]
    assert nx.is_frozen(result)
    result.nodes["z"]["label"] = "changed"
    assert graph.nodes["z"]["label"] == "original"
    assert result.nodes["z"]["vo"] is graph.nodes["z"]["vo"]
    assert set(tg.top_n_degree_subgraph(graph, -1, [])) == {"z", "a"}


def test_must_keep_strips_hydrogens_without_mutation_and_matches_topology_only():
    reference = colored_path("C", "O", "H")
    graph = nx.DiGraph()
    graph.add_node("match", vo=colored_path("N", "N"))
    graph.add_node("other", vo=colored_path("C"))

    result = tg.top_n_degree_subgraph(graph, 0, [reference])

    assert list(result) == ["match"]
    assert list(reference.nodes(data="color")) == [(0, "C"), (1, "O"), (2, "H")]


def test_zero_indegree_stripping_is_one_pass_and_returns_a_view_of_a_copy():
    graph = nx.DiGraph([(0, 1), (1, 2)])
    graph.nodes[1]["label"] = "original"

    result = tg.strip_digraph_zero_indegree(graph)

    assert list(result) == [1, 2]
    assert nx.is_frozen(result)
    result.nodes[1]["label"] = "changed"
    assert graph.nodes[1]["label"] == "original"


def test_canonical_labels_follow_iteration_order_and_preserve_attributes():
    graph = nx.MultiDiGraph(name="original")
    graph.add_node("z", color="C")
    graph.add_node(42, color="O")
    graph.add_edge(42, "z", key="bond", color=2)

    result = tg.canonicalize_node_labels(graph)

    assert isinstance(result, nx.MultiDiGraph)
    assert list(result.nodes(data="color")) == [(0, "C"), (1, "O")]
    assert result.edges[1, 0, "bond"]["color"] == 2
    assert result.name == "original"
    assert list(graph) == ["z", 42]


def test_scrambling_preserves_labels_and_seeded_global_random_behavior():
    graph = nx.path_graph(["a", "b", "c", "d"])
    random_state = random.getstate()
    try:
        expected_random = random.Random(42)
        expected_labels = list(graph)
        expected_random.shuffle(expected_labels)

        result = tg.scramble_node_indices(graph, seed=42)

        assert list(result) == expected_labels
        assert set(result) == set(graph)
        assert random.random() == expected_random.random()
        assert list(graph) == ["a", "b", "c", "d"]
    finally:
        random.setstate(random_state)


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
def test_assembly_file_preserves_insertion_order_and_exact_line_format(
    tmp_path, graph_type
):
    graph = graph_type(name="example")
    graph.add_node(2, color="O")
    graph.add_node(0, color="C")
    graph.add_node(1, color="N")
    graph.add_edge(2, 1, color=2)
    graph.add_edge(0, 1, color=1)
    path = tmp_path / "graph_info"

    assert tg.write_ass_graph_file(graph, path) is None
    assert path.read_text() == "example\n3\n3 2 1 2\nO C N\n2 1"


def test_assembly_file_validates_colors_before_overwriting(tmp_path):
    graph = colored_path("C", "O")
    graph.nodes[0]["color"] = "C H"
    path = tmp_path / "graph_info"
    path.write_text("keep me")

    with pytest.raises(AssertionError, match="Node color for node 0 contains a space"):
        tg.write_ass_graph_file(graph, path)
    assert path.read_text() == "keep me"


def test_graphml_roundtrip_uses_string_node_ids_and_retains_attributes(
    tmp_path, monkeypatch
):
    graph = colored_path("C", "O")
    monkeypatch.chdir(tmp_path)

    assert tg.write_graphml(graph) is None
    result = tg.read_graphml()

    assert list(result.nodes(data="color")) == [("0", "C"), ("1", "O")]
    assert result.edges["0", "1"]["color"] == 1


@pytest.mark.parametrize(
    "smiles, expected_edges",
    [
        ("C.O", []),
        ("[Na+].O", []),
        ("[Na+].[Cl-]", [(0, 1)]),
        ("[Na+].[Cl-].[K+].[F-]", [(2, 3)]),
    ],
)
def test_ionic_molecules_always_return_components_and_join_the_last_charged_pair(
    smiles, expected_edges
):
    graph, mols = tg.create_ionic_molecule(smiles, add_hydrogens=False, sanitize=False)

    assert isinstance(mols, list)
    assert len(mols) == len(smiles.split("."))
    assert all(isinstance(mol, Chem.Mol) for mol in mols)
    assert list(graph.edges()) == expected_edges
    assert all(data["color"] == 6 for *_, data in graph.edges(data=True))


def test_bond_smiles_are_unique_alphabetized_and_use_fallback_for_aromatic_bonds():
    mol = Chem.MolFromSmiles("OCCN.C=O.N#C.c1ccccc1")
    assert tg.get_bond_smi(mol) == {"C-O", "C-C", "C-N", "C=O", "C#N", "C~C"}
