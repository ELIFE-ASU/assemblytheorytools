"""Compatibility checks for reaction data, graph metadata and sampling helpers."""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import networkx as nx
import pytest
from rdkit import Chem

from assemblytheorytools import reassembler


def test_reaction_library_preserves_every_pattern_and_its_position():
    smarts, reactions = reassembler.assemble_smarts()

    assert [len(group) for group in smarts] == [14, 512, 14]
    # These fingerprints capture the complete pre-refactor reaction library,
    # including atom mappings, duplicate templates and ordering.
    assert [
        hashlib.sha256(json.dumps(group, separators=(",", ":")).encode()).hexdigest()
        for group in smarts
    ] == [
        "f2389f964c4e96327a5e1ccc4b77ad6650ba773d93f8b637598dbdd5c38eae5a",
        "2fdeca721153f71e2721069c98244e88b94e5d267bc060bc74a8c2f555fe7bcb",
        "a5708401e62f77fcd8659628e1a09328eb417a21b8acf809ce1fcaaa0d66ed12",
    ]
    assert reactions == [[None] * 14, [None] * 512, [None] * 14]
    assert reassembler.origami_smarts() == smarts[2]


def test_reaction_library_returns_independent_mutable_lists():
    smarts, reactions = reassembler.assemble_smarts()
    expected = deepcopy(smarts)
    origami = reassembler.origami_smarts()

    smarts[0][0] = "changed"
    smarts[2].pop()
    reactions[0][0] = object()
    reactions[1].clear()
    origami.clear()

    fresh_smarts, fresh_reactions = reassembler.assemble_smarts()
    assert fresh_smarts == expected
    assert fresh_reactions == [[None] * 14, [None] * 512, [None] * 14]
    assert reassembler.origami_smarts() == expected[2]
    assert reactions[2] == [None] * 14


@pytest.mark.parametrize("count", [0, 50, 51])
def test_printer_preserves_batch_boundaries_labels_and_filenames(monkeypatch, count):
    molecules = [object() for _ in range(count)]
    images = [Mock(), Mock()]
    draw = Mock(side_effect=images)
    monkeypatch.setattr(reassembler.Draw, "MolsToGridImage", draw)

    assert reassembler.printer(molecules) is None

    assert draw.call_count == (count + 49) // 50
    if count:
        assert draw.call_args_list[0].args == (molecules[:50],)
        assert draw.call_args_list[0].kwargs == {
            "molsPerRow": 10,
            "legends": [str(index) for index in range(50)],
        }
        images[0].save.assert_called_once_with("Mols_0-50.png")
    if count == 51:
        assert draw.call_args_list[1].args == (molecules[50:],)
        assert draw.call_args_list[1].kwargs == {"molsPerRow": 10, "legends": ["50"]}
        images[1].save.assert_called_once_with("Mols_50-100.png")


@pytest.mark.parametrize(
    ("formula", "element", "count"),
    [
        ("C12H22O11", "C", 12),
        ("C12H22O11", "O", 11),
        ("H2O", "O", 1),
        ("H2O", "N", 0),
        ("CaCl2", "Cl", 2),
        ("CH3OH", "H", 3),  # The first occurrence determines the count.
        ("", "C", 0),
    ],
)
def test_get_num_atom_retains_formula_parsing(formula, element, count):
    assert reassembler.get_num_atom(formula, element) == count


def test_pick_two_selects_positions_and_preserves_remaining_order(monkeypatch):
    values = ["same", "middle", "same", "last"]
    choice = Mock(return_value=[2, 0])
    monkeypatch.setattr(reassembler.np.random, "choice", choice)

    assert reassembler.pick_two(values) == (["same", "same"], ["middle", "last"])
    choice.assert_called_once_with(4, 2, replace=False)
    assert values == ["same", "middle", "same", "last"]
    assert reassembler.pick_two([]) == ([], [])
    assert reassembler.pick_two(["only"]) == ([], [])


def test_interval_count_retains_input_sorting_and_empty_result():
    intervals = [[5, 7], [0, 2], [2, 4], [3, 6]]
    assert reassembler.count_non_overlapping_sublists(intervals) == 2
    assert intervals == [[0, 2], [2, 4], [3, 6], [5, 7]]
    assert reassembler.count_non_overlapping_sublists([]) == 1


def test_allowed_pairs_reject_conflicts_without_mutating_candidates(monkeypatch):
    candidates = [[0, 1], [0, 2], [3, 1], [3, 2]]
    choice = Mock(side_effect=[candidates[0], candidates[1], candidates[2], candidates[3]])
    monkeypatch.setattr(reassembler.random, "choice", choice)

    selected = reassembler.get_allowed_pairs(candidates, k=3, max_iterations=4)

    assert selected == [[0, 1], [3, 2]]
    assert selected[0] is candidates[0]
    assert selected[1] is candidates[3]
    assert candidates == [[0, 1], [0, 2], [3, 1], [3, 2]]
    assert choice.call_count == 4
    assert reassembler.get_allowed_pairs([], k=0) == []


def test_possible_combinations_filter_valence_and_shuffle_once(monkeypatch):
    first = {5: ["C", 3], 2: ["O", 1], 0: ["C", 1]}
    second = {7: ["C", 1], 4: ["O", 1]}
    shuffle = Mock(side_effect=lambda pairs: pairs.reverse())
    monkeypatch.setattr(reassembler.random, "shuffle", shuffle)

    assert reassembler.get_possible_combinations(first, second) == [[2, 4], [5, 7]]
    shuffle.assert_called_once()
    assert reassembler.get_possible_combinations({0: ["N", 1]}, second) is None
    assert shuffle.call_count == 1


def test_unique_molecules_keep_first_objects_in_input_order():
    first = Chem.MolFromSmiles("CCO")
    duplicate = Chem.MolFromSmiles("OCC")
    second = Chem.MolFromSmiles("CO")
    molecules = [None, first, duplicate, second, first]

    assert reassembler.get_unique_mols(molecules) == [first, second]
    assert molecules == [None, first, duplicate, second, first]
    assert reassembler.get_unique_mols([]) == []


def test_legacy_reassembly_retries_failed_stages_without_changing_sampling(monkeypatch, capsys):
    first = Chem.MolFromSmiles("CC")
    second = Chem.MolFromSmiles("CO")
    product = Chem.MolFromSmiles("CCO")
    molecules = [first, second]
    sampled_indices = iter([0, 1, 0, 0] * 4)
    events = []

    def randint(lower, upper):
        events.append(("randint", lower, upper))
        return next(sampled_indices)

    def pick_indices(size, count, replace):
        events.append(("choice", size, count, replace))
        return [0, 1]

    def record_stage(name, results):
        values = iter(results)

        def run(*args):
            events.append((name,))
            return next(values)

        return Mock(side_effect=run)

    assembly = record_stage("assemble", [None, product, product, product])
    structure_filter = record_stage("filter", [None, product, product])
    conformation_filter = record_stage("conformation", [None, product])
    monkeypatch.setattr(reassembler.random, "randint", randint)
    monkeypatch.setattr(reassembler.np.random, "choice", pick_indices)
    monkeypatch.setattr(reassembler.rdMolDescriptors, "CalcExactMolWt", lambda mol, onlyHeavy: 20)
    monkeypatch.setattr(reassembler, "assemble", assembly)
    monkeypatch.setattr(reassembler, "filter_mol", structure_filter)
    monkeypatch.setattr(reassembler, "conformation_filter", conformation_filter)

    result = reassembler.reassemble_old(
        molecules,
        n_mol_needed=1,
        mw_min=30,
        mw_max=50,
        mw_delta=0,
        one_atom_weight=0,
        unsat_min=0,
        unsat_max=10,
    )

    assert result == [product]
    sampling = [("randint", 0, 1)] * 3 + [("randint", 0, 0), ("choice", 2, 2, False)]
    assert events == (
        sampling
        + [("assemble",)]
        + sampling
        + [("assemble",), ("filter",)]
        + sampling
        + [("assemble",), ("filter",), ("conformation",)]
        + sampling
        + [("assemble",), ("filter",), ("conformation",)]
    )
    assert all(call.args == (first, second, 1) for call in assembly.call_args_list)
    assert molecules == [first, second]
    assert "nFiltered = 2" in capsys.readouterr().out


def test_fragment_combination_preserves_both_input_molecules():
    first = Chem.MolFromSmiles("CC")
    second = Chem.MolFromSmiles("CO")
    before = [Chem.MolToMolBlock(mol) for mol in (first, second)]

    assert reassembler.combine_fragments(first, second, [(0, 0)]) == "CCO"
    assert [Chem.MolToMolBlock(mol) for mol in (first, second)] == before


def test_atomic_distribution_distinguishes_invalid_and_unreactive_nodes():
    graph = nx.DiGraph()
    graph.add_nodes_from(["CCO", "N#N", "not a molecule"])

    distribution = reassembler.get_atomic_distribution(graph)

    assert dict(distribution) == {"CCO": {6, 8}, "not a molecule": None}
    assert distribution["N#N"] == set()


def test_atom_mapping_removes_explicit_hydrogens_and_reports_free_valence():
    mol, mapping = reassembler.get_atom_type_index_mapping("[H]OC")

    assert Chem.MolToSmiles(mol) == "CO"
    assert mapping == {0: ["O", 1], 1: ["C", 3]}
    assert reassembler.get_atom_type_index_mapping("not a molecule") == (None, None)


def test_graph_composition_preserves_order_and_accumulates_parallel_edges():
    first = nx.MultiDiGraph(source="first", retained="metadata")
    first.add_nodes_from(
        [
            ("CO", {"level": 0, "custom": "discarded"}),
            ("CC", {"level": 2}),
            ("CCO", {"level": 2}),
        ]
    )
    first.add_edge("CC", "CCO", custom="discarded")
    first.add_edge("CC", "CCO")
    first.add_edge("CO", "CCO")
    second = nx.DiGraph(source="second")
    second.add_nodes_from(
        [
            ("CCO", {"level": 1}),
            ("CC", {"level": 0}),
            ("CCOC", {"level": 2}),
        ]
    )
    second.add_edges_from([("CC", "CCO"), ("CCO", "CCOC")])

    graph = reassembler.compose_all([first, second], get_atomic_count=False)

    assert type(graph) is nx.DiGraph
    assert list(graph) == ["CO", "CC", "CCO", "CCOC"]
    assert graph.graph == {"source": "second", "retained": "metadata"}
    assert dict(graph.nodes(data=True)) == {
        "CO": {"level": 0, "count": 1, "usage": [1]},
        "CC": {"level": 0, "count": 2, "usage": [1]},
        "CCO": {"level": 1, "count": 2, "usage": [2]},
        "CCOC": {"level": 2, "count": 1, "usage": []},
    }
    assert list(graph.edges(data=True)) == [
        ("CO", "CCO", {"count": 1}),
        ("CC", "CCO", {"count": 3}),
        ("CCO", "CCOC", {"count": 1}),
    ]
    assert first.nodes["CC"]["level"] == 2
    assert first.nodes["CO"]["custom"] == "discarded"


def test_empty_graph_composition_is_rejected():
    # Older implementations reach NetworkX with None before the ValueError.
    with pytest.raises((AttributeError, ValueError)):
        reassembler.compose_all([])


def test_layer_sampling_sums_powered_counts_and_preserves_layer_order(monkeypatch):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            ("a", {"level": 2, "count": 2}),
            ("b", {"level": 0, "count": 3}),
            ("c", {"level": 2, "count": 3}),
            ("d", {"level": 1, "count": 4}),
        ]
    )
    pool = reassembler.MoleculeGenerationAssemblyPool(SimpleNamespace(joined_assembly_graph=graph))
    choice = Mock(return_value=[0])
    monkeypatch.setattr(reassembler.random, "choices", choice)

    assert pool.sample_layer(curr_depth=2, black_listed_layers=[1]) == 0
    assert dict(pool.layer_sampling_weights) == {2: 13, 0: 9, 1: 16}
    choice.assert_called_once_with([2, 0], weights=[13, 9], k=1)
    pool.sample_layer(exponent=1, curr_depth=0)
    choice.assert_called_with([0], weights=[9], k=1)


def test_fragment_sampling_preserves_compatibility_and_inverse_weights(monkeypatch):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            ("base", {"atomic_count": {6}}),
            ("rare", {"atomic_count": {6, 8}, "count": 2}),
            ("incompatible", {"atomic_count": {8}, "count": 10}),
            ("common", {"atomic_count": {6}, "count": 4}),
        ]
    )
    pool = reassembler.MoleculeGenerationAssemblyPool(
        SimpleNamespace(joined_assembly_graph_minus_x=graph)
    )
    pool.level_to_fragment = {0: ["rare", "incompatible", "common"], 1: ["incompatible"]}
    choice = Mock(return_value=["rare"])
    monkeypatch.setattr(reassembler.random, "choices", choice)

    # The existing blacklist parameter is accepted but does not exclude nodes.
    assert pool.wrs_from_layer("base", inverse=True, exponent=2, blacklist=["rare"]) == ("rare", 2)
    choice.assert_called_once_with(["rare", "common"], weights=[0.25, 0.0625], k=1)
    assert pool.wrs_from_layer("base", layer=1) == (None, 0)
    assert choice.call_count == 1


def test_step_sampling_keeps_zero_weight_gaps(monkeypatch):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [
            ("branch", {"level": 1}),
            ("middle", {"level": 2}),
            ("last", {"level": 4}),
        ]
    )
    graph.add_edge("branch", "last")
    pool = reassembler.MoleculeGenerationAssemblyPool(
        SimpleNamespace(
            joined_assembly_graph=graph,
            leaf_nodes=["last", "branch", "middle"],
            max_assembly_index=4,
        )
    )
    choice = Mock(return_value=[3])
    monkeypatch.setattr(reassembler.random, "choices", choice)

    assert dict(pool.get_leaf_counts_per_level()) == {4: 1, 2: 1, 1: 0, 3: 0}
    pool.set_sw_n_steps()
    assert pool.n_steps_sampling_weights == [0, 0, 1, 0, 1]
    assert pool.weighted_n_steps_sampler(3, min_level=1) == 3
    choice.assert_called_once_with([1, 2, 3], weights=[1, 0, 1], k=1)


def test_assembled_molecule_results_keep_success_order_and_empty_distinction():
    pool = reassembler.MoleculeGenerationAssemblyPool(None)
    assert pool.get_assembled_molecules() is None
    pool.assembled_molecules[0] = []
    assert pool.get_assembled_molecules() == []
    pool.assembled_molecules[3] = [["CC", "CO", "CCO"], ["CCO", "CC", "CCCO"]]
    pool.assembled_molecules[1] = [["CC", "CC", "CCC"]]
    assert pool.get_assembled_molecules() == ["CCCO", "CCC"]
