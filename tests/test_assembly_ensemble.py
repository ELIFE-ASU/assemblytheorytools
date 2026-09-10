"""
Tests for the ensemble assembly quantities.

These cover the assembly equation evaluated from precomputed indices, the
copy-number aggregation that feeds it, the joint assembly space built from
individual pathways, and the exploration ratio measured over that space.
None of them needs the external C++ calculator.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

import assemblytheorytools as att
from assemblytheorytools import assembly

# Indices and copy numbers of the molecular ensemble in
# `test_assembly_mols.test_calculate_assembly`: glycine, alanine, glycine,
# ethane and glycine, with hydrogens stripped.
MOL_INDICES = [3, 4, 3, 0, 3]
MOL_COPIES = [1, 2, 3, 4, 5]
MOL_REFERENCE = 11.87409143815135

# The string ensemble in `test_assembly_strings.test_string_ensemble_assembly`: abab, cdcdcdcd
# and c.
STR_INDICES = [2, 3, 0]
STR_COPIES = [10, 100, 40]
STR_REFERENCE = 13.9597977352397


@pytest.mark.parametrize(
    "indices, copies, expected",
    [
        (MOL_INDICES, MOL_COPIES, MOL_REFERENCE),
        (STR_INDICES, STR_COPIES, STR_REFERENCE),
    ],
    ids=["molecular", "string"],
)
def test_assembly_equation_matches_reference_ensembles(indices, copies, expected):
    result = att.calculate_assembly_from_indices(indices, copies)

    assert type(result) is float
    assert result == expected


def test_calculate_assembly_from_indices_regularises_failed_indices():
    """`None` and negative values mark a calculation that failed or timed out.
    They are treated as zero, so the object still contributes its copy-number
    weight."""
    failed = att.calculate_assembly_from_indices([None, -1, 2], [2, 2, 2])
    zeroed = att.calculate_assembly_from_indices([0, 0, 2], [2, 2, 2])

    assert failed == zeroed


def test_calculate_assembly_from_indices_ignores_single_copies():
    """Its `n_i - 1` factor is zero, so its assembly index cannot affect the
    result. It still enters the total copy number, which dilutes every other
    term, so the value is lower than if it had not been observed at all."""
    complex_singleton = att.calculate_assembly_from_indices([1, 9], [100.0, 1.0])
    simple_singleton = att.calculate_assembly_from_indices([1, 2], [100.0, 1.0])
    unobserved = att.calculate_assembly_from_indices([1], [100.0])

    assert complex_singleton == simple_singleton
    assert complex_singleton < unobserved


@pytest.mark.parametrize(
    "ai_list, n_i, match",
    [
        ([1, 2], [1], "same length"),
        ([1], [1, 2], "same length"),
        ([], [], "not be empty"),
        ([1, 2], [0, 0], "sum to zero"),
    ],
)
def test_calculate_assembly_from_indices_rejects_bad_input(ai_list, n_i, match):
    """Mismatched lengths used to be truncated silently by `zip`, and copy
    numbers summing to zero used to raise `ZeroDivisionError`."""
    with pytest.raises(ValueError, match=match):
        att.calculate_assembly_from_indices(ai_list, n_i)


def test_count_copies_collapses_repeats_in_first_seen_order():
    unique, counts = att.count_copies(["abab", "cdcd", "abab", "abab"])

    assert unique == ["abab", "cdcd"]
    assert counts == [3, 1]


def test_count_copies_of_an_empty_input():
    assert att.count_copies([]) == ([], [])


def test_count_copies_with_a_key_function():
    """Two RDKit molecules parsed from the same SMILES are distinct objects, so
    they only collapse when an InChI key function is supplied."""
    mols = [Chem.MolFromSmiles(smi) for smi in ("CCO", "CCO", "CC")]

    assert att.count_copies(mols)[1] == [1, 1, 1]

    unique, counts = att.count_copies(mols, key=Chem.MolToInchi)

    assert counts == [2, 1]
    assert [Chem.MolToSmiles(mol) for mol in unique] == ["CCO", "CC"]


def test_count_copies_feeds_the_assembly_equation():
    strings, n_i = att.count_copies(["ab", "ab", "cd"])

    assert strings == ["ab", "cd"]
    assert att.calculate_assembly_from_indices([1, 1], n_i) == (
        att.calculate_assembly_from_indices([1, 1], [2, 1])
    )


@pytest.fixture
def shared_string_pathways():
    return [
        nx.DiGraph([("a", "ab"), ("b", "ab")]),
        nx.DiGraph([("a", "ab"), ("b", "ab"), ("ab", "abb"), ("b", "abb")]),
    ]


def test_joint_assembly_space_merges_shared_intermediates(shared_string_pathways):
    space = att.joint_assembly_space(shared_string_pathways)

    assert set(space) == {"a", "b", "ab", "abb"}
    assert set(space.edges) == {("a", "ab"), ("b", "ab"), ("ab", "abb"), ("b", "abb")}


def test_joint_assembly_space_relabels_by_node_key():
    """The molecular calculator labels nodes `step_N` and `virtual_object_N` and
    carries the object itself in a `vo` attribute, so two pathways only share
    intermediates once they are relabelled by it."""
    first = nx.DiGraph()
    first.add_edge("virtual_object_0", "step_1")
    nx.set_node_attributes(first, {"virtual_object_0": "a", "step_1": "ab"}, "vo")

    second = nx.DiGraph()
    second.add_edge("virtual_object_1", "step_1")
    nx.set_node_attributes(second, {"virtual_object_1": "b", "step_1": "ab"}, "vo")

    assert att.joint_assembly_space([first, second]).number_of_nodes() == 3
    assert set(att.joint_assembly_space([first, second], node_key="vo").nodes) == {
        "a",
        "b",
        "ab",
    }


def test_joint_assembly_space_rejects_an_empty_input():
    with pytest.raises(ValueError, match="not be empty"):
        att.joint_assembly_space([])


@pytest.mark.parametrize(
    "observed", [None, ["ab", "abb"]], ids=["inferred", "explicit"]
)
def test_exploration_ratio_counts_observed_against_contingent(
    shared_string_pathways, observed
):
    # The union holds four objects; ab and abb are the two observed targets.
    assert att.exploration_ratio(shared_string_pathways, observed=observed) == 0.5


def test_exploration_ratio_of_a_single_pathway():
    """The sequence `gavhp` repeats no character, so its minimum pathway is nine
    nodes: five units, three intermediates and the sequence itself."""
    pathway = att.calculate_string_assembly_index("gavhp", mode="cfg")[2]

    assert pathway.number_of_nodes() == 9
    assert att.exploration_ratio([pathway]) == pytest.approx(1 / 9)


def test_exploration_ratio_is_higher_for_a_shared_ensemble():
    """An ensemble that also observes its own intermediates covers more of the
    joint assembly space than one that observes only the endpoints, which is
    the contrast the ratio is designed to detect."""
    pathways = [
        att.calculate_string_assembly_index(s, mode="cfg")[2]
        for s in ("gav", "gavh", "gavhp")
    ]

    endpoints_only = att.exploration_ratio(pathways)
    everything = att.exploration_ratio(
        pathways, observed=att.joint_assembly_space(pathways).nodes
    )

    assert endpoints_only < everything == 1.0


def test_exploration_ratio_rejects_an_empty_input():
    with pytest.raises(ValueError, match="not be empty"):
        att.exploration_ratio([])


@pytest.mark.parametrize(
    "container", [tuple, np.array, pd.Series], ids=["tuple", "array", "series"]
)
def test_assembly_equation_accepts_sized_sequences(container):
    assert (
        att.calculate_assembly_from_indices(
            container(MOL_INDICES), container(MOL_COPIES)
        )
        == MOL_REFERENCE
    )


def test_exploration_infers_observed_objects_before_relabelled_paths_merge():
    first = nx.DiGraph([(0, 1)])
    second = nx.DiGraph([(0, 1), (1, 2)])
    nx.set_node_attributes(first, {0: "a", 1: "ab"}, "vo")
    nx.set_node_attributes(second, {0: "a", 1: "ab", 2: "abb"}, "vo")

    assert assembly.exploration_ratio([first, second], node_key="vo") == 2 / 3
    assert (
        assembly.exploration_ratio(
            [first, second], observed=["ab", "ab", "unobserved"], node_key="vo"
        )
        == 1 / 3
    )
    assert list(first) == [0, 1]
    assert list(second) == [0, 1, 2]
