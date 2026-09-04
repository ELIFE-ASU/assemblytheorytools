"""
Tests for the ensemble assembly quantities.

These cover the assembly equation evaluated from precomputed indices, the
copy-number aggregation that feeds it, the joint assembly space built from
individual pathways, and the exploration ratio measured over that space.
None of them needs the external C++ calculator.
"""

import networkx as nx
import numpy as np
import pytest
from rdkit import Chem

import assemblytheorytools as att


# Indices and copy numbers of the molecular ensemble in
# `test_assembly_mols.test_calculate_assembly`: glycine, alanine, glycine,
# ethane and glycine, with hydrogens stripped.
MOL_INDICES = [3, 4, 3, 0, 3]
MOL_COPIES = [1, 2, 3, 4, 5]
MOL_REFERENCE = 11.87409143815135

# The string ensemble in `test_assembly_strings.test_bigA`: abab, cdcdcdcd
# and c.
STR_INDICES = [2, 3, 0]
STR_COPIES = [10, 100, 40]
STR_REFERENCE = 13.9597977352397


def test_calculate_assembly_from_indices_molecular_reference():
    """
    Test that the assembly equation reproduces the molecular reference value.

    The inputs are the assembly indices and copy numbers of the ensemble in
    `test_calculate_assembly`, so this pins the same arithmetic without
    invoking the external calculator.

    Asserts:
        - The result equals the reference value exactly.
    """
    assert att.calculate_assembly_from_indices(
        MOL_INDICES, MOL_COPIES) == MOL_REFERENCE


def test_calculate_assembly_from_indices_string_reference():
    """
    Test that the assembly equation reproduces the string reference value.

    Asserts:
        - The result equals the value `test_bigA` computes term by term.
    """
    n_t = sum(STR_COPIES)
    expected = sum(np.exp(ai) * ((n - 1) / n_t)
                   for ai, n in zip(STR_INDICES, STR_COPIES))

    assert att.calculate_assembly_from_indices(
        STR_INDICES, STR_COPIES) == expected == STR_REFERENCE


def test_calculate_assembly_from_indices_returns_a_float():
    """
    Test that the return value is a plain float, as annotated.

    Asserts:
        - The result is exactly a `float`, not a NumPy scalar.
    """
    result = att.calculate_assembly_from_indices(MOL_INDICES, MOL_COPIES)

    assert type(result) is float


def test_calculate_assembly_from_indices_regularises_failed_indices():
    """
    Test that failed assembly indices are regularised rather than dropped.

    `None` and negative values mark a calculation that failed or timed out.
    They are treated as zero, so the object still contributes its copy-number
    weight.

    Asserts:
        - `None` and a negative index give the same result as zero.
    """
    failed = att.calculate_assembly_from_indices([None, -1, 2], [2, 2, 2])
    zeroed = att.calculate_assembly_from_indices([0, 0, 2], [2, 2, 2])

    assert failed == zeroed


def test_calculate_assembly_from_indices_ignores_single_copies():
    """
    Test that an object observed once contributes nothing of its own.

    Its `n_i - 1` factor is zero, so its assembly index cannot affect the
    result. It still enters the total copy number, which dilutes every other
    term, so the value is lower than if it had not been observed at all.

    Asserts:
        - Changing the singleton's assembly index leaves the result unchanged.
        - Observing the singleton lowers the result through `N_T` alone.
    """
    complex_singleton = att.calculate_assembly_from_indices(
        [1, 9], [100.0, 1.0])
    simple_singleton = att.calculate_assembly_from_indices(
        [1, 2], [100.0, 1.0])
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
    """
    Test that malformed input raises `ValueError` rather than misbehaving.

    Mismatched lengths used to be truncated silently by `zip`, and copy
    numbers summing to zero used to raise `ZeroDivisionError`.

    Asserts:
        - Each case raises `ValueError` with an explanatory message.
    """
    with pytest.raises(ValueError, match=match):
        att.calculate_assembly_from_indices(ai_list, n_i)


def test_count_copies_collapses_repeats_in_first_seen_order():
    """
    Test that repeated objects collapse to one entry carrying the count.

    Asserts:
        - The unique objects keep their first-seen order.
        - The counts align with them.
    """
    unique, counts = att.count_copies(["abab", "cdcd", "abab", "abab"])

    assert unique == ["abab", "cdcd"]
    assert counts == [3, 1]


def test_count_copies_of_an_empty_input():
    """
    Test that an empty input gives two empty lists.

    Asserts:
        - Both returned lists are empty.
    """
    assert att.count_copies([]) == ([], [])


def test_count_copies_with_a_key_function():
    """
    Test that `key` decides identity, for objects that do not compare by value.

    Two RDKit molecules parsed from the same SMILES are distinct objects, so
    they only collapse when an InChI key function is supplied.

    Asserts:
        - Without a key the two molecules stay separate.
        - With `Chem.MolToInchi` they collapse to one entry counted twice.
    """
    mols = [Chem.MolFromSmiles(smi) for smi in ("CCO", "CCO", "CC")]

    assert att.count_copies(mols)[1] == [1, 1, 1]

    unique, counts = att.count_copies(mols, key=Chem.MolToInchi)

    assert counts == [2, 1]
    assert [Chem.MolToSmiles(mol) for mol in unique] == ["CCO", "CC"]


def test_count_copies_feeds_the_assembly_equation():
    """
    Test that the counted output is accepted by the assembly equation.

    Asserts:
        - Counting three strings and passing the pair through gives the value
          computed from the collapsed ensemble directly.
    """
    strings, n_i = att.count_copies(["ab", "ab", "cd"])

    assert strings == ["ab", "cd"]
    assert att.calculate_assembly_from_indices([1, 1], n_i) == (
        att.calculate_assembly_from_indices([1, 1], [2, 1]))


def _string_pathway(*edges):
    """
    Build a small pathway whose nodes are the substrings they represent.

    Parameters
    ----------
    *edges : tuple of str
        Directed edges, each a `(source, target)` pair.

    Returns
    -------
    nx.DiGraph
        The pathway.
    """
    path = nx.DiGraph()
    path.add_edges_from(edges)
    return path


def test_joint_assembly_space_merges_shared_intermediates():
    """
    Test that composing pathways shares the intermediates they have in common.

    Asserts:
        - The union holds each distinct object once.
        - Edges from both pathways survive.
    """
    first = _string_pathway(("a", "ab"), ("b", "ab"))
    second = _string_pathway(("a", "ab"), ("b", "ab"), ("ab", "abb"),
                             ("b", "abb"))

    space = att.joint_assembly_space([first, second])

    assert set(space.nodes) == {"a", "b", "ab", "abb"}
    assert space.number_of_edges() == 4


def test_joint_assembly_space_relabels_by_node_key():
    """
    Test that `node_key` merges pathways whose node identifiers are positional.

    The molecular calculator labels nodes `step_N` and `virtual_object_N` and
    carries the object itself in a `vo` attribute, so two pathways only share
    intermediates once they are relabelled by it.

    Asserts:
        - Without `node_key` the positional labels keep the pathways apart.
        - With it the shared object appears once.
    """
    first = nx.DiGraph()
    first.add_edge("virtual_object_0", "step_1")
    nx.set_node_attributes(first, {"virtual_object_0": "a", "step_1": "ab"},
                           "vo")

    second = nx.DiGraph()
    second.add_edge("virtual_object_1", "step_1")
    nx.set_node_attributes(second, {"virtual_object_1": "b", "step_1": "ab"},
                           "vo")

    assert att.joint_assembly_space([first, second]).number_of_nodes() == 3
    assert set(att.joint_assembly_space([first, second],
                                        node_key="vo").nodes) == {"a", "b", "ab"}


def test_joint_assembly_space_rejects_an_empty_input():
    """
    Test that composing no pathways raises rather than returning an empty graph.

    Asserts:
        - `ValueError` is raised.
    """
    with pytest.raises(ValueError, match="not be empty"):
        att.joint_assembly_space([])


def test_exploration_ratio_counts_observed_against_contingent():
    """
    Test the exploration ratio on a pathway pair with a known answer.

    The union holds four objects, `abab` and `abb` are the pathway targets,
    so two of the four were observed.

    Asserts:
        - The ratio is 0.5.
        - Passing `observed` explicitly gives the same answer.
    """
    first = _string_pathway(("a", "ab"), ("b", "ab"))
    second = _string_pathway(("a", "ab"), ("b", "ab"), ("ab", "abb"),
                             ("b", "abb"))

    assert att.exploration_ratio([first, second]) == 0.5
    assert att.exploration_ratio([first, second],
                                 observed=["ab", "abb"]) == 0.5


def test_exploration_ratio_of_a_single_pathway():
    """
    Test the exploration ratio of one sequence built from its own pathway.

    The sequence `gavhp` repeats no character, so its minimum pathway is nine
    nodes: five units, three intermediates and the sequence itself.

    Asserts:
        - Exactly one of the nine objects was observed.
    """
    pathway = att.calculate_string_assembly_index("gavhp", mode="cfg")[2]

    assert pathway.number_of_nodes() == 9
    assert att.exploration_ratio([pathway]) == pytest.approx(1 / 9)


def test_exploration_ratio_is_higher_for_a_shared_ensemble():
    """
    Test that sequences sharing intermediates explore their space more fully.

    An ensemble that also observes its own intermediates covers more of the
    joint assembly space than one that observes only the endpoints, which is
    the contrast the ratio is designed to detect.

    Asserts:
        - Adding the intermediates to the observed set raises the ratio.
    """
    pathways = [att.calculate_string_assembly_index(s, mode="cfg")[2]
                for s in ("gav", "gavh", "gavhp")]

    endpoints_only = att.exploration_ratio(pathways)
    everything = att.exploration_ratio(
        pathways, observed=att.joint_assembly_space(pathways).nodes)

    assert endpoints_only < everything == 1.0


def test_exploration_ratio_rejects_an_empty_input():
    """
    Test that an empty pathway list raises rather than dividing by zero.

    Asserts:
        - `ValueError` is raised.
    """
    with pytest.raises(ValueError, match="not be empty"):
        att.exploration_ratio([])


def test_calculate_assembly_from_indices_accepts_arrays_and_columns():
    """
    Test that any sized sequence works, not only a list.

    NumPy arrays and DataFrame columns are what a tabular workflow has to
    hand, and both raise on a plain truth test, so the emptiness check has to
    go through `len`.

    Asserts:
        - Arrays and DataFrame columns give the same value as lists.
    """
    import pandas as pd

    frame = pd.DataFrame({'a_i': MOL_INDICES, 'n_i': MOL_COPIES})

    assert att.calculate_assembly_from_indices(
        np.array(MOL_INDICES), np.array(MOL_COPIES, dtype=float)
    ) == MOL_REFERENCE
    assert att.calculate_assembly_from_indices(
        frame['a_i'], frame['n_i']) == MOL_REFERENCE
    assert att.calculate_assembly_from_indices(
        tuple(MOL_INDICES), tuple(MOL_COPIES)) == MOL_REFERENCE
