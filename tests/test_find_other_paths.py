"""Sampling alternative assembly pathways and evaluating their fragments."""

import numpy as np
import pytest
from rdkit import Chem

import assemblytheorytools as att
from assemblytheorytools import find_other_paths


def test_shortest_path_sampling_returns_unique_valid_fragments():
    molecule = att.smi_to_mol("C#CCC=C")

    paths = att.all_shortest_paths(molecule, f_graph_care=False)

    assert isinstance(paths, list)
    assert paths
    assert len(paths) == len(set(paths))
    for smiles in paths:
        assert isinstance(smiles, str)
        fragment = Chem.MolFromSmiles(smiles)
        assert fragment is not None
        assert 0 < fragment.GetNumAtoms() <= molecule.GetNumAtoms()
        assert fragment.GetNumBonds() <= molecule.GetNumBonds()


@pytest.mark.integration
def test_virtual_object_energy_is_finite_for_every_path(orca_path):
    paths = att.all_shortest_paths(att.smi_to_mol("CC"), f_graph_care=False)
    molecules = [att.smi_to_mol(smiles) for smiles in paths]

    energies = att.get_virtual_objects_energy(molecules, orca_path=orca_path)

    assert paths
    assert len(energies) == len(paths)
    assert np.isfinite(energies).all()


def test_atom_order_is_a_canonical_permutation_without_mutating_molecule():
    mol = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
    original = Chem.MolToMolBlock(mol)

    order = find_other_paths._get_atom_order(mol)
    ranks = list(Chem.CanonicalRankAtoms(mol, includeChirality=True))

    assert sorted(order) == list(range(mol.GetNumAtoms()))
    assert [ranks[index] for index in order] == sorted(ranks)
    assert Chem.MolToMolBlock(mol) == original


@pytest.mark.parametrize("items", [[], [1], list(range(20))])
def test_scramble_copies_input_and_respects_numpy_seed(items):
    original = items.copy()
    random_state = np.random.get_state()
    try:
        np.random.seed(42)
        first = find_other_paths._scramble_list(items)
        np.random.seed(42)
        second = find_other_paths._scramble_list(items)
    finally:
        np.random.set_state(random_state)

    assert first is not items
    assert items == original
    assert sorted(first) == sorted(items)
    assert first == second
    if len(items) > 1:
        assert first != items


@pytest.mark.parametrize(
    "smiles, batches, max_attempts, expected, expected_calls",
    [
        (
            "CCC",
            [["CC", "CC"], ["CC"], ["CCC", "CC"], ["CCC"], [], ["C"]],
            2,
            {"CC", "CCC"},
            5,
        ),
        ("CCC", [[], [], ["CC"]], 2, set(), 2),
        (
            "CC",
            [["C"], ["CC"], ["CCC"], ["CCCC"], ["CCCCC"]],
            3,
            {"C", "CC", "CCC", "CCCC"},
            4,
        ),
    ],
    ids=["deduplicate-and-reset-stale-count", "empty-results-stop", "bond-budget"],
)
def test_sampling_stops_at_stale_or_bond_budget(
    monkeypatch, smiles, batches, max_attempts, expected, expected_calls
):
    calls = []

    def calculate(mol, **settings):
        virtual_objects = batches[len(calls)]
        calls.append(mol)
        return 0, virtual_objects, None

    monkeypatch.setattr(find_other_paths, "calculate_assembly_index", calculate)

    result = att.all_shortest_paths(
        Chem.MolFromSmiles(smiles), max_attempts=max_attempts
    )

    assert isinstance(result, list)
    assert set(result) == expected
    assert len(result) == len(expected)
    assert len(calls) == expected_calls


@pytest.mark.parametrize(
    "smiles, max_attempts", [("CC", 0), ("CC", -1), ("C", 3), ("", 3)]
)
def test_sampling_without_attempts_skips_calculator(monkeypatch, smiles, max_attempts):
    def calculate(*args, **kwargs):
        pytest.fail("The calculator must not run without an attempt budget")

    monkeypatch.setattr(find_other_paths, "calculate_assembly_index", calculate)

    mol = Chem.MolFromSmiles(smiles)
    assert att.all_shortest_paths(mol, max_attempts=max_attempts) == []


@pytest.mark.parametrize("mol", [None, "CC", 42, []])
def test_sampling_rejects_non_molecules(mol):
    with pytest.raises(ValueError, match=r"^Input must be an RDKit molecule object\.$"):
        att.all_shortest_paths(mol)


@pytest.mark.parametrize("settings", [None, {}, {"canonicalize": True, "timeout": 12}])
def test_sampling_forwards_settings_and_preserves_existing_mutation(
    monkeypatch, settings
):
    supplied_settings = settings.copy() if settings is not None else None
    expected = {**(settings or {}), "canonicalize": False}
    calls = []

    def calculate(mol, **kwargs):
        calls.append(kwargs)
        return 0, [], None

    monkeypatch.setattr(find_other_paths, "calculate_assembly_index", calculate)

    att.all_shortest_paths(Chem.MolFromSmiles("CC"), supplied_settings, max_attempts=1)

    assert calls == [expected]
    assert supplied_settings == (expected if settings else settings)


@pytest.mark.parametrize("f_graph_care", [False, True])
def test_sampling_kekulizes_only_the_renumbered_copy(monkeypatch, f_graph_care):
    mol = Chem.MolFromSmiles("c1ccccc1")
    original = Chem.MolToMolBlock(mol, kekulize=False)
    calls = []

    def calculate(sample, **settings):
        calls.append(sample)
        return 0, [], None

    monkeypatch.setattr(find_other_paths, "calculate_assembly_index", calculate)

    att.all_shortest_paths(mol, f_graph_care=f_graph_care, max_attempts=1)

    assert len(calls) == 1
    assert calls[0] is not mol
    expected = (
        {Chem.BondType.SINGLE, Chem.BondType.DOUBLE}
        if f_graph_care
        else {Chem.BondType.AROMATIC}
    )
    assert {bond.GetBondType() for bond in calls[0].GetBonds()} == expected
    assert Chem.MolToMolBlock(mol, kekulize=False) == original
