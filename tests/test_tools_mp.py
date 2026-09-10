"""Parallel mapping preserves result order across worker backends."""

import pytest

import assemblytheorytools as att


def _assembly_index(smiles):
    # Process workers need an importable, module-level callable.
    return att.calculate_assembly_index(att.smi_to_nx(smiles), strip_hydrogen=True)[0]


def _add(a, b):
    return a + b


@pytest.mark.parametrize("mapper", [att.mp_calc, att.tp_calc, att.mp_calc_chunked])
def test_parallel_mapping_preserves_assembly_indices_and_input_order(mapper):
    smiles = [
        "C(C(=O)O)N",  # Glycine
        "C[C@@H](C(=O)O)N",  # Alanine
        "C([C@@H](C(=O)O)N)O",  # Serine
        "C1C[C@H](NC1)C(=O)O",  # Proline
        "CC(C)C(C(=O)O)N",  # Valine
        "CC(C)CC(C(=O)O)N",  # Leucine
        "CCC(C)CC(C(=O)O)N",  # Isoleucine
        "C1CCCCC1C(=O)O",  # Cyclohexane carboxylic acid
        "C1=CC=CC=C1C(=O)O",  # Benzoic acid
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    ]

    assert mapper(_assembly_index, smiles, n=2) == [3, 4, 4, 6, 5, 6, 6, 6, 6, 8]


def test_process_starmap_unpacks_arguments_in_order():
    assert att.mp_calc_star(_add, [(1, 2), (3, 4), (5, 6), (7, 8)], n=2) == [
        3,
        7,
        11,
        15,
    ]
