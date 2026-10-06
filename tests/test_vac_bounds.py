"""Vector addition chain lower bounds, delegated to assemblycfg."""

import warnings

import pytest

import assemblytheorytools as att


def test_molecules_are_bounded_on_the_calculator_graph(monkeypatch):
    """A Mol is converted as calculate_assembly_index converts it, hydrogens included."""
    mol = att.smi_to_mol("[H]C#C[H]")
    _, info = att.calculate_assembly_index_vac_lower_bound(mol, use_vac=False,
                                                           return_info=True)
    assert dict(zip(info["units"], info["vectors"][0])) == {
        ("C", "H", 1): 2, ("C", "C", 3): 1}
    # The caller's molecule is left as it was.
    assert mol.GetNumAtoms() == att.smi_to_mol("[H]C#C[H]").GetNumAtoms()

    graph = att.mol_to_nx(att.smi_to_mol("CCO"))
    assert (att.calculate_assembly_index_vac_lower_bound(att.smi_to_mol("CCO"), use_vac=False)
            == att.calculate_assembly_index_vac_lower_bound(graph, use_vac=False) == 6)


def test_lists_and_strings_are_passed_through():
    assert att.calculate_assembly_index_vac_lower_bound("abab", use_vac=False) == 2
    mols = [att.smi_to_mol("CC"), att.smi_to_mol("CCC")]
    assert att.calculate_assembly_index_vac_lower_bound(mols, strip_hydrogen=True,
                                                        use_vac=False) == 1


@pytest.mark.integration
@pytest.mark.parametrize("smi", ["CCO", "c1ccccc1", "OC(=O)CCC(=O)O",
                                 "NCC(=O)NCC(=O)O", "CC(=O)Oc1ccccc1C(=O)O"])
@pytest.mark.parametrize("strip_hydrogen", [False, True])
def test_vac_bound_brackets_between_scalar_bound_and_index(smi, strip_hydrogen):
    """The vac bound is at least the scalar bound and at most the exact index."""
    mol = att.smi_to_mol(smi)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # vac must actually run
        bound = att.calculate_assembly_index_vac_lower_bound(mol, strip_hydrogen=strip_hydrogen)
    scalar = att.calculate_assembly_index_lower_bound(mol, strip_hydrogen=strip_hydrogen)
    ai = att.calculate_assembly_index(mol, strip_hydrogen=strip_hydrogen)[0]
    assert scalar <= bound <= ai


@pytest.mark.integration
@pytest.mark.parametrize("data", ["bbcbaabab", "cabacbbc", "abracadabra",
                                  ["abcabc", "abcab"], ["aaaa", "bbbb"]])
def test_vac_bound_never_exceeds_string_index(data):
    bound = att.calculate_assembly_index_vac_lower_bound(data)
    assert bound <= att.calculate_string_assembly_index(data)[0]
