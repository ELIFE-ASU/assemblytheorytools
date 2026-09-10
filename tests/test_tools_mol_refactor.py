"""Regression coverage for molecule-helper behavior preserved during cleanup."""

import networkx as nx
import pytest
from rdkit import Chem

import assemblytheorytools.tools_mol as mol_tools


@pytest.mark.parametrize(
    "standardize", [mol_tools.safe_standardize_mol, mol_tools.standardize_mol]
)
@pytest.mark.parametrize("add_hydrogens", [False, True])
def test_standardization_mutates_input_and_only_copies_when_adding_hydrogens(
    standardize, add_hydrogens
):
    molecule = Chem.MolFromSmiles("c1ccccc1", sanitize=False)

    result = standardize(molecule, add_hydrogens=add_hydrogens)

    assert (result is molecule) == (not add_hydrogens)
    assert molecule.GetNumAtoms() == 6
    assert result.GetNumAtoms() == (12 if add_hydrogens else 6)
    assert sorted(bond.GetBondTypeAsDouble() for bond in molecule.GetBonds()) == [
        1,
        1,
        1,
        2,
        2,
        2,
    ]


@pytest.fixture(params=["smiles", "inchi", "molfile"])
def conversion(request, tmp_path):
    if request.param == "smiles":
        return mol_tools.smi_to_mol, "O", "invalid smiles"
    if request.param == "inchi":
        return mol_tools.inchi_to_mol, "InChI=1S/H2O/h1H2", "invalid inchi"

    valid_path = tmp_path / "water.mol"
    invalid_path = tmp_path / "invalid.mol"
    Chem.MolToMolFile(Chem.MolFromSmiles("O"), str(valid_path))
    invalid_path.write_text("invalid molfile\n")
    return mol_tools.molfile_to_mol, str(valid_path), str(invalid_path)


@pytest.mark.parametrize(
    "sanitize, add_hydrogens, atom_count",
    [
        (False, False, 1),
        (False, True, 1),
        (True, False, 1),
        (True, True, 3),
    ],
)
def test_conversion_only_adds_hydrogens_during_sanitization(
    conversion, sanitize, add_hydrogens, atom_count
):
    convert, source, _ = conversion

    molecule = convert(source, sanitize=sanitize, add_hydrogens=add_hydrogens)

    assert molecule.GetNumAtoms() == atom_count


def test_conversion_preserves_invalid_input_behavior(conversion):
    convert, _, source = conversion

    assert convert(source, sanitize=False) is None
    with pytest.raises(AttributeError, match="UpdatePropertyCache"):
        convert(source)


def test_disconnected_smiles_warns_and_retains_both_fragments():
    with pytest.warns(
        UserWarning, match="Disconnected molecules detected in SMILES string"
    ):
        molecule = mol_tools.smi_to_mol("C.O", sanitize=False)

    assert len(Chem.GetMolFrags(molecule)) == 2


def test_smiles_standardization_retains_hydrogen_and_error_behavior():
    assert mol_tools.standardise_smiles("O", sanitize=False) == "[H]O[H]"
    assert mol_tools.standardise_smiles("OCC", add_hydrogens=False) == "CCO"
    with pytest.raises(ValueError, match="^Invalid SMILES: invalid smiles$"):
        mol_tools.standardise_smiles("invalid smiles", add_hydrogens=False)
    with pytest.raises(AttributeError, match="UpdatePropertyCache"):
        mol_tools.standardise_smiles("invalid smiles")


@pytest.mark.parametrize("method, expected", [("1", 3), ("2", -1), ("3", 5)])
def test_free_valence_methods_distinguish_bracket_hydrogens(method, expected):
    molecule = Chem.MolFromSmiles("[NH4+]")

    assert (
        mol_tools.get_free_valence(molecule.GetAtomWithIdx(0), method=method)
        == expected
    )
    assert mol_tools.get_total_free_valence(molecule, method=method) == expected


def test_free_valence_rejects_unknown_methods():
    atom = Chem.MolFromSmiles("C").GetAtomWithIdx(0)

    with pytest.raises(
        ValueError,
        match="^Unknown method unknown for calculating free valence of atom C$",
    ):
        mol_tools.get_free_valence(atom, method="unknown")


def test_graph_valence_truncates_each_edge_and_ignores_molecule_method():
    graph = nx.cycle_graph(3)
    nx.set_node_attributes(graph, "C", "color")
    nx.set_edge_attributes(graph, 1.5, "color")

    assert mol_tools.get_total_free_valence(graph, method="unused") == 6


def test_charge_reset_returns_a_copy_and_leaves_original_charges_unchanged():
    molecule = Chem.MolFromSmiles("C=O")

    result = mol_tools.reset_mol_charge(molecule)

    assert result is not molecule
    assert [atom.GetFormalCharge() for atom in molecule.GetAtoms()] == [0, 0]
    assert [atom.GetFormalCharge() for atom in result.GetAtoms()] == [2, 0]


def test_combine_molecules_preserves_empty_and_single_input_contracts():
    molecule = Chem.MolFromSmiles("CO")
    molecules = (molecule,)

    empty = mol_tools.combine_mols([])
    assert isinstance(empty, Chem.RWMol)
    assert empty.GetNumAtoms() == 0
    assert mol_tools.combine_mols(molecule) is molecule
    assert mol_tools.combine_mols(molecules) is molecules

    combined = mol_tools.combine_mols([molecule])
    assert combined is not molecule
    assert Chem.MolToSmiles(combined) == "CO"


def test_element_set_ignores_missing_and_empty_molecules():
    assert mol_tools.get_element_set_from_mols(
        [None, Chem.Mol(), Chem.MolFromSmiles("CO"), Chem.MolFromSmiles("NCl")]
    ) == {"C", "O", "N", "Cl"}


def test_bracket_token_replacement_retains_existing_matching_rules():
    source = "[CH3][Cl][SiH2][nH][Na+][13CH3][C@H][C:1]"

    assert mol_tools.smi_remove_implicit_hydrogen(source) == (
        "[C][C][S][n][Na+][13CH3][C@H][C:1]"
    )


@pytest.mark.parametrize(
    "canonical, expected",
    [
        (True, "C[C@H](N)C(=O)NCC(=O)O"),
        (False, "N[C@H](C(=O)NCC(=O)O)C"),
    ],
)
def test_peptide_normalizes_sequence_and_respects_canonical_flag(canonical, expected):
    assert mol_tools.peptide_to_smiles(" a\n g\t", canonical=canonical) == expected


@pytest.mark.parametrize(
    "sequence, message",
    [
        (" \n\t", "Empty sequence"),
        ("GAZ", "Invalid amino acid code(s). Allowed: ACDEFGHIKLMNPQRSTVWY"),
    ],
)
def test_peptide_validation_preserves_error_messages(sequence, message):
    with pytest.raises(ValueError) as error:
        mol_tools.peptide_to_smiles(sequence)

    assert str(error.value) == message


def test_peptide_reports_rdkit_build_failure(monkeypatch):
    monkeypatch.setattr(mol_tools.Chem, "MolFromFASTA", lambda sequence: None)

    with pytest.raises(
        ValueError, match="RDKit could not parse/build the peptide from this sequence"
    ):
        mol_tools.peptide_to_smiles("AG")
