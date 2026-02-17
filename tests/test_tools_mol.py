from rdkit import Chem
from rdkit.Chem import AllChem as Chem

import assemblytheorytools as att


def test_graph_to_mol():
    """
    Test the conversion of a SMILES string to a molecular graph and back to a molecule.

    This function performs the following steps:
    1. Converts a SMILES string to a molecule object.
    2. Converts the molecule object to a NetworkX graph.
    3. Converts the NetworkX graph back to a molecule object.
    4. Checks if the original graph and the graph obtained from the converted molecule are isomorphic.

    Asserts:
        - The graph obtained from the converted molecule is isomorphic to the original graph.
    """
    print(flush=True)
    smi_in = "[Mo](Cl)(Cl)(C#N)(C=O)-[Mo](Cl)(Cl)(C#N)(C=O)"
    # Convert the SMILES string to a molecule object
    mol = att.smi_to_mol(smi_in)
    # Convert the molecule object to a NetworkX graph
    graph = att.mol_to_nx(mol)
    # Convert the NetworkX graph back to a molecule object
    mol_out = att.nx_to_mol(graph)
    # Check if the original graph and the graph obtained from the converted molecule are isomorphic
    assert att.is_graph_isomorphic(graph, att.mol_to_nx(mol_out))


def test_reset_mol_charge():
    """
    Test the functionality of resetting the formal charge of a molecule.

    This function performs multiple tests to verify the behavior of the `reset_mol_charge` function
    and related utilities. It checks the formal charge of molecules in various scenarios, including
    charged and uncharged cases, as well as SMILES-based molecule creation.

    Steps:
    ------
    1. Test a charged molecule graph and verify its formal charge.
    2. Test an uncharged molecule graph and verify its formal charge.
    3. Test a molecule graph with multiple possible charged cases and verify the simplest one is picked.
    4. Test an uncharged molecule created from a SMILES string and verify its formal charge.
    5. Test a charged molecule created from a SMILES string and verify its formal charge.

    Asserts:
    -------
    - The formal charge of the molecule matches the expected value in each test case.

    Notes:
    ------
    - The function uses RDKit utilities to calculate the formal charge of molecules.
    - Molecules are created from graphs or SMILES strings using `assemblytheorytools`.
    """
    print(flush=True)
    print('Testing charged case', flush=True)
    graph = att.ph_2p_graph()
    mol = att.nx_to_mol(graph)
    charge = Chem.GetFormalCharge(mol)
    print("Charge of the molecule:", charge, flush=True)
    assert charge == 0

    print('Testing uncharged case', flush=True)
    graph = att.water_graph()
    mol = att.nx_to_mol(graph)
    charge = Chem.GetFormalCharge(mol)
    print("Charge of the molecule:", charge, flush=True)
    assert charge == 0

    print('Testing case where there are multiple possible charged cases and simplest one is picked', flush=True)
    graph = att.phosphine_graph()
    mol = att.nx_to_mol(graph)
    charge = Chem.GetFormalCharge(mol)
    print("Charge of the molecule:", charge, flush=True)
    assert charge == 0

    print('Testing uncharged SMILES case', flush=True)
    mol = att.smi_to_mol("[H]O[H]")
    charge = Chem.GetFormalCharge(mol)
    print("Charge of the molecule:", charge, flush=True)
    assert charge == 0

    graph = att.ph_2p_graph()
    mol = att.nx_to_mol(graph)
    # print the smiles of the molecule
    smi_out = Chem.MolToSmiles(mol, allHsExplicit=True)
    print(smi_out, flush=True)
    print('Testing charged SMILES case', flush=True)
    smi_out = '[H][P]'
    print(smi_out)
    graph_out = att.smi_to_nx(smi_out)
    mol = att.nx_to_mol(graph_out)
    charge = Chem.GetFormalCharge(mol)
    print("Charge of the molecule:", charge, flush=True)
    att.print_graph_details(graph_out)
    assert charge == 0

    smi_out = '[N]=O'
    graph = att.smi_to_nx(smi_out)
    mol = att.nx_to_mol(graph)
    charge = Chem.GetFormalCharge(mol)
    print("Charge of the molecule:", charge, flush=True)
    assert charge == 0

    att.print_graph_details(graph)
    mol = att.smi_to_mol(smi_out)
    charge = Chem.GetFormalCharge(mol)
    print("Charge of the molecule:", charge, flush=True)
    assert charge == 0


def test_implicit_hydrogens():
    str1 = '[H]-[C]'  # -> '[H]-[C]'
    str2 = '[H]-[CH](-[H])-[N]'  # -> '[H]-[C](-[H])-[N]'
    str3 = '[CH3]-[CH2]-[CH1]'  # -> '[C]-[C]'

    assert att.smi_remove_implicit_hydrogen(str1) == '[H]-[C]', "Test failed for str1"
    assert att.smi_remove_implicit_hydrogen(str2) == '[H]-[C](-[H])-[N]', "Test failed for str2"
    assert att.smi_remove_implicit_hydrogen(str3) == '[C]-[C]-[C]', "Test failed for str3"


def test_create_ionic_molecule():
    """
    Test the creation and validation of an ionic molecule.

    This function performs the following steps:
    1. Defines a SMILES string for an ionic molecule.
    2. Creates the ionic molecule from the SMILES string.
    3. Checks that the combined graph has the correct number of nodes and edges.
    4. Checks that the combined graph contains the ionic bond.
    5. Calculates the assembly index of the combined graph.
    6. Adjusts the assembly index for ionic molecules.
    7. Asserts that the adjusted assembly index is equal to 3.

    Asserts:
        - The combined graph has the correct number of nodes and edges.
        - The combined graph contains the ionic bond.
        - The adjusted assembly index is equal to 3.
    """
    print(flush=True)
    smiles = "[NH4+].[SH-]"
    # Create the ionic molecule
    combined, mols = att.create_ionic_molecule(smiles)

    # Check that the combined graph has the correct number of nodes and edges
    assert combined.number_of_nodes() == sum(mol.GetNumAtoms() for mol in mols)
    assert combined.number_of_edges() == sum(mol.GetNumBonds() for mol in mols) + len(mols) - 1

    # Check that the graph contains the ionic bond, between the relevant charged atoms (here N+ and S-)
    ionic_bond_found = False
    h_s_bond_found = False
    for u, v, data in combined.edges(data=True):
        if data.get('color') == 6:
            ionic_bond_found = True
            # Check node labels
            label_u = combined.nodes[u].get('color')
            label_v = combined.nodes[v].get('color')
            if (label_u == 'N' and label_v == 'S') or (label_u == 'S' and label_v == 'N'):
                h_s_bond_found = True
            break
    assert ionic_bond_found
    assert h_s_bond_found

    # Check that the assembly index is 3
    ai, _, _ = att.calculate_assembly_index(combined)

    # Subtract 1 from assembly index for ionic molecules
    if '.' in smiles:
        ai -= 1

    assert ai == 3


def test_get_total_free_valence():
    """
    Test the calculation of the total free valence for a molecule and its graph representation.

    This function performs the following steps:
    1. Converts a SMILES string to a molecule object.
    2. Calculates the total free valence of the molecule.
    3. Converts the SMILES string to a NetworkX graph.
    4. Calculates the total free valence of the graph.
    5. Removes all hydrogen atoms from the graph.
    6. Calculates the total free valence of the modified molecule.

    Asserts:
        - The total free valence of the molecule is 0.
        - The total free valence of the graph is 0.
        - The total free valence of the modified molecule is 2.

    Notes:
        - The function uses `att.smi_to_mol` to convert SMILES strings to RDKit molecule objects.
        - The function uses `att.smi_to_nx` to convert SMILES strings to NetworkX graphs.
        - Hydrogen atoms are removed using `att.remove_hydrogen_from_graph`.
    """
    print(flush=True)
    smi_in = "[H]C#C[H]"
    # Convert the SMILES string to a molecule object
    mol = att.smi_to_mol(smi_in)
    fv = att.get_total_free_valence(mol)
    print("Total free valence:", fv, flush=True)
    assert fv == 0

    graph = att.smi_to_nx(smi_in)
    fv = att.get_total_free_valence(graph)
    print("Total free valence from graph:", fv, flush=True)
    assert fv == 0

    # delete all the hydrogens
    mol = att.remove_hydrogen_from_graph(graph)
    fv = att.get_total_free_valence(mol)
    print("Total free valence:", fv, flush=True)
    assert fv == 2


def test_standardise_smiles():
    """
    Test the `standardise_smiles` function for standardizing SMILES strings.

    This function performs the following steps:
    1. Defines an input SMILES string.
    2. Standardizes the SMILES string with hydrogen addition enabled.
    3. Asserts that the output is a string and matches the expected standardized SMILES.
    4. Standardizes the SMILES string with hydrogen addition disabled.
    5. Asserts that the output matches the expected SMILES without added hydrogens.

    Asserts:
        - The output is a string.
        - The standardized SMILES matches the expected value with hydrogens added.
        - The standardized SMILES matches the expected value without hydrogens added.
    """
    smi_in = 'O'  # Input SMILES string
    out = att.standardise_smiles(smi_in)  # Standardize the SMILES string
    # Check that the output is a string
    assert isinstance(out, str)
    # Check that the output matches the expected standardized SMILES
    assert out == '[H]O[H]'

    out = att.standardise_smiles(smi_in, add_hydrogens=False)
    assert out == 'O'


def test_peptide_to_smiles():
    print(flush=True)
    peptide = "GGG"
    smi_out = att.peptide_to_smiles(peptide)
    print(smi_out, flush=True)
    assert isinstance(smi_out, str)
    assert smi_out == 'NCC(=O)NCC(=O)NCC(=O)O'
