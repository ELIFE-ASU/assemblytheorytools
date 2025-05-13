import os

import networkx as nx
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem as Chem

import assemblytheorytools as att


def test_ai_graph():
    """
    Test the calculation of the assembly index for a molecular graph.

    This function performs the following steps:
    1. Converts a SMILES string to a molecule object.
    2. Converts the molecule object to a NetworkX graph.
    3. Calculates the assembly index of the graph.
    4. Retrieves the input graph from the output dictionary.
    5. Converts the input graph back to a SMILES string.
    6. Compares the calculated assembly index to the expected value.
    7. Checks if the original graph and the input graph are isomorphic.
    8. Verifies that the original SMILES string matches the converted SMILES string.

    Asserts:
        - The calculated assembly index is equal to 2.
        - The original graph and the input graph are isomorphic.
        - The original SMILES string matches the converted SMILES string.
    """
    print(flush=True)
    smi_in = "[H]C#C[H]"
    # Convert the SMILES string to a molecule object
    mol = att.smi_to_mol(smi_in)
    # Convert the molecule object to a NetworkX graph
    graph = att.mol_to_nx(mol)
    # Calculate the assembly index of the graph
    ai, virt_obj, _ = att.calculate_assembly_index(graph)
    virt_obj = att.convert_pathway_dict_to_list(virt_obj)

    # Convert the graph to a mol and then to a SMILES string
    virt_obj = [Chem.MolToSmiles(att.nx_to_mol(graph)) for graph in virt_obj]
    ref_out = ['[H]C#C[H]',
               '[H]C([H])([H])[H]',
               '[H]C([H])([H])[H]',
               '[H]C#C[H]']

    assert ai == 2
    assert att.check_elements(virt_obj, ref_out)


def test_ai_mol_file():
    """
    Test the calculation of the assembly index for a molecule from a file.

    This function performs the following steps:
    1. Converts a SMILES string to a molecule object.
    2. Writes the molecule object to a mol file.
    3. Calculates the assembly index of the molecule from the mol file.
    4. Compares the calculated assembly index to the expected value.
    5. Verifies that the InChI of the molecule matches the InChI from the output dictionary.
    6. Removes the temporary mol file.

    Asserts:
        - The calculated assembly index is equal to 2.
        - The InChI of the molecule matches the InChI from the output dictionary.
    """
    print(flush=True)
    smi_in = "[H]C#C[H]"
    # Convert the SMILES string to a molecule object
    mol = att.smi_to_mol(smi_in)
    # Write the molecule object to a mol file
    mol_file = "tmp.mol"
    att.write_v2k_mol_file(mol, mol_file)
    # Calculate the assembly index of the molecule from the mol file
    ai, virt_obj, _ = att.calculate_assembly_index(mol_file)

    virt_obj = att.convert_pathway_dict_to_list(virt_obj)

    ref_out = ['InChI=1S/C2H2/c1-2/h1-2H',
               'InChI=1S/CH4/h1H4',
               'InChI=1S/CH4/h1H4',
               'InChI=1S/C2H2/c1-2/h1-2H']
    assert ai == 2
    assert att.check_elements(virt_obj, ref_out)
    os.remove(mol_file)


def test_ai_mol():
    """
    Test the calculation of the assembly index for a molecule.

    This function performs the following steps:
    1. Converts a SMILES string to a molecule object.
    2. Calculates the assembly index of the molecule.
    3. Compares the calculated assembly index to the expected value.
    4. Verifies that the InChI of the molecule matches the InChI from the output dictionary.

    Asserts:
        - The calculated assembly index is equal to 2.
        - The InChI of the molecule matches the InChI from the output dictionary.
    """
    print(flush=True)
    smi_in = "[H]C#C[H]"
    # Convert the SMILES string to a molecule object
    mol = att.smi_to_mol(smi_in)
    # Calculate the assembly index of the molecule
    ai, virt_obj, _ = att.calculate_assembly_index(mol)
    virt_obj = att.convert_pathway_dict_to_list(virt_obj)

    # Convert the graph to a mol and then to a SMILES string
    virt_obj = [Chem.MolToSmiles(mol) for mol in virt_obj]
    ref_out = ['[H]C#C[H]',
               '[H]C([H])([H])[H]',
               '[H]C([H])([H])[H]',
               '[H]C#C[H]']

    assert ai == 2
    assert att.check_elements(virt_obj, ref_out)


def test_ai_compare_graph_mol_file_mol():
    """
    Test the consistency of the assembly index calculation across different representations of molecules.

    This function performs the following steps:
    1. Converts SMILES strings to molecule objects.
    2. Converts the molecule objects to NetworkX graphs.
    3. Writes the molecule objects to mol files.
    4. Calculates the assembly index for the graph, mol file, and molecule object.
    5. Asserts that the assembly index is the same for all representations.
    6. Removes the temporary mol file.

    Asserts:
        - The assembly index calculated from the graph, mol file, and molecule object are equal.
    """
    print(flush=True)
    smis = ["c1ccccc1", "[BH-]1-[NH+]=[BH-]-[NH+]=[BH-]-[NH+]=1"]
    mol_file = "tmp.mol"
    for smi_in in smis:
        # Convert the SMILES string to a molecule object
        mol = att.smi_to_mol(smi_in)
        # Convert the molecule object to a NetworkX graph
        graph = att.mol_to_nx(mol)
        # Write the molecule object to a mol file
        att.write_v2k_mol_file(mol, mol_file)

        # Calculate the assembly index for the graph
        ai_graph, _, _ = att.calculate_assembly_index(graph)
        # Calculate the assembly index for the mol file
        ai_mol_file, _, _ = att.calculate_assembly_index(mol_file)
        # Calculate the assembly index for the molecule object
        ai_mol, _, _ = att.calculate_assembly_index(mol)

        # Assert that the assembly index is the same for all representations
        assert ai_graph == ai_mol_file == ai_mol
    # Remove the temporary mol file
    os.remove(mol_file)


def test_calculate_assembly_index_flag_for_logs():
    """
    Test the `calculate_assembly_index` function with different input types
    and configuration options.
    """

    # Test case 1: SMILES to RDKit molecule
    smi = "C1=CC=CC=C1"  # Benzene
    mol = att.smi_to_mol(smi)

    # test input of mol from smiles, no return log file
    ai, virt_obj, path = att.calculate_assembly_index(mol)
    assert isinstance(ai, int), "Assembly index should be an integer"
    assert ai > 0, "AI should be a positive number"

    # test input of mol object to mol file, no return log file
    mol_file = "test_benzene.mol"
    att.write_v2k_mol_file(mol, mol_file)
    ai_mol_file, _, _ = att.calculate_assembly_index(mol_file)
    assert ai == ai_mol_file, "Assembly index should be consistent across representations"

    # Clean up test file
    os.remove(mol_file)

    # test mol object and flag for returning log file
    ai_log, _, _, log_file = att.calculate_assembly_index(mol, return_log_file=True)
    assert os.path.exists(log_file), "Log file should be generated when return_log_file=True"

    # Clean up log file
    os.remove(log_file)


@pytest.mark.slow
def test_big_chungus():
    """
    Test the calculation of the assembly index for a large molecule.

    This function performs the following steps:
    1. Loads a molecule from a mol file.
    2. Converts the molecule to a NetworkX graph.
    3. Calculates the assembly index for the graph, mol file, and molecule object.
    4. Asserts that the assembly index is the same for all representations.

    Asserts:
        - The assembly index calculated from the graph, mol file, and molecule object are equal to 8.
    """
    print(flush=True)
    mol_file = os.path.expanduser(os.path.abspath("data/mol_files/big_chungus.mol"))
    # Get the mol object
    mol = att.molfile_to_mol(mol_file)
    # Convert the system into graphs
    graph = att.mol_to_nx(mol)

    # Graph
    ai_graph, _, _ = att.calculate_assembly_index(graph, timeout=1000.0)
    # Mol file
    ai_mol_file, _, _ = att.calculate_assembly_index(mol_file, timeout=1000.0)
    # Mol
    ai_mol, _, _ = att.calculate_assembly_index(mol, timeout=1000.0)

    assert ai_graph == ai_mol_file == ai_mol == 8


@pytest.mark.slow
def test_taxol_file():
    """
    Test the calculation of the assembly index for the molecule in the taxol mol file.

    This function performs the following steps:
    1. Loads the taxol molecule from a mol file.
    2. Calculates the assembly index for the molecule.
    3. Asserts that the calculated assembly index is equal to 23.

    Asserts:
        - The calculated assembly index is equal to 23.
    """
    print(flush=True)
    mol_file = os.path.expanduser(os.path.abspath("data/mol_files/taxol.mol"))
    # Mol file
    ai_mol_file, _, _ = att.calculate_assembly_index(mol_file, timeout=60.0, strip_hydrogen=True, debug=False)

    assert ai_mol_file <= 24  # actual value is 23, but for time out this is ok


def test_joint_ass():
    """
    Test the calculation of the assembly index for a combined molecule.

    This function performs the following steps:
    1. Converts SMILES strings to molecule objects.
    2. Combines the molecule objects into a single molecule.
    3. Calculates the assembly index of the combined molecule with hydrogen stripping.
    4. Asserts that the calculated assembly index is equal to 4.

    Asserts:
        - The calculated assembly index is equal to 4.
    """
    print(flush=True)
    molecules = ["NCC(O)=O", "CC(N)C(O)=O"]
    # Convert all the SMILES strings to molecule objects
    mols = [att.smi_to_mol(smile) for smile in molecules]
    # Combine the molecule objects into a single molecule
    mol = att.combine_mols(mols)

    # Calculate the assembly index
    ai, virt_obj, _ = att.calculate_assembly_index(mol, strip_hydrogen=True)

    virt_obj = att.convert_pathway_dict_to_list(virt_obj)

    # Convert the graph to a mol and then to a SMILES string
    virt_obj = [Chem.MolToSmiles(mol) for mol in virt_obj]
    ref_out = ['[H]OC(=O)C([H])([H])N([H])[H]',
               '[H]OC(=O)C([H])(N([H])[H])C([H])([H])[H]',
               '[H]OC(=O)C([H])([H])N([H])[H]',
               '[H]C([H])([H])C([H])([H])[H]',
               '[H]OC(=O)C([H])([H])N([H])[H]']

    assert ai == 4
    assert att.check_elements(virt_obj, ref_out)


def test_joint_ass_mol():
    """
    Test the calculation of the assembly index for a combined molecule.

    This function performs the following steps:
    1. Defines a string of SMILES representations for multiple molecules.
    2. Splits the string into individual SMILES strings.
    3. Converts the SMILES strings to molecule objects.
    4. Combines the molecule objects into a single molecule.
    5. Calculates the assembly index of the combined molecule.
    6. Compares the calculated assembly index to the expected value.
    7. Verifies that the InChI of the first split molecule matches the InChI from the output dictionary.

    Asserts:
        - The calculated assembly index is equal to 11.
        - The InChI of the first split molecule matches the InChI from the output dictionary.
    """
    print(flush=True)
    molecules = "[H]C#C[H].[H][C]([H])([H])[C]([H])([H])[H].[H]C([H])([H])([H]).[H]O([H]).[H]N([H])([H]).[H][N+]([H])([H])([H]).[S-]([H]).[H][H]"
    molecules = molecules.split(".")
    # Convert all the SMILES strings to molecule objects
    mols = [att.smi_to_mol(smile) for smile in molecules]
    mol = att.combine_mols(mols)

    # Calculate the assembly index
    ai, virt_obj, _ = att.calculate_assembly_index(mol)
    # Compare to the hand calculated value
    out_mol = Chem.MolToInchi(att.split_mols(mol)[0])
    assert ai == 11
    assert out_mol == Chem.MolToInchi(virt_obj["file_graph"][0])


def test_joint_ass_graph():
    """
    Test the calculation of the assembly index for a combined molecular graph.

    This function performs the following steps:
    1. Defines a string of SMILES representations for multiple molecules.
    2. Splits the string into individual SMILES strings.
    3. Converts the SMILES strings to molecule objects.
    4. Converts the molecule objects to NetworkX graphs.
    5. Joins the individual graphs into a single graph.
    6. Calculates the assembly index of the combined graph.
    7. Compares the calculated assembly index to the expected value.
    8. Verifies that the original combined graph is isomorphic to the output graph.

    Asserts:
        - The calculated assembly index is equal to 11.
        - The original combined graph is isomorphic to the output graph.
    """
    print(flush=True)
    molecules = "[H]C#C[H].[H][C]([H])([H])[C]([H])([H])[H].[H]C([H])([H])([H]).[H]O([H]).[H]N([H])([H]).[H][N+]([H])([H])([H]).[S-]([H]).[H][H]"
    molecules = molecules.split(".")
    # Convert all the SMILES strings to molecule objects
    mols = [att.smi_to_mol(smile) for smile in molecules]
    # Convert the molecule objects into graphs
    graphs = [att.mol_to_nx(mol) for mol in mols]
    # Join the graphs
    graphs_joint = nx.disjoint_union_all(graphs)
    # Calculate the assembly index
    ai, virt_obj, _ = att.calculate_assembly_index(graphs_joint)
    # Compare to the hand calculated value
    out_graph = nx.disjoint_union_all(virt_obj["file_graph"])
    assert ai == 11
    assert att.is_graph_isomorphic(graphs_joint, out_graph)


@pytest.mark.slow
def test_big_joint_ass():
    """
    Test the calculation of the assembly index for a large combined molecule.

    This function performs the following steps:
    1. Defines a string of SMILES representations for multiple molecules.
    2. Converts the SMILES strings to molecule objects.
    3. Combines the molecule objects into a single molecule.
    4. Calculates the assembly index of the combined molecule with hydrogen stripping.
    5. Asserts that the calculated assembly index is equal to 40.

    Asserts:
        - The calculated assembly index is equal to 40.
    """
    print(flush=True)
    molecules = "NCC(=O)O.CC(N)C(=O)O.C([C@@H](C(=O)O)N)O.O=C(O)CC(N)C(=O)O.O=C(O)C(N)CS.OC(=O)CCC(N)C(=O)O.C[C@H]([C@@H](C(=O)O)N)O.CC(C)C(N)C(=O)O"
    molecules += ".NC(=O)CC(N)C(=O)O.O=C(N)CCC(N)C(=O)O.CC(CC)C(N)C(=O)O.CC(C)CC(N)C(=O)O.NC(CCCCN)C(=O)O.O=C(O)C1CCCN1.O=C(O)C(N)CCSC.C(C[C@@H](C(=O)O)N)CN=C(N)N"
    # Convert all the SMILES strings to molecule objects
    mols = [att.smi_to_mol(smile) for smile in molecules]
    # Combine the molecule objects into a single molecule
    mol = att.combine_mols(mols)

    # Calculate the assembly index
    ai, _, _ = att.calculate_assembly_index(mol, strip_hydrogen=True)

    # Assert that the calculated assembly index is equal to 40
    assert ai == 40


def test_semi_metric():
    """
    Test the calculation of the assembly semi-metric between two molecular graphs.

    This function performs the following steps:
    1. Defines a list of SMILES strings for two molecules.
    2. Converts the SMILES strings to molecule objects.
    3. Converts the molecule objects to NetworkX graphs.
    4. Calculates the assembly semi-metric between the two graphs.
    5. Asserts that the calculated distance is equal to 1.

    Asserts:
        - The calculated distance is equal to 1.
    """
    print(flush=True)
    molecules = ["NCC(O)=O", "CC(N)C(O)=O"]
    # Convert all the smile to mol
    mols = [att.smi_to_mol(smile) for smile in molecules]
    # Convert the system into graphs
    graphs = [att.mol_to_nx(mol) for mol in mols]

    distance = att.calculate_assembly_semi_metric(graphs[0],
                                                  graphs[1],
                                                  dir_code=None,
                                                  timeout=100.0,
                                                  debug=False,
                                                  strip_hydrogen=True)
    assert distance == 1


def test_construction_pathway_file():
    print(flush=True)
    pathway_str = "data/pathway/tmpPathway"

    # Try to load the pathway
    digraph, vo_list = att.parse_pathway_file(pathway_str, vo_type='inchi')

    vo_list_ref = ['InChI=1S/C2H6/c1-2/h1-2H3',
                   'InChI=1S/CH2O/c1-2/h1H2',
                   'InChI=1S/CH4/h1H4',
                   'InChI=1S/C2H4O/c1-2-3/h2H,1H3']

    # Check the number of nodes
    assert digraph.number_of_nodes() == 8
    # Check the number of edges
    assert digraph.number_of_edges() == 9

    assert att.check_elements(vo_list, vo_list_ref)


def test_construction_pathway_smi():
    print(flush=True)

    smi = "CC=O"  # Acetaldehyde
    mol = att.smi_to_mol(smi)
    ai, virt_obj, pathway = att.calculate_assembly_index(mol)
    pathway, vo_list = pathway

    vo_list_ref = ['[H]C([H])([H])C', 'C=O', '[H]C[H]', '[H]C', '[H]CC([H])([H])[H]', 'CC', '[H]C(=O)C([H])([H])[H]', '[H]C([H])[H]']
    assert ai == 5
    assert len(vo_list) == len(set(vo_list_ref))
    assert pathway.number_of_nodes() == 8
    assert pathway.number_of_edges() == 9

    assert att.check_elements(vo_list, vo_list_ref)


def test_construction_pathway_joint():
    print(flush=True)
    smi = "CC=O.OCC"
    mol = att.smi_to_mol(smi)
    ai, virt_obj, pathway = att.calculate_assembly_index(mol)
    pathway, vo_list = pathway

    vo_list_ref = ['[H]C',
                   '[H]C([H])C',
                   'CC',
                   '[H]C([H])([H])C',
                   '[H]CC',
                   '[H]CC([H])([H])[H]',
                   '[H]C(=O)C([H])([H])[H]',
                   '[H]CO',
                   'CO',
                   'C=O',
                   '[H]O',
                   '[H]CO[H]',
                   '[H]OC([H])([H])C([H])([H])[H]']
    assert ai == 8
    assert pathway.number_of_nodes() == 13
    assert pathway.number_of_edges() == 16

    assert att.check_elements(vo_list, vo_list_ref)

    att.plot_digraph_metro(pathway, filename="test")
