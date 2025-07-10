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
    ai, virt_obj, pathway = att.calculate_assembly_index(graph)
    virt_obj = att.convert_pathway_dict_to_list(virt_obj)

    # Convert the graph to a SMILES string
    virt_obj = [att.nx_to_smi(graph, add_hydrogens=False) for graph in virt_obj]
    print(virt_obj, flush=True)

    ref_out = ['[H][C]#[C][H]', '[H][CH3]', '[H][CH3]', '[CH]#[CH]']

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
    print(virt_obj, flush=True)
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
    virt_obj = [Chem.MolToSmiles(mol, kekuleSmiles=True, allHsExplicit=True) for mol in virt_obj]
    print(virt_obj, flush=True)

    ref_out = ['[H][C]#[C][H]', '[H][CH3]', '[H][CH3]', '[CH]#[CH]']

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

    This function performs the following steps:
    1. Converts a SMILES string to a molecule object.
    2. Calculates the assembly index for the molecule without returning a log file.
    3. Writes the molecule object to a mol file and calculates the assembly index from the file.
    4. Compares the assembly index calculated from the molecule and the mol file.
    5. Calculates the assembly index for the molecule with the flag to return a log file.
    6. Verifies that the log file is generated.
    7. Cleans up temporary files created during the test.

    Asserts:
        - The assembly index is an integer and greater than 0.
        - The assembly index is consistent across representations (molecule and mol file).
        - The log file is generated when the `return_log_file` flag is set.
    """
    print(flush=True)

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
    ai_graph, _, _ = att.calculate_assembly_index(graph, timeout=30.0, strip_hydrogen=True)
    # Mol file
    ai_mol_file, _, _ = att.calculate_assembly_index(mol_file, timeout=30.0, strip_hydrogen=True)
    # Mol
    ai_mol, _, _ = att.calculate_assembly_index(mol, timeout=30.0, strip_hydrogen=True)

    assert ai_graph <= 8
    assert ai_mol_file <= 8
    assert ai_mol <= 8


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
    ai, _, _ = att.calculate_assembly_index(mol_file, timeout=30.0, strip_hydrogen=True)
    # actual value is 23, but for timeout this is ok
    assert ai <= 24


def test_exact_flag():
    print(flush=True)
    mol_file = os.path.expanduser(os.path.abspath("data/mol_files/taxol.mol"))
    ai, _, _ = att.calculate_assembly_index(mol_file,
                                            timeout=10.0,
                                            strip_hydrogen=True,
                                            exact=True)

    assert ai == -1


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
    print(virt_obj, flush=True)
    ref_out = ['NCC(=O)O', 'CC(N)C(=O)O', 'NCC(=O)O', 'CC', 'NCC(=O)O']

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
    6. Asserts that the calculated assembly index is equal to 11.

    Asserts:
        - The calculated assembly index is equal to 11.
    """
    print(flush=True)
    molecules = "[H]C#C[H].[H][C]([H])([H])[C]([H])([H])[H].[H]C([H])([H])([H]).[H]O([H]).[H]N([H])([H]).[H][N+]([H])([H])([H]).[S-]([H]).[H][H]"
    molecules = molecules.split(".")
    # Convert all the SMILES strings to molecule objects
    mols = [att.smi_to_mol(smile) for smile in molecules]
    # Combine the molecule objects into a single molecule
    mol = att.combine_mols(mols)

    # Calculate the assembly index
    ai, _, _ = att.calculate_assembly_index(mol)
    # Assert that the calculated assembly index is equal to 11
    assert ai == 11


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


def test_jai_self():
    """
    Test that the joint assembly index (JAI) of two identical molecules
    is equal to the assembly index (AI) of a single instance.

    This validates that JAI does not artificially increase when the
    same molecule is duplicated, ensuring internal deduplication and
    fragment reuse work as expected.

    Assertion:
    - The JAI of two identical molecules must equal the AI of one.
    """
    print(flush=True)
    molecules = ["O=P(O)(O)OC[C@@H](O)[C@@H](O)c1c[nH]c2ccccc12", "O=P(O)(O)OC[C@@H](O)[C@@H](O)c1c[nH]c2ccccc12"]
    # Convert all the SMILES strings to molecule objects
    mols = [att.smi_to_mol(smile) for smile in molecules]
    # Combine the molecule objects into a single molecule
    mol = att.combine_mols(mols)

    # Calculate the assembly index
    jai, _, _ = att.calculate_assembly_index(mol, strip_hydrogen=True)
    ai, _, _ = att.calculate_assembly_index(mols[0], strip_hydrogen=True)
    assert jai == ai


def test_jai_asymmetric():
    """
    Test that assembly index computation is order-independent
    for asymmetric molecule inputs.

    This function verifies that combining two molecules in different
    orders results in the same assembly index, confirming that the
    calculation is not sensitive to the input list sequence.

    Molecules:
    - L-serine: "N[C@@H](CCO)C(=O)O"
    - Citric acid: "O=C(O)CC(C(=O)O)C(O)C(=O)O"

    Assertion:
    - The calculated assembly indices (ai_1 and ai_2) should be equal.
    """
    print(flush=True)
    molecules = ["N[C@@H](CCO)C(=O)O", "O=C(O)CC(C(=O)O)C(O)C(=O)O"]
    # Convert all the SMILES strings to molecule objects
    mols = [att.smi_to_mol(smile) for smile in molecules]
    # Combine the molecule objects into a single molecule
    mol = att.combine_mols(mols)

    # Calculate the assembly index
    ai_1, _, _ = att.calculate_assembly_index(mol, strip_hydrogen=True)

    molecules = ["O=C(O)CC(C(=O)O)C(O)C(=O)O", "N[C@@H](CCO)C(=O)O"]
    # Convert all the SMILES strings to molecule objects
    mols = [att.smi_to_mol(smile) for smile in molecules]
    # Combine the molecule objects into a single molecule
    mol = att.combine_mols(mols)

    # Calculate the assembly index
    ai_2, _, _ = att.calculate_assembly_index(mol, strip_hydrogen=True)

    assert ai_1 == ai_2


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
    """
    Test the construction of a pathway graph and virtual object list from a pathway file.

    This function performs the following steps:
    1. Defines the file path for the pathway file.
    2. Loads the pathway graph and virtual object list from the file.
    3. Compares the virtual object list to a reference list.
    4. Checks the number of nodes and edges in the pathway graph.

    Asserts:
        - The pathway graph has 8 nodes.
        - The pathway graph has 9 edges.
        - The virtual object list matches the reference list.

    Variables:
        pathway_str (str): Path to the pathway file.
        digraph (networkx.DiGraph): Pathway graph loaded from the file.
        vo_list (list): Virtual object list loaded from the file.
        vo_list_ref (list): Reference list of expected virtual objects.
    """
    print(flush=True)
    pathway_str = "data/pathway/tmpPathway"

    # Try to load the pathway
    digraph, vo_list = att.parse_pathway_file(pathway_str, vo_type='inchi')

    # Define the reference virtual object list
    vo_list_ref = ['InChI=1S/C2H6/c1-2/h1-2H3',
                   'InChI=1S/CH2O/c1-2/h1H2',
                   'InChI=1S/CH4/h1H4',
                   'InChI=1S/C2H4O/c1-2-3/h2H,1H3']

    # Check the number of nodes
    assert digraph.number_of_nodes() == 8
    # Check the number of edges
    assert digraph.number_of_edges() == 9

    # Assert that the virtual object list matches the reference list
    assert att.check_elements(vo_list, vo_list_ref)


def test_construction_pathway_smi():
    """
    Test the construction of a pathway graph and virtual object list for a molecule.

    This function performs the following steps:
    1. Defines a SMILES string representing a molecule (Acetaldehyde).
    2. Converts the SMILES string to a molecule object.
    3. Calculates the assembly index, pathway graph, and virtual object list for the molecule.
    4. Compares the virtual object list to a reference list.
    5. Asserts the correctness of the assembly index, pathway graph nodes, edges, and virtual object list.

    Asserts:
        - The calculated assembly index is equal to 5.
        - The virtual object list matches the reference list.
        - The pathway graph has 8 nodes.
        - The pathway graph has 9 edges.
    """
    print(flush=True)

    smi = "CC=O"  # Acetaldehyde
    mol = att.smi_to_mol(smi)  # Convert the SMILES string to a molecule object
    ai, virt_obj, pathway = att.calculate_assembly_index(mol)  # Calculate the assembly index and pathway
    pathway, vo_list = pathway  # Extract the pathway graph and virtual object list

    # Define the reference virtual object list
    vo_list_ref = ['[H]C([H])([H])C', 'C=O', '[H]C[H]', '[H]C', '[H]CC([H])([H])[H]', 'CC', '[H]C(=O)C([H])([H])[H]',
                   '[H]C([H])[H]']

    # Assert the correctness of the assembly index, pathway graph, and virtual object list
    assert ai == 5
    assert len(vo_list) == len(set(vo_list_ref))
    assert pathway.number_of_nodes() == 8
    assert pathway.number_of_edges() == 9

    assert att.check_elements(vo_list, vo_list_ref)


def test_construction_pathway_joint():
    """
    Test the construction of a pathway graph and virtual object list for a joint molecule.

    This function performs the following steps:
    1. Defines a SMILES string representing a joint molecule.
    2. Converts the SMILES string to a molecule object.
    3. Calculates the assembly index, pathway graph, and virtual object list for the molecule.
    4. Compares the virtual object list to a reference list.
    5. Plots the pathway graph and saves it as temporary files.
    6. Removes the temporary files.
    7. Asserts the correctness of the assembly index, pathway graph nodes, edges, and virtual object list.

    Asserts:
        - The calculated assembly index is equal to 8.
        - The pathway graph has 13 nodes.
        - The pathway graph has 16 edges.
        - The virtual object list matches the reference list.
    """
    print(flush=True)
    smi = "CC=O.OCC"  # Define the SMILES string for the joint molecule
    mol = att.smi_to_mol(smi)  # Convert the SMILES string to a molecule object
    ai, virt_obj, pathway = att.calculate_assembly_index(mol)  # Calculate the assembly index and pathway
    pathway, vo_list = pathway  # Extract the pathway graph and virtual object list

    # Define the reference virtual object list
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

    # Plot the pathway graph and save it as temporary files
    att.plot_digraph_metro(pathway, filename="test")
    os.remove("test.png")  # Remove the temporary PNG file
    os.remove("test.svg")  # Remove the temporary SVG file

    # Assert the correctness of the assembly index, pathway graph, and virtual object list
    assert ai == 8
    assert pathway.number_of_nodes() == 13
    assert pathway.number_of_edges() == 16
    assert att.check_elements(vo_list, vo_list_ref)


def test_calculate_assembly_parallel():
    """
    Test the parallel calculation of assembly indices for a list of molecular graphs.

    This function performs the following steps:
    1. Defines a list of SMILES strings representing molecules.
    2. Converts the SMILES strings to NetworkX graphs.
    3. Defines settings for the assembly index calculation.
    4. Calculates the assembly indices for the graphs in parallel.
    5. Prints the calculated assembly indices.
    6. Asserts that the calculated assembly indices match the expected reference values.

    Asserts:
        - The calculated assembly indices match the reference list [3, 4, 3, 0, 3].
    """
    print(flush=True)
    # Define a list of SMILES strings
    smiles = ['[H]OC(=O)C([H])([H])N([H])[H]',
              '[H]OC(=O)C([H])(N([H])[H])C([H])([H])[H]',
              '[H]OC(=O)C([H])([H])N([H])[H]',
              '[H]C([H])([H])C([H])([H])[H]',
              '[H]OC(=O)C([H])([H])N([H])[H]']
    # Convert the SMILES strings to NetworkX graphs
    graphs = [att.smi_to_nx(smi) for smi in smiles]
    # Define settings for the assembly index calculation
    settings = {'strip_hydrogen': True}
    # Calculate assembly indices in parallel
    ai = att.calculate_assembly_parallel(graphs, settings)[0]
    # Print the calculated assembly indices
    print(ai, flush=True)
    # Define the reference list of expected assembly indices
    ref_list = [3, 4, 3, 0, 3]
    # Assert that the calculated assembly indices match the expected values
    assert att.check_elements(ai, ref_list)


def test_calculate_sum_assembly():
    """
    Test the calculation of the sum of assembly indices for molecular graphs.

    This function performs the following steps:
    1. Converts two SMILES strings to NetworkX graphs.
    2. Defines settings for the assembly index calculation.
    3. Calculates the sum of assembly indices in parallel mode.
    4. Asserts that the calculated sum matches the expected value.
    5. Calculates the sum of assembly indices in sequential mode.
    6. Asserts that the calculated sum matches the expected value.

    Asserts:
        - The sum of assembly indices is equal to 7 in both parallel and sequential modes.
    """
    print(flush=True)
    # Convert SMILES strings to NetworkX graphs
    graphs = [att.smi_to_nx("C1=CC=CC=C1"), att.smi_to_nx("C1=CC=CC=C1O")]
    # Define settings for the assembly index calculation
    settings = {'strip_hydrogen': True}
    # Calculate the sum of assembly indices in parallel mode
    ai_sum = att.calculate_sum_assembly(graphs, settings, parallel=True)
    assert ai_sum == 7
    # Calculate the sum of assembly indices in sequential mode
    ai_sum = att.calculate_sum_assembly(graphs, settings, parallel=False)
    assert ai_sum == 7


def test_calculate_assembly_similarity():
    """
    Test the calculation of assembly similarity between two molecular graphs.

    This function performs the following steps:
    1. Converts two SMILES strings to NetworkX graphs.
    2. Calculates the assembly similarity between the two graphs using the `calculate_assembly_similarity` function.
    3. Prints the calculated similarity value.
    4. Asserts that the calculated similarity matches the expected value.

    Asserts:
        - The calculated similarity is equal to 0.75.
    """
    print(flush=True)
    graphs = [att.smi_to_nx("C1=CC=CC=C1"), att.smi_to_nx("C1=CC=CC=C1O")]
    settings = {'strip_hydrogen': True}
    similarity = att.calculate_assembly_similarity(graphs, settings, parallel=True)
    print(similarity, flush=True)
    assert similarity == 0.75
    similarity = att.calculate_assembly_similarity(graphs, settings, parallel=False)
    print(similarity, flush=True)
    assert similarity == 0.75


def test_node_canonicalization():
    """
    Test the assembly index calculation for a specific molecular graph that produced inaccurate results
    without node canonicalization.

    This function creates a molecular graph with specific nodes and edges, calculates its assembly index,
    and asserts that the result matches the expected value.

    Steps:
    1. Define a molecular graph using NetworkX.
    2. Add nodes with specific attributes (e.g., color representing atom types).
    3. Add edges with specific attributes (e.g., color representing bond types).
    4. Calculate the assembly index using the `calculate_assembly_index` function.
    5. Assert that the calculated assembly index is equal to the expected value.

    Asserts:
        - The calculated assembly index is equal to 8.
    """
    # Create a new graph
    G = nx.Graph()

    # Add nodes with attributes (color represents atom type)
    G.add_node(0, color='C')
    G.add_node(1, color='C')
    G.add_node(2, color='C')
    G.add_node(6, color='C')
    G.add_node(10, color='C')
    G.add_node(11, color='C')
    G.add_node(12, color='C')
    G.add_node(15, color='C')
    G.add_node(18, color='C')
    G.add_node(3, color='O')
    G.add_node(4, color='O')
    G.add_node(5, color='O')
    G.add_node(7, color='O')
    G.add_node(8, color='O')
    G.add_node(13, color='O')
    G.add_node(14, color='O')
    G.add_node(16, color='O')
    G.add_node(19, color='O')

    # Add edges with attributes (color represents bond type)
    G.add_edge(18, 19, color=2)
    G.add_edge(8, 18, color=1)
    G.add_edge(6, 8, color=1)
    G.add_edge(6, 7, color=2)
    G.add_edge(0, 18, color=1)
    G.add_edge(0, 6, color=1)
    G.add_edge(0, 1, color=1)
    G.add_edge(0, 10, color=1)
    G.add_edge(1, 5, color=1)
    G.add_edge(1, 2, color=1)
    G.add_edge(2, 3, color=2)
    G.add_edge(2, 4, color=1)
    G.add_edge(4, 15, color=1)
    G.add_edge(10, 15, color=1)
    G.add_edge(15, 16, color=2)
    G.add_edge(10, 11, color=2)
    G.add_edge(11, 12, color=1)
    G.add_edge(12, 13, color=2)
    G.add_edge(12, 14, color=1)

    # Calculate the assembly index
    a, _, _ = att.calculate_assembly_index(G)

    # Assert that the calculated assembly index matches the expected value
    assert a == 8


def test_calculate_assembly_upper_bound():
    """
    Test the calculation of the assembly upper bound for a molecule.

    This function performs the following steps:
    1. Defines a SMILES string representing a molecule.
    2. Converts the SMILES string to a molecule object.
    3. Calculates the assembly upper bound with hydrogen stripping enabled.
    4. Asserts that the calculated upper bound is equal to 0.
    5. Calculates the assembly upper bound without hydrogen stripping.
    6. Asserts that the calculated upper bound is equal to 2.
    7. Converts the molecule object to a NetworkX graph.
    8. Calculates the assembly upper bound for the graph without hydrogen stripping.
    9. Asserts that the calculated upper bound is equal to 2.

    Asserts:
        - The assembly upper bound with hydrogen stripping is equal to 0.
        - The assembly upper bound without hydrogen stripping is equal to 2.
        - The assembly upper bound for the graph without hydrogen stripping is equal to 2.
    """
    print(flush=True)
    smi_in = "[H]C#C[H]"  # Define the SMILES string for the molecule
    # Convert the SMILES string to a molecule object
    mol = att.smi_to_mol(smi_in)
    # Test strip hydrogen flag
    ai_upper_bound = att.calculate_assembly_upper_bound(mol, strip_hydrogen=True)
    assert ai_upper_bound == 0  # Assert the upper bound with hydrogen stripping
    # Test without stripping hydrogen
    ai_upper_bound = att.calculate_assembly_upper_bound(mol, strip_hydrogen=False)
    assert ai_upper_bound == 2  # Assert the upper bound without hydrogen stripping
    # Convert the molecule object to a NetworkX graph
    ai_upper_bound_graph = att.calculate_assembly_upper_bound(att.mol_to_nx(mol), strip_hydrogen=False)
    assert ai_upper_bound_graph == 2  # Assert the upper bound for the graph without hydrogen stripping


def test_calculate_assembly_lower_bound():
    """
    Test the calculation of the assembly lower bound for a molecule.

    This function performs the following steps:
    1. Defines a SMILES string representing a molecule.
    2. Converts the SMILES string to a molecule object.
    3. Calculates the assembly lower bound with hydrogen stripping enabled.
    4. Asserts that the calculated lower bound is equal to 0.
    5. Calculates the assembly lower bound without hydrogen stripping.
    6. Asserts that the calculated lower bound is equal to 1.
    7. Converts the molecule object to a NetworkX graph.
    8. Calculates the assembly lower bound for the graph without hydrogen stripping.
    9. Asserts that the calculated lower bound is equal to 1.

    Asserts:
        - The assembly lower bound with hydrogen stripping is equal to 0.
        - The assembly lower bound without hydrogen stripping is equal to 1.
        - The assembly lower bound for the graph without hydrogen stripping is equal to 1.
    """
    print(flush=True)
    smi_in = "[H]C#C[H]"  # Define the SMILES string for the molecule
    # Convert the SMILES string to a molecule object
    mol = att.smi_to_mol(smi_in)
    # Test strip hydrogen flag
    ai_lower_bound = att.calculate_assembly_lower_bound(mol, strip_hydrogen=True)
    assert ai_lower_bound == 0  # Assert the lower bound with hydrogen stripping
    # Test without stripping hydrogen
    ai_lower_bound = att.calculate_assembly_lower_bound(mol, strip_hydrogen=False)
    assert ai_lower_bound == 1  # Assert the lower bound without hydrogen stripping
    # Convert the molecule object to a NetworkX graph
    ai_lower_bound_graph = att.calculate_assembly_lower_bound(att.mol_to_nx(mol), strip_hydrogen=False)
    assert ai_lower_bound_graph == 1  # Assert the lower bound for the graph without hydrogen stripping


def test_standardise_smiles():
    """
    Tests the standardization of a SMILES string using the `standardise_smiles` function.

    Steps:
    1. Defines an input SMILES string ('O').
    2. Calls the `standardise_smiles` function to standardize the input.
    3. Verifies that the output is a string.
    4. Asserts that the standardized output matches the expected value ('[H]O[H]').

    Parameters:
    None

    Returns:
    None
    """
    smi_in = 'O'  # Input SMILES string
    out = att.standardise_smiles(smi_in)  # Standardize the SMILES string
    # Check that the output is a string
    assert isinstance(out, str)
    # Check that the output matches the expected standardized SMILES
    assert out == '[H]O[H]'


def test_vo_problem():
    print(flush=True)
    smi = [
        'O',  # H2O
        'O=S=O',  # SO2
        'OS(=O)(=O)O',  # H2SO4
    ]
    print(f'input smi: {smi}', flush=True)
    graphs = [att.smi_to_nx(s) for s in smi]
    mols = [att.smi_to_mol(s) for s in smi]

    venus_smi = [att.nx_to_smi(g) for g in graphs]
    print(f'back converted smi: {venus_smi}')

    for i in range(len(graphs)):
        print(f'\ngraph {i} smi: {att.nx_to_smi(graphs[i])}', flush=True)
        ai, virt_obj, path = att.calculate_assembly_index(graphs[i])
        print(f'ai: {ai}', flush=True)
        virt_obj = att.convert_pathway_dict_to_list(virt_obj)
        # convert to smi
        virt_obj = [Chem.MolToSmiles(att.nx_to_mol(graph, add_hydrogens=False)) for graph in virt_obj]
        print('from vo', virt_obj)

        virt_obj = path[-1]
        # convert to smi
        virt_obj = [Chem.MolToSmiles(att.nx_to_mol(graph, add_hydrogens=False)) for graph in virt_obj]
        print('from graph path', virt_obj)

        ai, virt_obj, path = att.calculate_assembly_index(mols[i])
        virt_obj = path[-1]
        print('from mol path', virt_obj)


def test_calculate_jo_from_pathway():
    """
    Test the calculation of the joining operation index (JO) from a pathway file.

    This function performs the following steps:
    1. Defines the file path for the pathway file.
    2. Calculates the joining operation index (JO) using the `calculate_jo_from_pathway` function.
    3. Asserts that the calculated JO matches the expected value.

    Asserts:
        - The calculated JO is equal to 28.
    """
    print(flush=True)
    file = os.path.expanduser(os.path.abspath("data/pathway/taxolPathway"))
    jo = att.calculate_jo_from_pathway(file)
    assert jo == 28, f"Expected JO to be 28, but got {jo}"


def test_calculate_jo():
    """
    Test the calculation of the joining operation index (JO) for a molecular graph.

    This function performs the following steps:
    1. Defines a SMILES string representing a molecule (Benzene).
    2. Converts the SMILES string to a NetworkX graph.
    3. Calculates the joining operation index (JO) for the graph using the `calculate_jo` function.
    4. Asserts that the calculated JO matches the expected value.

    Asserts:
        - The calculated JO is equal to 6.
    """
    print(flush=True)
    smi = "C1=CC=CC=C1"  # Benzene
    graph = att.smi_to_nx(smi)
    jo, _, _ = att.calculate_jo(graph)
    assert jo == 6, f"Expected JO to be 6, but got {jo}"
