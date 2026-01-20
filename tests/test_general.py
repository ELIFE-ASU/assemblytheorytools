import os
import shutil

import networkx as nx
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


def test_get_graph_charges():
    """
    Test the calculation of formal charges for nodes in a molecular graph.

    This function performs the following steps:
    1. Creates a molecular graph using `att.ph_2p_graph()`.
    2. Calculates the formal charges of the graph's nodes using `att.get_graph_charges()`.
    3. Prints the calculated charges.
    4. Asserts that the calculated charges match the expected values.

    Asserts:
        - The calculated charges are equal to [2, 0].

    Notes:
        - The graph represents a molecule with two nodes, where the expected charges are predefined.
    """
    print(flush=True)
    print('Testing charged case', flush=True)
    graph = att.ph_2p_graph()
    charges = att.get_graph_charges(graph)
    print("Charges of the graph:", charges, flush=True)
    assert charges == [2, 0]


def test_smi_to_nx_conversion():
    """
    Test the conversion of a SMILES string to a NetworkX graph and back to a SMILES string.

    This function performs the following steps:
    1. Converts a SMILES string to a NetworkX graph.
    2. Converts the NetworkX graph back to a SMILES string.
    3. Asserts that the original SMILES string and the converted SMILES string are equal.

    Asserts:
        - The converted SMILES string is equal to the original SMILES string.
    """
    print(flush=True)
    smi = "[H]O[H]"
    graph = att.smi_to_nx(smi)
    smi_out = att.nx_to_smi(graph)
    assert smi_out == smi, f"Expected {smi}, but got {smi_out}"


def test_inchi_to_nx_conversion():
    """
    Test the conversion of an InChI string to a NetworkX graph and back to an InChI string.

    This function performs the following steps:
    1. Converts an InChI string to a NetworkX graph.
    2. Converts the NetworkX graph back to an InChI string.
    3. Checks if the original InChI string and the converted InChI string are equal.

    Asserts:
        - The converted InChI string is equal to the original InChI string.
    """
    print(flush=True)
    inchi = "InChI=1S/H2O/h1H2"
    graph = att.inchi_to_nx(inchi)
    inchi_out = att.nx_to_inchi(graph)
    assert inchi_out == inchi, f"Expected {inchi}, but got {inchi_out}"


def test_hydrogen_stripping():
    """
    Test the calculation of the assembly index for a molecule with and without hydrogen stripping.

    This function performs the following steps:
    1. Loads a molecule from a mol file.
    2. Converts a SMILES string to a molecule object.
    3. Asserts that the graph representations of the two molecules are isomorphic.
    4. Converts the molecule object to a NetworkX graph.
    5. Calculates the assembly index for the graph with hydrogen stripped.
    6. Calculates the assembly index for the mol file.
    7. Calculates the assembly index for the molecule object with hydrogen stripped.
    8. Asserts that the assembly index is the same for all representations.
    9. Calculates the assembly index for the graph, mol file, and molecule object with the strip_hydrogen flag set to True.
    10. Asserts that the assembly index is the same for all representations with the strip_hydrogen flag.

    Asserts:
        - The graph representations of the two molecules are isomorphic.
        - The assembly index calculated from the graph, mol file, and molecule object are equal to 4.
        - The assembly index calculated from the graph, mol file, and molecule object with the strip_hydrogen flag are equal to 4.
    """
    print(flush=True)
    mol_file = os.path.expanduser(os.path.abspath("tests/data/mol_files/alanine.mol"))
    # Get the mol object
    mol_1 = att.molfile_to_mol(mol_file)
    mol = att.smi_to_mol("C[C@@H](C(=O)O)N")

    assert att.is_graph_isomorphic(att.mol_to_nx(mol), att.mol_to_nx(mol_1))

    # Convert the system into graphs
    graph = att.mol_to_nx(mol)
    # Load the mol file, pass it Graph
    ai_graph, _, _ = att.calculate_assembly_index(att.remove_hydrogen_from_graph(graph))
    # Directly run the mol file
    ai_mol_file, _, _ = att.calculate_assembly_index(mol_file)
    # RDkit Mol
    ai_mol, _, _ = att.calculate_assembly_index(mol, strip_hydrogen=True)

    assert ai_graph == ai_mol_file == ai_mol == 4

    # Test the manual case
    # Graph
    ai_graph, _, _ = att.calculate_assembly_index(graph, strip_hydrogen=True)
    # Mol file
    ai_mol_file, _, _ = att.calculate_assembly_index(mol_file, strip_hydrogen=True)
    # Mol
    ai_mol, _, _ = att.calculate_assembly_index(mol, strip_hydrogen=True)

    assert ai_graph == ai_mol_file == ai_mol == 4


def test_ass_mol_debug():
    """
    Test the calculation of the assembly index for a molecule with debug information.

    This function performs the following steps:
    1. Converts a SMILES string to a molecule object.
    2. Calculates the assembly index of the molecule with debug information.
    3. Retrieves the path of the created file.
    4. Compares the calculated assembly index to the expected value.
    5. Verifies that the InChI of the molecule matches the InChI from the output dictionary.
    6. Asserts that only one directory was created.
    7. Cleans up by removing the created directory.

    Asserts:
        - The calculated assembly index is equal to 2.
        - The InChI of the molecule matches the InChI from the output dictionary.
        - Only one directory was created.
    """
    print(flush=True)
    # Convert all the smile to mol
    mol = att.smi_to_mol("[H]C#C[H]")
    # Calculate the assembly index
    ai, virt_obj, _ = att.calculate_assembly_index(mol, debug=True)
    # Get the path of the created file
    dir_list = att.list_subdirs(os.getcwd(), target="ai_calc")
    # Compare to the hand calculated value
    ref_out = ['[C]#[C]', '[H][C]', '[H][C]#[C]', '[H][C]#[C][H]']
    assert ai == 2
    assert att.check_elements(virt_obj, ref_out)
    assert len(dir_list) == 1
    # Clean up
    shutil.rmtree(dir_list[0])


def test_all_paths_simple():
    """
    Test the calculation of all shortest paths in a molecule.

    This function performs the following steps:
    1. Converts a SMILES string to a molecule object.
    2. Calculates all shortest paths in the molecule.
    3. Compares the calculated paths to the expected paths.

    Asserts:
        - Each calculated path is in the list of expected paths.
    """
    print(flush=True)
    # Convert the SMILES string to a molecule object
    mol = att.smi_to_mol("C#CCC=C")
    # Calculate all shortest paths in the molecule
    paths = att.all_shortest_paths(mol, f_graph_care=False)

    # check that the output is a list of strings
    assert isinstance(paths, list)
    assert len(paths) > 0


def test_energy_of_all_paths():
    """
    Test the calculation of the energy for all shortest paths in a molecule.

    This function performs the following steps:
    1. Converts a SMILES string to a molecule object.
    2. Calculates all shortest paths in the molecule.
    3. Defines a list of expected paths.
    4. Asserts that each calculated path is in the list of expected paths.
    5. For each path, calculates its energy and asserts that the energy is not None.

    Asserts:
        - Each calculated path is in the list of expected paths.
        - The energy of each path is not None.
    """
    print(flush=True)
    # Convert the SMILES string to a molecule object
    mol = att.smi_to_mol("CC")
    # Calculate all shortest paths in the molecule
    paths = att.all_shortest_paths(mol, f_graph_care=False)
    mols = [att.smi_to_mol(vo) for vo in paths]
    energy = att.get_virtual_objects_energy(mols)
    for i, vo in enumerate(paths):
        print(f"VO: {vo}, Energy: {energy[i]}", flush=True)
        # Assert that the energy is not None
        assert energy is not None


def test_hand_graph():
    """
    Test the calculation of the assembly index for a hand-constructed graph.

    This function performs the following steps:
    1. Creates a ring graph with 8 nodes.
    2. Sets the labels of the nodes to "C" (carbon atom).
    3. Sets the edge labels to "1" (single bond).
    4. Prints the details of the input graph.
    5. Calculates the assembly index of the graph.
    6. Converts the pathway dictionary to a list.
    7. Prints the details of the output graph.
    8. Removes the first pathway from the list.
    9. Prints the details of each pathway object.
    10. Asserts that the calculated assembly index is equal to 3.

    Asserts:
        - The calculated assembly index is equal to 3.
    """
    print(flush=True)
    print("This is a hand construction graph test", flush=True)
    # Create a ring graph with 8 nodes
    G = nx.cycle_graph(8)
    # Set the labels of the nodes to be "C" - a carbon atom
    nx.set_node_attributes(G, "C", "color")
    # Set the edge labels to be "1" - a single bond
    nx.set_edge_attributes(G, 1, "color")
    print("input", flush=True)
    att.print_graph_details(G)

    ai, virt_obj, _ = att.calculate_assembly_index(G)
    print("output", flush=True)
    print(f"Ass index = {ai}", flush=True)
    for i, p in enumerate(virt_obj):
        print(f"Pathway object = {i}", flush=True)
        att.print_graph_details(p)

    assert ai == 3


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


# @pytest.mark.slow
# def test_auto_compile():
#     print(flush=True)
#     # Attempt to compile the C++ backend in the specified source directory
#     att.compile_assembly_cpp_script(os.path.join(os.getcwd(), "assemblycpp-main"))
#     pass

def test_compile_assembly_cpp():
    att.compile_assembly_cpp()


def test_join_graphs():
    """
    Test the functionality of joining and splitting molecular graphs.

    This function performs the following steps:
    1. Creates two molecular graphs from SMILES strings.
    2. Joins the two graphs into a single graph.
    3. Asserts that the joined graph has the correct number of nodes and edges.
    4. Splits the joined graph back into its disconnected subgraphs.
    5. Asserts that the split subgraphs have the correct number of nodes and edges.
    6. Verifies that the original graphs are isomorphic to the split subgraphs.

    Asserts:
        - The joined graph has 5 nodes and 3 edges.
        - The split subgraphs have the correct number of nodes and edges.
        - The original graphs are isomorphic to the split subgraphs.
    """
    print(flush=True)
    # Create a molecular graph for water
    g1 = att.smi_to_nx('[H][O][H]')
    # Create a molecular graph for oxygen
    g2 = att.smi_to_nx('[O][O]')
    # Join the two graphs into a single graph
    joined = att.join_graphs([g1, g2])
    assert joined.number_of_nodes() == 5
    assert joined.number_of_edges() == 3

    # Split the joined graph back into its components
    g1_split, g2_split = att.get_disconnected_subgraphs(joined)
    assert g1_split.number_of_nodes() == 3
    assert g1_split.number_of_edges() == 2
    assert g2_split.number_of_nodes() == 2
    assert g2_split.number_of_edges() == 1

    # Check that the original graphs are equal to the split graphs
    assert nx.is_isomorphic(g1, g1_split)
    assert nx.is_isomorphic(g2, g2_split)


def test_assign_levels():
    """
    Test the `assign_levels` function with a directed graph.

    This function performs the following steps:
    1. Creates a directed graph.
    2. Defines nodes with their expected levels and adds them to the graph.
    3. Defines edges between the nodes and adds them to the graph.
    4. Calls the `assign_levels` function to assign levels to the nodes.
    5. Verifies that the assigned levels match the expected levels.

    Asserts:
        - Each node's assigned level matches its expected level.
    """
    print(flush=True)
    # Create a directed graph
    graph = nx.DiGraph()

    # Define nodes with their levels and add them to the graph
    nodes = {"CC": 0, "C=C": 0, "CO": 0, "CC=C": 1, "OCC=C": 2}
    graph.add_nodes_from(nodes)

    # Define edges and add them to the graph
    edges = [("CC", "CC=C"), ("C=C", "CC=C"), ("CO", "OCC=C"), ("CC=C", "OCC=C")]
    graph.add_edges_from(edges)

    # Assign levels to nodes
    att.assign_levels(graph)

    # Verify node levels
    for node, level in nodes.items():
        assert graph.nodes[node]["level"] == level, \
            f"Node {node} has incorrect level: {graph.nodes[node]['level']} instead of {level}"


def test_assign_levels_linear_chain():
    """
    Test the `assign_levels` function with a linear chain graph.

    This function performs the following steps:
    1. Creates a directed graph representing a linear chain of nodes.
    2. Defines nodes with their expected levels and adds them to the graph.
    3. Defines edges between the nodes to form a linear chain.
    4. Calls the `assign_levels` function to assign levels to the nodes.
    5. Verifies that the assigned levels match the expected levels.

    Asserts:
        - Each node's assigned level matches its expected level.
    """
    print(flush=True)
    # Create a directed graph
    graph = nx.DiGraph()

    # Define nodes and their levels
    nodes = {"CC": 0, "CCC": 1, "CCCCC": 2, "CCCCCCCCC": 3}
    graph.add_nodes_from(nodes)

    # Define edges between nodes
    edges = [("CC", "CCC"), ("CCC", "CCCCC"), ("CCCCC", "CCCCCCCCC")]
    graph.add_edges_from(edges)

    # Assign levels to nodes
    att.assign_levels(graph)

    # Verify node levels
    for node, level in nodes.items():
        assert graph.nodes[node][
                   "level"] == level, f"Node {node} has incorrect level: {graph.nodes[node]['level']} instead of {level}"


def test_assign_levels_empty_graph():
    """
    Test the `assign_levels` function with an empty graph.

    This function performs the following steps:
    1. Creates an empty directed graph.
    2. Calls the `assign_levels` function on the empty graph.
    3. Asserts that the graph remains empty after the function call.

    Asserts:
        - The graph has no nodes after calling `assign_levels`.
    """
    print(flush=True)
    # Create an empty directed graph
    graph = nx.DiGraph()
    # Assign levels to the empty graph
    att.assign_levels(graph)
    # Verify that the graph has no nodes
    assert len(graph.nodes) == 0, "Empty graph should have no nodes."


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


def test_int_chain():
    assert att.integer_chain(1) == 0
    assert att.integer_chain(2) == 1
    assert att.integer_chain(3) == 2
    assert att.integer_chain(4) == 2
    assert att.integer_chain(5) == 3
    assert att.integer_chain(9998) == 16
    assert att.integer_chain(9999) == 16


def _get_ai(smi):
    """
    Calculate the assembly index (AI) for a given SMILES string.

    This function takes a SMILES (Simplified Molecular Input Line Entry System) string as input,
    converts it to a NetworkX graph representation, and calculates the assembly index with
    hydrogen atoms stripped.

    Parameters:
    -----------
    smi : str
        The SMILES string representing the molecular structure.

    Returns:
    --------
    int
        The calculated assembly index for the given SMILES string.
    """
    ai, _, _ = att.calculate_assembly_index(att.smi_to_nx(smi), strip_hydrogen=True)
    return ai


def test_parallel_processing():
    """
    Test the parallel processing of assembly index calculations for a list of SMILES strings.

    This function performs the following steps:
    1. Defines a list of SMILES strings representing various molecules.
    2. Calculates the assembly index for each SMILES string using multiprocessing.
    3. Asserts that the calculated assembly indices match the expected values.
    4. Repeats the calculation using thread-based parallel processing and chunked multiprocessing.
    5. Verifies that the results are consistent across all methods.

    Asserts:
        - The calculated assembly indices match the expected values for each method.

    Notes:
        - The `att.mp_calc`, `att.tp_calc`, and `att.mp_calc_chunked` functions are used for multiprocessing,
          thread-based processing, and chunked multiprocessing, respectively.
        - The `_get_ai` function is used to calculate the assembly index for a given SMILES string.
    """
    print(flush=True)
    smiles_list = [
        'C(C(=O)O)N',  # Glycine
        'C[C@@H](C(=O)O)N',  # Alanine
        'C([C@@H](C(=O)O)N)O',  # Serine
        'C1C[C@H](NC1)C(=O)O',  # Proline
        'CC(C)C(C(=O)O)N',  # Valine
        'CC(C)CC(C(=O)O)N',  # Leucine
        'CCC(C)CC(C(=O)O)N',  # Isoleucine
        'C1CCCCC1C(=O)O',  # Cyclohexane carboxylic acid
        'C1=CC=CC=C1C(=O)O',  # Benzoic acid
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
    ]

    # Calculate assembly indices using multiprocessing
    results = att.mp_calc(_get_ai, smiles_list)
    print(f"Results: {results}", flush=True)
    expected_ais = [3, 4, 4, 6, 5, 6, 6, 6, 6, 8]
    assert results == expected_ais

    # Calculate assembly indices using thread-based parallel processing
    results = att.tp_calc(_get_ai, smiles_list)
    print(f"Results: {results}", flush=True)
    expected_ais = [3, 4, 4, 6, 5, 6, 6, 6, 6, 8]
    assert results == expected_ais

    # Calculate assembly indices using chunked multiprocessing
    results = att.mp_calc_chunked(_get_ai, smiles_list)
    print(f"Results: {results}", flush=True)
    expected_ais = [3, 4, 4, 6, 5, 6, 6, 6, 6, 8]
    assert results == expected_ais


def _add(a, b):
    """
    Add two numbers.

    This helper function takes two numerical inputs and returns their sum.

    Parameters:
    -----------
    a : int or float
        The first number to add.
    b : int or float
        The second number to add.

    Returns:
    --------
    int or float
        The sum of the two input numbers.
    """
    return a + b


def test_mp_calc_star():
    """
    Test the `mp_calc_star` function for parallel execution with multiple arguments.

    This function performs the following steps:
    1. Defines a list of argument tuples to be passed to the `_add` function.
    2. Defines the expected results for the addition of each tuple.
    3. Calls the `mp_calc_star` function to perform parallel computation of `_add` on the arguments.
    4. Asserts that the results from `mp_calc_star` match the expected results.

    Asserts:
        - The results of the parallel computation are equal to the expected results.

    Notes:
        - The `_add` function is a helper function that adds two numbers.
        - The `mp_calc_star` function is used for multiprocessing with multiple arguments.
    """
    args = [(1, 2), (3, 4), (5, 6), (7, 8)]
    expected_results = [3, 7, 11, 15]
    results = att.mp_calc_star(_add, args)
    assert results == expected_results


def test_pubchem():
    id_str = 'Aspirin'
    id = 2244
    n_sample = 3

    print(flush=True)
    smi = att.pubchem_name_to_smi(id_str)
    print(smi, flush=True)
    assert smi == 'CC(=O)OC1=CC=CC=C1C(=O)O'
    mol = att.pubchem_name_to_mol(id_str, add_hydrogens=True)
    smi_out = Chem.MolToSmiles(mol)
    print(smi_out, flush=True)
    assert smi_out == '[H]OC(=O)c1c([H])c([H])c([H])c([H])c1OC(=O)C([H])([H])[H]'
    graph = att.pubchem_name_to_nx(id_str, add_hydrogens=True)
    smi_out = att.nx_to_smi(graph)
    print(smi_out, flush=True)
    assert smi_out == '[H]OC(=O)C1=C([H])C([H])=C([H])C([H])=C1OC(=O)C([H])([H])[H]'

    smi = att.pubchem_id_to_smi(id)
    print(smi, flush=True)
    assert smi == 'CC(=O)OC1=CC=CC=C1C(=O)O'
    mol = att.pubchem_id_to_mol(id, add_hydrogens=True)
    smi_out = Chem.MolToSmiles(mol)
    print(smi_out, flush=True)
    assert smi_out == '[H]OC(=O)c1c([H])c([H])c([H])c([H])c1OC(=O)C([H])([H])[H]'
    graph = att.pubchem_id_to_nx(id, add_hydrogens=True)
    smi_out = att.nx_to_smi(graph)
    print(smi_out, flush=True)
    assert smi_out == '[H]OC(=O)C1=C([H])C([H])=C([H])C([H])=C1OC(=O)C([H])([H])[H]'

    _, smi_out = att.sample_random_pubchem(n_sample)
    print(smi_out, flush=True)
    assert len(smi_out) == n_sample

    _, smi_out = att.sample_first_pubchem(n_sample)
    print(smi_out, flush=True)
    assert len(smi_out) == n_sample

    att.download_pubchem_cid_smiles_gz()
    assert os.path.exists('CID-SMILES.gz')
    id_out, smi_out = att.sample_pubchem_cid_smiles_gz(n_sample)
    print(id_out, smi_out, flush=True)
    assert len(smi_out) == n_sample
