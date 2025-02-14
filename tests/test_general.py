import os
import shutil

import networkx as nx
import numpy as np
import pytest
from ase.io import read
from ase.visualize import view
from rdkit import Chem
from rdkit.Chem import AllChem as Chem

import assemblytheorytools as att


def list_subdirs(directory, target="ai_calc"):
    """
    List subdirectories in a given directory that start with a specific target string.

    Args:
        directory (str): The path to the directory to search within.
        target (str, optional): The prefix string that subdirectories must start with. Defaults to "ai_calc".

    Returns:
        list: A list of subdirectory names that start with the target string.
    """
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and d.startswith(target)]


def print_graph_details(graph):
    """
    Print the details of a graph, including node indices, node colors, edge connections, and edge colors.

    Args:
        graph (networkx.Graph): The graph whose details are to be printed.

    Returns:
        None
    """
    print("{", flush=True)
    for node in graph.nodes(data=True):
        node_index = node[0]
        node_color = node[1].get('color', 'No color')
        edge_connections = list(graph.edges(node_index))
        edge_colors = [graph.get_edge_data(*edge)['color'] for edge in edge_connections]
        print(f"({node_index}, {node_color}): {edge_connections}, {edge_colors}", flush=True)
    print("}", flush=True)


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


def test_ass_graph():
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
    # Get the input graph from the output dictionary
    input_graph = virt_obj["file_graph"][0]
    # Convert the input graph back to a SMILES string
    smi_out = Chem.MolToSmiles(att.nx_to_mol(input_graph))
    # Check the assembly index
    assert ai == 2
    # Check the output graph is the same as the input
    assert att.is_graph_isomorphic(graph, input_graph)
    # Check the graph conversion to and from RDKit
    assert smi_in == smi_out


def test_ass_mol_file():
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

    # Compare to the hand calculated value
    assert ai == 2
    assert Chem.MolToInchi(mol) == virt_obj["file_graph"][0]
    # Remove the temporary mol file
    os.remove(mol_file)


def test_ass_mol():
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
    # Compare to the hand calculated value
    assert ai == 2
    assert Chem.MolToInchi(mol) == Chem.MolToInchi(virt_obj["file_graph"][0])


def test_compare_ass_graph_mol_file_mol():
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
    ai_mol_file, _, _ = att.calculate_assembly_index(mol_file, timeout=1000.0)

    assert ai_mol_file == 23


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
    mol_file = os.path.expanduser(os.path.abspath("data/mol_files/alanine.mol"))
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
    ai_mol, _, _ = att.calculate_assembly_index(Chem.RemoveHs(mol))

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
    dir_list = list_subdirs(os.getcwd())
    # Compare to the hand calculated value
    assert ai == 2
    assert Chem.MolToInchi(mol) == Chem.MolToInchi(virt_obj["file_graph"][0])
    assert len(dir_list) == 1
    # Clean up
    shutil.rmtree(dir_list[0])


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
    ai, _, _ = att.calculate_assembly_index(mol, strip_hydrogen=True)

    # Assert that the calculated assembly index is equal to 4
    assert ai == 4


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


def test_joint_ass_str():
    """
    Test the calculation of the assembly index for a set of strings.

    This function performs the following steps:
    1. Define a list of strings
    2. Calculate their assembly index
    3. Assert ai = 4


    Asserts:
        - The calculated assembly index is equal to 4.
    """
    strs = ["aaaa", "bbbb", "aa"]
    ai, v_obj, path = att.calculate_string_assembly_index(strs)
    assert ai == 4


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
    # Define the expected paths
    expected = ['InChI=1S/CH4/h1H4',
                'InChI=1S/C2H2/c1-2/h1-2H',
                'InChI=1S/C2H4/c1-2/h1-2H2',
                'InChI=1S/C2H6/c1-2/h1-2H3',
                'InChI=1S/C3H8/c1-3-2/h3H2,1-2H3',
                'InChI=1S/C5H6/c1-3-5-4-2/h1,4H,2,5H2']
    # Assert that each calculated path is in the list of expected paths
    for p in paths:
        assert p in expected


def test_node_scramble():
    """
    Test the scrambling of node indices in a molecular graph.

    This function performs the following steps:
    1. Defines a SMILES string for a molecule.
    2. Converts the SMILES string to a molecule object.
    3. Converts the molecule object to a NetworkX graph.
    4. Calculates the assembly index of the graph.
    5. Retrieves the input graph from the output dictionary.
    6. Plots the molecular graph.
    7. Converts the input graph back to a SMILES string.
    8. Scrambles the node indices of the graph.
    9. Calculates the assembly index of the scrambled graph.
    10. Retrieves the input graph from the output dictionary.
    11. Plots the scrambled molecular graph.
    12. Converts the scrambled input graph back to a SMILES string.
    13. Asserts that the SMILES string of the original graph matches the SMILES string of the scrambled graph.

    Asserts:
        - The SMILES string of the original graph matches the SMILES string of the scrambled graph.
    """
    print(flush=True)
    smi_in = "[H]OC(=O)C([H])(N([H])C(=O)C([H])([H])N([H])[H])C([H])([H])C([H])(C([H])([H])[H])C([H])([H])[H]"
    # smi_in = "CCC"
    # Convert all the smile to mol
    mol = att.smi_to_mol(smi_in)
    # Convert the system into graphs
    graph = att.mol_to_nx(mol, add_hydrogens=False)
    # Calculate the assembly index
    ai, virt_obj, _ = att.calculate_assembly_index(graph)
    # Get the input graph from the output dict
    input_graph = virt_obj["file_graph"][0]
    att.plot_mol_graph(input_graph, f_labs=True, filename="graph")
    smi_out = Chem.MolToSmiles(att.nx_to_mol(input_graph))

    graph_sc = att.scramble_node_indices(graph)
    # Calculate the assembly index
    ai_sc, virt_obj_sc, _ = att.calculate_assembly_index(graph_sc)
    # Get the input graph from the output dict
    input_graph = virt_obj_sc["file_graph"][0]
    att.plot_mol_graph(input_graph, f_labs=True, filename="scrambled")
    smi_out_sc = Chem.MolToSmiles(att.nx_to_mol(input_graph))

    # assert ai == ai_sc
    assert smi_out == smi_out_sc  # Check the graph conversion to and from RDKit


def test_undir_str_ass():
    """
    Test the string assembly index calculation in mol mode for an undirected string.
    
    This function performs the following steps:
    1. Defines an input string.
    2. Calculates the assembly index of the input string.
    3. Compares the calculated assembly index to the expected value.

    Asserts:
        - The calculated assembly index is equal to the expected value.
    """
    s_inpt = "abracadabra"
    ai, _, _ = att.calculate_string_assembly_index(s_inpt, directed=False, mode='mol', debug=True)
    ai_ref = 7
    assert ai == ai_ref


def test_dir_str_ass():
    """
    Test the string assembly index calculation in mol mode for a directed string.
    
    This function performs the following steps:
    1. Defines an input string.
    2. Calculates the assembly index of the input string.
    3. Compares the calculated assembly index to the expected value.

    Asserts:
        - The calculated assembly index is equal to the expected value.
    """
    s_inpt = "abracadabra"
    ai, _, _ = att.calculate_string_assembly_index(s_inpt, directed=True, mode='mol', debug=True)
    ai_ref = 7
    assert ai == ai_ref


def test_CFG_str_ass():
    """
    Test the CFG upperbound to string assembly index for a directed string.
    
    This function performs the following steps:
    1. Defines an input string.
    2. Calculates the assembly index upper bound for the input string.
    3. Compares the upper bound to the exact value.

    Asserts:
        - The calculated upper bound is <= the exact value.
    """
    s_inpt = "abracadabra"
    ai, _, _ = att.calculate_string_assembly_index(s_inpt, directed=True, mode="cfg", debug=True)
    ai_ref = 7
    assert ai <= ai_ref


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
    print_graph_details(G)

    ai, virt_obj, _ = att.calculate_assembly_index(G)
    # Convert the dict to a list
    virt_obj = att.convert_pathway_dict_to_list(virt_obj)
    print("output", flush=True)
    print(f"Ass index = {ai}", flush=True)
    # Remove the first pathway
    virt_obj.pop(0)
    for i, p in enumerate(virt_obj):
        print(f"Pathway object = {i}", flush=True)
        print_graph_details(p)

    assert ai == 3


def test_path_vis_strings():
    # COMPLETE ME
    pass


def test_path_vis_mols():
    # COMPLETE ME
    pass


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

    # Check that the combined graph contains the ionic bond
    ionic_bond_found = False
    for u, v, data in combined.edges(data=True):
        if data.get('bond_type') == 'ionic':
            ionic_bond_found = True
            break
    assert ionic_bond_found

    # Check that the assembly index is 3
    ai, _, _ = att.calculate_assembly_index(combined)

    # Subtract 1 from assembly index for ionic molecules
    if '.' in smiles:
        ai -= 1

    assert ai == 3


def test_cif_loading():
    """
    Test the loading and validation of CIF files.

    This function performs the following steps:
    1. Lists all CIF files in the target directory.
    2. Skips specific invalid CIF files.
    3. Reads each CIF file and converts it to a molecule object.
    4. Reduces the lattice of the molecule.
    5. Visualizes the molecule.
    6. Writes the molecule to a mol file.
    7. Reads the mol file back into a molecule object.
    8. Visualizes the molecule read from the mol file.
    9. Asserts that the positions and atomic numbers of the atoms are consistent between the original and read molecules.

    Asserts:
        - The positions of the atoms in the original and read molecules are close within a tolerance.
        - The atomic numbers of the atoms in the original and read molecules are the same.
    """
    print(flush=True)
    target_dir = "data/cif_files/"
    dirs = att.file_list_all(os.path.expanduser(os.path.abspath(target_dir)))
    dirs.sort()
    print(dirs, flush=True)
    for file in dirs:
        print(file, flush=True)
        if 'Attakolite_0' in file:  # Attakolite_0 invalid spacegroup C 1 2/m 1
            continue
        if 'Wodginite_3' in file:  # Wodginite_3 invalid spacegroup C 1 2/c 1
            continue
        # input mol file
        atoms = att.read_cif_file(file)
        # tmp = ase.geometry.minkowski_reduce(atoms)
        # print(tmp)

        import ase.build
        # ase.build.niggli_reduce(atoms)
        ase.build.tools.niggli_reduce(atoms)
        ase.build.tools.reduce_lattice(atoms)

        view(atoms)
        tmp_file = file.split('.')[0] + ".mol"
        att.atoms_to_mol_file(atoms, fname=tmp_file)
        atoms2 = read(tmp_file)
        view(atoms2)
        # os.remove(tmp_file)
        # check that the atoms are the same
        assert np.allclose(atoms.get_positions(), atoms2.get_positions(), rtol=1e-04, atol=1e-04)
        assert np.allclose(atoms.get_atomic_numbers(), atoms2.get_atomic_numbers())
        exit()
        # input("Press Enter to continue...")


def test_cif_ai():
    """
    Test the calculation of the assembly index for a molecule from a CIF file.

    This function performs the following steps:
    1. Lists all CIF files in the target directory.
    2. Reads the first CIF file and converts it to a molecule object.
    3. Writes the molecule object to a mol file.
    4. Calculates the assembly index of the molecule from the mol file.
    5. Removes the temporary mol file.
    6. Converts the molecule object to a NetworkX graph.
    7. Calculates the assembly index of the graph.
    8. Asserts that the assembly index calculated from the mol file and the graph are equal to 4.

    Asserts:
        - The assembly index calculated from the mol file and the graph are equal to 4.
    """
    print(flush=True)
    target_dir = "data/cif_files/"
    dirs = att.file_list_all(os.path.expanduser(os.path.abspath(target_dir)))
    file = dirs[0]

    # input mol file
    atoms = att.read_cif_file(file)
    tmp_file = file.split('.')[0] + ".mol"
    att.atoms_to_mol_file(atoms, fname=tmp_file)
    ai_mol, _, _ = att.calculate_assembly_index(tmp_file, joint_corr=False)

    os.remove(tmp_file)

    graph = att.atoms_to_nx(atoms)
    ai_graph, _, _ = att.calculate_assembly_index(graph, joint_corr=False)

    assert ai_mol == ai_graph == 4


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
                                                  debug=True,
                                                  strip_hydrogen=True)
    assert distance == 1


@pytest.mark.slow
def test_auto_compile():
    print(flush=True)
    att.compile_assembly_code(os.path.join(os.getcwd(), "assemblycpp-main"))
    pass


def test_construction_pathway_file():
    print(flush=True)
    pathway_str = "data/pathway/tmpPathway"

    # Try to load the pathway
    digraph, inchi_list = att.parse_pathway_file(pathway_str)

    print(inchi_list)
    print(digraph.nodes(data=True))
    # # Get the inchi from the digraph
    # for i,_ in enumerate(digraph.nodes()):
    #     print(digraph.nodes[i]["inchi"])

    inchi_list_ref = ['InChI=1S/C2H6/c1-2/h1-2H3',
                      'InChI=1S/CH4/h1H4',
                      'InChI=1S/CH2O/c1-2/h1H2',
                      'InChI=1S/CH4/h1H4',
                      'InChI=1S/CH4/h1H4',
                      'InChI=1S/C2H6/c1-2/h1-2H3',
                      'InChI=1S/C2H6/c1-2/h1-2H3',
                      'InChI=1S/C2H4O/c1-2-3/h2H,1H3']

    # Check the number of nodes
    assert digraph.number_of_nodes() == 8
    # Check the number of edges
    assert digraph.number_of_edges() == 9

    # Check the inchi
    for ref, node in zip(inchi_list_ref, inchi_list):
        assert ref == node


def test_construction_pathway_smi():
    print(flush=True)

    smi = "CC=O"  # Acetaldehyde
    mol = att.smi_to_mol(smi)
    ai, virt_obj, pathway = att.calculate_assembly_index(mol, debug=False)
    pathway, inchi_list = pathway
    inchi_list_ref = ['InChI=1S/C2H6/c1-2/h1-2H3',
                      'InChI=1S/CH4/h1H4',
                      'InChI=1S/CH2O/c1-2/h1H2',
                      'InChI=1S/CH4/h1H4',
                      'InChI=1S/CH4/h1H4',
                      'InChI=1S/C2H6/c1-2/h1-2H3',
                      'InChI=1S/C2H6/c1-2/h1-2H3',
                      'InChI=1S/C2H4O/c1-2-3/h2H,1H3']
    assert ai == 5
    assert len(virt_obj) == len(set(inchi_list_ref))
    assert pathway.number_of_nodes() == 8
    assert pathway.number_of_edges() == 9

    # Check the information on each node
    for ref, node in zip(inchi_list_ref, inchi_list):
        assert ref == node


def test_construction_pathway_joint():
    print(flush=True)
    smi = "CC=O.OCC"
    inchi_acetaldehyde = "InChI=1S/C2H4O/c1-2-3/h2H,1H3"
    inchi_ethanol = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
    mol = att.smi_to_mol(smi)
    ai, virt_obj, pathway = att.calculate_assembly_index(mol, debug=False)
    pathway, inchi_list = pathway
    inchi_list_ref = ['InChI=1S/C2H6/c1-2/h1-2H3',
                      'InChI=1S/CH4/h1H4',
                      'InChI=1S/CH2O/c1-2/h1H2',
                      'InChI=1S/CH4O/c1-2/h2H,1H3',
                      'InChI=1S/H2O/h1H2',
                      'InChI=1S/C2H6/c1-2/h1-2H3',
                      'InChI=1S/C2H6/c1-2/h1-2H3',
                      'InChI=1S/C2H6/c1-2/h1-2H3',
                      'InChI=1S/C2H6/c1-2/h1-2H3',
                      'InChI=1S/CH4O/c1-2/h2H,1H3',
                      'InChI=1S/CH4O/c1-2/h2H,1H3',
                      'InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3',
                      'InChI=1S/C2H4O/c1-2-3/h2H,1H3']
    assert ai == 8
    assert inchi_list[-2] == inchi_ethanol
    assert inchi_list[-1] == inchi_acetaldehyde
    assert pathway.number_of_nodes() == 13
    assert pathway.number_of_edges() == 16

    # Check the information on each node
    for ref, node in zip(inchi_list_ref, inchi_list):
        assert ref == node
    pass


def test_plot_construction():
    print(flush=True)
    pathway_str = "data/pathway/tmpPathway"

    # Try to load the pathway
    digraph = att.parse_pathway_file(pathway_str)

    # Find all the files
    files = att.file_list_all("path_images/")
    att.plot_digraph_with_images(digraph, files)
    pass


def test_get_mol_descriptors():
    doravirine = Chem.MolFromSmiles('Cn1c(n[nH]c1=O)Cn2ccc(c(c2=O)Oc3cc(cc(c3)Cl)C#N)C(F)(F)F')

    desc = att.get_mol_descriptors(doravirine)
    assert desc['MolWt'] == 425.754
    assert desc['BertzCT'] == 1236.821427505276


def test_tanimoto_similarity():
    # https://www.rdkit.org/docs/GettingStartedInPython.html#rdkit-topological-fingerprints
    ms = [Chem.MolFromSmiles('CCOC'),
          Chem.MolFromSmiles('CCO'),
          Chem.MolFromSmiles('COC')]

    sim = att.tanimoto_similarity(ms[0], ms[1])
    assert sim == 0.6
    sim = att.tanimoto_similarity(ms[0], ms[2])
    assert sim == 0.4
    sim = att.tanimoto_similarity(ms[1], ms[2])
    assert sim == 0.25


def test_dice_morgan_similarity():
    # https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
    m1 = Chem.MolFromSmiles('Cc1ccccc1')
    m2 = Chem.MolFromSmiles('Cc1ncccc1')
    sim = att.dice_morgan_similarity(m1, m2, radius=2)
    assert sim == 0.55
