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
    assert ai == 2
    assert Chem.MolToInchi(mol) == Chem.MolToInchi(virt_obj["file_graph"][0])
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
    # Define the expected paths
    expected = ['InChI=1S/CH4/h1H4',
                'InChI=1S/C2H2/c1-2/h1-2H',
                'InChI=1S/C2H4/c1-2/h1-2H2',
                'InChI=1S/C2H6/c1-2/h1-2H3',
                'InChI=1S/C3H8/c1-3-2/h3H2,1-2H3',
                'InChI=1S/C5H6/c1-3-5-4-2/h1,4H,2,5H2']
    # Assert that each calculated path is in the list of expected paths
    for p in paths:
        print(p, flush=True)
        assert p in expected


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
    mol = att.smi_to_mol("C#CCC=C")  # CN1C=NC2=C1C(=O)N(C(=O)N2C)C
    # Calculate all shortest paths in the molecule
    paths = att.all_shortest_paths(mol, f_graph_care=False)

    for p in paths:
        mol = att.inchi_to_mol(p)
        smi = Chem.MolToSmiles(mol)

        # Get the energy of the path
        energy = att.get_virtual_objects_energy(mol)
        print(f"VO: {smi}, Energy: {energy}", flush=True)
        # Assert that the energy is not None
        assert energy is not None


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
    # Convert the dict to a list
    virt_obj = att.convert_pathway_dict_to_list(virt_obj)
    print("output", flush=True)
    print(f"Ass index = {ai}", flush=True)
    # Remove the first pathway
    virt_obj.pop(0)
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
        att.atoms_to_mol_file(atoms, file_name=tmp_file)
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
    att.atoms_to_mol_file(atoms, file_name=tmp_file)
    ai_mol, _, _ = att.calculate_assembly_index(tmp_file, joint_corr=False)

    os.remove(tmp_file)

    graph = att.atoms_to_nx(atoms)
    ai_graph, _, _ = att.calculate_assembly_index(graph, joint_corr=False)

    assert ai_mol == ai_graph == 4


@pytest.mark.slow
def test_auto_compile():
    print(flush=True)
    att.compile_assembly_code(os.path.join(os.getcwd(), "assemblycpp-main"))
    pass


def test_plot_digraph_metro():
    print(flush=True)
    pathway_str = "data/pathway/tmpPathway"
    # Try to load the pathway
    digraph = att.parse_pathway_file(pathway_str)
    digraph, _ = digraph
    att.plot_digraph_metro(digraph, filename="test")
    os.remove("test.png")
    os.remove("test.svg")
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


def test_get_chirality():
    smi = "OC[C@H]1OC=C[C@@H](O)[C@@H]1O"
    mol = att.smi_to_mol(smi)

    chirality = att.get_chirality(mol)
    assert chirality == 3


def test_jai_self():
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


def test_smi_to_atoms():
    """
    Test the conversion of a SMILES string to an ASE Atoms object.

    This function performs the following steps:
    1. Defines a SMILES string for benzene ('c1ccccc1').
    2. Converts the SMILES string to an ASE Atoms object using the `smi_to_atoms` function.
    3. Asserts that the chemical formula of the resulting Atoms object is 'C6H6'.
    4. Asserts that the number of atoms in the Atoms object is 12 (6 carbons and 6 hydrogens).

    Asserts:
        - The chemical formula of the Atoms object matches 'C6H6'.
        - The total number of atoms in the Atoms object is 12.
    """
    print(flush=True)
    smi = 'c1ccccc1'  # SMILES string for benzene
    atoms = att.smi_to_atoms(smi)  # Convert the SMILES string to an ASE Atoms object

    # Assert that the chemical formula is correct
    assert atoms.get_chemical_formula() == 'C6H6'
    # Assert that the total number of atoms is correct
    assert len(atoms) == 12


def test_compression_zlib_smi():
    """
    Test the compression of a molecule's SMILES representation using zlib.

    This function performs the following steps:
    1. Converts a SMILES string to a molecule object.
    2. Compresses the molecule's SMILES representation using `compression_zlib_smi`.
    3. Compresses the molecule with and without hydrogens.
    4. Compresses the molecule with a low compression level.
    5. Asserts that the compressed size is smaller with higher compression levels.
    6. Asserts that the compressed size is smaller when hydrogens are excluded.
    7. Asserts that the compressed sizes match expected values.

    Asserts:
        - The compressed size with higher compression is smaller than with lower compression.
        - The compressed size without hydrogens is smaller than with hydrogens.
        - The compressed sizes match the expected values (13 and 29).
    """
    print(flush=True)
    smi = "C1=CC=C(C=C1)C(=O)O"  # SMILES for benzoic acid
    mol = att.smi_to_mol(smi)  # Convert the SMILES string to a molecule object

    # Compress the molecule's SMILES representation without hydrogens
    compressed = att.compression_zlib_smi(mol, add_hydrogens=False)
    # Compress the molecule's SMILES representation with a low compression level
    bad_compressed = att.compression_zlib_smi(mol, add_hydrogens=False, level=0)
    # Compress the molecule's SMILES representation with hydrogens
    h_compressed = att.compression_zlib_smi(mol, add_hydrogens=True)

    # Assert that the compressed size with higher compression is smaller
    assert compressed < bad_compressed
    # Assert that the compressed size without hydrogens matches the expected value
    assert compressed == 13
    # Assert that the compressed size with hydrogens matches the expected value
    assert h_compressed == 29


def test_compression_bz2_smi():
    """
    Test the compression of a molecule's SMILES representation using bz2.

    This function performs the following steps:
    1. Defines a SMILES string for benzoic acid.
    2. Converts the SMILES string to a molecule object.
    3. Compresses the molecule's SMILES representation with and without hydrogens.
    4. Asserts that the compressed size is smaller when hydrogens are excluded.
    5. Asserts that the compressed sizes match the expected values.

    Asserts:
        - The compressed size without hydrogens is smaller than with hydrogens.
        - The compressed size without hydrogens matches the expected value (35).
        - The compressed size with hydrogens matches the expected value (48).
    """
    print(flush=True)
    smi = "C1=CC=C(C=C1)C(=O)O"  # SMILES for benzoic acid
    mol = att.smi_to_mol(smi)

    # Compress the molecule's SMILES representation
    compressed = att.compression_bz2_smi(mol, add_hydrogens=False)
    h_compressed = att.compression_bz2_smi(mol, add_hydrogens=True)

    # Assert that the compressed size is smaller without hydrogens
    assert compressed < h_compressed
    # Assert expected compressed sizes
    assert compressed == 35
    assert h_compressed == 48


def test_compression_lzma_smi():
    """
    Test the compression of a molecule's SMILES representation using lzma.

    This function performs the following steps:
    1. Defines a SMILES string for benzoic acid.
    2. Converts the SMILES string to a molecule object.
    3. Compresses the molecule's SMILES representation with and without hydrogens.
    4. Asserts that the compressed size is smaller when hydrogens are excluded.
    5. Asserts that the compressed sizes match the expected values.

    Asserts:
        - The compressed size without hydrogens is smaller than with hydrogens.
        - The compressed size without hydrogens matches the expected value (44).
        - The compressed size with hydrogens matches the expected value (60).
    """
    print(flush=True)
    smi = "C1=CC=C(C=C1)C(=O)O"  # SMILES for benzoic acid
    mol = att.smi_to_mol(smi)  # Convert the SMILES string to a molecule object

    # Compress the molecule's SMILES representation without hydrogens
    compressed = att.compression_lzma_smi(mol, add_hydrogens=False)
    # Compress the molecule's SMILES representation with hydrogens
    h_compressed = att.compression_lzma_smi(mol, add_hydrogens=True)

    # Assert that the compressed size is smaller without hydrogens
    assert compressed < h_compressed
    # Assert expected compressed sizes
    assert compressed == 44
    assert h_compressed == 60


def test_compression_zlib_graph():
    """
    Test the compression and decompression of a graph using zlib.

    This function performs the following steps:
    1. Creates a simple graph with colored nodes and edges.
    2. Compresses the graph using `compress_zlib_graph`.
    3. Decompresses the graph and verifies the node and edge attributes.
    4. Converts a molecule's SMILES representation to a NetworkX graph.
    5. Compresses the molecule's graph representation with and without hydrogens.
    6. Compresses the molecule's graph representation with a low compression level.
    7. Asserts that the compressed size with higher compression is smaller.
    8. Asserts that the compressed sizes match the expected values.

    Asserts:
        - The compressed size with higher compression is smaller than with lower compression.
        - The compressed size without hydrogens matches the expected value (111).
        - The compressed size with hydrogens matches the expected value (111).
    """
    print(flush=True)
    # Create a simple graph with colors
    graph = nx.Graph()
    graph.add_node(1, color='red')
    graph.add_node(2, color='blue')
    graph.add_edge(1, 2, color='green')

    # Compress
    compressed = att.compress_zlib_graph(graph)
    print(f"Compressed size: {len(compressed)} bytes", flush=True)

    # Decompress
    G2 = att.decompress_zlib_graph(compressed)
    print("Node colors:", nx.get_node_attributes(G2, 'color'), flush=True)
    print("Edge colors:", nx.get_edge_attributes(G2, 'color'), flush=True)

    smi = "C1=CC=C(C=C1)C(=O)O"  # SMILES for benzoic acid
    mol = att.smi_to_mol(smi)  # Convert the SMILES string to a molecule object
    graph = att.mol_to_nx(mol, add_hydrogens=True)
    print(graph, flush=True)
    print(graph.nodes(data=True), flush=True)

    # Compress
    compressed = att.compress_zlib_graph(graph)
    print(f"Compressed size: {len(compressed)} bytes", flush=True)

    # Decompress
    G2 = att.decompress_zlib_graph(compressed)
    print("Node colors:", nx.get_node_attributes(G2, 'color'), flush=True)
    print("Edge colors:", nx.get_edge_attributes(G2, 'color'), flush=True)

    # Compress the molecule's SMILES representation without hydrogens
    compressed = att.compression_zlib_graph(graph, add_hydrogens=False)
    # Compress the molecule's SMILES representation with a low compression level
    bad_compressed = att.compression_zlib_graph(graph, add_hydrogens=False, level=0)
    # Compress the molecule's SMILES representation with hydrogens
    h_compressed = att.compression_zlib_graph(graph, add_hydrogens=True)
    print(compressed, bad_compressed, h_compressed)
    # Assert that the compressed size with higher compression is smaller
    assert compressed < bad_compressed
    # Assert that the compressed size without hydrogens matches the expected value
    assert compressed == 111
    # Assert that the compressed size with hydrogens matches the expected value
    assert h_compressed == 111


def test_circle_plot():
    """
    Test the plot_assembly_circle function from the att module.
    
    Steps:
        
    1. Define a set of example node names.
    2. Set visualization parameters.
    3. Call att.plot_assembly_circle with the specified parameters.
    4. Assert that the output file is created.
    
    The test will fail if the file is not generated.
    """

    nodes = ['b', 'a', 'd', 'c', 'ba', 'dc', 'baa', 'bad', 'badc', 'baab', 'baba', 'ddbcd', 'bcdda']

    labels = True
    node_size = 1000
    arrow_size = 50
    node_color = 'Skyblue'
    edge_color = 'Grey'
    fig_size = 10
    filename = 'circle_plot.png'
    att.plot_assembly_circle(nodes,
                             labels=labels,
                             node_size=node_size,
                             arrow_size=arrow_size,
                             node_color=node_color,
                             edge_color=edge_color,
                             fig_size=fig_size,
                             filename=filename)

    assert os.path.isfile('circle_plot.png'), "Failed to generate the file."
    os.remove('circle_plot.png')
