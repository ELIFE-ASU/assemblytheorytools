import random

import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem as Chem

import assemblytheorytools as att


def test_origami():
    print(flush=True)
    mol = Chem.MolFromSmiles('OCC(O)CO')
    mol = att.origami(mol)

    smi_list = [Chem.MolToSmiles(m) for m in mol]

    assert 'OCC1CO1' in smi_list
    assert 'OC1COC1' in smi_list

    mol_out = att.origami(mol[0])
    assert Chem.MolToSmiles(mol_out[0]) == Chem.MolToSmiles(mol[0])


def test_get_num_atom():
    print(flush=True)
    res = att.get_num_atom('CH3', 'C')
    assert res == 1
    res = att.get_num_atom('CH3', 'H')
    assert res == 3


def test_degree_unsaturation():
    print(flush=True)
    # Ethane
    sat = att.degree_unsaturation(Chem.MolFromSmiles('CC'))
    assert sat == 0.0
    # Ethene
    sat = att.degree_unsaturation(Chem.MolFromSmiles('C=C'))
    assert sat == 1.0
    # Cyclopropane
    sat = att.degree_unsaturation(Chem.MolFromSmiles('C1CC1'))
    assert sat == 1.0
    # Benzene
    sat = att.degree_unsaturation(Chem.MolFromSmiles('c1ccccc1'))
    assert sat == 4.0


def test_assemble():
    print(flush=True)
    mol = att.assemble(Chem.MolFromSmiles('OCC(O)CO'), Chem.MolFromSmiles('C=C'), 1)
    if mol is not None:
        print(Chem.MolToSmiles(mol), flush=True)
    assert Chem.MolToSmiles(mol) == 'C=C(O)C(O)CO'


def test_reassemble_old():
    print(flush=True)
    molecules = ['[H]OC(=O)C([H])([H])N([H])[H]']
    # Convert all the SMILES strings to molecule objects
    mols = [att.smi_to_mol(smile) for smile in molecules]
    # Combine the molecule objects into a single molecule
    mol = att.combine_mols(mols)

    # Calculate the assembly index without removing hydrogens
    ai, virt_obj, _ = att.calculate_assembly_index(mol, strip_hydrogen=True)
    print(virt_obj, flush=True)

    # Convert pathway molecules to InChI strings for easy comparison/printing
    mols_out = [att.smi_to_mol(smile) for smile in virt_obj]
    print([Chem.MolToInchi(mol) for mol in mols_out], flush=True)

    # Reassemble the original molecule(s) from substructures, multiple times
    re_mols = att.reassemble_old(mols_out, n_mol_needed=4)

    # Convert reassembled molecules to InChI for output
    print([Chem.MolToInchi(mol) for mol in re_mols], flush=True)


def test_seb_molecule_construction():
    """
    Test random molecule construction between 2 fragments
    The extent of atom overlap is random.

    This function performs the following steps:
    1. Creates a minimal directed graph that includes only the 'fragments' (SMILES strings) to be joined
    2. Generates a MoleculeGeneration object, which allows usage of the combine_fragments() method
    3. Combines the two fragments. (Defining an assemble_object is required for the use of the function)

    Asserts that the resulting molecule is 'CC(N)C(=O)OC(=O)CN'
    """
    print(flush=True)
    random.seed(10)

    # Create minimal graph as a mock assembly pool
    graph = nx.DiGraph()
    graph.add_nodes_from(["NCC(O)=O", "CC(N)C(O)=O"])  # Glycine, alanine
    molecule_generation = att.MoleculeGenerationAssemblyPool(graph)

    # Combine two fragments together to form new molecule
    list_fragments = list(graph.nodes)
    fragment1 = list_fragments[0]
    fragment2 = list_fragments[1]
    product = molecule_generation.combine_fragments_layer(fragment1=fragment1, fragment2=fragment2,
                                                          assemble_object=att.Assemble())

    assert product == "CC(N)C(=O)OC(=O)CN"


def test_seb_combine_pathways():
    """
    Test the combination of two molecules' assembly pathways for an estimated JAS.
    Only one node of each molecule is included in the joined directed graph.
    Edges that are in multiple graphs are only included once.

    This function performs the following steps:
    1. Converts molecules into Molecule objects.
    2. Combines the each individual molecules' pathways together.
    3. Asserts that resulting directed graph matches Sebastian's code
    """
    print(flush=True)

    # Construct expected graph
    nodes = ['CC', 'CO', 'CCO', 'C=O', 'CC=O']
    edges = [('CC', 'CCO'), ('CC', 'CC=O'), ('CO', 'CCO'), ('C=O', 'CC=O')]
    H = nx.DiGraph()
    H.add_nodes_from(nodes)
    H.add_edges_from(edges)

    smiles = ['CCO', 'CC=O']  # Ethanol, acetaldehyade
    molecule_list = []  # To store Molecule objects
    # For each SMILES, construct a Molecule object, including its assembly pathway
    # represented as a directed graph.
    for smiles_str in smiles:
        molecule = att.Molecule(smiles=smiles_str)
        molecule.reconstruct_pathway()  #
        molecule.construct_layered_graph()
        molecule_list.append(molecule)

    # Construct MoleculeSpace
    mol_space = att.MoleculeSpace(molecules=molecule_list)
    mol_space.construct_joined_graph()  # Creates the estimated joint assembly
    G = mol_space.joined_assembly_graph  # Grab the graph - this is the assembly pool (I think)

    assert nx.is_isomorphic(G, H)


def test_seb_assembly_layer_removal():
    """
    Test the removal of molecules from given 'layer(s)' of the assembly pool (all molecules with a given assembly).
    Default for a_minus_x_assembly_pool x paramter (number of layers to remove) is 1.
    In this example, original assembly pool has an assembly depth of 1. Thus, only building blocks
    are expected to remain following the removal of 1 layer.

    Asserts that resulting assembly pool is isomorphic with expected graph.
    """
    print(flush=True)

    # Construct expected graph
    nodes = ['CC', 'CO', 'C=O']  # No edges, just building blocks
    H = nx.MultiDiGraph()
    H.add_nodes_from(nodes)

    smiles = ['CCO', 'CC=O']  # Ethanol, acetaldehyade
    molecule_list = []  # To store Molecule objects
    # For each SMILES, construct a Molecule object, including its assembly pathway
    # represented as a directed graph.
    for smiles_str in smiles:
        molecule = att.Molecule(smiles=smiles_str)
        molecule.reconstruct_pathway()  #
        molecule.construct_layered_graph()
        molecule_list.append(molecule)

    # Construct MoleculeSpace
    mol_space = att.MoleculeSpace(molecules=molecule_list)
    mol_space.construct_joined_graph()  # Creates the estimated joint assembly

    # Remove layer (all molecules with assembly index of 1)
    G, _ = mol_space.a_minus_x_assembly_pool(X=1)  # Returns the resulting graph

    assert nx.is_isomorphic(G, H)


def test_seb_constructing_n_molecules():
    """
    Test the construction of n molecules from the assembly pool and the resulting diverged
    assembly pool graph.

    Example assembly pool has assembly depth of 1. The random_construct_n_molecules() method at this time requires
    the removal of at least 1 layer. Thus, molecules are reconstructed from original building blocks in this test.

    This function performs the following:
    1. Creates the expected resulting graph (H) to compare with generated assembly pool
    2. Prepares a MoleculeSpace between 2 molecules (ethanol and acetaldehyde)
    3. Constructs 2 new molecules from the assembly pool of ethanol and acetaldehyde

    Asserts
    1. The resulting assembled molecules are 'CCC' and 'O=CO', which did not previously exist in assembly pool.
    2. The diverged assembly pool appropriately depicts this new construction.
    """
    print(flush=True)
    # This particular seed generates novel products that don't exist in the pool: propane and acetic acid
    random.seed(2)

    # Constructed expected graph
    nodes = ['CC', 'CO', 'C=O', 'CCC', 'O=CO']
    edges = [('CC', 'CCC'), ('CO', 'O=CO'), ('C=O', 'O=CO')]
    H = nx.DiGraph()
    H.add_nodes_from(nodes)
    H.add_edges_from(edges)

    smiles = ['CCO', 'CC=O']  # Ethanol, acetaldehyade
    molecule_list = []  # To store Molecule objects
    # For each SMILES, construct a Molecule object, including its assembly pathway
    # represented as a directed graph.
    for smiles_str in smiles:
        molecule = att.Molecule(smiles=smiles_str)
        molecule.reconstruct_pathway()
        molecule.construct_layered_graph()
        molecule_list.append(molecule)

    # Construct MoleculeSpace
    mol_space = att.MoleculeSpace(molecules=molecule_list)
    mol_space.construct_joined_graph()  # Creates the estimated joint assembly

    # Testing novel molecule generation, insert assembly pool? Or the whole mol space?
    molecule_generation = att.MoleculeGenerationAssemblyPool(mol_space)
    # Number of molecules to generate, number of steps in fragments, number of layers to remove
    molecule_generation.random_construct_n_molecules(2, 1, x=1)

    # Grab assembled molecules (SMILES) only from results
    assembled_molecules = [sublist[0][2] for sublist in molecule_generation.assembled_molecules.values()]

    # Grab diverged assembly
    diverged_assembly_pool = molecule_generation.diverged_assembly_graph

    assert assembled_molecules == ['CCC', 'O=CO']
    assert nx.is_isomorphic(diverged_assembly_pool, H)
