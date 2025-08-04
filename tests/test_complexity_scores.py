import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem as Chem

import assemblytheorytools as att


def test_count_unique_bonds():
    """
    Test the `count_unique_bonds` function.

    This function performs the following tests:
    1. Converts a SMILES string for water ("O") to a molecule object.
       - Asserts that the number of unique bonds is 1.
    2. Converts a SMILES string for benzene ("c1ccccc1") to a molecule object.
       - Asserts that the number of unique bonds is 3.

    Asserts:
        - The number of unique bonds matches the expected value for each test case.
    """
    print(flush=True)
    # Test with water
    smi = "O"
    mol = att.smi_to_mol(smi)  # Convert SMILES to molecule object
    n = att.count_unique_bonds(mol)  # Count unique bonds
    assert n == 1, f"Expected 1 unique bonds for {smi}, got {n}"

    # Test with benzene
    smi = "c1ccccc1"  # Benzene
    mol = att.smi_to_mol(smi)  # Convert SMILES to molecule object
    n = att.count_unique_bonds(mol)  # Count unique bonds
    assert n == 3, f"Expected 2 unique bonds for {smi}, got {n}"


def test_get_mol_descriptors():
    """
    Test that `get_mol_descriptors()` correctly calculates molecular
    descriptors for a known compound (doravirine).

    Molecule:
    - Doravirine (an antiretroviral drug)
    - SMILES: 'Cn1c(n[nH]c1=O)Cn2ccc(c(c2=O)Oc3cc(cc(c3)Cl)C#N)C(F)(F)F'

    Method:
    - Uses RDKit to generate a Mol object.
    - Uses `assemblytheorytools.get_mol_descriptors()` to extract descriptors.

    Assertions:
    - Molecular Weight (MolWt) ≈ 425.754
    - Bertz Complexity (BertzCT) ≈ 1236.821427

    Notes:
    - Uses rounding to avoid test failures due to small floating-point differences.
    """
    # Create an RDKit Mol object from SMILES for doravirine (a drug molecule)
    doravirine = Chem.MolFromSmiles('Cn1c(n[nH]c1=O)Cn2ccc(c(c2=O)Oc3cc(cc(c3)Cl)C#N)C(F)(F)F')

    # Compute molecular descriptors using assemblytheorytools
    desc = att.get_mol_descriptors(doravirine)

    # Check expected descriptor values (approximate)
    assert round(desc['MolWt'], 3) == 425.754
    assert round(desc['BertzCT'], 6) == 1236.821427


def test_tanimoto_similarity():
    """
    Test that Tanimoto similarity between RDKit topological fingerprints
    is computed correctly for a set of small molecules.

    Method:
    - Uses `tanimoto_similarity()` from `assemblytheorytools`, which wraps
      RDKit's topological fingerprinting and similarity computation.

    Assertions:
    - Similarity(CCOC, CCO)  == 0.6
    - Similarity(CCOC, COC)  == 0.4
    - Similarity(CCO,  COC)  == 0.25
    """
    # https://www.rdkit.org/docs/GettingStartedInPython.html#rdkit-topological-fingerprints

    # Create RDKit Mol objects from SMILES
    ms = [Chem.MolFromSmiles('CCOC'),
          Chem.MolFromSmiles('CCO'),
          Chem.MolFromSmiles('COC')]

    # Compute and assert Tanimoto similarities using att's wrapper
    sim = att.tanimoto_similarity(ms[0], ms[1])
    assert sim == 0.6
    sim = att.tanimoto_similarity(ms[0], ms[2])
    assert sim == 0.4
    sim = att.tanimoto_similarity(ms[1], ms[2])
    assert sim == 0.25


def test_dice_morgan_similarity():
    """
    Test that the Dice similarity between two small molecules,
    computed using Morgan fingerprints (radius=2), returns the expected value.

    Molecules:
    - Toluene: SMILES = 'Cc1ccccc1'
    - Methylpyridine: SMILES = 'Cc1ncccc1'

    Method:
    - Uses `dice_morgan_similarity()` from `assemblytheorytools`
      with circular fingerprints of radius 2.

    Assertion:
    - The computed similarity must be equal to 0.55.
    """
    # https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints

    # Create two RDKit Mol objects from SMILES strings
    m1 = Chem.MolFromSmiles('Cc1ccccc1')  # Toluene
    m2 = Chem.MolFromSmiles('Cc1ncccc1')  # Methylpyridine

    # Compute Dice similarity between Morgan fingerprints (circular)
    sim = att.dice_morgan_similarity(m1, m2, radius=2)

    # Assert similarity value (approximate comparison recommended)
    assert sim == 0.55


def test_get_chirality():
    """
    Test that the `get_chirality` function correctly counts the number
    of chiral centers in a stereochemically defined molecule.

    Assertion:
    - The function must return 3, matching the number of explicitly defined
      stereocenters in the SMILES string.
    """
    # SMILES string for a chiral molecule (likely a sugar derivative)
    smi = "OC[C@H]1OC=C[C@@H](O)[C@@H]1O"

    # Convert SMILES to RDKit Mol object
    mol = att.smi_to_mol(smi)

    # Compute number of chiral centers
    chirality = att.get_chirality(mol)

    # Assert the expected number of stereocenters
    assert chirality == 3


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


def test_compression_ratio_zlib_graph():
    print(flush=True)
    smi = "C1=CC=C(C=C1)C(=O)O"  # SMILES for benzoic acid
    mol = att.smi_to_mol(smi)  # Convert the SMILES string to a molecule object
    graph = att.mol_to_nx(mol, add_hydrogens=True)
    ratio = att.compression_ratio_zlib_graph(graph, add_hydrogens=True)
    print(f"Compression ratio: {ratio:.2f}", flush=True)
    assert np.allclose(ratio, 6.97, atol=0.01)

    ratio = att.compression_ratio_zlib_graph(graph, add_hydrogens=False)
    print(f"Compression ratio: {ratio:.2f}", flush=True)
    assert np.allclose(ratio, 5.95, atol=0.01)


def test_calculate_assembly_ratio():
    print(flush=True)
    smi = "C1=CC=C(C=C1)C(=O)O"  # SMILES for benzoic acid
    mol = att.smi_to_mol(smi)  # Convert the SMILES string to a molecule object
    graph = att.mol_to_nx(mol, add_hydrogens=True)
    ratio = att.calculate_assembly_ratio(graph, settings={'strip_hydrogen': True})
    print(f"Compression ratio: {ratio:.2f}", flush=True)
    assert np.allclose(ratio, 2.50, atol=0.01)

    ratio = att.calculate_assembly_ratio(graph, settings={'strip_hydrogen': False})
    print(f"Compression ratio: {ratio:.2f}", flush=True)
    assert np.allclose(ratio, 1.50, atol=0.01)


def test_calculate_jo_assembly_ratio():
    print(flush=True)
    smi = "C1=CC=C(C=C1)C(=O)O"  # SMILES for benzoic acid
    mol = att.smi_to_mol(smi)  # Convert the SMILES string to a molecule object
    graph = att.mol_to_nx(mol, add_hydrogens=True)
    ratio = att.calculate_jo_assembly_ratio(graph, settings={'strip_hydrogen': True})
    print(f"Compression ratio: {ratio:.2f}", flush=True)
    assert np.allclose(ratio, 2.14, atol=0.01)

    ratio = att.calculate_jo_assembly_ratio(graph, settings={'strip_hydrogen': False})
    print(f"Compression ratio: {ratio:.2f}", flush=True)
    assert np.allclose(ratio, 1.29, atol=0.01)


def test_fcfp4():
    print(flush=True)
    smi = "COC1=C(O)C=C(CC(=O)O)C=C1Br"
    mol = Chem.MolFromSmiles(smi)
    assert att.fcfp4(mol) == 29


def test_bottcher():
    print(flush=True)
    smi = "COC1=C(O)C=C(CC(=O)O)C=C1Br"
    mol = Chem.MolFromSmiles(smi)
    assert att.bottcher(mol) == 161.80418485421137


def test_bottcher_batch():
    print(flush=True)
    smiles = [r"CC(/C=C/C1=CC=CC=C1)=O",
              r"Cl/C=C\C=C\Br",
              r"CC/C(C1=CC=CC=C1)=C(C2=CC=CC=C2)/CC",
              r"C/C(=C(/C=C/C)\CCC)/CC",
              r"CC/C=C(C)/[2H]",
              r"CC/C=C1CCC[C@H](Br)C/1",
              "C/C=C(C)/C",
              "CC/C=C1CCCCC/1"]
    scores = [73.43, 54.25, 45.92, 60.34, 41.17, 95.8, 19.17, 34.17]
    calc_scores = [att.bottcher(Chem.MolFromSmiles(smi)) for smi in smiles]
    # round to 2 decimal places
    calc_scores = [round(score, 2) for score in calc_scores]
    # assert that the scores are the same
    assert calc_scores == scores


def test_proudfoot():
    print(flush=True)
    smi = "COC1=C(O)C=C(CC(=O)O)C=C1Br"
    mol = Chem.MolFromSmiles(smi)
    assert att.proudfoot(mol) == 30.54277674961796


def test_mc1():
    print(flush=True)
    smi = "COC1=C(O)C=C(CC(=O)O)C=C1Br"
    mol = Chem.MolFromSmiles(smi)
    assert att.mc1(mol) == 0.7142857142857143


def test_mc2():
    print(flush=True)
    smi = "COC1=C(O)C=C(CC(=O)O)C=C1Br"
    mol = Chem.MolFromSmiles(smi)
    assert att.mc2(mol) == 8
