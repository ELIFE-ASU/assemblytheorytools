import bz2
import json
import lzma
import math
import os
import sys
import traceback
import zlib
from typing import Dict, Any, Optional

import networkx as nx
import numpy as np
import rdkit
from networkx.readwrite import json_graph
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import RDConfig
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.SpacialScore import SPS
from rdkit.Chem.rdchem import Mol

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

sys.path.append(os.path.join(RDConfig.RDContribDir, 'ChiralPairs'))
import ChiralDescriptors

from .tools_graph import remove_hydrogen_from_graph
from .tools_mol import standardize_mol


def count_unique_bonds(mol: Mol) -> int:
    """
    Counts the number of unique bonds in a molecule.

    This function iterates over all the bonds in the given RDKit molecule object,
    identifies unique bonds based on the atom types (symbols) and bond type,
    and returns the count of these unique bonds.

    A bond is considered unique if the pair of atom types (sorted alphabetically)
    and the bond type are distinct.

    Parameters:
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule object whose bonds are to be analyzed.

    Returns:
    -------
    int
        The number of unique bonds in the molecule.
    """
    unique_bonds = set()  # A set to store unique bonds
    for bond in mol.GetBonds():
        # Get the atom types (symbols) of the bonded atoms and sort them
        atom_types = tuple(sorted([bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()]))
        # Get the bond type (e.g., single, double, etc.)
        bond_type = bond.GetBondType()
        # Add the unique bond (atom types and bond type) to the set
        unique_bonds.add((atom_types, bond_type))
    # Return the count of unique bonds
    return len(unique_bonds)


def molecular_weight(mol: Mol) -> float:
    """
    Calculates the molecular weight of a molecule.

    This function uses RDKit's molecular descriptors to compute the
    molecular weight of the given molecule.

    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule object for which the molecular weight is to be calculated.

    Returns:
    --------
    float
        The molecular weight of the molecule.
    """
    return Descriptors.MolWt(mol)


def bertz_complexity(mol: Mol) -> float:
    """
    Calculates the Bertz complexity of a molecule.

    The Bertz complexity is a molecular descriptor that quantifies the
    structural complexity of a molecule. It is based on graph theory
    and considers factors such as the number of atoms, bonds, and
    branching in the molecular structure.

    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule object for which the Bertz complexity is to be calculated.

    Returns:
    --------
    float
        The Bertz complexity of the molecule.
    """
    return BertzCT(mol)


def wiener_index(mol: Mol) -> int:
    """
    Calculates the Wiener index of a molecule.

    The Wiener index is a topological descriptor that represents the sum of
    all shortest path distances between pairs of vertices in a molecular graph.
    It is used in cheminformatics to study molecular structure-activity
    relationships and predict physicochemical properties.

    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule object for which the Wiener index is to be calculated.

    Returns:
    --------
    int
        The Wiener index of the molecule.
    """
    distance_matrix = Chem.rdmolops.GetDistanceMatrix(mol)
    graph = nx.Graph()

    # Add nodes to the graph
    for i in range(len(distance_matrix)):
        graph.add_node(i)

    # Add edges with weights based on the distance matrix
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            graph.add_edge(i, j, weight=distance_matrix[i, j])

    return nx.wiener_index(graph)


def balaban_index(mol: Mol) -> float:
    """
    Calculates the Balaban index of a molecule.

    The Balaban index is a topological descriptor that provides a measure of
    molecular connectivity. It is used in cheminformatics to study molecular
    structure-activity relationships and predict physicochemical properties.

    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule object for which the Balaban index is to be calculated.

    Returns:
    --------
    float
        The Balaban index of the molecule.
    """
    return Descriptors.BalabanJ(mol)


def randic_index(mol: Mol) -> float:
    """
    Calculates the Randic index of a molecule.

    Randic index is a topological descriptor calculated by summing the
    inverse square roots of the product of the degrees of connected atom pairs.
    It is used for the study of molecular structure-activity
    relationships and the prediction of physicochemical properties.

    :param mol: An RDKit molecule object.
    :return: The Randic index.
    """
    adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
    degrees = [sum(row) for row in adj_matrix]
    randic_sum = 0
    for i, row in enumerate(adj_matrix):
        for j, val in enumerate(row):
            if val == 1:
                randic_sum += 1 / (degrees[i] * degrees[j]) ** 0.5
    return randic_sum / 2


def kirchhoff_index(mol: Mol) -> float:
    """
    Calculates the Kirchhoff index of a molecule.

    Kirchhoff index is a topological index calculated as the sum of the effective
    resistances between all pairs of vertices in the molecular graph. It is used for
    predicting physicochemical properties and molecular activities.

    :param mol: An RDKit molecule object.
    :return: The Kirchhoff index.
    """
    adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol).astype(np.float64)
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix
    pseudo_inverse_laplacian = np.linalg.pinv(laplacian_matrix)
    diagonal_elements = np.diagonal(pseudo_inverse_laplacian)

    kirchhoff_sum = 0
    for i in range(len(diagonal_elements)):
        for j in range(i + 1, len(diagonal_elements)):
            kirchhoff_sum += (diagonal_elements[i] + diagonal_elements[j] - 2 * pseudo_inverse_laplacian[i, j])

    return kirchhoff_sum


def spacial_score(mol: Mol, normalise: bool = False) -> float:
    """
    Calculates the spacial score of a molecule. https://github.com/frog2000/Spacial-Score

    Spacial score is a descriptor that quantifies the spatial arrangement
    of atoms in a molecule. It can be used to predict various molecular properties.

    :param mol: An RDKit molecule object.
    :param normalise: A boolean indicating whether to normalise the score.
    :return: The spacial score of the molecule.
    """
    return rdkit.Chem.SpacialScore.SPS(mol, normalise)


def get_mol_descriptors(mol: Mol, missingval: Optional[Any] = None) -> Dict[str, Any]:
    """
    Calculates molecular descriptors for a given molecule. Please note that there are a lot of descriptors.

    https://greglandrum.github.io/rdkit-blog/posts/2022-12-23-descriptor-tutorial.html

    This function iterates over all available molecular descriptors in RDKit,
    calculates each descriptor for the provided molecule, and stores the results
    in a dictionary. If a descriptor calculation fails, a specified missing value
    is assigned.

    :param mol: An RDKit molecule object.
    :param missingval: The value to assign if a descriptor calculation fails. Default is None.
    :return: A dictionary with descriptor names as keys and their calculated values as values.
    """
    res = {}
    for nm, fn in Descriptors._descList:
        try:
            res[nm] = fn(mol)
        except:
            traceback.print_exc()
            res[nm] = missingval
    return res


def tanimoto_similarity(mol1: Mol, mol2: Mol) -> float:
    """
    Calculates the Tanimoto similarity between two molecules.

    Tanimoto similarity is a measure of the similarity between two sets of
    molecular fingerprints. It is commonly used in cheminformatics to compare
    the structural similarity of molecules.

    :param mol1: An RDKit molecule object representing the first molecule.
    :param mol2: An RDKit molecule object representing the second molecule.
    :return: The Tanimoto similarity between the two molecules.
    """
    fpgen = Chem.GetRDKitFPGenerator()
    fp1 = fpgen.GetFingerprint(mol1)
    fp2 = fpgen.GetFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def dice_morgan_similarity(mol1: Mol, mol2: Mol, radius: int = 3) -> float:
    """
    Calculates the Dice similarity between two molecules using Morgan fingerprints.

    Dice similarity is a measure of the similarity between two sets of
    molecular fingerprints. It is commonly used in cheminformatics to compare
    the structural similarity of molecules.

    :param mol1: An RDKit molecule object representing the first molecule.
    :param mol2: An RDKit molecule object representing the second molecule.
    :param radius: The radius parameter for the Morgan fingerprint. Default is 3.
    :return: The Dice similarity between the two molecules.
    """
    fpgen = Chem.GetMorganGenerator(radius=radius)
    fp1 = fpgen.GetSparseCountFingerprint(mol1)
    fp2 = fpgen.GetSparseCountFingerprint(mol2)
    return DataStructs.DiceSimilarity(fp1, fp2)


def get_chirality(mol: Mol) -> int:
    """
    Determine the chirality of a molecule.

    This function calculates the number of chiral centres in a given RDKit molecule object.

    Parameters:
        mol (rdkit.Chem.rdchem.Mol): An RDKit molecule object.

    Returns:
        int: The number of chiral centres in the molecule.
    """
    nc = len(Chem.FindMolChiralCenters(mol,
                                       useLegacyImplementation=False,
                                       includeUnassigned=True,
                                       includeCIP=False))
    return nc


def compression_zlib_smi(mol: Mol,
                         add_hydrogens: bool = True,
                         level: int = 9,
                         check: bool = True,
                         rm_overhead: bool = True) -> int:
    """
    Compresses the SMILES representation of a molecule using zlib.

    This function standardises the molecule, converts it to a SMILES string,
    compresses the string using zlib, and optionally removes compression overhead.
    It can also verify the integrity of the compressed data.

    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule object to be compressed.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens to the molecule during standardisation (default is True).
    level : int, optional
        The compression level for zlib (default is 9, maximum compression).
    check : bool, optional
        Whether to verify that the compressed data can be decompressed and matches the original (default is True).
    rm_overhead : bool, optional
        Whether to remove the overhead of compressing an empty string (default is True).

    Returns:
    --------
    int
        The length of the compressed SMILES string, adjusted for overhead if specified.

    Raises:
    -------
    Exception
        If decompression fails during the integrity check.
    """
    # Standardise the molecule
    mol = standardize_mol(mol, add_hydrogens=add_hydrogens)

    # Remove all hydrogens from the molecule
    if not add_hydrogens:
        mol = Chem.RemoveHs(mol)

    # Convert the molecule to SMILES
    smiles = Chem.MolToSmiles(mol,
                              canonical=True,
                              kekuleSmiles=True,
                              isomericSmiles=True,
                              allHsExplicit=add_hydrogens)

    # Compress the SMILES string using zlib
    compressed = zlib.compress(smiles.encode("utf-8"), level=level)
    val = len(compressed)

    # Check if the compressed data can be decompressed and matches the original data
    if check:
        try:
            zlib.decompress(compressed).decode("utf-8")
        except Exception as e:
            print(f"Decompression failed: {e}")
            raise

    # Calculate the overhead of the compression
    if rm_overhead:
        overhead = zlib.compress("".encode("utf-8"), level=level)
        val -= len(overhead)

    return val


def compression_bz2_smi(mol: Mol,
                        add_hydrogens: bool = True,
                        check: bool = True,
                        rm_overhead: bool = True) -> int:
    """
    Compresses the SMILES representation of a molecule using bz2.

    This function standardises the molecule, converts it to a SMILES string,
    compresses the string using bz2, and optionally removes compression overhead.
    It can also verify the integrity of the compressed data.

    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule object to be compressed.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens to the molecule during standardisation (default is True).
    check : bool, optional
        Whether to verify that the compressed data can be decompressed and matches the original (default is True).
    rm_overhead : bool, optional
        Whether to remove the overhead of compressing an empty string (default is True).

    Returns:
    --------
    int
        The length of the compressed SMILES string, adjusted for overhead if specified.

    Raises:
    -------
    Exception
        If decompression fails during the integrity check.
    """
    # Standardise the molecule
    mol = standardize_mol(mol, add_hydrogens=add_hydrogens)

    # Remove all hydrogens from the molecule
    if not add_hydrogens:
        mol = Chem.RemoveHs(mol)

    # Convert the molecule to SMILES
    smiles = Chem.MolToSmiles(mol,
                              canonical=True,
                              kekuleSmiles=True,
                              isomericSmiles=True,
                              allHsExplicit=add_hydrogens)

    # Compress the SMILES string using bz2
    compressed = bz2.compress(smiles.encode("utf-8"))
    val = len(compressed)

    # Check if the compressed data can be decompressed and matches the original data
    if check:
        try:
            bz2.decompress(compressed).decode("utf-8")
        except Exception as e:
            print(f"Decompression failed: {e}")
            raise

    # Calculate the overhead of the compression
    if rm_overhead:
        overhead = bz2.compress("".encode("utf-8"))
        val -= len(overhead)

    return val


def compression_lzma_smi(mol: Mol,
                         add_hydrogens: bool = True,
                         check: bool = True,
                         rm_overhead: bool = True) -> int:
    """
    Compresses the SMILES representation of a molecule using lzma.

    This function standardises the molecule, converts it to a SMILES string,
    compresses the string using lzma, and optionally removes compression overhead.
    It can also verify the integrity of the compressed data.

    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule object to be compressed.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens to the molecule during standardisation (default is True).
    check : bool, optional
        Whether to verify that the compressed data can be decompressed and matches the original (default is True).
    rm_overhead : bool, optional
        Whether to remove the overhead of compressing an empty string (default is True).

    Returns:
    --------
    int
        The length of the compressed SMILES string, adjusted for overhead if specified.

    Raises:
    -------
    Exception
        If decompression fails during the integrity check.
    """
    # Standardize the molecule
    mol = standardize_mol(mol, add_hydrogens=add_hydrogens)

    # Remove all hydrogens from the molecule
    if not add_hydrogens:
        mol = Chem.RemoveHs(mol)

    # Convert the molecule to SMILES
    smiles = Chem.MolToSmiles(mol,
                              canonical=True,
                              kekuleSmiles=True,
                              isomericSmiles=True,
                              allHsExplicit=add_hydrogens)

    # Compress the SMILES string using lzma
    compressed = lzma.compress(smiles.encode("utf-8"))
    val = len(compressed)

    # Check if the compressed data can be decompressed and matches the original data
    if check:
        try:
            lzma.decompress(compressed).decode("utf-8")
        except Exception as e:
            print(f"Decompression failed: {e}")
            raise

    # Calculate the overhead of the compression
    if rm_overhead:
        overhead = lzma.compress("".encode("utf-8"))
        val -= len(overhead)

    return val


def compress_zlib_graph(graph: nx.Graph, level: int = 9) -> bytes:
    """
    Compress a NetworkX graph using zlib compression.

    This function converts a NetworkX graph into a JSON-serializable 
    node-link format, encodes it to a UTF-8 JSON string, and compresses 
    it using zlib.

    Parameters:
    -----------
        graph (nx.Graph): The NetworkX graph to compress.
        
        level (int, optional): Compression level (0-9). Default is 9 
            (maximum compression).

     Returns:
    ---------
        bytes: The compressed graph data as a byte string.
    """
    # Convert graph to node-link data (JSON-serialisable)
    data = json_graph.node_link_data(graph)

    # Serialise to JSON string
    json_str = json.dumps(data)

    # Compress the JSON bytes
    return zlib.compress(json_str.encode('utf-8'), level)


def decompress_zlib_graph(compressed_data: bytes) -> nx.Graph:
    """
    Decompress a zlib-compressed NetworkX graph.

    This function takes zlib-compressed bytes representing a 
    NetworkX graph in JSON node-link format, decompresses and 
    decodes them, and reconstructs the original graph.

    Parameters:
    -----------
        compressed_data (bytes): The compressed graph data as a byte string.

    Returns:
    --------
        nx.Graph: The reconstructed NetworkX graph.
    """
    # Decompress to JSON string
    json_str = zlib.decompress(compressed_data).decode('utf-8')

    # Parse JSON back to node-link format and rebuild graph
    data = json.loads(json_str)
    return json_graph.node_link_graph(data)


def compression_zlib_graph(graph: nx.Graph,
                           add_hydrogens: bool = True,
                           level: int = 9,
                           check: bool = True,
                           rm_overhead: bool = True
                           ) -> int:
    """
    Compresses a graph representation using zlib.

    This function serialises a graph into a JSON-compatible format, compresses it using zlib,
    and optionally removes compression overhead. It can also verify the integrity of the
    compressed data.

    Parameters:
    -----------
    graph : nx.Graph
        The NetworkX graph object to be compressed.
    add_hydrogens : bool, optional
        Whether to include hydrogens in the graph representation (default is True).
    level : int, optional
        The compression level for zlib (default is 9, maximum compression).
    check : bool, optional
        Whether to verify that the compressed data can be decompressed and matches the original (default is True).
    rm_overhead : bool, optional
        Whether to remove the overhead of compressing an empty graph (default is True).

    Returns:
    --------
    int
        The length of the compressed graph data, adjusted for overhead if specified.

    Raises:
    -------
    Exception
        If decompression fails during the integrity check.
    """
    # Remove hydrogens from the graph if specified
    if not add_hydrogens:
        graph = remove_hydrogen_from_graph(graph)

    # Compress the graph using zlib
    comp = compress_zlib_graph(graph, level=level)

    # Get the length of the compressed data
    val = len(comp)

    # Check if the compressed data can be decompressed and matches the original data
    if check:
        try:
            decompress_zlib_graph(comp)
        except Exception as e:
            print(f"Decompression failed: {e}")
            raise

    # Calculate the overhead of the compression
    if rm_overhead:
        overhead = compress_zlib_graph(nx.Graph())
        val -= len(overhead)

    return val


def fcfp4(mol: Mol) -> int:
    # https://doi.org/10.1021/ci0503558
    # Generate FCFP_4 fingerprint (functional-based ECFP4)
    # Use the 'useFeatures=True' argument to focus on functional groups
    fp = Chem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useFeatures=True)
    return fp.GetNumOnBits()


def _get_chemical_non_equivs(atom, themol):
    num_unique_substituents = 0
    substituents = [[], [], [], []]
    for item, key in enumerate(
            ChiralDescriptors.determineAtomSubstituents(atom.GetIdx(), themol, Chem.GetDistanceMatrix(themol))[0]):
        for subatom in \
                ChiralDescriptors.determineAtomSubstituents(atom.GetIdx(), themol, Chem.GetDistanceMatrix(themol))[0][
                    key]:
            substituents[item].append(themol.GetAtomWithIdx(subatom).GetSymbol())
            num_unique_substituents = len(set(tuple(tuple(substituent) for substituent in substituents if substituent)))
            #
            # Logic to determine e.g. whether repeats of CCCCC are cyclopentyl and pentyl or two of either
            #
    return num_unique_substituents


def _get_bottcher_local_diversity(atom):
    neighbors = []
    for neighbor in atom.GetNeighbors():
        neighbors.append(str(neighbor.GetSymbol()))
    if atom.GetSymbol() in set(neighbors):
        return len(set(neighbors))
    else:
        return len(set(neighbors)) + 1


def _get_num_isomeric_possibilities(atom):
    try:
        if (atom.GetProp('_CIPCode')):
            return 2
    except:
        return 1


def _get_num_valence_electrons(atom):
    valence = {1: ['H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'],  # Alkali Metals
               2: ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'],  # Alkali Earth Metals
               # transition metals???
               3: ['B', 'Al', 'Ga', 'In', 'Tl', 'Nh'],
               4: ['C', 'Si', 'Ge', 'Sn', 'Pb', 'Fl'],
               5: ['N', 'P', 'As', 'Sb', 'Bi', 'Mc'],  # Pnictogens
               6: ['O', 'S', 'Se', 'Te', 'Po', 'Lv'],  # Chalcogens
               7: ['F', 'Cl', 'Br', 'I', 'At', 'Ts'],  # Halogens
               8: ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Og']}  # Noble Gases
    for k in valence:
        if atom.GetSymbol() in valence[k]:
            return k
    return 0


def _get_bottcher_bond_index(atom):
    b_sub_i_ranking = 0
    bonds = []
    for bond in atom.GetBonds():
        bonds.append(str(bond.GetBondType()))
    for bond in bonds:
        if bond == 'SINGLE':
            b_sub_i_ranking += 1
        if bond == 'DOUBLE':
            b_sub_i_ranking += 2
        if bond == 'TRIPLE':
            b_sub_i_ranking += 3
    if 'AROMATIC' in bonds:
        # This list can be expanded as errors arise.
        if atom.GetSymbol() == 'C':
            b_sub_i_ranking += 3
        elif atom.GetSymbol() == 'N':
            b_sub_i_ranking += 2
    return b_sub_i_ranking


def bottcher(mol: Mol) -> float:
    # Current failures: Does not distinguish between cyclopentyl and pentyl (etc.)
    #                   and so unfairly underestimates complexity.
    # https://github.com/boskovicgroup/bottchercomplexity
    complexity = 0.0
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    atoms = mol.GetAtoms()
    atom_stereo_classes = []
    atoms_corrected_for_symmetry = []
    pt = Chem.GetPeriodicTable()
    for atom in atoms:
        if atom.GetProp('_CIPRank') in atom_stereo_classes:
            continue
        else:
            atoms_corrected_for_symmetry.append(atom)
            atom_stereo_classes.append(atom.GetProp('_CIPRank'))
    for atom in atoms_corrected_for_symmetry:
        d = _get_chemical_non_equivs(atom, mol)
        e = _get_bottcher_local_diversity(atom)
        s = _get_num_isomeric_possibilities(atom)
        v = _get_num_valence_electrons(atom)
        b = _get_bottcher_bond_index(atom)
        complexity += d * e * s * math.log(v * b, 2)

    return complexity
