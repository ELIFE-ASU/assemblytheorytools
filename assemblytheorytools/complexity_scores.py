import bz2
import lzma
import traceback
import zlib
from typing import Dict, Any, Optional

import networkx as nx
import numpy as np
import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.rdchem import Mol

from .tools_mol import standardize_mol


def molecular_weight(mol: Mol) -> float:
    """
    Calculates the molecular weight of a molecule.

    Molecular weight is the sum of the atomic weights of all the atoms in a molecule.
    It is a simple measure of the molecule's size and is often used to estimate
    properties like solubility and boiling point.

    :param mol: An RDKit molecule object.
    :return: The molecular weight.
    """
    return Descriptors.MolWt(mol)


def bertz_complexity(mol: Mol) -> float:
    """
    Calculates the Bertz complexity of a molecule.

    Bertz complexity is a topological index that combines both bond and atom
    information. It is a measure of the structural complexity of the molecule
    and is used to compare the complexity of different molecules.

    :param mol: An RDKit molecule object.
    :return: The Bertz complexity.
    """
    return BertzCT(mol)


def wiener_index(mol: Mol) -> float:
    """
    Calculates the Wiener index of a molecule.

    Wiener index is a topological descriptor calculated as the sum of the shortest
    path lengths between all pairs of atoms in a molecule. It is a measure of the
    molecule's branching and can be used to estimate various physico-chemical properties.

    :param mol: An RDKit molecule object.
    :return: The Wiener index.
    """
    distance_matrix = Chem.rdmolops.GetDistanceMatrix(mol)
    G = nx.Graph()

    # Add nodes
    for i in range(len(distance_matrix)):
        G.add_node(i)

    # Add edges
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            G.add_edge(i, j, weight=distance_matrix[i, j])

    return nx.wiener_index(G)


def balaban_index(mol: Mol) -> float:
    """
    Calculates the Balaban index of a molecule.

    Balaban index is a topological descriptor that quantifies the degree of
    branching in a molecule. It is useful for predicting physico-chemical
    properties and biological activities of molecules.

    :param mol: An RDKit molecule object.
    :return: The Balaban index.
    """
    return Descriptors.BalabanJ(mol)


def randic_index(mol: Mol) -> float:
    """
    Calculates the Randic index of a molecule.

    Randic index is a topological descriptor calculated by summing the
    inverse square roots of the product of the degrees of connected atom pairs.
    It is used for the study of molecular structure-activity
    relationships and the prediction of physico-chemical properties.

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
    predicting physico-chemical properties and molecular activities.

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


def spacial_score(mol: Mol, normalize: bool = False) -> float:
    """
    Calculates the spacial score of a molecule. https://github.com/frog2000/Spacial-Score

    Spacial score is a descriptor that quantifies the spatial arrangement
    of atoms in a molecule. It can be used to predict various molecular properties.

    :param mol: An RDKit molecule object.
    :param normalize: A boolean indicating whether to normalize the score.
    :return: The spacial score of the molecule.
    """
    return rdkit.Chem.SpacialScore.SPS(mol, normalize)


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

    This function calculates the number of chiral centers in a given RDKit molecule object.

    Parameters:
        mol (rdkit.Chem.rdchem.Mol): An RDKit molecule object.

    Returns:
        int: The number of chiral centers in the molecule.
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

    This function standardizes the molecule, converts it to a SMILES string,
    compresses the string using zlib, and optionally removes compression overhead.
    It can also verify the integrity of the compressed data.

    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule object to be compressed.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens to the molecule during standardization (default is True).
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

    This function standardizes the molecule, converts it to a SMILES string,
    compresses the string using bz2, and optionally removes compression overhead.
    It can also verify the integrity of the compressed data.

    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule object to be compressed.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens to the molecule during standardization (default is True).
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

    This function standardizes the molecule, converts it to a SMILES string,
    compresses the string using lzma, and optionally removes compression overhead.
    It can also verify the integrity of the compressed data.

    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule object to be compressed.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens to the molecule during standardization (default is True).
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
