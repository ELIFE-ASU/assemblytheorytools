import bz2
import json
import lzma
import math
import os
import sys
import traceback
import zlib
from collections import defaultdict
from typing import Dict, Any, Optional

import networkx as nx
import numpy as np
import rdkit
from networkx.readwrite import json_graph
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors, RDConfig, rdMolDescriptors
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
    """
    Generates the FCFP_4 fingerprint (functional-based ECFP4) for a molecule.

    https://doi.org/10.1021/ci0503558

    This function computes the FCFP_4 fingerprint of a molecule using RDKit's
    Morgan fingerprinting method. The fingerprint is generated with a radius of 2
    and 2048 bits, focusing on functional groups by setting `useFeatures=True`.

    Parameters:
    -----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule object for which the FCFP_4 fingerprint is to be generated.

    Returns:
    --------
    int
        The number of bits set to 1 in the generated FCFP_4 fingerprint.
    """
    # Generate the FCFP_4 fingerprint with functional group focus
    fp = Chem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useFeatures=True)
    # Return the number of bits set to 1
    return fp.GetNumOnBits()


def _get_chemical_non_equivs(atom: Chem.rdchem.Atom, mol: Mol) -> float:
    """
    Calculates the chemical non-equivalence of an atom in a molecule.

    This function determines the number of unique substituent groups attached to an atom
    in a molecule. It uses the distance matrix of the molecule and the atom's substituents
    to identify unique groups based on their atomic symbols.

    Parameters:
    ----------
    atom : Chem.rdchem.Atom
        The RDKit atom object for which chemical non-equivalence is to be calculated.
    mol : Mol
        The RDKit molecule object containing the atom.

    Returns:
    -------
    float
        The number of unique substituent groups attached to the atom.
    """
    # Initialize a list to store substituents for up to 4 groups
    substituents = [[] for _ in range(4)]

    # Get the distance matrix of the molecule
    distance_matrix = Chem.GetDistanceMatrix(mol)

    # Determine the substituents of the atom using ChiralDescriptors
    atom_substituents = ChiralDescriptors.determineAtomSubstituents(atom.GetIdx(), mol, distance_matrix)[0]

    # Populate the substituents list with atomic symbols of neighboring atoms
    for item, key in enumerate(atom_substituents):
        for subatom in atom_substituents[key]:
            substituents[item].append(mol.GetAtomWithIdx(subatom).GetSymbol())

    # Calculate the number of unique substituent groups
    return float(len(set(tuple(sub) for sub in substituents if sub)))


def _get_bottcher_local_diversity(atom: Chem.rdchem.Atom) -> float:
    """
    Calculates the Bottcher local diversity of an atom.

    This function determines the diversity of an atom based on the unique
    neighboring atom types. It adds an additional value of 1.0 if the atom's
    own type is not present among its neighbors.

    Parameters:
    ----------
    atom : Chem.rdchem.Atom
        The RDKit atom object for which the Bottcher local diversity is to be calculated.

    Returns:
    -------
    float
        The Bottcher local diversity score of the atom.
    """
    # Get the set of unique symbols of neighboring atoms
    neighbors = {neighbor.GetSymbol() for neighbor in atom.GetNeighbors()}
    # Calculate diversity, adding 1.0 if the atom's symbol is not in its neighbors
    return len(neighbors) + (1.0 if atom.GetSymbol() not in neighbors else 0.0)


def _get_num_isomeric_possibilities(atom: Chem.rdchem.Atom) -> float:
    """
    Determines the number of isomeric possibilities for an atom.

    This function checks if the atom has a '_CIPCode' property, which indicates
    the presence of stereochemical information. If the property exists, the atom
    has two isomeric possibilities (e.g., R/S or E/Z). Otherwise, it has only one.

    Parameters:
    ----------
    atom : Chem.rdchem.Atom
        The RDKit atom object for which the number of isomeric possibilities is to be determined.

    Returns:
    -------
    float
        The number of isomeric possibilities: 2.0 if the '_CIPCode' property exists, otherwise 1.0.
    """
    return 2.0 if atom.HasProp('_CIPCode') else 1.0


def _get_num_valence_electrons(atom: Chem.rdchem.Atom, pt: Chem.rdchem.PeriodicTable) -> float:
    """
    Calculates the number of valence electrons for a given atom.

    This function uses the periodic table to determine the number of outer-shell
    (valence) electrons for the specified atom based on its atomic number.

    Parameters:
    ----------
    atom : Chem.rdchem.Atom
        The RDKit atom object for which the number of valence electrons is to be calculated.
    pt : Chem.rdchem.PeriodicTable
        The RDKit periodic table object used to retrieve atomic properties.

    Returns:
    -------
    float
        The number of valence electrons for the given atom.
    """
    return float(pt.GetNOuterElecs(pt.GetAtomicNumber(atom.GetSymbol())))


def _get_bottcher_bond_index(atom: Chem.rdchem.Atom) -> float:
    """
    Calculates the Bottcher bond index for a given atom.

    This function computes a ranking value based on the bond types connected to the atom.
    Each bond type is assigned a specific weight, and additional adjustments are made
    for aromatic bonds involving carbon or nitrogen atoms.

    Parameters:
    ----------
    atom : Chem.rdchem.Atom
        The RDKit atom object for which the Bottcher bond index is to be calculated.

    Returns:
    -------
    float
        The Bottcher bond index for the given atom.

    Raises:
    ------
    ValueError
        If an unsupported bond type is encountered.
    """
    b_sub_i_ranking = 0.0
    bond_weights = {
        'SINGLE': 1.0,
        'DOUBLE': 2.0,
        'TRIPLE': 3.0,
        'QUADRUPLE': 4.0,
        'QUINTUPLE': 5.0,
        'HEXTUPLE': 6.0
    }
    bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
    for bond in bonds:
        b_sub_i_ranking += bond_weights.get(bond, 0.0)
        if bond not in bond_weights and bond != 'AROMATIC':
            raise ValueError(f"Unsupported bond type {bond}")

    if 'AROMATIC' in bonds:
        if atom.GetSymbol() == 'C':
            b_sub_i_ranking += 3.0
        elif atom.GetSymbol() == 'N':
            b_sub_i_ranking += 2.0
    return b_sub_i_ranking


def bottcher(mol: Mol) -> float:
    """
    Calculates the Bottcher complexity of a molecule.

    https://github.com/boskovicgroup/bottchercomplexity
    https://doi.org/10.1021/acs.jcim.5b00723

    The Bottcher complexity is a molecular descriptor that quantifies the structural
    complexity of a molecule. It considers factors such as chemical non-equivalence,
    local diversity, isomeric possibilities, valence electrons, and bond indices
    for each atom in the molecule.

    Parameters:
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule object for which the Bottcher complexity is to be calculated.

    Returns:
    -------
    float
        The Bottcher complexity of the molecule.
    """
    complexity = 0.0
    # Assign stereochemistry to the molecule
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    pt = Chem.GetPeriodicTable()

    # Filter atoms to correct for symmetry
    atoms_corrected_for_symmetry = []
    atom_stereo_classes = set()
    for atom in mol.GetAtoms():
        cip_rank = atom.GetProp('_CIPRank')
        if cip_rank not in atom_stereo_classes:
            atoms_corrected_for_symmetry.append(atom)
            atom_stereo_classes.add(cip_rank)

    # Calculate complexity
    for atom in atoms_corrected_for_symmetry:
        d = _get_chemical_non_equivs(atom, mol)  # Chemical non-equivalence
        e = _get_bottcher_local_diversity(atom)  # Local diversity
        s = _get_num_isomeric_possibilities(atom)  # Isomeric possibilities
        v = _get_num_valence_electrons(atom, pt)  # Number of valence electrons
        b = _get_bottcher_bond_index(atom)  # Bond index
        # Update complexity using the calculated factors
        complexity += d * e * s * math.log(v * b, 2)

    return complexity


def proudfoot(mol: Mol) -> float:
    """
    Calculates the Proudfoot complexity of a molecule.

    https://doi.org/10.1016/j.bmcl.2017.03.008

    The Proudfoot complexity is a molecular descriptor that quantifies the structural
    complexity of a molecule. It is based on the distribution of molecular paths,
    atomic complexity, molecular complexity, log-sum complexity, and structural entropy.

    Parameters:
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule object for which the Proudfoot complexity is to be calculated.

    Returns:
    -------
    float
        The Proudfoot complexity of the molecule.
    """
    # Generate the Morgan fingerprint for the molecule with a radius of 2
    fingerprint = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    paths = fingerprint.GetNonzeroElements()

    # Calculate path frequencies per atom environment
    atom_paths = defaultdict(list)
    for path, count in paths.items():
        # Determine the atoms involved in the path
        atoms_in_path = path % mol.GetNumAtoms()
        atom_paths[atoms_in_path].append(count)

    # Step 1: Calculate atomic complexity (C_A)
    c_a_values = {}
    for atom, path_counts in atom_paths.items():
        total_paths = sum(path_counts)
        # Calculate the fraction of each path
        path_fractions = [count / total_paths for count in path_counts]
        # Compute atomic complexity using Shannon entropy
        ca = -sum(p * math.log2(p) for p in path_fractions) + math.log2(total_paths)
        c_a_values[atom] = ca

    # Step 2: Calculate molecular complexity (C_M)
    c_m = sum(c_a_values.values())

    # Step 3: Calculate log-sum complexity (C_M*)
    c_m_star = math.log2(sum(2 ** ca for ca in c_a_values.values()))

    # Step 4: Calculate structural entropy complexity (C_SE)
    atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    total_atoms = len(atom_types)
    # Calculate the frequency of each atom type
    type_frequencies = {atype: atom_types.count(atype) / total_atoms for atype in set(atom_types)}
    # Compute structural entropy using Shannon entropy
    c_se = -sum(freq * math.log2(freq) for freq in type_frequencies.values())

    return c_m


def sascore(mol: Mol) -> float:
    """
    Calculates the synthetic accessibility (SA) score of a molecule.

    https://doi.org/10.1021/acs.jmedchem.3c00689

    The SA score is a measure of how easily a molecule can be synthesized.
    It is calculated using the `sascorer` module, which evaluates various
    molecular properties to estimate synthetic accessibility.

    Parameters:
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule object for which the SA score is to be calculated.

    Returns:
    -------
    float
        The synthetic accessibility score of the molecule.
    """
    return sascorer.calculateScore(mol)


def mc1(mol: Mol) -> float:
    atom_number = 0.0
    divalent_node = 0.0

    for atom in mol.GetAtoms():
        atom_number += 1.0
        degree = atom.GetDegree()

        if degree == 2:
            divalent_node += 1.0
        else:
            continue

    return 1.0 - (divalent_node / atom_number)


def mc2(mol: Mol) -> int:
    atoms_in_C_O_X_double_bond = set()

    for bond in mol.GetBonds():
        # Check for a C=O double bond
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()

            if {begin_atom.GetAtomicNum(), end_atom.GetAtomicNum()} == {6, 8}:  # C and O
                # Identify which one is the carbon
                carbon = begin_atom if begin_atom.GetAtomicNum() == 6 else end_atom
                oxygen = end_atom if carbon == begin_atom else begin_atom

                # Check carbon's neighbors for N or O (excluding the double-bonded O)
                for neighbor in carbon.GetNeighbors():
                    if neighbor.GetIdx() != oxygen.GetIdx() and neighbor.GetAtomicNum() in [7, 8]:
                        atoms_in_C_O_X_double_bond.update([carbon.GetIdx(), oxygen.GetIdx()])
                        break  # Only need one N/O neighbor to satisfy the condition

    # Count non-divalent atoms not in C=O-X double bonds
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetDegree() != 2 and atom.GetIdx() not in atoms_in_C_O_X_double_bond:
            count += 1

    return count
