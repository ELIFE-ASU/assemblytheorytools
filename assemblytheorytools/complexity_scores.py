import bz2
import json
import lzma
import traceback
import zlib
from collections import OrderedDict
from math import log
from typing import Dict, Any, Optional

import networkx as nx
import numpy as np
import rdkit
from networkx.readwrite import json_graph
from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.rdchem import Mol

from .tools_graph import remove_hydrogen_from_graph
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
    molecule's branching and can be used to estimate various physicochemical properties.

    :param mol: An RDKit molecule object.
    :return: The Wiener index.
    """
    distance_matrix = Chem.rdmolops.GetDistanceMatrix(mol)
    graph = nx.Graph()

    # Add nodes
    for i in range(len(distance_matrix)):
        graph.add_node(i)

    # Add edges
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            graph.add_edge(i, j, weight=distance_matrix[i, j])

    return nx.wiener_index(graph)


def balaban_index(mol: Mol) -> float:
    """
    Calculates the Balaban index of a molecule.

    Balaban index is a topological descriptor that quantifies the degree of
    branching in a molecule. It is useful for predicting physicochemical
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
    # Convert graph to node-link data (JSON-serialisable)
    data = json_graph.node_link_data(graph)

    # Serialise to JSON string
    json_str = json.dumps(data)

    # Compress the JSON bytes
    return zlib.compress(json_str.encode('utf-8'), level)


def decompress_zlib_graph(compressed_data: bytes) -> nx.Graph:
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


class BottcherScore:
    """
    A class to calculate the Böttcher score of molecules.
    https://github.com/forlilab/bottchscore

    This class provides methods to calculate the Böttcher score, which is a measure of the complexity
    of a molecule based on various atomic and molecular properties.

    Attributes:
        converter (OBConversion): An Open Babel conversion object for SMILES format.
        automorp_memory_maxsize (int): Maximum size for automorphism memory.
        _mesomery_patterns (dict): Dictionary of SMARTS patterns for mesomeric structures.
    """

    def __init__(self, automorp_memory_maxsize=3000000):
        """
        Initialize the BottcherScore object.

        Args:
            automorp_memory_maxsize (int, optional): Maximum size for automorphism memory. Defaults to 3000000.
        """
        self.converter = ob.OBConversion()
        self.converter.SetOutFormat('smi')
        self.automorp_memory_maxsize = automorp_memory_maxsize
        self._mesomery_patterns = {
            '[$([#8;X1])]=*-[$([#8;X1])]': [[[0, 2]], 1.5],
            '[$([#7;X2](=*))](=*)(-*=*)': [[[2, 1]], 1.5],
        }

    def score(self, mol, disable_mesomeric=False) -> float:
        """
        Calculate the Bottcher score for the given molecule.

        This method calculates the Bottcher score for the molecule by initializing the molecule,
        calculating various terms, identifying cis-trans double bonds, and calculating the total complexity score.

        Args:
            mol (OBMol): The molecule object to score.
            disable_mesomeric (bool, optional): Flag to disable mesomeric calculations. Defaults to False.

        Returns:
            float: The calculated Bottcher score for the molecule. Returns 0 if the molecule has fewer than 2 atoms.
        """
        if mol.NumAtoms() < 2:
            return 0
        self._initialize_mol(mol, disable_mesomeric)
        self._calculate_terms()
        self._find_cistrans_bond_atoms()
        self._calculate_score()
        return self._intrinsic_complexity

    def _initialize_mol(self, mol, disable_mesomeric):
        """
        Initialize the molecule and its properties.

        This method initializes the molecule by setting the molecule object, generating its SMILES representation,
        building automorphisms, and initializing indices and mesomery equivalence dictionaries. If mesomeric
        calculations are not disabled, it also calculates mesomeric contributions.

        Args:
            mol (OBMol): The molecule object to initialize.
            disable_mesomeric (bool): Flag to disable mesomeric calculations.

        Returns:
            None
        """
        self.mol = mol
        self._smiles = self.converter.WriteString(self.mol).split()[0]
        self._build_automorphism()
        self._indices = OrderedDict()
        self._mesomery_equivalence = {}
        if not disable_mesomeric:
            self._calc_mesomery()

    def _calc_mesomery(self):
        """
        Calculate mesomeric contributions and update automorphisms.

        This method uses SMARTS patterns to identify mesomeric structures in the molecule.
        It updates the mesomery equivalence dictionary and automorphs dictionary with the
        contributions and automorphisms found.

        Returns:
            None
        """
        matcher = ob.OBSmartsPattern()
        for patt, (idx_pairs, contribution) in self._mesomery_patterns.items():
            matcher.Init(patt)
            if matcher.Match(self.mol):
                for f in matcher.GetUMapList():
                    for pair in idx_pairs:
                        for idx in pair:
                            self._mesomery_equivalence[f[idx]] = contribution
                        p0, p1 = f[pair[0]], f[pair[1]]
                        self.automorphs.setdefault(p0, set()).add(p1)
                        self.automorphs.setdefault(p1, set()).add(p0)

    def _calculate_terms(self):
        """
        Calculate various terms for each atom in the molecule.

        This method iterates over all atoms in the molecule, skipping hydrogen atoms,
        and calculates several properties for each atom, including its degree (di),
        valence indicator (Vi), bond order sum (bi), number of unique element types (ei),
        and stereochemistry indicator (si). These properties are stored in the _indices dictionary.

        Returns:
            None
        """
        for idx in range(1, self.mol.NumAtoms() + 1):
            atom = self.mol.GetAtom(idx)
            if self._is_hydrogen(idx):
                continue
            self._indices[idx] = {}
            self._calc_di(idx, atom)
            self._calc_vi(idx, atom)
            self._calc_bi_ei_si(idx, atom)

    def _calculate_score(self):
        """
        Calculate the total complexity score for the molecule.

        This method calculates the total complexity score for the molecule by summing the complexities
        of individual atoms and adjusting for cis-trans double bonds and equivalent groups.

        Returns:
            None
        """
        total_complexity = 0
        for idx in self._indices:
            if not any(idx in double_bond for double_bond in self._cistrans_double_bond_atoms):
                complexity = self._calculate_complexity(idx)
                self._indices[idx]['complexity'] = complexity
                total_complexity += complexity

        for db_idx1, db_idx2 in self._cistrans_double_bond_atoms:
            complexity_db_idx1 = self._calculate_complexity(db_idx1)
            complexity_db_idx2 = self._calculate_complexity(db_idx2)
            if complexity_db_idx1 <= complexity_db_idx2:
                complexity_db_idx1 *= 2
                self._indices[db_idx1]['si'] += 1
            else:
                complexity_db_idx2 *= 2
                self._indices[db_idx2]['si'] += 1
            self._indices[db_idx1]['complexity'] = complexity_db_idx1
            self._indices[db_idx2]['complexity'] = complexity_db_idx2
            total_complexity += complexity_db_idx1 + complexity_db_idx2

        for idx, eq_groups in self._equivalents.items():
            for e in eq_groups:
                total_complexity -= 0.5 * self._indices[idx]['complexity'] / len(eq_groups)
        self._intrinsic_complexity = total_complexity

    def _calculate_complexity(self, idx):
        """
        Calculate the complexity of the atom at the given index.

        This method calculates the complexity of an atom based on its degree (di),
        the number of unique element types (ei), the stereochemistry indicator (si),
        the valence indicator (Vi), and the bond order sum (bi).

        Args:
            idx (int): The index of the atom in the molecule.

        Returns:
            float: The calculated complexity of the atom. Returns 0 if an error occurs.
        """
        data = self._indices[idx]
        try:
            return data['di'] * data['ei'] * data['si'] * log(data['Vi'] * data['bi'], 2)
        except:
            print(f"[ *** Error calculating complexity: atom_idx[{idx}] *** ]")
            return 0

    def _find_cistrans_bond_atoms(self):
        """
        Identify and register cis-trans double bonds in the molecule.

        This method finds potential cis-trans double bonds in the molecule and registers them
        by checking the neighboring atoms and their branches.

        Returns:
            None
        """
        self._cistrans_double_bond_atoms = []
        for bond in self._find_potential_cistrans_bonds():
            db_atom_idx1, db_atom_idx2 = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
            double_bond_atoms = (self.mol.GetAtom(db_atom_idx1), self.mol.GetAtom(db_atom_idx2))
            register_cistrans = False
            for index, db_atom in enumerate(double_bond_atoms):
                self.excluded_atoms = [double_bond_atoms[index - 1]]
                neigh_atoms_db = self._find_neighbours(db_atom)
                neighbours_atom_numbers_db = self._find_neighbours(db_atom, return_atomic_nums=True)
                if len(set(neighbours_atom_numbers_db)) > 1 or len(neighbours_atom_numbers_db) == 1:
                    register_cistrans = True
                else:
                    branch1_atoms, branch2_atoms = [neigh_atoms_db[0]], [neigh_atoms_db[1]]
                    branches_equal = True
                    while branches_equal:
                        self.excluded_atoms += branch1_atoms + branch2_atoms
                        branches_equal = self._branch_levels_are_equal(branch1_atoms, branch2_atoms)
                        branch1_atoms = [neigh for atom in branch1_atoms for neigh in self._find_neighbours(atom)]
                        branch2_atoms = [neigh for atom in branch2_atoms for neigh in self._find_neighbours(atom)]
                        if not branch1_atoms and not branch2_atoms:
                            break
                    if branches_equal:
                        register_cistrans = False
                        break
                    else:
                        register_cistrans = True
            if register_cistrans:
                self._cistrans_double_bond_atoms.append((db_atom_idx1, db_atom_idx2))

    def _find_potential_cistrans_bonds(self):
        """
        Find potential cis-trans bonds in the molecule.

        This method identifies bonds in the molecule that have cis-trans stereochemistry
        and are specified as such.

        Returns:
            list: A list of bonds that have cis-trans stereochemistry.
        """
        double_bonds = []
        facade = ob.OBStereoFacade(self.mol)
        for bond in ob.OBMolBondIter(self.mol):
            if facade.HasCisTransStereo(bond.GetId()) and facade.GetCisTransStereo(bond.GetId()).IsSpecified():
                double_bonds.append(bond)
        return double_bonds

    def _branch_levels_are_equal(self, branch1_atoms, branch2_atoms):
        """
        Check if the branch levels of two sets of atoms are equal.

        This method compares the atomic numbers and bond orders of two branches of atoms
        to determine if they are equal.

        Args:
            branch1_atoms (list of OBAtom): The first branch of atoms to compare.
            branch2_atoms (list of OBAtom): The second branch of atoms to compare.

        Returns:
            bool: True if the branch levels are equal, False otherwise.
        """
        branch1_atomic_nums = sorted(self._find_neighbours(atom, return_atomic_nums=True) for atom in branch1_atoms)
        branch2_atomic_nums = sorted(self._find_neighbours(atom, return_atomic_nums=True) for atom in branch2_atoms)
        branch1_bond_orders = sorted(self._indices[atom.GetIdx()]['bi'] for atom in branch1_atoms)
        branch2_bond_orders = sorted(self._indices[atom.GetIdx()]['bi'] for atom in branch2_atoms)
        return branch1_atomic_nums == branch2_atomic_nums and branch1_bond_orders == branch2_bond_orders

    def _find_neighbours(self, atom, return_atomic_nums=False):
        """
        Find the neighboring atoms of the given atom.

        This method retrieves the neighboring atoms of the specified atom. If return_atomic_nums
        is True, it returns the atomic numbers of the neighboring atoms instead of the atom objects.

        Args:
            atom (OBAtom): The atom object for which the neighbors are being found.
            return_atomic_nums (bool, optional): Flag to return atomic numbers instead of atom objects. Defaults to False.

        Returns:
            list: A list of neighboring atom objects or their atomic numbers.
        """
        neigh_atoms = []
        neigh_atomic_numbers = []
        for neigh in ob.OBAtomAtomIter(atom):
            if neigh in self.excluded_atoms:
                continue
            neigh_atoms.append(neigh)
            neigh_atomic_numbers.append(self._get_atomic_num(neigh))
        return neigh_atomic_numbers if return_atomic_nums else neigh_atoms

    def _calc_vi(self, idx, atom):
        """
        Calculate the valence indicator (Vi) for the atom at the given index.

        This method calculates the valence indicator based on the total valence and formal charge of the atom.

        Args:
            idx (int): The index of the atom in the molecule.
            atom (OBAtom): The atom object for which the valence indicator is being calculated.

        Returns:
            None
        """
        self._indices[idx]['Vi'] = 8 - atom.GetTotalValence() + atom.GetFormalCharge()

    def _calc_bi_ei_si(self, idx, atom):
        """
        Calculate the bond order sum (bi), the number of unique element types (ei), and the stereochemistry indicator (si) for the atom at the given index.

        Args:
            idx (int): The index of the atom in the molecule.
            atom (OBAtom): The atom object for which the properties are being calculated.

        Returns:
            None
        """
        bi = 0
        ei = [self._get_atomic_num(atom)]
        si = 1 + int(atom.IsChiral())
        for neigh in ob.OBAtomAtomIter(atom):
            if self._is_hydrogen(neigh.GetIdx()):
                continue
            contribution = self._mesomery_equivalence.get(idx, self.mol.GetBond(atom, neigh).GetBondOrder())
            bi += contribution
            ei.append(self._get_atomic_num(neigh))
        self._indices[idx]['bi'] = bi
        self._indices[idx]['ei'] = len(set(ei))
        self._indices[idx]['si'] = si

    @staticmethod
    def _get_atomic_num(atom):
        """
        Get the atomic number of the given atom.

        This method returns the isotope number if it exists, otherwise it returns the atomic number.

        Args:
            atom (OBAtom): The atom object for which the atomic number is being retrieved.

        Returns:
            int: The isotope number if it exists, otherwise the atomic number.
        """
        return atom.GetIsotope() or atom.GetAtomicNum()

    def _calc_di(self, idx, atom):
        """
        Calculate the degree of the atom at the given index.

        This method updates the equivalents dictionary with automorphs and calculates
        the degree of the atom, storing it in the indices dictionary.

        Args:
            idx (int): The index of the atom in the molecule.
            atom (OBAtom): The atom object for which the degree is being calculated.

        Returns:
            None
        """
        self._equivalents[idx] = self.automorphs.get(idx, set())
        if idx in self.automorphs:
            self._equivalents[idx].update(self.automorphs[idx])
        groups = []
        for neigh in ob.OBAtomAtomIter(atom):
            if self._is_hydrogen(neigh.GetIdx()):
                continue
            if neigh.GetIdx() in self.automorphs and set(groups) & self.automorphs[neigh.GetIdx()]:
                continue
            groups.append(neigh.GetIdx())
        self._indices[idx]['di'] = len(groups)

    def _build_automorphism(self):
        """
        Build the automorphism groups for the molecule.

        This method initializes the equivalents dictionary and finds the automorphisms
        of the molecule, storing them in the automorphs dictionary.

        The automorphisms are found using the Open Babel library.

        Returns:
            None
        """
        self._equivalents = {}
        automorphs = ob.vvpairUIntUInt()
        mol_copy = ob.OBMol(self.mol)
        for i in ob.OBMolAtomIter(mol_copy):
            if i.GetIsotope():
                i.SetAtomicNum(i.GetAtomicNum() + i.GetIsotope())
        ob.FindAutomorphisms(mol_copy, automorphs, pybel.ob.OBBitVec(), self.automorp_memory_maxsize)
        self.automorphs = {}
        for am in automorphs:
            for i, j in am:
                if i != j and not self._is_hydrogen(i + 1) and not self._is_hydrogen(j + 1):
                    self.automorphs.setdefault(i + 1, set()).add(j + 1)
        for k, v in self.automorphs.items():
            self.automorphs[k] = set(v)

    def _is_hydrogen(self, idx):
        """
        Check if the atom at the given index is a hydrogen atom.

        Args:
            idx (int): The index of the atom in the molecule.

        Returns:
            bool: True if the atom is hydrogen, False otherwise.
        """
        return self.mol.GetAtom(idx).GetAtomicNum() == 1


def calculate_bottcher_score(mol: Mol, mesomer: bool = False, automorp_max: int = 3000000) -> float:
    """
    Calculate the Böttcher score for a given molecule.
    https://github.com/forlilab/bottchscore
    https://pubs.acs.org/doi/10.1021/jacs.0c08231
    https://pubs.acs.org/doi/10.1021/acs.jcim.5b00723

    The Böttcher score is a measure of molecular complexity based on various
    atomic and molecular properties. This function converts the RDKit molecule
    to a SMILES string, reads it into a Pybel molecule, and calculates the score
    using the `BottcherScore` class.

    Args:
        mol (Mol): An RDKit molecule object.
        mesomer (bool, optional): Whether to include mesomeric contributions in the score calculation. Defaults to False.
        automorp_max (int, optional): Maximum size for automorphism memory. Defaults to 3000000.

    Returns:
        float: The calculated Böttcher score for the molecule.
    """
    smi: str = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True, allHsExplicit=True)
    mol_input: pybel.Molecule = pybel.readstring("smi", smi)
    return BottcherScore(automorp_memory_maxsize=automorp_max).score(mol_input.OBMol, mesomer)
