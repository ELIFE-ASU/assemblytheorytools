"""
Molecular complexity and similarity measures.

This module collects scalar descriptors used to compare molecules against the
assembly index. It covers simple counts (bonds, molecular weight), graph
invariants (Bertz, Wiener, Balaban, Randic and Kirchhoff indices), published
complexity scores (spacial score, Boettcher, Proudfoot, MC1 and MC2),
compression-based proxies using ``zlib``, ``bz2`` and ``lzma``, and fingerprint
similarity measures.

Each published measure carries the DOI of the paper defining it in its own
docstring; the compression proxies and ``shannon_entropy`` are not published
molecular complexity measures. See the citing page for the full list.
"""

import bz2
import json
import lzma
import math
import traceback
import zlib
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Callable, Dict, Optional, Tuple

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.SpacialScore import SPS
from rdkit.Chem.rdchem import Mol

from .tools_graph import remove_hydrogen_from_graph
from .tools_mol import standardize_mol


def count_unique_bonds(mol: Mol) -> int:
    """
    Count distinct unordered atom-symbol pairs and bond types.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.

    Returns
    -------
    int
        The number of unique bonds in the molecule.
    """
    unique_bonds = {
        (
            tuple(
                sorted((bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()))
            ),
            bond.GetBondType(),
        )
        for bond in mol.GetBonds()
    }
    return len(unique_bonds)


def count_bonds(mol: Mol) -> int:
    """
    Count all bonds, including those to explicit hydrogens.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.

    Returns
    -------
    int
        The total number of bonds in the molecule.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> mol = att.smi_to_mol("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")  # caffeine
    >>> att.count_bonds(mol)
    25
    >>> att.count_non_h_bonds(mol)
    15
    """
    return mol.GetNumBonds()


def count_non_h_bonds(mol: Mol) -> int:
    """
    Count bonds whose endpoints are both non-hydrogen atoms.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.

    Returns
    -------
    int
        The number of bonds in the molecule that do not involve hydrogen
        atoms.
    """
    return sum(
        bond.GetBeginAtom().GetSymbol() != "H" and bond.GetEndAtom().GetSymbol() != "H"
        for bond in mol.GetBonds()
    )


def molecular_weight(mol: Mol) -> float:
    """
    Calculate molecular weight with RDKit's ``MolWt`` descriptor.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.

    Returns
    -------
    float
        The molecular weight of the molecule.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> round(att.molecular_weight(
    ...     att.smi_to_mol("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")), 2)
    194.19
    """
    return Descriptors.MolWt(mol)


def bertz_complexity(mol: Mol) -> float:
    """
    Calculate the Bertz structural complexity with RDKit.

    Reference: https://doi.org/10.1021/ja00402a071.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.

    Returns
    -------
    float
        The Bertz complexity of the molecule.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> round(att.bertz_complexity(att.smi_to_mol("c1ccccc1")), 2)
    226.3
    >>> round(att.bertz_complexity(
    ...     att.smi_to_mol("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")), 2)
    924.42
    """
    return BertzCT(mol)


def wiener_index(mol: Mol) -> int:
    """
    Sum shortest-path distances over all unordered atom pairs.

    Reference: https://doi.org/10.1021/ja01193a005.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.

    Returns
    -------
    int
        The Wiener index of the molecule.

    Notes
    -----
    - Every atom present in ``mol`` is a vertex of the graph, so molecules
      carrying explicit hydrogens give a larger index than the heavy-atom
      skeleton alone. Strip the hydrogens first if the heavy-atom value is
      wanted.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> att.wiener_index(att.smi_to_mol("c1ccccc1"))
    174
    >>> att.wiener_index(att.smi_to_mol("CN1C=NC2=C1C(=O)N(C(=O)N2C)C"))
    1089
    """
    distance_matrix = Chem.rdmolops.GetDistanceMatrix(mol)
    # The distance matrix is symmetric with a zero diagonal, so summing every
    # entry counts each unordered pair of atoms exactly twice.
    return int(distance_matrix.sum()) // 2


def balaban_index(mol: Mol) -> float:
    """
    Calculate the Balaban connectivity index with RDKit's ``BalabanJ``.

    Reference: https://doi.org/10.1016/0009-2614(82)80009-2.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.

    Returns
    -------
    float
        The Balaban index of the molecule.
    """
    return Descriptors.BalabanJ(mol)


def randic_index(mol: Mol) -> float:
    """
    Sum inverse square roots of endpoint-degree products over all bonds.

    Reference: https://doi.org/10.1021/ja00856a001.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.

    Returns
    -------
    float
        The Randic index.
    """
    adjacency = Chem.rdmolops.GetAdjacencyMatrix(mol)
    degrees = adjacency.sum(axis=1)
    rows, columns = np.nonzero(adjacency == 1)
    # Visit both directions in matrix order to preserve numerical summation.
    return sum(1 / (degrees[i] * degrees[j]) ** 0.5 for i, j in zip(rows, columns)) / 2


def kirchhoff_index(mol: Mol) -> float:
    """
    Sum pairwise effective resistances using the Laplacian pseudoinverse.

    Reference: https://doi.org/10.1007/BF01164627.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.

    Returns
    -------
    float
        The Kirchhoff index.
    """
    adjacency = Chem.rdmolops.GetAdjacencyMatrix(mol).astype(np.float64)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    inverse = np.linalg.pinv(laplacian)
    diagonal = inverse.diagonal()
    return sum(
        diagonal[i] + diagonal[j] - 2 * inverse[i, j]
        for i, j in combinations(range(len(diagonal)), 2)
    )


def spacial_score(mol: Mol, normalise: bool = False) -> float:
    """
    Calculate the spacial score, optionally normalised per heavy atom.

    Reference: https://doi.org/10.1021/acs.jmedchem.3c00689.
    See https://github.com/frog2000/Spacial-Score for the reference
    implementation.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.
    normalise : bool, optional
        Whether to normalise the score, by default False.

    Returns
    -------
    float
        The spacial score of the molecule.
    """
    return SPS(mol, normalise)


def get_mol_descriptors(mol: Mol, missingval: Optional[Any] = None) -> Dict[str, Any]:
    """
    Calculate all registered RDKit descriptors, substituting failures.

    Failed calculations print a traceback and receive ``missingval``. See
    https://greglandrum.github.io/rdkit-blog/posts/2022-12-23-descriptor-tutorial.html
    for an overview of RDKit descriptors.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.
    missingval : Optional[Any], optional
        Value assigned to failed descriptor calculations, by default None.

    Returns
    -------
    Dict[str, Any]
        A dictionary with descriptor names as keys and their calculated
        values as values.
    """
    descriptors = {}
    for name, calculate in Descriptors._descList:
        try:
            descriptors[name] = calculate(mol)
        except Exception:
            traceback.print_exc()
            descriptors[name] = missingval
    return descriptors


def tanimoto_similarity(mol1: Mol, mol2: Mol) -> float:
    """
    Calculate Tanimoto similarity between RDKit topological fingerprints.

    Parameters
    ----------
    mol1 : rdkit.Chem.rdchem.Mol
        The first RDKit molecule.
    mol2 : rdkit.Chem.rdchem.Mol
        The second RDKit molecule.

    Returns
    -------
    float
        The Tanimoto similarity between the two molecules.

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> round(att.tanimoto_similarity(att.smi_to_mol("NCC(=O)O"),
    ...                               att.smi_to_mol("CC(N)C(=O)O")), 4)
    0.3265

    This compares fingerprint bits. With hydrogens stripped, the
    assembly-theoretic score for the same pair is 0.75 -- see
    :func:`~assemblytheorytools.assembly.calculate_assembly_index_similarity`
    -- which asks instead how much of the construction work is shared.
    """
    fpgen = Chem.GetRDKitFPGenerator()
    return DataStructs.TanimotoSimilarity(
        fpgen.GetFingerprint(mol1), fpgen.GetFingerprint(mol2)
    )


def dice_morgan_similarity(mol1: Mol, mol2: Mol, radius: int = 3) -> float:
    """
    Calculate Dice similarity between sparse Morgan count fingerprints.

    Parameters
    ----------
    mol1 : rdkit.Chem.rdchem.Mol
        The first RDKit molecule.
    mol2 : rdkit.Chem.rdchem.Mol
        The second RDKit molecule.
    radius : int, optional
        Morgan fingerprint radius, by default 3.

    Returns
    -------
    float
        The Dice similarity between the two molecules.
    """
    fpgen = Chem.GetMorganGenerator(radius=radius)
    return DataStructs.DiceSimilarity(
        fpgen.GetSparseCountFingerprint(mol1), fpgen.GetSparseCountFingerprint(mol2)
    )


def get_chirality(mol: Mol) -> int:
    """
    Count chiral centres, including those with unassigned stereochemistry.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.

    Returns
    -------
    int
        The number of chiral centres in the molecule.
    """
    return len(
        Chem.FindMolChiralCenters(
            mol, useLegacyImplementation=False, includeUnassigned=True, includeCIP=False
        )
    )


def _standardised_smiles(mol: Mol, add_hydrogens: bool) -> str:
    """Return canonical, kekulised, isomeric SMILES with optional hydrogens."""
    mol = standardize_mol(mol, add_hydrogens=add_hydrogens)
    if not add_hydrogens:
        mol = Chem.RemoveHs(mol)

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        kekuleSmiles=True,
        isomericSmiles=True,
        allHsExplicit=add_hydrogens,
    )


def _compressed_length(
    payload: bytes,
    compress: Callable[[bytes], bytes],
    decompress: Callable[[bytes], Any],
    check: bool,
    rm_overhead: bool,
) -> int:
    """
    Return compressed payload length, optionally net of empty-byte overhead.

    When ``check`` is true, decompression errors are printed and re-raised.
    """
    compressed = compress(payload)
    if check:
        _check_decompression(compressed, decompress)
    overhead = len(compress(b"")) if rm_overhead else 0
    return len(compressed) - overhead


def _check_decompression(compressed: bytes, decompress: Callable[[bytes], Any]) -> None:
    """Report decompression errors and re-raise them unchanged."""
    try:
        decompress(compressed)
    except Exception as error:
        print(f"Decompression failed: {error}")
        raise


def compression_zlib_smi(
    mol: Mol,
    add_hydrogens: bool = True,
    level: int = 9,
    check: bool = True,
    rm_overhead: bool = True,
) -> int:
    """
    Return the zlib-compressed size of a standardised canonical SMILES
    string.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens, by default True.
    level : int, optional
        zlib compression level (0-9), by default 9.
    check : bool, optional
        Whether to verify successful decompression, by default True.
    rm_overhead : bool, optional
        Whether to subtract compressed empty-string overhead, by default
        True.

    Returns
    -------
    int
        The length of the compressed SMILES string, adjusted for overhead if
        specified.

    Raises
    ------
    Exception
        If decompression fails during the integrity check.
    """
    return _compressed_length(
        _standardised_smiles(mol, add_hydrogens).encode("utf-8"),
        compress=lambda data: zlib.compress(data, level=level),
        decompress=lambda data: zlib.decompress(data).decode("utf-8"),
        check=check,
        rm_overhead=rm_overhead,
    )


def compression_bz2_smi(
    mol: Mol, add_hydrogens: bool = True, check: bool = True, rm_overhead: bool = True
) -> int:
    """
    Return the bz2-compressed size of a standardised canonical SMILES
    string.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens, by default True.
    check : bool, optional
        Whether to verify successful decompression, by default True.
    rm_overhead : bool, optional
        Whether to subtract compressed empty-string overhead, by default
        True.

    Returns
    -------
    int
        The length of the compressed SMILES string, adjusted for overhead if
        specified.

    Raises
    ------
    Exception
        If decompression fails during the integrity check.
    """
    return _compressed_length(
        _standardised_smiles(mol, add_hydrogens).encode("utf-8"),
        compress=bz2.compress,
        decompress=lambda data: bz2.decompress(data).decode("utf-8"),
        check=check,
        rm_overhead=rm_overhead,
    )


def compression_lzma_smi(
    mol: Mol, add_hydrogens: bool = True, check: bool = True, rm_overhead: bool = True
) -> int:
    """
    Return the lzma-compressed size of a standardised canonical SMILES
    string.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens, by default True.
    check : bool, optional
        Whether to verify successful decompression, by default True.
    rm_overhead : bool, optional
        Whether to subtract compressed empty-string overhead, by default
        True.

    Returns
    -------
    int
        The length of the compressed SMILES string, adjusted for overhead if
        specified.

    Raises
    ------
    Exception
        If decompression fails during the integrity check.
    """
    return _compressed_length(
        _standardised_smiles(mol, add_hydrogens).encode("utf-8"),
        compress=lzma.compress,
        decompress=lambda data: lzma.decompress(data).decode("utf-8"),
        check=check,
        rm_overhead=rm_overhead,
    )


def compress_zlib_graph(graph: nx.Graph, level: int = 9) -> bytes:
    """
    Compress a graph as UTF-8 node-link JSON using zlib.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph to compress.
    level : int, optional
        zlib compression level (0-9), by default 9.

    Returns
    -------
    bytes
        The compressed graph data as a byte string.
    """
    return zlib.compress(_graph_json_bytes(graph), level)


def _graph_json_bytes(graph: nx.Graph) -> bytes:
    """Serialise node-link data with the formatting used by graph scores."""
    return json.dumps(json_graph.node_link_data(graph)).encode("utf-8")


def decompress_zlib_graph(compressed_data: bytes) -> nx.Graph:
    """
    Reconstruct a graph from zlib-compressed UTF-8 node-link JSON.

    Parameters
    ----------
    compressed_data : bytes
        The compressed node-link JSON bytes.

    Returns
    -------
    nx.Graph
        The reconstructed NetworkX graph.
    """
    data = json.loads(zlib.decompress(compressed_data).decode("utf-8"))
    return json_graph.node_link_graph(data)


def _compressed_zlib_graph_size(
    graph: nx.Graph, level: int, check: bool, rm_overhead: bool
) -> int:
    """
    Return compressed graph length, optionally net of empty-graph overhead.

    The empty graph always uses the default compression level. The caller
    handles hydrogen removal; ``check`` verifies successful decompression.
    """
    compressed = compress_zlib_graph(graph, level=level)
    if check:
        _check_decompression(compressed, decompress_zlib_graph)
    # The empty-graph baseline always uses the default compression level.
    overhead = len(compress_zlib_graph(nx.Graph())) if rm_overhead else 0
    return len(compressed) - overhead


def compression_zlib_graph(
    graph: nx.Graph,
    add_hydrogens: bool = True,
    level: int = 9,
    check: bool = True,
    rm_overhead: bool = True,
) -> int:
    """
    Return the zlib-compressed graph size, optionally subtracting overhead.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph to compress.
    add_hydrogens : bool, optional
        Whether to retain hydrogens, by default True. False strips them from
        a copy of the graph.
    level : int, optional
        zlib compression level (0-9), by default 9.
    check : bool, optional
        Whether to verify successful decompression, by default True.
    rm_overhead : bool, optional
        Whether to subtract the empty-graph size at compression level 9, by
        default True.

    Returns
    -------
    int
        The length of the compressed graph data, adjusted for overhead if
        specified.

    Raises
    ------
    Exception
        If decompression fails during the integrity check.
    """
    if not add_hydrogens:
        graph = remove_hydrogen_from_graph(graph)

    return _compressed_zlib_graph_size(graph, level, check, rm_overhead)


def compression_ratio_zlib_graph(
    graph: nx.Graph,
    add_hydrogens: bool = True,
    level: int = 9,
    check: bool = True,
    rm_overhead: bool = True,
) -> float:
    """
    Divide the graph's uncompressed JSON size by its compressed size.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph to compress.
    add_hydrogens : bool, optional
        Whether to retain hydrogens, by default True. False strips them from
        a copy of the graph.
    level : int, optional
        zlib compression level (0-9), by default 9.
    check : bool, optional
        Whether to verify successful decompression, by default True.
    rm_overhead : bool, optional
        Whether to subtract the empty-graph size at compression level 9, by
        default True.

    Returns
    -------
    float
        The compression ratio of the graph, calculated as the size of the
        uncompressed graph divided by the size of the compressed graph.

    Raises
    ------
    Exception
        If decompression fails during the integrity check.
    """
    if not add_hydrogens:
        graph = remove_hydrogen_from_graph(graph)

    uncompressed_size = len(_graph_json_bytes(graph))
    compressed_size = _compressed_zlib_graph_size(graph, level, check, rm_overhead)
    return uncompressed_size / compressed_size


def fcfp4(mol: Mol) -> int:
    """
    Count set bits in a 2048-bit, radius-2 feature-based Morgan fingerprint.

    Reference: https://doi.org/10.1021/ci0503558.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.

    Returns
    -------
    int
        The number of bits set to 1 in the generated FCFP_4 fingerprint.
    """
    fingerprint = Chem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=2048, useFeatures=True
    )
    return fingerprint.GetNumOnBits()


def _determine_atom_substituents(
    atom_id: int, mol: Mol, distance_matrix: np.ndarray
) -> Tuple[Dict[int, list], Dict[int, int], Dict[int, int]]:
    """
    Return ordered substituents, shared-atom counts and maximum shell
    depths.

    Each substituent starts at a direct neighbour. Atoms can belong to
    multiple substituents in rings. Shells follow the supplied shortest-path
    distances, with direct neighbours assigned depth zero.
    """
    atom_paths = distance_matrix[atom_id]
    shells = defaultdict(list)
    for index, distance in enumerate(atom_paths):
        shells[distance].append(index)

    # Preserve atom-index order within each shell and substituent.
    subs = defaultdict(list)
    shared_neighbors = defaultdict(int)
    max_shell = defaultdict(int)
    for neighbor in shells[1]:
        subs[neighbor].append(neighbor)
        shared_neighbors[neighbor] += 1
        max_shell[neighbor] = 0

    # Disconnected atoms have a large distance sentinel; skip empty shells.
    for distance in sorted(distance for distance in shells if distance >= 2):
        for index in shells[distance]:
            for neighbor in mol.GetAtomWithIdx(index).GetNeighbors():
                neighbor_index = neighbor.GetIdx()
                if atom_paths[neighbor_index] == distance:
                    continue
                for root, substituent in subs.items():
                    if neighbor_index in substituent and index not in substituent:
                        substituent.append(index)
                        shared_neighbors[index] += 1
                        max_shell[root] = int(distance)

    return subs, shared_neighbors, max_shell


def _get_chemical_non_equivs(atom: Chem.rdchem.Atom, mol: Mol) -> float:
    """
    Count distinct ordered element sequences among up to four substituents.

    Log conversion errors and return 0.0 if the four-substituent limit is
    exceeded or an element sequence cannot be built.
    """
    # Retain the four-substituent limit and its error fallback.
    substituents = [[] for _ in range(4)]
    distance_matrix = Chem.GetDistanceMatrix(mol)
    atom_substituents = _determine_atom_substituents(
        atom.GetIdx(), mol, distance_matrix
    )[0]
    try:
        for index, atom_indices in enumerate(atom_substituents.values()):
            substituents[index].extend(
                mol.GetAtomWithIdx(atom_index).GetSymbol()
                for atom_index in atom_indices
            )
        return float(
            len({tuple(substituent) for substituent in substituents if substituent})
        )
    except Exception as error:
        print(
            f"Error calculating chemical non-equivalence for atom {atom.GetIdx()}: "
            f"{error}"
        )
        print(traceback.format_exc())
        return 0.0


def _get_bottcher_local_diversity(atom: Chem.rdchem.Atom) -> float:
    """Count distinct elements across an atom and its neighbours."""
    symbols = {neighbor.GetSymbol() for neighbor in atom.GetNeighbors()}
    symbols.add(atom.GetSymbol())
    return float(len(symbols))


def _get_num_isomeric_possibilities(atom: Chem.rdchem.Atom) -> float:
    """Return 2.0 for atoms with a CIP code, otherwise 1.0."""
    return 2.0 if atom.HasProp("_CIPCode") else 1.0


def _get_num_valence_electrons(
    atom: Chem.rdchem.Atom, pt: Chem.rdchem.PeriodicTable
) -> float:
    """Return the number of outer-shell electrons for an atom."""
    return float(pt.GetNOuterElecs(atom.GetAtomicNum()))


def _get_bottcher_bond_index(atom: Chem.rdchem.Atom) -> float:
    """
    Sum bond orders with one aromatic correction per carbon or nitrogen.

    Raise ``ValueError`` for unsupported bond types.
    """
    bond_weights = {
        "SINGLE": 1.0,
        "DOUBLE": 2.0,
        "TRIPLE": 3.0,
        "QUADRUPLE": 4.0,
        "QUINTUPLE": 5.0,
        "HEXTUPLE": 6.0,
        "AROMATIC": 0.0,
    }
    bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
    bond_index = 0.0
    for bond in bonds:
        if bond not in bond_weights:
            raise ValueError(f"Unsupported bond type {bond}")
        bond_index += bond_weights[bond]

    # Aromatic bonds are weighted by the ring atom they belong to
    if "AROMATIC" in bonds:
        bond_index += {"C": 3.0, "N": 2.0}.get(atom.GetSymbol(), 0.0)
    return bond_index


def bottcher(mol: Mol) -> float:
    """
    Calculate symmetry-corrected Bottcher molecular complexity.

    Each atom contributes chemical non-equivalence, local diversity,
    stereochemistry, valence electrons and bond weights. Stereochemistry is
    assigned on the input molecule.

    References:
    https://github.com/boskovicgroup/bottchercomplexity
    https://doi.org/10.1021/acs.jcim.5b00723

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.

    Returns
    -------
    float
        The Bottcher complexity of the molecule.
    """
    Chem.AssignStereochemistry(
        mol, cleanIt=True, force=True, flagPossibleStereoCenters=True
    )
    pt = Chem.GetPeriodicTable()
    # Keep the first atom per canonical rank; _CIPRank is not always populated.
    canonical_ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
    seen_ranks = set()
    complexity = 0.0
    for atom in mol.GetAtoms():
        rank = canonical_ranks[atom.GetIdx()]
        if rank in seen_ranks:
            continue
        seen_ranks.add(rank)

        d = _get_chemical_non_equivs(atom, mol)  # Chemical non-equivalence
        e = _get_bottcher_local_diversity(atom)  # Local diversity
        s = _get_num_isomeric_possibilities(atom)  # Isomeric possibilities
        v = _get_num_valence_electrons(atom, pt)  # Number of valence electrons
        b = _get_bottcher_bond_index(atom)  # Bond index
        complexity += d * e * s * np.log2(v * b)

    return complexity


def proudfoot(mol: Mol) -> float:
    """
    Calculate Proudfoot molecular complexity (C_M).

    Sum the Shannon entropy of each Morgan path-count group plus ``log2`` of
    its total path count.

    Reference: https://doi.org/10.1016/j.bmcl.2017.03.008.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.

    Returns
    -------
    float
        The Proudfoot molecular complexity C_M of the molecule.

    Notes
    -----
    The paper also defines a log-sum complexity (C_M*) and a structural
    entropy term (C_SE); neither is returned here.
    """
    fingerprint = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    atom_paths = defaultdict(list)
    for path, count in fingerprint.GetNonzeroElements().items():
        atom_paths[path % mol.GetNumAtoms()].append(count)

    complexities = []
    for path_counts in atom_paths.values():
        total_paths = sum(path_counts)
        fractions = (count / total_paths for count in path_counts)
        entropy = -sum(p * math.log2(p) for p in fractions)
        complexities.append(entropy + math.log2(total_paths))
    return sum(complexities)


def mc1(mol: Mol) -> float:
    """
    Calculate MC1: one minus the fraction of atoms with degree two.

    Reference: https://doi.org/10.1021/acs.jcim.5c00334.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.

    Returns
    -------
    float
        The MC1 index of the molecule.
    """
    divalent_nodes = sum(atom.GetDegree() == 2 for atom in mol.GetAtoms())
    return 1.0 - divalent_nodes / mol.GetNumAtoms()


def mc2(mol: Mol) -> int:
    """
    Count non-divalent atoms outside qualifying carbonyl groups.

    Exclude both atoms of each C=O bond whose carbon also has a nitrogen or
    oxygen neighbour.

    Reference: https://doi.org/10.1021/acs.jcim.5c00334.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to measure.

    Returns
    -------
    int
        The count of non-divalent atoms not involved in C=O-X double bonds.
    """
    excluded_atoms = set()
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.rdchem.BondType.DOUBLE:
            continue
        carbon, oxygen = bond.GetBeginAtom(), bond.GetEndAtom()
        if carbon.GetAtomicNum() == 8:
            carbon, oxygen = oxygen, carbon
        if carbon.GetAtomicNum() != 6 or oxygen.GetAtomicNum() != 8:
            continue
        if any(
            neighbor.GetIdx() != oxygen.GetIdx() and neighbor.GetAtomicNum() in (7, 8)
            for neighbor in carbon.GetNeighbors()
        ):
            excluded_atoms.update((carbon.GetIdx(), oxygen.GetIdx()))

    return sum(
        atom.GetDegree() != 2 and atom.GetIdx() not in excluded_atoms
        for atom in mol.GetAtoms()
    )


def shannon_entropy(s: str) -> float:
    """
    Calculate character entropy in bits: ``-sum(p * log2(p))``.

    Here ``p`` is the frequency of each distinct character divided by the
    string length.

    Usage example: https://doi.org/10.1126/sciadv.abj2465

    Parameters
    ----------
    s : str
        The input string.

    Returns
    -------
    float
        Entropy in bits. Returns 0.0 for empty strings.
    """
    if not s:
        return 0.0

    # Accumulated term by term rather than via sum(), whose compensated
    # summation would shift results in the last bit
    entropy = 0.0
    for count in Counter(s).values():
        p = count / len(s)
        entropy -= p * math.log2(p)
    return entropy
