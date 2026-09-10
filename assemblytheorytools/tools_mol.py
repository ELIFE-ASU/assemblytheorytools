"""
RDKit molecule helpers.

This module wraps the RDKit operations used throughout the package:
standardisation and sanitisation, free-valence and formal-charge handling,
construction from SMILES, InChI and mol files, combining and splitting
multi-fragment molecules, V2000 mol file writing, and peptide sequence
conversion.
"""

import re
import warnings
from typing import List, Union

import networkx as nx
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdchem import GetPeriodicTable

_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def safe_standardize_mol(mol: Chem.Mol, add_hydrogens: bool = True) -> Chem.Mol:
    """
    Standardise a molecule with relaxed valence checks.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The molecule to standardise in place.
    add_hydrogens : bool, optional
        Whether to return a copy with explicit hydrogens. Default is True.

    Returns
    -------
    rdkit.Chem.Mol
        The standardised molecule, or a hydrogenated copy if requested.

    Notes
    -----
    Skips cleanup and property sanitisation, but propagates other RDKit
    sanitisation errors. Without added hydrogens, returns the input object.
    """
    mol.UpdatePropertyCache(strict=False)
    Chem.SetConjugation(mol)
    Chem.SetHybridization(mol)
    Chem.SanitizeMol(
        mol,
        sanitizeOps=(
            Chem.SANITIZE_ALL ^ Chem.SANITIZE_CLEANUP ^ Chem.SANITIZE_PROPERTIES
        ),
        catchErrors=False,
    )
    rdMolStandardize.NormalizeInPlace(mol)
    Chem.Kekulize(mol)
    return Chem.AddHs(mol) if add_hydrogens else mol


def standardize_mol(mol: Chem.Mol, add_hydrogens: bool = True) -> Chem.Mol:
    """
    Standardise a molecule with full sanitisation and strict valence checks.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to standardise in place.
    add_hydrogens : bool, optional
        Whether to return a copy with explicit hydrogens. Default is True.

    Returns
    -------
    Chem.Mol
        The standardised molecule, or a hydrogenated copy if requested.

    Notes
    -----
    RDKit sanitisation errors propagate to the caller. Without added
    hydrogens, returns the input object.
    """
    Chem.SanitizeMol(mol, catchErrors=False)
    rdMolStandardize.NormalizeInPlace(mol)
    mol.UpdatePropertyCache(strict=True)
    Chem.Kekulize(mol)
    return Chem.AddHs(mol) if add_hydrogens else mol


def get_free_valence(
    atom: Chem.Atom,
    pt: Chem.rdchem.PeriodicTable = None,
    method: str = "1",
) -> int:
    """
    Calculate an atom's free valence using the selected method.

    Parameters
    ----------
    atom : Chem.Atom
        The atom to inspect.
    pt : Chem.rdchem.PeriodicTable, optional
        Periodic table to use. Defaults to RDKit's global table.
    method : str, optional
        ``'1'`` (default): minimum allowed valence minus the integer part of
        the sum of bond orders. ``'2'``: default minus explicit valence.
        ``'3'``: number of outer electrons.

    Returns
    -------
    int
        The calculated free valence of the atom.

    Raises
    ------
    ValueError
        If an unknown method is provided.
    """
    pt = pt or GetPeriodicTable()
    symbol = atom.GetSymbol()
    atomic_number = pt.GetAtomicNumber(symbol)

    if method == "1":
        bond_order = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
        return min(pt.GetValenceList(atomic_number)) - int(bond_order)
    if method == "2":
        return pt.GetDefaultValence(symbol) - atom.GetExplicitValence()
    if method == "3":
        return pt.GetNOuterElecs(atomic_number)
    raise ValueError(
        f"Unknown method {method} for calculating free valence of atom {symbol}"
    )


def reset_mol_charge(mol: Chem.Mol, pt: Chem.rdchem.PeriodicTable = None) -> Chem.Mol:
    """
    Return a copy with formal charges set to each atom's free valence.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to copy and adjust.
    pt : Chem.rdchem.PeriodicTable, optional
        Periodic table to use. Defaults to RDKit's global table.

    Returns
    -------
    Chem.Mol
        A new RDKit molecule object with updated formal charges.

    Notes
    -----
    Uses :func:`get_free_valence` with its default method. The input is
    unchanged; the result is not sanitised or given a property-cache update.
    """
    pt = pt or GetPeriodicTable()
    editable = Chem.RWMol(mol)

    for atom in editable.GetAtoms():
        atom.SetFormalCharge(get_free_valence(atom, pt=pt))

    return editable.GetMol()


def get_total_free_valence(
    mol: Chem.Mol | nx.Graph,
    pt: Chem.rdchem.PeriodicTable = None,
    method: str = "1",
) -> int:
    """
    Calculate the total free valence of a molecule or graph.

    Parameters
    ----------
    mol : Chem.Mol | nx.Graph
        A molecule or graph. Graphs store element symbols in node ``color``
        attributes and bond orders in edge ``color`` attributes.
    pt : Chem.rdchem.PeriodicTable, optional
        Periodic table to use. Defaults to RDKit's global table.
    method : str, optional
        Passed to :func:`get_free_valence` for RDKit molecules. Default is
        ``'1'``. Ignored for graphs.

    Returns
    -------
    int
        The total free valence of the molecule or graph.

    Notes
    -----
    Graphs use each element's minimum allowed valence minus the sum of its
    incident bond orders, converting each edge's ``color`` to an integer.
    """
    pt = pt or GetPeriodicTable()
    if isinstance(mol, nx.Graph):
        total = 0
        for node, data in mol.nodes(data=True):
            atomic_number = pt.GetAtomicNumber(data.get("color"))
            bond_order = sum(
                int(mol.edges[edge].get("color")) for edge in mol.edges(node)
            )
            total += min(pt.GetValenceList(atomic_number)) - bond_order
        return total

    return sum(get_free_valence(atom, pt=pt, method=method) for atom in mol.GetAtoms())


def _maybe_sanitize(mol: Chem.Mol, sanitize: bool, add_hydrogens: bool) -> Chem.Mol:
    """Standardise if requested; otherwise ignore the hydrogen flag."""
    return safe_standardize_mol(mol, add_hydrogens=add_hydrogens) if sanitize else mol


def smi_to_mol(smi: str, add_hydrogens: bool = True, sanitize: bool = True) -> Chem.Mol:
    """
    Convert a SMILES string to an optionally standardised RDKit molecule.

    Parameters
    ----------
    smi : str
        A SMILES string representing the molecular structure.
    add_hydrogens : bool, optional
        Add explicit hydrogens during standardisation. Default is True;
        ignored when ``sanitize=False``.
    sanitize : bool, optional
        Run :func:`safe_standardize_mol` after parsing. Default is True.

    Returns
    -------
    Chem.Mol or None
        The parsed molecule. Failed parsing returns None if sanitisation
        is disabled; otherwise standardisation raises an error.

    Warns
    -----
    UserWarning
        If the SMILES string contains ``'.'`` (disconnected molecules).

    Examples
    --------
    >>> import assemblytheorytools as att
    >>> mol = att.smi_to_mol("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")  # caffeine
    >>> mol.GetNumAtoms()
    24
    >>> round(att.molecular_weight(mol), 2)
    194.19

    Hydrogens are explicit by default; pass ``add_hydrogens=False`` for the
    heavy-atom molecule only:

    >>> att.smi_to_mol("CCO", add_hydrogens=False).GetNumAtoms()
    3
    """
    if "." in smi:
        warnings.warn(
            "Disconnected molecules detected in SMILES string. "
            "Ensure proper handling of these molecules."
        )
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    return _maybe_sanitize(mol, sanitize, add_hydrogens)


def inchi_to_mol(
    inchi: str, add_hydrogens: bool = True, sanitize: bool = True
) -> Chem.Mol:
    """
    Convert an InChI string to an optionally standardised RDKit molecule.

    Parameters
    ----------
    inchi : str
        An InChI string representing the molecular structure.
    add_hydrogens : bool, optional
        Add explicit hydrogens during standardisation. Default is True;
        ignored when ``sanitize=False``.
    sanitize : bool, optional
        Run :func:`safe_standardize_mol` after parsing. Default is True.

    Returns
    -------
    Chem.Mol or None
        The parsed molecule, retaining existing hydrogens. Failed parsing
        returns None if sanitisation is disabled; otherwise standardisation
        raises an error.
    """
    mol = Chem.MolFromInchi(inchi, sanitize=False, removeHs=False)
    return _maybe_sanitize(mol, sanitize, add_hydrogens)


def molfile_to_mol(
    mol: str, add_hydrogens: bool = True, sanitize: bool = True
) -> Chem.Mol:
    """
    Read a mol file into an optionally standardised RDKit molecule.

    Parameters
    ----------
    mol : str
        Path to the mol file.
    add_hydrogens : bool, optional
        Add explicit hydrogens during standardisation. Default is True;
        ignored when ``sanitize=False``.
    sanitize : bool, optional
        Run :func:`safe_standardize_mol` after parsing. Default is True.

    Returns
    -------
    Chem.Mol or None
        The parsed molecule. Failed parsing returns None if sanitisation
        is disabled; otherwise standardisation raises an error. File-reading
        errors propagate from RDKit.
    """
    mol = Chem.MolFromMolFile(mol, sanitize=False)
    return _maybe_sanitize(mol, sanitize, add_hydrogens)


def combine_mols(mols: Union[List[Chem.Mol], Chem.Mol]) -> Chem.Mol:
    """
    Combine a list of molecules as disconnected fragments.

    Parameters
    ----------
    mols : Union[List[Chem.Mol], Chem.Mol]
        Molecules in fragment order, or a single molecule to return as is.

    Returns
    -------
    Chem.Mol
        A combined copy for a list input; otherwise the input itself.
        An empty list produces an empty editable molecule.
    """
    if not isinstance(mols, list):
        return mols

    combined = Chem.RWMol()
    for mol in mols:
        combined = Chem.CombineMols(combined, mol)
    return combined


def split_mols(mol: Chem.Mol) -> tuple[Chem.Mol, ...]:
    """
    Split a molecule into its connected components.

    Parameters
    ----------
    mol : Chem.Mol
        The input RDKit molecule to be split.

    Returns
    -------
    tuple[Chem.Mol, ...]
        Sanitised fragments in RDKit's component order.
    """
    return Chem.GetMolFrags(mol, asMols=True)


def write_v2k_mol_file(mol: Chem.Mol, file_path: str) -> None:
    """
    Write a molecule to a mol file, forcing V2000 format.

    Parameters
    ----------
    mol : Chem.Mol
        The RDKit molecule to be written to the file.
    file_path : str
        The path to the file where the molecule will be written.

    Returns
    -------
    None
    """
    with open(file_path, "w") as mol_file:
        mol_file.write(Chem.MolToV2KMolBlock(mol))


def get_element_set_from_mols(mols: List[Chem.Mol]) -> set:
    """
    Extract unique element symbols from a list of RDKit molecules.

    Parameters
    ----------
    mols : List[Chem.Mol]
        Molecules to inspect. False-valued entries, including None, are
        skipped.

    Returns
    -------
    set
        A set containing unique element symbols found in all molecules.
    """
    return {atom.GetSymbol() for mol in mols if mol for atom in mol.GetAtoms()}


def standardise_smiles(
    smi: str, add_hydrogens: bool = True, sanitize: bool = True
) -> str:
    """
    Return canonical, isomeric SMILES with Kekule bond notation.

    Parameters
    ----------
    smi : str
        A SMILES string representing the molecular structure.
    add_hydrogens : bool, optional
        Run :func:`safe_standardize_mol` and add explicit hydrogens, even
        when ``sanitize=False``. Default is True.
    sanitize : bool, optional
        Sanitise during SMILES parsing. Default is True.

    Returns
    -------
    str
        The canonical SMILES string.

    Raises
    ------
    ValueError
        If parsing fails and ``add_hydrogens=False``. With added hydrogens,
        standardisation errors propagate before this check.
    """
    mol = Chem.MolFromSmiles(smi, sanitize=sanitize)
    if add_hydrogens:
        mol = safe_standardize_mol(mol, add_hydrogens=True)

    if not mol:
        raise ValueError(f"Invalid SMILES: {smi}")

    return Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True, canonical=True)


def smi_remove_implicit_hydrogen(input_string: str) -> str:
    """
    Remove implicit hydrogen counts from SMILES strings.

    Parameters
    ----------
    input_string : str
        A SMILES string containing bracketed atom tokens.

    Returns
    -------
    str
        A modified SMILES string with implicit hydrogen counts removed.

    Notes
    -----
    Bracketed letters followed by optional digits are reduced to their first
    letter: ``[CH3]`` becomes ``[C]``, as does ``[Cl]``. Tokens containing
    charges, isotope prefixes, or stereochemistry are left unchanged.
    """
    return re.sub(r"\[([a-zA-Z])[a-zA-Z]*[0-9]*\]", r"[\1]", input_string)


def peptide_to_smiles(seq: str, *, canonical: bool = True) -> str:
    """
    Convert a peptide sequence to an isomeric SMILES string.

    Parameters
    ----------
    seq : str
        Standard single-letter amino acid codes. Whitespace is removed and
        letters are converted to uppercase before validation.
    canonical : bool, optional
        Whether to generate canonical SMILES. Default is True.

    Returns
    -------
    str
        A SMILES string representing the peptide.

    Raises
    ------
    ValueError
        If the sequence is empty, contains invalid amino acid codes, or
        cannot be built by RDKit.

    Notes
    -----
    Accepts the amino acid codes ``ACDEFGHIKLMNPQRSTVWY`` and uses RDKit's
    ``MolFromFASTA`` to build the peptide.
    """
    sequence = "".join(seq.split()).upper()
    if not sequence:
        raise ValueError("Empty sequence")

    if any(code not in _AMINO_ACIDS for code in sequence):
        raise ValueError(f"Invalid amino acid code(s). Allowed: {_AMINO_ACIDS}")

    mol = Chem.MolFromFASTA(sequence)
    if mol is None:
        raise ValueError("RDKit could not parse/build the peptide from this sequence")

    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=canonical)
