import re
import warnings
from typing import List, Union

import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdchem import GetPeriodicTable


def safe_standardize_mol(mol: Chem.Mol, add_hydrogens: bool = True) -> Chem.Mol:
    """
    Standardise the given RDKit molecule with additional safety checks.

    Args:
        mol (rdkit.Chem.Mol): The input RDKit molecule to be standardised.
        add_hydrogens (bool, optional): Whether to add hydrogens to the molecule. Default is True.

    Returns:
        rdkit.Chem.Mol: The standardised RDKit molecule.
    """
    # Update the molecule's property cache without strict checking
    mol.UpdatePropertyCache(strict=False)
    # Set conjugation and hybridisation states
    Chem.SetConjugation(mol)
    Chem.SetHybridization(mol)
    # Normalise the molecule, excluding clean-up and property sanitisation
    Chem.SanitizeMol(mol,
                     sanitizeOps=(Chem.SANITIZE_ALL ^ Chem.SANITIZE_CLEANUP ^ Chem.SANITIZE_PROPERTIES),
                     catchErrors=False)
    # Normalise the molecule in place using RDKit's MolStandardize
    rdMolStandardize.NormalizeInPlace(mol)
    # Kekulise the molecule (convert aromatic bonds to alternating single and double bonds)
    Chem.Kekulize(mol)
    if add_hydrogens:
        # Add hydrogens
        mol = Chem.AddHs(mol)
    return mol


def standardize_mol(mol: Chem.Mol, add_hydrogens: bool = True) -> Chem.Mol:
    """
    Standardize the given RDKit molecule.

    Args:
        mol (Chem.Mol): The input RDKit molecule to be standardised.
        add_hydrogens (bool, optional): Whether to add hydrogens to the molecule. Default is True.

    Returns:
        Chem.Mol: The standardized RDKit molecule.
    """
    # Sanitise the molecule
    Chem.SanitizeMol(mol, catchErrors=False)
    # Normalise the molecule in place using RDKit's MolStandardize
    rdMolStandardize.NormalizeInPlace(mol)
    # Update the molecule's property cache without strict checking
    mol.UpdatePropertyCache(strict=True)
    # Kekulise the molecule (convert aromatic bonds to alternating single and double bonds)
    Chem.Kekulize(mol)
    if add_hydrogens:
        # Add hydrogens
        mol = Chem.AddHs(mol)
    # Return the molecule
    return mol


def get_free_valence(atom: Chem.Atom,
                     pt: Chem.rdchem.PeriodicTable = None,
                     method: str = '1') -> int:
    """
    Calculate the free valence of an atom based on the specified method.

    Parameters:
    -----------
    atom : Chem.Atom
        The RDKit atom object for which the free valence is calculated.
    pt : Chem.rdchem.PeriodicTable, optional
        The RDKit periodic table object. If not provided, it defaults to the global periodic table.
    method : str, optional
        The method used to calculate the free valence. Options are:
        - '1': Uses the minimum valence from the periodic table's valence list minus the atom's degree.
        - '2': Uses the default valence minus the atom's explicit valence.
        - '3': Uses the number of outer electrons of the atom.

    Returns:
    --------
    int
        The calculated free valence of the atom.

    Raises:
    -------
    ValueError
        If an unknown method is provided.

    Notes:
    ------
    - The free valence is calculated differently depending on the method specified.
    - The periodic table is used to retrieve atomic properties such as valence and outer electrons.
    """
    pt = pt or GetPeriodicTable()
    symbol = atom.GetSymbol()
    atomic_number = pt.GetAtomicNumber(symbol)

    if method == '1':
        return min(pt.GetValenceList(atomic_number)) - int(
            sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]))
    elif method == '2':
        return pt.GetDefaultValence(symbol) - atom.GetExplicitValence()
    elif method == '3':
        return pt.GetNOuterElecs(atomic_number)
    else:
        raise ValueError(f"Unknown method {method} for calculating free valence of atom {symbol}")


def reset_mol_charge(mol: Chem.Mol,
                     pt: Chem.rdchem.PeriodicTable = None) -> Chem.Mol:
    """
    Adjusts the formal charges of atoms in a molecule to match their free valence.

    Parameters:
    -----------
    mol : Chem.Mol
        An RDKit molecule object whose atom charges need to be reset.
    pt : Chem.rdchem.PeriodicTable, optional
        The RDKit periodic table object. If not provided, it defaults to the global periodic table.

    Returns:
    --------
    Chem.Mol
        A new RDKit molecule object with updated formal charges.

    Notes:
    ------
    - The function iterates over all atoms in the molecule and sets their formal charge
      based on the free valence calculated using the `get_free_valence` function.
    - The molecule's property cache is updated after modifying the charges.
    """
    pt = pt or GetPeriodicTable()
    rw = Chem.RWMol(mol)

    for atom in rw.GetAtoms():
        atom.SetFormalCharge(get_free_valence(atom, pt=pt))

    return rw.GetMol()


def get_total_free_valence(mol: Chem.Mol | nx.Graph,
                           pt: Chem.rdchem.PeriodicTable = None,
                           method: str = '1') -> int:
    if isinstance(mol, nx.Graph):
        pt = pt or GetPeriodicTable()

        # loop through all nodes in the graph
        free = 0
        for node in mol.nodes:
            symbol = mol.nodes[node].get('color')
            atomic_number = pt.GetAtomicNumber(symbol)

            edge_total = 0
            for edge in mol.edges(node):
                edge_total += int(mol.edges[edge].get('color'))

            free += min(pt.GetValenceList(atomic_number)) - edge_total
    else:
        free = 0
        pt = pt or GetPeriodicTable()
        for atom in mol.GetAtoms():
            free += get_free_valence(atom, pt=pt, method=method)
    return free


def smi_to_mol(smi: str, add_hydrogens: bool = True, sanitize: bool = True) -> Chem.Mol:
    """
    Convert a SMILES string to an RDKit molecule object.

    This function takes a SMILES (Simplified Molecular Input Line Entry System) string
    and converts it into an RDKit molecule object. Optionally, the molecule can be
    sanitized and explicit hydrogens can be added.

    Parameters:
    -----------
    smi : str
        A SMILES string representing the molecular structure.
    add_hydrogens : bool, optional
        If True, adds explicit hydrogens to the molecule during sanitization. Defaults to True.
    sanitize : bool, optional
        If True, sanitizes the molecule after conversion. Defaults to True.

    Returns:
    --------
    Chem.Mol
        An RDKit molecule object representing the input SMILES string.

    Raises:
    -------
    Warning
        If the SMILES string contains disconnected molecules (indicated by a '.' character).

    Notes:
    ------
    - The function uses RDKit's `MolFromSmiles` to create the molecule object.
    - Sanitization ensures the molecule is chemically valid and standardized.
    - Disconnected molecules in the SMILES string are flagged with a warning.
    """
    if '.' in smi:
        warnings.warn("Disconnected molecules detected in SMILES string. Ensure proper handling of these molecules.")
    # Convert the SMILES string to an RDKit molecule object
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    # Sanitise the molecule
    if sanitize:
        return safe_standardize_mol(mol, add_hydrogens=add_hydrogens)
    else:
        return mol


def inchi_to_mol(inchi: str, add_hydrogens: bool = True, sanitize: bool = True) -> Chem.Mol:
    """
    Convert an InChI string to an RDKit molecule object.

    This function takes an InChI (International Chemical Identifier) string
    and converts it into an RDKit molecule object. Optionally, the molecule
    can be sanitized and explicit hydrogens can be added.

    Parameters:
    -----------
    inchi : str
        An InChI string representing the molecular structure.
    add_hydrogens : bool, optional
        If True, adds explicit hydrogens to the molecule during sanitization. Defaults to True.
    sanitize : bool, optional
        If True, sanitizes the molecule after conversion. Defaults to True.

    Returns:
    --------
    Chem.Mol
        An RDKit molecule object representing the input InChI string.

    Raises:
    -------
    ValueError
        If the InChI string is invalid or the molecule could not be created.

    Notes:
    ------
    - The function uses RDKit's `MolFromInchi` to create the molecule object.
    - Sanitization ensures the molecule is chemically valid and standardized.
    """
    mol = Chem.MolFromInchi(inchi, sanitize=False, removeHs=False)
    # Sanitise the molecule
    if sanitize:
        return safe_standardize_mol(mol, add_hydrogens=add_hydrogens)
    else:
        return mol


def molfile_to_mol(mol: str, add_hydrogens: bool = True, sanitize: bool = True) -> Chem.Mol:
    """
    Convert a Molfile to an RDKit molecule object.

    This function takes a Molfile (a file format for storing molecular structures),
    reads it, and converts it into an RDKit molecule object. Optionally, the molecule
    can be sanitized and explicit hydrogens can be added.

    Parameters:
    -----------
    mol : str
        The path to the Molfile containing the molecular structure.
    add_hydrogens : bool, optional
        If True, adds explicit hydrogens to the molecule during sanitization. Defaults to True.
    sanitize : bool, optional
        If True, sanitizes the molecule after conversion. Defaults to True.

    Returns:
    --------
    Chem.Mol
        An RDKit molecule object representing the input Molfile.

    Raises:
    -------
    ValueError
        If the Molfile is invalid or the molecule could not be created.

    Notes:
    ------
    - The function uses RDKit's `MolFromMolFile` to create the molecule object.
    - Sanitization ensures the molecule is chemically valid and standardized.
    """
    # Convert the Molfile to an RDKit molecule
    mol = Chem.MolFromMolFile(mol, sanitize=False)

    # Sanitise the molecule
    if sanitize:
        return safe_standardize_mol(mol, add_hydrogens=add_hydrogens)
    else:
        return mol


def combine_mols(mols: Union[List[Chem.Mol], Chem.Mol]) -> Chem.Mol:
    """
    Combine multiple RDKit molecules into a single molecule.

    Args:
        mols (Union[List[Chem.Mol], Chem.Mol]): A list of RDKit molecules to be combined or a single RDKit molecule.

    Returns:
        Chem.Mol: The combined RDKit molecule if input is a list, otherwise returns the input molecule.
    """
    if isinstance(mols, list):
        combined_mol = Chem.RWMol()
        for mol in mols:
            combined_mol = Chem.CombineMols(combined_mol, mol)
        return combined_mol
    else:
        return mols


def split_mols(mol: Chem.Mol) -> tuple[Chem.Mol, ...]:
    """
    Split an RDKit molecule into its individual components.

    Args:
        mol (Chem.Mol): The input RDKit molecule to be split.

    Returns:
        tuple[Chem.Mol, ...]: A tuple of RDKit molecule fragments.
    """
    return Chem.GetMolFrags(mol, asMols=True)


def write_v2k_mol_file(mol: Chem.Mol, file_path: str) -> None:
    """
    Write an RDKit molecule to a file in V2K Mol block format.

    Args:
        mol (Chem.Mol): The RDKit molecule to be written to the file.
        file_path (str): The path to the file where the molecule will be written.
    """
    # Need to force rdkit to use V2k mol block format
    with open(file_path, "w") as f:
        f.write(Chem.MolToV2KMolBlock(mol))
    return None


def get_element_set_from_mols(mols: List[Chem.Mol]) -> set:
    element_set = set()
    for mol in mols:
        if mol:
            for atom in mol.GetAtoms():
                element_set.add(atom.GetSymbol())
    return element_set


def standardise_smiles(smi: str, add_hydrogens: bool = True, sanitize: bool = True) -> str:
    """
    Standardize a SMILES string.

    This function converts a SMILES (Simplified Molecular Input Line Entry System) string
    into a standardized format using RDKit. Optionally, explicit hydrogens can be added
    and the molecule can be sanitized.

    Parameters:
    -----------
    smi : str
        A SMILES string representing the molecular structure.
    add_hydrogens : bool, optional
        If True, adds explicit hydrogens to the molecule during standardization. Defaults to True.
    sanitize : bool, optional
        If True, sanitizes the molecule during conversion. Defaults to True.

    Returns:
    --------
    str
        A standardized SMILES string with isomeric, kekule, and canonical options enabled.

    Raises:
    -------
    ValueError
        If the SMILES string is invalid or the molecule could not be created.

    Notes:
    ------
    - The function uses RDKit's `MolFromSmiles` to create the molecule object.
    - Sanitization ensures the molecule is chemically valid and standardized.
    - The standardized SMILES string is returned with specified options.
    """
    # Convert the SMILES string to an RDKit molecule object and add explicit hydrogens
    mol = Chem.MolFromSmiles(smi, sanitize=sanitize)
    if add_hydrogens:
        # Standardise the molecule with hydrogens
        mol = safe_standardize_mol(mol, add_hydrogens=True)

    # Raise an error if the molecule could not be created
    if not mol:
        raise ValueError(f"Invalid SMILES: {smi}")

    # Return the standardised SMILES string with specified options
    return Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True, canonical=True)


def smi_remove_implicit_hydrogen(input_string: str) -> str:
    """
    Remove implicit hydrogen counts from SMILES strings.

    This function processes a SMILES (Simplified Molecular Input Line Entry System) string
    and removes implicit hydrogen counts from atomic symbols enclosed in square brackets.

    Parameters:
    -----------
    input_string : str
        A SMILES string containing atomic symbols with implicit hydrogen counts.

    Returns:
    --------
    str
        A modified SMILES string with implicit hydrogen counts removed.

    Notes:
    ------
    - The function uses a regular expression to identify atomic symbols with implicit hydrogen counts.
    - Only the atomic symbol is retained, and the hydrogen count is removed.
    """
    pattern = r'\[([a-zA-Z]+[0-9]*)\]'

    def update_match(match):
        group = match.group(1)
        return f"[{group[0]}]"

    return re.sub(pattern, update_match, input_string)
