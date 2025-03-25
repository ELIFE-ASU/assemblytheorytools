from typing import List, Union

from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def safe_standardize_mol(mol: Chem.Mol, add_hydrogens: bool = True) -> Chem.Mol:
    """
    Standardize the given RDKit molecule with additional safety checks.

    Args:
        mol (rdkit.Chem.Mol): The input RDKit molecule to be standardized.
        add_hydrogens (bool, optional): Whether to add hydrogens to the molecule. Default is True.

    Returns:
        rdkit.Chem.Mol: The standardized RDKit molecule.
    """
    # Update the molecule's property cache without strict checking
    mol.UpdatePropertyCache(strict=False)
    # Set conjugation and hybridization states
    Chem.SetConjugation(mol)
    Chem.SetHybridization(mol)
    # Normalize the molecule, excluding cleanup and property sanitization
    Chem.SanitizeMol(mol, sanitizeOps=(Chem.SANITIZE_ALL ^ Chem.SANITIZE_CLEANUP ^ Chem.SANITIZE_PROPERTIES),
                     catchErrors=False)
    # Normalize the molecule in place using RDKit's MolStandardize
    rdMolStandardize.NormalizeInPlace(mol)
    # Kekulize the molecule (convert aromatic bonds to alternating single and double bonds)
    Chem.Kekulize(mol)
    if add_hydrogens:
        # Add hydrogens
        mol = Chem.AddHs(mol)
    return mol


def standardize_mol(mol: Chem.Mol, add_hydrogens: bool = True) -> Chem.Mol:
    """
    Standardize the given RDKit molecule.

    Args:
        mol (Chem.Mol): The input RDKit molecule to be standardized.
        add_hydrogens (bool, optional): Whether to add hydrogens to the molecule. Default is True.

    Returns:
        Chem.Mol: The standardized RDKit molecule.
    """
    # Sanitise the molecule
    Chem.SanitizeMol(mol, catchErrors=False)
    # Normalize the molecule in place using RDKit's MolStandardize
    rdMolStandardize.NormalizeInPlace(mol)
    # Update the molecule's property cache without strict checking
    mol.UpdatePropertyCache(strict=True)
    # Kekulize the molecule (convert aromatic bonds to alternating single and double bonds)
    Chem.Kekulize(mol)
    if add_hydrogens:
        # Add hydrogens
        mol = Chem.AddHs(mol)
    # Return the molecule
    return mol


def smi_to_mol(smi: str, add_hydrogens: bool = True, safe_sanitise: bool = False) -> Chem.Mol:
    """
    Convert a SMILES string to a standardized RDKit molecule.

    Args:
        smi (str): The SMILES string representing the molecule.
        add_hydrogens (bool, optional): Whether to add hydrogens to the molecule. Default is True.
        safe_sanitise (bool, optional): Whether to use safe sanitization. Default is False.

    Returns:
        Chem.Mol: The standardized RDKit molecule.
    """
    mol = Chem.MolFromSmiles(smi)
    # Sanitise the molecule
    if safe_sanitise:
        return safe_standardize_mol(mol, add_hydrogens=add_hydrogens)
    else:
        return standardize_mol(mol, add_hydrogens=add_hydrogens)


def inchi_to_mol(inchi: str, add_hydrogens: bool = True, safe_sanitise: bool = False) -> Chem.Mol:
    """
    Convert an InChI string to a standardized RDKit molecule.

    Args:
        inchi (str): The InChI string representing the molecule.
        add_hydrogens (bool, optional): Whether to add hydrogens to the molecule. Default is True.
        safe_sanitise (bool, optional): Whether to use safe sanitization. Default is False.

    Returns:
        Chem.Mol: The standardized RDKit molecule.
    """
    mol = Chem.MolFromInchi(inchi)
    # Sanitise the molecule
    if safe_sanitise:
        return safe_standardize_mol(mol, add_hydrogens=add_hydrogens)
    else:
        return standardize_mol(mol, add_hydrogens=add_hydrogens)


def molfile_to_mol(mol: str, add_hydrogens: bool = True, safe_sanitise: bool = False) -> Chem.Mol:
    """
    Convert a Molfile to a standardized RDKit molecule.

    Args:
        mol (str): The path to the Molfile representing the molecule.
        add_hydrogens (bool, optional): Whether to add hydrogens to the molecule. Default is True.
        safe_sanitise (bool, optional): Whether to use safe sanitization. Default is False.

    Returns:
        Chem.Mol: The standardized RDKit molecule.
    """
    # Convert the Molfile to an RDKit molecule
    mol = Chem.MolFromMolFile(mol)
    # Standardize the molecule
    if safe_sanitise:
        return safe_standardize_mol(mol, add_hydrogens=add_hydrogens)
    else:
        return standardize_mol(mol, add_hydrogens=add_hydrogens)


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


def get_element_set_from_mols(mols):
    """
    Determine the set of unique elements present in a list of RDKit molecules.

    Parameters:
    -----------
    mols : list
        List of RDKit molecule objects.

    Returns:
    --------
    set
        Set of element symbols (strings) present in the molecules.
    """
    element_set = set()
    for mol in mols:
        if mol:
            for atom in mol.GetAtoms():
                element_set.add(atom.GetSymbol())
    return element_set