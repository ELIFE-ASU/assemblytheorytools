from typing import List, Union

from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


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


def smi_to_mol(smi: str, add_hydrogens: bool = True, sanitize: bool = True) -> Chem.Mol:
    if '.' in smi:
        print("You have ionic molecules in your set, make sure you handle them appropriately. "
              "Have a look at the create_ionic_molecule function in tools_graphs.py", flush=True)
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    # Sanitise the molecule
    if sanitize:
        return safe_standardize_mol(mol, add_hydrogens=add_hydrogens)
    else:
        return mol


def inchi_to_mol(inchi: str, add_hydrogens: bool = True, sanitize: bool = True) -> Chem.Mol:
    mol = Chem.MolFromInchi(inchi, sanitize=False)
    # Sanitise the molecule
    if sanitize:
        return safe_standardize_mol(mol, add_hydrogens=add_hydrogens)
    else:
        return mol


def molfile_to_mol(mol: str, add_hydrogens: bool = True, sanitize: bool = True) -> Chem.Mol:
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
