from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def safe_standardize_mol(mol):
    """
    Standardize the given RDKit molecule.

    Args:
        mol (Chem.Mol): The input RDKit molecule to be standardized.

    Returns:
        Chem.Mol: The standardized RDKit molecule.
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
    # Add explicit hydrogens to the molecule
    mol = Chem.AddHs(mol)
    return mol


def standardize_mol(mol):
    """
    Standardize the given RDKit molecule.

    Args:
        mol (Chem.Mol): The input RDKit molecule to be standardized.

    Returns:
        Chem.Mol: The standardized RDKit molecule.
    """
    # Sanitise the molecule
    Chem.SanitizeMol(mol, catchErrors=False)
    rdMolStandardize.NormalizeInPlace(mol)
    # Update the properties
    mol.UpdatePropertyCache(strict=False)
    Chem.Kekulize(mol)
    # Add hydrogens
    mol = Chem.AddHs(mol)
    # Return the molecule
    return mol


def smi_to_mol(smi):
    """
    Convert a SMILES string to a standardized RDKit molecule.

    Args:
        smi (str): The SMILES string representing the molecule.

    Returns:
        Chem.Mol: The standardized RDKit molecule.
    """
    mol = Chem.MolFromSmiles(smi)
    # Sanitise the molecule
    return standardize_mol(mol)


def inchi_to_mol(inchi):
    """
    Convert an InChI string to a standardized RDKit molecule.

    Args:
        inchi (str): The InChI string representing the molecule.

    Returns:
        Chem.Mol: The standardized RDKit molecule.
    """
    mol = Chem.MolFromInchi(inchi)
    # Sanitise the molecule
    return standardize_mol(mol)


def molfile_to_mol(mol):
    """
    Convert a Molfile to a standardized RDKit molecule.

    Args:
        mol (str): The path to the Molfile representing the molecule.

    Returns:
        Chem.Mol: The standardized RDKit molecule.
    """
    mol = Chem.MolFromMolFile(mol)
    # Sanitise the molecule
    return standardize_mol(mol)


def combine_mols(mols):
    """
    Combine multiple RDKit molecules into a single molecule.

    Args:
        mols (list or Chem.Mol): A list of RDKit molecules to be combined or a single RDKit molecule.

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


def split_mols(mol):
    """
    Split an RDKit molecule into its individual fragments.

    Args:
        mol (Chem.Mol): The input RDKit molecule to be split.

    Returns:
        tuple: A tuple of RDKit molecule fragments.
    """
    return Chem.GetMolFrags(mol, asMols=True)


def write_v2k_mol_file(mol, file_path):
    """
    Write an RDKit molecule to a file in V2K Mol block format.

    Args:
        mol (Chem.Mol): The RDKit molecule to be written to the file.
        file_path (str): The path to the file where the molecule will be written.
    """
    # Need to force rdkit to use V2k mol block format
    with open(file_path, "w") as f:
        f.write(Chem.MolToV2KMolBlock(mol))
