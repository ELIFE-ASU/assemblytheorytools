from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


# def standardize_mol(mol):
#     # Standardize the molecule
#     mol.UpdatePropertyCache(strict=False)
#     Chem.SetConjugation(mol)
#     Chem.SetHybridization(mol)
#     # Normalize the molecule
#     Chem.SanitizeMol(mol, sanitizeOps=(Chem.SANITIZE_ALL ^ Chem.SANITIZE_CLEANUP ^ Chem.SANITIZE_PROPERTIES), catchErrors=False)
#     rdMolStandardize.NormalizeInPlace(mol)
#     # kekulize the molecule
#     Chem.Kekulize(mol)
#     Chem.AddHs(mol)

def standardize_mol(mol):
    # Sanitise the molecule
    Chem.SanitizeMol(mol, catchErrors=False)
    rdMolStandardize.NormalizeInPlace(mol)
    # Update the properties
    mol.UpdatePropertyCache()
    Chem.Kekulize(mol)
    # Add hydrogens
    mol = Chem.AddHs(mol)
    # Return the molecule
    return mol

# def standardize_mol(mol):
#     Chem.SetConjugation(mol)
#     Chem.SetHybridization(mol)
#     Chem.SanitizeMol(mol, sanitizeOps=(Chem.SANITIZE_ALL ^ Chem.SANITIZE_CLEANUP ^ Chem.SANITIZE_PROPERTIES))
#     #Chem.SanitizeMol(mol, catchErrors=False)
#     # Normalize the molecule
#     rdMolStandardize.NormalizeInPlace(mol)
#     mol.UpdatePropertyCache(strict=False)
#     # kekulize the molecule
#     # Chem.Kekulize(mol)
#     # mol = Chem.AddHs(mol)
#     return mol


def smi_to_mol(smi):
    mol = Chem.MolFromSmiles(smi)
    # Sanitise the molecule
    return standardize_mol(mol)


def inchi_to_mol(inchi):
    mol = Chem.MolFromInchi(inchi)
    # Sanitise the molecule
    return standardize_mol(mol)


def molfile_to_mol(mol):
    mol = Chem.MolFromMolFile(mol)
    # Sanitise the molecule
    return standardize_mol(mol)


def combine_mols(mols):
    combined_mol = Chem.RWMol()
    for mol in mols:
        combined_mol = Chem.CombineMols(combined_mol, mol)
    return combined_mol


def write_v2k_mol_file(mol, file_path):
    # Need to force rdkit to use V2k mol block format
    with open(file_path, "w") as f:
        f.write(Chem.MolToV2KMolBlock(mol))
