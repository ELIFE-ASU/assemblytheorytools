import os
import re
import subprocess
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem

exec_path = "/home/mshahjah/assembly-theory/target/release/assembly-theory"  # Path to executable

def smiles_to_molfile(smiles, molfile_path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    has_wildcard = any(atom.GetSymbol() == '*' for atom in mol.GetAtoms())
    if not has_wildcard:
        ret = AllChem.EmbedMolecule(mol)
        if ret != 0:
            return False
    with open(molfile_path, 'w') as f:
        f.write(Chem.MolToMolBlock(mol))
    return True

def calculate_rust_ai(smiles, exec_path):
    if not smiles:
        return "No SMILES provided."
    with tempfile.NamedTemporaryFile(suffix='.mol', delete=False) as tmp_file:
        tmp_molfile = tmp_file.name
    try:
        if not smiles_to_molfile(smiles, tmp_molfile):
            return f"Invalid SMILES: {smiles}"
        result = subprocess.run([exec_path, tmp_molfile], capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return f"Error processing {smiles}: {result.stderr.strip()}"
        match = re.search(r'\b\d+\b', result.stdout)
        rust_ai = match.group(0) if match else result.stdout.strip()
        return rust_ai
    except Exception as e:
        return f"Exception: {e}"
    finally:
        if os.path.exists(tmp_molfile):
            os.remove(tmp_molfile)

# You can change the SMILES string here or use input()
smiles = input("Enter SMILES: ").strip()
rust_ai = calculate_rust_ai(smiles, exec_path)
print(f"Result: {rust_ai}")