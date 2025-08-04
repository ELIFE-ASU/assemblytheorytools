import re
import subprocess
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem


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


def calculate_rust_ai(smiles, exec_path=None, timeout=300):
    if exec_path is None:
        exec_path = "/home/mshahjah/assembly-theory/target/release/assembly-theory"
    if not smiles:
        print("No SMILES provided.", flush=True)
        return -1

    with tempfile.NamedTemporaryFile(suffix='.mol', delete=True) as tmp_file:
        tmp_molfile = tmp_file.name

        if not smiles_to_molfile(smiles, tmp_molfile):
            print(f"Invalid SMILES: {smiles}", flush=True)
            return -1

        try:
            result = subprocess.run([exec_path, tmp_molfile], capture_output=True, text=True, timeout=timeout)

            if result.returncode != 0:
                print(f"Error processing {smiles}: {result.stderr.strip()}", flush=True)
                return -1

            match = re.search(r'\b\d+\b', result.stdout)
            return match.group(0) if match else result.stdout.strip()
        except Exception as e:
            print(f"Exception occurred: {e}", flush=True)
            return -1
