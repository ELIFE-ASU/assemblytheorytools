import re
import subprocess
import tempfile

from rdkit import Chem


def _input_helper(mol, molfile_path, add_hydrogens=True):
    if mol is None:
        return False
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    # Check for wildcard atoms (e.g., '*')
    has_wildcard = any(atom.GetSymbol() == '*' for atom in mol.GetAtoms())
    # If there are wildcard atoms, we cannot write a valid molfile as not supported
    if has_wildcard:
        return False
    with open(molfile_path, 'w') as f:
        f.write(Chem.MolToMolBlock(mol))
    return True


def calculate_rust_ai(mol, exec_path=None, timeout=300):
    # Input validation
    if exec_path is None:
        exec_path = "/home/mshahjah/assembly-theory/target/release/assembly-theory"
    if not mol:
        print("No input provided.", flush=True)
        return -1

    with tempfile.NamedTemporaryFile(suffix='.mol', delete=True) as tmp_file:
        tmp_molfile = tmp_file.name

        if not _input_helper(mol, tmp_molfile):
            print(f"Invalid input", flush=True)
            return -1

        try:
            result = subprocess.run([exec_path, tmp_molfile], capture_output=True, text=True, timeout=timeout)

            if result.returncode != 0:
                print(f"Error processing {mol}: {result.stderr.strip()}", flush=True)
                return -1

            match = re.search(r'\b\d+\b', result.stdout)
            return match.group(0) if match else result.stdout.strip()
        except Exception as e:
            print(f"Exception occurred: {e}", flush=True)
            return -1
