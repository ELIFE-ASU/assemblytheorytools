import re
import subprocess
import tempfile

from rdkit import Chem


def _input_helper(mol: Chem.Mol,
                  file_path: str,
                  add_hydrogens: bool = True) -> bool:
    if mol is None:
        return False
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)
    # Check for wildcard atoms (e.g., '*')
    has_wildcard = any(atom.GetSymbol() == '*' for atom in mol.GetAtoms())
    # If there are wildcard atoms, we cannot write a valid molfile as not supported
    if has_wildcard:
        return False
    with open(file_path, 'w') as f:
        f.write(Chem.MolToMolBlock(mol))
    return True


def calculate_rust_ai(mol: Chem.Mol,
                      exec_path: str | None = None,
                      timeout: int = 300,
                      add_hydrogens: bool = False) -> int:
    # Input validation
    if exec_path is None:
        exec_path = "/home/mshahjah/assembly-theory/target/release/assembly-theory"
    if not mol:
        print("No input provided.", flush=True)
        return -1

    with tempfile.NamedTemporaryFile(suffix='.mol', delete=True) as tmp_file:
        tmp_molfile = tmp_file.name

        if not _input_helper(mol, tmp_molfile, add_hydrogens=add_hydrogens):
            print(f"Invalid input", flush=True)
            return -1

        try:
            result = subprocess.run([exec_path, tmp_molfile], capture_output=True, text=True, timeout=timeout)

            if result.returncode != 0:
                print(f"Error processing {mol}: {result.stderr.strip()}", flush=True)
                return -1

            match = re.search(r'\b\d+\b', result.stdout)
            return int(match.group(0) if match else result.stdout.strip())
        except Exception as e:
            print(f"Exception occurred: {e}", flush=True)
            return -1
