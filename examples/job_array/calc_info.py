import fcntl
import os
import sys
import assemblytheorytools as att
import pandas as pd
from rdkit.Chem import AllChem as Chem


def write_to_shared_file(message, shared_file):
    with open(shared_file, 'a') as f:
        # Acquire an exclusive lock before writing
        fcntl.flock(f, fcntl.LOCK_EX)
        # Write the message to the file
        f.write(message)
        # Release the lock after writing
        fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    timeout = 86400.0  # 24 hours
    data_dir = os.path.abspath(r"/scratch/lslocomb/mol_data/patent/")
    in_file = "SureChEMBL_map_20240101.csv.zip"
    at_out_file = in_file.split(".")[0] + "_at.csv"
    # Load the data
    df = pd.read_csv(os.path.join(data_dir, in_file))
    # Get the smiles
    smiles = df["smiles"].values
    # Get the names
    mol_id = df["cid"].values

    # Running AT in parallel
    i = int(sys.argv[1])
    # Convert the molecule to a mol object
    mol = att.safe_standardize_mol(Chem.MolFromSmiles(smiles[i]))
    # Calculate the assembly index
    at, path = att.calculate_assembly_index(mol, timeout=timeout)
    # convert the path into absolute smiles strings
    path = att.get_mol_pathway_to_smi(path)
    # Write the data to the shared file
    write_to_shared_file(f"{i}, {mol_id[i]}, {at}, {path}\n", at_out_file)
