import os
import sys

import pandas as pd

import assemblytheorytools as att

if __name__ == "__main__":
    timeout = 59.0 * 60.0 * 4  # Just shy of 4 hours
    data_dir = os.path.abspath(r"/scratch/lslocomb/mol_data/")
    in_file = "CBRdb_C.csv.zip"
    at_out_file = in_file.split(".")[0] + "_at.csv"
    # Load the data
    df = pd.read_csv(os.path.join(data_dir, in_file), low_memory=False)
    # Get the smiles
    smiles = df["smiles"].values
    # Get the ID
    mol_id = df["id"].values

    # Running AT in parallel
    i = int(sys.argv[1])
    # Convert the molecule to a mol object
    mol = att.smi_to_mol(smiles[i])
    # Calculate the assembly index
    at, virt_obj, _ = att.calculate_assembly_index(mol, timeout=timeout, strip_hydrogen=True)
    # Write the data to the shared file
    att.write_to_shared_file(f"{i}, {mol_id[i]}, {at}, {virt_obj}\n", at_out_file)
