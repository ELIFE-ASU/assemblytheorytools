import os
import sys

import pandas as pd
from rdkit import Chem
import assemblytheorytools as att

if __name__ == "__main__":
    max_heavy = 50
    timeout = 59.0 * 60.0 * 4  # Just shy of 4 hours
    data_dir = os.path.abspath(r"/scratch/lslocomb/mol_data/")
    in_file = "CBRdb_C.csv.zip"
    at_out_file = in_file.split(".")[0] + "_at.csv"
    # Load the data
    df = pd.read_csv(os.path.join(data_dir, in_file), low_memory=False)
    # Make subselction to speed it up
    df = df[['compound_id', 'smiles', 'n_heavy_atoms']]
    # Remove . in the smiles column
    df = df[~df['smiles'].str.contains(r"\.")]
    # Remove * in the smiles column
    df = df[~df['smiles'].str.contains(r"\*")]
    # Remove diative bonds in the smiles column
    df = df[~df['smiles'].str.contains(r"\->")]
    # Only select the cases where there are less than n heavy atoms
    df = df[df['n_heavy_atoms'] <= max_heavy]
    # Only select the cases where there are more than 2 heavy atoms
    df = df[df['n_heavy_atoms'] >= 2]
    # Remove rows which cannot be parsed by rdkit
    df = df[df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)]
    # Remove all the rows that will not sanitize
    df = df[df['smiles'].apply(
        lambda x: Chem.SanitizeMol(Chem.MolFromSmiles(x), catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE)]
    # remove nan values
    df = df.dropna(subset=['smiles'])
    # Get the smiles
    smiles = df["smiles"].values
    # Get the ID
    mol_id = df["compound_id"].values

    # Running AT in parallel
    i = int(sys.argv[1])
    # Convert the molecule to a mol object
    mol = att.smi_to_mol(smiles[i])
    # Calculate the assembly index
    at, virt_obj, _ = att.calculate_assembly_index(mol, timeout=timeout, strip_hydrogen=True)
    # Write the data to the shared file
    att.write_to_shared_file(f"{i}, {mol_id[i]}, {at}, {virt_obj}\n", at_out_file)
