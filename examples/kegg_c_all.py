import os

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem

import assemblytheorytools as att

# set the plot axis
plt.rcParams['axes.linewidth'] = 2.0


def get_ai(smi):
    ai, _, _ = att.calculate_assembly_index(att.smi_to_nx(smi), strip_hydrogen=True)
    return ai


if __name__ == "__main__":
    max_heavy = 30
    data_file_in = "CBRdb_C.csv.zip"
    kegg_data_in_path = os.path.expanduser(os.path.abspath(f"..//..//{data_file_in}"))
    target_url = f'https://raw.githubusercontent.com/ELIFE-ASU/CBRdb/refs/heads/main/{data_file_in}'

    if not os.path.exists(kegg_data_in_path):
        os.system(f"wget {target_url} -O ../../{data_file_in}")
    else:
        print("File already exists, skipping download.")

    df = pd.read_csv(kegg_data_in_path, low_memory=False)
    # Only select the compound_id, smiles, n_heavy_atoms, n_chiral_centers columns
    df = df[['compound_id', 'smiles', 'n_heavy_atoms']]
    # remove duplicates based on the smiles column
    df = df.drop_duplicates(subset=['smiles'])
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
    print(f"Number of molecules: {len(df)}")

    df['assembly_index'] = att.mp_calc(get_ai, df['smiles'])
    df = df.dropna(subset=['assembly_index'])
    df = df[df['assembly_index'] >= 0]

    att.scatter_plot(df['n_heavy_atoms'],
                     df['assembly_index'],
                     xlab="Heavy atom count",
                     ylab="Assembly index")
    plt.show()
