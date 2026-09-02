import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from rdkit import Chem

import assemblytheorytools as att

# set the plot axis
plt.rcParams['axes.linewidth'] = 2.0


def get_ai(smi):
    ai, _, _ = att.calculate_assembly_index(
        att.smi_to_nx(smi), strip_hydrogen=True, exact=True)
    return ai


def get_bertz_complexity(smi):
    return att.bertz_complexity(Chem.MolFromSmiles(smi, sanitize=True))


def get_bottcher_complexity(smi):
    return att.bottcher(Chem.MolFromSmiles(smi, sanitize=True))


def get_wiener_index(smi):
    return att.wiener_index(Chem.MolFromSmiles(smi, sanitize=True))


def get_balaban_index(smi):
    return att.balaban_index(Chem.MolFromSmiles(smi, sanitize=True))


def get_spacial_score(smi):
    return att.spacial_score(Chem.MolFromSmiles(smi, sanitize=True))


def get_proudfoot_complexity(smi):
    return att.proudfoot(Chem.MolFromSmiles(smi, sanitize=True))


def get_mc1(smi):
    return att.mc1(Chem.MolFromSmiles(smi, sanitize=True))


# get_ai's strip_hydrogen=True reduces a molecule to its heavy-atom skeleton;
# Chem.MolFromSmiles keeps hydrogens implicit rather than as explicit nodes, so
# every score below is computed over that same heavy-atom skeleton and stays
# directly comparable to the assembly index.
# Column name -> (axis label, function computing it from a SMILES string).
COMPLEXITY_SCORES = [
    ("assembly_index", "Assembly index", get_ai),
    ("bertz", "Bertz", get_bertz_complexity),
    ("bottcher", "Böttcher", get_bottcher_complexity),
    ("wiener", "Wiener", get_wiener_index),
    ("balaban", "Balaban", get_balaban_index),
    ("spacial_score", "Spacial score", get_spacial_score),
    ("proudfoot", "Proudfoot", get_proudfoot_complexity),
    ("mc1", "MC1", get_mc1),
]


def plot_complexity_matrix(df, columns, fontsize=10, panel_size=2.4):
    # Lower-triangular grid: diagonal is each score's distribution, each
    # off-diagonal panel is column-score vs row-score with Pearson's r
    # annotated. The upper triangle is skipped since it would only mirror
    # the lower one.
    n = len(columns)
    fig, axes = plt.subplots(n, n, figsize=(panel_size * n, panel_size * n))

    for i, (row_col, row_lab) in enumerate(columns):
        y = df[row_col].to_numpy(dtype=float)
        for j, (col_col, col_lab) in enumerate(columns):
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
                continue

            x = df[col_col].to_numpy(dtype=float)
            if i == j:
                ax.hist(x, bins=30, color='black', alpha=0.7)
            else:
                ax.scatter(x, y, s=8, color='black', alpha=0.3, edgecolors='none', rasterized=True)
                r = att.get_r(x, y)
                ax.text(0.05, 0.90, f"r = {r:.2f}", transform=ax.transAxes,
                        fontsize=fontsize - 1, va='top')

            ax.minorticks_on()
            ax.tick_params(axis='both', which='both', direction='in',
                           top=True, right=True, labelsize=fontsize - 2)
            ax.set_xlabel(col_lab if i == n - 1 else "", fontsize=fontsize)
            ax.set_ylabel(row_lab if j == 0 else "", fontsize=fontsize)
            if i != n - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])

    fig.tight_layout()
    return fig, axes


if __name__ == "__main__":
    max_heavy = 15
    data_file_in = "CBRdb_C.csv.zip"
    kegg_data_in_path = os.path.expanduser(os.path.abspath(f"..//..//{data_file_in}"))
    target_url = f'https://raw.githubusercontent.com/ELIFE-ASU/CBRdb/refs/heads/main/{data_file_in}'

    if not os.path.exists(kegg_data_in_path):
        os.system(f"wget {target_url} -O ../../{data_file_in}")
    else:
        print("File already exists, skipping download.")

    df = pd.read_csv(kegg_data_in_path, low_memory=False)
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

    df = df[['compound_id', 'smiles', 'n_heavy_atoms']]

    for col, _, fn in COMPLEXITY_SCORES:
        df[col] = att.mp_calc(fn, df['smiles'])

    # Drop rows where any score failed (assembly index uses -1 as its own
    # failure sentinel, handled separately below)
    score_cols = [col for col, _, _ in COMPLEXITY_SCORES]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=score_cols)
    df = df[df['assembly_index'] >= 0]
    print(f"Number of molecules with all scores computed: {len(df)}")

    # Write the dataframe to a csv file
    df.to_csv("kegg_c_complexity_scores.csv", index=False)
    df.to_csv("kegg_c_complexity_scores.csv.zip", index=False)

    plot_complexity_matrix(df, [(col, lab) for col, lab, _ in COMPLEXITY_SCORES])
    plt.savefig("kegg_c_complexity_matrix.svg", dpi=150)
    plt.show()
