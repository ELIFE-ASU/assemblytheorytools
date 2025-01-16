import os
import matplotlib.pyplot as plt
import networkx as nx

import assemblytheorytools as att
import numpy as np
import pandas as pd

from rdkit import Chem

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# set the plot axis
plt.rcParams['axes.linewidth'] = 2.0


if __name__ == "__main__":
    kegg_data_path = os.path.expanduser(os.path.abspath("..//..//kegg_data_C.csv.zip"))

    kegg_data = pd.read_csv(kegg_data_path)
    # Print the size
    print(kegg_data.shape,flush=True)
    print(kegg_data.columns,flush=True)

    # sort by the number of heavy atoms
    kegg_data = kegg_data.sort_values(by="n_heavy_atoms")

    # only select the data that has less than 20 heavy atoms
    kegg_data = kegg_data[kegg_data["n_heavy_atoms"]<10]

    # Get the smiles
    smi = kegg_data["smiles"]
    n_heavy_atoms = kegg_data["n_heavy_atoms"].tolist()
    array_ai = []
    array_heavy_atoms = []

    for i, s in enumerate(smi):
        print(i, s,flush=True)
        # Convert all the smile to mol

        mol = Chem.MolFromSmiles(s, sanitize=False)

        # convert to a networkx graph
        graph = att.mol_to_nx(mol)

        # mol = att.smi_to_mol(s)
        # # Calculate the assembly index
        ai, _, _ = att.calculate_assembly_index(graph, strip_hydrogen=True)
        print(f"Assembly index: {ai}", flush=True)
        array_ai.append(ai)
        array_heavy_atoms.append(n_heavy_atoms[i])
        # except Exception as e:
        #     print(f"Error: {e}", flush=True)

    # add the assembly index to the dataframe
    kegg_data["ai"] = array_ai

    # Plot the assembly index
    plt.scatter(array_heavy_atoms, array_ai, c="black", alpha=0.5)
    att.n_plot("Heavy atom count", "Assembly index")
    plt.savefig("ai_vs_heavy_atom_count.png")
    plt.close()



