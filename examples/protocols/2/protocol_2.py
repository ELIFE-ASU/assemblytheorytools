import time as t

import matplotlib.pyplot as plt
import numpy as np

import assemblytheorytools as att

if __name__ == "__main__":
    t_0 = t.time()
    n_sample = 20_000
    max_mw = 550.0
    max_bonds = 50

    att.download_pubchem_cid_smiles_gz()
    sample = att.sample_pubchem_cid_smiles_gz_mw(n_sample, max_mw=max_mw, max_bonds=max_bonds)
    mw = sample['mw'].tolist()
    smis = sample['smiles'].tolist()
    t_1 = t.time()
    print(f"Time to sample {n_sample} molecules: {t_1 - t_0:.1f} seconds", flush=True)

    t_2 = t.time()
    # Convert SMILES strings to RDKit molecule objects
    mols = att.mp_calc(att.smi_to_mol, smis)

    # Visualize the first 16 molecules in a grid
    img = att.draw_mol_grid(smis[:16], legends=smis[:16], n_cols=4)
    img.show()

    # Calculate the assembly index for the list of molecules
    ai = att.calculate_assembly_parallel(mols, settings={'strip_hydrogen': True,
                                                               'timeout': 120.0})[0]
    t_3 = t.time()
    print(f"Time to calculate assembly indices for molecules: {t_3 - t_2:.1f} seconds", flush=True)

    plt.scatter(mw, ai)

    plt.xlabel("Molecular Weight")
    plt.ylabel("Assembly Index")
    plt.show()

    n_x_bins = len(set(int(x) // 10 * 10 for x in mw))
    n_y_bins = len(set(ai))

    fig, ax = att.plot_heatmap(np.array(mw),
                               np.array(ai),
                               "Molecular Weight",
                               "Assembly Index",
                               nbins=(n_x_bins, n_y_bins),
                               )
    plt.show()
