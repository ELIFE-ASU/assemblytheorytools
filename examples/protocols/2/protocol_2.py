import os

import matplotlib.pyplot as plt

import assemblytheorytools as att

import time as t

if __name__ == "__main__":
    t_0 = t.time()
    n_sample = 10_000
    sample_file = 'pubchem_sample.smi'

    # Check if sample file exists
    if not os.path.isfile(sample_file):
        # If not, create it by sampling random PubChem molecules
        _, smis = att.sample_random_pubchem(n_sample, seed=42)
        t_1 = t.time()
        print(f"Time to sample {n_sample} molecules: {t_1 - t_0:.1f} seconds", flush=True)

        # Write the sampled SMILES to a file
        with open(sample_file, 'w') as f:
            for smi in smis:
                f.write(f"{smi}\n")
    else:
        # If the file exists, read the SMILES from it
        with open(sample_file, 'r') as f:
            smis = [line.strip() for line in f.readlines()]

    t_2 = t.time()
    # Convert SMILES strings to RDKit molecule objects
    mols = [att.smi_to_mol(smi, sanitize=True, add_hydrogens=True) for smi in smis]

    # Get the molecular weights for each molecule
    mw = [att.molecular_weight(mol) for mol in mols]

    # Calculate the assembly index for the list of molecules
    ai, _, _ = att.calculate_assembly_parallel(mols, settings={'strip_hydrogen': True,
                                                               'timeout': 30.0})
    t_3 = t.time()
    print(f"Time to calculate assembly indices for molecules: {t_3 - t_2:.1f} seconds", flush=True)

    plt.scatter(mw, ai)
    plt.show()
