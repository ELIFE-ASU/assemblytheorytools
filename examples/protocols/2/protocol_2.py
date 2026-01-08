import assemblytheorytools as att

import matplotlib.pyplot as plt

if __name__ == "__main__":
    _, smis = att.sample_random_pubchem(100, seed=42)

    mols = [att.smi_to_mol(smi, sanitize=True, add_hydrogens=True) for smi in smis]

    graphs = [att.smi_to_nx(smi, sanitize=True, add_hydrogens=True) for smi in smis]
    mw = [att.molecular_weight(mol) for mol in mols]

    # Calculate the assembly index, virtual objects, and pathway for the molecule
    # The assembly index is a measure of molecular complexity
    # Virtual objects represent intermediate structures in the assembly process
    # The pathway describes the sequence of steps in the assembly process
    ai, _, _ = att.calculate_assembly_parallel(graphs, settings={'strip_hydrogen': True, 'timeout': 10.0})

    # Print the calculated assembly index to the console
    print(f"Assembly index: {ai}", flush=True)


    plt.scatter(mw, ai)
    plt.show()
