from rdkit import Chem

import assemblytheorytools as att

if __name__ == "__main__":
    # Define a list of SMILES strings representing molecules.
    smiles = [
        '[H]OC(=O)C([H])([H])N([H])[H]',  # Glycine
        '[H]OC(=O)C([H])(N([H])[H])C([H])([H])[H]',  # Alanine
        '[H]OC(=O)C([H])([H])N([H])[H]',  # Glycine (duplicate)
        '[H]C([H])([H])C([H])([H])[H]',  # Ethane
        '[H]OC(=O)C([H])([H])N([H])[H]'  # Glycine (again)
    ]

    graphs = [att.smi_to_nx(smi) for smi in smiles]
    # Compute assembly index in parallel for all molecules
    ai, vo, pathway = att.calculate_assembly_parallel(graphs, dict(strip_hydrogen=True))

    # Display the results for each molecule
    for i, (ai_i, vo_i, pathway_i) in enumerate(zip(ai, vo, pathway)):
        print(f"SMILES: {smiles[i]}", flush=True)
        print(f"Assembly Index: {ai_i}", flush=True)
        print(f"VOs: {vo_i}", flush=True)
        print(f"Pathway: {pathway_i}", flush=True)
        print(flush=True)
