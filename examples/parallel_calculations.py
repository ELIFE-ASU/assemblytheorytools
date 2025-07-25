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

    # Convert SMILES to RDKit Mol objects and ensure explicit hydrogens
    mols = [Chem.AddHs(Chem.MolFromSmiles(smi, sanitize=True)) for smi in smiles]

    # Set the computation timeout to 3 minutes
    timeout = 3.0 * 60.0

    # Compute assembly index in parallel for all molecules
    ai, vo, pathway = att.calculate_assembly_index_parallel(
        mols,
        dict(timeout=timeout, strip_hydrogen=True)
    )

    # Display the results for each molecule
    for i, (ai_i, vo_i, pathway_i) in enumerate(zip(ai, vo, pathway)):
        print(f"SMILES: {smiles[i]}")
        print(f"Assembly Index: {ai_i}")
        print(f"VOs: {vo_i}")
        print(f"Pathway: {pathway_i}")
        print()
