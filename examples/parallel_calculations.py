from rdkit import Chem

import assemblytheorytools as att

if __name__ == "__main__":
    smiles = ['[H]OC(=O)C([H])([H])N([H])[H]',
              '[H]OC(=O)C([H])(N([H])[H])C([H])([H])[H]',
              '[H]OC(=O)C([H])([H])N([H])[H]',
              '[H]C([H])([H])C([H])([H])[H]',
              '[H]OC(=O)C([H])([H])N([H])[H]']
    mols = [Chem.AddHs(Chem.MolFromSmiles(smi, sanitize=True)) for smi in smiles]
    timeout = 3.0 * 60.0  # 3 mins

    # Calculate the assembly index in parallel
    ai, vo, pathway = att.calculate_assembly_index_parallel(mols, dict(timeout=timeout, strip_hydrogen=True))

    # Loop over the results
    for i, (ai_i, vo_i, pathway_i) in enumerate(zip(ai, vo, pathway)):
        print(f"SMILES: {smiles[i]}")
        print(f"Assembly Index: {ai_i}")
        print(f"VOs: {vo_i}")
        print(f"Pathway: {pathway_i}")
        print()
