from rdkit.Chem import AllChem as Chem

import assemblytheorytools as att

if __name__ == "__main__":
    print(flush=True)
    molecules = ['[H]OC(=O)C([H])([H])N([H])[H]']
    # Convert all the SMILES strings to molecule objects
    mols = [att.smi_to_mol(smile) for smile in molecules]
    # Combine the molecule objects into a single molecule
    mol = att.combine_mols(mols)

    # Calculate the assembly index
    ai, virt_obj, _ = att.calculate_assembly_index(mol, strip_hydrogen=False)

    mols_out = att.convert_pathway_dict_to_list(virt_obj)

    re_mols = [Chem.MolToInchi(mol) for mol in mols_out]
    print(re_mols, flush=True)

    re_mols = att.reassemble_old(mols_out, n_mol_needed=4)
    re_mols = [Chem.MolToInchi(mol) for mol in re_mols]
    print(re_mols, flush=True)
