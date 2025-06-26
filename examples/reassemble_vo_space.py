from rdkit.Chem import AllChem as Chem
import assemblytheorytools as att

if __name__ == "__main__":
    print(flush=True)

    # List of SMILES strings (example here: glycine with explicit hydrogens)
    molecules = ['[H]OC(=O)C([H])([H])N([H])[H]']

    # Convert SMILES strings to RDKit Mol objects using att's helper
    mols = [att.smi_to_mol(smile) for smile in molecules]

    # Combine all molecules into one superstructure
    mol = att.combine_mols(mols)

    # Calculate the assembly index without removing hydrogens
    ai, virt_obj, _ = att.calculate_assembly_index(mol, strip_hydrogen=False)

    # Convert the internal assembly pathway (dict) to a list of molecules
    mols_out = att.convert_pathway_dict_to_list(virt_obj)

    # Convert pathway molecules to InChI strings for easy comparison/printing
    re_mols = [Chem.MolToInchi(mol) for mol in mols_out]
    print(re_mols, flush=True)

    # Reassemble the original molecule(s) from substructures, multiple times
    re_mols = att.reassemble_old(mols_out, n_mol_needed=4)

    # Convert reassembled molecules to InChI for output
    re_mols = [Chem.MolToInchi(mol) for mol in re_mols]
    print(re_mols, flush=True)
