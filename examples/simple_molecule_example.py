import assemblytheorytools as att

if __name__ == "__main__":
    # Convert all the smile to mol
    mol = att.smi_to_mol("c1ccccc1")
    # Calculate the assembly index
    ai, path = att.calculate_assembly_index(mol)
    # Convert the pathway into smiles
    smi_out = att.get_mol_pathway_to_smi(path)

    print(f"Assembly index: {ai}", flush=True)
    print(f"Path: {smi_out}", flush=True)
