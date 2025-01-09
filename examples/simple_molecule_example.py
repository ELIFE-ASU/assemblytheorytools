import assemblytheorytools as att

if __name__ == "__main__":
    # Convert all the smile to mol
    mol = att.smi_to_mol("c1ccccc1")
    # Calculate the assembly index
    ai, virt_obj, path = att.calculate_assembly_index(mol)
    # Convert the pathway into smiles
    smi_out = att.get_mol_pathway_to_smi(virt_obj)

    print(f"Assembly index: {ai}", flush=True)
    print(f"virt_obj: {smi_out}", flush=True)

    # Convert to graph
    graph = att.mol_to_nx(mol)

    # dry run the assembly calculation
    att.assembly_dry_run(graph)
