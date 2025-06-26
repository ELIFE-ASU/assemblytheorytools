import assemblytheorytools as att

if __name__ == "__main__":
    print(flush=True)

    # Define the SMILES string for glycine
    smi = "C(C(=O)O)N"

    # Convert the SMILES string to an RDKit Mol object
    mol = att.smi_to_mol(smi)

    # Compute the assembly index and associated data
    ai, virt_obj, pathway = att.calculate_assembly_index(mol, debug=False)

    # Display the calculated assembly index
    print(f"Assembly Index: {ai}")

    # Unpack pathway information
    pathway, vo_list = pathway

    # Generate and save a metro-style plot of the assembly graph
    att.plot_digraph_metro(pathway, filename="example")
