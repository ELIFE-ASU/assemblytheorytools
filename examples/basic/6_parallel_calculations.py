import assemblytheorytools as att

if __name__ == "__main__":
    # Define a list of SMILES strings representing molecules.
    # Each SMILES string corresponds to a specific molecule, including stereochemistry.
    # Example molecules include various amino acids and other organic compounds.
    smiles = ['C([C@@H](C(=O)O)N)S',  # Cysteine
              'OC(=O)CC[C@H](N)C(=O)O',  # Glutamic acid
              'NC(=O)CC[C@H](N)C(=O)O',  # Glutamine
              'NCC(=O)O',  # Glycine
              'C1=C(NC=N1)C[C@H](N)C(=O)O',  # Histidine
              'CC[C@H](C)[C@H](N)C(=O)O',  # Isoleucine
              'CC(C)C[C@H](N)C(=O)O',  # Leucine
              'NCCCC[C@H](N)C(=O)O',  # Lysine
              'CSCC[C@H](N)C(=O)O',  # Methionine
              'c1ccc(cc1)C[C@H](N)C(=O)O',  # Phenylalanine
              'OC(=O)[C@@H]1CCCN1',  # Proline
              'OC[C@H](N)C(=O)O',  # Serine
              'C[C@@H](O)[C@H](N)C(=O)O',  # Threonine
              'c1ccc2c(c1)c(c[nH]2)C[C@H](N)C(=O)O',  # Tryptophan
              'Oc1ccc(cc1)C[C@H](N)C(=O)O',  # Tyrosine
              'CC(C)[C@H](N)C(=O)O'  # Valine
              ]

    # Convert each SMILES string into a NetworkX graph representation.
    # This step prepares the molecules for assembly index calculations.
    graphs = [att.smi_to_nx(smi) for smi in smiles]

    # Compute the assembly index, virtual objects, and pathways in parallel for all molecules.
    # The `strip_hydrogen` parameter specifies whether hydrogen atoms are removed during calculations.
    ai, vo, pathway = att.calculate_assembly_index_parallel(graphs, dict(strip_hydrogen=True))

    # Display the results for each molecule.
    # For each SMILES string, print the corresponding assembly index, virtual objects, and pathway.
    for i, (ai_i, vo_i, pathway_i) in enumerate(zip(ai, vo, pathway)):
        print(f"SMILES: {smiles[i]}", flush=True)  # Print the SMILES string.
        print(f"Assembly Index: {ai_i}", flush=True)  # Print the calculated assembly index.
        print(f"VOs: {vo_i}", flush=True)  # Print the virtual objects.
        print(f"Pathway: {pathway_i}", flush=True)  # Print the assembly pathway.
        print(flush=True)  # Print a blank line for better readability.
