import assemblytheorytools as att

if __name__ == "__main__":
    # Define a list of molecule names to analyze
    mols_str = ["codeine",
                "morphine"]
    # Set the timeout duration for assembly index calculations (in seconds)
    timeout = 5.0 * 60.0

    # Convert molecule names to their corresponding SMILES strings using PubChem
    smis = [att.pubchem_name_to_smi(name) for name in mols_str]

    # Visualize the common bonds between the molecules and display the image
    img = att.show_common_bonds(*smis, legends=mols_str)
    img.show()

    # Print the SMILES strings of the input molecules
    print(f"Input SMILES strings: {smis}", flush=True)

    # Convert the SMILES strings to NetworkX graph representations
    graphs = [att.smi_to_nx(smi) for smi in smis]

    # Calculate the assembly indices for the individual molecules in parallel
    # Settings:
    # - strip_hydrogen: Removes hydrogen atoms from the graph
    # - timeout: Maximum time allowed for the calculation per graph (in seconds)
    ai_i = att.calculate_assembly_index_parallel(graphs,
                                                 settings={'strip_hydrogen': True,
                                                           'timeout': timeout})[0]
    # Print the assembly indices for each individual molecule
    print(f"Individual assembly indices:", flush=True)
    for i, name in enumerate(mols_str):
        print(f"{name}: {ai_i[i]}", flush=True)

    # Combine the individual molecule graphs into a single graph
    combined_graph = att.join_graphs(graphs)

    # Calculate the assembly index, virtual objects, and pathway for the combined graph
    # Settings:
    # - strip_hydrogen: Removes hydrogen atoms from the graph
    # - timeout: Maximum time allowed for the calculation (in seconds)
    ai, virt_obj, pathway = att.calculate_assembly_index(combined_graph,
                                                         strip_hydrogen=True,
                                                         timeout=timeout)
    # Convert the virtual objects (graphs) back to SMILES strings
    virt_obj = [att.nx_to_smi(vo, add_hydrogens=False) for vo in virt_obj]

    # Print the assembly index and virtual objects for the combined graph
    print(f"Joint assembly index: {ai}", flush=True)
    print(f"Virtual objects in pathway: {virt_obj}", flush=True)

    # Plot the directed graph representation of the assembly pathway
    # - vo_names: Display synonyms for virtual object names
    att.plot_digraph_metro(pathway, vo_names='synonym')
