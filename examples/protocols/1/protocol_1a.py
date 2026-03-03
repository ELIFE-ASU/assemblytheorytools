import matplotlib.pyplot as plt

import assemblytheorytools as att

if __name__ == "__main__":
    # Set the timeout duration for assembly index calculations (in seconds)
    timeout = 5.0 * 60.0

    mols_str = ['Glycine',
                'Alanine',
                'Proline',
                'Valine',
                'Histidine']

    smis = ['C(C(=O)O)N',
            'C[C@@H](C(=O)O)N',
            'C1C[C@H](NC1)C(=O)O',
            'CC(C)[C@@H](C(=O)O)N',
            'O=C([C@H](CC1=CNC=N1)N)O']

    # Convert the SMILES strings to NetworkX graph representations
    graphs = [att.smi_to_nx(smi) for smi in smis]

    # Print the SMILES strings of the input molecules
    print(f"Input SMILES strings: {smis}", flush=True)
    # Calculate the assembly index for each individual molecule in parallel
    ai_i = att.calculate_assembly_index_parallel(graphs,
                                                 settings={'strip_hydrogen': True,
                                                           'timeout': timeout})[0]
    # Print the assembly indices for each individual molecule
    print(f"Individual assembly indices:", flush=True)
    for i, name in enumerate(mols_str):
        print(f"    {name}: {ai_i[i]}", flush=True)

    # Combine the individual molecule graphs into a single graph
    combined_graph = att.join_graphs(graphs)

    # Calculate the assembly index, virtual objects, and pathway for the combined graph
    jai, virt_obj, pathway = att.calculate_assembly_index(combined_graph,
                                                         strip_hydrogen=True,
                                                         timeout=timeout)
    # Convert the virtual objects (graphs) back to SMILES strings
    virt_obj = [att.nx_to_smi(vo, add_hydrogens=False) for vo in virt_obj]
    print(f"Joint assembly index: {jai}", flush=True)
    print(f"Virtual objects in pathway:", flush=True)
    for vo in virt_obj:
        print(f"    {vo}", flush=True)


    att.plot_pathway(pathway,
                     show_icons=True,
                     frame_on=True,
                     fig_size=(16, 10),
                     layout_style='crossmin_long')
    # Save the pathway plot as SVG and PNG files
    plt.savefig("mol_pathway_example.svg")
    plt.savefig("mol_pathway_example.png", dpi=300)
    plt.show()
