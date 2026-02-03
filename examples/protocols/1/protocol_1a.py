import matplotlib.pyplot as plt

import assemblytheorytools as att

# Main execution block
if __name__ == "__main__":
    # Convert a compound name to its SMILES representation using PubChem
    smi = att.pubchem_name_to_smi('diethyl phthalate')
    print(f"Input SMILES: {smi}", flush=True)

    # Convert the SMILES string to a NetworkX graph representation
    graph = att.smi_to_nx(smi)

    # Calculate the assembly index, virtual objects, and pathway for the graph
    ai, virt_obj, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)

    # Convert the virtual objects (graphs) back to SMILES strings
    virt_obj = [att.nx_to_smi(vo, add_hydrogens=False) for vo in virt_obj]

    # Print the calculated assembly index and virtual objects
    print(f"Assembly index: {ai}", flush=True)
    print(f"Virtual objects in pathway: {virt_obj}", flush=True)

    # Plot the assembly pathway with specified visualization parameters
    att.plot_pathway(pathway,
                     frame_on=True,
                     plot_type='mol',  # Plot molecules in the pathway
                     fig_size=(14, 7),  # Set figure size
                     layout_style='crossmin_long')  # Use a specific layout style

    # Save the pathway plot as SVG and PNG files
    plt.savefig("mol_pathway_example.svg")
    plt.savefig("mol_pathway_example.png", dpi=300)

    # Display the plot
    plt.show()
