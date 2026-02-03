import platform

import matplotlib.pyplot as plt

import assemblytheorytools as att

if __name__ == "__main__":
    # Print a blank line to the console for better readability
    print(flush=True)

    # Define the SMILES string representing a molecule
    # This string encodes the structure of two molecules: glycine and alanine
    smi = 'C(C(=O)O)N.C[C@@H](C(=O)O)N'

    # Convert the SMILES string to an RDKit Mol object
    # This object is used for molecular computations
    mol = att.smi_to_mol(smi)

    # Compute the assembly index and associated data for the molecule
    # The assembly index measures the structural complexity of the molecule
    # `virt_obj` represents virtual objects, and `pathway` represents the assembly pathway
    ai, virt_obj, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)

    # Display the calculated assembly index in the console
    print(f"Assembly Index: {ai}", flush=True)

    # Generate and save a metro-style plot of the assembly graph
    # This visualization is only generated on Linux systems
    if platform.system().lower() == "linux":
        att.plot_digraph_metro(pathway, filename="metro_pathway_example")

    # Convert the SMILES string to an RDKit Mol object again
    # This step is repeated for further visualization purposes
    mol = att.smi_to_mol(smi)
    ai, virt_obj, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)

    # Visualize the assembly pathway and save it as an SVG file
    # The pathway is displayed with icons and a frame
    att.plot_pathway(pathway, show_icons=True, frame_on=True)
    plt.savefig("mol_pathway_example.svg")
    plt.show()

    # Convert the SMILES string to a NetworkX graph representation
    # This graph representation is used for graph-based computations
    graph = att.smi_to_nx(smi)

    # Compute the assembly index and associated data for the graph
    ai, virt_obj, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)

    # Visualize the pathway graph and save it as an SVG file
    # The graph-based pathway is displayed with icons and a frame
    att.plot_pathway(pathway, show_icons=True, plot_type='graph', frame_on=True)
    plt.savefig("graph_pathway_example.svg")
    plt.show()
