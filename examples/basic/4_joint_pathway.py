import platform

import matplotlib.pyplot as plt

import assemblytheorytools as att

if __name__ == "__main__":
    # Print a blank line to the console
    print(flush=True)

    # Define the SMILES strings for glycine and alanine
    smi = ['C(C(=O)O)N', 'C[C@@H](C(=O)O)N']

    # Convert each SMILES string into a NetworkX graph
    graphs = [att.smi_to_nx(s) for s in smi]

    # Convert each SMILES string into an RDKit Mol object
    mols = [att.smi_to_mol(s) for s in smi]

    # Calculate the assembly similarity between the graphs
    # The `settings` parameter specifies options such as whether to strip hydrogen atoms
    sim = att.calculate_assembly_similarity(graphs, settings={'strip_hydrogen': True})
    print(f'Assembly similarity: {sim}', flush=True)

    # Calculate and print the Bertz complexity for each molecule
    for mol in mols:
        score = att.bertz_complexity(mol)
        print(f'Bertz complexity: {score}', flush=True)

    # Calculate the Tanimoto similarity between the two molecules
    sim = att.tanimoto_similarity(mols[0], mols[1])
    print(f'Tanimoto similarity: {sim}', flush=True)

    # Combine the SMILES strings into a single RDKit Mol object
    mol = att.smi_to_mol('.'.join(smi))

    # Calculate the assembly index, virtual objects, and pathway for the combined molecule
    # The `strip_hydrogen` parameter determines whether hydrogen atoms are removed
    ai, virt_obj, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)

    # Display the calculated assembly index
    print(f"Assembly Index: {ai}", flush=True)

    # Generate and save a metro-style plot of the assembly graph (Linux only)
    if platform.system().lower() == "linux":
        att.plot_digraph_metro(pathway, filename="metro_pathway_example")

    # Visualize the assembly pathway and save it as an SVG file
    att.plot_pathway(pathway, show_icons=True, frame_on=True)
    plt.savefig("mol_pathway_example.svg")
    plt.show()

    # Combine the individual graphs into a single graph
    graph = att.join_graphs(graphs)

    # Calculate the assembly index, virtual objects, and pathway for the combined graph
    ai, virt_obj, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)

    # Visualize the pathway graph and save it as an SVG file
    att.plot_pathway(pathway, show_icons=True, plot_type='graph', frame_on=True)
    plt.savefig("graph_pathway_example.svg")
    plt.show()
