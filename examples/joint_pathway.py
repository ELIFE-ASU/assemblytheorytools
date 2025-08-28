import platform

import matplotlib.pyplot as plt

import assemblytheorytools as att

if __name__ == "__main__":
    print(flush=True)

    # Define the SMILES string for glycine and alanine
    smi = ['C(C(=O)O)N', 'C[C@@H](C(=O)O)N']
    graphs = [att.smi_to_nx(s) for s in smi]
    mols = [att.smi_to_mol(s) for s in smi]
    sim = att.calculate_assembly_similarity(graphs, {'strip_hydrogen': True})
    print(f'Assembly similarity: {sim}', flush=True)

    for mol in mols:
        score = att.bertz_complexity(mol)
        print(f'Bertz complexity: {score}', flush=True)

    sim = att.tanimoto_similarity(mols[0], mols[1])
    print(f'Tanimoto similarity: {sim}', flush=True)

    # Convert the SMILES string to an RDKit Mol object
    mol = att.smi_to_mol('.'.join(smi))
    ai, virt_obj, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)
    # Display the calculated assembly index
    print(f"Assembly Index: {ai}", flush=True)

    # Generate and save a metro-style plot of the assembly graph
    if platform.system().lower() == "linux":
        att.plot_digraph_metro(pathway, filename="metro_pathway_example")
    att.plot_pathway_mol(pathway, show_icons=True, frame_on=True)
    plt.savefig("mol_pathway_example.svg")
    plt.show()

    graph = att.join_graphs(graphs)
    ai, virt_obj, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)
    att.plot_pathway_mol(pathway, show_icons=True, frame_on=True, plot_type='graph')
    plt.savefig("graph_pathway_example.svg")
    plt.show()
