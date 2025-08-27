import platform

import matplotlib.pyplot as plt

import assemblytheorytools as att

if __name__ == "__main__":
    print(flush=True)

    # Define the SMILES string for glycine and alanine
    smi = 'C(C(=O)O)N.C[C@@H](C(=O)O)N'
    # L-Serine: C([C@@H](C(=O)O)N)O
    # Tryptophan: c1[nH]c2ccccc2c1C[C@H](N)C(=O)O
    smi = 'C([C@@H](C(=O)O)N)O'

    # Convert the SMILES string to an RDKit Mol object
    mol = att.smi_to_mol(smi)

    # Compute the assembly index and associated data
    ai, virt_obj, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)

    # Display the calculated assembly index
    print(f"Assembly Index: {ai}", flush=True)

    # Generate and save a metro-style plot of the assembly graph
    if platform.system().lower() == "linux":
        att.plot_digraph_metro(pathway, filename="metro_pathway_example")

    # Convert the SMILES string to an RDKit Mol object
    mol = att.smi_to_mol(smi)
    ai, virt_obj, pathway = att.calculate_assembly_index(mol, strip_hydrogen=True)
    att.plot_pathway_mol(pathway, show_icons=True, frame_on=True)
    plt.savefig("mol_pathway_example.svg")
    plt.show()

    graph = att.smi_to_nx(smi)
    ai, virt_obj, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)
    att.plot_pathway_graph(pathway, show_icons=True, frame_on=True)
    plt.savefig("graph_pathway_example.svg")
    plt.show()
