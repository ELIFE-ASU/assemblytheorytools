import matplotlib.pyplot as plt
import networkx as nx

import assemblytheorytools as att

if __name__ == "__main__":
    print(flush=True)

    # Define the SMILES string for glycine, alanine, serine, and proline

    smi = ['C(C(=O)O)N', 'C[C@@H](C(=O)O)N', 'C([C@@H](C(=O)O)N)O', 'C1C[C@H](NC1)C(=O)O']
    # Convert the SMILES string to an RDKit Mol object
    mol = att.smi_to_mol('.'.join(smi))
    graphs = [att.smi_to_nx(s) for s in smi]
    graph = att.join_graphs(graphs)

    # Calculate the assembly index without removing hydrogens
    ai, virt_obj, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)

    att.plot_pathway_mol(pathway, show_icons=False, frame_on=True)
    plt.savefig("mol_pathway_example.svg")
    plt.show()
    n_before = len(pathway)

    new_graphs = att.enumerate_up(virt_obj[-1], virt_obj[-2])
    print(len(new_graphs))

    G = nx.DiGraph()
    # add the input nodes
    G.add_nodes_from([virt_obj[-1], virt_obj[-2]])
    G.add_nodes_from(new_graphs)
    # add edges from the input nodes to the new graphs
    for g in new_graphs:
        G.add_edge(virt_obj[-1], g)
        G.add_edge(virt_obj[-2], g)
    att.plot_pathway_mol(G, show_icons=False, frame_on=True)
    plt.show()
