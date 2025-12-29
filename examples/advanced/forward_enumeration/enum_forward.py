import matplotlib.pyplot as plt
import networkx as nx

import assemblytheorytools as att

if __name__ == "__main__":
    print(flush=True)

    # Define the SMILES strings for glycine, alanine, serine, and proline
    smi = ['C(C(=O)O)N',
           'C[C@@H](C(=O)O)N',
           'C([C@@H](C(=O)O)N)O',
           'C1C[C@H](NC1)C(=O)O'
           ]

    # Convert the concatenated SMILES strings into an RDKit Mol object
    mol = att.smi_to_mol('.'.join(smi))

    # Convert each SMILES string into a NetworkX graph
    graphs = [att.smi_to_nx(s) for s in smi]

    # Combine the individual graphs into a single graph
    graph = att.join_graphs(graphs)

    # Calculate the assembly index, virtual objects, and pathway for the combined graph
    # The `strip_hydrogen` parameter determines whether hydrogen atoms are removed
    ai, virt_obj, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)

    # Visualize the assembly pathway and save it
    att.plot_pathway(pathway, show_icons=False, frame_on=True)
    plt.savefig("mol_pathway_example.png")
    plt.show()

    # Store the number of steps in the pathway
    n_before = len(pathway)

    # Generate new graphs by enumerating upward from the last two virtual objects
    new_graphs = att.enumerate_up(virt_obj[-1], virt_obj[-2])

    # Print the number of new graphs generated
    print(f'N {len(new_graphs)} new graphs generated', flush=True)

    # Create a directed graph to represent the relationships between the input nodes and new graphs
    G = nx.DiGraph()

    # Add the last two virtual objects as input nodes
    G.add_nodes_from([virt_obj[-1], virt_obj[-2]])

    # Add the newly generated graphs as nodes
    G.add_nodes_from(new_graphs)

    # Add edges from the input nodes to the new graphs
    for g in new_graphs:
        G.add_edge(virt_obj[-1], g)
        G.add_edge(virt_obj[-2], g)

    # Visualize the updated pathway graph
    att.plot_pathway(G, show_icons=False, frame_on=True)
    plt.show()
