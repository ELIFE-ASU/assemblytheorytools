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

    # Select two virtual objects by structure. The virtual-object list order is
    # deliberately not used because it is not stable between calculations.
    vo_by_smiles = {
        att.nx_to_smi(vo, add_hydrogens=False): vo
        for vo in virt_obj
    }
    input_graphs = [vo_by_smiles[smiles] for smiles in ("CN", "C=O")]

    # Generate every graph obtainable by identifying compatible vertices in
    # these two virtual objects.
    new_graphs = att.enumerate_up(*input_graphs)

    # Print the number of new graphs generated
    print(f'N {len(new_graphs)} new graphs generated', flush=True)

    # Create a directed graph to represent the relationships between the input nodes and new graphs
    G = nx.DiGraph()

    # Add the selected virtual objects as input nodes
    G.add_nodes_from(input_graphs)

    # Add the newly generated graphs as nodes
    G.add_nodes_from(new_graphs)

    # Add edges from the input nodes to the new graphs
    for g in new_graphs:
        G.add_edge(input_graphs[0], g)
        G.add_edge(input_graphs[1], g)

    # Visualize the updated pathway graph
    att.plot_pathway(G, plot_type='graph', show_icons=False, frame_on=True)
    plt.show()
