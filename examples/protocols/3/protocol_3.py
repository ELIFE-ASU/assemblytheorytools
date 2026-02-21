import matplotlib.pyplot as plt
import networkx as nx

import assemblytheorytools as att


def strip_digraph_zero_indegree(G: nx.DiGraph) -> nx.DiGraph:
    G = G.copy()
    nodes = [n for n, indeg in G.in_degree() if indeg > 0]
    return G.subgraph(nodes)


if __name__ == "__main__":
    # Set the timeout duration for assembly index calculations (in seconds)
    timeout = 5.0 * 60.0

    # Define a list of molecule names to analyze
    mols_str = ["Papaverine",
                "Thebaine",
                "Codeine"]
    # Corresponding SMILES strings for the molecules
    smis = ['COc1ccc(cc1OC)Cc2c3cc(c(cc3ccn2)OC)OC',
            'COC1=CC=C2[C@@H](C3)N(C)CC[C@@]24C5=C3C=CC(OC)=C5O[C@@H]14',
            'CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)OC)O[C@H]3[C@H](C=C4)O',
            ]

    smis = ['C(C(=O)O)N',
            'C[C@@H](C(=O)O)N',
            'C1C[C@H](NC1)C(=O)O',
            'CC(C)[C@@H](C(=O)O)N',
            'O=C([C@H](CC1=CNC=N1)N)O']

    # Convert the SMILES strings to NetworkX graph representations
    graphs = [att.smi_to_nx(smi) for smi in smis]

    # # Visualize the common bonds between the molecules and display the image
    # img = att.show_common_bonds(*smis, legends=mols_str)
    # img.save("common_bonds.png")
    # img.show()

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
    ai, virt_obj, pathway = att.calculate_assembly_index(combined_graph,
                                                         strip_hydrogen=True,
                                                         timeout=timeout)
    # Print the assembly index and virtual objects for the combined graph
    print(f"Joint assembly index: {ai}", flush=True)

    att.plot_pathway(pathway,
                     show_icons=True,
                     frame_on=True,
                     fig_size=(16, 9),
                     layout_style='crossmin_long')
    plt.show()

    # att.plot_pathway(pathway,
    #                  show_icons=True,
    #                  fig_size=(30, 15),
    #                  layout_style='crossmin_long')
    # plt.show()
    #
    # pathway = strip_digraph_zero_indegree(pathway)
    # pathway = strip_digraph_zero_indegree(pathway)
    # att.plot_pathway(pathway,
    #                  show_icons=True,
    #                  fig_size=(18, 12),
    #                  layout_style='crossmin_long')
    # plt.show()
