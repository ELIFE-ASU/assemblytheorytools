import assemblytheorytools as att

import matplotlib.pyplot as plt

if __name__ == "__main__":
    import os
    os.environ["ASS_PATH"] = '../../../assemblytheorytools/precompiled/asscpp_combined_static_linux'

    smi = att.pubchem_name_to_smi('diethyl phthalate')
    print(f"SMILES: {smi}", flush=True)
    graph = att.smi_to_nx(smi, sanitize=True, add_hydrogens=True)

    ai, virt_obj, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)

    # Print the calculated assembly index to the console
    print(f"Assembly index: {ai}", flush=True)

    # Convert the graphs in the pathway to smi, so I can use 'mol' plot type
    for node in pathway.nodes():
        node_graph = pathway.nodes[node]['vo']
        node_smi = att.nx_to_smi(node_graph, add_hydrogens=False)
        # print(f"Node {node}: {node_smi}")
        pathway.nodes[node]['vo'] = node_smi

    att.plot_pathway(pathway, show_icons=True, frame_on=False, plot_type='mol', fig_size=(14, 7), layout_style='crossmin_long')
    plt.savefig("mol_pathway_example.svg")
    plt.show()

    #att.plot_digraph_metro(pathway, filename="metro_pathway_example")
