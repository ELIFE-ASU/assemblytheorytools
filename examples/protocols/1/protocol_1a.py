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

    # Print the virtual objects to the console
    print(f"virt_obj: {virt_obj}", flush=True)

    att.plot_pathway(pathway, show_icons=False, frame_on=True, plot_type='graph')
    plt.savefig("mol_pathway_example.svg")
    plt.show()

    att.plot_digraph_metro(pathway, filename="metro_pathway_example")
