import matplotlib.pyplot as plt

import assemblytheorytools as att

if __name__ == "__main__":
    smi = att.pubchem_name_to_smi('diethyl phthalate')
    print(f"SMILES: {smi}", flush=True)
    graph = att.smi_to_nx(smi, sanitize=True, add_hydrogens=True)
    ai, virt_obj, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True)
    print(f"Assembly index: {ai}", flush=True)

    virt_obj = [att.nx_to_smi(vo, add_hydrogens=False) for vo in virt_obj]
    print(f"Virtual objects in pathway: {virt_obj}", flush=True)

    att.plot_pathway(pathway,
                     frame_on=False,
                     plot_type='mol', fig_size=(14, 7), layout_style='crossmin_long')
    plt.savefig("mol_pathway_example.svg")
    plt.savefig("mol_pathway_example.png", dpi=300)
    plt.show()
