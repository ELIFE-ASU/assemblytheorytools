import assemblytheorytools as att

if __name__ == "__main__":
    mols_str = ["codeine",
                "morphine"]
    timeout = 5.0 * 60.0

    smis = [att.pubchem_name_to_smi(name) for name in mols_str]

    img = att.show_common_bonds(*smis, legends=mols_str)
    img.show()

    print(f"Input SMILES strings: {smis}", flush=True)

    graphs = [att.smi_to_nx(smi) for smi in smis]
    ai_i = att.calculate_assembly_index_parallel(graphs,
                                                 settings={'strip_hydrogen': True,
                                                           'timeout': timeout})[0]
    print(f"Individual assembly indices:", flush=True)
    for i, name in enumerate(mols_str):
        print(f"{name}: {ai_i[i]}", flush=True)

    combined_graph = att.join_graphs(graphs)
    ai, virt_obj, pathway = att.calculate_assembly_index(combined_graph,
                                                         strip_hydrogen=True,
                                                         timeout=timeout)
    virt_obj = [att.nx_to_smi(vo, add_hydrogens=False) for vo in virt_obj]
    print(f"Joint assembly index: {ai}", flush=True)
    print(f"Virtual objects in pathway: {virt_obj}", flush=True)
    att.plot_digraph_metro(pathway, vo_names='synonym')
