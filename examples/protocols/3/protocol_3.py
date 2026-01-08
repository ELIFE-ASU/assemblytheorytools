import assemblytheorytools as att

if __name__ == "__main__":
    # mols_str = ["papaverine",
    #             "thebaine",
    #             "codeine",
    #             "morphine",
    #             "diamorphine",
    #             "fentanyl",
    #             "methadone",
    #             "remifentanil",
    #             "salvinorin a",
    #             "pethidine"]
    #
    # smis = [att.pubchem_name_to_smi(name) for name in mols_str]
    # print(smis)
    smis = ['COC1=C(C=C(C=C1)CC2=NC=CC3=CC(=C(C=C32)OC)OC)OC', 'CN1CC[C@]23[C@@H]4C(=CC=C2[C@H]1CC5=C3C(=C(C=C5)OC)O4)OC', 'CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)OC)O[C@H]3[C@H](C=C4)O', 'CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O', 'CC(=O)O[C@H]1C=C[C@H]2[C@H]3CC4=C5[C@]2([C@H]1OC5=C(C=C4)OC(=O)C)CCN3C', 'CCC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3', 'CCC(=O)C(CC(C)N(C)C)(C1=CC=CC=C1)C2=CC=CC=C2', 'CCC(=O)N(C1=CC=CC=C1)C2(CCN(CC2)CCC(=O)OC)C(=O)OC', 'CC(=O)O[C@H]1C[C@H]([C@@]2(CC[C@H]3C(=O)O[C@@H](C[C@@]3([C@H]2C1=O)C)C4=COC=C4)C)C(=O)OC', 'CCOC(=O)C1(CCN(CC1)C)C2=CC=CC=C2']

    mols = [att.smi_to_mol(smi) for smi in smis]
    # convert the SMILES strings to graphs
    graphs = [att.smi_to_nx(smi) for smi in smis]
    # Combine the graphs into a single graph
    combined_graph = att.join_graphs(graphs)
    ai, virt_obj, pathway = att.calculate_assembly_index(combined_graph, dir_code="/home/louie/skunkworks/assemblytheorytools/assemblytheorytools/precompiled/asscpp_combined_static_linux", strip_hydrogen=True, timeout=1.0)

    print(ai)
    att.plot_digraph_metro(pathway, filename="metro_pathway_example")
