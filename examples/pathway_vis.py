import assemblytheorytools as att

if __name__ == "__main__":
    print(flush=True)
    smi = "C(C(=O)O)N"
    mol = att.smi_to_mol(smi)
    ai, virt_obj, pathway = att.calculate_assembly_index(mol, debug=False)
    print(f"Assembly Index: {ai}")
    pathway, vo_list = pathway
    att.plot_digraph_metro(pathway, filename="example")
