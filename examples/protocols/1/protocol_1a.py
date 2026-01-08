import assemblytheorytools as att

import matplotlib.pyplot as plt

if __name__ == "__main__":
    mol_str = 6781
    mol_str = 'diethyl phthalate'
    smi = att.pubchem_id_to_smi(mol_str)
    print(f"SMILES: {smi}", flush=True)
    mol = att.pubchem_id_to_mol(mol_str)
    graph = att.smi_to_nx('CCOC(=O)C1=CC=CC=C1C(=O)OCC', sanitize=True, add_hydrogens=True)

    ai, virt_obj, pathway = att.calculate_assembly_index(graph, strip_hydrogen=True, timeout=1000.0)

    # Print the calculated assembly index to the console
    print(f"Assembly index: {ai}", flush=True)

    # Print the virtual objects to the console
    print(f"virt_obj: {virt_obj}", flush=True)

    att.plot_pathway(pathway, show_icons=True, frame_on=True)
    plt.savefig("mol_pathway_example.svg")
    plt.show()

    att.plot_digraph_metro(pathway, filename="metro_pathway_example")
