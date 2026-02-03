import networkx as nx
from rdkit import Chem as Chem

import assemblytheorytools as att

if __name__ == "__main__":
    # Set the timeout duration for the assembly index calculation
    timeout = 600.0

    # List of SMILES strings representing the input molecules
    smiles_list = ['C(C(=O)O)N',  # Glycine
                   'C[C@@H](C(=O)O)N',  # Alanine
                   'C([C@@H](C(=O)O)N)O',  # Serine
                   ]

    # Convert all SMILES strings to molecular graphs
    # Each SMILES string is converted to an RDKit molecule, then to a NetworkX graph,
    # and finally hydrogen atoms are removed from the graph
    graphs = [att.remove_hydrogen_from_graph(att.mol_to_nx(att.smi_to_mol(smile))) for smile in smiles_list]

    # Convert the list of molecular graphs into progressive union graphs (joint assembly)
    # This creates a series of disjoint union graphs, where each graph is the union of the
    # previous graph and the current one
    for i in range(1, len(graphs)):
        graphs[i] = nx.disjoint_union(graphs[i - 1], graphs[i])

    # Calculate the assembly index for each joint graph
    for i, graph in enumerate(graphs):
        print(f"Running joint: {i + 1}", flush=True)

        # Calculate the assembly index and virtual objects for the current graph
        # The assembly index is a measure of molecular complexity, and virtual objects
        # represent intermediate structures in the assembly process
        ai, virt_obj, _ = att.calculate_assembly_index(graph, timeout=timeout)

        # Convert the virtual objects into SMILES strings
        # This step generates SMILES representations of the virtual objects for output
        smiles_output = [Chem.MolToSmiles(att.nx_to_mol(graph)) for graph in virt_obj]

        # Print the assembly index and the SMILES representation of the input graph
        print(f"Assembly index: {ai}", flush=True)
        print(f"Input graph: {smiles_output[0]}", flush=True)

        # Print the SMILES strings of the virtual objects
        print("VO SMILES:", flush=True)
        for smi in smiles_output[1:]:
            print(smi, flush=True)
        print(flush=True)
