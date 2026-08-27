import matplotlib.pyplot as plt

import assemblytheorytools as att

if __name__ == "__main__":
    # Print a blank line to the console for better readability
    print(flush=True)

    # Convert a SMILES string to a molecular graph
    # The SMILES string "c1ccc2cc3ccccc3cc2c1" represents anthracene
    graph = att.smi_to_nx("c1ccc2cc3ccccc3cc2c1")

    # Calculate the assembly index with the Rust backend
    # This returns the index alone, and is faster than the default C++ backend
    print(f"Assembly index: {att.calculate_assembly_index_rust(graph)}", flush=True)

    # Calculate the assembly depth, which counts the joining steps along the
    # longest branch of a pathway rather than the total number of steps
    # The depth search is much more expensive than the index search and takes no
    # timeout, so it is only run here on a small molecule
    benzene = att.smi_to_nx("c1ccccc1")
    print(f"Benzene assembly depth: {att.calculate_assembly_depth_rust(benzene)}",
          flush=True)

    # Inspect the graph the backend actually builds, which confirms that
    # hydrogens have been dropped and that the rings have been kekulised
    info = att.get_molecule_info_rust(graph)
    print(f"Atoms seen by the backend: {info.count('label = \"Atom')}", flush=True)

    # Run the search with explicit options and report what it did
    # 'parallel="none"' makes the number of states searched reproducible
    result = att.calculate_assembly_index_rust_search(graph, parallel="none")
    print(f"Index: {result.index}, matching pairs: {result.num_matches}, "
          f"states searched: {result.states_searched}", flush=True)

    # Reconstruct a minimum assembly pathway and plot it
    # Pathway reconstruction needs an assembly-theory release newer than 0.6.1,
    # so fall back gracefully when the installed one cannot do it
    try:
        result = att.calculate_assembly_index_rust_search(
            graph, parallel="none", max_pathways=1)
    except NotImplementedError as e:
        print(f"Skipping pathway reconstruction: {e}", flush=True)
    else:
        pathway = result.pathways[0]
        print(f"Pathway: {pathway.number_of_nodes()} virtual objects, "
              f"{pathway.number_of_edges()} joining operations", flush=True)

        # The pathway is an ordinary ATT pathway graph, so it plots like any other
        fig, ax = att.plot_pathway(pathway, plot_type="mol")
        plt.show()
