import assemblytheorytools as att

if __name__ == "__main__":
    # Define the input string for which the assembly index will be calculated
    s_inpt = "gggfhhhvg"

    # Calculate the assembly index, virtual objects, and pathway for the input string
    # Parameters:
    # - directed: Whether the graph is directed (False for undirected)
    # - mode: The mode of calculation ('mol' for molecular assembly)
    ai, virt_obj, pathway = att.calculate_string_assembly_index(
        s_inpt,
        directed=False,
        mode='mol'
    )

    # Print the calculated assembly index
    print(f"Assembly index: {ai}", flush=True)

    # Print the virtual objects in the assembly pathway
    print(f"Virtual objects in pathway: {virt_obj}", flush=True)

    # Plot the directed graph representation of the assembly pathway
    att.plot_digraph_metro(pathway)
