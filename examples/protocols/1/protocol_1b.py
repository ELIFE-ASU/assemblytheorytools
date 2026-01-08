import assemblytheorytools as att

if __name__ == "__main__":
    # Define a single input string for analysis
    s_inpt = "gggfhhhvg"

    # Calculate the string assembly index
    ai, virt_obj, pathway = att.calculate_string_assembly_index(
        s_inpt,
        directed=False,  # Treat as undirected assembly
        mode='mol'
    )

    # Output the results
    # Print the calculated assembly index, virtual objects, and assembly pathway to the console
    print(f"Assembly index: {ai}", flush=True)
    print(f"virt_obj: {virt_obj}", flush=True)
    print(f"path: {pathway}", flush=True)

    att.plot_digraph_metro(pathway, filename="metro_pathway_example")
