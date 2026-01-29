import assemblytheorytools as att

if __name__ == "__main__":
    s_inpt = "gggfhhhvg"

    ai, virt_obj, pathway = att.calculate_string_assembly_index(
        s_inpt,
        directed=False,
        mode='mol'
    )
    print(f"Assembly index: {ai}", flush=True)
    print(f"Virtual objects in pathway: {virt_obj}", flush=True)
    att.plot_digraph_metro(pathway, filename="metro_pathway_example")
