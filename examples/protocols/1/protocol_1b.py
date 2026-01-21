import assemblytheorytools as att
import matplotlib.pyplot as plt

if __name__ == "__main__":
    s_inpt = "gggfhhhvg"

    ai, virt_obj, pathway = att.calculate_string_assembly_index(
        s_inpt,
        directed=False,  # Treat as undirected assembly
        mode='mol'
    )
    print(f"Assembly index: {ai}", flush=True)
    print(f"Virtual objects in pathway: {virt_obj}", flush=True)

    att.plot_pathway(pathway,
                     plot_type='string',
                     layout_style='crossmin')
    plt.savefig("str_pathway_example.svg")
    plt.savefig("str_pathway_example.png", dpi=300)
    plt.show()
