import matplotlib.pyplot as plt

import assemblytheorytools as att

if __name__ == "__main__":
    # Define the input string for which the assembly index will be calculated
    s_inpt = "gggfhhhvg"

    # Calculate the assembly index, virtual objects, and pathway for the input string
    ai, virt_obj, pathway = att.calculate_string_assembly_index(
        s_inpt,
        directed=False,
    )

    # Print the calculated assembly index
    print(f"Assembly index: {ai}", flush=True)

    # Print the virtual objects in the assembly pathway
    print(f"Virtual objects in pathway: {virt_obj}", flush=True)

    att.plot_pathway(pathway,
                     plot_type="string",
                     font_size=24,
                     fig_size=(16, 10),
                     layout_style='crossmin_long')
    plt.savefig("str_pathway_example.svg")
    plt.savefig("str_pathway_example.png", dpi=300)
    plt.show()