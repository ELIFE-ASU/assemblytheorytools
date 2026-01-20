import assemblytheorytools as att
import matplotlib.pyplot as plt

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


    att.plot_pathway(pathway, show_icons=True, frame_on=True, fig_size=(14, 7), plot_type='string', layout_style = 'crossmin', node_color='white') #'crossmin', 'crossmin_long', 'sa'
    plt.savefig("str_pathway_example.svg")
    plt.show()
