import matplotlib.pyplot as plt
import networkx as nx

import assemblytheorytools as att


def draw_edges_from_metabolites(graph, color_to_index, metabolites):
    """
    Draws edges between nodes in a graph based on metabolite connections.

    Args:
        graph (networkx.Graph): The graph to which edges will be added.
        color_to_index (dict): A dictionary mapping metabolite names to node indices.
        metabolites (dict): A dictionary where keys are metabolite names and values are lists of connected metabolites.

    Returns:
        networkx.Graph: The graph with edges added based on metabolite connections.
    """
    for metabolite, connections in metabolites.items():
        if metabolite in color_to_index:
            for connection in connections:
                if connection in color_to_index:
                    graph.add_edge(color_to_index[metabolite], color_to_index[connection])
    return graph


def flatten_dict_to_list(d):
    """
    Flattens a dictionary into a list containing all keys and values.

    Args:
        d (dict): The dictionary to flatten.

    Returns:
        list: A list containing all keys and values from the dictionary.
    """
    flattened_list = []
    for key, values in d.items():
        flattened_list.append(key)
        flattened_list.extend(values if isinstance(values, list) else [values])
    return flattened_list


if __name__ == "__main__":
    G = nx.Graph()

    network1_metabolites = {"CO2": ["H2", "MFR", "Formyl-MFR"],
                            "Formyl-MFR": ["H+", "THMPT", "CO2", "Methane"],
                            "Methyl-S-coenzyme M": ["Coenzyme B"],
                            "Methane": ["H+"]}

    # Flatten the dictionary and get a set of items
    flattened_list = list(set(flatten_dict_to_list(network1_metabolites)))
    print("Nodes: ", flattened_list, flush=True)

    for i, item in enumerate(flattened_list):
        G.add_node(i, color=item)

    # Draw a map between the node color and its node index
    color_to_index = {item: i for i, item in enumerate(flattened_list)}
    print("color map:", color_to_index, flush=True)

    # Draw edges
    G = draw_edges_from_metabolites(G, color_to_index, network1_metabolites)

    # Plot the graph
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.savefig("graph.png")
    plt.close()

    # Calculate the assembly index
    ai, virt_obj, path = att.calculate_assembly_index(G, debug=False)
    print(f"Assembly index: {ai}", flush=True)
