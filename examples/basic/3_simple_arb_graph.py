import matplotlib.pyplot as plt
import networkx as nx

import assemblytheorytools as att

if __name__ == "__main__":
    # Create a simple undirected graph using NetworkX
    graph = nx.Graph()

    # Add nodes to the graph with a "color" attribute as a label
    # Node indices must start from 0
    graph.add_node(0, color="0")  # Node 0 with color label "0"
    graph.add_node(1, color="1")  # Node 1 with color label "1"
    graph.add_node(2, color="2")  # Node 2 with color label "2"
    graph.add_node(3, color="3")  # Node 3 with color label "3"

    # Add edges between nodes with a "color" attribute as a label
    # The "color" attribute is an integer starting from 1
    graph.add_edge(0, 1, color=1)  # Edge between nodes 0 and 1 with color label 1
    graph.add_edge(1, 2, color=1)  # Edge between nodes 1 and 2 with color label 1
    graph.add_edge(2, 3, color=1)  # Edge between nodes 2 and 3 with color label 1

    # Visualize the graph structure
    # The graph is drawn with node labels and bold font for better visibility
    nx.draw(graph, with_labels=True, font_weight='bold')
    plt.show()

    # Calculate the assembly index for the graph
    # The assembly index is a measure of the graph's structural complexity
    # `virt_obj` represents virtual objects, and `path` represents the assembly pathway
    ai, virt_obj, path = att.calculate_assembly_index(graph)

    # Print the calculated assembly index to the console
    print(f"Assembly index: {ai}", flush=True)
