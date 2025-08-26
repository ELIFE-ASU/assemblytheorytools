import matplotlib.pyplot as plt
import networkx as nx

import assemblytheorytools as att

if __name__ == "__main__":
    # Create a simple graph
    graph = nx.Graph()

    # Add nodes, the color attribute (string) is used to label the node
    graph.add_node(0, color="0")  # Graph index must start from 0
    graph.add_node(1, color="1")
    graph.add_node(2, color="2")
    graph.add_node(3, color="3")

    # Add edges, the color attribute (integer from 1-inf) is used to label the edge
    graph.add_edge(0, 1, color=1)
    graph.add_edge(1, 2, color=1)
    graph.add_edge(2, 3, color=1)

    # Plot the graph
    nx.draw(graph, with_labels=True, font_weight='bold')
    plt.show()

    # Calculate the assembly index
    ai, virt_obj, path = att.calculate_assembly_index(graph)

    print(f"Assembly index: {ai}", flush=True)
