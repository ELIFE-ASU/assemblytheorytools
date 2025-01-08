import assemblytheorytools as att
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Create a simple graph
    G = nx.Graph()

    # Add nodes, the color attribute (string) is used to label the node
    G.add_node(0, color="0") # Graph index must start from 0
    G.add_node(1, color="1")
    G.add_node(2, color="2")
    G.add_node(3, color="3")

    # Add edges, the color attribute (integer from 1-inf) is used to label the edge
    G.add_edge(0, 1, color=1)
    G.add_edge(1,2, color=1)
    G.add_edge(2,3, color=1)

    # Plot the graph
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()

    # Calculate the assembly index
    ai, path = att.calculate_assembly_index(G)

    print(f"Assembly index: {ai}", flush=True)

