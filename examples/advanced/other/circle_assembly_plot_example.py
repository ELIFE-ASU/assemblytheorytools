import numpy as np

import assemblytheorytools as att

'''
Here is example a code to plot a graph, where objects are displayed in concentric
circles according to their assembly index. 

adj_matrix: numpy.ndarray
A square adjacency matrix representing the relationships between nodes.
If adj_matrix[i, j] >= 1, it signifies that node i points to node j.
It must be assembly-consistent: all nodes
must be pointed to by at least one node with a lower assembly index, 
except for nodes with the minimum assembly index, which will be considered
the building blocks. 
'''

if __name__ == "__main__":
    # Define our example set of nodes and adjacency matrix
    nodes = ['b', 'a', 'd', 'c', 'ba', 'dc', 'baa', 'bad', 'badc', 'baab', 'baba', 'ddbcd', 'bcdda']

    adj_matrix = np.array([
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    assembly_indices = [
        att.calculate_string_assembly_index(node, mode="cfg")[0]
        for node in nodes
    ]

    # Minimal call:
    att.plot_assembly_circle(
        nodes, adj_matrix, assembly_indices,
        filename='circle_plot.svg', node_size=800)

    # Styled call:
    labels = nodes
    node_size = 1000
    arrow_size = 50
    node_color = 'Skyblue'
    edge_color = 'Grey'
    fig_size = 10
    filename = 'circle_plot.pdf'
    att.plot_assembly_circle(nodes,
                             labels=labels,
                             adj_matrix=adj_matrix,
                             assembly_indices=assembly_indices,
                             node_size=node_size,
                             arrow_size=arrow_size,
                             node_color=node_color,
                             edge_color=edge_color,
                             fig_size=fig_size,
                             filename=filename)

    # Assembly indices can be substituted with another integer criterion.
    att.plot_assembly_circle(nodes, adj_matrix=adj_matrix, assembly_indices=assembly_indices)
