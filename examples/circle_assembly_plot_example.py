import numpy as np

import CFG
import assemblytheorytools as att

'''
Here is a function to plot a graph, where objects are displayed in concentric
circles according to their assembly index. 

    Parameters:
    ----------
    nodes : list
        A list of nodes in the network that are to be visualized.
    
    assembly_indices (OPTIONAL): list or numpy.ndarray
        If not provided, they will be calculated for strings.
        
    adj_matrix (OPTIONAL, but recommended): numpy.ndarray
        A square adjacency matrix representing the relationships between nodes.
        If adj_matrix[i, j] >= 1, it signifies that node i points to node j.
        If not provided, rules_graph will be used as adjacency matrix.
        IMPORTANT: if provided, must be assembly-consistent, that is, all nodes
        must be pointed to by at least one node with a lower assembly index, 
        except for nodes with the minimum assembly index, which will be considered
        the building blocks. 
    
    labels : list
        A list of labels corresponding to the nodes. These labels can be used for debugging or display purposes.
    
    node_size : float
    
    arrow_size (OPTIONAL): float
    
    node_color (OPTIONAL): str or list
    
    edge_color (OPTIONAL): str or list
    
    fig_size (OPTIONAL): float
    
    filename (OPTIONAL): str

'''

# Define our example set of nodes and adjacency matrix

nodes = ['b', 'a', 'd', 'c', 'ba', 'dc', 'baa', 'bad', 'badc', 'baab', 'baba', 'ddbcd', 'bcdda']

adj_matrix = np.array([
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # Nodo 0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Nodo 1
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # Nodo 2
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # Nodo 3
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1],  # Nodo 4
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Nodo 5
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Nodo 6
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # Nodo 7
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Nodo 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Nodo 9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Nodo 10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Nodo 11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Nodo 12
])

# Specifying adjacency matrix:

att.plot_assembly_circle(nodes, adj_matrix)

# Without specifying adjacency matrix with given parameters:

labels = True
node_size = 1000
arrow_size = 50
node_color = 'Skyblue'
edge_color = 'Grey'
fig_size = 10
filename = 'circle_plot.png'
att.plot_assembly_circle(nodes,
                         labels=True,
                         node_size=node_size,
                         arrow_size=arrow_size,
                         node_color=node_color,
                         edge_color=edge_color,
                         fig_size=fig_size,
                         filename=filename)

# Example with providing assembly indices (can be substituted by other criteria)
n_nodes = len(nodes)
assembly_indices = np.zeros(n_nodes, dtype=int)
for i in range(n_nodes):
    assembly_indices[i] = CFG.ai_with_pathways(nodes[i], f_print=False)[0]

att.plot_assembly_circle(nodes, adj_matrix=adj_matrix, assembly_indices=assembly_indices)
