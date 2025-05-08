import matplotlib.pyplot as plt
import numpy as np

import CFG
import assemblytheorytools as att

nodes = ['a', 'b', 'c', 'd', 'ab', 'cd', 'abb', 'abc', 'abcd', 'abba', 'abab', 'ccadc', 'adccb']
n_nodes = len(nodes)

adj_matrix = np.array([
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # Nodo 0
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Nodo 1
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # Nodo 2
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # Nodo 3
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1],  # Nodo 4
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Nodo 5
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Nodo 6
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Nodo 7
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Nodo 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Nodo 9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Nodo 10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Nodo 11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # Nodo 12
])

# Calculate assembly indices 
assembly_indices = np.zeros(n_nodes, dtype=int)
for i in range(n_nodes):
    assembly_indices[i] = CFG.ai_with_pathways(nodes[i], f_print=False)[0]

labels = True
node_size = 1000
arrow_size = 80
node_color = 'Skyblue'
edge_color = 'Grey'
fig_size = 10

att.plot_assembly_circle(nodes, assembly_indices, adj_matrix, labels, node_size, arrow_size, node_color, edge_color,
                         fig_size)
plt.show()
