import os
import matplotlib.pyplot as plt
import numpy as np

import CFG
import assemblytheorytools as att

from typing import List

nodes = ['b', 'a', 'd', 'c', 'ba', 'dc', 'baa', 'bad', 'badc', 'baab', 'baba', 'ddbcd', 'bcdda']


labels = True
node_size = 1000
arrow_size = 50
node_color = 'Skyblue'
edge_color = 'Grey'
fig_size = 10
filename = 'circle_plot.png'
att.plot_assembly_circle(nodes, labels = True, node_size = node_size, arrow_size = arrow_size, node_color = node_color, edge_color = edge_color,
                             fig_size = fig_size, filename = filename)


if os.path.isfile('circle_plot.png'):
    print(f"The file was successfully generated.")
else:
    print("Failed to generate the file.")
    

