import os

import networkx as nx
import numpy as np
import pandas as pd

import assemblytheorytools as att

if __name__ == "__main__":
    data_file_in = "AA20_rTCA_distance_matrix.csv"
    kegg_data_in_path = os.path.expanduser(os.path.abspath(f"..//..//{data_file_in}"))

    # load the data
    data = pd.read_csv(kegg_data_in_path)
    abbreviations = data["Abbreviation"].values

    print(data.head(), flush=True)
    print(data.columns, flush=True)

    # Create a graph
    G = nx.Graph()
    # Add nodes to the graph
    for abbr in abbreviations:
        G.add_node(abbr)

    # convert data to numpy array
    data = data.to_numpy()[:, 1:]  # Remove the first column
    print(data, flush=True)

    # Loop over each row in the data
    for i, row in enumerate(data):
        # Find the indices of the minimum non-zero value in the row
        non_zero_row = row[row > 0]
        min_value = np.min(non_zero_row)
        min_indices = np.where(row == min_value)[0]
        min_value = row[min_indices]
        # Add an edge between the corresponding nodes
        for j in range(len(min_indices)):
            G.add_edge(abbreviations[i], abbreviations[min_indices[j]], weight=min_value[j])

    # plot  the graph
    att.plot_graph(G, f_labs=True, node_size=500)
