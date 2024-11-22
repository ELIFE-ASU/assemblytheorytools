import numpy as np


def load_fasta(file_path):
    """
    Load a FASTA file and return its contents as a single string.

    Args:
        file_path (str): The path to the FASTA file.

    Returns:
        str: The contents of the FASTA file as a single string.
    """
    # Load the file contents into a NumPy array
    fasta_array = np.genfromtxt(file_path, dtype=str, delimiter='\n', comments='>')
    # Join the array elements into a single string
    fasta_content = ''.join(fasta_array)
    return fasta_content


# Hack to compute joint assembly index of strings
def prep_joint_string_ai(input_list, reversible=False, timeout=10.0):
    if "" in input_list:
        raise ValueError("Empty string in input list")
    # Build a string of all the inputs seperated by unique dummy characters
    delimiters = []
    amalgam_string = input_list[0]
    for string in input_list[1:]:
        unique_char = get_unique_char(amalgam_string + string)
        amalgam_string += unique_char + string
        delimiters.append(unique_char)

    # To use this to calculate joint assembly index, use formula:
    # ai(amalgum_string)-2*len(delimiters) = joint_ai(input_list)
    # 
    # delimiters can be used to process the pathway 
    return amalgam_string, delimiters


def get_unique_char(input_str):
    for i in range(1,
                   1114111):  # Max Unicode code point is 0x10FFFF, which is 1114111 in decimal, start at 1 to skip null
        char = chr(i)
        if char not in input_str:
            return char


def generate_and_visualize_string_pathway(file_path):
    """
    Generate and visualize a pathway graph using string pathway data from v5.

    Args:
        file_path (str): Path to the pathway file.

    Returns:
        networkx.DiGraph: The constructed directed graph.
    """
    import string_pathway as sp
    import networkx as nx
    import matplotlib.pyplot as plt

    # Generate the pathway object
    construction_object = sp.generate_string_pathway_ian(file_path)

    # Create a directed graph
    G = nx.DiGraph()
    for edge in sp.get_graph_string_explicit(construction_object[0])[1]:
        G.add_edge(f"{edge[0]}", f"{edge[1]}")

    # Visualize the graph
    pos = nx.spring_layout(G, k=0.9)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue",
            font_size=10, font_color="black", edge_color="gray", width=1.5)
    plt.show()

    return G


def generate_and_visualize_cfg_pathway(file_path):
    """
    Generate and visualize a directed graph from CFG pathway data.

    Args:
        file_path (str): Path to the file containing pathway data.

    Returns:
        networkx.DiGraph: The constructed directed graph.
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import re

    # Read the file content
    with open(file_path, 'r') as file:
        file_content = file.readlines()

    # Extract pathway line
    pathway_line = [line for line in file_content if line.startswith("Pathway:")][0]

    # Extract pathway data
    pathway_data = re.findall(r"'(.*?)'", pathway_line)

    # Create a directed graph
    G = nx.DiGraph()
    for relationship in pathway_data:
        reactants, product = relationship.split(' = ')
        reactant_1, reactant_2 = reactants.split(' + ')

        # Add directed edges
        G.add_edge(reactant_1, product)
        G.add_edge(reactant_2, product)

    # Visualize the graph
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue",
            font_size=10, font_color="black", font_weight="bold",
            arrows=True, arrowstyle="-|>", arrowsize=15)
    plt.title("CFG")
    plt.show()

    return G
